#!/usr/bin/env python
"""
stable_resume.py - 稳定版恢复训练（防止OOM）
"""
import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
os.environ["UNSLOTH_SKIP_INIT_CHECK"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.environ["OMP_NUM_THREADS"] = "2"

import torch
import json
import gc
import numpy as np
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"

# 查找最新checkpoint
def find_checkpoint():
    results_dir = os.path.join(PROJECT_ROOT, "results")
    if not os.path.exists(results_dir):
        return None, 0
    
    checkpoints = []
    for d in os.listdir(results_dir):
        if d.startswith("checkpoint-"):
            step = int(d.split("-")[1])
            checkpoints.append((step, d))
    
    if not checkpoints:
        return None, 0
    
    checkpoints.sort()
    latest_step, latest_dir = checkpoints[-1]
    return os.path.join(results_dir, latest_dir), latest_step

print("="*60)
print("稳定版恢复训练脚本")
print("="*60)

# 加载数据集
print("\n📂 加载数据集...")
dataset_path = os.path.join(PROJECT_ROOT, "3training_0_011.json")
with open(dataset_path, "r") as f:
    dataset_dicts = json.load(f)

# 限制样本数防止OOM
MAX_SAMPLES = 30000
if len(dataset_dicts) > MAX_SAMPLES:
    dataset_dicts = dataset_dicts[:MAX_SAMPLES]
    print(f"⚠️ 限制样本数: {MAX_SAMPLES}")

ds = Dataset.from_list(dataset_dicts)
print(f"✅ 数据集: {len(ds)} 条样本")

# 加载模型
print("\n✓ 加载模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/user/Llama-3.1-8B",
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    device_map={"": 0},
)

# LoRA配置
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # 从32降到16，减少显存
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none"
)
model.config.output_hidden_states = True

# 查找checkpoint
checkpoint_path, last_step = find_checkpoint()
if checkpoint_path:
    print(f"✅ 找到checkpoint: step {last_step}")
else:
    print("⚠️ 未找到checkpoint，从头训练")
    last_step = 0

# 训练配置 - 显存友好版
BATCH_SIZE = 2          # 进一步降低
GRAD_ACCUM = 8          # 增加累积
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM

total_samples = len(ds)
steps_per_epoch = total_samples // EFFECTIVE_BATCH
total_steps = steps_per_epoch * 10

print(f"\n📊 训练配置:")
print(f"   批次大小: {BATCH_SIZE}")
print(f"   梯度累积: {GRAD_ACCUM}")
print(f"   有效批次: {EFFECTIVE_BATCH}")
print(f"   已完成: {last_step} 步")
print(f"   总步数: {total_steps}")
print(f"   剩余步数: {total_steps - last_step}")

args = TrainingArguments(
    output_dir=os.path.join(PROJECT_ROOT, "results"),
    num_train_epochs=10,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    optim="adamw_8bit",
    learning_rate=5e-5,
    bf16=True,
    logging_steps=50,           # 减少日志频率
    report_to="none",
    save_strategy="steps",
    save_steps=500,              # 每500步保存
    dataloader_num_workers=1,
    dataloader_pin_memory=False, # 减少内存
    remove_unused_columns=False,
    gradient_checkpointing=True,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,           # 梯度裁剪
    save_only_model=True,
    use_cpu=False,
)

# 自定义训练器（简化版，减少打印）
class SimpleMAETrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tok = getattr(self, "processing_class", self.tokenizer)
        self.id_to_val = {}
        digit_ids = []
        for i in range(tok.vocab_size):
            t = tok.decode([i]).strip()
            if t and t.isdigit():
                try:
                    self.id_to_val[i] = float(t)
                    digit_ids.append(i)
                except:
                    pass
        self.digit_ids = torch.tensor(digit_ids).to(self.args.device)
        self.target_bracket_id = 510
        self.print_counter = 0
        print(f"✅ 数字Token: {len(digit_ids)} 个")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs["output_hidden_states"] = True
        outputs = model(**inputs)
        last_hidden = outputs.hidden_states[-1]
        logits = model.lm_head(last_hidden.to(next(model.lm_head.parameters()).dtype))
        ce_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        
        labels = inputs.get("labels")
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        
        all_digit_mask = torch.isin(shift_labels, self.digit_ids)
        digit_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for b in range(shift_labels.shape[0]):
            pos = (shift_labels[b] == self.target_bracket_id).nonzero(as_tuple=True)[0]
            if len(pos) > 0:
                digit_mask[b, pos[-1]+1:] = all_digit_mask[b, pos[-1]+1:]
        digit_mask &= (shift_labels != -100)
        
        mae_loss = torch.tensor(0.0, device=self.args.device)
        total_loss = ce_loss
        
        if digit_mask.any():
            dig_logits = shift_logits[digit_mask][:, self.digit_ids]
            probs = torch.softmax(dig_logits.float(), dim=-1)
            val_vec = torch.tensor([self.id_to_val[int(t)] for t in self.digit_ids], device=self.args.device)
            expected = (probs * val_vec).sum(dim=-1)
            
            target_labels = shift_labels[digit_mask]
            target_vals = torch.zeros_like(expected)
            for i, tid in enumerate(target_labels):
                target_vals[i] = self.id_to_val.get(int(tid), 0.0)
            
            mask = (target_vals != 0).float()
            if mask.mean() > 0:
                mask = mask / mask.mean()
            mae_loss = torch.mean(torch.abs(expected - target_vals) * mask)
            total_loss = ce_loss + 0.11 * mae_loss
        
        # 每200步打印一次
        self.print_counter += 1
        if self.print_counter % 200 == 0 and self.state.global_step > 0:
            print(f"\nStep {self.state.global_step}: Loss={total_loss.item():.4f}, CE={ce_loss.item():.4f}, MAE={mae_loss.item():.4f}")
        
        return (total_loss, outputs) if return_outputs else total_loss

# 初始化trainer
trainer = SimpleMAETrainer(
    model=model,
    train_dataset=ds,
    tokenizer=tokenizer,
    args=args,
    max_seq_length=1024,
    packing=False,
)

# 清空缓存
torch.cuda.empty_cache()
gc.collect()

# 显存状态
print(f"\n📊 显存: {torch.cuda.memory_allocated(0)/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

print("\n🚀 开始恢复训练...")
print("="*60)

# 从checkpoint恢复
if checkpoint_path:
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    trainer.train()

# 保存模型
out_path = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-final")
model.save_pretrained(out_path)
print(f"\n✅ 完成！模型: {out_path}")

# 清理
del model, trainer
gc.collect()
torch.cuda.empty_cache()