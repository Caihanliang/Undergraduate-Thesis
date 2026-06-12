#!/usr/bin/env python
""" 
resume_training_final.py
从checkpoint恢复训练 - 显存优化版
"""
import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
os.environ["UNSLOTH_SKIP_INIT_CHECK"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["OMP_NUM_THREADS"] = "4"

import logging
import random
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from chinese_calendar import is_workday
import sys
import gc
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 设置项目根路径
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"

# 重定向输出
class Logger(object):
    def __init__(self, filename=os.path.join(PROJECT_ROOT, "training_resume.txt")):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")  # 改为追加模式
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
print("="*60)
print("✓ 正在从checkpoint恢复训练...")
print("="*60)

# --- 路径配置 ---
BASE_DATA_PATH = PROJECT_ROOT
FINETUNE_OUTPUT_PATH = PROJECT_ROOT
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "results-quick")

# 查找最新的checkpoint
def find_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    checkpoints = [d for d in os.listdir(CHECKPOINT_PATH) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    # 按步数排序
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    latest = os.path.join(CHECKPOINT_PATH, checkpoints[-1])
    step = int(checkpoints[-1].split("-")[1])
    print(f"✅ 找到最新checkpoint: {latest} (step {step})")
    return latest, step

# --- 加载数据集 ---
print("\n📂 加载已保存的数据集...")
dataset_path = os.path.join(PROJECT_ROOT, "quick.json")
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

with open(dataset_path, "r", encoding="utf-8") as f:
    dataset_dicts = json.load(f)

ds = Dataset.from_list(dataset_dicts)
print(f"✅ 数据集加载完成: {len(ds)} 条样本")

# --- 预计算数字token映射 ---
def precompute_digit_mapping(tokenizer, device):
    id_to_val = {}
    digit_ids = []
    for i in range(tokenizer.vocab_size):
        t = tokenizer.decode([i]).strip()
        if t and t.isdigit():
            try:
                id_to_val[i] = float(t)
                digit_ids.append(i)
            except:
                pass
    digit_ids_tensor = torch.tensor(digit_ids, device=device)
    val_vec = torch.tensor([id_to_val[int(t)] for t in digit_ids], device=device)
    target_bracket_id = 510
    print(f"✅ 数字Token预加载: {len(digit_ids)} 个")
    return id_to_val, digit_ids, digit_ids_tensor, val_vec, target_bracket_id

# --- 优化版训练器 ---
class OptimizedMAEHybridTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tok = getattr(self, "processing_class", self.tokenizer)
        device = self.args.device
        self.id_to_val, self.digit_ids, self.digit_ids_tensor, self.val_vec, self.target_bracket_id = \
            precompute_digit_mapping(tok, device)
        self.digit_set = set(self.digit_ids)
        self.print_interval = 100
        self.step_counter = 0
        print(f"✅ 优化版Trainer初始化完成")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs["output_hidden_states"] = True
        outputs = model(**inputs)
        last_hidden = outputs.hidden_states[-1]
        logits = model.lm_head(last_hidden.to(next(model.lm_head.parameters()).dtype))
        ce_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        
        labels = inputs.get("labels")
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        
        all_digit_mask = torch.isin(shift_labels, self.digit_ids_tensor)
        
        bracket_mask = (shift_labels == self.target_bracket_id)
        bracket_positions = []
        for b in range(shift_labels.shape[0]):
            pos = torch.where(bracket_mask[b])[0]
            if len(pos) > 0:
                bracket_positions.append(pos[-1].item())
            else:
                bracket_positions.append(-1)
        
        digit_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for b, pos in enumerate(bracket_positions):
            if pos >= 0:
                digit_mask[b, pos+1:] = all_digit_mask[b, pos+1:]
        
        digit_mask &= (shift_labels != -100)
        
        mae_loss = torch.tensor(0.0, device=self.args.device)
        total_loss = ce_loss
        
        if digit_mask.any():
            dig_logits = shift_logits[digit_mask][:, self.digit_ids_tensor]
            probs = torch.softmax(dig_logits.to(torch.float32), dim=-1)
            expected_values = (probs * self.val_vec).sum(dim=-1)
            
            target_labels = shift_labels[digit_mask]
            target_vals = torch.zeros_like(expected_values)
            for i, tid in enumerate(target_labels):
                tid_int = int(tid.item())
                if tid_int in self.id_to_val:
                    target_vals[i] = self.id_to_val[tid_int]
            
            mask = (target_vals != 0).float()
            if mask.mean() > 0:
                mask = mask / mask.mean()
                mae_loss = torch.mean(torch.abs(expected_values - target_vals) * mask)
                total_loss = ce_loss + 0.11 * mae_loss
        
        self.step_counter += 1
        if self.step_counter % self.print_interval == 0 and self.state.global_step > 0:
            try:
                tok = getattr(self, "processing_class", self.tokenizer)
                s_idx = 0
                s_lab = shift_labels[s_idx].cpu().numpy()
                b_pos = np.where(s_lab == self.target_bracket_id)[0]
                if len(b_pos) > 0:
                    start = max(0, b_pos[-1] - 150)
                    safe = [x for x in s_lab[start:] if x != -100]
                    txt = tok.decode(safe, skip_special_tokens=False).split('<|eot_id|>')[0] + " <|eot_id|>"
                    print(f"\n" + "="*15 + " 核心客流对账单 " + "="*15)
                    print(f"Step {self.state.global_step}")
                    print(f"预览:\n{txt[:200]}...")
                    print(f"Loss: {total_loss.item():.4f} | CE:{ce_loss.item():.4f} | MAE:{mae_loss.item():.4f}")
                    print(f"训练进度: {self.state.global_step}/{self.state.max_steps} ({100*self.state.global_step/self.state.max_steps:.1f}%)")
                    print("="*60)
            except:
                pass
        
        return (total_loss, outputs) if return_outputs else total_loss

# --- 加载模型 ---
print(f"\n✓ 正在加载模型...")
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
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none"
)
model.config.output_hidden_states = True

# 查找checkpoint
latest_checkpoint, last_step = find_latest_checkpoint()
if latest_checkpoint is None:
    print("⚠️ 未找到checkpoint，将从头开始训练")
    resume_from = False
else:
    print(f"✅ 将从 step {last_step} 恢复训练")
    resume_from = latest_checkpoint

# 训练配置 - 显存优化版
BATCH_SIZE = 4
GRAD_ACCUM = 8
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM
total_samples = len(ds)
steps_per_epoch = total_samples // EFFECTIVE_BATCH
total_steps = steps_per_epoch * 10

print(f"\n📊 训练配置:")
print(f"   总样本数: {total_samples}")
print(f"   批次大小: {BATCH_SIZE}")
print(f"   梯度累积: {GRAD_ACCUM}")
print(f"   有效批次: {EFFECTIVE_BATCH}")
print(f"   已完成步数: {last_step if latest_checkpoint else 0}")
print(f"   剩余步数: {total_steps - (last_step if latest_checkpoint else 0)}")
print(f"   预计剩余时间: ~{(total_steps - (last_step if latest_checkpoint else 0)) * 4 / 3600:.1f} 小时")

args = TrainingArguments(
    output_dir=CHECKPOINT_PATH,
    num_train_epochs=10,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    optim="adamw_8bit",
    learning_rate=5e-5,
    bf16=True,
    logging_steps=100,
    report_to="none",
    save_strategy="steps",
    save_steps=500,
    dataloader_num_workers=1,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    save_only_model=True,
    use_cpu=False,
    ddp_find_unused_parameters=False,
)

# 初始化trainer
trainer = OptimizedMAEHybridTrainer(
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

# 显存监控
print(f"\n📊 显存状态:")
print(f"   总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"   已用显存: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"   空闲显存: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")

print("\n🚀 开始恢复训练...")
print("="*60)

# 关键：从checkpoint恢复
if resume_from:
    trainer.train(resume_from_checkpoint=resume_from)
else:
    trainer.train()

# 保存最终模型
out_path = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick")
model.save_pretrained(out_path)
print(f"\n✅ 训练完成！模型已保存至: {out_path}")

# 清理
del model, trainer
gc.collect()
torch.cuda.empty_cache()
print("✅ 显存已清理")