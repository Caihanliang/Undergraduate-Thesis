#!/usr/bin/env python3
"""
真正的恢复训练脚本 - 绝不生成新数据
"""
import os
import sys
import json
import random
import torch
import numpy as np
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# 设置路径
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"

# 1. 强制检查：必须已有数据
print("="*60)
print("🤖 真正恢复训练 - 检查必要条件")
print("="*60)

# 检查checkpoint
checkpoint_path = os.path.join(PROJECT_ROOT, "results", "checkpoint-6993")
if not os.path.exists(checkpoint_path):
    print(f"❌ 错误: checkpoint不存在: {checkpoint_path}")
    sys.exit(1)
print(f"✅ checkpoint存在: {checkpoint_path}")

# 检查训练数据
data_path = os.path.join(PROJECT_ROOT, "3training_0_011.json")
if not os.path.exists(data_path):
    print(f"❌ 错误: 训练数据不存在: {data_path}")
    print("请先运行一次完整训练生成数据")
    sys.exit(1)
print(f"✅ 训练数据存在: {data_path}")

# 2. 加载训练数据
print(f"\n📥 加载训练数据...")
with open(data_path, "r", encoding="utf-8") as f:
    dataset_data = json.load(f)

print(f"📊 数据集统计:")
print(f"  总样本数: {len(dataset_data)}")
print(f"  事件样本: {sum(1 for d in dataset_data if d.get('reason')=='Event')}")
print(f"  误差样本: {sum(1 for d in dataset_data if d.get('reason')=='Error')}")
print(f"  普通样本: {sum(1 for d in dataset_data if d.get('reason')=='Normal')}")

# 打乱数据
random.shuffle(dataset_data)
dataset = Dataset.from_list(dataset_data)

# 3. 加载模型（从checkpoint）
print(f"\n🤖 加载模型（从checkpoint）...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint_path,  # 关键：从checkpoint加载
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    device_map={"": 0},
)

# 4. 导入你的trainer
print(f"导入NoLogitMAEHybridTrainer...")
# 创建一个简化的trainer，避免导入原文件的所有代码
class SimpleMAETrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        local_tokenizer = getattr(self, "processing_class", self.tokenizer)
        self.id_to_val = {}
        all_digit_ids = []
        for i in range(local_tokenizer.vocab_size):
            t = local_tokenizer.decode([i]).strip()
            if t and all(c.isdigit() for c in t):
                try:
                    self.id_to_val[i] = float(t)
                    all_digit_ids.append(i)
                except:
                    pass
        self.digit_ids = torch.tensor(all_digit_ids).to(self.args.device)
        self.target_bracket_id = 510
        print(f"✅ 数字Token: {len(all_digit_ids)} 个")

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
            bracket_pos = (shift_labels[b] == self.target_bracket_id).nonzero(as_tuple=True)[0]
            if len(bracket_pos) > 0:
                start_from = bracket_pos[-1].item() + 1
                digit_mask[b, start_from:] = all_digit_mask[b, start_from:]
        digit_mask &= (shift_labels != -100)

        mae_loss = torch.tensor(0.0).to(self.args.device)
        total_loss = ce_loss
        expected_values = torch.tensor([]).to(self.args.device)

        if digit_mask.any():
            dig_logits = shift_logits[digit_mask][:, self.digit_ids]
            probs = torch.softmax(dig_logits.to(torch.float32), dim=-1)
            val_vec = torch.tensor([self.id_to_val[int(t)] for t in self.digit_ids]).to(self.args.device)
            expected_values = (probs * val_vec).sum(dim=-1)
            target_labels = shift_labels[digit_mask]
            target_vals = torch.zeros_like(expected_values)
            for i in range(len(target_labels)):
                tid = int(target_labels[i])
                target_vals[i] = self.id_to_val.get(tid, 0.0)
            
            mask = (target_vals != 0).float()
            if mask.mean() > 0:
                mask /= mask.mean()
            
            mae_loss = torch.mean(torch.abs(expected_values - target_vals) * mask)
            total_loss = ce_loss + 0.11 * mae_loss

            if self.state.global_step % 1 == 0 and self.state.global_step < 10:
                print(f"\n[Step {self.state.global_step}] Loss: {total_loss.item():.4f} | MAE: {mae_loss.item():.4f}")

        return (total_loss, outputs) if return_outputs else total_loss

# 5. 配置LoRA
model.config.output_hidden_states = True
model = FastLanguageModel.get_peft_model(
    model, r=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                   "gate_proj","up_proj","down_proj",
                   "embed_tokens","lm_head"],
    lora_alpha=16, lora_dropout=0, bias="none"
)

# 6. 训练参数
args = TrainingArguments(
    output_dir=os.path.join(PROJECT_ROOT, "results"),
    num_train_epochs=10,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    optim="paged_adamw_8bit",
    learning_rate=5e-5,
    bf16=True,
    logging_steps=10,
    report_to="none",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=5,
    remove_unused_columns=False,
    push_to_hub=False,
    warmup_steps=0,
    max_steps=20000,  # 总共训练到20000步
)

# 7. 创建trainer
trainer = SimpleMAETrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=args,
    max_seq_length=1024,
    packing=False
)

# 8. 开始训练
print(f"\n{'='*60}")
print(f"🚀 开始训练 - 从 Step 6993 继续")
print(f"{'='*60}")
trainer.train(resume_from_checkpoint=True)

# 9. 保存模型
output_path = os.path.join(PROJECT_ROOT, "llama-resumed-final")
model.save_pretrained(output_path)
print(f"\n✅ 训练完成！模型保存到: {output_path}")