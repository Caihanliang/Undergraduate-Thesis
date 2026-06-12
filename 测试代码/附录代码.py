import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# 1. 指令模板（核心COT/指令）
instruction_text = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a conservative highway traffic flow refiner.\n"
    "Rules: 1. GNN合理则不改；2.避免跳变；3.仅输出Final Correction.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Station: {} | Feature: {} | Time: {} | Weather: {} | GNN: {} | Event: {}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

# 2. 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    "/home/user/Llama-3.1-8B", max_seq_length=1024, load_in_4bit=True
)
tokenizer.pad_token = tokenizer.eos_token

# 3. 构造样本（核心格式：prompt+答案）
def build_sample(desc, feat, t, w, gnn, ev, target):
    prompt = instruction_text.format(desc, feat, t, w, gnn, ev)
    answer = f"Final Correction: {target}<|eot_id|>"
    full = prompt + answer
    enc = tokenizer(full, truncation=True, add_special_tokens=False)
    p_enc = tokenizer(prompt, add_special_tokens=False)
    return {"input_ids": enc["input_ids"],
            "labels": [-100]*len(p_enc["input_ids"])+enc["input_ids"][len(p_enc["input_ids"]):]}

# 4. 示例样本+数据集
samples = [build_sample("站点A", "小客车", "2025-01-01", "晴", [100,110], "无", [102,108])]
ds = Dataset.from_list(samples)

# 5. LoRA配置
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj"]
)

# 6. 训练参数
args = TrainingArguments(
    output_dir="./results", max_steps=2200, per_device_train_batch_size=4,
    gradient_accumulation_steps=8, learning_rate=5e-5, bf16=True
)

# 7. 微调+保存
trainer = SFTTrainer(model=model, train_dataset=ds, tokenizer=tokenizer, args)
trainer.train()
model.save_pretrained("llama_finetuned")