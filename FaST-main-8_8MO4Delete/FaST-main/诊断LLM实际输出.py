"""
LLM输出诊断脚本：查看模型实际生成的内容
运行方式：python 诊断LLM实际输出.py
"""
import torch
import numpy as np
import pandas as pd
from unsloth import FastLanguageModel
import re
from chinese_calendar import is_workday
import os

PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"
MODEL_PATH = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick430")
YTRUE_PATH = os.path.join(PROJECT_ROOT, "finetune_real_traffic.npz")
YPRED_PATH = os.path.join(PROJECT_ROOT, "finetune_data.npz")
STATION_LIST = os.path.join(PROJECT_ROOT, "station_list_hngs.txt")
PATTERN_LIST = os.path.join(PROJECT_ROOT, "station_natural_list_4feat.txt")
HIS_DATA_PATH = os.path.join(PROJECT_ROOT, "his_data_with_index.csv")

TEST_STATION_ID = 10
FEATURE_NAMES = {
    0: "Passenger Car Up",
    1: "Passenger Car Down",
    2: "Non-Passenger Car Up",
    3: "Non-Passenger Car Down",
}

print("="*80)
print("🔍 LLM输出诊断工具")
print("="*80)

# 加载模型
print("\n【步骤1】加载模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    device_map="auto",
)
FastLanguageModel.for_inference(model)
model.config.use_cache = True
model.eval()

# 加载数据
print("\n【步骤2】加载数据...")
true_npz = np.load(YTRUE_PATH)
pred_npz = np.load(YPRED_PATH)

if 'target' in true_npz:
    true_data = true_npz['target']
elif 'arr_0' in true_npz:
    true_data = true_npz['arr_0']

if 'prediction' in pred_npz:
    pred_data = pred_npz['prediction']
elif 'arr_0' in pred_npz:
    pred_data = pred_npz['arr_0']

his_df = pd.read_csv(HIS_DATA_PATH)
his_df['时间'] = pd.to_datetime(his_df['时间'])
timestamps = his_df['时间'].values[:len(true_data)]

with open(STATION_LIST, "r", encoding="utf-8") as f:
    stations = [l.strip() for l in f if l.strip()]
with open(PATTERN_LIST, "r", encoding="utf-8") as f:
    patterns = [l.strip() for l in f if l.strip()]

print(f"✅ 数据加载完成")
print(f"   真实值形状: {true_data.shape}")
print(f"   预测值形状: {pred_data.shape}")

# 构造Prompt
def get_prompt(j, f, i):
    station = stations[j]
    feature = FEATURE_NAMES[f]
    ts = pd.to_datetime(timestamps[i])
    day_type = "Workday" if is_workday(ts) else "Holiday"
    gnn_seq = pred_data[i, j, f].round().astype(int).tolist()
    pattern = patterns[j*4 + f] if (j*4 + f) < len(patterns) else "General"
    
    weather = "Unknown"
    event = "None"
    
    t3s = ts.strftime('%Y-%m-%d %H:%M')
    t4 = ts + pd.Timedelta(hours=7)
    t4s = t4.strftime('%Y-%m-%d %H:%M')
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a professional highway traffic flow refiner.
Analyze step by step and output the final corrected traffic flow values.<|eot_id|><|start_header_id|>user<|end_header_id|>

Refine:
1. Station: {station}
2. Feature: {feature}
3. Time: {t3s} to {t4s} | DayType: {day_type}
4. Flow Pattern: {pattern}
5. Weather: {weather}
6. GNN: {gnn_seq}
7. Event: {event}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt

# 测试单个样本
print("\n【步骤3】测试单个样本推理...")
print("-" * 80)

j = TEST_STATION_ID
f = 0
i = 0

prompt = get_prompt(j, f, i)
print(f"\n 输入Prompt（前200字符）:")
print(prompt[:200] + "...")

print(f"\n🎯 测试样本信息:")
print(f"   站点: {stations[j]}")
print(f"   特征: {FEATURE_NAMES[f]}")
print(f"   GNN输入: {pred_data[i, j, f].round().astype(int).tolist()}")
print(f"   真实值: {true_data[i, j, f].tolist()}")

# 推理
print(f"\n 开始推理...")
inputs = tokenizer([prompt], return_tensors="pt", truncation=True).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,  # 增加生成长度
        temperature=0.01,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
    )

full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "="*80)
print("📄 LLM完整输出:")
print("="*80)
print(full_text)
print("="*80)

# 尝试提取Final Correction
match = re.search(r"Final Correction:\s*\[([0-9,\s]+)\]", full_text)
if match:
    print(f"\n✅ 成功提取Final Correction:")
    extracted = [int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()]
    print(f"   {extracted}")
else:
    print(f"\n❌ 未能提取Final Correction！")
    print(f"\n🔍 问题分析:")
    print(f"   1. 检查输出中是否包含'Final Correction'关键词:")
    if "Final Correction" in full_text:
        print(f"      ✅ 包含关键词，但格式可能不匹配")
        # 显示包含关键词的上下文
        idx = full_text.find("Final Correction")
        print(f"      上下文: ...{full_text[max(0,idx-50):idx+100]}...")
    else:
        print(f"      ❌ 完全不包含该关键词")
    
    print(f"\n   2. 可能的原因:")
    print(f"      - 模型未正确学习输出格式")
    print(f"      - Prompt格式与微调时不一致")
    print(f"      - 模型加载的不是微调后的权重")
    print(f"      - 训练数据质量问题")

# 测试多个样本
print("\n" + "="*80)
print("📊 批量测试5个样本...")
print("="*80)

success_count = 0
fail_count = 0

for test_i in range(5):
    prompt = get_prompt(j, f, test_i)
    inputs = tokenizer([prompt], return_tensors="pt", truncation=True).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.01,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    match = re.search(r"Final Correction:\s*\[([0-9,\s]+)\]", text)
    
    if match:
        success_count += 1
        print(f"\n样本{test_i}: ✅ 成功提取")
    else:
        fail_count += 1
        print(f"\n样本{test_i}: ❌ 失败")
        # 显示输出的最后100字符
        print(f"   输出结尾: ...{text[-200:]}")

print(f"\n 统计结果:")
print(f"   成功: {success_count}/5")
print(f"   失败: {fail_count}/5")

if fail_count > 0:
    print(f"\n⚠️  诊断结论:")
    print(f"   LLM未能正确输出预期格式，所有样本都回退到GNN值")
    print(f"   这就是为什么微调和微调前指标完全相同！")
    
    print(f"\n🔧 建议检查:")
    print(f"   1. 确认加载的是微调后的模型（不是原始预训练模型）")
    print(f"   2. 检查微调时的Prompt格式是否与推理完全一致")
    print(f"   3. 查看训练日志，确认Loss确实在下降")
    print(f"   4. 检查微调数据中是否包含正确的'Final Correction'格式")

print("\n" + "="*80)
