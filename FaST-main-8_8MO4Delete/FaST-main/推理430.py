import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel
from transformers import TextStreamer
import re
from chinese_calendar import is_workday
from tqdm import tqdm
import os
import glob
import json

# ====================== 你的配置（不用改）======================
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"
# 【关键修复】使用正确的模型路径！
# 选项1：使用430专用模型（只训练站点10，对站点10效果最好）
# MODEL_PATH = os.path.join(PROJECT_ROOT, "results-quick430/checkpoint-780")

# 选项2：使用506通用模型（训练全部157个站点，泛化能力强）
MODEL_PATH = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick506")

YTRUE_PATH = os.path.join(PROJECT_ROOT, "finetune_real_traffic.npz")
YPRED_PATH = os.path.join(PROJECT_ROOT, "finetune_data.npz")
STATION_LIST = os.path.join(PROJECT_ROOT, "station_list_hngs.txt")
PATTERN_LIST = os.path.join(PROJECT_ROOT, "station_natural_list_4feat.txt")
HIS_DATA_PATH = os.path.join(PROJECT_ROOT, "his_data_with_index.csv")

# 评测 1 个站点（你可以改数字）
TEST_STATION_ID = 8
FEATURE_NAMES = {
    0: "Passenger Car Up",      # 【修改】改为英文，与微调代码一致
    1: "Passenger Car Down",    # 【修改】改为英文，与微调代码一致
    2: "Non-Passenger Car Up",  # 【修改】改为英文，与微调代码一致
    3: "Non-Passenger Car Down" # 【修改】改为英文，与微调代码一致
}

# ====================== 加载模型（已优化最高速）======================
print("🔄 加载 LLM 模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    device_map="auto",
)
FastLanguageModel.for_inference(model)

# ========== 核心加速：开启 KV 缓存 + 推理优化 ==========
model.config.use_cache = True
model.eval()
tokenizer.padding_side = "left"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ====================== 加载数据 ======================
print("📂 加载真实值和预测值数据...")
true_npz = np.load(YTRUE_PATH)
pred_npz = np.load(YPRED_PATH)

# 正确的键名检测
if 'target' in true_npz:
    true_data = true_npz['target']
elif 'arr_0' in true_npz:
    true_data = true_npz['arr_0']
else:
    raise KeyError(f"❌ ytrue_train.npz 未找到有效键，可用键: {list(true_npz.keys())}")

if 'prediction' in pred_npz:
    pred_data = pred_npz['prediction']
elif 'arr_0' in pred_npz:
    pred_data = pred_npz['arr_0']
else:
    raise KeyError(f"❌ ypred_train.npz 未找到有效键，可用键: {list(pred_npz.keys())}")

print(f"✓ true_data shape: {true_data.shape}")
print(f"✓ pred_data shape: {pred_data.shape}")

his_df = pd.read_csv(HIS_DATA_PATH)
his_df['时间'] = pd.to_datetime(his_df['时间'])
timestamps = his_df['时间'].values[:len(true_data)]

# 自动检查站点是否越界
max_station = pred_data.shape[1] - 1
if TEST_STATION_ID > max_station:
    print(f"⚠️  站点 {TEST_STATION_ID} 越界，自动切换为 {max_station}")
    TEST_STATION_ID = max_station

with open(STATION_LIST, "r", encoding="utf-8") as f:
    stations = [l.strip() for l in f if l.strip()]
with open(PATTERN_LIST, "r", encoding="utf-8") as f:
    patterns = [l.strip() for l in f if l.strip()]

# ====================== 推理函数（【修复版】批量推理）======================
def extract_final_correction(output_text):
    match = re.search(r"Final Correction:\s*\[([0-9,\s]+)\]", output_text)
    if not match:
        return None
    try:
        return [int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()]
    except:
        return None

def predict_batch(prompts, batch_size=64):
    """【最终优化版】批量推理"""
    all_results = []
    total = len(prompts)
    success_count = 0
    fail_count = 0
    
    for start_idx in tqdm(range(0, total, batch_size), desc="批量推理中"):
        end_idx = min(start_idx + batch_size, total)
        batch_prompts = prompts[start_idx:end_idx]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64, # 【优化】配合引导符，64足够输出4个数字
                temperature=0.01,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1, # 【新增】防止复读
            )
        
        decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, text in enumerate(decoded_texts):
            res = extract_final_correction(text)
            
            # 【诊断】打印第一个样本的最后100字符
            if start_idx == 0 and i == 0:
                print(f"\n{'='*80}")
                print(f"🔍 [DEBUG] 样本 0 输出末尾:")
                print(f"{'='*80}")
                print(text[-100:]) 
                print(f"{'='*80}")
                print(f"✅ 提取结果: {res}")
                print(f"{'='*80}\n")
            
            if res:
                success_count += 1
            else:
                fail_count += 1
            
            all_results.append(res)
    
    print(f"\n📊 [诊断报告] 提取成功: {success_count}, 提取失败: {fail_count}")
    if fail_count > 0:
        print("⚠️ 警告：存在提取失败，LLM 指标可能因回退逻辑而与 GNN 完全一致！")
            
    return all_results

# ====================== 构造 Prompt（【全量对齐版】一次性包含4个特征）======================
def get_prompt_all_features(j, i):
    """构造包含所有 4 个特征的全量 Prompt"""
    station = stations[j]
    ts = pd.to_datetime(timestamps[i])
    day_type = "Workday" if is_workday(ts) else "Holiday"
    t3s = ts.strftime('%Y-%m-%d %H:%M')
    t4s = (ts + pd.Timedelta(hours=7)).strftime('%Y-%m-%d %H:%M')
    
    # 获取该站点该时间步所有 4 个特征的 GNN 预测序列
    gnn_seqs = pred_data[i, j, :].round().astype(int).tolist() # shape: (4, 8)
    
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a professional highway traffic flow refiner.\n"
        "Analyze step by step and output the final corrected traffic flow values for all 4 features.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Refine:\n"
        f"1. Station: {station}\n"
        f"2. Time: {t3s} to {t4s} | DayType: {day_type}\n"
        f"3. Weather: Unknown\n"
        f"4. Event: None\n"
        f"5. GNN Predictions:\n"
        f"   - Passenger Car Up: {gnn_seqs[0]}\n"
        f"   - Passenger Car Down: {gnn_seqs[1]}\n"
        f"   - Non-Passenger Car Up: {gnn_seqs[2]}\n"
        f"   - Non-Passenger Car Down: {gnn_seqs[3]}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "Final Correction: ["
    )
    return prompt

def predict_batch_all_features(prompts, batch_size=64):
    """【全量对齐版】一次性推理 4 个特征"""
    all_results = []
    total = len(prompts)
    success_count = 0
    fail_count = 0
    
    for start_idx in tqdm(range(0, total, batch_size), desc="批量推理中"):
        end_idx = min(start_idx + batch_size, total)
        batch_prompts = prompts[start_idx:end_idx]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256, # 给足空间让模型完成分析并输出4个数
                temperature=0.01,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for text in decoded_texts:
            res = extract_final_correction(text)
            
            # 【诊断】打印第一个样本的最后100字符
            if start_idx == 0 and len(all_results) == 0:
                print(f"\n{'='*80}")
                print(f"🔍 [DEBUG] 样本 0 输出末尾:")
                print(f"{'='*80}")
                print(text[-100:]) 
                print(f"{'='*80}")
                print(f"✅ 提取结果: {res}")
                print(f"{'='*80}\n")
            
            if res:
                success_count += 1
            else:
                fail_count += 1
            
            all_results.append(res)
    
    print(f"\n📊 [诊断报告] 提取成功: {success_count}, 提取失败: {fail_count}")
    if fail_count > 0:
        print("⚠️ 警告：存在提取失败，LLM 指标可能因回退逻辑而与 GNN 完全一致！")
            
    return all_results

# ====================== 开始推理（【核心优化】按站点-时间步一次性推理4个特征）======================
print(f"🚀 开始测试站点：{TEST_STATION_ID} | {stations[TEST_STATION_ID]}")

N_TEST = min(300, len(true_data))

# 收集全量 Prompt
print("📝 收集全量推理Prompt...")
all_prompts = [get_prompt_all_features(TEST_STATION_ID, i) for i in range(N_TEST)]

# 批量推理
print("📌 一次性推理 4 个特征（与训练格式完全一致）")
batch_results = predict_batch_all_features(all_prompts, batch_size=64)

# 整理结果
llm_results = [[] for _ in range(4)]
gnn_results = [[] for _ in range(4)]
true_results = [[] for _ in range(4)]

for i in range(N_TEST):
    res = batch_results[i]
    for f in range(4):
        true_seq = true_data[i, TEST_STATION_ID, f].tolist()
        gnn_seq = pred_data[i, TEST_STATION_ID, f].tolist()
        
        true_results[f].append(true_seq[0])
        gnn_results[f].append(gnn_seq[0])
        
        # 【关键对齐】从模型输出的4个数中提取对应特征的值
        if res and len(res) == 4:
            llm_results[f].append(res[f])
        else:
            llm_results[f].append(gnn_seq[0]) # 回退

# ====================== 指标计算 ======================
def calc_metrics(y_true, y_pred, eps=1e-6):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 【新增】非负约束，防止 GNN 或 LLM 输出微小负值影响指标
    y_pred = np.maximum(y_pred, 0)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    rmae = mae / (np.mean(y_true) + eps) * 100
    return round(mae, 2), round(mape, 2), round(rmae, 2)

# ====================== 打印指标 ======================
print("\n" + "="*80)
print(f"📊 评测结果 - 站点 {TEST_STATION_ID}：{stations[TEST_STATION_ID]}")
print("="*80)

all_metrics = []
for f in range(4):
    # 【新增】打印前5个样本的详细对比，用于诊断
    print(f"\n🔍 [DEBUG] 特征 {f} ({FEATURE_NAMES[f]}) 前5个样本对比:")
    for k in range(min(5, N_TEST)):
        true_val = true_results[f][k]
        gnn_val = gnn_results[f][k]
        llm_val = llm_results[f][k]
        print(f"   样本{k}: 真实={true_val:.1f}, GNN={gnn_val:.1f}, LLM={llm_val:.1f}")
    print("-" * 50)

    mae_g, mape_g, rmae_g = calc_metrics(true_results[f], gnn_results[f])
    mae_l, mape_l, rmae_l = calc_metrics(true_results[f], llm_results[f])

    improvement = mae_g - mae_l

    print(f"\n✅ 特征 {f}: {FEATURE_NAMES[f]}")
    print(f"GNN   | MAE: {mae_g:>7.2f} | MAPE: {mape_g:>6.2f}% | RMAE: {rmae_g:>5.2f}%")
    print(f"LLM   | MAE: {mae_l:>7.2f} | MAPE: {mape_l:>6.2f}% | RMAE: {rmae_l:>5.2f}%")
    if improvement > 0:
        print(f"提升 → MAE ↓{improvement:.2f}")
    else:
        print(f"退化 → MAE ↑{-improvement:.2f}")
        
    all_metrics.append([mae_g, mape_g, rmae_g, mae_l, mape_l, rmae_l])

# ====================== 可视化绘图 ======================
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"站点 {TEST_STATION_ID}：{stations[TEST_STATION_ID]} 对比图", fontsize=16)

axs = axs.flatten()
x = np.arange(N_TEST)

for f in range(4):
    ax = axs[f]
    ax.plot(x, true_results[f], label="真实值", linewidth=2, color="#2E8B57")
    ax.plot(x, gnn_results[f], label="GNN预测", linewidth=1.5, color="#FF6347")
    ax.plot(x, llm_results[f], label="LLM修正", linewidth=2, color="#1E90FF")
    ax.set_title(f"{FEATURE_NAMES[f]}")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("comparison_plot.png", dpi=300)
print("\n📸 图表已保存：comparison_plot.png")

# ====================== 保存结果（符合规范：JSON指标 + CSV详细误差）======================
df = pd.DataFrame(all_metrics, columns=[
    "GNN_MAE", "GNN_MAPE", "GNN_RMAE",
    "LLM_MAE", "LLM_MAPE", "LLM_RMAE"
])
df.index = [FEATURE_NAMES[i] for i in range(4)]
df.to_csv("metrics_result.csv", encoding="utf-8-sig")

# 【规范】保存JSON格式的评估指标
metrics_dict = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "station_id": TEST_STATION_ID,
    "station_name": stations[TEST_STATION_ID],
    "num_samples": N_TEST,
    "metrics": {}
}
for f in range(4):
    metrics_dict["metrics"][FEATURE_NAMES[f]] = {
        "GNN": {"MAE": float(all_metrics[f][0]), "MAPE": float(all_metrics[f][1]), "RMAE": float(all_metrics[f][2])},
        "LLM": {"MAE": float(all_metrics[f][3]), "MAPE": float(all_metrics[f][4]), "RMAE": float(all_metrics[f][5])}
    }

with open("evaluation_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

# 【规范】保存带误差分析的详细预测结果
detail_rows = []
for f in range(4):
    for i in range(N_TEST):
        detail_rows.append({
            "feature": FEATURE_NAMES[f],
            "sample_idx": i,
            "true_value": true_results[f][i],
            "gnn_prediction": gnn_results[f][i],
            "llm_prediction": llm_results[f][i],
            "gnn_error": abs(true_results[f][i] - gnn_results[f][i]),
            "llm_error": abs(true_results[f][i] - llm_results[f][i]),
            "gnn_percentage_error": abs(true_results[f][i] - gnn_results[f][i]) / (true_results[f][i] + 1e-6) * 100,
            "llm_percentage_error": abs(true_results[f][i] - llm_results[f][i]) / (true_results[f][i] + 1e-6) * 100,
        })

detail_df = pd.DataFrame(detail_rows)
detail_df.to_csv("detailed_predictions.csv", index=False, encoding="utf-8-sig")

print("📄 指标已保存：metrics_result.csv")
print("📄 JSON评估报告已保存：evaluation_metrics.json")
print("📄 详细预测结果已保存：detailed_predictions.csv")

print("\n🎉 全部完成！")
