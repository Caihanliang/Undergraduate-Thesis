import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel
import re
from chinese_calendar import is_workday
from tqdm import tqdm
import os
import json

# ====================== 配置区域 ======================
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "infer_all_157_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick506")
YTRUE_PATH = os.path.join(PROJECT_ROOT, "finetune_real_traffic.npz")
YPRED_PATH = os.path.join(PROJECT_ROOT, "finetune_data.npz")
STATION_LIST = os.path.join(PROJECT_ROOT, "station_list_hngs.txt")
PATTERN_LIST = os.path.join(PROJECT_ROOT, "station_natural_list_4feat.txt")
HIS_DATA_PATH = os.path.join(PROJECT_ROOT, "his_data_with_index.csv")

FEATURE_NAMES = {
    0: "Passenger Car Up",
    1: "Passenger Car Down",
    2: "Non-Passenger Car Up",
    3: "Non-Passenger Car Down"
}

N_TEST = min(300, 1000) # 每个站点推理的样本数，可根据显存调整

# ====================== 加载模型 ======================
print("🔄 加载 LLM 模型...")
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
tokenizer.padding_side = "left"

# ====================== 加载数据 ======================
print("📂 加载数据...")
true_npz = np.load(YTRUE_PATH)
pred_npz = np.load(YPRED_PATH)

true_data = true_npz['target'] if 'target' in true_npz else true_npz['arr_0']
pred_data = pred_npz['prediction'] if 'prediction' in pred_npz else pred_npz['arr_0']

his_df = pd.read_csv(HIS_DATA_PATH)
his_df['时间'] = pd.to_datetime(his_df['时间'])
timestamps = his_df['时间'].values[:len(true_data)]

with open(STATION_LIST, "r", encoding="utf-8") as f:
    stations = [l.strip() for l in f if l.strip()]
with open(PATTERN_LIST, "r", encoding="utf-8") as f:
    patterns = [l.strip() for l in f if l.strip()]

TOTAL_STATIONS = len(stations)
print(f"✓ 数据加载完成。总站点数: {TOTAL_STATIONS}, 样本数: {len(true_data)}")

# ====================== 辅助函数 ======================
def extract_final_correction(output_text):
    match = re.search(r"Final Correction:\s*\[([0-9,\s]+)\]", output_text)
    if not match:
        return None
    try:
        return [int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()]
    except:
        return None

def predict_batch(prompts, batch_size=64):
    """【极速诱导版】批量推理"""
    all_results = []
    total = len(prompts)
    debug_count = 0
    
    for start_idx in tqdm(range(0, total, batch_size), desc="批量推理中"):
        end_idx = min(start_idx + batch_size, total)
        batch_prompts = prompts[start_idx:end_idx]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30, # 【关键】只给输出数字的空间，防止废话
                temperature=0.01,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1, # 【新增】防止复读
            )
        
        decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for text in decoded_texts:
            res = extract_final_correction(text)
            
            # 【诊断】打印前2个样本
            if debug_count < 2:
                print(f"\n{'='*80}")
                print(f"🔍 LLM原始输出 (样本 {start_idx + debug_count}):")
                print(f"{'='*80}")
                print(text[-150:]) # 只打印最后150字符，看是否包含数字
                print(f"{'='*80}")
                print(f"✅ 提取结果: {res}")
                print(f"{'='*80}\n")
                debug_count += 1
            
            all_results.append(res)
            
    return all_results

def get_prompt(j, f, i):
    station = stations[j]
    feature = FEATURE_NAMES[f]
    ts = pd.to_datetime(timestamps[i])
    day_type = "Workday" if is_workday(ts) else "Holiday"
    
    # 【关键对齐】获取该站点该特征的8个时间步序列
    gnn_seq = pred_data[i, j, f].round().astype(int).tolist()
    
    pattern = patterns[j*4 + f] if (j*4 + f) < len(patterns) else "General"
    t3s = ts.strftime('%Y-%m-%d %H:%M')
    t4s = (ts + pd.Timedelta(hours=7)).strftime('%Y-%m-%d %H:%M')
    
    # 【关键修复】在 Prompt 末尾增加引导符，强制模型续写
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a professional highway traffic flow refiner.\n"
        "Analyze step by step and output the final corrected traffic flow values.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Refine:\n"
        f"1. Station: {station}\n"
        f"2. Feature: {feature}\n"
        f"3. Time: {t3s} to {t4s} | DayType: {day_type}\n"
        f"4. Flow Pattern: {pattern}\n"
        f"5. Weather: Unknown\n"
        f"6. GNN: {gnn_seq}\n"
        f"7. Event: None<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "Final Correction: [" # 【诱导式生成】
    )
    return prompt

def calc_metrics(y_true, y_pred, eps=1e-6):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    rmae = mae / (np.mean(y_true) + eps) * 100
    return round(mae, 2), round(mape, 2), round(rmae, 2)

# ====================== 主循环：推理全部 157 个站点 ======================
all_station_metrics = []

print(f"\n🚀 开始全量推理 {TOTAL_STATIONS} 个站点...")

for station_id in range(TOTAL_STATIONS):
    print(f"\n📍 正在处理站点 [{station_id+1}/{TOTAL_STATIONS}]: {stations[station_id]}")
    
    station_res = {"station_id": station_id, "station_name": stations[station_id]}
    
    for f in range(4):
        # 收集 Prompt
        prompts = [get_prompt(station_id, f, i) for i in range(N_TEST)]
        
        # 批量推理
        batch_results = predict_batch(prompts, batch_size=16)
        
        true_vals, gnn_vals, llm_vals = [], [], []
        for i in range(N_TEST):
            true_seq = true_data[i, station_id, f].tolist()
            gnn_seq = pred_data[i, station_id, f].tolist()
            res = batch_results[i]
            
            true_vals.append(true_seq[0])
            gnn_vals.append(gnn_seq[0])
            
            # 【关键对齐】模型输出的是4个数的列表 [v0, v1, v2, v3]
            # 我们当前在处理特征 f，所以取 res[f]
            if res and len(res) > f:
                llm_vals.append(res[f])
            else:
                llm_vals.append(gnn_seq[0]) # 回退
        
        # 计算指标
        mae_g, mape_g, rmae_g = calc_metrics(true_vals, gnn_vals)
        mae_l, mape_l, rmae_l = calc_metrics(true_vals, llm_vals)
        
        station_res[f"{FEATURE_NAMES[f]}_GNN_MAE"] = mae_g
        station_res[f"{FEATURE_NAMES[f]}_GNN_MAPE"] = mape_g
        station_res[f"{FEATURE_NAMES[f]}_GNN_RMAE"] = rmae_g
        station_res[f"{FEATURE_NAMES[f]}_LLM_MAE"] = mae_l
        station_res[f"{FEATURE_NAMES[f]}_LLM_MAPE"] = mape_l
        station_res[f"{FEATURE_NAMES[f]}_LLM_RMAE"] = rmae_l
        
        print(f"   特征 {f}: GNN MAE={mae_g:.2f}, LLM MAE={mae_l:.2f}")

    all_station_metrics.append(station_res)

# ====================== 结果汇总与保存 ======================
df_all = pd.DataFrame(all_station_metrics)
csv_path = os.path.join(OUTPUT_DIR, "157_stations_full_metrics.csv")
df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")

# 计算全局平均指标（157个站点，4个特征的平均值）
global_avg = {}
for metric in ["MAE", "MAPE", "RMAE"]:
    for model in ["GNN", "LLM"]:
        cols = [c for c in df_all.columns if c.endswith(f"{model}_{metric}")]
        global_avg[f"Global_Avg_{model}_{metric}"] = df_all[cols].mean().mean()

print("\n" + "="*80)
print("📊 157个站点全局平均指标 (Global Average across all stations & features)")
print("="*80)
for k, v in global_avg.items():
    print(f"{k}: {v:.4f}")

# 保存全局平均指标到 JSON
avg_report = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "total_stations": TOTAL_STATIONS,
    "samples_per_station": N_TEST,
    "global_average_metrics": global_avg
}
json_path = os.path.join(OUTPUT_DIR, "global_average_metrics.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(avg_report, f, indent=2, ensure_ascii=False)

print(f"\n✅ 全部完成！")
print(f"📁 详细结果: {csv_path}")
print(f"📄 全局平均报告: {json_path}")

# ====================== 你原有单个站点测试逻辑【完全原封不动】======================
TEST_STATION_ID = 0
print(f"\n🚀 开始测试站点：{TEST_STATION_ID} | {stations[TEST_STATION_ID]}")

all_prompts_by_feature = {}
for f in range(4):
    print(f"   特征 {f}: {FEATURE_NAMES[f]}")
    j = TEST_STATION_ID
    prompts = []
    for i in range(N_TEST):
        prompts.append(get_prompt(j, f, i))
    all_prompts_by_feature[f] = prompts

llm_results = [[] for _ in range(4)]
gnn_results = [[] for _ in range(4)]
true_results = [[] for _ in range(4)]

for f in range(4):
    print(f"\n📌 批量推理特征 {f}: {FEATURE_NAMES[f]}")
    j = TEST_STATION_ID
    batch_results = predict_batch(all_prompts_by_feature[f], batch_size=32)

    for i in range(N_TEST):
        true_seq = true_data[i, j, f].tolist()
        gnn_seq = pred_data[i, j, f].tolist()
        res = batch_results[i]

        true_results[f].append(true_seq[0])
        gnn_results[f].append(gnn_seq[0])
        
        # 【关键对齐】模型输出的是4个数的列表 [v0, v1, v2, v3]
        # 我们当前在处理特征 f，所以取 res[f]
        if res and len(res) > f:
            llm_results[f].append(res[f])
        else:
            llm_results[f].append(gnn_seq[0]) # 回退

# 打印指标
print("\n" + "="*80)
print(f"📊 评测结果 - 站点 {TEST_STATION_ID}：{stations[TEST_STATION_ID]}")
print("="*80)

all_metrics = []
for f in range(4):
    mae_g, mape_g, rmae_g = calc_metrics(true_results[f], gnn_results[f])
    mae_l, mape_l, rmae_l = calc_metrics(true_results[f], llm_results[f])

    print(f"\n✅ 特征 {f}: {FEATURE_NAMES[f]}")
    print(f"GNN   | MAE: {mae_g:5.2f} | MAPE: {mape_g:6.2f}% | RMAE: {rmae_g:5.2f}%")
    print(f"LLM   | MAE: {mae_l:5.2f} | MAPE: {mape_l:6.2f}% | RMAE: {rmae_l:5.2f}%")
    print(f"提升 → MAE ↓{mae_g - mae_l:.2f}")
    all_metrics.append([mae_g, mape_g, rmae_g, mae_l, mape_l, rmae_l])

# 绘图【完全原版】
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

# 图片保存到统一文件夹
pic_path = os.path.join(OUTPUT_DIR, "comparison_plot.png")
plt.tight_layout()
plt.savefig(pic_path, dpi=300)
print(f"\n📸 图表已保存：{pic_path}")

# 所有原输出文件全部迁移到统一文件夹
df = pd.DataFrame(all_metrics, columns=[
    "GNN_MAE", "GNN_MAPE", "GNN_RMAE",
    "LLM_MAE", "LLM_MAPE", "LLM_RMAE"
])
df.index = [FEATURE_NAMES[i] for i in range(4)]
df.to_csv(os.path.join(OUTPUT_DIR, "metrics_result.csv"), encoding="utf-8-sig")

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

with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

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
detail_df.to_csv(os.path.join(OUTPUT_DIR, "detailed_predictions.csv"), index=False, encoding="utf-8-sig")

print("📄 单站点指标已保存至统一文件夹")
print("📄 JSON评估报告已保存至统一文件夹")
print("📄 详细预测结果已保存至统一文件夹")
print("\n🎉 全部完成！")