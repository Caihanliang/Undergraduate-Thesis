import os
import re
import gc
import json
import math
import torch
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from unsloth import FastLanguageModel
from chinese_calendar import is_workday

# ====================== 环境设置 ======================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# ====================== 路径配置 ======================
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"
MODEL_PATH = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick509")

YTRUE_PATH = os.path.join(PROJECT_ROOT, "finetune_real_traffic.npz")
YPRED_PATH = os.path.join(PROJECT_ROOT, "finetune_data.npz")
STATION_LIST = os.path.join(PROJECT_ROOT, "station_list_hngs.txt")
PATTERN_LIST = os.path.join(PROJECT_ROOT, "station_natural_list_4feat.txt")
HIS_DATA_PATH = os.path.join(PROJECT_ROOT, "his_data_with_index.csv")
WEATHER_DATA_PATH = os.path.join(PROJECT_ROOT, "160站点天气信息/")
EVENTS_CSV_PATH = os.path.join(PROJECT_ROOT, "events_list_quan.csv")

# ====================== 评测配置 ======================
TEST_STATION_ID = 100
N_TEST = 300
BATCH_SIZE = 16   # 更稳，避免长输出时OOM或截断副作用

FEATURE_NAMES = {
    0: "Passenger Car Up",
    1: "Passenger Car Down",
    2: "Non-Passenger Car Up",
    3: "Non-Passenger Car Down"
}

# 与训练代码保持一致
INSTRUCTION_TEXT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a professional highway traffic flow refiner.\n"
    "Analyze step by step and output the final corrected traffic flow values.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Refine:\n"
    "1. Station: {}\n"
    "2. Feature: {}\n"
    "3. Time: {} to {} | DayType: {}\n"
    "4. Flow Pattern: {}\n"
    "5. Weather: {}\n"
    "6. GNN: {}\n"
    "7. Event: {}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

# ====================== 工具函数 ======================
def load_npz_array(npz_path, preferred_key):
    data = np.load(npz_path)
    if preferred_key in data:
        return data[preferred_key]
    elif "arr_0" in data:
        return data["arr_0"]
    else:
        raise KeyError(f"{npz_path} 未找到有效键，可用键: {list(data.keys())}")

def align_4d_arrays(true_data, pred_data):
    """
    对齐到 [num_samples, num_stations, num_features, seq_len]
    与训练代码保持一致
    """
    if true_data.ndim != 4 or pred_data.ndim != 4:
        raise ValueError(f"数据维度错误: true={true_data.shape}, pred={pred_data.shape}")

    if true_data.shape[1] == 8 and true_data.shape[3] == 4:
        true_data = true_data.transpose(0, 2, 3, 1)
        pred_data = pred_data.transpose(0, 2, 3, 1)
    elif true_data.shape[1] == 4 and true_data.shape[3] == 8:
        true_data = true_data.transpose(0, 2, 1, 3)
        pred_data = pred_data.transpose(0, 2, 1, 3)

    return true_data, pred_data

def get_station_short_name(desc: str):
    parts = str(desc).split()
    return parts[0] if parts else str(desc)

def preload_all_weather(weather_dir, total_stations):
    weather_dict = {}
    if not os.path.exists(weather_dir):
        print("⚠️ 天气目录不存在")
        return weather_dict

    files = os.listdir(weather_dir)
    for j in tqdm(range(total_stations), desc="预加载天气"):
        try:
            prefix = f"{j:03d}"
            match = [f for f in files if f.startswith(prefix)]
            if not match:
                continue

            path = os.path.join(weather_dir, match[0])

            if path.endswith(".csv"):
                df = pd.read_csv(path, encoding="utf-8")
            elif path.endswith(".xlsx") or path.endswith(".xls"):
                df = pd.read_excel(path)
            else:
                continue

            d_col = None
            w_col = None
            for c in df.columns:
                c_lower = str(c).lower()
                if "日" in str(c) or "date" in c_lower:
                    d_col = c
                if "天气" in str(c) or "weather" in c_lower:
                    w_col = c

            if not d_col or not w_col:
                continue

            df[d_col] = pd.to_datetime(df[d_col]).dt.date.astype(str)
            df = df.drop_duplicates(subset=[d_col]).set_index(d_col)
            weather_dict[j] = df[w_col].astype(str).to_dict()
        except Exception:
            continue

    print(f"✅ 天气预加载完成：{len(weather_dict)} 个站点")
    return weather_dict

def get_weather_fast(weather_cache, station_idx, date_str):
    return weather_cache.get(station_idx, {}).get(date_str, "Unknown")

def load_events(path):
    if not os.path.exists(path):
        print(f"⚠️ 未找到事件文件: {path}")
        return {}

    try:
        df_ev = pd.read_csv(path, encoding="utf-8")
        date_col = "日期"
        station_col = "站点名称"
        event_col = "事件描述"

        for c in [date_col, station_col, event_col]:
            if c not in df_ev.columns:
                raise ValueError(f"事件文件缺少列: {c}")

        df_ev[date_col] = pd.to_datetime(df_ev[date_col]).dt.strftime("%Y-%m-%d")
        df_ev[station_col] = df_ev[station_col].astype(str).str.strip()
        df_ev[event_col] = df_ev[event_col].astype(str).str.strip()

        event_map = {}
        for _, r in df_ev.iterrows():
            event_map[(r[date_col], r[station_col])] = r[event_col]

        print(f"✅ 事件加载完成：{len(event_map)} 条")
        return event_map
    except Exception as e:
        print(f"❌ 事件加载失败: {e}")
        return {}

def calc_metrics(y_true, y_pred, eps=1e-6):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_pred = np.maximum(y_pred, 0)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    rmae = mae / (np.mean(y_true) + eps) * 100

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE": round(float(mape), 2),
        "RMAE": round(float(rmae), 2),
    }

# ====================== 解析函数：核心修复 ======================
def extract_final_correction(text, expected_len, allow_naked_numbers=False):
    """
    三层解析：
    1. 严格匹配完整 Final Correction: [....]
    2. 宽松匹配 Final Correction 后出现的前 expected_len 个数字
    3. （仅强约束重试时）如果文本开头就是数字列表，则直接取前 expected_len 个数字
    """
    if text is None:
        return None

    # ---------- 1) 严格匹配完整 Final Correction ----------
    matches = list(re.finditer(r"Final\s*Correction\s*:\s*\[([^\]]+)\]", text, flags=re.I | re.S))
    if matches:
        content = matches[-1].group(1)
        nums = re.findall(r"-?\d+", content)
        if len(nums) >= expected_len:
            return [max(0, int(x)) for x in nums[:expected_len]]

    # ---------- 2) 宽松匹配：从 Final Correction 后往后抓数字 ----------
    idx = text.lower().rfind("final correction")
    if idx != -1:
        tail = text[idx:]
        nums = re.findall(r"-?\d+", tail)
        if len(nums) >= expected_len:
            # 第一个数字很可能是 step number / 噪音时，优先取最后 expected_len 个更稳
            # 但这里 tail 已经从 Final Correction 开始，通常直接前 expected_len 就行
            return [max(0, int(x)) for x in nums[:expected_len]]

    # ---------- 3) 强约束 prompt 下：文本可能直接从数字开始 ----------
    if allow_naked_numbers:
        stripped = text.strip()
        if stripped.startswith(tuple("0123456789-")):
            nums = re.findall(r"-?\d+", stripped)
            if len(nums) >= expected_len:
                return [max(0, int(x)) for x in nums[:expected_len]]

    return None

# ====================== 模型加载 ======================
print("🔄 加载 LLM 模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    device_map="auto",
)
# ========== 新增打印：显示当前使用的模型路径 ==========
print(f"✅ 模型加载完成，使用模型路径：{MODEL_PATH}")

FastLanguageModel.for_inference(model)
model.config.use_cache = True
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# eot 停止符
STOP_IDS = []
if tokenizer.eos_token_id is not None:
    STOP_IDS.append(tokenizer.eos_token_id)
try:
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in STOP_IDS:
        STOP_IDS.append(eot_id)
except Exception:
    pass

# ====================== 加载数据 ======================
print("📂 加载真实值和预测值数据...")
true_data = load_npz_array(YTRUE_PATH, "target")
pred_data = load_npz_array(YPRED_PATH, "prediction")
true_data, pred_data = align_4d_arrays(true_data, pred_data)

num_samples, num_stations, num_features, seq_len = true_data.shape
print(f"✓ 对齐后 shape: true={true_data.shape}, pred={pred_data.shape}")

if TEST_STATION_ID >= num_stations:
    print(f"⚠️ 站点 {TEST_STATION_ID} 越界，自动切换到 {num_stations - 1}")
    TEST_STATION_ID = num_stations - 1

with open(STATION_LIST, "r", encoding="utf-8") as f:
    stations = [l.strip() for l in f if l.strip()]

with open(PATTERN_LIST, "r", encoding="utf-8") as f:
    patterns = [l.strip() for l in f if l.strip()]

his_df = pd.read_csv(HIS_DATA_PATH)
his_df["时间"] = pd.to_datetime(his_df["时间"])
parking_data = his_df.set_index("时间").sort_index()

weather_cache = preload_all_weather(WEATHER_DATA_PATH, num_stations)
event_map = load_events(EVENTS_CSV_PATH)

# 与训练阶段一致的时间映射
rows = np.arange(num_samples) * num_stations + TEST_STATION_ID
rows = rows[rows < len(parking_data)]
station_timestamps = parking_data.index[rows]

usable_samples = min(N_TEST, len(station_timestamps), num_samples)
station_desc = stations[TEST_STATION_ID]
station_short_name = get_station_short_name(station_desc)

print(f"🚀 开始测试站点：{TEST_STATION_ID} | {station_desc}")
print(f"📌 可评测样本数：{usable_samples}")

# ====================== Prompt 构造 ======================
def build_prompts_for_sample(station_id, sample_idx, feature_idx):
    desc = stations[station_id]
    s_name = get_station_short_name(desc)

    ts = station_timestamps[sample_idx]
    t3s = ts.strftime("%Y-%m-%d %H:%M")
    t4s = (ts + pd.Timedelta(hours=7)).strftime("%Y-%m-%d %H:%M")
    dayt = "Workday" if is_workday(ts) else "Off-day"

    date_str = ts.strftime("%Y-%m-%d")
    weather = get_weather_fast(weather_cache, station_id, date_str)
    event = event_map.get((date_str, s_name), "None")

    pattern_idx = station_id * num_features + feature_idx
    pattern = patterns[pattern_idx] if pattern_idx < len(patterns) else "General"

    gnn_seq = pred_data[sample_idx, station_id, feature_idx].round().astype(int).tolist()

    # 标准 prompt：与训练完全同构
    prompt_std = INSTRUCTION_TEXT.format(
        desc,
        FEATURE_NAMES[feature_idx],
        t3s,
        t4s,
        dayt,
        pattern,
        weather,
        gnn_seq,
        event
    )

    # 强约束 prompt：只在失败重试时使用
    # 目的：跳过长COT，直接续写数字
    prompt_force = prompt_std + "Final Correction: ["

    meta = {
        "sample_idx": sample_idx,
        "station_id": station_id,
        "station_desc": desc,
        "station_short_name": s_name,
        "feature_idx": feature_idx,
        "feature_name": FEATURE_NAMES[feature_idx],
        "timestamp": str(ts),
        "date": date_str,
        "weather": weather,
        "event": event,
        "pattern": pattern,
        "day_type": dayt,
        "gnn_seq": gnn_seq,
        "prompt_std": prompt_std,
        "prompt_force": prompt_force,
    }
    return meta

# ====================== 生成函数：只解码新生成文本 ======================
def generate_texts(prompts, max_new_tokens, batch_size, desc="生成中"):
    all_texts = []
    total = len(prompts)

    for start in tqdm(range(0, total, batch_size), desc=desc):
        batch_prompts = prompts[start:start + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to("cuda")

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=STOP_IDS if len(STOP_IDS) > 1 else STOP_IDS[0],
                repetition_penalty=1.03,
            )

        # 只取“新生成部分”，不要把 prompt 再解码进去
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[:, prompt_len:]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=False)
        all_texts.extend(texts)

        del inputs, outputs, gen_ids
        torch.cuda.empty_cache()

    return all_texts

# ====================== 带重试的推理：彻底修复核心 ======================
def predict_feature_with_retry(prompt_items, expected_len, batch_size=16):
    """
    三轮推理：
    1) 标准prompt，320 token
    2) 标准prompt，480 token
    3) 强约束prompt，64 token（直接续写 Final Correction: [ 后面的数字）
    """
    results = [None] * len(prompt_items)
    raw_outputs = [""] * len(prompt_items)
    fail_indices = list(range(len(prompt_items)))

    strategies = [
        ("第1轮 标准推理", "prompt_std", 320, False),
        ("第2轮 长输出重试", "prompt_std", 480, False),
        ("第3轮 强约束数字重试", "prompt_force", 64, True),
    ]

    for round_name, prompt_key, max_new_tokens, allow_naked in strategies:
        if not fail_indices:
            break

        prompts = [prompt_items[i][prompt_key] for i in fail_indices]
        texts = generate_texts(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            desc=round_name
        )

        new_fail_indices = []
        success_this_round = 0

        for origin_idx, text in zip(fail_indices, texts):
            raw_outputs[origin_idx] = text
            seq = extract_final_correction(
                text=text,
                expected_len=expected_len,
                allow_naked_numbers=allow_naked
            )
            if seq is not None and len(seq) == expected_len:
                results[origin_idx] = seq
                success_this_round += 1
            else:
                new_fail_indices.append(origin_idx)

        print(f"✅ {round_name} 成功: {success_this_round} | 剩余失败: {len(new_fail_indices)}")

        # 打印当前轮一个失败样本，便于排查
        if new_fail_indices:
            debug_idx = new_fail_indices[0]
            debug_text = raw_outputs[debug_idx]
            print("\n" + "=" * 90)
            print(f"🔍 [DEBUG] {round_name} 失败样本输出末尾：")
            print("=" * 90)
            print(debug_text[-500:])
            print("=" * 90 + "\n")

        fail_indices = new_fail_indices

    return results, raw_outputs, fail_indices

# ====================== 主推理流程 ======================
llm_flat = {f: [] for f in range(num_features)}
gnn_flat = {f: [] for f in range(num_features)}
true_flat = {f: [] for f in range(num_features)}

# 绘图使用：首步值
plot_true = {f: [] for f in range(num_features)}
plot_gnn = {f: [] for f in range(num_features)}
plot_llm = {f: [] for f in range(num_features)}

detail_rows = []
summary_rows = []

total_fail_after_retry = 0

for f in range(num_features):
    print(f"\n🚀 开始推理特征 {f}: {FEATURE_NAMES[f]}")

    prompt_items = [build_prompts_for_sample(TEST_STATION_ID, i, f) for i in range(usable_samples)]

    pred_results, raw_outputs, final_fail_indices = predict_feature_with_retry(
        prompt_items=prompt_items,
        expected_len=seq_len,
        batch_size=BATCH_SIZE
    )

    total_fail_after_retry += len(final_fail_indices)

    # 汇总
    for i in range(usable_samples):
        meta = prompt_items[i]

        true_seq = true_data[i, TEST_STATION_ID, f].round().astype(int).tolist()
        gnn_seq = pred_data[i, TEST_STATION_ID, f].round().astype(int).tolist()

        if pred_results[i] is not None:
            llm_seq = pred_results[i]
            is_fallback = False
        else:
            llm_seq = gnn_seq
            is_fallback = True

        true_flat[f].extend(true_seq)
        gnn_flat[f].extend(gnn_seq)
        llm_flat[f].extend(llm_seq)

        plot_true[f].append(true_seq[0])
        plot_gnn[f].append(gnn_seq[0])
        plot_llm[f].append(llm_seq[0])

        for h in range(seq_len):
            detail_rows.append({
                "station_id": TEST_STATION_ID,
                "station_short_name": station_short_name,
                "station_desc": station_desc,
                "feature_id": f,
                "feature_name": FEATURE_NAMES[f],
                "sample_idx": i,
                "horizon": h + 1,
                "timestamp": meta["timestamp"],
                "date": meta["date"],
                "day_type": meta["day_type"],
                "weather": meta["weather"],
                "event": meta["event"],
                "pattern": meta["pattern"],
                "true_value": true_seq[h],
                "gnn_prediction": gnn_seq[h],
                "llm_prediction": llm_seq[h],
                "gnn_error": abs(true_seq[h] - gnn_seq[h]),
                "llm_error": abs(true_seq[h] - llm_seq[h]),
                "gnn_percentage_error": abs(true_seq[h] - gnn_seq[h]) / (true_seq[h] + 1e-6) * 100,
                "llm_percentage_error": abs(true_seq[h] - llm_seq[h]) / (true_seq[h] + 1e-6) * 100,
                "is_fallback": is_fallback,
                "raw_output_tail": raw_outputs[i][-300:] if isinstance(raw_outputs[i], str) else "",
            })

# ====================== 指标计算 ======================
print("\n" + "=" * 100)
print(f"📊 评测结果 - 站点 {TEST_STATION_ID}: {station_desc}")
print("=" * 100)

metrics_dict = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "station_id": TEST_STATION_ID,
    "station_short_name": station_short_name,
    "station_desc": station_desc,
    "num_samples": usable_samples,
    "seq_len": seq_len,
    "final_fallback_count": total_fail_after_retry,
    "metrics": {}
}

metrics_table_rows = []

for f in range(num_features):
    print(f"\n🔍 [DEBUG] 特征 {f} ({FEATURE_NAMES[f]}) 前5个样本首步对比:")
    for k in range(min(5, usable_samples)):
        print(
            f"   样本{k}: "
            f"真实={plot_true[f][k]}, "
            f"GNN={plot_gnn[f][k]}, "
            f"LLM={plot_llm[f][k]}"
        )
    print("-" * 60)

    gnn_metrics = calc_metrics(true_flat[f], gnn_flat[f])
    llm_metrics = calc_metrics(true_flat[f], llm_flat[f])

    mae_improve = round(gnn_metrics["MAE"] - llm_metrics["MAE"], 2)
    rmse_improve = round(gnn_metrics["RMSE"] - llm_metrics["RMSE"], 2)

    print(f"✅ 特征 {f}: {FEATURE_NAMES[f]}")
    print(
        f"GNN | MAE: {gnn_metrics['MAE']:>7.2f} | RMSE: {gnn_metrics['RMSE']:>7.2f} | "
        f"MAPE: {gnn_metrics['MAPE']:>7.2f}% | RMAE: {gnn_metrics['RMAE']:>7.2f}%"
    )
    print(
        f"LLM | MAE: {llm_metrics['MAE']:>7.2f} | RMSE: {llm_metrics['RMSE']:>7.2f} | "
        f"MAPE: {llm_metrics['MAPE']:>7.2f}% | RMAE: {llm_metrics['RMAE']:>7.2f}%"
    )

    if mae_improve > 0:
        print(f"提升 → MAE ↓ {mae_improve:.2f}")
    else:
        print(f"退化 → MAE ↑ {-mae_improve:.2f}")

    metrics_dict["metrics"][FEATURE_NAMES[f]] = {
        "GNN": gnn_metrics,
        "LLM": llm_metrics,
        "MAE_improvement": mae_improve,
        "RMSE_improvement": rmse_improve,
    }

    metrics_table_rows.append({
        "Feature": FEATURE_NAMES[f],
        "GNN_MAE": gnn_metrics["MAE"],
        "GNN_RMSE": gnn_metrics["RMSE"],
        "GNN_MAPE": gnn_metrics["MAPE"],
        "GNN_RMAE": gnn_metrics["RMAE"],
        "LLM_MAE": llm_metrics["MAE"],
        "LLM_RMSE": llm_metrics["RMSE"],
        "LLM_MAPE": llm_metrics["MAPE"],
        "LLM_RMAE": llm_metrics["RMAE"],
        "MAE_Improve": mae_improve,
        "RMSE_Improve": rmse_improve,
    })

print(f"\n📌 最终仍解析失败并回退到GNN的样本数: {total_fail_after_retry}")

# ====================== 可视化 ======================
# 用全英文避免中文字形警告
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"Station {TEST_STATION_ID}: {station_short_name} (first-step view)", fontsize=16)

axs = axs.flatten()
x = np.arange(usable_samples)

for f in range(num_features):
    ax = axs[f]
    ax.plot(x, plot_true[f], label="True", linewidth=2, color="#2E8B57")
    ax.plot(x, plot_gnn[f], label="GNN", linewidth=1.5, color="#FF6347")
    ax.plot(x, plot_llm[f], label="LLM", linewidth=1.8, color="#1E90FF")
    ax.set_title(FEATURE_NAMES[f])
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(PROJECT_ROOT, f"comparison_plot_station_{TEST_STATION_ID}.png")
plt.savefig(plot_path, dpi=300)
plt.close()

# ====================== 保存结果 ======================
metrics_df = pd.DataFrame(metrics_table_rows)
metrics_csv_path = os.path.join(PROJECT_ROOT, f"metrics_result_station_{TEST_STATION_ID}.csv")
metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

metrics_json_path = os.path.join(PROJECT_ROOT, f"evaluation_metrics_station_{TEST_STATION_ID}.json")
with open(metrics_json_path, "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

detail_df = pd.DataFrame(detail_rows)
detail_csv_path = os.path.join(PROJECT_ROOT, f"detailed_predictions_station_{TEST_STATION_ID}.csv")
detail_df.to_csv(detail_csv_path, index=False, encoding="utf-8-sig")

print(f"\n📸 图表已保存：{plot_path}")
print(f"📄 指标已保存：{metrics_csv_path}")
print(f"📄 JSON评估报告已保存：{metrics_json_path}")
print(f"📄 详细预测结果已保存：{detail_csv_path}")

# ====================== 释放显存 ======================
gc.collect()
torch.cuda.empty_cache()

print("\n🎉 全部完成！")