import os
import re
import gc
import csv
import json
import torch
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from unsloth import FastLanguageModel
from chinese_calendar import is_workday

# ====================== 环境设置 ======================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# ====================== 路径配置 ======================
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"
MODEL_PATH = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick506")

YTRUE_PATH = os.path.join(PROJECT_ROOT, "finetune_real_traffic.npz")
YPRED_PATH = os.path.join(PROJECT_ROOT, "finetune_data.npz")
STATION_LIST = os.path.join(PROJECT_ROOT, "station_list_hngs.txt")
PATTERN_LIST = os.path.join(PROJECT_ROOT, "station_natural_list_4feat.txt")
HIS_DATA_PATH = os.path.join(PROJECT_ROOT, "his_data_with_index.csv")
WEATHER_DATA_PATH = os.path.join(PROJECT_ROOT, "160站点天气信息/")
EVENTS_CSV_PATH = os.path.join(PROJECT_ROOT, "events_list_quan.csv")

# ====================== 推理配置 ======================
DEBUG_MAX_STATIONS = None
DEBUG_MAX_SAMPLES_PER_STATION = None

# 👇 👇 👇 【关键】从断掉的 104 号站点开始继续跑 👇 👇 👇
START_FROM_STATION = 112

# 不跳过任何站点
SKIP_STATION_IDS = set()

# 👇 👇 👇 【关键】降低批大小防止再次被杀死 👇 👇 👇
BATCH_SIZE = 2

# 输出长度只需要 8 个数字左右，不需要 320/480
FIRST_PASS_MAX_NEW_TOKENS = 32
RETRY_MAX_NEW_TOKENS = 48

FEATURE_NAMES = {
    0: "Passenger Car Up",
    1: "Passenger Car Down",
    2: "Non-Passenger Car Up",
    3: "Non-Passenger Car Up"
}

# ====================== Prompt 模板 ======================
BASE_INSTRUCTION_TEXT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a highway traffic flow refiner. "
    "Return only the corrected traffic flow sequence. "
    "No explanation.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Refine traffic flow.\n"
    "Station: {}\n"
    "Feature: {}\n"
    "Time: {} to {} | DayType: {}\n"
    "Pattern: {}\n"
    "Weather: {}\n"
    "GNN: {}\n"
    "Event: {}\n"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

# ====================== 输出文件 ======================
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "inference_results_all_stations_with_predictions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DETAIL_PATH = os.path.join(OUTPUT_DIR, "all_station_timeslot_feature_metrics.csv")
STATION_FEATURE_MEAN_PATH = os.path.join(OUTPUT_DIR, "all_station_feature_mean_metrics.csv")
GLOBAL_FEATURE_MEAN_PATH = os.path.join(OUTPUT_DIR, "global_feature_mean_metrics.csv")
GLOBAL_OVERALL_MEAN_PATH = os.path.join(OUTPUT_DIR, "global_overall_mean_metrics.csv")

# 新增：保存 true / GNN / LLM 预测序列，方便对比
PREDICTION_COMPARE_PATH = os.path.join(OUTPUT_DIR, "llm_gnn_prediction_compare.csv")


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
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.maximum(y_pred, 0)

    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    rmae = mae / (np.mean(y_true) + eps) * 100

    return {
        "MAE": round(float(mae), 4),
        "MAPE": round(float(mape), 4),
        "RMAE": round(float(rmae), 4),
    }


class MetricAccumulator:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.count = 0
        self.sum_abs_err = 0.0
        self.sum_pct_err = 0.0
        self.sum_true = 0.0

    def update(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        y_pred = np.maximum(y_pred, 0)

        abs_err = np.abs(y_true - y_pred)
        pct_err = abs_err / (y_true + self.eps) * 100

        self.count += y_true.size
        self.sum_abs_err += abs_err.sum()
        self.sum_pct_err += pct_err.sum()
        self.sum_true += y_true.sum()

    def result(self):
        if self.count == 0:
            return {"MAE": 0.0, "MAPE": 0.0, "RMAE": 0.0}

        mae = self.sum_abs_err / self.count
        mape = self.sum_pct_err / self.count
        mean_true = self.sum_true / self.count
        rmae = mae / (mean_true + self.eps) * 100

        return {
            "MAE": round(float(mae), 4),
            "MAPE": round(float(mape), 4),
            "RMAE": round(float(rmae), 4),
        }


# ====================== 解析函数 ======================
def extract_final_correction(text, expected_len, allow_naked_numbers=False):
    if text is None:
        return None

    # 1. 严格匹配 Final Correction: [...]
    matches = list(
        re.finditer(
            r"Final\s*Correction\s*:\s*\[([^\]]+)\]",
            text,
            flags=re.I | re.S
        )
    )
    if matches:
        content = matches[-1].group(1)
        nums = re.findall(r"-?\d+", content)
        if len(nums) >= expected_len:
            return [max(0, int(x)) for x in nums[:expected_len]]

    # 2. 宽松匹配 final correction 后面的数字
    idx = text.lower().rfind("final correction")
    if idx != -1:
        tail = text[idx:]
        nums = re.findall(r"-?\d+", tail)
        if len(nums) >= expected_len:
            return [max(0, int(x)) for x in nums[:expected_len]]

    # 3. 如果 prompt 已经以 "Final Correction: [" 结尾，
    #    生成内容可能直接是 "123, 456, ..."
    if allow_naked_numbers:
        stripped = text.strip()
        if stripped.startswith(tuple("0123456789-")):
            nums = re.findall(r"-?\d+", stripped)
            if len(nums) >= expected_len:
                return [max(0, int(x)) for x in nums[:expected_len]]

    return None


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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

STOP_IDS = []
if tokenizer.eos_token_id is not None:
    STOP_IDS.append(tokenizer.eos_token_id)

try:
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in STOP_IDS:
        STOP_IDS.append(eot_id)
except Exception:
    pass

if len(STOP_IDS) == 0:
    STOP_IDS = [tokenizer.eos_token_id]


# ====================== 加载数据 ======================
print("📂 加载真实值和预测值数据...")

true_data = load_npz_array(YTRUE_PATH, "target")
pred_data = load_npz_array(YPRED_PATH, "prediction")
true_data, pred_data = align_4d_arrays(true_data, pred_data)

num_samples, num_stations, num_features, seq_len = true_data.shape
print(f"✓ 对齐后 shape: true={true_data.shape}, pred={pred_data.shape}")

with open(STATION_LIST, "r", encoding="utf-8") as f:
    stations = [l.strip() for l in f if l.strip()]

with open(PATTERN_LIST, "r", encoding="utf-8") as f:
    patterns = [l.strip() for l in f if l.strip()]

his_df = pd.read_csv(HIS_DATA_PATH)
his_df["时间"] = pd.to_datetime(his_df["时间"])
parking_data = his_df.set_index("时间").sort_index()

weather_cache = preload_all_weather(WEATHER_DATA_PATH, num_stations)
event_map = load_events(EVENTS_CSV_PATH)


# ====================== 生成实际推理站点列表 ======================
if DEBUG_MAX_STATIONS is None:
    station_range = list(range(num_stations))
else:
    station_range = list(range(min(DEBUG_MAX_STATIONS, num_stations)))

# 👇 👇 👇 【关键】只保留 >= 104 的站点 👇 👇 👇
station_range = [s for s in station_range if s >= START_FROM_STATION]

print(f"🚫 跳过站点: 0~{START_FROM_STATION-1}")
print(f"🚀 实际推理站点数: {len(station_range)} / {num_stations}")
print(f"📁 输出目录: {OUTPUT_DIR}")
print(f"🚀 每站样本数限制: {DEBUG_MAX_SAMPLES_PER_STATION if DEBUG_MAX_SAMPLES_PER_STATION is not None else '全部'}")
print(f"🚀 批大小: {BATCH_SIZE}")
print(f"🚀 第一轮 max_new_tokens: {FIRST_PASS_MAX_NEW_TOKENS}")
print(f"🚀 第二轮 max_new_tokens: {RETRY_MAX_NEW_TOKENS}")


# ====================== 预缓存每个站点时间索引 ======================
print("🕒 预缓存站点时间索引...")

station_timestamp_cache = []

for station_id in range(num_stations):
    rows = np.arange(num_samples) * num_stations + station_id
    rows = rows[rows < len(parking_data)]
    station_timestamp_cache.append(parking_data.index[rows])

print("✅ 时间索引缓存完成")


# ====================== Prompt 构造 ======================
def build_prompt_meta(station_id, sample_idx, feature_idx, station_timestamps):
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

    prompt_std = BASE_INSTRUCTION_TEXT.format(
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

    # 强约束：让模型直接续写数字
    prompt_force = prompt_std + "Final Correction: ["

    return {
        "station_id": station_id,
        "station_name": desc,
        "station_short_name": s_name,
        "sample_idx": sample_idx,
        "timestamp": str(ts),
        "date": date_str,
        "feature_id": feature_idx,
        "feature_name": FEATURE_NAMES[feature_idx],
        "weather": weather,
        "event": event,
        "pattern": pattern,
        "day_type": dayt,
        "gnn_seq": gnn_seq,
        "prompt_std": prompt_std,
        "prompt_force": prompt_force,
    }


# ====================== 推理函数 ======================
def generate_texts(prompts, max_new_tokens, batch_size, desc="生成中"):
    all_texts = []
    total = len(prompts)

    for start in tqdm(range(0, total, batch_size), desc=desc, leave=False):
        batch_prompts = prompts[start:start + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to("cuda", non_blocking=True)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=STOP_IDS if len(STOP_IDS) > 1 else STOP_IDS[0],
                repetition_penalty=1.0,
            )

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[:, prompt_len:]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=False)

        all_texts.extend(texts)

        del inputs, outputs, gen_ids

    return all_texts


def predict_feature_with_retry(prompt_items, expected_len, batch_size=32, print_debug=False):
    results = [None] * len(prompt_items)
    raw_outputs = [""] * len(prompt_items)
    fail_indices = list(range(len(prompt_items)))

    strategies = [
        ("第1轮 强约束推理", "prompt_force", FIRST_PASS_MAX_NEW_TOKENS, True),
        ("第2轮 短重试", "prompt_force", RETRY_MAX_NEW_TOKENS, True),
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

        if print_debug:
            print(f"✅ {round_name} 成功: {success_this_round} | 剩余失败: {len(new_fail_indices)}")
            if new_fail_indices:
                debug_idx = new_fail_indices[0]
                print("\n" + "=" * 90)
                print(f"🔍 [DEBUG] {round_name} 失败样本输出：")
                print("=" * 90)
                print(raw_outputs[debug_idx][-500:])
                print("=" * 90 + "\n")

        fail_indices = new_fail_indices

    return results, raw_outputs, fail_indices


# ====================== 输出表头 ======================
detail_header = [
    "station_id", "station_short_name", "station_name",
    "sample_idx", "timestamp",
    "feature_id", "feature_name",
    "GNN_MAE", "GNN_MAPE", "GNN_RMAE",
    "LLM_MAE", "LLM_MAPE", "LLM_RMAE",
    "MAE_improvement", "MAPE_improvement", "RMAE_improvement",
    "is_fallback"
]

station_feature_mean_header = [
    "station_id", "station_short_name", "station_name",
    "feature_id", "feature_name",
    "num_samples", "num_points", "fallback_count",
    "GNN_MAE", "GNN_MAPE", "GNN_RMAE",
    "LLM_MAE", "LLM_MAPE", "LLM_RMAE",
    "MAE_improvement", "MAPE_improvement", "RMAE_improvement"
]

global_feature_mean_header = [
    "feature_id", "feature_name",
    "station_count", "total_samples", "total_points", "fallback_count",
    "GNN_MAE", "GNN_MAPE", "GNN_RMAE",
    "LLM_MAE", "LLM_MAPE", "LLM_RMAE",
    "MAE_improvement", "MAPE_improvement", "RMAE_improvement"
]

global_overall_header = [
    "scope",
    "station_count", "feature_count", "total_samples", "total_points", "fallback_count",
    "GNN_MAE", "GNN_MAPE", "GNN_RMAE",
    "LLM_MAE", "LLM_MAPE", "LLM_RMAE",
    "MAE_improvement", "MAPE_improvement", "RMAE_improvement"
]

# 新增：逐样本预测值对比表
prediction_compare_header = [
    "station_id",
    "station_short_name",
    "station_name",
    "sample_idx",
    "timestamp",
    "feature_id",
    "feature_name",
    "is_fallback",
    "true_seq",
    "gnn_seq",
    "llm_seq"
]


# ====================== 全局汇总器 ======================
global_feature_acc = {
    f: {
        "gnn": MetricAccumulator(),
        "llm": MetricAccumulator(),
        "fallback_count": 0,
        "total_samples": 0,
        "station_ids": set(),
    }
    for f in range(num_features)
}

overall_acc = {
    "gnn": MetricAccumulator(),
    "llm": MetricAccumulator(),
    "fallback_count": 0,
    "total_samples": 0,
    "station_ids": set(),
    "feature_ids": set(),
}


# ====================== 主循环 ======================
print("\n🚀 开始全站点推理...\n")

# 👇 👇 👇 【关键】以追加模式打开，不覆盖之前结果 👇 👇 👇
detail_f = open(DETAIL_PATH, "a", newline="", encoding="utf-8-sig")
detail_writer = csv.writer(detail_f)

station_feature_f = open(STATION_FEATURE_MEAN_PATH, "a", newline="", encoding="utf-8-sig")
station_feature_writer = csv.writer(station_feature_f)

prediction_compare_f = open(PREDICTION_COMPARE_PATH, "a", newline="", encoding="utf-8-sig")
prediction_compare_writer = csv.writer(prediction_compare_f)

try:
    for station_id in tqdm(station_range, desc="总站点进度"):
        station_desc = stations[station_id]
        station_short_name = get_station_short_name(station_desc)

        station_timestamps = station_timestamp_cache[station_id]

        usable_samples = min(len(station_timestamps), num_samples)

        if DEBUG_MAX_SAMPLES_PER_STATION is not None:
            usable_samples = min(usable_samples, DEBUG_MAX_SAMPLES_PER_STATION)

        if usable_samples <= 0:
            continue

        for feature_id in range(num_features):
            feature_name = FEATURE_NAMES[feature_id]

            prompt_items = [
                build_prompt_meta(station_id, i, feature_id, station_timestamps)
                for i in range(usable_samples)
            ]

            print_debug = (station_id == station_range[0] and feature_id == 0)

            pred_results, raw_outputs, final_fail_indices = predict_feature_with_retry(
                prompt_items=prompt_items,
                expected_len=seq_len,
                batch_size=BATCH_SIZE,
                print_debug=print_debug
            )

            station_feature_gnn_acc = MetricAccumulator()
            station_feature_llm_acc = MetricAccumulator()
            station_feature_fallback_count = 0

            detail_rows_buffer = []
            prediction_compare_rows_buffer = []

            for i in range(usable_samples):
                true_seq = np.rint(true_data[i, station_id, feature_id]).astype(np.int32)
                true_seq = np.maximum(true_seq, 0)

                gnn_seq = np.rint(pred_data[i, station_id, feature_id]).astype(np.int32)
                gnn_seq = np.maximum(gnn_seq, 0)

                if pred_results[i] is not None:
                    llm_seq = np.asarray(pred_results[i], dtype=np.int32)
                    llm_seq = np.maximum(llm_seq, 0)
                    is_fallback = 0
                else:
                    llm_seq = gnn_seq
                    is_fallback = 1
                    station_feature_fallback_count += 1

                gnn_m = calc_metrics(true_seq, gnn_seq)
                llm_m = calc_metrics(true_seq, llm_seq)

                mae_improve = round(gnn_m["MAE"] - llm_m["MAE"], 4)
                mape_improve = round(gnn_m["MAPE"] - llm_m["MAPE"], 4)
                rmae_improve = round(gnn_m["RMAE"] - llm_m["RMAE"], 4)

                timestamp_value = prompt_items[i]["timestamp"]

                detail_rows_buffer.append([
                    station_id,
                    station_short_name,
                    station_desc,
                    i,
                    timestamp_value,
                    feature_id,
                    feature_name,
                    gnn_m["MAE"], gnn_m["MAPE"], gnn_m["RMAE"],
                    llm_m["MAE"], llm_m["MAPE"], llm_m["RMAE"],
                    mae_improve, mape_improve, rmae_improve,
                    is_fallback
                ])

                # 新增：保存 true / GNN / LLM 预测序列
                prediction_compare_rows_buffer.append([
                    station_id,
                    station_short_name,
                    station_desc,
                    i,
                    timestamp_value,
                    feature_id,
                    feature_name,
                    is_fallback,
                    json.dumps(true_seq.astype(int).tolist(), ensure_ascii=False),
                    json.dumps(gnn_seq.astype(int).tolist(), ensure_ascii=False),
                    json.dumps(llm_seq.astype(int).tolist(), ensure_ascii=False),
                ])

                # 站点-特征汇总
                station_feature_gnn_acc.update(true_seq, gnn_seq)
                station_feature_llm_acc.update(true_seq, llm_seq)

                # 全局-特征汇总
                global_feature_acc[feature_id]["gnn"].update(true_seq, gnn_seq)
                global_feature_acc[feature_id]["llm"].update(true_seq, llm_seq)
                global_feature_acc[feature_id]["total_samples"] += 1
                global_feature_acc[feature_id]["station_ids"].add(station_id)
                global_feature_acc[feature_id]["fallback_count"] += is_fallback

                # 全局总汇总
                overall_acc["gnn"].update(true_seq, gnn_seq)
                overall_acc["llm"].update(true_seq, llm_seq)
                overall_acc["total_samples"] += 1
                overall_acc["station_ids"].add(station_id)
                overall_acc["feature_ids"].add(feature_id)
                overall_acc["fallback_count"] += is_fallback

            # 写详细结果
            detail_writer.writerows(detail_rows_buffer)

            # 写 true / GNN / LLM 预测值对比表
            prediction_compare_writer.writerows(prediction_compare_rows_buffer)

            # 写站点-特征平均
            sf_gnn = station_feature_gnn_acc.result()
            sf_llm = station_feature_llm_acc.result()

            sf_row = [
                station_id,
                station_short_name,
                station_desc,
                feature_id,
                feature_name,
                usable_samples,
                usable_samples * seq_len,
                station_feature_fallback_count,
                sf_gnn["MAE"], sf_gnn["MAPE"], sf_gnn["RMAE"],
                sf_llm["MAE"], sf_llm["MAPE"], sf_llm["RMAE"],
                round(sf_gnn["MAE"] - sf_llm["MAE"], 4),
                round(sf_gnn["MAPE"] - sf_llm["MAPE"], 4),
                round(sf_gnn["RMAE"] - sf_llm["RMAE"], 4),
            ]

            station_feature_writer.writerow(sf_row)

        gc.collect()

finally:
    detail_f.close()
    station_feature_f.close()
    prediction_compare_f.close()


# ====================== 写全局特征平均 ======================
with open(GLOBAL_FEATURE_MEAN_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(global_feature_mean_header)

    for feature_id in range(num_features):
        gnn_m = global_feature_acc[feature_id]["gnn"].result()
        llm_m = global_feature_acc[feature_id]["llm"].result()

        row = [
            feature_id,
            FEATURE_NAMES[feature_id],
            len(global_feature_acc[feature_id]["station_ids"]),
            global_feature_acc[feature_id]["total_samples"],
            global_feature_acc[feature_id]["gnn"].count,
            global_feature_acc[feature_id]["fallback_count"],
            gnn_m["MAE"], gnn_m["MAPE"], gnn_m["RMAE"],
            llm_m["MAE"], llm_m["MAPE"], llm_m["RMAE"],
            round(gnn_m["MAE"] - llm_m["MAE"], 4),
            round(gnn_m["MAPE"] - llm_m["MAPE"], 4),
            round(gnn_m["RMAE"] - llm_m["RMAE"], 4),
        ]

        writer.writerow(row)


# ====================== 写全局总体平均 ======================
overall_gnn = overall_acc["gnn"].result()
overall_llm = overall_acc["llm"].result()

with open(GLOBAL_OVERALL_MEAN_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(global_overall_header)

    writer.writerow([
        "ALL_STATIONS_ALL_FEATURES",
        len(overall_acc["station_ids"]),
        len(overall_acc["feature_ids"]),
        overall_acc["total_samples"],
        overall_acc["gnn"].count,
        overall_acc["fallback_count"],
        overall_gnn["MAE"], overall_gnn["MAPE"], overall_gnn["RMAE"],
        overall_llm["MAE"], overall_llm["MAPE"], overall_llm["RMAE"],
        round(overall_gnn["MAE"] - overall_llm["MAE"], 4),
        round(overall_gnn["MAPE"] - overall_llm["MAPE"], 4),
        round(overall_gnn["RMAE"] - overall_llm["RMAE"], 4),
    ])


# ====================== 最终打印 ======================
print("\n" + "=" * 100)
print("✅ 全站点推理完成")
print("=" * 100)
print(f"🚫 已跳过站点: 0~{START_FROM_STATION-1}")
print(f"📌 实际推理站点数: {len(overall_acc['station_ids'])} / {num_stations}")
print(f"📁 输出目录: {OUTPUT_DIR}")
print(f"📄 每站点每时段每特征指标: {DETAIL_PATH}")
print(f"📄 每站点每特征平均指标: {STATION_FEATURE_MEAN_PATH}")
print(f"📄 全部站点四特征平均指标: {GLOBAL_FEATURE_MEAN_PATH}")
print(f"📄 全部站点全部特征总平均: {GLOBAL_OVERALL_MEAN_PATH}")
print(f"📄 LLM/GNN预测值对比表: {PREDICTION_COMPARE_PATH}")
print(f"📌 全局 fallback 总数: {overall_acc['fallback_count']}")
print("=" * 100)

gc.collect()
torch.cuda.empty_cache()

print("🎉 全部完成！")