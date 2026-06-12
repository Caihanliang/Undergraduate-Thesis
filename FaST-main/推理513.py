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
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/config/cai/"
MODEL_PATH = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick513-fixed")

YTRUE_PATH = os.path.join(PROJECT_ROOT, "finetune_real_traffic.npz")
YPRED_PATH = os.path.join(PROJECT_ROOT, "finetune_data.npz")

STATION_LIST = os.path.join(PROJECT_ROOT, "station_list_hngs_98.txt")
PATTERN_LIST = os.path.join(PROJECT_ROOT, "station_natural_list_4feat_98.txt")
HIS_DATA_PATH = os.path.join(PROJECT_ROOT, "his_data_with_index_98.csv")
WEATHER_DATA_PATH = os.path.join(PROJECT_ROOT, "98站点天气信息/")
EVENTS_CSV_PATH = os.path.join(PROJECT_ROOT, "events_list_quan.csv")

# ====================== 推理配置 ======================
DEBUG_MAX_STATIONS = None
DEBUG_MAX_SAMPLES_PER_STATION = None

# 不跳过任何站点
SKIP_STATION_IDS = set()

BATCH_SIZE = 32

FIRST_PASS_MAX_NEW_TOKENS = 32
RETRY_MAX_NEW_TOKENS = 48

FEATURE_NAMES = {
    0: "Passenger Car Up",
    1: "Passenger Car Down",
    2: "Non-Passenger Car Up",
    3: "Non-Passenger Car Down"
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

PREDICTION_COMPARE_PATH = os.path.join(OUTPUT_DIR, "llm_gnn_prediction_compare.csv")

# 断点续推文件
PROGRESS_PATH = os.path.join(OUTPUT_DIR, "completed_station_features.json")


# ====================== 断点工具函数 ======================
def load_completed_keys(progress_path):
    """
    加载已经完成的 station-feature 任务。
    key 格式: "{station_id}_{feature_id}"
    """
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data)
        except Exception as e:
            print(f"⚠️ 断点文件读取失败，将从空进度开始: {progress_path}, error={e}")
            return set()
    return set()


def save_completed_keys(progress_path, completed_keys):
    """
    保存已经完成的 station-feature 任务。
    """
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(completed_keys)), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, progress_path)


def open_csv_append_with_header(path, header):
    """
    append 模式打开 CSV。
    文件不存在或为空时写入表头。
    """
    file_exists = os.path.exists(path)
    need_header = (not file_exists) or (os.path.getsize(path) == 0)

    f = open(path, "a", newline="", encoding="utf-8-sig")
    writer = csv.writer(f)

    if need_header:
        writer.writerow(header)
        f.flush()

    return f, writer


# ====================== 基础工具函数 ======================
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
                c_str = str(c)
                c_lower = c_str.lower()

                if "日" in c_str or "date" in c_lower:
                    d_col = c
                if "天气" in c_str or "weather" in c_lower:
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

    idx = text.lower().rfind("final correction")
    if idx != -1:
        tail = text[idx:]
        nums = re.findall(r"-?\d+", tail)
        if len(nums) >= expected_len:
            return [max(0, int(x)) for x in nums[:expected_len]]

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

if len(stations) != num_stations:
    raise ValueError(f"站点描述数量不匹配: stations={len(stations)}, num_stations={num_stations}")

if len(patterns) != num_stations * num_features:
    raise ValueError(
        f"pattern数量不匹配: patterns={len(patterns)}, expected={num_stations * num_features}"
    )

his_df = pd.read_csv(HIS_DATA_PATH)
his_df["时间"] = pd.to_datetime(his_df["时间"])
parking_data = his_df.set_index("时间").sort_index()

if len(parking_data) < num_samples * num_stations:
    raise ValueError(
        f"his_data 行数不足: len={len(parking_data)}, expected>={num_samples * num_stations}"
    )

weather_cache = preload_all_weather(WEATHER_DATA_PATH, num_stations)
event_map = load_events(EVENTS_CSV_PATH)


# ====================== 生成实际推理站点列表 ======================
if DEBUG_MAX_STATIONS is None:
    station_range = list(range(num_stations))
else:
    station_range = list(range(min(DEBUG_MAX_STATIONS, num_stations)))

print(f"🚫 跳过站点: 无")
print(f"🚀 实际推理站点数: {len(station_range)} / {num_stations}")
print(f"📁 输出目录: {OUTPUT_DIR}")
print(f"📌 断点文件: {PROGRESS_PATH}")
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


# ====================== 主循环 ======================
print("\n🚀 开始全站点推理...\n")

completed_keys = load_completed_keys(PROGRESS_PATH)
print(f"🔁 已加载断点进度: {len(completed_keys)} 个 station-feature 已完成")

detail_f, detail_writer = open_csv_append_with_header(DETAIL_PATH, detail_header)
station_feature_f, station_feature_writer = open_csv_append_with_header(
    STATION_FEATURE_MEAN_PATH,
    station_feature_mean_header
)
prediction_compare_f, prediction_compare_writer = open_csv_append_with_header(
    PREDICTION_COMPARE_PATH,
    prediction_compare_header
)

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
            task_key = f"{station_id}_{feature_id}"

            if task_key in completed_keys:
                print(f"⏭️ 跳过已完成任务: station={station_id}, feature={feature_id}")
                continue

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

                station_feature_gnn_acc.update(true_seq, gnn_seq)
                station_feature_llm_acc.update(true_seq, llm_seq)

            detail_writer.writerows(detail_rows_buffer)
            prediction_compare_writer.writerows(prediction_compare_rows_buffer)

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

            detail_f.flush()
            station_feature_f.flush()
            prediction_compare_f.flush()

            completed_keys.add(task_key)
            save_completed_keys(PROGRESS_PATH, completed_keys)

            print(f"✅ 已完成并保存断点: station={station_id}, feature={feature_id}")

        gc.collect()

finally:
    detail_f.close()
    station_feature_f.close()
    prediction_compare_f.close()


# ====================== 从 prediction_compare 重新计算全局汇总 ======================
def parse_seq_cell(x):
    if isinstance(x, list):
        return np.asarray(x, dtype=float)
    try:
        return np.asarray(json.loads(x), dtype=float)
    except Exception:
        nums = re.findall(r"-?\d+", str(x))
        return np.asarray([float(v) for v in nums], dtype=float)


def recompute_global_from_prediction_compare():
    if not os.path.exists(PREDICTION_COMPARE_PATH):
        print("⚠️ prediction_compare 文件不存在，无法重新计算全局汇总")
        return

    print("\n🔄 从 llm_gnn_prediction_compare.csv 重新计算全局指标...")

    df = pd.read_csv(PREDICTION_COMPARE_PATH)

    if len(df) == 0:
        print("⚠️ prediction_compare 为空")
        return

    feature_acc = {
        f: {
            "gnn": MetricAccumulator(),
            "llm": MetricAccumulator(),
            "fallback_count": 0,
            "total_samples": 0,
            "station_ids": set(),
        }
        for f in range(num_features)
    }

    overall = {
        "gnn": MetricAccumulator(),
        "llm": MetricAccumulator(),
        "fallback_count": 0,
        "total_samples": 0,
        "station_ids": set(),
        "feature_ids": set(),
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="重算全局指标"):
        station_id = int(row["station_id"])
        feature_id = int(row["feature_id"])
        is_fallback = int(row["is_fallback"])

        true_seq = parse_seq_cell(row["true_seq"])
        gnn_seq = parse_seq_cell(row["gnn_seq"])
        llm_seq = parse_seq_cell(row["llm_seq"])

        feature_acc[feature_id]["gnn"].update(true_seq, gnn_seq)
        feature_acc[feature_id]["llm"].update(true_seq, llm_seq)
        feature_acc[feature_id]["fallback_count"] += is_fallback
        feature_acc[feature_id]["total_samples"] += 1
        feature_acc[feature_id]["station_ids"].add(station_id)

        overall["gnn"].update(true_seq, gnn_seq)
        overall["llm"].update(true_seq, llm_seq)
        overall["fallback_count"] += is_fallback
        overall["total_samples"] += 1
        overall["station_ids"].add(station_id)
        overall["feature_ids"].add(feature_id)

    with open(GLOBAL_FEATURE_MEAN_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(global_feature_mean_header)

        for feature_id in range(num_features):
            gnn_m = feature_acc[feature_id]["gnn"].result()
            llm_m = feature_acc[feature_id]["llm"].result()

            writer.writerow([
                feature_id,
                FEATURE_NAMES[feature_id],
                len(feature_acc[feature_id]["station_ids"]),
                feature_acc[feature_id]["total_samples"],
                feature_acc[feature_id]["gnn"].count,
                feature_acc[feature_id]["fallback_count"],
                gnn_m["MAE"], gnn_m["MAPE"], gnn_m["RMAE"],
                llm_m["MAE"], llm_m["MAPE"], llm_m["RMAE"],
                round(gnn_m["MAE"] - llm_m["MAE"], 4),
                round(gnn_m["MAPE"] - llm_m["MAPE"], 4),
                round(gnn_m["RMAE"] - llm_m["RMAE"], 4),
            ])

    overall_gnn = overall["gnn"].result()
    overall_llm = overall["llm"].result()

    with open(GLOBAL_OVERALL_MEAN_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(global_overall_header)

        writer.writerow([
            "ALL_STATIONS_ALL_FEATURES",
            len(overall["station_ids"]),
            len(overall["feature_ids"]),
            overall["total_samples"],
            overall["gnn"].count,
            overall["fallback_count"],
            overall_gnn["MAE"], overall_gnn["MAPE"], overall_gnn["RMAE"],
            overall_llm["MAE"], overall_llm["MAPE"], overall_llm["RMAE"],
            round(overall_gnn["MAE"] - overall_llm["MAE"], 4),
            round(overall_gnn["MAPE"] - overall_llm["MAPE"], 4),
            round(overall_gnn["RMAE"] - overall_llm["RMAE"], 4),
        ])

    print("✅ 全局指标重新计算完成")


recompute_global_from_prediction_compare()


# ====================== 最终打印 ======================
completed_keys = load_completed_keys(PROGRESS_PATH)
expected_total_tasks = len(station_range) * num_features

print("\n" + "=" * 100)
print("✅ 推理流程结束")
print("=" * 100)
print(f"🚫 已跳过站点: 无")
print(f"📌 站点数: {num_stations}")
print(f"📌 理论 station-feature 总任务数: {expected_total_tasks}")
print(f"📌 已完成 station-feature 任务数: {len(completed_keys)}")
print(f"📁 输出目录: {OUTPUT_DIR}")
print(f"📄 每站点每时段每特征指标: {DETAIL_PATH}")
print(f"📄 每站点每特征平均指标: {STATION_FEATURE_MEAN_PATH}")
print(f"📄 全部站点四特征平均指标: {GLOBAL_FEATURE_MEAN_PATH}")
print(f"📄 全部站点全部特征总平均: {GLOBAL_OVERALL_MEAN_PATH}")
print(f"📄 LLM/GNN预测值对比表: {PREDICTION_COMPARE_PATH}")
print(f"📌 断点文件: {PROGRESS_PATH}")

if len(completed_keys) < expected_total_tasks:
    print("⚠️ 当前还没有全部推理完成。下次重新运行会自动从断点继续。")
else:
    print("🎉 所有 station-feature 任务均已完成！")

print("=" * 100)

gc.collect()
torch.cuda.empty_cache()

print("🎉 全部完成！")
