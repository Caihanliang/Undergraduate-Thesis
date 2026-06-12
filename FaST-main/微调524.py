"""
python 微调524.py

目标：
1. 解决 LLM 微调后过度修正的问题。
2. 让模型学会：GNN 准时保持 GNN，GNN 明显错时才保守修正。
3. 避免预测值突然跳大/跳小。
4. 训练输出与推理输出格式一致：只输出 Final Correction。
5. 保留天气、事件、GDP、时段等上下文信息，但不再使用长 COT。
"""

import os

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["UNSLOTH_SKIP_INIT_CHECK"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import re
import sys
import glob
import json
import random
import logging
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count

import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from chinese_calendar import is_workday

warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


# ============================== 项目路径 ==============================
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/config/cai/"

BASE_MODEL_PATH = "/home/user/Llama-3.1-8B"


# ============================== 日志 ==============================
class Logger(object):
    def __init__(
        self,
        filename=os.path.join(
            PROJECT_ROOT,
            f"quick524_conservative_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
    ):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger()

print("=" * 100)
print("🚀 启动 quick24_conservative 微调")
print("核心策略：GNN 准则保持，GNN 明显错才保守修正，避免过度修正")
print("=" * 100)
print(f"日志文件: {sys.stdout.log.name}")


# ============================== 路径配置 ==============================
BASE_DATA_PATH = PROJECT_ROOT
FINETUNE_OUTPUT_PATH = PROJECT_ROOT

YTRUE_TRAIN_PATH = os.path.join(FINETUNE_OUTPUT_PATH, "finetune_real_traffic.npz")
YPRED_TRAIN_PATH = os.path.join(FINETUNE_OUTPUT_PATH, "finetune_data.npz")

CARPARK_DES_PATH = os.path.join(BASE_DATA_PATH, "station_list_hngs_98.txt")
NATURAL_PATTERN_PATH = os.path.join(BASE_DATA_PATH, "station_natural_list_4feat_98.txt")
WEATHER_DATA_PATH = os.path.join(BASE_DATA_PATH, "98站点天气信息/")
EVENTS_CSV_PATH = os.path.join(BASE_DATA_PATH, "events_list_quan.csv")
HIS_DATA_CSV_PATH = os.path.join(BASE_DATA_PATH, "his_data_with_index_98.csv")

DATASET_CACHE_PATH = os.path.join(PROJECT_ROOT, "quick524_conservative.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results-quick524-conservative")
FINAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick524-conservative")


# ============================== 训练配置 ==============================
# 不建议 10 epoch，容易过拟合并强化过度修正
MAX_SAMPLES = 22000
NUM_EPOCHS = 3
MAX_TRAIN_STEPS = 2200

TRAIN_BATCH_SIZE = 4
GRAD_ACCUM = 8

LORA_R = 16
LORA_ALPHA = 32

USE_GRADIENT_CHECKPOINTING = True

# 第一次建议 True，确保重新生成“保守修正”数据集
FORCE_REBUILD_DATASET = True

RANDOM_SEED = 3407
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================== 采样配置 ==============================
# GNN 本来很不错，所以正常样本一定要保留一部分，让模型学会“不改”
NORMAL_KEEP_RATIO = 0.20

# 事件样本全部保留
EVENT_KEEP_RATIO = 1.00

# 恶劣天气样本保留概率
BAD_WEATHER_KEEP_RATIO = 0.60


# ============================== 特征名 ==============================
FEATURE_NAMES = {
    0: "Passenger Car Up",
    1: "Passenger Car Down",
    2: "Non-Passenger Car Up",
    3: "Non-Passenger Car Down"
}


# ============================== 保守修正配置 ==============================
# 判断 GNN 是否明显错误的阈值
ABS_ERROR_THRESHOLD_BY_FEATURE = {
    0: 90,
    1: 90,
    2: 25,
    3: 25,
}

REL_ERROR_THRESHOLD_BY_FEATURE = {
    0: 0.12,
    1: 0.12,
    2: 0.18,
    3: 0.18,
}

# 修正强度 beta：越大越靠近真实值
# 客车特征之前 LLM 容易变差，所以 beta 低
# 非客车特征 LLM 更有价值，所以 beta 稍高
CORRECTION_BETA_BY_FEATURE = {
    0: 0.30,
    1: 0.30,
    2: 0.60,
    3: 0.60,
}

# 每个点最大允许修正幅度
MAX_ABS_DELTA_BY_FEATURE = {
    0: 80,
    1: 80,
    2: 35,
    3: 35,
}

MAX_REL_DELTA_BY_FEATURE = {
    0: 0.15,
    1: 0.15,
    2: 0.35,
    3: 0.35,
}


# ============================== 数据加载 ==============================
print("\n📂 正在加载数据...")

his_df = pd.read_csv(HIS_DATA_CSV_PATH)
his_df["时间"] = pd.to_datetime(his_df["时间"])
parking_data = his_df.set_index("时间").sort_index()

npz_true = np.load(YTRUE_TRAIN_PATH)
npz_pred = np.load(YPRED_TRAIN_PATH)

print("✓ NPZ 文件键名检测:")
print(f" true npz keys: {list(npz_true.keys())}")
print(f" pred npz keys: {list(npz_pred.keys())}")

if "target" in npz_true:
    true_array = npz_true["target"]
elif "arr_0" in npz_true:
    true_array = npz_true["arr_0"]
else:
    raise KeyError("真实值 npz 未找到 target 或 arr_0")

if "prediction" in npz_pred:
    pred_array = npz_pred["prediction"]
elif "arr_0" in npz_pred:
    pred_array = npz_pred["arr_0"]
else:
    raise KeyError("预测值 npz 未找到 prediction 或 arr_0")

print(f"true_array shape: {true_array.shape}")
print(f"pred_array shape: {pred_array.shape}")

if true_array.ndim == 4:
    if true_array.shape[1] == 8 and true_array.shape[3] == 4:
        true_array = true_array.transpose(0, 2, 3, 1)
        pred_array = pred_array.transpose(0, 2, 3, 1)
    elif true_array.shape[1] == 4 and true_array.shape[3] == 8:
        true_array = true_array.transpose(0, 2, 1, 3)
        pred_array = pred_array.transpose(0, 2, 1, 3)

num_samples, num_stations, num_features, seq_len = true_array.shape

print("\n✓ 维度对齐完成:")
print(f"样本数={num_samples}, 站点数={num_stations}, 特征数={num_features}, 序列长度={seq_len}")

if np.allclose(true_array, pred_array):
    raise ValueError("❌ 预测值和真实值完全相同，请检查数据文件")

diff = np.abs(true_array - pred_array)
print(f"✓ GNN 初始平均 MAE: {diff.mean():.4f}")


# ============================== 加载站点和 pattern ==============================
with open(CARPARK_DES_PATH, "r", encoding="utf-8") as f:
    carpark_des_list = [line.strip() for line in f if line.strip()]

with open(NATURAL_PATTERN_PATH, "r", encoding="utf-8") as f:
    natural_pattern_list = [line.strip() for line in f if line.strip()]

if len(carpark_des_list) != num_stations:
    raise ValueError(
        f"站点描述数量不匹配: station_list={len(carpark_des_list)}, num_stations={num_stations}"
    )

if len(natural_pattern_list) != num_stations * num_features:
    raise ValueError(
        f"pattern 数量不匹配: pattern={len(natural_pattern_list)}, expected={num_stations * num_features}"
    )

expected_his_rows = num_samples * num_stations
if len(parking_data) < expected_his_rows:
    raise ValueError(
        f"his_data 行数不足: len={len(parking_data)}, expected>={expected_his_rows}"
    )

print("✅ station_list / pattern_list / his_data 检查通过")


# ============================== GDP 解析 ==============================
def parse_gdp_from_description(desc):
    pattern = r"GDP\s+is\s+([\d,]+\.?\d*)\s+billion\s+yuan"
    match = re.search(pattern, str(desc), re.IGNORECASE)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except Exception:
            return None
    return None


def infer_gdp_level(desc, threshold=1000.0):
    gdp = parse_gdp_from_description(desc)
    if gdp is None:
        desc_lower = str(desc).lower()
        if any(k in desc_lower for k in ["city", "urban", "hub", "central", "center"]):
            return "High", None
        return "Low", None

    return ("High" if gdp >= threshold else "Low"), gdp


# ============================== 天气处理 ==============================
WEATHER_SEVERITY_MAP = {
    "晴": 0,
    "多云": 0,
    "阴": 1,
    "小雨": 2,
    "阵雨": 2,
    "雾": 2,
    "霾": 2,
    "中雨": 3,
    "大雨": 4,
    "暴雨": 5,
    "小雪": 3,
    "中雪": 4,
    "大雪": 5,
    "暴雪": 5,
    "冰雹": 5,
}


def get_weather_severity(weather_str):
    if not weather_str or weather_str == "Unknown":
        return 0

    w = str(weather_str)

    if w in WEATHER_SEVERITY_MAP:
        return WEATHER_SEVERITY_MAP[w]

    wl = w.lower()

    if any(k in wl for k in ["暴雨", "大雨", "heavy rain", "rainstorm"]):
        return 4
    if any(k in wl for k in ["中雨", "moderate rain"]):
        return 3
    if any(k in wl for k in ["小雨", "light rain", "drizzle"]):
        return 2
    if any(k in wl for k in ["大雪", "暴雪", "heavy snow", "blizzard"]):
        return 5
    if any(k in wl for k in ["中雪", "moderate snow"]):
        return 4
    if any(k in wl for k in ["小雪", "light snow"]):
        return 3
    if any(k in wl for k in ["雾", "fog", "霾", "haze"]):
        return 2
    if any(k in wl for k in ["阴", "overcast"]):
        return 1

    return 0


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

            if d_col is None or w_col is None:
                continue

            df[d_col] = pd.to_datetime(df[d_col]).dt.date.astype(str)
            df = df.drop_duplicates(subset=[d_col]).set_index(d_col)

            weather_dict[j] = df[w_col].astype(str).to_dict()

        except Exception:
            continue

    print(f"✅ 天气预加载完成：{len(weather_dict)} 个站点")
    return weather_dict


weather_cache = preload_all_weather(WEATHER_DATA_PATH, num_stations)


def get_weather_fast(station_idx, date_str):
    return weather_cache.get(station_idx, {}).get(date_str, "Unknown")


# ============================== 事件处理 ==============================
def load_events(path):
    if not os.path.exists(path):
        print(f"⚠️ 事件文件不存在: {path}")
        return {}

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, encoding="gb18030")

    date_col = "日期"
    station_col = "站点名称"
    event_col = "事件描述"

    for c in [date_col, station_col, event_col]:
        if c not in df.columns:
            raise ValueError(f"事件文件缺少列: {c}")

    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    df[station_col] = df[station_col].astype(str).str.strip()
    df[event_col] = df[event_col].astype(str).str.strip()

    event_map = {}

    for _, r in df.iterrows():
        event_map[(r[date_col], r[station_col])] = r[event_col]

    print(f"✅ 事件加载完成：{len(event_map)} 条")
    return event_map


event_map = load_events(EVENTS_CSV_PATH)


def classify_event(ev_text, is_passenger):
    if ev_text == "None" or not ev_text:
        return "none"

    ev = str(ev_text).lower()

    holiday_keys = ["festival", "holiday", "national day", "spring festival", "mid-autumn"]
    incident_keys = ["accident", "construction", "congestion", "control", "closed", "blocked", "delay"]
    activity_keys = ["concert", "match", "game", "activity", "exhibition", "show"]

    if any(k in ev for k in holiday_keys):
        return "passenger_increase_freight_decrease"

    if any(k in ev for k in incident_keys):
        return "decrease"

    if any(k in ev for k in activity_keys):
        return "increase" if is_passenger else "fluctuate"

    return "fluctuate"


# ============================== 修正目标生成 ==============================
def is_abnormal_sequence(seq):
    seq = np.asarray(seq, dtype=float)

    if np.any(np.isnan(seq)) or np.any(np.isinf(seq)):
        return True

    if np.any(seq < 0):
        return True

    if np.max(seq) > 10000:
        return True

    return False


def should_correct_sample(gnn_seq, true_seq, feature_idx):
    gnn_seq = np.asarray(gnn_seq, dtype=float)
    true_seq = np.asarray(true_seq, dtype=float)

    mae = np.mean(np.abs(gnn_seq - true_seq))
    mean_true = np.mean(true_seq)
    rel = mae / (mean_true + 1e-6)

    abs_th = ABS_ERROR_THRESHOLD_BY_FEATURE.get(feature_idx, 60)
    rel_th = REL_ERROR_THRESHOLD_BY_FEATURE.get(feature_idx, 0.15)

    return (mae > abs_th) and (rel > rel_th), mae, rel


def make_conservative_target(gnn_seq, true_seq, feature_idx, force_keep=False):
    """
    核心函数：
    - force_keep=True 或 GNN误差不大：直接输出 GNN
    - 否则：只向真实值方向移动一部分，并限制最大修正幅度
    """
    gnn_seq = np.asarray(gnn_seq, dtype=float)
    true_seq = np.asarray(true_seq, dtype=float)

    if force_keep:
        target = gnn_seq.copy()
        return np.maximum(np.rint(target), 0).astype(int).tolist()

    beta = CORRECTION_BETA_BY_FEATURE.get(feature_idx, 0.5)

    raw_target = gnn_seq + beta * (true_seq - gnn_seq)

    max_abs_delta = MAX_ABS_DELTA_BY_FEATURE.get(feature_idx, 50)
    max_rel_delta = MAX_REL_DELTA_BY_FEATURE.get(feature_idx, 0.25)

    allowed_delta = np.maximum(max_abs_delta, np.abs(gnn_seq) * max_rel_delta)

    lower = gnn_seq - allowed_delta
    upper = gnn_seq + allowed_delta

    target = np.clip(raw_target, lower, upper)
    target = np.maximum(np.rint(target), 0).astype(int)

    return target.tolist()


# ============================== 加载模型 ==============================
print("\n🔄 加载基础模型...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_PATH,
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    device_map={"": 0},
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ============================== Prompt 模板 ==============================
instruction_text = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a conservative highway traffic flow refiner.\n"
    "Your task is to refine GNN traffic predictions safely.\n"
    "Important rules:\n"
    "1. If the GNN prediction is reasonable, keep it almost unchanged.\n"
    "2. Avoid over-correction.\n"
    "3. Avoid abnormal jumps between adjacent time steps.\n"
    "4. Output only: Final Correction: [8 non-negative integers].<|eot_id|>"
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


# ============================== 样本生成 ==============================
def process_chunk(stations_chunk):
    samples = []
    local_log = []

    for station_id in stations_chunk:
        if station_id >= len(carpark_des_list):
            continue

        desc = carpark_des_list[station_id]
        station_short = desc.split()[0]

        gdp_level, gdp_value = infer_gdp_level(desc)

        rows = np.arange(num_samples) * num_stations + station_id
        rows = rows[rows < len(parking_data)]
        station_ts = parking_data.index[rows]

        if len(station_ts) == 0:
            continue

        dates = [t.strftime("%Y-%m-%d") for t in station_ts]
        is_work_arr = np.array([is_workday(t) for t in station_ts])
        hours = np.array([t.hour for t in station_ts])

        for feature_idx in range(num_features):
            feature_name = FEATURE_NAMES[feature_idx]
            is_passenger = feature_idx in [0, 1]

            pattern_idx = station_id * num_features + feature_idx
            pattern = natural_pattern_list[pattern_idx] if pattern_idx < len(natural_pattern_list) else "General"

            gnn_all = pred_array[:, station_id, feature_idx, :]
            true_all = true_array[:, station_id, feature_idx, :]

            keep_indices = []

            for i in range(min(len(station_ts), num_samples)):
                gnn_seq = np.rint(gnn_all[i]).astype(int)
                true_seq = np.rint(true_all[i]).astype(int)

                if is_abnormal_sequence(gnn_seq) or is_abnormal_sequence(true_seq):
                    continue

                need_correct, mae, rel = should_correct_sample(gnn_seq, true_seq, feature_idx)

                date_str = dates[i]
                weather = get_weather_fast(station_id, date_str)
                weather_sev = get_weather_severity(weather)

                event_text = event_map.get((date_str, station_short), "None")
                has_event = event_text != "None"

                keep = False

                if need_correct:
                    keep = True
                elif has_event and random.random() < EVENT_KEEP_RATIO:
                    keep = True
                elif weather_sev >= 3 and random.random() < BAD_WEATHER_KEEP_RATIO:
                    keep = True
                elif random.random() < NORMAL_KEEP_RATIO:
                    keep = True

                if keep:
                    keep_indices.append((i, need_correct, mae, rel, weather, weather_sev, event_text))

            if station_id % 30 == 0 and feature_idx == 0:
                print(
                    f"站点{station_id:3d} [{station_short:15s}] "
                    f"{feature_name:22s}: 采样 {len(keep_indices):4d}"
                )

            for item in keep_indices:
                i, need_correct, mae, rel, weather, weather_sev, event_text = item

                try:
                    ts = station_ts[i]
                    h = hours[i]
                    iw = is_work_arr[i]
                    day_type = "Workday" if iw else "Off-day"

                    t3s = ts.strftime("%Y-%m-%d %H:%M")
                    t4s = (ts + pd.Timedelta(hours=7)).strftime("%Y-%m-%d %H:%M")

                    gnn_seq = np.rint(gnn_all[i]).astype(int)
                    true_seq = np.rint(true_all[i]).astype(int)

                    # 核心：正常样本保持 GNN，高误差样本保守修正
                    final_values = make_conservative_target(
                        gnn_seq=gnn_seq,
                        true_seq=true_seq,
                        feature_idx=feature_idx,
                        force_keep=not need_correct
                    )

                    prompt = instruction_text.format(
                        desc,
                        feature_name,
                        t3s,
                        t4s,
                        day_type,
                        pattern,
                        weather,
                        gnn_seq.tolist(),
                        event_text if event_text != "None" else "None"
                    )

                    answer = f"Final Correction: [{', '.join(map(str, final_values))}] <|eot_id|>"
                    full_text = prompt + answer

                    samples.append({
                        "prompt": prompt,
                        "full_text": full_text
                    })

                    if need_correct:
                        local_log.append({
                            "station_id": station_id,
                            "station": station_short,
                            "feature": feature_name,
                            "timestamp": str(ts),
                            "mae": float(mae),
                            "rel_error": float(rel),
                            "gnn": gnn_seq.tolist(),
                            "true": true_seq.tolist(),
                            "target": final_values,
                        })

                except Exception:
                    continue

    return samples, local_log


def build_dataset_fast():
    n_proc = min(8, cpu_count())
    station_ids = list(range(num_stations))

    chunk_size = max(1, len(station_ids) // n_proc)
    chunks = [station_ids[i:i + chunk_size] for i in range(0, len(station_ids), chunk_size)]

    print(f"\n🚀 多进程生成样本：{n_proc} 进程，{len(chunks)} 块")

    all_samples = []
    all_logs = []

    with Pool(n_proc) as pool:
        for samples_part, logs_part in tqdm(
            pool.imap(process_chunk, chunks),
            total=len(chunks),
            desc="生成样本"
        ):
            all_samples.extend(samples_part)
            all_logs.extend(logs_part)

    print(f"生成原始样本数: {len(all_samples)}")

    random.shuffle(all_samples)

    if len(all_samples) > MAX_SAMPLES:
        all_samples = all_samples[:MAX_SAMPLES]

    print(f"截断后训练样本数: {len(all_samples)}")

    if all_logs:
        log_df = pd.DataFrame(all_logs)
        log_path = os.path.join(PROJECT_ROOT, "quick524_conservative_sampling_log.csv")
        log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
        print(f"采样日志已保存: {log_path}")

    tokenized_samples = []

    print("📝 Tokenizing...")

    for sample in tqdm(all_samples, desc="Tokenizing"):
        full = sample["full_text"]
        prompt = sample["prompt"]

        enc = tokenizer(
            full,
            truncation=True,
            max_length=1024,
            add_special_tokens=False
        )

        p_enc = tokenizer(
            prompt,
            add_special_tokens=False
        )

        prompt_len = len(p_enc["input_ids"])
        labels = [-100] * prompt_len + enc["input_ids"][prompt_len:]

        tokenized_samples.append({
            "input_ids": enc["input_ids"],
            "labels": labels,
            "attention_mask": [1] * len(enc["input_ids"]),
        })

    return tokenized_samples


# ============================== 构建或加载数据集 ==============================
print("\n🚀 开始生成/加载训练样本...")

if os.path.exists(DATASET_CACHE_PATH) and not FORCE_REBUILD_DATASET:
    print(f"发现已有数据集缓存: {DATASET_CACHE_PATH}")
    with open(DATASET_CACHE_PATH, "r", encoding="utf-8") as f:
        generated_dataset = json.load(f)
    print(f"加载缓存样本数: {len(generated_dataset)}")
else:
    generated_dataset = build_dataset_fast()
    with open(DATASET_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(generated_dataset, f, ensure_ascii=False)
    print(f"数据集缓存已保存: {DATASET_CACHE_PATH}")

random.shuffle(generated_dataset)

if len(generated_dataset) > MAX_SAMPLES:
    generated_dataset = generated_dataset[:MAX_SAMPLES]

print(f"最终训练样本数: {len(generated_dataset)}")


# ============================== 训练 ==============================
if not generated_dataset:
    raise RuntimeError("❌ 未生成训练样本")

ds = Dataset.from_list(generated_dataset)

model.config.output_hidden_states = False

print("\n🔧 配置 LoRA...")

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    random_state=RANDOM_SEED,
)

steps_per_epoch = max(1, len(generated_dataset) // (TRAIN_BATCH_SIZE * GRAD_ACCUM))
estimated_steps = steps_per_epoch * NUM_EPOCHS
actual_steps = min(estimated_steps, MAX_TRAIN_STEPS)

print("\n📊 训练统计:")
print(f"样本数: {len(generated_dataset)}")
print(f"batch size: {TRAIN_BATCH_SIZE}")
print(f"grad accum: {GRAD_ACCUM}")
print(f"effective batch: {TRAIN_BATCH_SIZE * GRAD_ACCUM}")
print(f"epochs: {NUM_EPOCHS}")
print(f"每轮步数约: {steps_per_epoch}")
print(f"最大训练步数: {MAX_TRAIN_STEPS}")
print(f"实际训练步数约: {actual_steps}")
print(f"LoRA r: {LORA_R}")
print(f"LoRA alpha: {LORA_ALPHA}")
print(f"gradient checkpointing: {USE_GRADIENT_CHECKPOINTING}")
print(f"输出目录: {RESULTS_DIR}")
print(f"最终模型目录: {FINAL_MODEL_DIR}")

args = TrainingArguments(
    output_dir=RESULTS_DIR,
    num_train_epochs=NUM_EPOCHS,
    max_steps=MAX_TRAIN_STEPS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    optim="paged_adamw_8bit",
    learning_rate=5e-5,
    bf16=True,
    logging_steps=10,
    report_to="none",
    save_strategy="steps",
    save_steps=500,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    save_only_model=True,
    use_cpu=False,
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    tokenizer=tokenizer,
    args=args,
    max_seq_length=1024,
    packing=False,
)

checkpoints = glob.glob(os.path.join(RESULTS_DIR, "checkpoint-*"))
resume_checkpoint = None

if checkpoints:
    checkpoint_nums = []
    for c in checkpoints:
        try:
            checkpoint_nums.append(int(c.split("-")[-1]))
        except Exception:
            pass

    if checkpoint_nums:
        max_num = max(checkpoint_nums)
        resume_checkpoint = os.path.join(RESULTS_DIR, f"checkpoint-{max_num}")
        print(f"发现 checkpoint: {resume_checkpoint}")
else:
    print("未发现 checkpoint，从头训练")

torch.cuda.empty_cache()
gc.collect()

print("\n🚀 开始训练...")
trainer.train(resume_from_checkpoint=resume_checkpoint)

print(f"\n💾 保存模型到: {FINAL_MODEL_DIR}")
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print("\n✅ 训练完成")
print(f"模型已保存至: {FINAL_MODEL_DIR}")

del model, trainer
gc.collect()
torch.cuda.empty_cache()
