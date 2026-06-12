"""
python 微调513_fixed.py

修复内容：
1. 修复 Unsloth 报错：
   embed_tokens & lm_head not trainable, which will cause NaNs.
2. target_modules 加回 embed_tokens 和 lm_head。
3. 使用标准 SFTTrainer，不使用自定义 hidden_states MAE Trainer，避免 30小时级别超慢训练。
4. 保持你的样本生成逻辑：阈值触发 + COT + 修正值=真实值。
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


# ============================== 日志 ==============================
class Logger(object):
    def __init__(self, filename=os.path.join(PROJECT_ROOT, f"quick513_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")):
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
print("🚀 启动 quick513_fixed 微调")
print("✅ 修复 embed_tokens / lm_head not trainable 报错")
print("=" * 100)
print(f"日志文件: {sys.stdout.log.name}")


# ============================== 路径配置 ==============================
BASE_DATA_PATH = PROJECT_ROOT
FINETUNE_OUTPUT_PATH = PROJECT_ROOT

# 如果你的 npz 已经是 98 站点但仍然叫原名，就保持这样
YTRUE_TRAIN_PATH = os.path.join(FINETUNE_OUTPUT_PATH, "finetune_real_traffic.npz")
YPRED_TRAIN_PATH = os.path.join(FINETUNE_OUTPUT_PATH, "finetune_data.npz")

CARPARK_DES_PATH = os.path.join(BASE_DATA_PATH, "station_list_hngs_98.txt") #索引读取
NATURAL_PATTERN_PATH = os.path.join(BASE_DATA_PATH, "station_natural_list_4feat_98.txt")
WEATHER_DATA_PATH = os.path.join(BASE_DATA_PATH, "98站点天气信息/")#索引读取
EVENTS_CSV_PATH = os.path.join(BASE_DATA_PATH, "events_list_quan.csv")
HIS_DATA_CSV_PATH = os.path.join(BASE_DATA_PATH, "his_data_with_index_98.csv")

DATASET_CACHE_PATH = os.path.join(PROJECT_ROOT, "quick513_fixed.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results-quick513-fixed")
FINAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick513-fixed")


# ============================== 训练配置 ==============================
MAX_SAMPLES = 30000
NUM_EPOCHS = 10

# 加回 embed_tokens/lm_head 后显存会增加。
# 如果显存够，可以改成 TRAIN_BATCH_SIZE=16, GRAD_ACCUM=2。
TRAIN_BATCH_SIZE = 8
GRAD_ACCUM = 4

# 如果你显存非常多，可以试：
# TRAIN_BATCH_SIZE = 16
# GRAD_ACCUM = 2

LORA_R = 32
LORA_ALPHA = 16

# 为了速度默认 False。
# 如果 OOM，改成 True。
USE_GRADIENT_CHECKPOINTING = False

FORCE_REBUILD_DATASET = False
NORMAL_SAMPLE_RATIO = 0.1


# ============================== 加载数据 ==============================
print("\n📂 正在加载数据...")

his_df = pd.read_csv(HIS_DATA_CSV_PATH)
his_df["时间"] = pd.to_datetime(his_df["时间"])
parking_data = his_df.set_index("时间").sort_index()

npz_true = np.load(YTRUE_TRAIN_PATH)
npz_pred = np.load(YPRED_TRAIN_PATH)

print("✓ NPZ文件键名检测:")
print(f" ytrue_train.npz 键名: {list(npz_true.keys())}")
print(f" ypred_train.npz 键名: {list(npz_pred.keys())}")

if "target" in npz_true:
    true_array = npz_true["target"]
elif "arr_0" in npz_true:
    true_array = npz_true["arr_0"]
else:
    raise KeyError("ytrue_train.npz 未找到有效键")

if "prediction" in npz_pred:
    pred_array = npz_pred["prediction"]
elif "arr_0" in npz_pred:
    pred_array = npz_pred["arr_0"]
else:
    raise KeyError("ypred_train.npz 未找到有效键")

print("✓ 流量数据加载成功:")
print(f" true_array shape: {true_array.shape}")
print(f" pred_array shape: {pred_array.shape}")

if np.allclose(true_array, pred_array):
    raise ValueError("❌ 预测值与真实值完全相同！请检查文件路径")
else:
    diff = np.abs(true_array - pred_array)
    print("\n✓ 数据验证通过:")
    print(f" 平均绝对误差 MAE: {diff.mean():.2f}")

print("\n直接使用 his_data_with_index_98.csv 的时间索引")
print(f" 起始时间: {parking_data.index[0]}")
print(f" 结束时间: {parking_data.index[-1]}")
print(f" 总行数: {len(parking_data)}")


# ============================== 加载站点与 pattern ==============================
print(f"\n正在加载站点列表: {CARPARK_DES_PATH}")
with open(CARPARK_DES_PATH, "r", encoding="utf-8") as f:
    carpark_des_list = [l.strip() for l in f if l.strip()]
print(f"✓ 站点列表加载成功: {len(carpark_des_list)} 个站点")

print(f"正在加载自然语言模式: {NATURAL_PATTERN_PATH}")
with open(NATURAL_PATTERN_PATH, "r", encoding="utf-8") as f:
    natural_pattern_list = [l.strip() for l in f if l.strip()]
print(f"✓ 自然语言模式加载成功: {len(natural_pattern_list)} 条描述")


# ============================== 维度对齐 ==============================
print("\n原始数据形状分析:")
print(f" true_array shape: {true_array.shape}")
print(f" pred_array shape: {pred_array.shape}")

if true_array.ndim == 4:
    if true_array.shape[1] == 8 and true_array.shape[3] == 4:
        true_array = true_array.transpose(0, 2, 3, 1)
        pred_array = pred_array.transpose(0, 2, 3, 1)
    elif true_array.shape[1] == 4 and true_array.shape[3] == 8:
        true_array = true_array.transpose(0, 2, 1, 3)
        pred_array = pred_array.transpose(0, 2, 1, 3)

num_samples, num_stations, num_features, seq_len = true_array.shape

print("\n✓ 维度对齐完成:")
print(f" 样本数: {num_samples} | 站点数: {num_stations} | 特征数: {num_features} | 序列长度: {seq_len}")

if len(carpark_des_list) != num_stations:
    raise ValueError(
        f"❌ 站点描述数量与流量数据站点数不一致: "
        f"station_list={len(carpark_des_list)}, num_stations={num_stations}"
    )

if len(natural_pattern_list) != num_stations * num_features:
    raise ValueError(
        f"❌ pattern数量与流量数据不一致: "
        f"pattern={len(natural_pattern_list)}, expected={num_stations * num_features}"
    )

expected_his_rows = num_samples * num_stations
if len(parking_data) < expected_his_rows:
    raise ValueError(
        f"❌ his_data 行数不足: len(parking_data)={len(parking_data)}, "
        f"expected>={expected_his_rows}"
    )

print("✅ station_list / pattern_list / his_data 数量检查通过")


FEATURE_NAMES = {
    0: "Passenger Car Up",
    1: "Passenger Car Down",
    2: "Non-Passenger Car Up",
    3: "Non-Passenger Car Down"
}


# ============================== 阈值配置 ==============================
ERROR_THRESHOLD_BY_FEATURE = {
    0: 100,
    1: 90,
    2: 27,
    3: 27,
}

print("\n" + "=" * 70)
print("📊 阈值触发修正配置:")
print("=" * 70)
for feat_idx, feat_name in FEATURE_NAMES.items():
    threshold = ERROR_THRESHOLD_BY_FEATURE.get(feat_idx, 60)
    print(f"   {feat_name:25s} : 阈值 = {threshold:3d}")
print("=" * 70 + "\n")


# ============================== 天气 ==============================
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

    print(f"✅ 天气预加载完成：{len(weather_dict)} 个站点在内存")
    return weather_dict


weather_cache = preload_all_weather(WEATHER_DATA_PATH, num_stations)


def get_weather_fast(station_idx, date_str):
    return weather_cache.get(station_idx, {}).get(date_str, "Unknown")


# ============================== 事件 ==============================
def load_chinese_events(path):
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
                raise ValueError(f"缺少列 {c}")

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


precise_events_map = load_chinese_events(EVENTS_CSV_PATH)


# ============================== 辅助函数 ==============================
def infer_gdp_level(desc):
    high_gdp_keywords = [
        "city", "urban", "hub", "central", "center",
        "downtown", "high-tech", "economic zone", "business"
    ]

    desc_lower = str(desc).lower()

    for kw in high_gdp_keywords:
        if kw in desc_lower:
            return "High"

    return "Low"


def infer_event_impact(ev_text):
    if ev_text == "None":
        return "none"

    ev_lower = str(ev_text).lower()

    decrease_keywords = [
        "accident", "construction", "congestion", "control",
        "closed", "rain", "snow", "collapse", "road closed",
        "blocked", "delay"
    ]

    increase_keywords = [
        "concert", "match", "game", "activity", "event",
        "fair", "exhibition", "show", "festival"
    ]

    for kw in decrease_keywords:
        if kw in ev_lower:
            return "decrease"

    for kw in increase_keywords:
        if kw in ev_lower:
            return "increase"

    return "fluctuate"


comparison_log = []


def log_comparison(station_name, feature_name, timestamp, gnn_value, true_value, llm_value, error_before, error_after):
    comparison_log.append({
        "station": station_name,
        "feature": feature_name,
        "timestamp": timestamp,
        "gnn_prediction": gnn_value,
        "true_value": true_value,
        "llm_output": llm_value,
        "gnn_error": error_before,
        "llm_error": error_after,
        "improvement": error_before - error_after
    })


# ============================== 加载模型 ==============================
print("\n✓ 正在加载基础模型...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/user/Llama-3.1-8B",
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


# ============================== 生成样本 ==============================
def process_chunk(stations_chunk):
    samples = []

    for j in stations_chunk:
        if j >= len(carpark_des_list):
            continue

        desc = carpark_des_list[j]
        s_name = desc.split()[0]

        for f in range(num_features):
            f_name = FEATURE_NAMES[f]
            is_p = f in (0, 1)

            pi = j * num_features + f
            pat = natural_pattern_list[pi] if pi < len(natural_pattern_list) else "General"

            pred_j = pred_array[:, j, f, :]
            true_j = true_array[:, j, f, :]
            mae_per_sample = np.abs(pred_j - true_j).mean(axis=1)

            feature_threshold = ERROR_THRESHOLD_BY_FEATURE.get(f, 60)

            rows = np.arange(num_samples) * num_stations + j
            rows = rows[rows < len(parking_data)]
            ts = parking_data.index[rows]

            if len(ts) == 0:
                continue

            dates = [t.strftime("%Y-%m-%d") for t in ts]
            is_work = np.array([is_workday(t) for t in ts])
            is_holi = ~is_work
            hours = np.array([t.hour for t in ts])

            events = [precise_events_map.get((d, s_name), "None") for d in dates]
            has_ev = np.array([e != "None" for e in events])

            need_correction = mae_per_sample > feature_threshold
            ev_mask = has_ev
            norm_mask = np.random.rand(len(mae_per_sample)) < NORMAL_SAMPLE_RATIO

            keep = np.where(need_correction | ev_mask | norm_mask)[0]

            if j % 50 == 0 and f == 0 and len(keep) > 0:
                print(
                    f"   站点{j:3d} [{s_name[:15]:15s}] {f_name:20s}: "
                    f"阈值={feature_threshold:3d}, "
                    f"触发修正={need_correction.sum():4d}, "
                    f"事件触发={ev_mask.sum():3d}, "
                    f"随机采样={norm_mask.sum():4d}, "
                    f"总采样={len(keep):4d}"
                )

            for i in keep:
                try:
                    t3 = ts[i]
                    dt = dates[i]
                    h = hours[i]
                    ih = is_holi[i]
                    iw = is_work[i]
                    ev = events[i]
                    w = get_weather_fast(j, dt)

                    gnn_original = pred_j[i]
                    gnn_array = gnn_original.round().astype(int).tolist()

                    true_value_seq = true_j[i]

                    # 保持你的原始逻辑：修正值 = 真实值
                    final_values = true_value_seq.round().astype(int).tolist()
                    final_values = [max(0, int(x)) for x in final_values]

                    is_above_threshold = mae_per_sample[i] > feature_threshold

                    if is_above_threshold:
                        mae_before = np.abs(gnn_original - true_value_seq).mean()
                        mae_after = np.abs(np.array(final_values) - true_value_seq).mean()
                        t3_str = t3.strftime("%Y-%m-%d %H:%M:%S")

                        log_comparison(
                            s_name,
                            f_name,
                            t3_str,
                            gnn_array,
                            true_value_seq.tolist(),
                            final_values,
                            mae_before,
                            mae_after
                        )

                    t4 = t3 + pd.Timedelta(hours=7)
                    t3s = t3.strftime("%Y-%m-%d %H:%M")
                    t4s = t4.strftime("%Y-%m-%d %H:%M")

                    dayt = "Workday" if iw else "Off-day"

                    cot = []
                    gdp_level = infer_gdp_level(desc)

                    if h < 6 or h >= 23:
                        time_period = "Nighttime (Low traffic base)"
                    elif 7 <= h <= 9 or 17 <= h <= 19:
                        time_period = "Peak hours (High traffic base)"
                    else:
                        time_period = "Off-peak daytime (Moderate traffic base)"

                    if is_above_threshold:
                        cot.append(
                            f"1. Economic Profile: Station {s_name} is located in a {gdp_level}-GDP region. "
                            f"This determines the base sensitivity to holiday and economic activities."
                        )

                        cot.append(
                            f"2. Temporal Dynamics: The target window ({t3s} to {t4s}) falls under {time_period}."
                        )

                        if ih:
                            if is_p:
                                surge_magnitude = "significantly high" if gdp_level == "High" else "moderate"
                                cot.append(
                                    f"3. Holiday Effect: Holiday drives passenger car ({f_name}) volume up. "
                                    f"Given the {gdp_level}-GDP region, this upward adjustment magnitude should be {surge_magnitude}."
                                )
                            else:
                                cot.append(
                                    f"3. Holiday Effect: As a freight/commercial feature ({f_name}), "
                                    f"holiday traffic restrictions dictate a drastic reduction in volume."
                                )
                        else:
                            cot.append(
                                "3. Holiday Effect: Routine workday pattern. "
                                "No holiday-induced vehicle type adjustments needed."
                            )

                        if w not in ["晴", "多云", "阴", "Unknown"]:
                            cot.append(
                                f"4. Weather Impact: Adverse weather ({w}) restricts road capacity, "
                                f"causing a drop in traffic volume, which is particularly impactful during {time_period}."
                            )
                        else:
                            cot.append(
                                f"4. Weather Impact: Routine weather ({w}) causes no disruption to baseline capacity."
                            )

                        if ev != "None":
                            impact_dir = infer_event_impact(ev)

                            if impact_dir == "decrease":
                                cot.append(
                                    f"5. Event Disruption: A negative special event ('{ev}') severely throttles typical traffic flow downwards."
                                )
                            elif impact_dir == "increase":
                                cot.append(
                                    f"5. Event Disruption: A positive special event ('{ev}') draws localized traffic, pushing volume upwards."
                                )
                            else:
                                cot.append(
                                    f"5. Event Disruption: Special event ('{ev}') causes significant anomalous fluctuations."
                                )
                        else:
                            cot.append("5. Event Disruption: No special events reported.")

                        cot.append(
                            "6. Strategy: The GNN baseline prediction fails to adequately account for these interacting dynamic factors. "
                            "Overriding with actual calibrated data."
                        )

                        cot.append(f"Final Correction: [{', '.join(map(str, final_values))}]")

                    else:
                        cot.append(f"1. Economic Profile: {gdp_level}-GDP region baseline.")
                        cot.append(f"2. Temporal Dynamics: {time_period}.")
                        cot.append(
                            "3. Conditions Evaluation: Normal weather, no holidays, and no disruptive events detected. "
                            "Traffic is operating under highly predictable standard patterns."
                        )
                        cot.append(
                            "4. Strategy: Standard GNN captures this routine variance perfectly. No manual adjustment required."
                        )
                        cot.append(f"Final Correction: [{', '.join(map(str, final_values))}]")

                    cot_text = " ".join(cot) + " <|eot_id|>"

                    prompt = instruction_text.format(
                        desc,
                        f_name,
                        t3s,
                        t4s,
                        dayt,
                        pat,
                        w,
                        gnn_array,
                        ev if ev != "None" else "None"
                    )

                    full = prompt + cot_text

                    samples.append({
                        "prompt": prompt,
                        "cot_text": cot_text,
                        "full_text": full
                    })

                except Exception:
                    continue

    return samples


def build_dataset_fast():
    n_proc = min(8, cpu_count())
    stations = list(range(num_stations))

    chunk_size = max(1, len(stations) // n_proc)
    chunks = [stations[i:i + chunk_size] for i in range(0, len(stations), chunk_size)]

    print(f"\n🚀 多进程生成：{n_proc} 进程，{len(chunks)} 块")

    with Pool(n_proc) as p:
        res = list(tqdm(p.imap(process_chunk, chunks), total=len(chunks), desc="生成样本"))

    all_samples = []
    for r in res:
        all_samples.extend(r)

    print(f"\n📝 批量 tokenize {len(all_samples)} 个样本...")

    tokenized_samples = []

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

        pl = len(p_enc["input_ids"])
        labs = [-100] * pl + enc["input_ids"][pl:]

        tokenized_samples.append({
            "input_ids": enc["input_ids"],
            "labels": labs,
            "attention_mask": [1] * len(enc["input_ids"]),
        })

    return tokenized_samples


# ============================== 构建或加载数据集 ==============================
print("\n🚀 开始生成/加载训练样本...")

dataset_path = DATASET_CACHE_PATH

if os.path.exists(dataset_path) and not FORCE_REBUILD_DATASET:
    print(f"✅ 发现已存在数据集: {dataset_path}")
    print("   加载已有数据集...")

    with open(dataset_path, "r", encoding="utf-8") as f:
        generated_dataset = json.load(f)

    print(f"✅ 加载完成：{len(generated_dataset)} 条样本")

else:
    print("   生成新数据集...")

    generated_dataset = build_dataset_fast()
    random.shuffle(generated_dataset)

    if len(generated_dataset) > MAX_SAMPLES:
        generated_dataset = generated_dataset[:MAX_SAMPLES]
        print(f"⚠️ 样本数已限制到 {MAX_SAMPLES} 条")

    print(f"\n✅ 样本生成完成：总计 {len(generated_dataset)} 条")

    with open(dataset_path, "w", encoding="utf-8") as f:
        simplified_dataset = [
            {
                "input_ids": d["input_ids"],
                "labels": d["labels"],
                "attention_mask": d["attention_mask"]
            }
            for d in generated_dataset
        ]

        json.dump(simplified_dataset, f, ensure_ascii=False)

    print(f"✅ 数据集已保存至: {dataset_path}")

random.shuffle(generated_dataset)

if len(generated_dataset) > MAX_SAMPLES:
    generated_dataset = generated_dataset[:MAX_SAMPLES]
    print(f"⚠️ 加载缓存后再次限制样本数到 {MAX_SAMPLES}")

print(f"✅ 最终训练样本数: {len(generated_dataset)}")


# ============================== 保存 comparison_log ==============================
if comparison_log:
    comparison_df = pd.DataFrame(comparison_log)
    comparison_df.to_csv(
        os.path.join(PROJECT_ROOT, "comparison_log_quick513_fixed.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n✅ 阈值触发修正统计:")
    print(f"   触发修正样本数: {len(comparison_log)}")
    print(f"   修正前GNN平均MAE: {comparison_df['gnn_error'].mean():.2f}")
    print(f"   修正后平均MAE: {comparison_df['llm_error'].mean():.2f}")
    print(f"   平均改善: {comparison_df['improvement'].mean():.2f}")


# ============================== 训练 ==============================
if generated_dataset:
    ds = Dataset.from_list(generated_dataset)

    # 关键：不要打开 hidden states，否则会非常慢
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

            # 关键修复：必须加回来，否则 Unsloth 报错/警告 NaN
            "embed_tokens",
            "lm_head",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        random_state=3407,
    )

    BATCH_SIZE = TRAIN_BATCH_SIZE

    total_samples = len(generated_dataset)
    steps_per_epoch = max(1, total_samples // (BATCH_SIZE * GRAD_ACCUM))
    total_steps = steps_per_epoch * NUM_EPOCHS

    print("\n📊 训练统计:")
    print(f"   总样本数: {total_samples}")
    print(f"   批次大小: {BATCH_SIZE}")
    print(f"   梯度累积: {GRAD_ACCUM}")
    print(f"   有效批次: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"   训练轮数: {NUM_EPOCHS}")
    print(f"   每轮步数: ~{steps_per_epoch}")
    print(f"   总步数: ~{total_steps}")
    print(f"   LoRA r: {LORA_R}")
    print(f"   LoRA alpha: {LORA_ALPHA}")
    print(f"   gradient_checkpointing: {USE_GRADIENT_CHECKPOINTING}")
    print(f"   输出目录: {RESULTS_DIR}")
    print(f"   模型保存目录: {FINAL_MODEL_DIR}")

    args = TrainingArguments(
        output_dir=RESULTS_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        optim="adamw_8bit",
        learning_rate=5e-5,
        bf16=True,
        logging_steps=10,
        report_to="none",
        save_strategy="epoch",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False} if USE_GRADIENT_CHECKPOINTING else None,
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
            print(f"\n✅ 发现已有 checkpoint: {resume_checkpoint}")
            print("   将从这里恢复训练")
    else:
        print("\n⚠️ 未发现 checkpoint，从头开始训练")

    torch.cuda.empty_cache()
    gc.collect()

    print("\n📊 显存状态:")
    print(f"   总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   已用显存: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(
        f"   空闲估计: "
        f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB"
    )

    print("\n🚀 开始训练 quick513_fixed...")
    print("⚡ 当前配置:")
    print("   - 标准 SFTTrainer")
    print("   - 不启用 output_hidden_states")
    print("   - 已加入 embed_tokens / lm_head，修复 Unsloth NaN 警告")
    print("   - 默认关闭 gradient_checkpointing")
    print("   - 若 OOM，请将 TRAIN_BATCH_SIZE=8, GRAD_ACCUM=4, USE_GRADIENT_CHECKPOINTING=True")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    if comparison_log:
        final_comparison_df = pd.DataFrame(comparison_log)
        final_comparison_df.to_csv(
            os.path.join(PROJECT_ROOT, "final_comparison_log_quick513_fixed.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    print(f"\n💾 保存模型到: {FINAL_MODEL_DIR}")
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    print(f"\n✅ 训练完成！模型已保存至: {FINAL_MODEL_DIR}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

else:
    print("\n❌ 未生成训练样本")
