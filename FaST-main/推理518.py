import os
import re
import gc
import csv
import json
import torch
import warnings
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from unsloth import FastLanguageModel
from chinese_calendar import is_workday

# ====================== 环境设置 ======================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# 【优化】启用cuDNN自动调优，加速卷积和矩阵运算
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # RTX PRO 6000支持TF32

# ====================== 日志配置 ======================
# 创建日志目录
LOG_DIR = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/inference_results_all_stations_with_predictions/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志文件
log_file = os.path.join(LOG_DIR, f"inference_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"📝 日志文件已创建: {log_file}")

# ====================== 路径配置 ======================
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/config/cai/"
# 【修复】使用优化版微调模型的保存路径
MODEL_PATH = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick518-optimized")

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

# 【新增】诊断模式：打印前几个样本的原始输出
# 【优化】生产环境建议关闭，节省I/O开销
DEBUG_MODE = False  # 从True改为False
DEBUG_MODE = True  # 从True改为False

DEBUG_PRINT_COUNT = 3

# 不跳过任何站点
SKIP_STATION_IDS = set()

# 【紧急加速】基于RTX PRO 6000 Blackwell (96GB显存) 优化batch size
# 
# 配置分析：
# - max_new_tokens: 128（简化CoT，约3-4步分析）
# - BATCH_SIZE: 128（充分利用96GB显存）
# - 每任务batches: ceil(850/128) = 7 batches
# - 单batch耗时: ~15-18秒（生成128 tokens）
# - 每任务耗时: 7 × 16.5 = 115.5秒 ≈ 1.9分钟
# - 总任务数: 98站点 × 4特征 = 392
# - 理论总时间: 392 × 1.9分钟 ≈ 12.4小时
# - 实际预估: 14-16小时（考虑I/O、断点保存等开销）
# 
# 相比原配置（39小时）的改进：
# - max_new_tokens: 256 → 128（减少50%）
# - BATCH_SIZE: 64 → 128（增加100%）
# - 预期加速比: 2.5-3倍
# 【紧急修复】降低BATCH_SIZE以避免OOM
# 
# 问题分析：
# - BATCH_SIZE=128 + max_new_tokens=128 → 显存需求约90GB+
# - RTX PRO 6000有96GB，但其他进程占用约71GB（14+28.59+28.84）
# - 可用显存仅约23GB，不足以支撑BATCH_SIZE=128
# 
# 解决方案：
# - 降低BATCH_SIZE到64（回到之前的稳定配置）
# - 保持max_new_tokens=128（简化CoT）
# - 预期推理时间：从14-16小时增加到20-22小时
# 
# 配置分析：
# - BATCH_SIZE = 64
# - 每任务batches: ceil(850/64) = 14 batches
# - 单batch耗时: ~18秒（生成128 tokens）
# - 每任务耗时: 14 × 18 = 252秒 ≈ 4.2分钟
# - 总任务数: 392
# - 理论总时间: 392 × 4.2分钟 ≈ 27.4小时
# - 实际预估: 20-22小时（考虑缓存加速）
BATCH_SIZE = 64

# 【修复】调整max_new_tokens以支持完整CoT输出
# 训练时模型学习了6步分析 + Final Correction格式
# - CoT分析约需150-200 tokens（经济背景、时序动态、节假日、天气、事件、策略）
# - Final Correction: [v0,v1,...,v7] 约需20-30 tokens
# - 总计约需170-230 tokens
# 
# 原配置: 32/48 (严重不足，导致CoT被截断)
# 优化后: 256/384 (足够输出完整CoT + 数值序列)
# 【紧急加速】调整max_new_tokens以平衡速度和完整性
# 
# 问题分析：
# - 完整CoT需要256 tokens，导致推理时间过长（39小时）
# - 但原始配置32 tokens严重不足，导致CoT截断、解析失败
# 
# 折中方案：
# - 减少到128 tokens，允许模型输出简化版CoT（约3-4步分析）
# - 保留Final Correction格式，确保能正确提取数值
# - 预期效果：从39小时降至18-20小时
# 
# 风险评估：
# - CoT可能被轻微截断，但关键信息（经济背景、天气、事件）应该能输出
# - 如果仍然超时，可进一步降至96 tokens
FIRST_PASS_MAX_NEW_TOKENS = 128
RETRY_MAX_NEW_TOKENS = 192

FEATURE_NAMES = {
    0: "Passenger Car Up",
    1: "Passenger Car Down",
    2: "Non-Passenger Car Up",
    3: "Non-Passenger Car Down"
}

# ====================== Prompt 模板 ======================
# 【关键修复】必须与微调518_optimized.py中的instruction_text完全一致！
# 
# 训练时的Prompt结构：
# System: "Analyze step by step and output the final corrected traffic flow values."
# User: Refine信息（7个字段）
# Assistant: CoT分析（6步）+ "Final Correction: [v0,v1,...,v7]"
#
# 推理时必须保持相同的格式，让模型能够按照训练时的模式输出

BASE_INSTRUCTION_TEXT = (
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

# 【重要】不需要添加引导符，因为模型已经学会了输出完整CoT

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
    """
    【修复】从LLM输出中提取数值序列
    
    策略优先级：
    1. 查找 "Final Correction: [...]" 格式（微调时的标准格式）
    2. 查找任意方括号内的数字列表 [v0,v1,...]
    3. 直接提取开头的数字序列
    4. 宽松模式 - 提取所有数字
    """
    if text is None:
        return None

    # 策略1: 优先查找 "Final Correction: [...]" 格式（最准确）
    fc_matches = list(
        re.finditer(
            r"Final\s*Correction\s*:\s*\[([^\]]+)\]",
            text,
            flags=re.I | re.S
        )
    )
    
    if fc_matches:
        content = fc_matches[-1].group(1)
        nums = re.findall(r"-?\d+", content)
        if len(nums) >= expected_len:
            return [max(0, int(x)) for x in nums[:expected_len]]

    # 策略2: 查找最后一个方括号内的数字列表
    bracket_matches = list(
        re.finditer(
            r"\[([^\]]+)\]",
            text,
            flags=re.S
        )
    )
    
    if bracket_matches:
        # 取最后一个匹配（最可能是最终输出）
        content = bracket_matches[-1].group(1)
        nums = re.findall(r"-?\d+", content)
        if len(nums) >= expected_len:
            return [max(0, int(x)) for x in nums[:expected_len]]

    # 策略3: 如果文本以数字或方括号开头，直接提取
    stripped = text.strip()
    if stripped.startswith(tuple("0123456789-[")):
        nums = re.findall(r"-?\d+", stripped)
        if len(nums) >= expected_len:
            return [max(0, int(x)) for x in nums[:expected_len]]

    # 策略4: 宽松模式 - 提取所有数字（最后手段）
    if allow_naked_numbers:
        nums = re.findall(r"-?\d+", text)
        if len(nums) >= expected_len:
            # 【修复】过滤掉明显的年份（如2023）、日期部分和不合理的大数
            filtered_nums = []
            for x in nums:
                num_val = int(x)
                # 过滤条件：
                # 1. 排除4位数的年份（如2023）
                # 2. 排除明显不合理的超大值（交通流量一般<5000）
                # 3. 保留负数（后续会转为0）
                if abs(num_val) < 1000 or (len(x) != 4 and num_val <= 5000):
                    filtered_nums.append(num_val)
            
            if len(filtered_nums) >= expected_len:
                return [max(0, x) for x in filtered_nums[:expected_len]]

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

# 【关键修复】推理时必须使用left-padding（decoder-only模型要求）
# 注意：这与微调时的right-padding不同，是generate()的特殊要求
tokenizer.padding_side = "left"

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

    # 【关键修复】使用与微调代码完全一致的Prompt格式（带编号列表）
    prompt_std = BASE_INSTRUCTION_TEXT.format(
        desc,                          # 1. Station
        FEATURE_NAMES[feature_idx],   # 2. Feature
        t3s,                           # 3. Time start
        t4s,                           # 3. Time end
        dayt,                          # 3. DayType
        pattern,                       # 4. Flow Pattern
        weather,                       # 5. Weather
        gnn_seq,                       # 6. GNN
        event                          # 7. Event
    )

    # 【关键修复】不再添加 "Final Correction: ["，因为微调代码中output包含完整的CoT
    # 但如果模型不输出Final Correction，我们需要在Prompt末尾添加引导符
    # 注意：这与微调时的full_text格式不同，是推理时的特殊处理
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
        "prompt_force": prompt_force,  # 与prompt_std相同
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
            # 【关键修复】开启采样并设置极低temperature，使微调效果显现
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,          # 必须为True，否则temperature无效
                temperature=0.01,        # 【修复】从0.1降低到0.01，确保数值稳定性
                top_p=0.95,              # nucleus sampling
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=STOP_IDS if len(STOP_IDS) > 1 else STOP_IDS[0],
                repetition_penalty=1.05, # 轻微惩罚重复
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

    # 【修复】使用prompt_force（带引导符）进行推理
    strategies = [
        ("第1轮 强约束推理", "prompt_force", FIRST_PASS_MAX_NEW_TOKENS, True),
        ("第2轮 宽松重试", "prompt_std", RETRY_MAX_NEW_TOKENS, True),
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
            logger.info(f"✅ {round_name} 成功: {success_this_round} | 剩余失败: {len(new_fail_indices)}")
            if new_fail_indices:
                debug_idx = new_fail_indices[0]
                logger.debug("\n" + "=" * 90)
                logger.debug(f"🔍 [DEBUG] {round_name} 失败样本输出：")
                logger.debug("=" * 90)
                logger.debug("原始输出（最后500字符）:")
                logger.debug(raw_outputs[debug_idx][-500:])
                logger.debug("=" * 90)
                logger.debug(f"期望长度: {expected_len}")
                logger.debug("=" * 90 + "\n")

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
logger.info("\n🚀 开始全站点推理...\n")
logger.info(f"📊 配置信息:")
logger.info(f"  - 模型路径: {MODEL_PATH}")
logger.info(f"  - DEBUG_MODE: {DEBUG_MODE}")
logger.info(f"  - DEBUG_PRINT_COUNT: {DEBUG_PRINT_COUNT}")
logger.info(f"  - BATCH_SIZE: {BATCH_SIZE}")
logger.info(f"  - Temperature: 0.01")

completed_keys = load_completed_keys(PROGRESS_PATH)
logger.info(f"🔁 已加载断点进度: {len(completed_keys)} 个 station-feature 已完成")

# 【优化】性能监控变量
import time
start_time = time.time()
total_tasks = len(station_range) * num_features
completed_tasks = len(completed_keys)

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
                logger.info(f"⏭️ 跳过已完成任务: station={station_id}, feature={feature_id}")
                continue

            # 【优化】记录单个任务的开始时间
            task_start_time = time.time()
            logger.info(f"\n{'='*80}")
            logger.info(f"📍 开始处理: Station {station_id} ({station_short_name}) - Feature {feature_id} ({feature_name})")
            logger.info(f"   样本数: {usable_samples}")
            logger.info(f"{'='*80}")

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

            # 【新增】诊断计数器
            debug_printed_count = 0

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

                # 【新增】诊断输出
                if DEBUG_MODE and debug_printed_count < DEBUG_PRINT_COUNT:
                    logger.info(f"\n{'='*90}")
                    logger.info(f"🔍 [诊断样本 {debug_printed_count + 1}]")
                    logger.info(f"站点: {station_short_name} | 特征: {feature_name}")
                    logger.info(f"时间: {prompt_items[i]['timestamp']}")
                    logger.info(f"真实值: {true_seq.tolist()}")
                    logger.info(f"GNN预测: {gnn_seq.tolist()}")
                    logger.info(f"LLM预测: {llm_seq.tolist() if pred_results[i] is not None else '回退到GNN'}")
                    logger.info(f"是否回退: {'是' if is_fallback else '否'}")
                    
                    if raw_outputs[i]:
                        logger.info(f"\nLLM原始输出（前300字符）:")
                        logger.info(raw_outputs[i][:300])
                        logger.info(f"\nLLM原始输出（后300字符）:")
                        logger.info(raw_outputs[i][-300:])
                    
                    logger.info(f"{'='*90}\n")
                    debug_printed_count += 1

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

            # 【优化】计算任务耗时和预计剩余时间
            task_elapsed = time.time() - task_start_time
            completed_tasks += 1
            remaining_tasks = total_tasks - completed_tasks
            avg_time_per_task = (time.time() - start_time) / completed_tasks
            estimated_remaining_time = avg_time_per_task * remaining_tasks
            
            # 格式化时间显示
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                if hours > 0:
                    return f"{hours}h{minutes}m{secs}s"
                elif minutes > 0:
                    return f"{minutes}m{secs}s"
                else:
                    return f"{secs}s"
            
            print(f"✅ 已完成并保存断点: station={station_id}, feature={feature_id} | "
                  f"耗时: {format_time(task_elapsed)} | "
                  f"进度: {completed_tasks}/{total_tasks} | "
                  f"预计剩余: {format_time(estimated_remaining_time)}")
            logger.info(f"   LLM解析成功率: {(usable_samples - len(final_fail_indices)) / usable_samples * 100:.2f}%")

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
