"""
python 微调430solo.py
"""
""" 
python fin_fast_vals_quick.py
 - 显存兼容最终版 - 阈值触发修正 - 支持断点恢复
 - 【已修改】直接使用真实值训练，修正值=真实值
 - 【已修改】全新5步COT推理：GDP+时段+节假日+天气+事件
训练就是用的这个代码
"""
import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
os.environ["UNSLOTH_SKIP_INIT_CHECK"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
# 显存优化设置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import random
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from chinese_calendar import is_workday
import sys
import gc
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from datetime import datetime
import glob

# 设置项目根路径
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"

# 重定向输出
class Logger(object):
    def __init__(self, filename=os.path.join(PROJECT_ROOT, f"quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")):
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
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
print("✓ 正在启动【显存兼容版】阈值触发修正训练...")
print(f"✓ 日志文件: {sys.stdout.log.name}")

# --- 路径配置 ---
BASE_DATA_PATH = PROJECT_ROOT
FINETUNE_OUTPUT_PATH = PROJECT_ROOT

YTRUE_TRAIN_PATH = os.path.join(FINETUNE_OUTPUT_PATH, "finetune_real_traffic.npz")
YPRED_TRAIN_PATH = os.path.join(FINETUNE_OUTPUT_PATH, "finetune_data.npz")
CARPARK_DES_PATH = os.path.join(BASE_DATA_PATH, "station_list_hngs.txt")
NATURAL_PATTERN_PATH = os.path.join(BASE_DATA_PATH, "station_natural_list_4feat.txt")
WEATHER_DATA_PATH = os.path.join(BASE_DATA_PATH, "160站点天气信息/")
EVENTS_CSV_PATH = os.path.join(BASE_DATA_PATH, "events_list_quan.csv")
HIS_DATA_CSV_PATH = os.path.join(BASE_DATA_PATH, "his_data_with_index.csv")

# --- 加载数据 ---
print("正在加载数据并精简天气信息...")
his_df = pd.read_csv(HIS_DATA_CSV_PATH)
his_df['时间'] = pd.to_datetime(his_df['时间'])
parking_data = his_df.set_index('时间').sort_index()

npz_true = np.load(YTRUE_TRAIN_PATH)
npz_pred = np.load(YPRED_TRAIN_PATH)

print(f"✓ NPZ文件键名检测:")
print(f" ytrue_train.npz 键名: {list(npz_true.keys())}")
print(f" ypred_train.npz 键名: {list(npz_pred.keys())}")

if 'target' in npz_true:
    true_array = npz_true['target']
elif 'arr_0' in npz_true:
    true_array = npz_true['arr_0']
else:
    raise KeyError(f"ytrue_train.npz 未找到有效键")

if 'prediction' in npz_pred:
    pred_array = npz_pred['prediction']
elif 'arr_0' in npz_pred:
    pred_array = npz_pred['arr_0']
else:
    raise KeyError(f"ypred_train.npz 未找到有效键")

print(f"✓ 流量数据加载成功:")
print(f" true_array shape: {true_array.shape}")
print(f" pred_array shape: {pred_array.shape}")

if np.allclose(true_array, pred_array):
    raise ValueError("❌ 预测值与真实值完全相同！请检查文件路径")
else:
    diff = np.abs(true_array - pred_array)
    print(f"\n✓ 数据验证通过:")
    print(f" 平均绝对误差 (MAE): {diff.mean():.2f}")

print(f"\n直接使用his_data_with_index.csv的时间索引")
print(f" 起始时间: {parking_data.index[0]}")
print(f" 结束时间: {parking_data.index[-1]}")
print(f" 总行数: {len(parking_data)}")

# 加载站点列表
print(f"\n正在加载站点列表: {CARPARK_DES_PATH}")
with open(CARPARK_DES_PATH, "r", encoding='utf-8') as f:
    carpark_des_list = [l.strip() for l in f if l.strip()]
print(f"✓ 站点列表加载成功: {len(carpark_des_list)} 个站点")

# 加载自然语言模式
print(f"正在加载自然语言模式: {NATURAL_PATTERN_PATH}")
with open(NATURAL_PATTERN_PATH, "r", encoding='utf-8') as f:
    natural_pattern_list = [l.strip() for l in f if l.strip()]
print(f"✓ 自然语言模式加载成功: {len(natural_pattern_list)} 条描述")

# 维度对齐
print(f"\n原始数据形状分析:")
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
print(f"\n✓ 维度对齐完成:")
print(f" 样本数: {num_samples} | 站点数: {num_stations} | 特征数: {num_features} | 序列长度: {seq_len}")

FEATURE_NAMES = {
    0: "Passenger Car Up",
    1: "Passenger Car Down", 
    2: "Non-Passenger Car Up",
    3: "Non-Passenger Car Down"
}

# ==============================
# 【核心配置】误差阈值 - 只有超过阈值才触发修正
# ==============================
ERROR_THRESHOLD_BY_FEATURE = {
    0: 100,  # 小客车上行: MAE > 100 才触发修正
    1: 90,   # 小客车下行: MAE > 90 才触发修正
    2: 27,   # 非小客车上行: MAE > 27 才触发修正
    3: 27    # 非小客车下行: MAE > 27 才触发修正
}

# 打印阈值配置
print("\n" + "="*70)
print("📊 【阈值触发修正配置】只有GNN误差超过阈值才执行修正:")
print("="*70)
for feat_idx, feat_name in FEATURE_NAMES.items():
    threshold = ERROR_THRESHOLD_BY_FEATURE.get(feat_idx, 60)
    print(f"   {feat_name:25s} : 阈值 = {threshold:3d}  (MAE > {threshold} 时触发修正)")
print("="*70 + "\n")

# ==============================
# 预加载天气
# ==============================
# ==============================
# 预加载天气 【已修复：支持csv+xlsx，不崩溃】
# ==============================
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
            
            # ====================== 修复点：支持 CSV / XLSX ======================
            if path.endswith('.csv'):
                df = pd.read_csv(path, encoding='utf-8')
            elif path.endswith('.xlsx') or path.endswith('.xls'):
                df = pd.read_excel(path)
            else:
                continue
            # ====================================================================
            
            # 自动查找日期列、天气列（兼容各种列名）
            d_col = None
            w_col = None
            for c in df.columns:
                c_lower = str(c).lower()
                if '日' in c or 'date' in c_lower:
                    d_col = c
                if '天气' in c or 'weather' in c_lower:
                    w_col = c
            
            if not d_col or not w_col:
                continue
            
            df[d_col] = pd.to_datetime(df[d_col]).dt.date.astype(str)
            df = df.drop_duplicates(subset=[d_col]).set_index(d_col)
            weather_dict[j] = df[w_col].to_dict()
        except:
            continue
    print(f"✅ 天气预加载完成：{len(weather_dict)} 个站点在内存")
    return weather_dict

weather_cache = preload_all_weather(WEATHER_DATA_PATH, num_stations)

def get_weather_fast(station_idx, date_str):
    return weather_cache.get(station_idx, {}).get(date_str, "Unknown")

# 加载事件
def load_chinese_events(path):
    if not os.path.exists(path):
        print(f"⚠️ 未找到事件文件: {path}")
        return {}
    try:
        df_ev = pd.read_csv(path, encoding='utf-8')
        date_col = "日期"
        station_col = "站点名称"
        event_col = "事件描述"
        for c in [date_col, station_col, event_col]:
            if c not in df_ev.columns:
                raise ValueError(f"缺少列 {c}")
        df_ev[date_col] = pd.to_datetime(df_ev[date_col]).dt.strftime('%Y-%m-%d')
        df_ev[station_col] = df_ev[station_col].astype(str).str.strip()
        df_ev[event_col] = df_ev[event_col].astype(str).str.strip()
        event_map = {}
        for _, r in df_ev.iterrows():
            k = (r[date_col], r[station_col])
            event_map[k] = r[event_col]
        print(f"✅ 事件加载完成：{len(event_map)} 条")
        return event_map
    except Exception as e:
        print(f"❌ 事件加载失败: {e}")
        return {}

precise_events_map = load_chinese_events(EVENTS_CSV_PATH)

# ==============================
# 预计算数字token映射
# ==============================
def precompute_digit_mapping(tokenizer, device):
    """预计算数字token的映射和ID列表"""
    id_to_val = {}
    digit_ids = []
    for i in range(tokenizer.vocab_size):
        t = tokenizer.decode([i]).strip()
        if t and t.isdigit():
            try:
                id_to_val[i] = float(t)
                digit_ids.append(i)
            except:
                pass
    
    digit_ids_tensor = torch.tensor(digit_ids, device=device)
    val_vec = torch.tensor([id_to_val[int(t)] for t in digit_ids], device=device)
    target_bracket_id = 510
    
    print(f"✅ 数字Token预加载: {len(digit_ids)} 个")
    return id_to_val, digit_ids, digit_ids_tensor, val_vec, target_bracket_id

# ==============================
# 新增：GDP水平推断
# ==============================
def infer_gdp_level(desc):
    """简单推断GDP水平：根据站点描述中的关键词"""
    # high_gdp_keywords = ["市", "主城", "枢纽", "高新区", "中心", "经济区"]
    high_gdp_keywords = ["city", "urban", "hub", "central", "center", 
                         "downtown", "high-tech", "economic zone", "business"]
    for kw in high_gdp_keywords:
        if kw in desc:
            return "High"
    return "Low"

# ==============================
# 新增：事件影响方向推断
# ==============================
def infer_event_impact(ev_text):
    """推断事件对车流量的影响方向"""
    if ev_text == "None":
        return "none"
    # decrease_keywords = ["事故", "施工", "拥堵", "管制", "封闭", "雨雪", "塌方"]
    # increase_keywords = ["演唱会", "比赛", "活动", "展会", "免费"]
     # 导致流量下降的英文关键词
    decrease_keywords = ["accident", "construction", "congestion", "control", 
                         "closed", "rain", "snow", "collapse", "road closed", 
                         "blocked", "delay"]
    # 导致流量上升的英文关键词
    increase_keywords = ["concert", "match", "game", "activity", "event", 
                         "fair", "exhibition", "show", "festival"]
    
    for kw in decrease_keywords:
        if kw in ev_text:
            return "decrease"
    for kw in increase_keywords:
        if kw in ev_text:
            return "increase"
    return "fluctuate" # 无法明确判断时使用波动

# ==============================
# 核心账单输出函数
# ==============================
comparison_log = []  # 全局对比日志

def log_comparison(station_name, feature_name, timestamp, gnn_value, true_value, llm_value, error_before, error_after):
    """记录GNN、真实值、LLM输出的对比"""
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
    
    # 打印核心账单
    print(f"\n{'='*30}")
    print(f"📊 【阈值触发修正对账单】")
    print(f"{'='*30}")
    print(f"站点: {station_name}")
    print(f"特征: {feature_name}")
    print(f"时间: {timestamp}")
    print(f"{'-'*30}")
    print(f"GNN预测值:     {gnn_value}")
    print(f"真实值:        {true_value}")
    print(f"修正后值:      {llm_value}")
    print(f"{'-'*30}")
    print(f"修正前MAE:     {error_before:.2f}")
    print(f"修正后MAE:     {error_after:.2f}")
    print(f"改善幅度:      {error_before - error_after:.2f} ({100*(error_before-error_after)/error_before:.1f}%)")
    print(f"{'='*30}\n")

# ==============================
# 优化版训练器
# ==============================
class OptimizedMAEHybridTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tok = getattr(self, "processing_class", self.tokenizer)
        
        device = self.args.device
        self.id_to_val, self.digit_ids, self.digit_ids_tensor, self.val_vec, self.target_bracket_id = \
            precompute_digit_mapping(tok, device)
        
        self.digit_set = set(self.digit_ids)
        self.print_interval = 10
        self.step_counter = 0
        
        print(f"✅ 优化版Trainer初始化完成，打印间隔: {self.print_interval} 步")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs["output_hidden_states"] = True
        outputs = model(**inputs)
        last_hidden = outputs.hidden_states[-1]
        logits = model.lm_head(last_hidden.to(next(model.lm_head.parameters()).dtype))
        ce_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        
        labels = inputs.get("labels")
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        
        all_digit_mask = torch.isin(shift_labels, self.digit_ids_tensor)
        
        bracket_mask = (shift_labels == self.target_bracket_id)
        bracket_positions = []
        for b in range(shift_labels.shape[0]):
            pos = torch.where(bracket_mask[b])[0]
            if len(pos) > 0:
                bracket_positions.append(pos[-1].item())
            else:
                bracket_positions.append(-1)
        
        digit_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for b, pos in enumerate(bracket_positions):
            if pos >= 0:
                digit_mask[b, pos+1:] = all_digit_mask[b, pos+1:]
        
        digit_mask &= (shift_labels != -100)
        
        mae_loss = torch.tensor(0.0, device=self.args.device)
        total_loss = ce_loss
        
        if digit_mask.any():
            dig_logits = shift_logits[digit_mask][:, self.digit_ids_tensor]
            probs = torch.softmax(dig_logits.to(torch.float32), dim=-1)
            expected_values = (probs * self.val_vec).sum(dim=-1)
            
            target_labels = shift_labels[digit_mask]
            target_vals = torch.zeros_like(expected_values)
            for i, tid in enumerate(target_labels):
                tid_int = int(tid.item())
                if tid_int in self.id_to_val:
                    target_vals[i] = self.id_to_val[tid_int]
            
            mask = (target_vals != 0).float()
            if mask.mean() > 0:
                mask = mask / mask.mean()
                mae_loss = torch.mean(torch.abs(expected_values - target_vals) * mask)
                total_loss = ce_loss + 0.11 * mae_loss
        
        self.step_counter += 1
        if self.step_counter % self.print_interval == 0 and self.state.global_step > 0:
            try:
                tok = getattr(self, "processing_class", self.tokenizer)
                s_idx = 0
                s_lab = shift_labels[s_idx].cpu().numpy()
                b_pos = np.where(s_lab == self.target_bracket_id)[0]
                if len(b_pos) > 0:
                    start = max(0, b_pos[-1] - 200)
                    safe = [x for x in s_lab[start:] if x != -100]
                    txt = tok.decode(safe, skip_special_tokens=False).split('<|eot_id|>')[0] + " <|eot_id|>"
                    
                    print(f"\n{'='*30}")
                    print(f"🎯 训练进度 - Step {self.state.global_step}")
                    print(f"{'='*30}")
                    print(f"Loss: {total_loss.item():.4f} | CE:{ce_loss.item():.4f} | MAE:{mae_loss.item():.4f}")
                    print(f"进度: {self.state.global_step}/{self.state.max_steps} ({100*self.state.global_step/self.state.max_steps:.1f}%)")
                    print(f"{'='*30}\n")
            except Exception as e:
                pass
        
        return (total_loss, outputs) if return_outputs else total_loss

# 加载模型
print(f"\n✓ 正在加载模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/user/Llama-3.1-8B",
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    device_map={"": 0},
)

# 指令模板（已删除旧规则）
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

# ==============================
# 批量生成样本（【核心】直接使用真实值训练）
# ==============================
NORMAL_SAMPLE_RATIO = 0.1  # 正常样本采样比例

def process_chunk(stations_chunk):
    """处理站点块 - 【核心】修正值 = 真实值"""
    samples = []
    for j in stations_chunk:
        if j >= len(carpark_des_list):
            continue
        desc = carpark_des_list[j]
        s_name = desc.split()[0]
        
        for f in range(num_features):
            f_name = FEATURE_NAMES[f]
            is_p = f in (0, 1)
            is_t = f in (2, 3)
            is_d = f == 1
            pi = j * num_features + f
            pat = natural_pattern_list[pi] if pi < len(natural_pattern_list) else "General"
            
            pred_j = pred_array[:, j, f, :]
            true_j = true_array[:, j, f, :]
            mae_per_sample = np.abs(pred_j - true_j).mean(axis=1)
            
            # 特征阈值
            feature_threshold = ERROR_THRESHOLD_BY_FEATURE.get(f, 60)
            
            rows = np.arange(num_samples) * num_stations + j
            rows = rows[rows < len(parking_data)]
            ts = parking_data.index[rows]
            if len(ts) == 0:
                continue
            
            dates = [t.strftime('%Y-%m-%d') for t in ts]
            is_work = np.array([is_workday(t) for t in ts])
            is_holi = ~is_work
            hours = np.array([t.hour for t in ts])
            events = [precise_events_map.get((d, s_name), "None") for d in dates]
            has_ev = np.array([e != "None" for e in events])
            
            # 触发条件
            need_correction = mae_per_sample > feature_threshold
            ev_mask = has_ev
            norm_mask = np.random.rand(len(mae_per_sample)) < NORMAL_SAMPLE_RATIO
            keep = np.where(need_correction | ev_mask | norm_mask)[0]
            
            # 打印采样统计
            if j % 50 == 0 and f == 0 and len(keep) > 0:
                print(f"   站点{j:3d} [{s_name[:15]:15s}] {f_name:20s}: "
                      f"阈值={feature_threshold:3d}, "
                      f"触发修正={need_correction.sum():4d}, "
                      f"事件触发={ev_mask.sum():3d}, "
                      f"随机采样={norm_mask.sum():4d}, "
                      f"总采样={len(keep):4d}")
            
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
                    
                    # ========== 【核心修改】直接使用真实值作为修正结果 ==========
                    final_values = true_value_seq.round().astype(int).tolist()
                    is_above_threshold = mae_per_sample[i] > feature_threshold
                    
                    # 记录日志
                    if is_above_threshold:
                        mae_before = np.abs(gnn_original - true_value_seq).mean()
                        mae_after = np.abs(np.array(final_values) - true_value_seq).mean()
                        t3_str = t3.strftime('%Y-%m-%d %H:%M:%S')
                        log_comparison(s_name, f_name, t3_str, 
                                     gnn_array, true_value_seq.tolist(), final_values,
                                     mae_before, mae_after)
                    # ========================================================
                    
                    t4 = t3 + pd.Timedelta(hours=7)
                    t3s = t3.strftime('%Y-%m-%d %H:%M')
                    t4s = t4.strftime('%Y-%m-%d %H:%M')
                    dy = t3.strftime('%A')
                    dayt = "Workday" if iw else "Off-day"
                    time_desc = f"{t3s} to {t4s} | {dy} | {dayt}"
                    
                    # ========== 【全新5步COT推理】 ==========
                    cot = []
                    gdp_level = infer_gdp_level(desc)
                    
                    # 时段划分
                    if h < 6 or h >= 23:
                        time_period = "Nighttime (Low traffic base)"
                    elif 7 <= h <= 9 or 17 <= h <= 19:
                        time_period = "Peak hours (High traffic base)"
                    else:
                        time_period = "Off-peak daytime (Moderate traffic base)"
                    
                    if is_above_threshold:
                        # Step 1: GDP分析
                        cot.append(f"1. Economic Profile: Station {s_name} is located in a {gdp_level}-GDP region. This determines the base sensitivity to holiday and economic activities.")
                        # Step 2: 时段分析
                        cot.append(f"2. Temporal Dynamics: The target window ({t3s} to {t4s}) falls under {time_period}.")
                        # Step 3: 节假日+车型分析
                        if ih:
                            if is_p:
                                surge_magnitude = "significantly high" if gdp_level == "High" else "moderate"
                                cot.append(f"3. Holiday Effect: Holiday drives passenger car ({f_name}) volume up. Given the {gdp_level}-GDP region, this upward adjustment magnitude should be {surge_magnitude}.")
                            else:
                                cot.append(f"3. Holiday Effect: As a freight/commercial feature ({f_name}), holiday traffic restrictions dictate a drastic reduction in volume.")
                        else:
                            cot.append("3. Holiday Effect: Routine workday pattern. No holiday-induced vehicle type adjustments needed.")
                        # Step 4: 天气分析
                        if w not in ["晴", "多云", "阴", "Unknown"]:
                            cot.append(f"4. Weather Impact: Adverse weather ({w}) restricts road capacity, causing a drop in traffic volume, which is particularly impactful during {time_period}.")
                        else:
                            cot.append(f"4. Weather Impact: Routine weather ({w}) causes no disruption to baseline capacity.")
                        # Step 5: 事件分析
                        if ev != "None":
                            impact_dir = infer_event_impact(ev)
                            if impact_dir == "decrease":
                                cot.append(f"5. Event Disruption: A negative special event ('{ev}') severely throttles typical traffic flow downwards.")
                            elif impact_dir == "increase":
                                cot.append(f"5. Event Disruption: A positive special event ('{ev}') draws localized traffic, pushing volume upwards.")
                            else:
                                cot.append(f"5. Event Disruption: Special event ('{ev}') causes significant anomalous fluctuations.")
                        else:
                            cot.append("5. Event Disruption: No special events reported.")
                        # 输出真实值
                        cot.append("6. Strategy: The GNN baseline prediction fails to adequately account for these interacting dynamic factors. Overriding with actual calibrated data.")
                        cot.append(f"Final Correction: [{', '.join(map(str, final_values))}]")
                    else:
                        # 低误差简化推理
                        cot.append(f"1. Economic Profile: {gdp_level}-GDP region baseline.")
                        cot.append(f"2. Temporal Dynamics: {time_period}.")
                        cot.append("3. Conditions Evaluation: Normal weather, no holidays, and no disruptive events detected. Traffic is operating under highly predictable standard patterns.")
                        cot.append("4. Strategy: Standard GNN captures this routine variance perfectly. No manual adjustment required.")
                        cot.append(f"Final Correction: [{', '.join(map(str, final_values))}]")
                    # ==========================================
                    
                    cot_text = " ".join(cot) + " <|eot_id|>"
                    
                    prompt = instruction_text.format(desc, f_name, t3s, t4s, dayt, pat, w, gnn_array, ev if ev != "None" else "None")
                    full = prompt + cot_text
                    
                    samples.append({
                        "prompt": prompt,
                        "cot_text": cot_text,
                        "full_text": full
                    })
                except Exception as e:
                    continue
    return samples

def build_dataset_fast():
    """多进程生成数据集"""
    n_proc = min(8, cpu_count())
    # stations = list(range(num_stations))
    stations = [10]
    chunk_size = max(1, len(stations) // n_proc)
    chunks = [stations[i:i+chunk_size] for i in range(0, len(stations), chunk_size)]
    
    print(f"\n🚀 多进程生成：{n_proc} 进程，{len(chunks)} 块")
    with Pool(n_proc) as p:
        res = list(tqdm(p.imap(process_chunk, chunks), total=len(chunks), desc="生成样本"))
    
    all_samples = []
    for r in res:
        all_samples.extend(r)
    
    print(f"\n📝 批量tokenize {len(all_samples)} 个样本...")
    tokenized_samples = []
    for sample in tqdm(all_samples, desc="Tokenizing"):
        full = sample["full_text"]
        prompt = sample["prompt"]
        
        enc = tokenizer(full, truncation=True, max_length=1024, add_special_tokens=False)
        p_enc = tokenizer(prompt, add_special_tokens=False)
        pl = len(p_enc["input_ids"])
        labs = [-100] * pl + enc["input_ids"][pl:]
        
        tokenized_samples.append({
            "input_ids": enc["input_ids"],
            "labels": labs,
            "attention_mask": [1] * len(enc["input_ids"]),
        })
    
    return tokenized_samples

# ==============================
# 开始生成
# ==============================
print("\n🚀 开始生成训练样本...")

# 检查是否已有生成的数据集，避免重复生成
dataset_path = os.path.join(PROJECT_ROOT, "quick30.json")
if os.path.exists(dataset_path):
    print(f"✅ 发现已存在数据集: {dataset_path}")
    print(f"   加载已有数据集...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        generated_dataset = json.load(f)
    print(f"✅ 加载完成：{len(generated_dataset)} 条样本")
else:
    print("   生成新数据集...")
    generated_dataset = build_dataset_fast()
    random.shuffle(generated_dataset)
    
    MAX_SAMPLES = 30000
    if len(generated_dataset) > MAX_SAMPLES:
        generated_dataset = generated_dataset[:MAX_SAMPLES]
        print(f"⚠️ 样本数已限制到 {MAX_SAMPLES} 条")
    
    print(f"\n✅ 样本生成完成：总计 {len(generated_dataset)} 条")
    
    # 保存数据集
    with open(dataset_path, "w", encoding="utf-8") as f:
        simplified_dataset = [{"input_ids": d["input_ids"], "labels": d["labels"], "attention_mask": d["attention_mask"]} 
                              for d in generated_dataset]
        json.dump(simplified_dataset, f, ensure_ascii=False)
    print(f"✅ 数据集已保存至: {dataset_path}")

# 保存对比日志
if comparison_log:
    comparison_df = pd.DataFrame(comparison_log)
    comparison_df.to_csv(os.path.join(PROJECT_ROOT, "comparison_log.csv"), index=False, encoding='utf-8-sig')
    print(f"\n✅ 【阈值触发修正统计】:")
    print(f"   触发修正样本数: {len(comparison_log)}")
    print(f"   修正前GNN平均MAE: {comparison_df['gnn_error'].mean():.2f}")
    print(f"   修正后平均MAE: {comparison_df['llm_error'].mean():.2f}")
    print(f"   平均改善: {comparison_df['improvement'].mean():.2f}")
    print(f"   平均改善率: {100 * (comparison_df['improvement'].mean() / comparison_df['gnn_error'].mean()):.1f}%")

# ==============================
# 【显存兼容配置】
# ==============================
if generated_dataset:
    ds = Dataset.from_list(generated_dataset)
    model.config.output_hidden_states = True
    
    # LoRA配置
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none"
    )
    
    # 显存友好的batch配置
    BATCH_SIZE = 8
    GRAD_ACCUM = 4
    total_samples = len(generated_dataset)
    steps_per_epoch = total_samples // (BATCH_SIZE * GRAD_ACCUM)
    total_steps = steps_per_epoch * 10
    
    print(f"\n📊 训练统计:")
    print(f"   总样本数: {total_samples}")
    print(f"   批次大小: {BATCH_SIZE}")
    print(f"   梯度累积: {GRAD_ACCUM}")
    print(f"   有效批次: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"   训练轮数: 10")
    print(f"   每轮步数: ~{steps_per_epoch}")
    print(f"   总步数: ~{total_steps}")
    
    # 显存优化配置
    args = TrainingArguments(
        output_dir=os.path.join(PROJECT_ROOT, "results-quick430"),
        num_train_epochs=10,
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        save_only_model=True,
        use_cpu=False,
        ddp_find_unused_parameters=False,
    )
    
    # 初始化trainer
    trainer = OptimizedMAEHybridTrainer(
        model=model,
        train_dataset=ds,
        tokenizer=tokenizer,
        args=args,
        max_seq_length=1024,
        packing=False,
    )
    
    # 断点恢复
    checkpoint_dir = os.path.join(PROJECT_ROOT, "results-quick430")
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    resume_checkpoint = None
    if checkpoints:
        checkpoint_nums = [int(c.split('-')[-1]) for c in checkpoints]
        max_num = max(checkpoint_nums)
        resume_checkpoint = os.path.join(checkpoint_dir, f"checkpoint-{max_num}")
        print(f"\n✅ 发现已有checkpoint: {resume_checkpoint}")
        print(f"   将从这里恢复训练（步数 ~{max_num}）")
    else:
        print(f"\n⚠️ 未发现checkpoint，从头开始训练")
    
    # 清空缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 显存监控
    print(f"\n📊 显存状态:")
    print(f"   总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   已用显存: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"   空闲显存: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")
    
    print("\n🚀 开始训练...")
    print(f"⚡ 【真实值训练模式】:")
    print(f"   - 修正值 = 真实值")
    print(f"   - 5步COT推理：GDP+时段+节假日+天气+事件")
    print(f"   - 批次大小: {BATCH_SIZE}")
    print(f"   - 梯度累积: {GRAD_ACCUM}")
    # if resume_checkpoint:
    #     print(f"   - 恢复点: {resume_checkpoint}")
    # print("="*60)
    
    # # 开始训练
    # trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.train()
    
    # 保存最终日志
    if comparison_log:
        final_comparison_df = pd.DataFrame(comparison_log)
        final_comparison_df.to_csv(os.path.join(PROJECT_ROOT, "final_comparison_log.csv"), index=False, encoding='utf-8-sig')
        print(f"\n✅ 【最终修正统计】:")
        print(f"   总触发修正样本数: {len(final_comparison_df)}")
        print(f"   GNN平均MAE: {final_comparison_df['gnn_error'].mean():.2f}")
        print(f"   修正后平均MAE: {final_comparison_df['llm_error'].mean():.2f}")
        print(f"   平均改善: {final_comparison_df['improvement'].mean():.2f}")
    
    out_path = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick430")
    model.save_pretrained(out_path)
    print(f"\n✅ 训练完成！模型已保存至: {out_path}")
    
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("\n❌ 未生成训练样本")