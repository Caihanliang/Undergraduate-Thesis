""" 
python 微调428.py
 - 优化版：更强的修正效果 - 带详细对账输出 - 支持157站点
 - 优化：限制样本数、减少轮数、增加批次
"""
import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
os.environ["UNSLOTH_SKIP_INIT_CHECK"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import random
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import re
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

PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"

class Logger(object):
    def __init__(self, filename=os.path.join(PROJECT_ROOT, f"optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")):
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
print("✓ 正在启动【优化版】阈值触发修正训练...")
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

# ==============================
# 优化配置参数
# ==============================
MAX_SAMPLES = 50000      # 最大训练样本数（从35万降到5万）
NUM_EPOCHS = 5           # 训练轮数（从15降到5）
BATCH_SIZE = 8           # 批次大小（从4增加到8）
GRAD_ACCUM = 4           # 梯度累积（从8降到4，有效batch=32）
LEARNING_RATE = 1e-4     # 学习率
MAE_WEIGHT = 0.5         # MAE损失权重
LOG_STEPS = 10           # 日志输出间隔
# ==============================

# --- 加载数据 ---
print("正在加载数据...")
his_df = pd.read_csv(HIS_DATA_CSV_PATH)
his_df['时间'] = pd.to_datetime(his_df['时间'])
parking_data = his_df.set_index('时间').sort_index()

npz_true = np.load(YTRUE_TRAIN_PATH)
npz_pred = np.load(YPRED_TRAIN_PATH)

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

diff = np.abs(true_array - pred_array)
print(f"\n✓ 数据验证通过: 平均绝对误差 (MAE): {diff.mean():.2f}")

# 加载站点列表
with open(CARPARK_DES_PATH, "r", encoding='utf-8') as f:
    carpark_des_list = [l.strip() for l in f if l.strip()]
print(f"✓ 站点列表加载成功: {len(carpark_des_list)} 个站点")

with open(NATURAL_PATTERN_PATH, "r", encoding='utf-8') as f:
    natural_pattern_list = [l.strip() for l in f if l.strip()]
print(f"✓ 自然语言模式加载成功: {len(natural_pattern_list)} 条")

# 维度对齐
if true_array.ndim == 4:
    if true_array.shape[1] == 8 and true_array.shape[3] == 4:
        true_array = true_array.transpose(0, 2, 3, 1)
        pred_array = pred_array.transpose(0, 2, 3, 1)

num_samples, num_stations, num_features, seq_len = true_array.shape
print(f"\n✓ 维度对齐完成: 样本数={num_samples}, 站点数={num_stations}, 特征数={num_features}")

FEATURE_NAMES = {
    0: "Passenger Car Up",
    1: "Passenger Car Down", 
    2: "Non-Passenger Car Up",
    3: "Non-Passenger Car Down"
}

# ==============================
# 【改进配置】降低阈值，增加修正样本
# ==============================
ERROR_THRESHOLD_BY_FEATURE = {
    0: 50,   # 小客车上行
    1: 45,   # 小客车下行
    2: 15,   # 非小客车上行
    3: 15    # 非小客车下行
}

print("\n" + "="*70)
print("📊 【优化配置】:")
print("="*70)
for feat_idx, feat_name in FEATURE_NAMES.items():
    threshold = ERROR_THRESHOLD_BY_FEATURE.get(feat_idx, 60)
    print(f"   {feat_name:25s} : 阈值 = {threshold:3d}")
print(f"\n📊 训练配置:")
print(f"   - 最大样本数: {MAX_SAMPLES}")
print(f"   - 训练轮数: {NUM_EPOCHS}")
print(f"   - 批次大小: {BATCH_SIZE}")
print(f"   - 梯度累积: {GRAD_ACCUM}")
print(f"   - 有效批次: {BATCH_SIZE * GRAD_ACCUM}")
print("="*70 + "\n")

# ==============================
# 修复版天气加载
# ==============================
def preload_all_weather(weather_dir, total_stations):
    weather_dict = {}
    
    if not os.path.exists(weather_dir):
        print(f"⚠️ 天气目录不存在: {weather_dir}")
        for j in range(total_stations):
            weather_dict[j] = {}
        return weather_dict
    
    files = os.listdir(weather_dir)
    print(f"📁 天气目录文件数: {len(files)}")
    
    csv_files = [f for f in files if f.endswith('.csv') or f.endswith('.xlsx')]
    print(f"📋 天气文件示例: {csv_files[:5]}")
    
    loaded_count = 0
    
    for j in tqdm(range(total_stations), desc="预加载天气"):
        matched_file = None
        
        possible_names = [
            f"{j:03d}.csv", f"{j:03d}.xlsx", f"{j}.csv", f"{j}.xlsx",
            f"station_{j}.csv", f"station{j}.csv",
        ]
        
        for name in possible_names:
            if name in files:
                matched_file = name
                break
        
        if not matched_file:
            for f in csv_files:
                if f.startswith(f"{j:03d}") or f.startswith(f"{j}_") or f.startswith(f"{j}."):
                    matched_file = f
                    break
        
        if not matched_file:
            weather_dict[j] = {}
            continue
        
        try:
            path = os.path.join(weather_dir, matched_file)
            if path.endswith('.csv'):
                df = pd.read_csv(path, encoding='utf-8')
            elif path.endswith('.xlsx'):
                df = pd.read_excel(path)
            else:
                weather_dict[j] = {}
                continue
            
            d_col = None
            for c in df.columns:
                if '日期' in str(c) or 'date' in str(c).lower() or '时间' in str(c):
                    d_col = c
                    break
            
            w_col = None
            for c in df.columns:
                if '天气' in str(c) or 'weather' in str(c).lower() or '天气状况' in str(c):
                    w_col = c
                    break
            
            if d_col is None or w_col is None:
                weather_dict[j] = {}
                continue
            
            df[d_col] = pd.to_datetime(df[d_col]).dt.date.astype(str)
            df = df.drop_duplicates(subset=[d_col]).set_index(d_col)
            weather_dict[j] = df[w_col].to_dict()
            loaded_count += 1
            
        except Exception as e:
            weather_dict[j] = {}
    
    print(f"✅ 天气预加载完成：{loaded_count}/{total_stations} 个站点")
    return weather_dict

weather_cache = preload_all_weather(WEATHER_DATA_PATH, num_stations)

def get_weather_fast(station_idx, date_str):
    return weather_cache.get(station_idx, {}).get(date_str, "Clear")

# ==============================
# 加载事件数据
# ==============================
def load_chinese_events(path):
    if not os.path.exists(path):
        print(f"⚠️ 事件文件不存在: {path}")
        return {}
    try:
        df_ev = pd.read_csv(path, encoding='utf-8')
        date_col, station_col, event_col = "日期", "站点名称", "事件描述"
        
        if date_col not in df_ev.columns or station_col not in df_ev.columns:
            return {}
        
        event_map = {}
        for _, r in df_ev.iterrows():
            try:
                dt = pd.to_datetime(r[date_col]).strftime('%Y-%m-%d')
                st = str(r[station_col]).strip()
                ev = str(r[event_col]).strip()
                if st and ev != 'nan' and ev != 'None':
                    event_map[(dt, st)] = ev
            except:
                continue
        print(f"✅ 事件加载完成：{len(event_map)} 条")
        return event_map
    except Exception as e:
        print(f"⚠️ 事件加载失败: {e}")
        return {}

precise_events_map = load_chinese_events(EVENTS_CSV_PATH)

# ==============================
# 【改进】更强的修正系数
# ==============================
def adjust_vectorized(gnp, is_p, is_t, is_holi, is_wk, h, w):
    a = np.array(gnp, dtype=np.float32)
    if is_p:
        if is_holi:
            a *= 1.5
        if "晴" in str(w) or "多云" in str(w):
            a *= 1.3
        if "雨" in str(w) or "雪" in str(w) or "雾" in str(w):
            a *= 0.7
        if h < 6 or h >= 23:
            a *= 0.8
        if 7 <= h <= 9 or 17 <= h <= 19:
            a *= 1.2
    if is_t and is_holi:
        a *= 0.6
    return np.clip(np.round(a), 0, None).astype(int).tolist()

# ==============================
# 预计算数字token映射
# ==============================
def precompute_digit_mapping(tokenizer, device):
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
        self.print_interval = LOG_STEPS
        self.step_counter = 0
        self.mae_weight = MAE_WEIGHT
        
        print(f"✅ 优化版Trainer初始化完成，MAE权重={self.mae_weight}")
        print(f"✅ 对账输出间隔: 每 {self.print_interval} 步")

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
        
        total_loss = ce_loss + self.mae_weight * mae_loss
        
        self.step_counter += 1
        
        if self.step_counter % self.print_interval == 0 and self.state.global_step > 0:
            try:
                tok = getattr(self, "processing_class", self.tokenizer)
                s_idx = 0
                s_lab = shift_labels[s_idx].cpu().numpy()
                b_pos = np.where(s_lab == self.target_bracket_id)[0]
                
                if len(b_pos) > 0:
                    start = max(0, b_pos[-1] - 300)
                    safe = [x for x in s_lab[start:] if x != -100]
                    txt = tok.decode(safe, skip_special_tokens=False).split('<|eot_id|>')[0]
                    
                    station_match = re.search(r'Station:\s*([^\n]+)', txt)
                    feature_match = re.search(r'Feature:\s*([^\n]+)', txt)
                    
                    prev = digit_mask[:s_idx].sum().item()
                    curr = digit_mask[s_idx].sum().item()
                    
                    if curr > 0:
                        preds = expected_values[prev:prev+curr].cpu().tolist()
                        
                        print(f"\n{'='*70}")
                        print(f"📊 【核心客流对账单】 Step {self.state.global_step}")
                        print(f"{'='*70}")
                        
                        if station_match:
                            print(f"📍 站点: {station_match.group(1).strip()}")
                        if feature_match:
                            print(f"📈 特征: {feature_match.group(1).strip()}")
                        
                        print(f"🔮 LLM修正值(前5步): {[round(p, 2) for p in preds[:5]]}")
                        print(f"{'-'*50}")
                        print(f"🎯 Loss: {total_loss.item():.4f} | CE: {ce_loss.item():.4f} | MAE: {mae_loss.item():.4f}")
                        print(f"📈 进度: {self.state.global_step}/{self.state.max_steps} ({100*self.state.global_step/self.state.max_steps:.1f}%)")
                        print(f"{'='*70}\n")
                        
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

# 指令模板
instruction_text = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a professional highway traffic flow refiner.\n"
    "Follow these strict rules:\n"
    "1. Passenger cars: strongly affected by weather, time, holidays.\n"
    "2. Trucks: only reduced on holidays.\n"
    "3. Sunny/holiday → increase passenger cars.\n"
    "4. Rain/snow/fog/night → decrease passenger cars.\n"
    "5. Peak hours (7-9, 17-19) → increase passenger cars.\n"
    "Output Final Correction as a list.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Station: {}\n"
    "Feature: {}\n"
    "Time: {}\n"
    "DayType: {}\n"
    "Weather: {}\n"
    "GNN: {}\n"
    "Event: {}\n"
    "Provide corrected 8-hour prediction.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "Final Correction: ["
)

# ==============================
# 批量生成样本
# ==============================
NORMAL_SAMPLE_RATIO = 0.2

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
            is_t = f in (2, 3)
            is_d = f == 1
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
            
            dates = [t.strftime('%Y-%m-%d') for t in ts]
            is_work = np.array([is_workday(t) for t in ts])
            is_holi = ~is_work
            hours = np.array([t.hour for t in ts])
            events = [precise_events_map.get((d, s_name), "None") for d in dates]
            has_ev = np.array([e != "None" for e in events])
            
            need_correction = mae_per_sample > feature_threshold
            ev_mask = has_ev
            norm_mask = np.random.rand(len(mae_per_sample)) < NORMAL_SAMPLE_RATIO
            keep = np.where(need_correction | ev_mask | norm_mask)[0]
            
            for i in keep:
                try:
                    t3 = ts[i]
                    dt = dates[i]
                    h = hours[i]
                    ih = is_holi[i]
                    iw = is_work[i]
                    ev = events[i]
                    w = get_weather_fast(j, dt)
                    gnn_array = pred_j[i].round().astype(int).tolist()
                    
                    is_above_threshold = mae_per_sample[i] > feature_threshold
                    
                    if is_above_threshold:
                        cor = adjust_vectorized(gnn_array, is_p, is_t, ih, iw, h, w)
                        final_values = cor
                    else:
                        final_values = gnn_array
                    
                    t3s = t3.strftime('%Y-%m-%d %H:%M')
                    t4s = (t3 + pd.Timedelta(hours=7)).strftime('%Y-%m-%d %H:%M')
                    dayt = "Workday" if iw else "Off-day"
                    
                    prompt = instruction_text.format(
                        s_name, f_name, f"{t3s} to {t4s}", dayt, w, 
                        gnn_array, ev if ev != "None" else "None"
                    )
                    cot_text = f"{', '.join(map(str, final_values))}] <|eot_id|>"
                    
                    full = prompt + cot_text
                    
                    enc = tokenizer(full, truncation=True, max_length=1024, add_special_tokens=False)
                    p_enc = tokenizer(prompt, add_special_tokens=False)
                    pl = len(p_enc["input_ids"])
                    labs = [-100] * pl + enc["input_ids"][pl:]
                    
                    samples.append({
                        "input_ids": enc["input_ids"],
                        "labels": labs,
                        "attention_mask": [1] * len(enc["input_ids"]),
                    })
                except Exception as e:
                    continue
    return samples

def build_dataset_fast():
    n_proc = min(8, cpu_count())
    stations = list(range(num_stations))
    chunk_size = max(1, len(stations) // n_proc)
    chunks = [stations[i:i+chunk_size] for i in range(0, len(stations), chunk_size)]
    
    print(f"\n🚀 多进程生成：{n_proc} 进程，{len(chunks)} 块")
    with Pool(n_proc) as p:
        res = list(tqdm(p.imap(process_chunk, chunks), total=len(chunks), desc="生成样本"))
    
    all_samples = []
    for r in res:
        all_samples.extend(r)
    
    print(f"\n✅ 原始样本生成完成：{len(all_samples)} 条")
    return all_samples

# 生成数据集
print("\n🚀 生成训练样本...")
dataset_path = os.path.join(PROJECT_ROOT, "optimized_dataset.json")
if os.path.exists(dataset_path):
    print(f"✅ 加载已有数据集: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        generated_dataset = json.load(f)
    print(f"✅ 加载完成：{len(generated_dataset)} 条")
else:
    generated_dataset = build_dataset_fast()
    random.shuffle(generated_dataset)
    
    # 【优化】限制样本数量
    if len(generated_dataset) > MAX_SAMPLES:
        generated_dataset = generated_dataset[:MAX_SAMPLES]
        print(f"⚠️ 样本数已限制到 {MAX_SAMPLES} 条（原{len(generated_dataset)}条）")
    
    with open(dataset_path, "w", encoding="utf-8") as f:
        simplified_dataset = [{"input_ids": d["input_ids"], "labels": d["labels"], "attention_mask": d["attention_mask"]} 
                              for d in generated_dataset]
        json.dump(simplified_dataset, f, ensure_ascii=False)
    print(f"✅ 数据集已保存: {dataset_path}")

print(f"\n📊 最终训练样本数: {len(generated_dataset)}")

# ==============================
# 训练
# ==============================
if generated_dataset:
    ds = Dataset.from_list(generated_dataset)
    model.config.output_hidden_states = True
    
    # LoRA配置 - 包含 embed_tokens 和 lm_head
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head"
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none"
    )
    
    # 计算训练步数
    effective_batch = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = len(generated_dataset) // effective_batch
    total_steps = steps_per_epoch * NUM_EPOCHS
    expected_hours = total_steps * 7.5 / 3600
    
    args = TrainingArguments(
        output_dir=os.path.join(PROJECT_ROOT, "optimized_results"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        optim="adamw_8bit",
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=LOG_STEPS,
        report_to="none",
        save_strategy="epoch",
        gradient_checkpointing=True,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
        save_only_model=True,
    )
    
    trainer = OptimizedMAEHybridTrainer(
        model=model,
        train_dataset=ds,
        tokenizer=tokenizer,
        args=args,
        max_seq_length=1024,
        packing=False,
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n🚀 开始训练...")
    print("="*70)
    print("📊 优化配置总结:")
    print(f"   - 训练样本数: {len(generated_dataset)}")
    print(f"   - 训练轮数: {NUM_EPOCHS}")
    print(f"   - 批次大小: {BATCH_SIZE}")
    print(f"   - 梯度累积: {GRAD_ACCUM}")
    print(f"   - 有效批次: {effective_batch}")
    print(f"   - 每轮步数: ~{steps_per_epoch}")
    print(f"   - 总步数: ~{total_steps}")
    print(f"   - 预计时间: ~{expected_hours:.1f} 小时")
    print(f"   - MAE权重: {MAE_WEIGHT}")
    print(f"   - 学习率: {LEARNING_RATE}")
    print("="*70)
    
    trainer.train()
    
    out_path = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-optimized")
    model.save_pretrained(out_path)
    print(f"\n✅ 训练完成！模型已保存: {out_path}")
    
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("\n❌ 未生成训练样本")