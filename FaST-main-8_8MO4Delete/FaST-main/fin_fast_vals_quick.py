""" 
python fin_fast_vals_quick.py
 - 显存兼容最终版
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

# 设置项目根路径
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"

# 重定向输出
class Logger(object):
    def __init__(self, filename=os.path.join(PROJECT_ROOT, "quick.txt")):
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
print("✓ 正在启动【显存兼容版】隐状态截获 MAE 混合训练...")

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
# 预加载天气
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
            if path.endswith('.csv'):
                df = pd.read_csv(path, encoding='utf-8')
            else:
                continue
            
            d_col = next((c for c in df.columns if '日期' in c or 'date' in c.lower()), None)
            w_col = next((c for c in df.columns if '天气' in c or 'weather' in c.lower()), None)
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
# 优化版训练器（减少打印频率）
# ==============================
class OptimizedMAEHybridTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tok = getattr(self, "processing_class", self.tokenizer)
        
        device = self.args.device
        self.id_to_val, self.digit_ids, self.digit_ids_tensor, self.val_vec, self.target_bracket_id = \
            precompute_digit_mapping(tok, device)
        
        self.digit_set = set(self.digit_ids)
        self.print_interval = 100
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
                    start = max(0, b_pos[-1] - 150)
                    safe = [x for x in s_lab[start:] if x != -100]
                    txt = tok.decode(safe, skip_special_tokens=False).split('<|eot_id|>')[0] + " <|eot_id|>"
                    
                    print(f"\n" + "="*15 + " 核心客流对账单 " + "="*15)
                    print(f"Step {self.state.global_step}")
                    print(f"预览:\n{txt[:300]}...")
                    print(f"Loss: {total_loss.item():.4f} | CE:{ce_loss.item():.4f} | MAE:{mae_loss.item():.4f}")
                    print(f"训练进度: {self.state.global_step}/{self.state.max_steps} ({100*self.state.global_step/self.state.max_steps:.1f}%)")
                    print("="*60)
            except:
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
    "1. Passenger cars: strongly affected by weather, time, holidays, GDP, population, events.\n"
    "2. Trucks: almost NOT affected by weather/time/GDP; only reduced on holidays.\n"
    "3. Sunny/holidays/peaks/high GDP → increase passenger cars.\n"
    "4. Rain/snow/fog/night/low GDP → decrease passenger cars.\n"
    "5. Up/down trends similar; downstream stable.\n"
    "Output reasoning + Final Correction.<|eot_id|>"
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
# 向量化修正函数
# ==============================
def adjust_vectorized(gnp, is_p, is_t, is_holi, is_wk, h, w):
    a = np.array(gnp, dtype=np.float32)
    if is_p:
        if is_holi:
            a *= 1.25
        if ((7 <= h <= 9) or (17 <= h <= 19)) and is_wk:
            a *= 1.15
        if h < 6 or h >= 23:
            a *= 0.7
        if "晴" in w or "多云" in w:
            a *= 1.1
        if "雨" in w or "雪" in w or "雾" in w:
            a *= 0.85
    if is_t and is_holi:
        a *= 0.8
    return np.clip(np.round(a), 0, None).astype(int).tolist()

# ==============================
# 批量生成样本
# ==============================
ERROR_THRESHOLD = 60
NORMAL_SAMPLE_RATIO = 0.1

def process_chunk(stations_chunk):
    """处理站点块"""
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
            
            err_mask = mae_per_sample > ERROR_THRESHOLD
            ev_mask = has_ev
            norm_mask = np.random.rand(len(mae_per_sample)) < NORMAL_SAMPLE_RATIO
            keep = np.where(err_mask | ev_mask | norm_mask)[0]
            
            for i in keep:
                try:
                    t3 = ts[i]
                    dt = dates[i]
                    h = hours[i]
                    ih = is_holi[i]
                    iw = is_work[i]
                    ev = events[i]
                    w = get_weather_fast(j, dt)
                    gnn = pred_j[i].round().astype(int).tolist()
                    cor = adjust_vectorized(gnn, is_p, is_t, ih, iw, h, w)
                    
                    t4 = t3 + pd.Timedelta(hours=7)
                    t3s = t3.strftime('%Y-%m-%d %H:%M')
                    t4s = t4.strftime('%Y-%m-%d %H:%M')
                    dy = t3.strftime('%A')
                    dayt = "Workday" if iw else "Off-day"
                    time_desc = f"{t3s} to {t4s} | {dy} | {dayt}"
                    
                    cot = []
                    cot.append(f"1. Station: {s_name} area.")
                    cot.append(f"2. Feature: {f_name}; Time: {time_desc}; Event: {'Yes ('+ev+')' if ev!='None' else 'No'}.")
                    if is_p:
                        if ev != "None":
                            cot.append(f"3. Reasoning: Special event ({ev}) → passenger car flow surges significantly.")
                        elif ih:
                            cot.append(f"3. Reasoning: Holiday → passenger cars stay high all day.")
                        elif 7 <= h <= 9 or 17 <= h <= 19:
                            cot.append(f"3. Reasoning: Workday peak hour → passenger cars rise sharply.")
                        elif h < 6 or h >= 23:
                            cot.append(f"3. Reasoning: Night low-peak → passenger cars are very low.")
                        else:
                            cot.append(f"3. Reasoning: Routine period → stable passenger car flow.")
                    else:
                        cot.append(f"3. Reasoning: Truck flow is stable; only limited on holidays.")
                    cot.append(f"4. Weather: {w} → adjust passenger car flow according to rules.")
                    cot.append(f"5. Economy: Traffic base depends on local GDP and population.")
                    cot.append(f"6. Trend: Downstream stable" if is_d else f"6. Trend: Up/down consistent.")
                    cot.append(f"Final Correction: [{', '.join(map(str, cor))}]")
                    cot_text = " ".join(cot) + " <|eot_id|>"
                    
                    prompt = instruction_text.format(desc, f_name, t3s, t4s, dayt, pat, w, gnn, ev if ev != "None" else "None")
                    full = prompt + cot_text
                    
                    samples.append({
                        "prompt": prompt,
                        "cot_text": cot_text,
                        "full_text": full
                    })
                except:
                    continue
    return samples

def build_dataset_fast():
    """多进程生成数据集"""
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
print("\n🚀 超快版生成训练样本...")
generated_dataset = build_dataset_fast()
random.shuffle(generated_dataset)

MAX_SAMPLES = 30000  # 减少到3万样本
if len(generated_dataset) > MAX_SAMPLES:
    generated_dataset = generated_dataset[:MAX_SAMPLES]
    print(f"⚠️ 样本数已限制到 {MAX_SAMPLES} 条")

print(f"\n✅ 样本生成完成：总计 {len(generated_dataset)} 条")

dataset_path = os.path.join(PROJECT_ROOT, "quick.json")
with open(dataset_path, "w", encoding="utf-8") as f:
    simplified_dataset = [{"input_ids": d["input_ids"], "labels": d["labels"], "attention_mask": d["attention_mask"]} 
                          for d in generated_dataset]
    json.dump(simplified_dataset, f, ensure_ascii=False)
print(f"✅ 数据集已保存至: {dataset_path}")

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
    BATCH_SIZE = 8      # 从32降到8
    GRAD_ACCUM = 4      # 增加到4，有效batch=32
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
        output_dir=os.path.join(PROJECT_ROOT, "results-quick"),
        num_train_epochs=10,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        optim="adamw_8bit",                    # 使用8bit优化器节省显存
        learning_rate=5e-5,
        bf16=True,
        logging_steps=100,
        report_to="none",
        save_strategy="epoch",
        dataloader_num_workers=2,              # 减少worker数
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,            # 开启梯度检查点节省显存
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
    
    # 清空缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 显存监控
    print(f"\n📊 显存状态:")
    print(f"   总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   已用显存: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"   空闲显存: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")
    
    print("\n🚀 开始训练...")
    print(f"⚡ 显存优化配置:")
    print(f"   - 批次大小: {BATCH_SIZE}")
    print(f"   - 梯度累积: {GRAD_ACCUM}")
    print(f"   - 有效批次: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"   - 梯度检查点: 开启")
    print(f"   - 打印间隔: 每100步")
    print(f"   - 预期速度: 3-5秒/步")
    print(f"   - 预期总时间: 12-16小时")
    print("="*60)
    
    trainer.train()
    
    out_path = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick")
    model.save_pretrained(out_path)
    print(f"\n✅ 训练完成！模型已保存至: {out_path}")
    
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("\n❌ 未生成训练样本")