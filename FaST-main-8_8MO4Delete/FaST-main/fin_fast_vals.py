"""
python fin_fast_vals.py
这个就是最终的版本了 
"""
import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
os.environ["UNSLOTH_SKIP_INIT_CHECK"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
import re
import sys
import gc
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 新增：设置项目根路径（适配你的文件位置）
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"

class Logger(object):
    def __init__(self, filename=os.path.join(PROJECT_ROOT, "3training_0_011.txt")):
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
print("✓ 正在启动【超快版】隐状态截获 MAE 混合训练...")

# --- 路径配置 ---
BASE_DATA_PATH = PROJECT_ROOT
FINETUNE_OUTPUT_PATH = PROJECT_ROOT

YTRUE_TRAIN_PATH = os.path.join(FINETUNE_OUTPUT_PATH, "finetune_real_traffic.npz")
YPRED_TRAIN_PATH = os.path.join(FINETUNE_OUTPUT_PATH, "finetune_data.npz")
CARPARK_DES_PATH = os.path.join(BASE_DATA_PATH, "station_list_hngs.txt")
NATURAL_PATTERN_PATH = os.path.join(BASE_DATA_PATH, "station_natural_list_4feat.txt")
WEATHER_DATA_PATH = os.path.join(BASE_DATA_PATH, "160站点天气信息/")
EVENTS_CSV_PATH = os.path.join(BASE_DATA_PATH, "events_list_quan.csv")
TIMESTAMPS_CSV_PATH = os.path.join(BASE_DATA_PATH, "timestamps.csv")
HIS_DATA_CSV_PATH = os.path.join(BASE_DATA_PATH, "his_data_with_index.csv")

# --- 加载数据 ---
print("正在加载数据并精简天气信息...")
his_df = pd.read_csv(HIS_DATA_CSV_PATH)
his_df['时间'] = pd.to_datetime(his_df['时间'])
parking_data = his_df.set_index('时间').sort_index()

npz_true = np.load(YTRUE_TRAIN_PATH)
npz_pred = np.load(YPRED_TRAIN_PATH)

print(f"✓ NPZ文件键名检测:")
print(f"  ytrue_train.npz 键名: {list(npz_true.keys())}")
print(f"  ypred_train.npz 键名: {list(npz_pred.keys())}")

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
print(f"  true_array shape: {true_array.shape}")
print(f"  pred_array shape: {pred_array.shape}")

if np.allclose(true_array, pred_array):
    raise ValueError("❌ 预测值与真实值完全相同！请检查文件路径")
else:
    diff = np.abs(true_array - pred_array)
    print(f"\n✓ 数据验证通过:")
    print(f"  平均绝对误差 (MAE): {diff.mean():.2f}")

print(f"\n直接使用his_data_with_index.csv的时间索引")
print(f"  起始时间: {parking_data.index[0]}")
print(f"  结束时间: {parking_data.index[-1]}")
print(f"  总行数: {len(parking_data)}")

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
print(f"  true_array shape: {true_array.shape}")
print(f"  pred_array shape: {pred_array.shape}")

if true_array.ndim == 4:
    if true_array.shape[1] == 8 and true_array.shape[3] == 4:
        true_array = true_array.transpose(0, 2, 3, 1)
        pred_array = pred_array.transpose(0, 2, 3, 1)
    elif true_array.shape[1] == 4 and true_array.shape[3] == 8:
        true_array = true_array.transpose(0, 2, 1, 3)
        pred_array = pred_array.transpose(0, 2, 1, 3)

num_samples, num_stations, num_features, seq_len = true_array.shape
print(f"\n✓ 维度对齐完成:")
print(f"  样本数: {num_samples} | 站点数: {num_stations} | 特征数: {num_features} | 序列长度: {seq_len}")

FEATURE_NAMES = {
    0: "Passenger Car Up",
    1: "Passenger Car Down",
    2: "Non-Passenger Car Up",
    3: "Non-Passenger Car Down"
}

# ==============================
# 【超快优化1】一次性预加载所有天气
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

# --- 加载事件 ---
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

# ==============================================
# 混合损失训练器（完全不变，保证效果一致）
# ==============================================
class NoLogitMAEHybridTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tok = getattr(self, "processing_class", self.tokenizer)
        self.id_to_val = {}
        dig_ids = []
        for i in range(tok.vocab_size):
            t = tok.decode([i]).strip()
            if t and t.isdigit():
                try:
                    self.id_to_val[i] = float(t)
                    dig_ids.append(i)
                except:
                    pass
        self.digit_ids = torch.tensor(dig_ids).to(self.args.device)
        self.target_bracket_id = 510
        print(f"✅ 数字Token加载: {len(dig_ids)} 个")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs["output_hidden_states"] = True
        outputs = model(**inputs)
        last_hidden = outputs.hidden_states[-1]
        logits = model.lm_head(last_hidden.to(next(model.lm_head.parameters()).dtype))
        ce_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        labels = inputs.get("labels")
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()

        all_digit_mask = torch.isin(shift_labels, self.digit_ids)
        digit_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for b in range(shift_labels.shape[0]):
            pos = (shift_labels[b] == self.target_bracket_id).nonzero(as_tuple=True)[0]
            if len(pos) > 0:
                s = pos[-1].item() + 1
                digit_mask[b, s:] = all_digit_mask[b, s:]
        digit_mask &= (shift_labels != -100)

        mae_loss = torch.tensor(0.0).to(self.args.device)
        total_loss = ce_loss
        expected_values = torch.tensor([]).to(self.args.device)

        if digit_mask.any():
            dig_logits = shift_logits[digit_mask][:, self.digit_ids]
            probs = torch.softmax(dig_logits.to(torch.float32), dim=-1)
            val_vec = torch.tensor([self.id_to_val[int(t)] for t in self.digit_ids]).to(self.args.device)
            expected_values = (probs * val_vec).sum(dim=-1)
            target_labels = shift_labels[digit_mask]
            target_vals = torch.zeros_like(expected_values)
            for i, tid in enumerate(target_labels):
                target_vals[i] = self.id_to_val.get(int(tid), 0.0)

            mask = (target_vals != 0).float()
            if mask.mean() > 0:
                mask /= mask.mean()

            mae_loss = torch.mean(torch.abs(expected_values - target_vals) * mask)
            total_loss = ce_loss + 0.11 * mae_loss

            if self.state.global_step % 1 == 0:
                try:
                    s_idx = 0
                    s_lab = shift_labels[s_idx].cpu().numpy()
                    b_pos = (s_lab == self.target_bracket_id).nonzero()[0]
                    if len(b_pos) == 0:
                        return (total_loss, outputs) if return_outputs else total_loss
                    start = max(0, b_pos[-1] - 150)
                    safe = [x for x in s_lab[start:] if x != -100]
                    txt = self.tokenizer.decode(safe, skip_special_tokens=False).split('<|eot_id|>')[0] + " <|eot_id|>"

                    prev = digit_mask[:s_idx].sum().item()
                    curr = digit_mask[s_idx].sum().item()
                    if curr > 0:
                        preds = expected_values[prev:prev+curr].cpu().tolist()
                        st = b_pos[-1].item() + 1
                        raw = shift_labels[s_idx, st:].cpu().tolist()

                        p_list, t_list = [], []
                        cp, ct = "", ""
                        ptr = 0
                        for tid in raw:
                            if tid == -100 or tid == 128009:
                                break
                            if tid in self.id_to_val:
                                c = self.tokenizer.decode([tid]).strip()
                                ct += c
                                if ptr < len(preds):
                                    cp += str(int(round(preds[ptr])))
                                    ptr += 1
                            elif tid in (11, 60):
                                if ct:
                                    t_list.append(int(ct) if ct.isdigit() else 0)
                                    p_list.append(int(cp) if cp.isdigit() else 0)
                                    ct, cp = "", ""
                        print(f"\n" + "="*15 + " 核心客流对账单 " + "="*15)
                        print(f"Step {self.state.global_step}")
                        print(f"预览:\n{txt}")
                        print(f"预测: {p_list[:8]}")
                        print(f"真实: {t_list[:8]}")
                        print(f"Loss: {total_loss.item():.4f} | CE:{ce_loss.item():.4f} | MAE:{mae_loss.item():.4f}")
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

# 指令模板（完全不变）
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
# 【超快优化2】向量化修正
# ==============================
def adjust_vectorized(gnp, is_p, is_t, is_holi, is_wk, h, w):
    a = np.array(gnp, dtype=np.float32)
    if is_p:
        if is_holi: a *= 1.25
        if ((7 <= h <=9) or (17<=h<=19)) and is_wk: a *= 1.15
        if h <6 or h >=23: a *= 0.7
        if "晴" in w or "多云" in w: a *= 1.1
        if "雨" in w or "雪" in w or "雾" in w: a *= 0.85
    if is_t and is_holi: a *= 0.8
    return np.clip(np.round(a), 0, None).astype(int).tolist()

# ==============================
# 【超快优化3】多进程批量生成样本
# ==============================
ERROR_THRESHOLD = 60
NORMAL_SAMPLE_RATIO = 0.1

def process_chunk (stations_chunk):
    samples = []
    for j in stations_chunk:
        if j >= len(carpark_des_list): continue
        desc = carpark_des_list[j]
        s_name = desc.split()[0]

        for f in range(num_features):
            f_name = FEATURE_NAMES[f]
            is_p = f in (0,1)
            is_t = f in (2,3)
            is_d = f == 1
            pi = j*num_features + f
            pat = natural_pattern_list[pi] if pi < len(natural_pattern_list) else "General"

            pred_j = pred_array[:, j, f, :]
            true_j = true_array[:, j, f, :]
            mae_per_sample = np.abs(pred_j - true_j).mean(axis=1)

            rows = np.arange(num_samples) * num_stations + j
            rows = rows[rows < len(parking_data)]
            ts = parking_data.index[rows]
            if len(ts) == 0: continue

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
                        elif 7<=h<=9 or 17<=h<=19:
                            cot.append(f"3. Reasoning: Workday peak hour → passenger cars rise sharply.")
                        elif h<6 or h>=23:
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

                    prompt = instruction_text.format(desc, f_name, t3s, t4s, dayt, pat, w, gnn, ev if ev!="None" else "None")
                    full = prompt + cot_text

                    enc = tokenizer(full, truncation=True, max_length=1024, add_special_tokens=False)
                    p_enc = tokenizer(prompt, add_special_tokens=False)
                    pl = len(p_enc["input_ids"])
                    labs = [-100]*pl + enc["input_ids"][pl:]

                    samples.append({
                        "input_ids": enc["input_ids"],
                        "labels": labs,
                        "attention_mask": [1]*len(enc["input_ids"]),
                    })
                except:
                    continue
    return samples

def build_dataset_fast():
    n_proc = min(8, cpu_count())
    stations = list(range(num_stations))
    chunk = max(1, len(stations)//n_proc)
    chunks = [stations[i:i+chunk] for i in range(0, len(stations), chunk)]
    print(f"\n🚀 多进程生成：{n_proc} 进程，{len(chunks)} 块")
    with Pool(n_proc) as p:
        res = list(tqdm(p.imap(process_chunk, chunks), total=len(chunks), desc="生成样本"))
    out = []
    for r in res:
        out.extend(r)
    return out

# ==============================
# 开始生成
# ==============================
print("\n🚀 超快版生成训练样本...")
generated_dataset = build_dataset_fast()
random.shuffle(generated_dataset)

event_cnt = sum(1 for d in generated_dataset if d.get('reason')=='Event')
err_cnt = sum(1 for d in generated_dataset if d.get('reason')=='Error')
norm_cnt = sum(1 for d in generated_dataset if d.get('reason')=='Normal')

print(f"\n✅ 样本生成完成：总计 {len(generated_dataset)} 条")

dataset_path = os.path.join(PROJECT_ROOT, "3training_0_011.json")
with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(generated_dataset, f, ensure_ascii=False, indent=2)
print(f"✅ 数据集已保存至: {dataset_path}")

# ==============================
# 训练
# ==============================
if generated_dataset:
    ds = Dataset.from_list(generated_dataset)
    model.config.output_hidden_states = True
    model = FastLanguageModel.get_peft_model(
        model, r=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","embed_tokens","lm_head"],
        lora_alpha=16, lora_dropout=0, bias="none"
    )

    args = TrainingArguments(
        output_dir=os.path.join(PROJECT_ROOT, "results"),
        num_train_epochs=10,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        optim="paged_adamw_8bit",
        learning_rate=5e-5,
        bf16=True,
        logging_steps=1,
        report_to="none",
        save_strategy="epoch"
    )

    trainer = NoLogitMAEHybridTrainer(
        model=model,
        train_dataset=ds,
        tokenizer=tokenizer,
        args=args,
        max_seq_length=1024,
        packing=False
    )

    print("\n🚀 开始训练...")
    trainer.train()

    out_path = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned")
    model.save_pretrained(out_path)
    print(f"\n✅ 训练完成！模型已保存至: {out_path}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("\n❌ 未生成训练样本")


# """
# python fin_fast_vals.py
# """
# import os
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
# os.environ["UNSLOTH_SKIP_INIT_CHECK"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# import logging
# import random
# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import json
# from unsloth import FastLanguageModel
# from datasets import Dataset
# from transformers import TrainingArguments
# from trl import SFTTrainer
# from chinese_calendar import is_workday
# import re
# import sys
# import gc
# import os

# # 新增：设置项目根路径（适配你的文件位置）
# PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"

# class Logger(object):
#     def __init__(self, filename=os.path.join(PROJECT_ROOT, "3training_0_011.txt")):
#         self.terminal = sys.stdout
#         self.log = open(filename, "w", encoding="utf-8")

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#         self.log.flush()

#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()

# sys.stdout = Logger()
# logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
# print("✓ 正在启动【隐状态截获版】MAE 混合训练...")

# # --- 路径配置（修改：使用/mnt下的文件）---
# BASE_DATA_PATH = PROJECT_ROOT
# FINETUNE_OUTPUT_PATH = PROJECT_ROOT

# # 关键修改：使用你上传的文件路径
# YTRUE_TRAIN_PATH = os.path.join(FINETUNE_OUTPUT_PATH, "finetune_real_traffic.npz")
# YPRED_TRAIN_PATH = os.path.join(FINETUNE_OUTPUT_PATH, "finetune_data.npz")
# CARPARK_DES_PATH = os.path.join(BASE_DATA_PATH, "station_list_hngs.txt")
# NATURAL_PATTERN_PATH = os.path.join(BASE_DATA_PATH, "station_natural_list_4feat.txt")
# WEATHER_DATA_PATH = os.path.join(BASE_DATA_PATH, "160站点天气信息/")
# EVENTS_CSV_PATH = os.path.join(BASE_DATA_PATH, "events_list_quan.csv")
# TIMESTAMPS_CSV_PATH = os.path.join(BASE_DATA_PATH, "timestamps.csv")
# HIS_DATA_CSV_PATH = os.path.join(BASE_DATA_PATH, "his_data_with_index.csv")

# # --- 加载数据 ---
# print("正在加载数据并精简天气信息...")
# his_df = pd.read_csv(HIS_DATA_CSV_PATH)
# his_df['时间'] = pd.to_datetime(his_df['时间'])
# parking_data = his_df.set_index('时间').sort_index()

# npz_true = np.load(YTRUE_TRAIN_PATH)
# npz_pred = np.load(YPRED_TRAIN_PATH)

# print(f"✓ NPZ文件键名检测:")
# print(f"  ytrue_train.npz 键名: {list(npz_true.keys())}")
# print(f"  ypred_train.npz 键名: {list(npz_pred.keys())}")

# if 'target' in npz_true:
#     true_array = npz_true['target']
# elif 'arr_0' in npz_true:
#     true_array = npz_true['arr_0']
# else:
#     raise KeyError(f"ytrue_train.npz 未找到有效键")

# if 'prediction' in npz_pred:
#     pred_array = npz_pred['prediction']
# elif 'arr_0' in npz_pred:
#     pred_array = npz_pred['arr_0']
# else:
#     raise KeyError(f"ypred_train.npz 未找到有效键")

# print(f"✓ 流量数据加载成功:")
# print(f"  true_array shape: {true_array.shape}")
# print(f"  pred_array shape: {pred_array.shape}")

# if np.allclose(true_array, pred_array):
#     raise ValueError("❌ 预测值与真实值完全相同！请检查文件路径")
# else:
#     diff = np.abs(true_array - pred_array)
#     print(f"\n✓ 数据验证通过:")
#     print(f"  平均绝对误差 (MAE): {diff.mean():.2f}")

# # ✅ 优化：不再加载timestamps.csv，直接从his_data获取时间
# # 原因：1) 避免DataFrame查找开销 2) 事件匹配只需日期+站点
# print(f"\n直接使用his_data_with_index.csv的时间索引")
# print(f"  起始时间: {parking_data.index[0]}")
# print(f"  结束时间: {parking_data.index[-1]}")
# print(f"  总行数: {len(parking_data)}")

# # 加载站点列表
# print(f"\n正在加载站点列表: {CARPARK_DES_PATH}")
# with open(CARPARK_DES_PATH, "r", encoding='utf-8') as f:
#     carpark_des_list = [l.strip() for l in f if l.strip()]
# print(f"✓ 站点列表加载成功: {len(carpark_des_list)} 个站点")

# # 加载自然语言模式
# print(f"正在加载自然语言模式: {NATURAL_PATTERN_PATH}")
# with open(NATURAL_PATTERN_PATH, "r", encoding='utf-8') as f:
#     natural_pattern_list = [l.strip() for l in f if l.strip()]
# print(f"✓ 自然语言模式加载成功: {len(natural_pattern_list)} 条描述")

# # 维度对齐
# print(f"\n原始数据形状分析:")
# print(f"  true_array shape: {true_array.shape}")
# print(f"  pred_array shape: {pred_array.shape}")

# if true_array.ndim == 4:
#     if true_array.shape[1] == 8 and true_array.shape[3] == 4:
#         true_array = true_array.transpose(0, 2, 3, 1)
#         pred_array = pred_array.transpose(0, 2, 3, 1)
#     elif true_array.shape[1] == 4 and true_array.shape[3] == 8:
#         true_array = true_array.transpose(0, 2, 1, 3)
#         pred_array = pred_array.transpose(0, 2, 1, 3)

# num_samples, num_stations, num_features, seq_len = true_array.shape
# print(f"\n✓ 维度对齐完成:")
# print(f"  样本数: {num_samples} | 站点数: {num_stations} | 特征数: {num_features} | 序列长度: {seq_len}")

# FEATURE_NAMES = {
#     0: "Passenger Car Up",
#     1: "Passenger Car Down",
#     2: "Non-Passenger Car Up",
#     3: "Non-Passenger Car Down"
# }

# # 天气函数
# def get_weather_at_time(station_idx, target_time):
#     try:
#         if not os.path.exists(WEATHER_DATA_PATH):
#             return "Unknown"
#         weather_files = os.listdir(WEATHER_DATA_PATH)
#         station_prefix = f"{station_idx:03d}"
#         matched_file = None
#         for wf in weather_files:
#             if wf.startswith(station_prefix):
#                 matched_file = wf
#                 break
#         if not matched_file:
#             return "Unknown"

#         weather_path = os.path.join(WEATHER_DATA_PATH, matched_file)
#         if matched_file.endswith('.csv'):
#             weather_df_local = pd.read_csv(weather_path, encoding='utf-8')
#         elif matched_file.endswith('.xlsx'):
#             weather_df_local = pd.read_excel(weather_path)
#         else:
#             return "Unknown"

#         date_col = None
#         for col in weather_df_local.columns:
#             if '日期' in col or 'date' in col.lower():
#                 date_col = col
#                 break
#         if not date_col:
#             return "Unknown"

#         weather_df_local[date_col] = pd.to_datetime(weather_df_local[date_col])
#         weather_df_local = weather_df_local.set_index(date_col).sort_index()
#         code_col = None
#         for col in weather_df_local.columns:
#             if "天气" in col or "weather" in col.lower():
#                 code_col = col
#                 break
#         if not code_col:
#             return "Unknown"

#         target_date = target_time.date()
#         weather_df_local.index = weather_df_local.index.date
#         if target_date in weather_df_local.index:
#             return str(weather_df_local.loc[target_date, code_col])
#         else:
#             return "Unknown"
#     except Exception as e:
#         print(f"⚠️ 天气获取失败: {e}")
#         return "Unknown"

# # --- 关键修改2：加载中文事件文件 ---
# def load_chinese_events(path):
#     if not os.path.exists(path):
#         print(f"⚠️ 未找到事件文件: {path}")
#         return {}
#     try:
#         df_ev = pd.read_csv(path, encoding='utf-8')
#         print(f"✓ 事件文件表头: {list(df_ev.columns)}")

#         date_col = "日期"
#         station_col = "站点名称"
#         event_col = "事件描述"

#         required_cols = [date_col, station_col, event_col]
#         for col in required_cols:
#             if col not in df_ev.columns:
#                 raise ValueError(f"❌ 事件文件缺少必要表头: {col}")

#         df_ev[date_col] = pd.to_datetime(df_ev[date_col]).dt.strftime('%Y-%m-%d')
#         df_ev[station_col] = df_ev[station_col].astype(str).str.strip()
#         df_ev[event_col] = df_ev[event_col].astype(str).str.strip()

#         event_dict = {}
#         for _, row in df_ev.iterrows():
#             dt = row[date_col]
#             st = row[station_col]
#             ev = row[event_col]
#             if st and st != 'nan' and ev and ev != 'nan':
#                 event_dict[(dt, st)] = ev

#         print(f"✅ 中文事件加载完成：{len(event_dict)} 条有效事件")
#         sample_events = list(event_dict.items())[:5]
#         for (dt, st), ev in sample_events:
#             print(f"  示例: {dt} | {st} → {ev}")
#         return event_dict
#     except Exception as e:
#         print(f"❌ 事件加载失败: {e}")
#         return {}

# precise_events_map = load_chinese_events(EVENTS_CSV_PATH)

# # ==============================================
# # 混合损失训练器
# # ==============================================
# class NoLogitMAEHybridTrainer(SFTTrainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         local_tokenizer = getattr(self, "processing_class", self.tokenizer)
#         self.id_to_val = {}
#         all_digit_ids = []
#         for i in range(local_tokenizer.vocab_size):
#             t = local_tokenizer.decode([i]).strip()
#             if t and all(c.isdigit() for c in t):
#                 try:
#                     self.id_to_val[i] = float(t)
#                     all_digit_ids.append(i)
#                 except:
#                     pass
#         self.digit_ids = torch.tensor(all_digit_ids).to(self.args.device)
#         self.target_bracket_id = 510
#         print(f"✅ 捕获数字Token: {len(all_digit_ids)} 个")

#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         inputs["output_hidden_states"] = True
#         outputs = model(**inputs)
#         last_hidden = outputs.hidden_states[-1]
#         logits = model.lm_head(last_hidden.to(next(model.lm_head.parameters()).dtype))
#         ce_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

#         labels = inputs.get("labels")
#         shift_labels = labels[..., 1:].contiguous()
#         shift_logits = logits[..., :-1, :].contiguous()

#         all_digit_mask = torch.isin(shift_labels, self.digit_ids)
#         digit_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
#         for b in range(shift_labels.shape[0]):
#             bracket_pos = (shift_labels[b] == self.target_bracket_id).nonzero(as_tuple=True)[0]
#             if len(bracket_pos) > 0:
#                 start_from = bracket_pos[-1].item() + 1
#                 digit_mask[b, start_from:] = all_digit_mask[b, start_from:]
#         digit_mask &= (shift_labels != -100)

#         mae_loss = torch.tensor(0.0).to(self.args.device)
#         total_loss = ce_loss
#         expected_values = torch.tensor([]).to(self.args.device)

#         if digit_mask.any():
#             dig_logits = shift_logits[digit_mask][:, self.digit_ids]
#             probs = torch.softmax(dig_logits.to(torch.float32), dim=-1)
#             val_vec = torch.tensor([self.id_to_val[int(t)] for t in self.digit_ids]).to(self.args.device)
#             expected_values = (probs * val_vec).sum(dim=-1)
#             target_labels = shift_labels[digit_mask]
#             target_vals = torch.zeros_like(expected_values)
#             for i in range(len(target_labels)):
#                 tid = int(target_labels[i])
#                 target_vals[i] = self.id_to_val.get(tid, 0.0)
            
#             mask = (target_vals != 0).float()
#             if mask.mean() > 0:
#                 mask /= mask.mean()
            
#             mae_loss = torch.mean(torch.abs(expected_values - target_vals) * mask)
#             total_loss = ce_loss + 0.11 * mae_loss

#             if self.state.global_step % 1 == 0:
#                 sample_idx = 0
#                 s_labels = shift_labels[sample_idx].cpu().numpy()
#                 b_indices = (s_labels == self.target_bracket_id).nonzero()[0]

#                 if len(b_indices) > 0:
#                     start_preview = max(0, b_indices[-1] - 150)
#                     relevant_ids = s_labels[start_preview:]
#                     safe_ids = [tid for tid in relevant_ids if tid != -100]
#                     raw_text_only = self.tokenizer.decode(safe_ids, skip_special_tokens=False).split('<|eot_id|>')[0] + " <|eot_id|>"

#                     prev_mae_bits = digit_mask[:sample_idx].sum().item()
#                     curr_mae_bits = digit_mask[sample_idx].sum().item()

#                     if curr_mae_bits > 0:
#                         batch_preds = expected_values[prev_mae_bits: prev_mae_bits + curr_mae_bits].cpu().tolist()
#                         start_ptr = b_indices[-1].item() + 1
#                         raw_token_seq = shift_labels[sample_idx, start_ptr:].cpu().tolist()

#                         final_pred_res = []
#                         final_true_res = []
#                         current_p_str = ""
#                         current_t_str = ""
#                         pred_ptr = 0

#                         for tid in raw_token_seq:
#                             if tid == -100 or tid == 128009: break
#                             if tid in self.id_to_val:
#                                 token_char = self.tokenizer.decode([tid]).strip()
#                                 current_t_str += token_char
#                                 if pred_ptr < len(batch_preds):
#                                     p_val = int(round(batch_preds[pred_ptr]))
#                                     current_p_str += str(p_val)
#                                     pred_ptr += 1
#                             elif tid == 11 or tid == 60:
#                                 if current_t_str:
#                                     final_true_res.append(int(current_t_str) if current_t_str.isdigit() else 0)
#                                     final_pred_res.append(int(current_p_str) if current_p_str.isdigit() else 0)
#                                     current_t_str = ""
#                                     current_p_str = ""

#                         print(f"\n" + "=" * 15 + " 核心客流对账单 (硬核对齐版) " + "=" * 15)
#                         print(f"Step: {self.state.global_step}")
#                         print(f"原文预览:\n{raw_text_only}")
#                         print(f"还原预测数值: {final_pred_res[:8]}")
#                         print(f"还原真实数值: {final_true_res[:8]}")
#                         print("-" * 60)
#                         print(f"Loss Total: {total_loss.item():.4f} | CE: {ce_loss.item():.4f} | MAE: {mae_loss.item():.4f}")
#                         print("=" * 65 + "\n")

#         return (total_loss, outputs) if return_outputs else total_loss

# # 加载模型
# print(f"\n✓ 正在加载模型...")
# # 使用checkpoint
# checkpoint_dir = "config/cai/results/checkpoint-6993"  # 改成你的checkpoint
# import os
# if os.path.exists(checkpoint_dir):
#      print(f"从checkpoint恢复: {checkpoint_dir}")
#      model_name = checkpoint_dir
#      resume_from_checkpoint = True
# else:
#     print("从头开始训练")
#     model_name = "/home/user/Llama-3.1-8B"
#     resume_from_checkpoint = False

# model, tokenizer = FastLanguageModel.from_pretrained(
#     # model_name="/home/user/Llama-3.1-8B",
#     model_name=model_name,
#     max_seq_length=1024,
#     load_in_4bit=True,
#     dtype=torch.bfloat16,
#     device_map={"": 0},
# )

# # 指令模板
# instruction_text = (
#     "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
#     "You are a professional highway traffic flow refiner.\n"
#     "Follow these strict rules:\n"
#     "1. Passenger cars: strongly affected by weather, time, holidays, GDP, population, events.\n"
#     "2. Trucks: almost NOT affected by weather/time/GDP; only reduced on holidays.\n"
#     "3. Sunny/holidays/peaks/high GDP → increase passenger cars.\n"
#     "4. Rain/snow/fog/night/low GDP → decrease passenger cars.\n"
#     "5. Up/down trends similar; downstream stable.\n"
#     "Output reasoning + Final Correction.<|eot_id|>"
#     "<|start_header_id|>user<|end_header_id|>\n\n"
#     "Refine:\n"
#     "1. Station: {}\n"
#     "2. Feature: {}\n"
#     "3. Time: {} to {} | DayType: {}\n"
#     "4. Flow Pattern: {}\n"
#     "5. Weather: {}\n"
#     "6. GNN: {}\n"
#     "7. Event: {}<|eot_id|>"
#     "<|start_header_id|>assistant<|end_header_id|>\n\n"
# )

# generated_dataset = []
# ERROR_THRESHOLD = 60
# NORMAL_SAMPLE_RATIO = 0.1

# print("\n🚀 开始生成 100% 对齐规则的 CoT 样本...")

# # 全局去重集合（修复位置）
# printed_events = set()

# for j in range(num_stations):
#     if j >= len(carpark_des_list):
#         print(f"⚠️ 站点索引超出范围: j={j}, 站点列表长度={len(carpark_des_list)}")
#         continue
#     full_desc = carpark_des_list[j]
#     station_name = full_desc.split(' ')[0].strip()
#     print(f"\n===== 处理站点 {j+1}/{num_stations}: {station_name} =====")

#     for feat_idx in range(num_features):
#         feat_name = FEATURE_NAMES[feat_idx]
#         print(f"   → 处理车流类型：{feat_name}")
#         is_passenger = feat_idx in (0, 1)
#         is_truck = feat_idx in (2, 3)
#         is_downstream = feat_idx == 1

#         pat_idx = j * num_features + feat_idx
#         pat_str = natural_pattern_list[pat_idx] if pat_idx < len(natural_pattern_list) else "General traffic pattern"

#         for i in range(num_samples):
#             try:
#                 # ✅ 优化：直接从his_data索引获取时间，避免DataFrame查找开销
#                 # his_data格式：每个时间步有157行（每个站点一行）
#                 # 样本i对应的时间索引 = i * num_stations + j (站点j在该时间的行)
#                 base_row_idx = i * num_stations + j
                
#                 if base_row_idx >= len(parking_data):
#                     break
                
#                 t3 = parking_data.index[base_row_idx]  # 预测起始时间
#                 t4 = t3 + pd.Timedelta(hours=7)  # 预测结束时间（8小时）
#                 date_key = t3.strftime('%Y-%m-%d')
#                 h = t3.hour

#                 # 先计算 day_type
#                 is_work = is_workday(t3)
#                 is_holiday = not is_work  # ✅ 修复：在每个样本循环中定义is_holiday
                
#                 # 🔍 调试：验证is_holiday已定义
#                 assert 'is_holiday' in locals(), f"❌ is_holiday未定义! i={i}, j={j}"
                
#                 day_type = "Workday" if is_work else "Off-day"
                
#                 # 再拼接时间描述
#                 t3_str = t3.strftime('%Y-%m-%d %H:%M')
#                 t4_str = t4.strftime('%Y-%m-%d %H:%M')
#                 weekday = t3.strftime('%A')
#                 time_desc = f"{t3_str} to {t4_str} | {weekday} | {day_type}"

#                 # 事件匹配 + 去重打印（已修复）
#                 event_info = precise_events_map.get((date_key, station_name), "None")
#                 has_event = event_info != "None"
#                 event_key = (station_name, date_key)

#                 if has_event and event_key not in printed_events:
#                     printed_events.add(event_key)
#                     # print(f"   ✅ 匹配到事件: {date_key} | {station_name} → {event_info}")

#                 pred_vals = pred_array[i, j, feat_idx]
#                 true_vals = true_array[i, j, feat_idx]
#                 mae = np.mean(np.abs(pred_vals - true_vals))
#                 weather = get_weather_at_time(j, t3)
#                 gnn_vals = [int(round(v)) for v in pred_vals]

#                 reason = ""
#                 if mae > ERROR_THRESHOLD:
#                     reason = "Error"
#                 elif has_event:
#                     reason = "Event"
#                 elif random.random() < NORMAL_SAMPLE_RATIO:
#                     reason = "Normal"
#                 if not reason:
#                     continue

#                 adjusted = [float(v) for v in gnn_vals]
#                 if is_passenger:
#                     if is_holiday:
#                         adjusted = [x*1.25 for x in adjusted]
#                     if ((7 <= h <= 9) or (17 <= h <= 19)) and is_work:
#                         adjusted = [x*1.15 for x in adjusted]
#                     if h < 6 or h >= 23:
#                         adjusted = [x*0.7 for x in adjusted]
#                     if "晴" in weather or "多云" in weather:
#                         adjusted = [x*1.1 for x in adjusted]
#                     if "雨" in weather or "雪" in weather or "雾" in weather:
#                         adjusted = [x*0.85 for x in adjusted]
#                 if is_truck and is_holiday:
#                     adjusted = [x*0.8 for x in adjusted]

#                 final_corr = [max(0, round(x)) for x in adjusted]

#                 cot = []
#                 cot.append(f"1. Station: {station_name} area.")
#                 cot.append(f"2. Feature: {feat_name}; Time: {time_desc}; Event: {'Yes (' + event_info + ')' if has_event else 'No'}.")
                
#                 if is_passenger:
#                     if has_event:
#                         cot.append(f"3. Reasoning: Special event ({event_info}) → passenger car flow surges significantly.")
#                     elif is_holiday:
#                         cot.append(f"3. Reasoning: Holiday → passenger cars stay high all day.")
#                     elif (7 <= h <= 9) or (17 <= h <= 19):
#                         cot.append(f"3. Reasoning: Workday peak hour → passenger cars rise sharply.")
#                     elif h < 6 or h >= 23:
#                         cot.append(f"3. Reasoning: Night low-peak → passenger cars are very low.")
#                     else:
#                         cot.append(f"3. Reasoning: Routine period → stable passenger car flow.")
#                 else:
#                     cot.append(f"3. Reasoning: Truck flow is stable; only limited on holidays.")

#                 cot.append(f"4. Weather: {weather} → adjust passenger car flow according to rules.")
#                 cot.append(f"5. Economy: Traffic base depends on local GDP and population.")
#                 cot.append(f"6. Trend: Downstream flow is more stable" if is_downstream else f"6. Trend: Upstream and downstream trends are consistent.")
#                 cot.append(f"Final Correction: [{', '.join(map(str, final_corr))}]")
                
#                 cot_text = " ".join(cot) + " <|eot_id|>"
#                 prompt = instruction_text.format(
#                     full_desc, feat_name, t3, t4, day_type,
#                     pat_str, weather, gnn_vals, event_info if has_event else "None"
#                 )
#                 full_text = prompt + cot_text

#                 enc = tokenizer(full_text, truncation=True, max_length=1024, add_special_tokens=False)
#                 prompt_enc = tokenizer(prompt, add_special_tokens=False)
#                 p_len = len(prompt_enc["input_ids"])
#                 labels = [-100] * p_len + enc["input_ids"][p_len:]

#                 generated_dataset.append({
#                     "input_ids": enc["input_ids"],
#                     "labels": labels,
#                     "attention_mask": [1]*len(enc["input_ids"]),
#                     "reason": reason,
#                     "station": station_name,
#                     "date": date_key,
#                     "event": event_info
#                 })
#             except Exception as e:
#                 print(f"⚠️ 样本生成失败 (i={i}, j={j}): {str(e)[:100]}")
#                 continue

# event_cnt = sum(1 for d in generated_dataset if d['reason']=='Event')
# err_cnt = sum(1 for d in generated_dataset if d['reason']=='Error')
# norm_cnt = sum(1 for d in generated_dataset if d['reason']=='Normal')

# print(f"\n✅ 样本生成完成：总计 {len(generated_dataset)} 条")
# print(f"   事件样本: {event_cnt} 条")
# print(f"   误差样本: {err_cnt} 条")
# print(f"   普通样本: {norm_cnt} 条")

# dataset_path = os.path.join(PROJECT_ROOT, "3training_0_011.json")
# with open(dataset_path, "w", encoding="utf-8") as f:
#     json.dump(generated_dataset, f, ensure_ascii=False, indent=2)
# print(f"✅ 数据集已保存至: {dataset_path}")

# if len(generated_dataset) > 0:
#     random.shuffle(generated_dataset)
#     ds = Dataset.from_list(generated_dataset)

#     model.config.output_hidden_states = True
#     model = FastLanguageModel.get_peft_model(
#         model, r=32,
#         target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","embed_tokens","lm_head"],
#         lora_alpha=16, lora_dropout=0, bias="none"
#     )

#     args = TrainingArguments(
#         output_dir=os.path.join(PROJECT_ROOT, "results"),
#         num_train_epochs=10,
#         per_device_train_batch_size=32,
#         gradient_accumulation_steps=1,
#         optim="paged_adamw_8bit",
#         learning_rate=5e-5,
#         bf16=True,
#         logging_steps=1,
#         report_to="none",
#         save_strategy="epoch"
#     )

#     trainer = NoLogitMAEHybridTrainer(
#         model=model,
#         train_dataset=ds,
#         tokenizer=tokenizer,
#         args=args,
#         max_seq_length=1024,
#         packing=False
#     )

#     print("\n🚀 开始训练...")
#     trainer.train()
    
#     output_model_path = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned")
#     model.save_pretrained(output_model_path)
#     print(f"\n✅ 训练完成！模型已保存至: {output_model_path}")

#     del model, trainer
#     gc.collect()
#     torch.cuda.empty_cache()
# else:
#     print("\n❌ 未生成任何训练样本，请检查数据路径和匹配逻辑！")
