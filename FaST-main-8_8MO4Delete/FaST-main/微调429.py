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
print("✓ 启动【真实数据校准版】阈值修正训练...")

BASE_DATA_PATH = PROJECT_ROOT
FINETUNE_OUTPUT_PATH = PROJECT_ROOT

YTRUE_TRAIN_PATH = os.path.join(FINETUNE_OUTPUT_PATH, "finetune_real_traffic.npz")
YPRED_TRAIN_PATH = os.path.join(FINETUNE_OUTPUT_PATH, "finetune_data.npz")
CARPARK_DES_PATH = os.path.join(BASE_DATA_PATH, "station_list_hngs.txt")
NATURAL_PATTERN_PATH = os.path.join(BASE_DATA_PATH, "station_natural_list_4feat.txt")
WEATHER_DATA_PATH = os.path.join(BASE_DATA_PATH, "160站点天气信息/")
EVENTS_CSV_PATH = os.path.join(BASE_DATA_PATH, "events_list_quan.csv")
HIS_DATA_CSV_PATH = os.path.join(BASE_DATA_PATH, "his_data_with_index.csv")

his_df = pd.read_csv(HIS_DATA_CSV_PATH)
his_df['时间'] = pd.to_datetime(his_df['时间'])
parking_data = his_df.set_index('时间').sort_index()

npz_true = np.load(YTRUE_TRAIN_PATH)
npz_pred = np.load(YPRED_TRAIN_PATH)

def get_array(npz, key):
    if key in npz: return npz[key]
    return npz['arr_0']

true_array = get_array(npz_true, 'target')
pred_array = get_array(npz_pred, 'prediction')

with open(CARPARK_DES_PATH, "r", encoding='utf-8') as f:
    carpark_des_list = [l.strip() for l in f if l.strip()]
with open(NATURAL_PATTERN_PATH, "r", encoding='utf-8') as f:
    natural_pattern_list = [l.strip() for l in f if l.strip()]

if true_array.ndim == 4:
    if true_array.shape[1] == 8 and true_array.shape[3] == 4:
        true_array = true_array.transpose(0,2,3,1)
        pred_array = pred_array.transpose(0,2,3,1)

num_samples, num_stations, num_features, seq_len = true_array.shape

FEATURE_NAMES = {
    0: "LittleCar_Up",
    1: "LittleCar_Down",
    2: "NonLittleCar_Up",
    3: "NonLittleCar_Down"
}

# ===================== 【根据你的真实CSV 100%精准设置】 =====================
ERROR_THRESHOLD_BY_FEATURE = {
    0: 135,   # 小客车上行
    1: 125,   # 小客车下行
    2: 32,    # 非小客车上行
    3: 30     # 非小客车下行
}
# ==========================================================================

def preload_all_weather(weather_dir, total_stations):
    d = {}
    if not os.path.exists(weather_dir): return d
    for j in tqdm(range(total_stations)):
        try:
            s = f"{j:03d}"
            m = [f for f in os.listdir(weather_dir) if f.startswith(s)]
            if not m: continue
            df = pd.read_csv(os.path.join(weather_dir, m[0]), encoding='utf-8')
            dc = next((c for c in df.columns if '日期' in c), None)
            wc = next((c for c in df.columns if '天气' in c), None)
            if not dc or not wc: continue
            df[dc] = pd.to_datetime(df[dc]).dt.date.astype(str)
            d[j] = df.set_index(dc)[wc].to_dict()
        except:
            continue
    return d

weather_cache = preload_all_weather(WEATHER_DATA_PATH, num_stations)

def load_chinese_events(p):
    m = {}
    if os.path.exists(p):
        df = pd.read_csv(p, encoding='utf-8')
        for _, r in df.iterrows():
            m[(r['日期'], r['站点名称'])] = r['事件描述']
    return m

precise_events_map = load_chinese_events(EVENTS_CSV_PATH)

# ===================== 【极轻微保守微调，绝对不崩】 =====================
def adjust_vectorized(gnp, is_p, is_t, is_holi, is_wk, h, w):
    a = np.array(gnp, dtype=np.float32)
    if is_p:
        if is_holi:
            a *= 1.07      # 1.05 → 1.07 极轻微加强
        if "晴" in w or "多云" in w:
            a *= 1.04      # 1.05 → 1.04 防止过冲
        if "雨" in w or "雪" in w or "雾" in w:
            a *= 0.92      # 0.95 → 0.92 刚好匹配数据
    if is_t and is_holi:
        a *= 0.83          # 0.85 → 0.83 极轻微加强
    return np.clip(np.round(a), 0, None).astype(int).tolist()
# ========================================================================

comparison_log = []

def log_comparison(station_name, feature_name, timestamp, gnn_value, true_value, llm_value, error_before, error_after):
    comparison_log.append({
        "station": station_name, "feature": feature_name, "time": timestamp,
        "gnn": gnn_value, "true": true_value, "llm": llm_value,
        "before": error_before, "after": error_after
    })

# def precompute_digit_mapping(tokenizer, device):
def precompute_digit_mapping(tokenizer, device):
    id_to_val = {}
    digit_ids = []
    for i in range(tokenizer.vocab_size):
        try:
            t = tokenizer.decode([i]).strip()
            # 【修复：只保留合法数字，跳过特殊字符】
            if t and t.isdigit():
                num = float(t)
                # 【再加一层保护：只保留 0-9 的单个数字】
                if num.is_integer() and 0 <= num <= 9:
                    id_to_val[i] = num
                    digit_ids.append(i)
        except:
            continue
    digit_ids_tensor = torch.tensor(digit_ids, device=device)
    val_vec = torch.tensor([id_to_val[i] for i in digit_ids], device=device)
    return id_to_val, digit_ids, digit_ids_tensor, val_vec, 510

    id_to_val = {}
    digit_ids = []
    for i in range(tokenizer.vocab_size):
        t = tokenizer.decode([i]).strip()
        if t and t.isdigit():
            id_to_val[i] = float(t)
            digit_ids.append(i)
    digit_ids_tensor = torch.tensor(digit_ids, device=device)
    val_vec = torch.tensor([id_to_val[i] for i in digit_ids], device=device)
    return id_to_val, digit_ids, digit_ids_tensor, val_vec, 510

class OptimizedMAEHybridTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tok = self.tokenizer
        dev = self.args.device
        self.id2v, self.dids, self.dt, self.vv, self.br = precompute_digit_mapping(tok, dev)
        self.step = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs["output_hidden_states"] = True
        out = model(**inputs)
        ce_loss = out.loss
        logits = model.lm_head(out.hidden_states[-1].to(model.lm_head.weight.dtype))
        labels = inputs["labels"][..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        digit_mask = torch.isin(labels, self.dt) & (labels != -100)
        mae_loss = torch.tensor(0.0, device=self.args.device)

        if digit_mask.any():
            ps = torch.softmax(shift_logits[digit_mask][:, self.dt].float(), dim=-1)
            ev = (ps * self.vv).sum(dim=-1)
            tg = torch.tensor([self.id2v[int(i)] for i in labels[digit_mask]], device=self.args.device)
            mae_loss = F.l1_loss(ev, tg)

        total_loss = ce_loss + 0.11 * mae_loss
        self.step += 1
        if self.step % 10 == 0:
            print(f"Step {self.state.global_step} | Loss {total_loss:.3f} | CE {ce_loss:.3f} | MAE {mae_loss:.3f}")
        return (total_loss, out) if return_outputs else total_loss

instruction_text = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a professional traffic refiner.\n"
    "Output reasoning + Final Correction: [x1,...,x8].<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Refine:\n1. Station: {}\n2. Feature: {}\n3. Time: {} to {}\n4. Pattern: {}\n5. Weather: {}\n6. GNN: {}\n7. Event: {}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

def process_chunk(stations_chunk):
    samples = []
    for j in stations_chunk:
        if j >= len(carpark_des_list): continue
        desc = carpark_des_list[j]
        s_name = desc.split()[0]
        for f in range(4):
            is_p = f in (0,1)
            is_t = f in (2,3)
            pi = j*4 + f
            pat = natural_pattern_list[pi] if pi < len(natural_pattern_list) else "General"
            p = pred_array[:,j,f,:]
            t = true_array[:,j,f,:]
            mae_s = np.abs(p-t).mean(axis=1)
            th = ERROR_THRESHOLD_BY_FEATURE[f]
            keep = np.where(mae_s > th)[0]
            for i in keep:
                try:
                    ts = parking_data.index[i*num_stations + j]
                    dt = ts.strftime('%Y-%m-%d')
                    is_holi = not is_workday(ts)
                    h = ts.hour
                    w = weather_cache.get(j, {}).get(dt, "Unknown")
                    ev = precise_events_map.get((dt, s_name), "None")
                    gnn_seq = p[i].round().astype(int).tolist()
                    true_seq = t[i]
                    cor = adjust_vectorized(gnn_seq, is_p, is_t, is_holi, None, h, w)
                    t3s = ts.strftime('%Y-%m-%d %H:%M')
                    t4s = (ts+pd.Timedelta(hours=7)).strftime('%Y-%m-%d %H:%M')
                    prompt = instruction_text.format(desc, FEATURE_NAMES[f], t3s, t4s, pat, w, gnn_seq, ev)
                    ans = f"Final Correction: [{','.join(map(str,cor))}] <|eot_id|>"
                    samples.append({"text": prompt + ans})
                    log_comparison(s_name, FEATURE_NAMES[f], t3s, gnn_seq, true_seq.tolist(), cor,
                                  np.abs(gnn_seq-true_seq).mean(), np.abs(cor-true_seq).mean())
                except:
                    continue
    return samples

def build_dataset_fast():
    n_proc = min(8, cpu_count())
    stations = list(range(num_stations))
    chunks = [stations[i::n_proc] for i in range(n_proc)]
    with Pool(n_proc) as pool:
        res = pool.map(process_chunk, chunks)
    all_samples = []
    for r in res: all_samples += r
    random.shuffle(all_samples)
    if len(all_samples) > 30000: all_samples = all_samples[:30000]
    return all_samples

dataset_path = os.path.join(PROJECT_ROOT, "quick429.json")
if os.path.exists(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        generated_dataset = json.load(f)
else:
    generated_dataset = build_dataset_fast()
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(generated_dataset, f, ensure_ascii=False)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/user/Llama-3.1-8B",
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    device_map={"":0}
)

model = FastLanguageModel.get_peft_model(
    model, r=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none"
)

args = TrainingArguments(
    output_dir=os.path.join(PROJECT_ROOT, "results-quick"),
    num_train_epochs=10,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    optim="adamw_8bit",
    learning_rate=5e-5,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant":False},
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    save_only_model=True,
    report_to="none",
)

trainer = OptimizedMAEHybridTrainer(
    model=model,
    train_dataset=Dataset.from_list(generated_dataset),
    tokenizer=tokenizer,
    args=args,
    max_seq_length=1024,
    packing=False
)
#  不从断点加载
# ckpts = glob.glob(os.path.join(PROJECT_ROOT, "results-quick/checkpoint-*"))
# resume = max(ckpts, key=os.path.getctime) if ckpts else None

trainer.train(resume_from_checkpoint=None)

if comparison_log:
    pd.DataFrame(comparison_log).to_csv(os.path.join(PROJECT_ROOT, "comparison_log.csv"), index=False, encoding='utf-8-sig')

model.save_pretrained(os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick429"))
print("✅ 训练完成！")