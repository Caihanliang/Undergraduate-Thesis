import json
import re
import numpy as np
import torch
import pandas as pd
import os
from tqdm import tqdm
from unsloth import FastLanguageModel
from chinese_calendar import is_workday

# ===================== 配置 =====================
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"
FINETUNED_MODEL_PATH = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick427")
DATA_PATH = PROJECT_ROOT

YTRUE_TEST_PATH = os.path.join(DATA_PATH, "finetune_real_traffic.npz")
YPRED_TEST_PATH = os.path.join(DATA_PATH, "finetune_data.npz")
CARPARK_DES_PATH = os.path.join(DATA_PATH, "station_list_hngs.txt")
NATURAL_PATTERN_PATH = os.path.join(DATA_PATH, "station_natural_list_4feat.txt")
WEATHER_DATA_PATH = os.path.join(DATA_PATH, "160站点天气信息/")
EVENTS_CSV_PATH = os.path.join(DATA_PATH, "events_list_quan.csv")
HIS_DATA_CSV_PATH = os.path.join(DATA_PATH, "his_data_with_index.csv")

# 只推理 1 个站点（超快！）
TARGET_STATION = 0
INFERENCE_OUTPUT = "fast_inference_result.json"

FEATURE_NAMES = {
    0: "Passenger Car Up",
    1: "Passenger Car Down",
    2: "Non-Passenger Car Up",
    3: "Non-Passenger Car Down"
}

INSTRUCTION_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a professional highway traffic flow refiner.\n"
    "Follow these strict rules:\n"
    "1. Passenger cars: strongly affected by weather, time, holidays.\n"
    "2. Trucks: almost NOT affected by weather; only reduced on holidays.\n"
    "3. Sunny/holidays → increase passenger cars.\n"
    "4. Rain/snow/fog/night → decrease passenger cars.\n"
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

# ===================== 工具函数 =====================
def extract_numbers(text):
    try:
        last_bracket_idx = text.rfind("[")
        if last_bracket_idx == -1:
            return None
        content = text[last_bracket_idx:]
        nums = [int(n) for n in re.findall(r"\d+", content)]
        if len(nums) >= 8:
            return nums[:8]
    except:
        return None
    return None

def masked_mae_np(preds, labels, null_val=0.0):
    mask = (labels != null_val).astype('float32')
    if np.mean(mask) == 0: return 0.0
    mask /= np.mean(mask)
    mae = np.abs(preds - labels).astype('float32')
    return np.mean(np.nan_to_num(mae * mask))

def masked_rmse_np(preds, labels, null_val=0.0):
    mask = (labels != null_val).astype('float32')
    if np.mean(mask) == 0: return 0.0
    mask /= np.mean(mask)
    mse = np.square(preds - labels).astype('float32')
    return np.sqrt(np.mean(np.nan_to_num(mse * mask)))

def masked_mape_np(preds, labels, null_val=0.0):
    mask = (labels > null_val).astype('float32')
    if np.mean(mask) == 0: return 0.0
    mask /= np.mean(mask)
    mape = np.abs((preds - labels) / labels).astype('float32')
    return np.mean(np.nan_to_num(mask * mape))

# ===================== 加载数据 =====================
def load_aux():
    his_df = pd.read_csv(HIS_DATA_CSV_PATH)
    his_df['时间'] = pd.to_datetime(his_df['时间'])
    parking = his_df.set_index('时间').sort_index()

    with open(CARPARK_DES_PATH, encoding='utf-8') as f:
        stations = [l.strip() for l in f if l.strip()]
    with open(NATURAL_PATTERN_PATH, encoding='utf-8') as f:
        patterns = [l.strip() for l in f if l.strip()]

    events = {}
    if os.path.exists(EVENTS_CSV_PATH):
        df = pd.read_csv(EVENTS_CSV_PATH, encoding='utf-8')
        df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
        for _, r in df.iterrows():
            events[(r['日期'], str(r['站点名称']).strip())] = r['事件描述']

    weather = {}
    if os.path.exists(WEATHER_DATA_PATH):
        for f in os.listdir(WEATHER_DATA_PATH):
            if not f.endswith('.csv'): continue
            j = int(f[:3])
            wd = pd.read_csv(os.path.join(WEATHER_DATA_PATH, f))
            wd['日期'] = pd.to_datetime(wd.iloc[:,0]).dt.date.astype(str)
            weather[j] = dict(zip(wd['日期'], wd.iloc[:,1]))
    return parking, stations, patterns, events, weather

# ===================== 极速推理：只推 1 个站 =====================
def main():
    parking, stations, patterns, events, weather = load_aux()
    print(f"✅ 加载数据成功，推理站点：{TARGET_STATION}")

    # 加载 GNN 数据
    yt = np.load(YTRUE_TEST_PATH)
    yp = np.load(YPRED_TEST_PATH)
    true = yt['target'] if 'target' in yt else yt['arr_0']
    pred = yp['prediction'] if 'prediction' in yp else yp['arr_0']

    if true.ndim == 4:
        if true.shape[1] == 8 and true.shape[3] == 4:
            true = true.transpose(0,2,3,1)
            pred = pred.transpose(0,2,3,1)
    N, S, F, T = true.shape
    print(f"✅ 数据：{N}样本 × {S}站 × {F}特征 × {T}步")

    # 加载模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=FINETUNED_MODEL_PATH,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        device_map={"":0},
    )
    FastLanguageModel.for_inference(model)
    print("✅ 模型加载完成，开始极速推理...")

    # 只推理目标站点
    llm_out = []
    gnn_all = []
    true_all = []
    fail = 0

    for i in tqdm(range(N)):
        sample_llm = []
        for f in range(4):
            # 构造输入
            g = pred[i, TARGET_STATION, f]
            row = i * S + TARGET_STATION
            t = parking.index[min(row, len(parking)-1)]
            date = t.strftime('%Y-%m-%d')
            station_name = stations[TARGET_STATION].split()[0]

            # 构建 prompt
            prompt = INSTRUCTION_TEMPLATE.format(
                stations[TARGET_STATION],
                FEATURE_NAMES[f],
                t.strftime('%Y-%m-%d %H:%M'),
                (t+pd.Timedelta(hours=7)).strftime('%Y-%m-%d %H:%M'),
                "Workday" if is_workday(t) else "Off-day",
                patterns[TARGET_STATION*4 + f] if (TARGET_STATION*4 + f)<len(patterns) else "General",
                weather.get(TARGET_STATION, {}).get(date, "Unknown"),
                g.round().astype(int).tolist(),
                events.get((date, station_name), "None")
            )

            # 生成 —— 【修复了 temperature 报错】
            inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,  # 关闭采样 = 确定性输出
                    top_p=1.0
                )
            res = extract_numbers(tokenizer.decode(out[0]))
            if res is None:
                res = g.round().astype(int).tolist()
                fail += 1
            sample_llm.append(res)
        llm_out.append(sample_llm)
        gnn_all.append(pred[i, TARGET_STATION].tolist())
        true_all.append(true[i, TARGET_STATION].tolist())

    # 保存结果
    with open(INFERENCE_OUTPUT, 'w') as f:
        json.dump({"llm": llm_out, "gnn": gnn_all, "true": true_all}, f)

    # ===================== 输出指标 =====================
    print("\n" + "="*80)
    print(f"📊 极速推理完成 | 站点 {TARGET_STATION} | 失败 {fail}")
    print("="*80)

    llm_np = np.array(llm_out, dtype=np.float32)
    gnn_np = np.array(gnn_all, dtype=np.float32)
    true_np = np.array(true_all, dtype=np.float32)

    # 总体指标
    mae_gnn = masked_mae_np(gnn_np, true_np)
    mae_llm = masked_mae_np(llm_np, true_np)
    rmse_gnn = masked_rmse_np(gnn_np, true_np)
    rmse_llm = masked_rmse_np(llm_np, true_np)
    mape_gnn = masked_mape_np(gnn_np, true_np)
    mape_llm = masked_mape_np(llm_np, true_np)

    imp_mae = (mae_gnn - mae_llm) / mae_gnn * 100
    imp_rmse = (rmse_gnn - rmse_llm) / rmse_gnn * 100
    imp_mape = (mape_gnn - mape_llm) / mape_gnn * 100

    print(f"\n📈 最终指标 (你要的三个值)")
    print(f"MAE:  GNN {mae_gnn:.2f} → LLM {mae_llm:.2f}   ↑ {imp_mae:.1f}%")
    print(f"RMSE: GNN {rmse_gnn:.2f} → LLM {rmse_llm:.2f}   ↑ {imp_rmse:.1f}%")
    print(f"MAPE: GNN {mape_gnn:.2f} → LLM {mape_llm:.2f}   ↑ {imp_mape:.1f}%")
    print("="*80)

if __name__ == "__main__":
    main()