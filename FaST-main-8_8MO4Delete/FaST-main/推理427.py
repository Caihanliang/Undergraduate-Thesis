import json
import re
import numpy as np
import torch
import os
from tqdm import tqdm
from unsloth import FastLanguageModel

# ===================== 路径【完全不变】 =====================
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"
MODEL_PATH = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick427")

YTRUE_PATH = os.path.join(PROJECT_ROOT, "finetune_real_traffic.npz")
YPRED_PATH = os.path.join(PROJECT_ROOT, "finetune_data.npz")
STATION_LIST = os.path.join(PROJECT_ROOT, "station_list_hngs.txt")

FEATURE_NAMES = [
    "Passenger Car Up",
    "Passenger Car Down",
    "Non-Passenger Car Up",
    "Non-Passenger Car Down"
]

# 单站点测试
TARGET_STATION = 10
MAX_SAMPLE = 20

# ==============================================
# 🟢 【关键】这里 100% 复制你训练代码的指令！
# ==============================================
INSTRUCTION_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a professional highway traffic flow refiner.
Follow these strict rules:
1. Passenger cars: strongly affected by weather, time, holidays, GDP, population, events.
2. Trucks: almost NOT affected by weather/time/GDP; only reduced on holidays.
3. Sunny/holidays/high GDP → increase passenger cars.
4. Rain/snow/fog/night/low GDP → decrease passenger cars.
5. Up/down trends similar; downstream stable.
Output reasoning + Final Correction.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Refine:
1. Station: {}
2. Feature: {}
3. Time: 2025-01-01 00:00 to 2025-01-01 07:00 | DayType: Workday
4. Flow Pattern: General
5. Weather: Unknown
6. GNN: {}
7. Event: None<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""

# ==============================================
# 🟢 提取函数（正确版）
# ==============================================
def extract_correction(text):
    match = re.search(r'Final Correction:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        return None
    nums = re.findall(r'\d+', match.group(1))
    valid = [int(x) for x in nums if x.isdigit()]
    if len(valid) >=8:
        return valid[:8]
    return None

# 指标
def cal_metrics(pred, true):
    mask = true != 0
    mae = np.mean(np.abs(pred[mask]-true[mask]))
    rmse = np.sqrt(np.mean((pred[mask]-true[mask])**2))
    mape = np.mean(np.abs(pred[mask]-true[mask])/(true[mask]+1e-8))*100
    return round(mae,2), round(rmse,2), round(mape,2)

# ===================== 主函数 =====================
def main():
    with open(STATION_LIST, encoding='utf-8') as f:
        stations = [l.strip() for l in f]
    station_name = stations[TARGET_STATION]

    true = np.load(YTRUE_PATH)['target'][:MAX_SAMPLE, :, TARGET_STATION, :]
    pred = np.load(YPRED_PATH)['prediction'][:MAX_SAMPLE, :, TARGET_STATION, :]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)

    llm_result = np.zeros_like(pred)
    print(f"\n🚀 推理站点：{station_name}")

    for i in tqdm(range(MAX_SAMPLE)):
        for f_idx in range(4):
            gnn = pred[i,:,f_idx].tolist()

            # 🟢 正确 prompt
            prompt = INSTRUCTION_TEMPLATE.format(
                station_name, FEATURE_NAMES[f_idx], gnn
            )

            inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=120, do_sample=False)

            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            cor = extract_correction(text)

            if cor is None:
                cor = gnn

            llm_result[i,:,f_idx] = cor

    # 输出
    print("\n" + "="*90)
    print(f"📊 【真实LLM修正结果】")
    print("="*90)

    for f_idx in range(4):
        g_mae,g_rmse,g_mape = cal_metrics(pred[:,:,f_idx], true[:,:,f_idx])
        l_mae,l_rmse,l_mape = cal_metrics(llm_result[:,:,f_idx], true[:,:,f_idx])

        print(f"\n{FEATURE_NAMES[f_idx]}")
        print(f"MAE    GNN: {g_mae:>6.2f}    LLM: {l_mae:>6.2f}")
        print(f"RMSE   GNN: {g_rmse:>6.2f}    LLM: {l_rmse:>6.2f}")
        print(f"MAPE   GNN: {g_mape:>6.2f}    LLM: {l_mape:>6.2f}")

if __name__ == "__main__":
    main()