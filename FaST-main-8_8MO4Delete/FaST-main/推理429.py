import json
import re
import numpy as np
import torch
import os
from tqdm import tqdm
from unsloth import FastLanguageModel

# ===================== 【1. 配置路径】 =====================
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"
MODEL_PATH = os.path.join(PROJECT_ROOT, "llama-3-1-8b-highway-finetuned-quick429")

DATA_PATH = "/home/user/FSTLLM/own/data/FSTLLM-main/FSTLLM_STGNN/data/"
SAVE_JSON = "inference_results_llama_final.json"

TARGET_STATION = 0
MAX_SAMPLE = 2603  # 和参考代码一致
FEATURE_NAMES = ["LittleCar_Up", "LittleCar_Down", "NonLittleCar_Up", "NonLittleCar_Down"]

# ===================== 【2. 指令模板（和训练100%一致）】 =====================
INSTRUCTION_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a professional traffic refiner.\n"
    "Output reasoning + Final Correction: [x1,...,x8].<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Refine:\n1. Station: {}\n2. Feature: {}\n3. Time: 2025-01-01 00:00 to 2025-01-01 07:00\n4. Pattern: General\n5. Weather: Unknown\n6. GNN: {}\n7. Event: None<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "Final Correction: ["
)

# ===================== 【3. 数字提取（和参考代码一样强）】 =====================
# def extract_numbers(text):
# ===================== 【你要的版本：只从 Final Correction 提取】 =====================
def extract_numbers_safe(text, fallback):
    try:
        # 1. 严格匹配：Final Correction: [数字,数字,...]
        pattern = r'Final Correction:\s*\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # 提取里面的数字部分
            num_str = match.group(1)
            # 分割成单个数字
            nums = re.findall(r'\d+', num_str)
            # 过滤合法流量值
            valid = [int(n) for n in nums if n.isdigit() and 0 <= int(n) <= 9999]
            if len(valid) >= 8:
                return valid[:8]
    
    except Exception as e:
        pass

    # 提取失败才用GNN兜底
    return fallback[:8]
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

# ===================== 【4. 指标计算（和参考代码完全一样）】 =====================
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

# ===================== 【5. 加载模型】 =====================
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        device_map={"": 0}
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

# ===================== 【6. 主推理流程】 =====================
def main():
    print("🚀 加载模型...")
    model, tokenizer = load_model()

    print("📂 加载 GNN 数据...")
    y_true = np.load(os.path.join(DATA_PATH, "ytrue.npy.npz"))['arr_0']
    y_pred = np.load(os.path.join(DATA_PATH, "ypred.npy.npz"))['arr_0']

    y_true = y_true.transpose(1, 2, 0)
    y_pred = y_pred.transpose(1, 2, 0)

    answer_list = []
    llm_all = []
    gnn_all = []
    true_all = []
    fail = 0

    print("🔥 开始推理...")
    for i in tqdm(range(min(MAX_SAMPLE, y_true.shape[0]))):
        gnn_seq = y_pred[i, TARGET_STATION, :].tolist()
        true_seq = y_true[i, TARGET_STATION, :].tolist()

        # 构造prompt（和训练完全一样）
        prompt = INSTRUCTION_TEMPLATE.format("Station", FEATURE_NAMES[0], gnn_seq)

        # 生成
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                temperature=0.01
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_list.append(text)

        # 提取
        pred = extract_numbers(text)
        if pred is None:
            pred = gnn_seq
            fail += 1

        llm_all.append(pred)
        gnn_all.append(gnn_seq)
        true_all.append(true_seq)

    # 保存推理结果
    with open(SAVE_JSON, "w", encoding="utf-8") as f:
        json.dump(answer_list, f, ensure_ascii=False, indent=2)

    # ===================== 【7. 输出指标（和参考代码一模一样）】 =====================
    llm_np = np.array(llm_all)
    gnn_np = np.array(gnn_all)
    true_np = np.array(true_all)

    print("\n" + "="*90)
    print(f"{'Step':<10} | {'Metric':<6} | {'GNN':<15} | {'LLM':<15} | {'Improve':<10}")
    print("-"*90)

    for step in range(8):
        t = true_np[:, step]
        l = llm_np[:, step]
        g = gnn_np[:, step]

        g_mae = masked_mae_np(g, t)
        l_mae = masked_mae_np(l, t)
        imp = (g_mae-l_mae)/g_mae*100 if g_mae>0 else 0

        g_rmse = masked_rmse_np(g, t)
        l_rmse = masked_rmse_np(l, t)

        print(f"Step {step:<5} | MAE    | {g_mae:<15.4f} | {l_mae:<15.4f} | {imp:>+8.2f}%")
        print(f"{'':<10} | RMSE   | {g_rmse:<15.4f} | {l_rmse:<15.4f} |")

    # 总体
    total_gnn_mae = masked_mae_np(gnn_np, true_np)
    total_llm_mae = masked_mae_np(llm_np, true_np)
    total_imp = (total_gnn_mae - total_llm_mae) / total_gnn_mae * 100

    print("-"*90)
    print(f"{'OVERALL':<10} | MAE    | {total_gnn_mae:<15.4f} | {total_llm_mae:<15.4f} | {total_imp:>+8.2f}%")
    print("="*90)
    print(f"✅ 完成 | 提取失败：{fail} | 结果已保存到 {SAVE_JSON}")

if __name__ == "__main__":
    main()