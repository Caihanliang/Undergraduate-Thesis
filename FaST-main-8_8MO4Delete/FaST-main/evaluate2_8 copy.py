import json
import re
import numpy as np

# --- 工具函数：从推理结果文本中提取方括号内的 8 个数字 ---
def extract_numbers(text):
    """
    针对 CoT 格式增强的提取函数：
    锁定 'Final Correction:' 之后且位于全文最后的方括号内容
    """
    try:
        # 1. 寻找最后一个方括号的位置，这是最安全的锚点
        last_bracket_idx = text.rfind("[")
        if last_bracket_idx == -1:
            return None

        # 2. 截取从最后一个 [ 开始到结尾的内容
        content_after_last_bracket = text[last_bracket_idx:]

        # 3. 提取该段内的所有整数
        nums = [int(n) for n in re.findall(r"\d+", content_after_last_bracket)]

        if len(nums) >= 8:
            return nums[:8]
    except Exception:
        return None
    return None

# --- 标准误差计算函数 (支持 Masked 过滤) ---
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

def main():
    # --- 1. 配置路径与参数 ---
    # 请确保路径下的 ytrue.npy.npz 和 ypred.npy.npz 存在
    data_path = '/home/user/FSTLLM/own/data/FSTLLM-main/FSTLLM_STGNN/data/'
    json_file = "inference_results_llama31_0_mae011.json"

    target_station_idx = 0
    num_samples_per_station = 2603

    # --- 2. 加载数据 ---
    print(f"📂 正在读取 LLM 推理结果: {json_file}...")
    with open(json_file, "r", encoding="utf-8") as f:
        answer_list = json.load(f)

    print(f"📂 正在加载原始 GNN 数据...")
    y_true_all = np.load(data_path + 'ytrue.npy.npz')['arr_0']
    y_pred_sim = np.load(data_path + 'ypred.npy.npz')['arr_0']

    # 维度转换: (Time, Sample, Node) -> (Sample, Node, TimeStep)
    y_true_reshaped = y_true_all.transpose(1, 2, 0)
    y_pred_reshaped = y_pred_sim.transpose(1, 2, 0)

    llm_preds = []
    gnn_preds = []
    trues = []
    fail_count = 0

    # --- 3. 提取与对齐数据 ---
    for i in range(num_samples_per_station):
        # 提取 LLM 预测值
        pred_nums = extract_numbers(answer_list[i])
        # 获取原始 GNN 仿真值
        gnn_nums = y_pred_reshaped[i, target_station_idx, :].tolist()
        # 获取真实值
        true_nums = y_true_reshaped[i, target_station_idx, :].tolist()

        if pred_nums is not None:
            llm_preds.append(pred_nums)
        else:
            llm_preds.append(gnn_nums)  # 提取失败时回退到原始 GNN 仿真值
            fail_count += 1

        gnn_preds.append(gnn_nums)
        trues.append(true_nums)

    # 转换为 NumPy 矩阵 (Sample, 8)
    trues_np = np.array(trues).astype('float32')
    llm_np = np.array(llm_preds).astype('float32')
    gnn_np = np.array(gnn_preds).astype('float32')

    # --- 4. 分时间步计算指标 ---
    print("\n" + "=" * 90)
    print(f"{'Time Step':<10} | {'Metric':<6} | {'Original GNN':<15} | {'LLM Enhanced':<15} | {'Improvement':<12}")
    print("-" * 90)

    for step in range(8):
        # 提取第 step 个时刻的所有样本
        t_s = trues_np[:, step]
        l_s = llm_np[:, step]
        g_s = gnn_np[:, step]

        # MAE
        g_mae = masked_mae_np(g_s, t_s)
        l_mae = masked_mae_np(l_s, t_s)
        imp_mae = (g_mae - l_mae) / g_mae * 100 if g_mae > 0 else 0

        # RMSE
        g_rmse = masked_rmse_np(g_s, t_s)
        l_rmse = masked_rmse_np(l_s, t_s)
        imp_rmse = (g_rmse - l_rmse) / g_rmse * 100 if g_rmse > 0 else 0

        # MAPE (新增)
        g_mape = masked_mape_np(g_s, t_s)
        l_mape = masked_mape_np(l_s, t_s)
        imp_mape = (g_mape - l_mape) / g_mape * 100 if g_mape > 0 else 0

        # 打印当前 Step 的结果
        print(f"Step {step:<5} | MAE    | {g_mae:<15.4f} | {l_mae:<15.4f} | {imp_mae:>+10.2f}%")
        print(f"{'':<10} | RMSE   | {g_rmse:<15.4f} | {l_rmse:<15.4f} | {imp_rmse:>+10.2f}%")
        print(f"{'':<10} | MAPE   | {g_mape:<15.4f} | {l_mape:<15.4f} | {imp_mape:>+10.2f}%")
        print("-" * 90)

    # --- 5. 总体平均指标汇总 ---
    total_g_mae = masked_mae_np(gnn_np, trues_np)
    total_l_mae = masked_mae_np(llm_np, trues_np)
    total_imp_mae = (total_g_mae - total_l_mae) / total_g_mae * 100

    total_g_rmse = masked_rmse_np(gnn_np, trues_np)
    total_l_rmse = masked_rmse_np(llm_np, trues_np)
    total_imp_rmse = (total_g_rmse - total_l_rmse) / total_g_rmse * 100

    total_g_mape = masked_mape_np(gnn_np, trues_np)
    total_l_mape = masked_mape_np(llm_np, trues_np)
    total_imp_mape = (total_g_mape - total_l_mape) / total_g_mape * 100

    print(f"OVERALL    | MAE    | {total_g_mae:<15.4f} | {total_l_mae:<15.4f} | {total_imp_mae:>+10.2f}%")
    print(f"{'':<10} | RMSE   | {total_g_rmse:<15.4f} | {total_l_rmse:<15.4f} | {total_imp_rmse:>+10.2f}%")
    print(f"{'':<10} | MAPE   | {total_g_mape:<15.4f} | {total_l_mape:<15.4f} | {total_imp_mape:>+10.2f}%")
    print("=" * 90)
    print(f"💡 站点索引: {target_station_idx} | 总计样本: {num_samples_per_station} | 提取失败(回退): {fail_count}")

if __name__ == "__main__":
    main()