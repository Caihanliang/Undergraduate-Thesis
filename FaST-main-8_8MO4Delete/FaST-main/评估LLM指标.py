import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import time
import threading
from datetime import datetime
from unsloth import FastLanguageModel
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =================强制离线模式=================
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
os.environ["UNSLOTH_SKIP_INIT_CHECK"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =================配置区域=================
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main"
MODEL_PATH = os.path.join(PROJECT_ROOT, "config/cai/llama-3-1-8b-highway-optimized") 
DATA_PATH = os.path.join(PROJECT_ROOT, "config/cai/finetune_data.npz")
TRUE_PATH = os.path.join(PROJECT_ROOT, "config/cai/finetune_real_traffic.npz")
STATION_LIST = os.path.join(PROJECT_ROOT, "config/cai/station_list_hngs.txt")
WEATHER_DATA_PATH = os.path.join(PROJECT_ROOT, "config/cai/160站点天气信息/")
EVENTS_CSV_PATH = os.path.join(PROJECT_ROOT, "config/cai/events_list_quan.csv")

MAX_SEQ_LENGTH = 1024
DTYPE = torch.float16
LOAD_IN_4BIT = True

# 评估配置 - 评估所有站点
MAX_SAMPLES = 20        # 每个站点测试的样本数
MAX_STATIONS = None     # None表示所有站点（157个）
MAX_FEATURES = 4        # 测试所有4个特征
TIMEOUT_SECONDS = 15    # 生成超时（秒）
# =========================================

def calculate_metrics(predictions, targets):
    """计算MAE, RMSE, MAPE"""
    pred_flat = np.array(predictions).flatten()
    true_flat = np.array(targets).flatten()
    
    mask = (~np.isnan(true_flat)) & (~np.isnan(pred_flat)) & (true_flat != 0)
    if not np.any(mask):
        return 0, 0, 0
    
    mae = np.mean(np.abs(pred_flat[mask] - true_flat[mask]))
    rmse = np.sqrt(np.mean((pred_flat[mask] - true_flat[mask])**2))
    
    valid_true = true_flat[mask]
    valid_pred = pred_flat[mask]
    mape = np.mean(np.abs((valid_true - valid_pred) / (valid_true + 1e-5))) * 100
    mape = min(mape, 1000)
    
    return mae, rmse, mape

def load_station_list():
    """加载站点列表"""
    with open(STATION_LIST, "r", encoding='utf-8') as f:
        stations = [line.strip() for line in f if line.strip()]
    station_names = [s.split(' ')[0] for s in stations]
    return station_names

def load_weather_data():
    """预加载天气数据"""
    weather_cache = {}
    if not os.path.exists(WEATHER_DATA_PATH):
        print("⚠️ 天气目录不存在，使用默认天气")
        return weather_cache
    
    files = os.listdir(WEATHER_DATA_PATH)
    station_names = load_station_list()
    for j in tqdm(range(len(station_names)), desc="加载天气数据"):
        try:
            prefix = f"{j:03d}"
            match = [f for f in files if f.startswith(prefix)]
            if not match:
                continue
            path = os.path.join(WEATHER_DATA_PATH, match[0])
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
            weather_cache[j] = df[w_col].to_dict()
        except:
            continue
    print(f"✅ 天气数据加载完成：{len(weather_cache)} 个站点")
    return weather_cache

def load_events_data():
    """加载事件数据"""
    if not os.path.exists(EVENTS_CSV_PATH):
        print("⚠️ 事件文件不存在")
        return {}
    try:
        df_ev = pd.read_csv(EVENTS_CSV_PATH, encoding='utf-8')
        date_col = "日期"
        station_col = "站点名称"
        event_col = "事件描述"
        
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
        print(f"✅ 事件数据加载完成：{len(event_map)} 条")
        return event_map
    except Exception as e:
        print(f"⚠️ 事件加载失败: {e}")
        return {}

def get_weather(weather_cache, station_idx, date_str):
    """获取天气"""
    return weather_cache.get(station_idx, {}).get(date_str, "Clear")

def get_event(events_map, station_name, date_str):
    """获取事件"""
    return events_map.get((date_str, station_name), "None")

def get_feature_name(feature_idx):
    """获取特征名称"""
    feature_names = {
        0: "Passenger Car Up",
        1: "Passenger Car Down",
        2: "Non-Passenger Car Up",
        3: "Non-Passenger Car Down"
    }
    return feature_names.get(feature_idx, f"Feature_{feature_idx}")

def build_instruction(station_name, feature_name, time_info, day_type, weather, event, gnn_pred):
    """构建完整的指令"""
    if isinstance(gnn_pred, list):
        pred_str = ', '.join([f"{x:.0f}" for x in gnn_pred[:8]])
    else:
        pred_str = str(gnn_pred)
    
    instruction = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Refine traffic prediction:\n"
        f"Station: {station_name}\n"
        f"Feature: {feature_name}\n"
        f"Time: {time_info}\n"
        f"GNN: [{pred_str}]\n"
        f"Output corrected list.<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"Final Correction: ["
    )
    return instruction

def extract_numbers_from_response(response, instruction):
    """从模型响应中提取数字序列"""
    if not response:
        return None
    
    full_text = instruction + response
    
    match = re.search(r'Final Correction:\s*\[(.*?)\]', full_text, re.IGNORECASE | re.DOTALL)
    if match:
        nums = re.findall(r'\d+\.?\d*', match.group(1))
        if len(nums) >= 1:
            return [float(x) for x in nums[:8]]
    
    match = re.search(r'\[(.*?)\]', full_text)
    if match:
        nums = re.findall(r'\d+\.?\d*', match.group(1))
        if len(nums) >= 1:
            return [float(x) for x in nums[:8]]
    
    all_nums = re.findall(r'\d+\.?\d*', response)
    if len(all_nums) >= 8:
        return [float(x) for x in all_nums[:8]]
    
    return None

def generate_with_timeout(model, tokenizer, inputs, max_new_tokens=128, timeout=TIMEOUT_SECONDS):
    """带超时的生成函数"""
    result = [None]
    error = [None]
    
    def generate():
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )
            result[0] = outputs
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=generate)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        return None, "timeout"
    if error[0]:
        return None, str(error[0])
    return result[0], None

def load_data():
    """加载所有数据"""
    print("加载数据集...")
    
    pred_npz = np.load(DATA_PATH)
    true_npz = np.load(TRUE_PATH)
    
    pred_key = 'prediction' if 'prediction' in pred_npz else 'arr_0'
    true_key = 'target' if 'target' in true_npz else 'arr_0'
    
    pred_array = pred_npz[pred_key]
    true_array = true_npz[true_key]
    
    station_names = load_station_list()
    
    if len(pred_array.shape) == 4:
        num_samples = pred_array.shape[0]
        num_steps = pred_array.shape[1]
        num_stations = pred_array.shape[2]
        num_features = pred_array.shape[3]
    elif len(pred_array.shape) == 3:
        num_stations = len(station_names)
        num_samples = pred_array.shape[0] // num_stations
        num_steps = pred_array.shape[1]
        num_features = pred_array.shape[2]
        pred_array = pred_array.reshape(num_samples, num_steps, num_stations, num_features)
        true_array = true_array.reshape(num_samples, num_steps, num_stations, num_features)
    else:
        raise ValueError(f"未知数据形状: {pred_array.shape}")
    
    print(f"数据形状: 样本数={num_samples}, 步长={num_steps}, 站点数={num_stations}, 特征数={num_features}")
    return pred_array, true_array, station_names, num_samples, num_steps, num_stations, num_features

def evaluate_all():
    """主评估函数 - 评估所有站点"""
    print("="*80)
    print("高速公路流量预测LLM修正评估系统 - 全站点评估")
    print("="*80)
    
    # 加载数据
    pred_array, true_array, station_names, num_samples, num_steps, num_stations, num_features = load_data()
    
    # 加载辅助数据
    print("\n加载辅助数据...")
    weather_cache = load_weather_data()
    events_map = load_events_data()
    
    # 确定测试范围 - 所有站点
    test_samples = min(MAX_SAMPLES, num_samples)
    test_stations = num_stations  # 所有157个站点
    test_features = list(range(min(MAX_FEATURES, num_features)))
    
    print(f"\n📊 评估配置:")
    print(f"  总站点数: {num_stations}")
    print(f"  测试站点数: {test_stations} (全部)")
    print(f"  每个站点测试样本数: {test_samples}")
    print(f"  特征数: {len(test_features)}")
    print(f"  总评估样本数: {test_stations * test_samples * len(test_features)}")
    print(f"  超时设置: {TIMEOUT_SECONDS}秒")
    
    # 加载模型
    print("\n🔧 加载模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        local_files_only=True,
    )
    FastLanguageModel.for_inference(model)
    print("✅ 模型加载完成")
    
    # 存储所有预测值（按特征分组）
    all_gnn_preds = {0: [], 1: [], 2: [], 3: []}
    all_llm_preds = {0: [], 1: [], 2: [], 3: []}
    all_true_vals = {0: [], 1: [], 2: [], 3: []}
    
    # 存储每个站点的详细结果
    station_results = []
    
    print(f"\n🚀 开始评估 {test_stations} 个站点，每个站点 {test_samples} 个样本...")
    print("预计需要较长时间，请耐心等待...")
    print("-"*80)
    
    for station_idx in tqdm(range(test_stations), desc="评估站点"):
        station_name = station_names[station_idx] if station_idx < len(station_names) else f"Station_{station_idx}"
        
        for sample_idx in range(test_samples):
            # 使用不同的日期来增加多样性
            date_strs = ["2023-09-01", "2023-09-02", "2023-09-03", "2023-09-04", "2023-09-05"]
            date_str = date_strs[sample_idx % len(date_strs)]
            day_type = "Workday" if sample_idx % 5 < 3 else "Weekend"
            weather = get_weather(weather_cache, station_idx, date_str)
            event = get_event(events_map, station_name, date_str)
            time_info = f"{date_str} 08:00"
            
            for feat_idx in test_features:
                try:
                    # 获取GNN预测序列
                    pred_sequence = []
                    for step in range(min(num_steps, 8)):
                        val = pred_array[sample_idx, step, station_idx, feat_idx]
                        pred_sequence.append(float(val) if hasattr(val, 'item') else float(val))
                    
                    # 真实值
                    true_val = true_array[sample_idx, 0, station_idx, feat_idx]
                    true_val = float(true_val) if hasattr(true_val, 'item') else float(true_val)
                    gnn_val = pred_sequence[0]
                    
                    feature_name = get_feature_name(feat_idx)
                    
                    # 构建指令并推理
                    instruction = build_instruction(
                        station_name, feature_name, time_info, 
                        day_type, weather, event, pred_sequence
                    )
                    
                    inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=512).to("cuda")
                    outputs, error = generate_with_timeout(model, tokenizer, inputs, max_new_tokens=128)
                    
                    if outputs is None:
                        llm_val = gnn_val
                    else:
                        input_len = inputs.input_ids.shape[1]
                        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                        extracted = extract_numbers_from_response(response, instruction)
                        
                        if extracted and len(extracted) > 0:
                            llm_val = extracted[0]
                        else:
                            llm_val = gnn_val
                    
                    # 存储数据
                    all_gnn_preds[feat_idx].append(gnn_val)
                    all_llm_preds[feat_idx].append(llm_val)
                    all_true_vals[feat_idx].append(true_val)
                    
                except Exception as e:
                    continue
        
        # 每10个站点打印一次进度
        if (station_idx + 1) % 10 == 0:
            print(f"  已完成 {station_idx + 1}/{test_stations} 个站点...")
    
    # ========== 输出表格 ==========
    print("\n" + "="*80)
    print("📈 各特征预测性能评估（MAE / RMSE / MAPE）")
    print("="*80)
    print(f"{'特征':<25} {'GNN_MAE':<12} {'LLM_MAE':<12} {'GNN_RMSE':<12} {'LLM_RMSE':<12} {'GNN_MAPE':<12} {'LLM_MAPE':<12}")
    print("-"*100)
    
    for feat_idx in range(4):
        if len(all_true_vals[feat_idx]) == 0:
            continue
        
        feature_name = get_feature_name(feat_idx)
        
        # 计算指标
        gnn_mae, gnn_rmse, gnn_mape = calculate_metrics(all_gnn_preds[feat_idx], all_true_vals[feat_idx])
        llm_mae, llm_rmse, llm_mape = calculate_metrics(all_llm_preds[feat_idx], all_true_vals[feat_idx])
        
        # 计算改进率
        mae_improve = ((gnn_mae - llm_mae) / (gnn_mae + 1e-6)) * 100
        rmse_improve = ((gnn_rmse - llm_rmse) / (gnn_rmse + 1e-6)) * 100
        
        print(f"{feature_name:<25} {gnn_mae:<12.2f} {llm_mae:<12.2f} "
              f"{gnn_rmse:<12.2f} {llm_rmse:<12.2f} "
              f"{gnn_mape:<12.2f}% {llm_mape:<12.2f}%")
    
    print("="*80)
    
    # ========== 输出改进率汇总 ==========
    print("\n📊 改进率汇总:")
    print("-"*60)
    for feat_idx in range(4):
        if len(all_true_vals[feat_idx]) == 0:
            continue
        feature_name = get_feature_name(feat_idx)
        gnn_mae, _, _ = calculate_metrics(all_gnn_preds[feat_idx], all_true_vals[feat_idx])
        llm_mae, _, _ = calculate_metrics(all_llm_preds[feat_idx], all_true_vals[feat_idx])
        mae_improve = ((gnn_mae - llm_mae) / (gnn_mae + 1e-6)) * 100
        print(f"  {feature_name}: MAE改进 = {mae_improve:+.2f}%")
    
    # ========== 总体统计 ==========
    print("\n" + "="*60)
    print("📊 总体统计（所有特征合并）")
    print("="*60)
    
    # 合并所有特征的数据
    all_gnn = []
    all_llm = []
    all_true = []
    for feat_idx in range(4):
        all_gnn.extend(all_gnn_preds[feat_idx])
        all_llm.extend(all_llm_preds[feat_idx])
        all_true.extend(all_true_vals[feat_idx])
    
    total_gnn_mae, total_gnn_rmse, total_gnn_mape = calculate_metrics(all_gnn, all_true)
    total_llm_mae, total_llm_rmse, total_llm_mape = calculate_metrics(all_llm, all_true)
    
    print(f"\n{'指标':<15} {'GNN':<15} {'LLM':<15} {'改进率':<15}")
    print("-"*60)
    print(f"{'MAE':<15} {total_gnn_mae:<15.2f} {total_llm_mae:<15.2f} {((total_gnn_mae - total_llm_mae)/(total_gnn_mae+1e-6))*100:>+14.2f}%")
    print(f"{'RMSE':<15} {total_gnn_rmse:<15.2f} {total_llm_rmse:<15.2f} {((total_gnn_rmse - total_llm_rmse)/(total_gnn_rmse+1e-6))*100:>+14.2f}%")
    print(f"{'MAPE(%)':<15} {total_gnn_mape:<15.2f} {total_llm_mape:<15.2f} {((total_gnn_mape - total_llm_mape)/(total_gnn_mape+1e-6))*100:>+14.2f}%")
    
    # ========== 保存结果 ==========
    output_dir = os.path.join(PROJECT_ROOT, "config/cai/evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存各特征详细结果
    results_data = []
    for feat_idx in range(4):
        feature_name = get_feature_name(feat_idx)
        for i in range(len(all_true_vals[feat_idx])):
            results_data.append({
                'feature': feature_name,
                'gnn_pred': all_gnn_preds[feat_idx][i],
                'llm_pred': all_llm_preds[feat_idx][i],
                'true_val': all_true_vals[feat_idx][i]
            })
    
    results_df = pd.DataFrame(results_data)
    results_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 详细结果已保存: {results_path}")
    
    # 保存指标汇总
    metrics_data = []
    for feat_idx in range(4):
        if len(all_true_vals[feat_idx]) == 0:
            continue
        feature_name = get_feature_name(feat_idx)
        gnn_mae, gnn_rmse, gnn_mape = calculate_metrics(all_gnn_preds[feat_idx], all_true_vals[feat_idx])
        llm_mae, llm_rmse, llm_mape = calculate_metrics(all_llm_preds[feat_idx], all_true_vals[feat_idx])
        metrics_data.append({
            'feature': feature_name,
            'gnn_mae': gnn_mae, 'llm_mae': llm_mae, 'mae_improvement': ((gnn_mae - llm_mae)/(gnn_mae+1e-6))*100,
            'gnn_rmse': gnn_rmse, 'llm_rmse': llm_rmse, 'rmse_improvement': ((gnn_rmse - llm_rmse)/(gnn_rmse+1e-6))*100,
            'gnn_mape': gnn_mape, 'llm_mape': llm_mape, 'mape_improvement': ((gnn_mape - llm_mape)/(gnn_mape+1e-6))*100,
            'sample_count': len(all_true_vals[feat_idx])
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(output_dir, f"evaluation_metrics_{timestamp}.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f"✅ 指标汇总已保存: {metrics_path}")
    
    print("\n✅ 评估完成！")

if __name__ == "__main__":
    evaluate_all()
"""
python 评估LLM指标.py
"""
