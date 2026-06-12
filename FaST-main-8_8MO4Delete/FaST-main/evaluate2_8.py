"""
python evaluate2_8.py
高速公路4特征大模型微调结果评估脚本（全站点版）
适配 fin_fast_vals_quick.py 微调模型
支持单站点或全站点评估
"""
import json
import re
import numpy as np
import os
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer

# 全局变量用于 build_sample_index_map
station_list_global = []
sample_counter_global = {}

# --- 工具函数：从训练数据的labels中提取真实数值 ---
def extract_numbers_from_labels(labels, tokenizer):
    """
    从训练数据的labels中提取Final Correction后的8个数值
    labels中-100表示被mask的部分，非-100的是模型需要预测的token
    """
    try:
        # 过滤掉-100的token
        valid_tokens = [t for t in labels if t != -100]
        
        if not valid_tokens:
            return None
        
        # 解码为文本
        text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
        
        # 从文本中提取最后一个方括号内的8个数字
        last_bracket_idx = text.rfind("[")
        if last_bracket_idx == -1:
            return None
        
        content_after_last_bracket = text[last_bracket_idx:]
        nums = [int(n) for n in re.findall(r"\d+", content_after_last_bracket)]
        
        if len(nums) >= 8:
            return nums[:8]
        else:
            # 调试：打印提取失败的样本
            print(f"  ⚠️ 提取到{len(nums)}个数字（不足8个）: {nums}")
            return None
    except Exception as e:
        # 打印具体错误信息以便调试
        print(f"  ⚠️ 提取失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None
    return None

def build_sample_index_map(answer_list, tokenizer, num_stations, num_samples):
    """
    构建样本索引映射表
    从quick.json中提取每条样本对应的(station_idx, feat_idx, sample_idx)
    
    Returns:
        dict: {(station_idx, feat_idx, sample_idx): answer_list_index}
    """
    global station_list_global, sample_counter_global
    
    print("\n🔍 正在构建样本索引映射表...")
    index_map = {}
    matched_count = 0
    unmatched_count = 0
    
    # 重置全局计数器
    sample_counter_global = {}
    
    for ans_idx, item in enumerate(tqdm(answer_list, desc="解析样本")):
        if not isinstance(item, dict) or 'labels' not in item:
            continue
        
        labels = item['labels']
        valid_tokens = [t for t in labels if t != -100]
        
        if not valid_tokens:
            continue
        
        # 解码前50个token用于匹配站点和特征
        text_preview = tokenizer.decode(valid_tokens[:50], skip_special_tokens=True)
        
        # 尝试从文本中提取站点名称和特征类型
        # 格式示例: "1. Station: Huangxing area. 2. Feature: Passenger Car Up; ..."
        station_match = re.search(r"Station:\s*([^.]+?)\s+area", text_preview)
        feature_match = re.search(r"Feature:\s*([^;]+?);", text_preview)
        
        if station_match and feature_match:
            station_name = station_match.group(1).strip()
            feature_name = feature_match.group(1).strip()
            
            # 映射特征名称到索引
            feat_idx_map = {
                "Passenger Car Up": 0,
                "Passenger Car Down": 1,
                "Non-Passenger Car Up": 2,
                "Non-Passenger Car Down": 3
            }
            
            if feature_name in feat_idx_map:
                feat_idx = feat_idx_map[feature_name]
                
                # 查找站点索引（模糊匹配）
                station_idx = None
                for idx, s_name in enumerate(station_list_global):
                    if station_name.lower() in s_name.lower() or s_name.lower() in station_name.lower():
                        station_idx = idx
                        break
                
                if station_idx is not None:
                    # 由于无法直接知道sample_idx，我们按顺序分配
                    # 使用计数器跟踪每个(station, feature)组合的样本数
                    key = (station_idx, feat_idx)
                    if key not in sample_counter_global:
                        sample_counter_global[key] = 0
                    
                    sample_idx = sample_counter_global[key]
                    sample_counter_global[key] += 1
                    
                    # 只记录有效的样本索引
                    if sample_idx < num_samples:
                        index_map[(station_idx, feat_idx, sample_idx)] = ans_idx
                        matched_count += 1
            else:
                unmatched_count += 1
        else:
            unmatched_count += 1
    
    print(f"✓ 索引映射构建完成:")
    print(f"   成功匹配: {matched_count} 条")
    print(f"   未匹配: {unmatched_count} 条")
    print(f"   映射表大小: {len(index_map)} 条")
    
    return index_map

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

def evaluate_single_station(target_station_idx, station_name, answer_list, true_array, pred_array, num_samples, FEATURE_NAMES, tokenizer, index_map):
    """评估单个站点的所有4个特征"""
    
    print(f"\n{'='*90}")
    print(f"🔍 正在评估站点 {target_station_idx}: {station_name}")
    print(f"{'='*90}")
    
    station_results = {}
    
    for feat_idx in range(4):
        feat_name = FEATURE_NAMES[feat_idx]
        
        llm_preds = []
        gnn_preds = []
        trues = []
        fail_count = 0
        success_count = 0
        
        # 【关键】遍历所有850个样本，检查是否在index_map中存在
        for i in range(num_samples):
            key = (target_station_idx, feat_idx, i)
            
            if key in index_map:
                # 从quick.json中提取LLM预测值
                ans_idx = index_map[key]
                item = answer_list[ans_idx]
                labels = item.get('labels', None)
                
                if labels is not None:
                    pred_nums = extract_numbers_from_labels(labels, tokenizer)
                else:
                    pred_nums = None
            else:
                # 该样本不在quick.json中，标记为缺失
                pred_nums = None
            
            gnn_nums = pred_array[i, target_station_idx, feat_idx, :].tolist()
            true_nums = true_array[i, target_station_idx, feat_idx, :].tolist()
            
            if pred_nums is not None and len(pred_nums) == 8:
                llm_preds.append(pred_nums)
                success_count += 1
            else:
                llm_preds.append(gnn_nums)  # 回退到GNN
                fail_count += 1
            
            gnn_preds.append(gnn_nums)
            trues.append(true_nums)
        
        print(f"   有效样本: {success_count}/{num_samples}, 缺失/失败: {fail_count}/{num_samples}")
        
        trues_np = np.array(trues).astype('float32')
        llm_np = np.array(llm_preds).astype('float32')
        gnn_np = np.array(gnn_preds).astype('float32')
        
        # 计算总体指标
        total_g_mae = masked_mae_np(gnn_np, trues_np)
        total_l_mae = masked_mae_np(llm_np, trues_np)
        total_imp_mae = (total_g_mae - total_l_mae) / total_g_mae * 100 if total_g_mae > 0 else 0
        
        total_g_rmse = masked_rmse_np(gnn_np, trues_np)
        total_l_rmse = masked_rmse_np(llm_np, trues_np)
        total_imp_rmse = (total_g_rmse - total_l_rmse) / total_g_rmse * 100 if total_g_rmse > 0 else 0
        
        total_g_mape = masked_mape_np(gnn_np, trues_np)
        total_l_mape = masked_mape_np(llm_np, trues_np)
        total_imp_mape = (total_g_mape - total_l_mape) / total_g_mape * 100 if total_g_mape > 0 else 0
        
        station_results[feat_idx] = {
            "name": feat_name,
            "overall": {
                "MAE": {"gnn": float(total_g_mae), "llm": float(total_l_mae), "imp": float(total_imp_mae)},
                "RMSE": {"gnn": float(total_g_rmse), "llm": float(total_l_rmse), "imp": float(total_imp_rmse)},
                "MAPE": {"gnn": float(total_g_mape), "llm": float(total_l_mape), "imp": float(total_imp_mape)}
            },
            "success_count": success_count,
            "fail_count": fail_count
        }
        
        # 打印该特征的结果
        print(f"\n特征 {feat_idx}: {feat_name}")
        print(f"  MAE:  GNN={total_g_mae:.4f}, LLM={total_l_mae:.4f}, 改进={total_imp_mae:+.2f}%")
        print(f"  RMSE: GNN={total_g_rmse:.4f}, LLM={total_l_rmse:.4f}, 改进={total_imp_rmse:+.2f}%")
        print(f"  MAPE: GNN={total_g_mape:.4f}, LLM={total_l_mape:.4f}, 改进={total_imp_mape:+.2f}%")
    
    return station_results

def main():
    # ==============================
    # 1. 解析命令行参数
    # ==============================
    parser = argparse.ArgumentParser(description='高速公路LLM微调结果评估')
    parser.add_argument('--station', type=int, default=-1, 
                       help='目标站点索引（-1表示评估所有站点，0-156评估单站点）')
    parser.add_argument('--json_file', type=str, default=None,
                       help='推理结果JSON文件路径（默认自动检测）')
    parser.add_argument('--tokenizer', type=str, default="/home/user/Llama-3.1-8B",
                       help='Tokenizer路径（用于解码labels）')
    args = parser.parse_args()
    
    # ==============================
    # 2. 配置路径与参数
    # ==============================
    PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/"
    
    # LLM推理结果文件
    if args.json_file:
        json_file = args.json_file
    else:
        json_file = os.path.join(PROJECT_ROOT, "quick.json")
    
    # GNN预测值和真实值数据
    ytrue_path = os.path.join(PROJECT_ROOT, "finetune_real_traffic.npz")
    ypred_path = os.path.join(PROJECT_ROOT, "finetune_data.npz")
    
    # 站点描述
    station_list_path = os.path.join(PROJECT_ROOT, "station_list_hngs.txt")
    
    # 配置参数
    num_samples = 850  # 样本数量
    
    # 4个特征的名称
    FEATURE_NAMES = {
        0: "Passenger Car Up",      # 小客车上行
        1: "Passenger Car Down",    # 小客车下行
        2: "Non-Passenger Car Up",  # 非小客车上行
        3: "Non-Passenger Car Down" # 非小客车下行
    }
    
    # ==============================
    # 3. 加载数据
    # ==============================
    print(f"📂 正在加载 Tokenizer: {args.tokenizer}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except Exception as e:
        print(f"❌ 错误: 无法加载 Tokenizer: {e}")
        return

    print(f"📂 正在读取 LLM 推理结果: {json_file}...")
    if not os.path.exists(json_file):
        print(f"❌ 错误: 推理结果文件不存在: {json_file}")
        print(f"💡 请先运行推理脚本生成结果")
        return
    
    with open(json_file, "r", encoding="utf-8") as f:
        answer_list = json.load(f)
    
    print(f"📂 正在加载 GNN 预测值和真实值...")
    npz_true = np.load(ytrue_path)
    npz_pred = np.load(ypred_path)
    
    if 'target' in npz_true:
        true_array = npz_true['target']
    elif 'arr_0' in npz_true:
        true_array = npz_true['arr_0']
    else:
        raise KeyError(f"{ytrue_path} 未找到有效键")
    
    if 'prediction' in npz_pred:
        pred_array = npz_pred['prediction']
    elif 'arr_0' in npz_pred:
        pred_array = npz_pred['arr_0']
    else:
        raise KeyError(f"{ypred_path} 未找到有效键")
    
    print(f"✓ 数据形状:")
    print(f"  true_array: {true_array.shape}")
    print(f"  pred_array: {pred_array.shape}")
    print(f"  推理结果条目数: {len(answer_list)}")
    
    # 【关键】先加载站点列表，获取站点数量
    with open(station_list_path, "r", encoding='utf-8') as f:
        station_list = [l.strip() for l in f if l.strip()]
    
    num_stations = len(station_list)
    print(f"✓ 站点总数: {num_stations}")
    
    # ==============================
    # 【关键修复】自动维度检测与转置
    # ==============================
    # 期望格式: (Sample, Station, Feature, TimeStep) = (850, 157, 4, 8)
    # 实际可能格式:
    #   1. (850, 8, 157, 4) - 需要转置为 (850, 157, 4, 8)
    #   2. (850, 157, 4, 8) - 正确格式，无需转置
    
    if true_array.ndim == 4:
        s, d1, d2, d3 = true_array.shape
        
        # 检测维度顺序
        if d1 == 8 and d2 == num_stations and d3 == 4:
            # 格式: (Sample, TimeStep, Station, Feature) -> 需要转置
            print(f"\n⚠️ 检测到数据维度需要转置:")
            print(f"   原始形状: {true_array.shape}")
            print(f"   转置为: (Sample, Station, Feature, TimeStep)")
            
            # 转置: (S, T, N, F) -> (S, N, F, T)
            true_array = true_array.transpose(0, 2, 3, 1)
            pred_array = pred_array.transpose(0, 2, 3, 1)
            
            print(f"   转置后形状: {true_array.shape}")
        elif d1 == num_stations and d2 == 4 and d3 == 8:
            # 格式: (Sample, Station, Feature, TimeStep) - 正确
            print(f"\n✓ 数据维度正确: (Sample, Station, Feature, TimeStep)")
        else:
            raise ValueError(f"❌ 未知的数据维度格式: {true_array.shape}")
    
    print(f"\n✓ 最终数据形状:")
    print(f"  true_array: {true_array.shape}")
    print(f"  pred_array: {pred_array.shape}")
    
    # ==============================
    # 3.5. 构建智能样本索引映射
    # ==============================
    # 设置全局站点列表供 build_sample_index_map 使用
    station_list_global = station_list
    
    index_map = build_sample_index_map(answer_list, tokenizer, num_stations, num_samples)
    
    # ==============================
    # 4. 确定评估范围
    # ==============================
    if args.station == -1:
        # 评估所有站点
        stations_to_eval = list(range(num_stations))
        eval_mode = "ALL_STATIONS"
        print(f"\n🚀 评估模式: 全站点评估 ({num_stations} 个站点)")
    else:
        # 评估单站点
        if args.station < 0 or args.station >= num_stations:
            print(f"❌ 错误: 站点索引 {args.station} 超出范围 (0-{num_stations-1})")
            return
        stations_to_eval = [args.station]
        eval_mode = "SINGLE_STATION"
        print(f"\n🚀 评估模式: 单站点评估 (站点 {args.station})")
    
    # ==============================
    # 5. 执行评估
    # ==============================
    all_results = {}
    
    for station_idx in tqdm(stations_to_eval, desc="评估站点进度"):
        station_name = station_list[station_idx] if station_idx < len(station_list) else f"Station_{station_idx}"
        
        try:
            station_results = evaluate_single_station(
                station_idx, station_name, answer_list, 
                true_array, pred_array, num_samples, FEATURE_NAMES, tokenizer, index_map
            )
            all_results[station_idx] = {
                "name": station_name,
                "features": station_results
            }
        except Exception as e:
            print(f"\n⚠️ 站点 {station_idx} 评估失败: {e}")
            continue
    
    # ==============================
    # 6. 汇总统计
    # ==============================
    print(f"\n\n{'='*120}")
    print(f"📊 评估结果汇总 ({eval_mode})")
    print(f"{'='*120}")
    
    if eval_mode == "ALL_STATIONS":
        # 计算所有站点的平均指标
        print(f"\n{'站点':<10} | {'特征':<30} | {'Metric':<6} | {'GNN':<12} | {'LLM':<12} | {'改进':<12}")
        print("-" * 120)
        
        # 按特征汇总所有站点的平均值
        for feat_idx in range(4):
            feat_name = FEATURE_NAMES[feat_idx]
            
            all_gnn_mae = []
            all_llm_mae = []
            all_gnn_rmse = []
            all_llm_rmse = []
            all_gnn_mape = []
            all_llm_mape = []
            
            for station_idx, station_data in all_results.items():
                if feat_idx in station_data["features"]:
                    metrics = station_data["features"][feat_idx]["overall"]
                    all_gnn_mae.append(metrics["MAE"]["gnn"])
                    all_llm_mae.append(metrics["MAE"]["llm"])
                    all_gnn_rmse.append(metrics["RMSE"]["gnn"])
                    all_llm_rmse.append(metrics["RMSE"]["llm"])
                    all_gnn_mape.append(metrics["MAPE"]["gnn"])
                    all_llm_mape.append(metrics["MAPE"]["llm"])
            
            if all_gnn_mae:
                avg_gnn_mae = np.mean(all_gnn_mae)
                avg_llm_mae = np.mean(all_llm_mae)
                avg_imp_mae = (avg_gnn_mae - avg_llm_mae) / avg_gnn_mae * 100
                
                avg_gnn_rmse = np.mean(all_gnn_rmse)
                avg_llm_rmse = np.mean(all_llm_rmse)
                avg_imp_rmse = (avg_gnn_rmse - avg_llm_rmse) / avg_gnn_rmse * 100
                
                avg_gnn_mape = np.mean(all_gnn_mape)
                avg_llm_mape = np.mean(all_llm_mape)
                avg_imp_mape = (avg_gnn_mape - avg_llm_mape) / avg_gnn_mape * 100
                
                print(f"{'AVG':<10} | {feat_name:<30} | MAE    | {avg_gnn_mae:<12.4f} | {avg_llm_mae:<12.4f} | {avg_imp_mae:>+10.2f}%")
                print(f"{'':<10} | {'':<30} | RMSE   | {avg_gnn_rmse:<12.4f} | {avg_llm_rmse:<12.4f} | {avg_imp_rmse:>+10.2f}%")
                print(f"{'':<10} | {'':<30} | MAPE   | {avg_gnn_mape:<12.4f} | {avg_llm_mape:<12.4f} | {avg_imp_mape:>+10.2f}%")
        
        print(f"{'='*120}")
        
        # 找出表现最好和最差的站点
        print(f"\n🏆 Top 5 最佳改进站点 (按MAE改进率):")
        station_improvements = []
        for station_idx, station_data in all_results.items():
            improvements = []
            for feat_idx in range(4):
                if feat_idx in station_data["features"]:
                    imp = station_data["features"][feat_idx]["overall"]["MAE"]["imp"]
                    improvements.append(imp)
            if improvements:
                avg_imp = np.mean(improvements)
                station_improvements.append((station_idx, station_data["name"], avg_imp))
        
        station_improvements.sort(key=lambda x: x[2], reverse=True)
        for rank, (idx, name, imp) in enumerate(station_improvements[:5], 1):
            print(f"  {rank}. 站点 {idx:3d} ({name:<20}): 平均改进 {imp:+.2f}%")
        
        print(f"\n📉 Top 5 需改进站点 (按MAE改进率):")
        for rank, (idx, name, imp) in enumerate(station_improvements[-5:], 1):
            print(f"  {rank}. 站点 {idx:3d} ({name:<20}): 平均改进 {imp:+.2f}%")
    
    else:
        # 单站点详细输出
        station_idx = stations_to_eval[0]
        station_data = all_results[station_idx]
        print(f"\n站点: {station_data['name']}")
        print(f"{'特征':<30} | {'Metric':<6} | {'GNN':<12} | {'LLM':<12} | {'改进':<12}")
        print("-" * 90)
        
        for feat_idx in range(4):
            if feat_idx in station_data["features"]:
                feat_name = station_data["features"][feat_idx]["name"]
                metrics = station_data["features"][feat_idx]["overall"]
                
                for metric_name in ["MAE", "RMSE", "MAPE"]:
                    m = metrics[metric_name]
                    print(f"{feat_name:<30} | {metric_name:<6} | {m['gnn']:<12.4f} | {m['llm']:<12.4f} | {m['imp']:>+10.2f}%")
    
    # ==============================
    # 7. 保存评估结果
    # ==============================
    output_path = os.path.join(PROJECT_ROOT, f"evaluation_results_{eval_mode.lower()}.json")
    eval_results = {
        "eval_mode": eval_mode,
        "num_stations_evaluated": len(all_results),
        "num_samples_per_station": num_samples,
        "results": all_results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 评估结果已保存至: {output_path}")
    print(f"💡 提示: 使用 --station <idx> 参数可评估单个站点")

if __name__ == "__main__":
    main()