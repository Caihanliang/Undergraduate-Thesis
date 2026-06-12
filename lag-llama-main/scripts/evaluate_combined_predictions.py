"""
计算Lag-Llama四个特征的评估指标

功能:
1. 直接从prediction_results目录读取4个独立的CSV文件
2. 计算每个特征的MAE、RMSE、MAPE、sMAPE
3. 输出CSV和JSON格式的评估结果
4. 包含全局平均行
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# 配置
RESULTS_DIR = Path('/home/user/Downloads/cai/lag-llama-main/prediction_results')
OUTPUT_CSV = RESULTS_DIR / 'evaluation_metrics.csv'
OUTPUT_JSON = RESULTS_DIR / 'evaluation_metrics.json'

# 特征文件映射 (文件名: 特征显示名称)
FEATURE_FILES = {
    'passenger_up_predictions.csv': '小客车上行 (Passenger Car Up)',
    'passenger_down_predictions.csv': '小客车下行 (Passenger Car Down)',
    'non_passenger_up_predictions.csv': '非小客车上行 (Non-Passenger Car Up)',
    'non_passenger_down_predictions.csv': '非小客车下行 (Non-Passenger Car Down)'
}

# MAPE计算时过滤阈值(真实值小于此值的样本不参与MAPE计算)
MAPE_THRESHOLD = 1.0

def calculate_mae(true, pred):
    """计算平均绝对误差"""
    return np.mean(np.abs(true - pred))

def calculate_rmse(true, pred):
    """计算均方根误差"""
    return np.sqrt(np.mean((true - pred) ** 2))

def calculate_mape(true, pred, threshold=MAPE_THRESHOLD):
    """
    计算平均绝对百分比误差
    过滤真实值小于threshold的样本,避免除零错误
    """
    # 过滤条件: 真实值 >= threshold
    mask = true >= threshold
    if mask.sum() == 0:
        return np.nan
    
    true_filtered = true[mask]
    pred_filtered = pred[mask]
    
    mape = np.mean(np.abs((true_filtered - pred_filtered) / true_filtered)) * 100
    return mape

def calculate_smape(true, pred):
    """
    计算对称平均绝对百分比误差
    更稳健,适合处理低值数据
    """
    denominator = (np.abs(true) + np.abs(pred)) / 2
    # 避免除零
    mask = denominator > 1e-8
    if mask.sum() == 0:
        return np.nan
    
    smape = np.mean(np.abs(true[mask] - pred[mask]) / denominator[mask]) * 100
    return smape

def evaluate_feature(true_values, pred_values, feature_name):
    """计算单个特征的所有评估指标"""
    # 转换为numpy数组
    true = np.array(true_values, dtype=float)
    pred = np.array(pred_values, dtype=float)
    
    # 过滤NaN值
    valid_mask = ~(np.isnan(true) | np.isnan(pred))
    true = true[valid_mask]
    pred = pred[valid_mask]
    
    if len(true) == 0:
        return {
            'feature': feature_name,
            'MAE': np.nan,
            'RMSE': np.nan,
            'MAPE': np.nan,
            'sMAPE': np.nan,
            '样本数': 0
        }
    
    metrics = {
        'feature': feature_name,
        'MAE': calculate_mae(true, pred),
        'RMSE': calculate_rmse(true, pred),
        'MAPE': calculate_mape(true, pred),
        'sMAPE': calculate_smape(true, pred),
        '样本数': len(true)
    }
    
    return metrics

print("=" * 80)
print("Lag-Llama 预测结果评估")
print("=" * 80)

# 1. 读取4个特征文件并计算指标
print(f"\n正在从目录读取数据: {RESULTS_DIR}")
results = []

for filename, feature_name in FEATURE_FILES.items():
    filepath = RESULTS_DIR / filename
    
    if not filepath.exists():
        print(f"  ⚠️ 文件不存在: {filename},跳过")
        continue
    
    print(f"\n正在处理: {filename}")
    print(f"  特征名称: {feature_name}")
    
    # 读取CSV文件
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    print(f"  总行数: {len(df)}")
    print(f"  站点数: {df['station_id'].nunique()}")
    print(f"  时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    
    # 获取真实值和预测值
    true_values = df['true_value'].values
    pred_values = df['pred_value'].values
    
    # 计算指标
    metrics = evaluate_feature(true_values, pred_values, feature_name)
    results.append(metrics)
    
    print(f"    MAE:  {metrics['MAE']:.4f}")
    print(f"    RMSE: {metrics['RMSE']:.4f}")
    print(f"    MAPE: {metrics['MAPE']:.2f}% (过滤真实值<{MAPE_THRESHOLD}的样本)")
    print(f"    sMAPE:{metrics['sMAPE']:.2f}%")
    print(f"    样本数: {metrics['样本数']}")

if len(results) == 0:
    print("\n❌ 错误: 没有找到任何有效的预测结果文件!")
    exit(1)

# 2. 计算全局平均
print("\n正在计算全局平均指标...")
global_metrics = {
    'feature': '全局平均 (Global Average)',
    'MAE': np.mean([r['MAE'] for r in results]),
    'RMSE': np.mean([r['RMSE'] for r in results]),
    'MAPE': np.mean([r['MAPE'] for r in results if not np.isnan(r['MAPE'])]),
    'sMAPE': np.mean([r['sMAPE'] for r in results if not np.isnan(r['sMAPE'])]),
    '样本数': sum([r['样本数'] for r in results])
}
results.append(global_metrics)

print(f"  MAE:  {global_metrics['MAE']:.4f}")
print(f"  RMSE: {global_metrics['RMSE']:.4f}")
print(f"  MAPE: {global_metrics['MAPE']:.2f}%")
print(f"  sMAPE:{global_metrics['sMAPE']:.2f}%")

# 3. 保存CSV结果
print(f"\n正在保存CSV结果: {OUTPUT_CSV}")
results_df = pd.DataFrame(results)
# 重命名列以符合规范
results_df = results_df.rename(columns={
    'feature': '特征名称',
    '样本数': '样本数'
})
results_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"  ✓ CSV保存成功")

# 4. 保存JSON结果
print(f"\n正在保存JSON结果: {OUTPUT_JSON}")
# 添加时间戳和元数据
output_data = {
    'timestamp': datetime.now().isoformat(),
    'description': 'Lag-Llama高速公路交通流量预测评估结果',
    'metrics': []
}

for r in results:
    output_data['metrics'].append({
        'feature': r['feature'],
        'MAE': round(float(r['MAE']), 4) if not np.isnan(r['MAE']) else None,
        'RMSE': round(float(r['RMSE']), 4) if not np.isnan(r['RMSE']) else None,
        'MAPE': round(float(r['MAPE']), 2) if not np.isnan(r['MAPE']) else None,
        'sMAPE': round(float(r['sMAPE']), 2) if not np.isnan(r['sMAPE']) else None,
        'sample_count': r['样本数']
    })

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
print(f"  ✓ JSON保存成功")

# 5. 打印格式化结果
print("\n" + "=" * 80)
print("评估结果汇总")
print("=" * 80)
print(f"{'特征':<35} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'sMAPE':>10} {'样本数':>10}")
print("-" * 80)

for r in results:
    mape_str = f"{r['MAPE']:.2f}%" if not np.isnan(r['MAPE']) else "N/A"
    smape_str = f"{r['sMAPE']:.2f}%" if not np.isnan(r['sMAPE']) else "N/A"
    
    print(f"{r['feature']:<35} {r['MAE']:>10.4f} {r['RMSE']:>10.4f} {mape_str:>10} {smape_str:>10} {r['样本数']:>10}")

print("=" * 80)
print("\n✓ 评估完成!")
print(f"\n结果已保存至:")
print(f"  CSV: {OUTPUT_CSV}")
print(f"  JSON: {OUTPUT_JSON}")
print("\n注意事项:")
print("  1. MAPE已过滤真实值<1的样本,避免除零错误")
print("  2. sMAPE是更稳健的指标,推荐作为主要评估依据")
print("  3. 如数据中存在大量零值,请优先参考sMAPE而非MAPE")