"""
GMAN预测结果指标计算脚本
功能: 从CSV文件中读取预测结果，计算每个特征的MAE、RMSE、MAPE指标
python calculate_metrics.py
"""

import os
import pandas as pd
import numpy as np

# ==================== 配置参数 ====================
RESULTS_DIR = '/home/user/Downloads/cai/GMAN/models/highway_4feat'
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'metrics_summary.csv')

# 特征列名映射（根据实际CSV文件的列名）
FEATURE_COLUMNS = {
    '小客车上行': ('小客车上行_预测', '小客车上行_真实'),
    '小客车下行': ('小客车下行_预测', '小客车下行_真实'),
    '非小客车上行': ('非小客车上行_预测', '非小客车上行_真实'),
    '非小客车下行': ('非小客车下行_预测', '非小客车下行_真实')
}


def calculate_metrics(y_true, y_pred):
    """
    计算预测指标
    
    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
    
    Returns:
        dict: 包含MAE、RMSE、MAPE的字典
    """
    # 去除无效值（NaN或Inf）
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    
    if len(y_true_valid) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, '样本数': 0}
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_true_valid - y_pred_valid))
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((y_true_valid - y_pred_valid) ** 2))
    
    # MAPE (Mean Absolute Percentage Error)
    # 关键修复：只考虑真实值 >= 阈值的样本，避免除以0或极小值
    THRESHOLD = 10  # 只计算车流量>=10的样本的MAPE
    mask_valid_mape = y_true_valid >= THRESHOLD
    
    if np.sum(mask_valid_mape) > 0:
        y_true_for_mape = y_true_valid[mask_valid_mape]
        y_pred_for_mape = y_pred_valid[mask_valid_mape]
        mape = np.mean(np.abs((y_true_for_mape - y_pred_for_mape) / y_true_for_mape)) * 100
        
        # 统计信息
        total_samples = len(y_true_valid)
        valid_mape_samples = np.sum(mask_valid_mape)
        filtered_samples = total_samples - valid_mape_samples
    else:
        mape = np.nan
        valid_mape_samples = 0
        filtered_samples = len(y_true_valid)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        '样本数': len(y_true_valid),
        'MAPE有效样本数': valid_mape_samples,
        'MAPE过滤样本数': filtered_samples,
        'MAPE阈值': THRESHOLD
    }


def process_csv_file(csv_path, dataset_name):
    """
    处理单个CSV文件，计算各特征指标
    
    Args:
        csv_path: CSV文件路径
        dataset_name: 数据集名称 (train/val/test)
    
    Returns:
        DataFrame: 包含各特征指标的DataFrame
    """
    print(f"\n📊 处理 {dataset_name} 集...")
    print(f"   文件路径: {csv_path}")
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    print(f"   数据形状: {df.shape}")
    print(f"   列名: {list(df.columns)}")
    
    # 检查必需列是否存在
    required_cols = []
    for pred_col, true_col in FEATURE_COLUMNS.values():
        required_cols.extend([pred_col, true_col])
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"   ❌ 缺少列: {missing_cols}")
        return None
    
    # 计算每个特征的指标
    results = []
    for feat_name, (pred_col, true_col) in FEATURE_COLUMNS.items():
        y_pred = df[pred_col].values
        y_true = df[true_col].values
        
        metrics = calculate_metrics(y_true, y_pred)
        metrics['特征'] = feat_name
        metrics['数据集'] = dataset_name
        results.append(metrics)
        
        print(f"   {feat_name}:")
        print(f"      MAE:  {metrics['MAE']:.4f}")
        print(f"      RMSE: {metrics['RMSE']:.4f}")
        print(f"      MAPE: {metrics['MAPE']:.4f}%")
        print(f"      样本数: {metrics['样本数']}")
    
    return pd.DataFrame(results)


def main():
    print("="*70)
    print("GMAN 预测结果指标计算")
    print("="*70)
    
    datasets = ['train', 'val', 'test']
    all_results = []
    
    for dataset_name in datasets:
        csv_path = os.path.join(RESULTS_DIR, f'{dataset_name}_predictions.csv')
        
        if not os.path.exists(csv_path):
            print(f"\n⚠️  文件不存在: {csv_path}")
            continue
        
        result_df = process_csv_file(csv_path, dataset_name)
        if result_df is not None:
            all_results.append(result_df)
    
    if not all_results:
        print("\n❌ 没有成功处理任何文件")
        return
    
    # 合并所有结果
    final_df = pd.concat(all_results, ignore_index=True)
    
    # 保存汇总结果
    print(f"\n{'='*70}")
    print("📊 指标汇总")
    print(f"{'='*70}")
    
    # 按数据集和特征排序
    final_df = final_df.sort_values(['数据集', '特征'])
    
    # 打印表格
    pivot_table = final_df.pivot_table(
        index='特征',
        columns='数据集',
        values=['MAE', 'RMSE', 'MAPE'],
        aggfunc='first'
    )
    
    print("\n📈 MAE 对比:")
    print(pivot_table['MAE'].to_string())
    
    print("\n📈 RMSE 对比:")
    print(pivot_table['RMSE'].to_string())
    
    print("\n📈 MAPE 对比:")
    print(pivot_table['MAPE'].to_string())
    
    # 保存到CSV
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ 详细结果已保存到: {OUTPUT_FILE}")
    
    # 保存pivot表格
    pivot_path = os.path.join(RESULTS_DIR, 'metrics_pivot.csv')
    pivot_table.to_csv(pivot_path, encoding='utf-8-sig')
    print(f"✅ 汇总表格已保存到: {pivot_path}")
    
    print(f"\n{'='*70}")
    print("✅ 指标计算完成!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
