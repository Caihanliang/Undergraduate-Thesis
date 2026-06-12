#!/usr/bin/env python
# coding: utf-8
"""
计算ASTGCN训练集预测结果的评估指标
MAE, RMSE, MAPE, sMAPE (按特征分组，仅评估第一个时间步T0)
"""

import pandas as pd
import numpy as np
import argparse
import os


def calculate_metrics(y_true, y_pred):
    """
    计算评估指标
    
    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
    
    Returns:
        dict: 包含MAE, RMSE, MAPE, sMAPE的字典
    """
    # 非负截断（符合物理意义）
    y_pred = np.maximum(y_pred, 0)
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAPE (排除真实值为0或极小值的样本)
    epsilon = 1e-8
    mask = y_true > epsilon  # 排除接近0的值
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))) * 100
    else:
        mape = np.nan
    
    # sMAPE (对称MAPE，更稳健)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
    
    # R² (决定系数)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'sMAPE': smape,
        'R²': r2,
        '样本数': len(y_true),
        '有效样本数(MAPE)': int(mask.sum()),
        '真实值均值': np.mean(y_true),
        '真实值标准差': np.std(y_true),
        '预测值均值': np.mean(y_pred),
        '预测值标准差': np.std(y_pred),
        '真实值范围': f"[{y_true.min():.2f}, {y_true.max():.2f}]",
        '预测值范围': f"[{y_pred.min():.2f}, {y_pred.max():.2f}]"
    }


def main():
    parser = argparse.ArgumentParser(description='计算ASTGCN训练预测结果评估指标')
    parser.add_argument('--csv_path', type=str, 
                        default='./experiments/highway_traffic/astgcn_r_h1d0w0_channel4_1.000000e-03/train_predictions.csv',
                        help='预测结果CSV文件路径')
    parser.add_argument('--time_step', type=int, default=0,
                        help='评估的时间步索引 (0=T0, 1=T1, ..., 7=T7)，默认只评估第一个时间步T0')
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.csv_path):
        print(f"❌ 文件不存在: {args.csv_path}")
        return
    
    print("=" * 70)
    print("ASTGCN 训练集预测结果评估")
    print(f"评估时间步: T{args.time_step} (第{args.time_step + 1}个预测小时)")
    print("=" * 70)
    
    # 加载数据
    print(f"\n📂 加载数据: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"   总行数: {len(df)}")
    print(f"   列名: {list(df.columns)}")
    
    # 过滤指定时间步的数据
    time_label = f'_T{args.time_step}'
    df_filtered = df[df['时间'].str.contains(time_label, na=False)]
    
    print(f"   时间步 T{args.time_step} 的数据量: {len(df_filtered)} 行")
    
    if len(df_filtered) == 0:
        print(f"❌ 未找到时间步 T{args.time_step} 的数据")
        return
    
    # 获取所有特征
    features = df_filtered['特征'].unique()
    print(f"   特征数量: {len(features)}")
    print(f"   特征列表: {list(features)}")
    
    # 按特征分组计算指标
    print(f"\n{'=' * 70}")
    print(f"各特征评估指标 (时间步 T{args.time_step})")
    print(f"{'=' * 70}")
    
    all_metrics = {}
    
    for feature in features:
        feature_data = df_filtered[df_filtered['特征'] == feature]
        
        y_true = feature_data['真实值'].values
        y_pred = feature_data['预测值'].values
        
        metrics = calculate_metrics(y_true, y_pred)
        all_metrics[feature] = metrics
        
        print(f"\n{feature}:")
        print(f"  MAE:  {metrics['MAE']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}% (基于{metrics['有效样本数(MAPE)']}个有效样本)")
        print(f"  sMAPE: {metrics['sMAPE']:.2f}%")
        print(f"  R²:   {metrics['R²']:.4f}")
        print(f"  总样本数: {metrics['样本数']}")
        print(f"  📊 数据统计:")
        print(f"     真实值 - 均值: {metrics['真实值均值']:.2f}, 标准差: {metrics['真实值标准差']:.2f}")
        print(f"     预测值 - 均值: {metrics['预测值均值']:.2f}, 标准差: {metrics['预测值标准差']:.2f}")
        print(f"     真实值范围: {metrics['真实值范围']}")
        print(f"     预测值范围: {metrics['预测值范围']}")
    
    # 整体评估指标
    print(f"\n{'=' * 70}")
    print(f"整体评估指标 (所有特征, 时间步 T{args.time_step})")
    print(f"{'=' * 70}")
    
    y_true_all = df_filtered['真实值'].values
    y_pred_all = df_filtered['预测值'].values
    overall_metrics = calculate_metrics(y_true_all, y_pred_all)
    
    print(f"  MAE:  {overall_metrics['MAE']:.2f}")
    print(f"  RMSE: {overall_metrics['RMSE']:.2f}")
    print(f"  MAPE: {overall_metrics['MAPE']:.2f}% (基于{overall_metrics['有效样本数(MAPE)']}个有效样本)")
    print(f"  sMAPE: {overall_metrics['sMAPE']:.2f}%")
    print(f"  R²:   {overall_metrics['R²']:.4f}")
    print(f"  总样本数: {overall_metrics['样本数']}")
    
    # 保存评估结果
    output_path = args.csv_path.replace('.csv', f'_metrics_T{args.time_step}.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"ASTGCN 训练集预测结果评估指标 (时间步 T{args.time_step})\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("各特征评估指标:\n")
        f.write("-" * 70 + "\n")
        for feature, metrics in all_metrics.items():
            f.write(f"\n{feature}:\n")
            f.write(f"  MAE:  {metrics['MAE']:.2f}\n")
            f.write(f"  RMSE: {metrics['RMSE']:.2f}\n")
            f.write(f"  MAPE: {metrics['MAPE']:.2f}% (基于{metrics['有效样本数(MAPE)']}个有效样本)\n")
            f.write(f"  sMAPE: {metrics['sMAPE']:.2f}%\n")
            f.write(f"  R²:   {metrics['R²']:.4f}\n")
            f.write(f"  总样本数: {metrics['样本数']}\n")
            f.write(f"  数据统计:\n")
            f.write(f"     真实值 - 均值: {metrics['真实值均值']:.2f}, 标准差: {metrics['真实值标准差']:.2f}\n")
            f.write(f"     预测值 - 均值: {metrics['预测值均值']:.2f}, 标准差: {metrics['预测值标准差']:.2f}\n")
            f.write(f"     真实值范围: {metrics['真实值范围']}\n")
            f.write(f"     预测值范围: {metrics['预测值范围']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"整体评估指标 (所有特征, 时间步 T{args.time_step}):\n")
        f.write("-" * 70 + "\n")
        f.write(f"  MAE:  {overall_metrics['MAE']:.2f}\n")
        f.write(f"  RMSE: {overall_metrics['RMSE']:.2f}\n")
        f.write(f"  MAPE: {overall_metrics['MAPE']:.2f}% (基于{overall_metrics['有效样本数(MAPE)']}个有效样本)\n")
        f.write(f"  sMAPE: {overall_metrics['sMAPE']:.2f}%\n")
        f.write(f"  R²:   {overall_metrics['R²']:.4f}\n")
        f.write(f"  总样本数: {overall_metrics['样本数']}\n")
    
    print(f"\n💾 评估结果已保存到: {output_path}")
    print(f"\n{'=' * 70}")
    print("✅ 评估完成!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
