#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断GMAN预测值的分布模式
分析为什么低谷时段预测值都是常数
"""

import pandas as pd
import numpy as np
import os

def diagnose_gman_predictions():
    """诊断GMAN预测值分布"""
    
    # 读取GMAN预测结果
    # gman_file = '/home/user/Downloads/cai/GMAN/results/train_predictions.csv'
    gman_file = '/home/user/Downloads/cai/结果汇总98/GMAN/models/highway_4feat/train_predictions.csv'

    
    if not os.path.exists(gman_file):
        print(f"❌ 文件不存在: {gman_file}")
        return
    
    gman_df = pd.read_csv(gman_file, encoding='utf-8-sig')
    
    print("=" * 80)
    print("GMAN预测值诊断分析")
    print("=" * 80)
    
    # 只保留time_step=0的数据
    if 'time_step' in gman_df.columns:
        gman_data = gman_df[gman_df['time_step'] == 0].copy()
        print(f"\n过滤后数据量（time_step=0）: {len(gman_data)} 条")
    
    # 获取站点0的数据
    station_col = 'station_index' if 'station_index' in gman_df.columns else 'station'
    station_0 = gman_data[gman_data[station_col] == 0].copy()
    
    print(f"\n站点0的数据量: {len(station_0)} 条")
    
    # 获取预测值和真实值
    pred_col = '小客车上行_预测值' if '小客车上行_预测值' in gman_df.columns else '小客车上行_预测'
    true_col = '小客车上行_真实值' if '小客车上行_真实值' in gman_df.columns else '小客车上行_真实'
    
    predictions = station_0[pred_col].values
    truths = station_0[true_col].values
    
    print(f"\n预测值统计:")
    print(f"  最小值: {predictions.min():.2f}")
    print(f"  最大值: {predictions.max():.2f}")
    print(f"  均值:   {predictions.mean():.2f}")
    print(f"  标准差: {predictions.std():.2f}")
    
    print(f"\n真实值统计:")
    print(f"  最小值: {truths.min():.2f}")
    print(f"  最大值: {truths.max():.2f}")
    print(f"  均值:   {truths.mean():.2f}")
    print(f"  标准差: {truths.std():.2f}")
    
    # 分析低谷时段（真实值 < 300）
    low_traffic_mask = truths < 300
    low_traffic_preds = predictions[low_traffic_mask]
    low_traffic_truths = truths[low_traffic_mask]
    
    print(f"\n{'=' * 80}")
    print("低谷时段分析（真实值 < 300）")
    print(f"{'=' * 80}")
    print(f"低谷时段样本数: {len(low_traffic_preds)}")
    
    if len(low_traffic_preds) > 0:
        print(f"\n低谷时段预测值统计:")
        print(f"  最小值: {low_traffic_preds.min():.2f}")
        print(f"  最大值: {low_traffic_preds.max():.2f}")
        print(f"  均值:   {low_traffic_preds.mean():.2f}")
        print(f"  标准差: {low_traffic_preds.std():.2f}")
        
        # 检查是否都是同一个值
        unique_values = np.unique(low_traffic_preds)
        print(f"\n低谷时段预测值唯一值数量: {len(unique_values)}")
        
        if len(unique_values) <= 5:
            print(f"⚠️ 警告：低谷时段预测值几乎都相同！")
            print(f"唯一值列表: {unique_values[:10]}")  # 只显示前10个
        
        # 计算MAE
        mae = np.abs(low_traffic_preds - low_traffic_truths).mean()
        print(f"\n低谷时段MAE: {mae:.2f}")
    
    # 分析高峰时段（真实值 >= 1000）
    high_traffic_mask = truths >= 1000
    high_traffic_preds = predictions[high_traffic_mask]
    high_traffic_truths = truths[high_traffic_mask]
    
    print(f"\n{'=' * 80}")
    print("高峰时段分析（真实值 >= 1000）")
    print(f"{'=' * 80}")
    print(f"高峰时段样本数: {len(high_traffic_preds)}")
    
    if len(high_traffic_preds) > 0:
        print(f"\n高峰时段预测值统计:")
        print(f"  最小值: {high_traffic_preds.min():.2f}")
        print(f"  最大值: {high_traffic_preds.max():.2f}")
        print(f"  均值:   {high_traffic_preds.mean():.2f}")
        print(f"  标准差: {high_traffic_preds.std():.2f}")
        
        # 检查是否有波动
        unique_values = np.unique(high_traffic_preds)
        print(f"\n高峰时段预测值唯一值数量: {len(unique_values)}")
        
        if len(unique_values) > 100:
            print(f"✅ 高峰时段预测值有正常波动")
        else:
            print(f"⚠️ 警告：高峰时段预测值变化较少")
        
        # 计算MAE
        mae = np.abs(high_traffic_preds - high_traffic_truths).mean()
        print(f"\n高峰时段MAE: {mae:.2f}")
    
    # 总结
    print(f"\n{'=' * 80}")
    print("诊断结论")
    print(f"{'=' * 80}")
    
    if len(low_traffic_preds) > 0 and low_traffic_preds.std() < 1.0:
        print("❌ 问题确认：低谷时段预测值几乎是常数（标准差 < 1.0）")
        print("   原因：模型在低流量时段无法学习时间序列模式")
        print("   建议：")
        print("     1. 增加训练数据中的低谷时段样本权重")
        print("     2. 使用峰值感知损失函数")
        print("     3. 检查输出层激活函数是否正确（应该是线性而非ReLU）")
    else:
        print("✅ 低谷时段预测值有正常波动")
    
    if len(high_traffic_preds) > 0 and high_traffic_preds.std() > 100:
        print("✅ 高峰时段预测值有正常波动")
    else:
        print("⚠️ 高峰时段预测值波动不足")

if __name__ == '__main__':
    diagnose_gman_predictions()
