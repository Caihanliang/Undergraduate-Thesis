#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入分析GMAN低谷时段预测值分布
找出为什么低谷时段预测值看起来像常数
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_gman_low_traffic():
    """分析GMAN低谷时段预测值"""
    
    # 读取GMAN预测结果
    gman_file = '/home/user/Downloads/cai/结果汇总98/GMAN/models/highway_4feat/train_predictions.csv'
    gman_df = pd.read_csv(gman_file, encoding='utf-8-sig')
    
    print("=" * 80)
    print("GMAN低谷时段预测值深度分析")
    print("=" * 80)
    
    # 只保留time_step=0的数据
    if 'time_step' in gman_df.columns:
        gman_data = gman_df[gman_df['time_step'] == 0].copy()
    
    # 获取站点0的数据
    station_col = 'station_index' if 'station_index' in gman_df.columns else 'station'
    station_0 = gman_data[gman_data[station_col] == 0].copy()
    
    # 获取预测值和真实值
    pred_col = '小客车上行_预测值' if '小客车上行_预测值' in gman_df.columns else '小客车上行_预测'
    true_col = '小客车上行_真实值' if '小客车上行_真实值' in gman_df.columns else '小客车上行_真实'
    
    predictions = station_0[pred_col].values
    truths = station_0[true_col].values
    
    # 定义不同流量水平的阈值
    thresholds = [
        (0, 200, "极低流量 (< 200)"),
        (200, 300, "低流量 (200-300)"),
        (300, 500, "中等流量 (300-500)"),
        (500, 1000, "中高流量 (500-1000)"),
        (1000, 2101, "高流量 (> 1000)")
    ]
    
    print(f"\n{'=' * 80}")
    print("不同流量水平下的预测值分布")
    print(f"{'=' * 80}")
    
    for low, high, label in thresholds:
        mask = (truths >= low) & (truths < high)
        subset_preds = predictions[mask]
        subset_truths = truths[mask]
        
        if len(subset_preds) > 0:
            unique_count = len(np.unique(subset_preds))
            std_dev = subset_preds.std()
            mae = np.abs(subset_preds - subset_truths).mean()
            
            print(f"\n{label}:")
            print(f"  样本数: {len(subset_preds)}")
            print(f"  真实值范围: [{subset_truths.min():.0f}, {subset_truths.max():.0f}]")
            print(f"  预测值范围: [{subset_preds.min():.2f}, {subset_preds.max():.2f}]")
            print(f"  预测值均值: {subset_preds.mean():.2f}")
            print(f"  预测值标准差: {std_dev:.2f}")
            print(f"  预测值唯一值数量: {unique_count}")
            print(f"  MAE: {mae:.2f}")
            
            # 如果标准差很小或唯一值很少，发出警告
            if std_dev < 20 or unique_count < len(subset_preds) * 0.3:
                print(f"  ⚠️ 警告：预测值变化不足！")
                
                # 显示最常见的几个预测值
                value_counts = pd.Series(subset_preds).value_counts().head(5)
                print(f"  最常见的预测值:")
                for val, count in value_counts.items():
                    print(f"    {val:.2f}: {count}次 ({count/len(subset_preds)*100:.1f}%)")
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 所有数据的散点图
    ax1 = axes[0, 0]
    scatter = ax1.scatter(truths, predictions, alpha=0.5, s=10, c=np.arange(len(truths)), cmap='viridis')
    ax1.plot([0, 2200], [0, 2200], 'r--', linewidth=2, label='完美预测线')
    ax1.set_xlabel('真实值', fontsize=12)
    ax1.set_ylabel('预测值', fontsize=12)
    ax1.set_title('GMAN预测值 vs 真实值（所有数据）', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 低谷时段的直方图
    ax2 = axes[0, 1]
    low_mask = truths < 300
    ax2.hist(predictions[low_mask], bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(x=predictions[low_mask].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'均值={predictions[low_mask].mean():.2f}')
    ax2.set_xlabel('预测值', fontsize=12)
    ax2.set_ylabel('频次', fontsize=12)
    ax2.set_title('低谷时段预测值分布（真实值<300）', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 高峰时段的直方图
    ax3 = axes[1, 0]
    high_mask = truths >= 1000
    ax3.hist(predictions[high_mask], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(x=predictions[high_mask].mean(), color='red', linestyle='--', linewidth=2,
                label=f'均值={predictions[high_mask].mean():.2f}')
    ax3.set_xlabel('预测值', fontsize=12)
    ax3.set_ylabel('频次', fontsize=12)
    ax3.set_title('高峰时段预测值分布（真实值>=1000）', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4: 时间序列对比
    ax4 = axes[1, 1]
    sample_indices = np.arange(len(truths))
    ax4.plot(sample_indices, truths, 'b-', linewidth=1, alpha=0.7, label='真实值')
    ax4.plot(sample_indices, predictions, 'brown', linewidth=1, alpha=0.7, label='预测值')
    ax4.axhline(y=300, color='gray', linestyle=':', linewidth=2, label='低谷阈值(300)')
    ax4.set_xlabel('样本索引', fontsize=12)
    ax4.set_ylabel('流量值', fontsize=12)
    ax4.set_title('GMAN预测值与真实值时间序列对比', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = '/home/user/Downloads/cai/GMAN/gman_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 已保存分析图表: {save_path}")
    
    # 总结
    print(f"\n{'=' * 80}")
    print("诊断结论")
    print(f"{'=' * 80}")
    
    low_std = predictions[low_mask].std() if np.any(low_mask) else 0
    if low_std < 20:
        print("❌ 问题确认：低谷时段预测值变化严重不足")
        print(f"   标准差仅为 {low_std:.2f}（正常应该>100）")
        print("   原因：模型在低流量时段无法学习时间序列模式，倾向于输出接近均值的常数")
        print("\n   建议：")
        print("     1. ✅ 检查输出层激活函数（应该是线性而非ReLU）")
        print("     2. 增加低谷时段样本的训练权重")
        print("     3. 使用峰值感知损失函数（如Huber loss + peak penalty）")
        print("     4. 增加训练轮数或使用更小的学习率")
    else:
        print("✅ 低谷时段预测值有正常波动")

if __name__ == '__main__':
    analyze_gman_low_traffic()
