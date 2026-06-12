#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注误差大的时间点 - MOMENT模型全量站点版
为157个站点分别生成4个特征的图，标注误差大的点（带文字标签）
保持与原始可视化一致的风格
 python 标注误差大.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import json

# ========== 配置参数 ==========
BASE_DIR = "/home/user/Downloads/cai/moment-main/moment-main"
PREDICTION_FILE = os.path.join(BASE_DIR, "visualization-results/predictions/train_predictions.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "visualization-results/train_figures_error_annotation")

# 误差配置
ERROR_CONFIG = {
    'top_n_per_feature': 15,         # 每个特征标注前15个最大误差点
    'start_date': '2023-09-01 08:00:00',  # 起始时间
    'time_step_hours': 1,            # 时间步长
    'figsize': (14, 5),              # 图片大小
    'dpi': 150,                      # 分辨率
}

# 特征名称映射（英文）
FEATURE_DISPLAY_NAMES = {
    'Passenger_Car_Up': 'Passenger Car Up',
    'Passenger_Car_Down': 'Passenger Car Down',
    'Non_Passenger_Car_Up': 'Non-Passenger Car Up',
    'Non_Passenger_Car_Down': 'Non-Passenger Car Down'
}

# 特征顺序
FEATURE_ORDER = ['Non_Passenger_Car_Down', 'Non_Passenger_Car_Up', 
                 'Passenger_Car_Down', 'Passenger_Car_Up']

# 颜色配置（与原始可视化一致）
COLORS = {
    'actual': '#1f77b4',
    'predicted': '#ff4b5c',
    'error_marker': 'red',
    'error_marker_edge': 'darkred',
    'annotation_bg': 'yellow',
    'annotation_edge': 'orange'
}


def load_and_prepare_data():
    """加载并准备数据"""
    print("=" * 80)
    print("Loading prediction data")
    print("=" * 80)
    
    if not os.path.exists(PREDICTION_FILE):
        print(f"\nFile not found: {PREDICTION_FILE}")
        raise FileNotFoundError("Prediction file not found")
    
    df = pd.read_csv(PREDICTION_FILE)
    
    # Extract sample numbers and calculate real time
    df['sample_num'] = df['时间'].str.extract(r'Sample_(\d+)')[0].astype(int)
    start_date = pd.Timestamp(ERROR_CONFIG['start_date'])
    df['datetime'] = start_date + pd.to_timedelta(df['sample_num'] * ERROR_CONFIG['time_step_hours'], unit='h')
    
    print(f"Data shape: {df.shape}")
    print(f"Stations: {df['站点编号'].nunique()}")
    print(f"Features: {df['特征'].unique().tolist()}")
    
    # Calculate errors
    df['absolute_error'] = np.abs(df['真实值'] - df['预测值'])
    df['relative_error'] = df['absolute_error'] / (np.abs(df['真实值']) + 1e-8) * 100
    
    return df


def plot_station_with_annotations(df, station_id, feature, output_dir):
    """为单个站点的单个特征绘制带标注的图"""
    
    # 筛选该站点该特征的数据
    station_data = df[(df['站点编号'] == station_id) & (df['特征'] == feature)].copy()
    station_data = station_data.sort_values('datetime')
    
    if len(station_data) == 0:
        return 0
    
    # 找出误差最大的点（只标注前N个）
    station_data_sorted = station_data.sort_values('absolute_error', ascending=False)
    top_errors = station_data_sorted.head(ERROR_CONFIG['top_n_per_feature'])
    
    # 创建图表（与原始可视化风格一致）
    fig, ax = plt.subplots(figsize=ERROR_CONFIG['figsize'])
    
    time_vals = station_data['datetime'].values
    true_vals = station_data['真实值'].values
    pred_vals = station_data['预测值'].values
    
    # 绘制真实值和预测值（与原始代码一致）
    ax.plot(time_vals, true_vals, label='Actual', linewidth=1.5, 
            color=COLORS['actual'], marker='o', markersize=3, alpha=0.8)
    ax.plot(time_vals, pred_vals, label='Predicted', linewidth=1.5, 
            color=COLORS['predicted'], linestyle='--', marker='s', markersize=3, alpha=0.8)
    
    # 标注误差大的点（带文字标签）
    for _, row in top_errors.iterrows():
        time_point = row['datetime']
        true_val = row['真实值']
        pred_val = row['预测值']
        error_pct = row['relative_error']
        
        # 画红圈（与原始代码一致）
        ax.plot(time_point, true_val, 'o', color=COLORS['error_marker'], markersize=10,
               markeredgecolor=COLORS['error_marker_edge'], markeredgewidth=1.5, 
               markerfacecolor='none', zorder=5)
        ax.plot(time_point, pred_val, 'o', color='orange', markersize=6,
               markeredgecolor='darkorange', markeredgewidth=1, zorder=5)
        
        # 添加文字标注（与原始代码一致的格式）
        time_str = pd.Timestamp(time_point).strftime('%m-%d %H:%M')
        ax.annotate(f'Error:{error_pct:.0f}%\n{time_str}',
                   xy=(time_point, true_val),
                   xytext=(0, 20),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['annotation_bg'], 
                            alpha=0.8, edgecolor=COLORS['annotation_edge']),
                   arrowprops=dict(arrowstyle='->', color=COLORS['error_marker'], lw=1.2),
                   fontsize=8,
                   ha='center',
                   zorder=10)
    
    # 计算MAE和RMSE
    mae = np.mean(station_data['absolute_error'])
    rmse = np.sqrt(np.mean(station_data['absolute_error'] ** 2))
    
    # 设置标题和标签（与原始代码一致）
    feature_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
    ax.set_title(f"Station_{station_id:03d} - {feature_name} | MAE: {mae:.2f} | RMSE: {rmse:.2f}", 
                fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Normalized Traffic Flow', fontsize=11)
    
    # 格式化x轴（与原始代码一致）- 修复numpy datetime64问题
    time_vals_pd = pd.to_datetime(time_vals)
    time_span_days = (time_vals_pd[-1] - time_vals_pd[0]).days
    
    if time_span_days <= 7:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    elif time_span_days <= 30:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    else:
        interval = max(1, time_span_days // 10)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = os.path.join(output_dir, f'station_{station_id:03d}_{feature}_annotated.png')
    plt.savefig(output_file, dpi=ERROR_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    return len(top_errors)


def plot_overview_with_annotations(df, features, output_dir):
    """生成带误差标注的概览图"""
    print("\nGenerating overview plots with annotations...")
    
    overview_dir = os.path.join(output_dir, 'overview')
    os.makedirs(overview_dir, exist_ok=True)
    
    for feature in features:
        feature_df = df[df['特征'] == feature]
        
        # 计算平均值
        mean_by_time = feature_df.groupby('datetime').agg({
            '真实值': 'mean',
            '预测值': 'mean',
            'absolute_error': 'mean',
            'relative_error': 'mean'
        }).reset_index()
        
        # 找出平均误差最大的点
        mean_by_time_sorted = mean_by_time.sort_values('absolute_error', ascending=False)
        top_errors = mean_by_time_sorted.head(ERROR_CONFIG['top_n_per_feature'])
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        time_vals = mean_by_time['datetime'].values
        true_mean = mean_by_time['真实值'].values
        pred_mean = mean_by_time['预测值'].values
        
        ax.plot(time_vals, true_mean, label='Actual (Mean)', linewidth=2, 
               color=COLORS['actual'], marker='o', markersize=4, alpha=0.8)
        ax.plot(time_vals, pred_mean, label='Predicted (Mean)', linewidth=2, 
               color=COLORS['predicted'], linestyle='--', marker='s', markersize=4, alpha=0.8)
        
        # 添加误差带
        std_by_time = feature_df.groupby('datetime').agg({
            '真实值': 'std',
            '预测值': 'std'
        }).reset_index()
        
        ax.fill_between(time_vals, 
                       pred_mean - std_by_time['预测值'].values,
                       pred_mean + std_by_time['预测值'].values,
                       alpha=0.2, color='orange', label='Prediction Std')
        
        # 标注平均误差大的点
        for _, row in top_errors.iterrows():
            time_point = row['datetime']
            true_val = row['真实值']
            error_pct = row['relative_error']
            
            # 画红圈
            ax.plot(time_point, true_val, 'o', color=COLORS['error_marker'], markersize=12,
                   markeredgecolor=COLORS['error_marker_edge'], markeredgewidth=1.5, 
                   markerfacecolor='none', zorder=5)
            
            # 添加文字标注
            time_str = pd.Timestamp(time_point).strftime('%Y-%m-%d')
            ax.annotate(f'Avg Error:{error_pct:.0f}%\n{time_str}',
                       xy=(time_point, true_val),
                       xytext=(0, 20),
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['annotation_bg'], 
                                alpha=0.8, edgecolor=COLORS['annotation_edge']),
                       arrowprops=dict(arrowstyle='->', color=COLORS['error_marker'], lw=1.2),
                       fontsize=8,
                       ha='center',
                       zorder=10)
        
        feature_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
        
        # 计算整体指标
        mae = np.mean(mean_by_time['absolute_error'])
        rmse = np.sqrt(np.mean(mean_by_time['absolute_error'] ** 2))
        
        ax.set_title(f"Average Across All Stations - {feature_name} | MAE: {mae:.2f} | RMSE: {rmse:.2f}", 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Normalized Traffic Flow (Mean)', fontsize=12)
        
        # 格式化x轴 - 修复numpy datetime64问题
        time_vals_pd = pd.to_datetime(time_vals)
        time_span_days = (time_vals_pd[-1] - time_vals_pd[0]).days
        
        if time_span_days <= 7:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
        elif time_span_days <= 30:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        else:
            interval = max(1, time_span_days // 10)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11)
        
        plt.tight_layout()
        
        save_path = os.path.join(overview_dir, f'{feature_name.replace(" ", "_")}_overview_annotated.png')
        plt.savefig(save_path, dpi=ERROR_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {feature_name.replace(' ', '_')}_overview_annotated.png")


def save_annotation_summary(df, output_dir):
    """保存标注摘要信息"""
    summary = {}
    
    for feature in FEATURE_ORDER:
        feature_df = df[df['特征'] == feature]
        feature_summary = {
            'total_samples': len(feature_df),
            'mean_absolute_error': float(feature_df['absolute_error'].mean()),
            'std_absolute_error': float(feature_df['absolute_error'].std()),
            'max_absolute_error': float(feature_df['absolute_error'].max()),
            'top_error_points': []
        }
        
        # 获取全局top误差点
        top_global = feature_df.nlargest(ERROR_CONFIG['top_n_per_feature'], 'absolute_error')
        for _, row in top_global.iterrows():
            feature_summary['top_error_points'].append({
                'station': int(row['站点编号']),
                'time': row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                'actual': float(row['真实值']),
                'predicted': float(row['预测值']),
                'absolute_error': float(row['absolute_error']),
                'relative_error': float(row['relative_error'])
            })
        
        summary[feature] = feature_summary
    
    # 保存到JSON
    summary_path = os.path.join(output_dir, 'annotation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAnnotation summary saved to: {summary_path}")


def main():
    """主函数"""
    print("\n" + "🚀" * 40)
    print("MOMENT - Generate Error Annotation Plots")
    print("🚀" * 40 + "\n")
    
    # 加载数据
    df = load_and_prepare_data()
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有站点和特征
    stations = sorted(df['站点编号'].unique())
    features = FEATURE_ORDER
    
    print(f"\n📍 Total stations: {len(stations)}")
    print(f"📊 Features: {features}")
    print(f"🎯 Top {ERROR_CONFIG['top_n_per_feature']} error points per feature\n")
    
    # 为每个站点的每个特征生成图
    total_plots = 0
    total_annotations = 0
    
    for station_idx, station_id in enumerate(stations, 1):
        print(f"\nProcessing station {station_idx}/{len(stations)}: Station_{station_id:03d}")
        
        for feature in features:
            num_annotations = plot_station_with_annotations(df, station_id, feature, OUTPUT_DIR)
            if num_annotations > 0:
                total_plots += 1
                total_annotations += num_annotations
                print(f"  ✓ {FEATURE_DISPLAY_NAMES.get(feature, feature)}: {num_annotations} points annotated")
        
        if station_idx % 20 == 0:
            print(f"\n📊 Progress: {station_idx}/{len(stations)} stations completed")
    
    # 生成概览图
    plot_overview_with_annotations(df, features, OUTPUT_DIR)
    
    # 保存标注摘要
    save_annotation_summary(df, OUTPUT_DIR)
    
    print("\n" + "✅" * 40)
    print(f"Task completed!")
    print(f"  - Total plots: {total_plots}")
    print(f"  - Total annotations: {total_annotations}")
    print("✅" * 40)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print("💡 Each plot contains:")
    print("   - Actual and predicted curves (same style as original)")
    print("   - Red circles marking high-error points")
    print("   - Text labels showing error percentage and time")
    print("   - MAE and RMSE in title")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()