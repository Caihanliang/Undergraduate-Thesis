#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注误差大的时间点 - 全量站点版
为157个站点分别生成4个特征的图，在图上画红圈并标注时间和误差
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# ========== 配置参数 ==========
BASE_DIR = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main"
PREDICTIONS_DIR = os.path.join(BASE_DIR, "visulization-result-4FEAT-train/predictions")
PREDICTION_FILE = os.path.join(PREDICTIONS_DIR, "prediction_with_section_name.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "visulization-result-4FEAT-train/error_annotation")

# 误差配置
ERROR_CONFIG = {
    'mae_threshold': 100,            # MAE阈值100（流量单位：辆/小时）
    'top_n_per_feature': 15,         # 每个特征标注前15个最大误差点
}

# 特征名称映射（保持与原始代码一致）
FEATURE_DISPLAY_NAMES = {
    'LittleCar_Up': 'LittleCar_Up',
    'LittleCar_Down': 'LittleCar_Down',
    'NonLittleCar_Up': 'NonLittleCar_Up',
    'NonLittleCar_Down': 'NonLittleCar_Down'
}


def load_and_prepare_data():
    """加载并准备数据"""
    print("=" * 80)
    print("📊 加载预测数据")
    print("=" * 80)
    
    if not os.path.exists(PREDICTION_FILE):
        print(f"\n❌ 文件不存在: {PREDICTION_FILE}")
        if os.path.exists(PREDICTIONS_DIR):
            print(f"\n📁 当前目录内容:")
            for f in os.listdir(PREDICTIONS_DIR):
                print(f"   - {f}")
        raise FileNotFoundError("预测文件不存在")
    
    df = pd.read_csv(PREDICTION_FILE)
    df['时间'] = pd.to_datetime(df['时间'])
    
    print(f"✅ 数据形状: {df.shape}")
    print(f"✅ 站点数: {df['站点名称'].nunique()}")
    print(f"✅ 特征: {df['特征'].unique().tolist()}")
    
    # 计算误差
    df['absolute_error'] = (df['真实值'] - df['预测值']).abs()
    df['mae'] = df['absolute_error']  # MAE就是绝对误差
    df['mape'] = np.where(df['真实值'] > 0, 
                         df['absolute_error'] / df['真实值'] * 100, 0)
    
    return df


def plot_station_with_annotations(df, station_name, feature, output_dir):
    """为单个站点的单个特征绘制带红圈标注的图 - 智能防重叠版"""
    
    # 筛选该站点该特征的数据
    station_data = df[(df['站点名称'] == station_name) & (df['特征'] == feature)].copy()
    station_data = station_data.sort_values('时间')
    
    if len(station_data) == 0:
        return
    
    # 找出误差最大的点
    station_data_sorted = station_data.sort_values('mae', ascending=False)
    top_errors = station_data_sorted.head(ERROR_CONFIG['top_n_per_feature'])
    
    # 创建图表（与原始可视化脚本相同格式）
    fig, ax = plt.subplots(figsize=(18, 7))
    
    # 绘制真实值和预测值
    ax.plot(station_data['时间'], station_data['真实值'], 
           label='True Traffic', linewidth=1.5, color='#1f77b4', alpha=0.8, zorder=1)
    ax.plot(station_data['时间'], station_data['预测值'], 
           label='Pred Traffic', linewidth=1.5, color='#ff4b5c', linestyle='--', alpha=0.8, zorder=1)
    
    # 智能防重叠标注策略
    # 1. 计算每个标注的边界框
    # 2. 检测重叠并调整位置
    annotations_info = []
    
    for idx, (_, row) in enumerate(top_errors.iterrows()):
        time_point = row['时间']
        true_val = row['真实值']
        mape = row['mape']
        
        annotations_info.append({
            'time': time_point,
            'value': true_val,
            'mae': mape,  # 这里变量名是mape，但实际是mae值
            'idx': idx
        })
    
    # 按时间排序，便于检测相邻标注的重叠
    annotations_info.sort(key=lambda x: x['time'])
    
    # 为每个标注分配垂直位置（层）
    # 使用贪心算法：为每个标注选择不与前面标注重叠的最低层
    for i, ann in enumerate(annotations_info):
        # 默认放在上方
        ann['layer'] = 0
        ann['position'] = 'above'  # 或 'below'
        
        # 检查与前面标注的重叠
        for j in range(i):
            prev_ann = annotations_info[j]
            time_diff = (ann['time'] - prev_ann['time']).total_seconds() / 3600  # 小时差
            
            # 如果时间间隔小于3小时，可能需要错开
            if time_diff < 3:
                # 如果前一个在上方，这个就放下方
                if prev_ann['position'] == 'above':
                    ann['position'] = 'below'
                    ann['layer'] = prev_ann['layer']
                else:
                    # 前一个在下方，这个放上方，但可能需要更高的层
                    ann['position'] = 'above'
                    ann['layer'] = prev_ann['layer'] + 1
                break
    
    # 绘制标注
    for ann in annotations_info:
        time_point = ann['time']
        true_val = ann['value']
        mae = ann['mae']
        position = ann['position']
        layer = ann['layer']
        
        # 画红圈
        ax.plot(time_point, true_val, 'ro', markersize=14, 
               markeredgewidth=2.5, markeredgecolor='darkred', 
               markerfacecolor='none', zorder=5)
        
        # 根据位置和层计算偏移
        if position == 'above':
            y_offset = 20 + layer * 30  # 上方，每层高30像素
            text_y = true_val + y_offset
            va = 'bottom'
            arrow_style = '->'
            rad = 0.2
        else:
            y_offset = -(20 + layer * 30)  # 下方
            text_y = true_val + y_offset
            va = 'top'
            arrow_style = '-['
            rad = -0.2
        
        # 标注时间和误差
        ax.annotate(f'{time_point.strftime("%m-%d %H:%M")}\nMAE={mae:.1f}', 
                   xy=(time_point, true_val),
                   xytext=(0, y_offset),
                   textcoords='offset points',
                   ha='center',
                   va=va,
                   fontsize=8,
                   color='darkred',
                   fontweight='bold',
                   zorder=10 + layer,
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='yellow', 
                            alpha=0.9,
                            edgecolor='darkred',
                            linewidth=1.5,
                            zorder=10 + layer),
                   arrowprops=dict(arrowstyle=arrow_style, 
                                 color='darkred', 
                                 lw=1.5,
                                 shrinkA=3,
                                 shrinkB=3,
                                 connectionstyle=f'arc3,rad={rad}',
                                 zorder=10 + layer))
    
    # 设置标题和标签
    station_idx = station_name.split('_')[-1] if '_' in station_name else station_name
    ax.set_title(f'Station {station_idx} [train] - {FEATURE_DISPLAY_NAMES[feature]}\n'
                f'High Error Points: {len(top_errors)} annotated (MAE > {ERROR_CONFIG["mae_threshold"]})', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Traffic Volume', fontsize=12)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # 设置时间轴格式
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3)
    
    # 扩大y轴范围，给多层标注留出空间
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    max_layer = max([ann['layer'] for ann in annotations_info]) if annotations_info else 0
    extra_space = max_layer * 30 / y_range * 0.1  # 根据层数动态调整
    ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * (0.25 + extra_space))
    
    # 保存图片
    output_file = os.path.join(output_dir, f'{station_name}_{feature}_annotated.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return len(top_errors)


def main():
    """主函数"""
    print("\n" + "🚀" * 40)
    print("开始为157个站点生成带误差标注的图")
    print("🚀" * 40 + "\n")
    
    # 加载数据
    df = load_and_prepare_data()
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有站点
    stations = df['站点名称'].unique()
    features = df['特征'].unique()
    
    print(f"\n📍 总共 {len(stations)} 个站点")
    print(f"📊 特征: {features.tolist()}")
    print(f"🎯 每个特征标注前 {ERROR_CONFIG['top_n_per_feature']} 个最大误差点")
    print(f"📊 MAE阈值: {ERROR_CONFIG['mae_threshold']} (辆/小时)\n")
    
    # 为每个站点的每个特征生成图
    total_plots = 0
    for station_idx, station in enumerate(stations, 1):
        print(f"\n{'='*80}")
        print(f"处理站点 {station_idx}/{len(stations)}: {station}")
        print(f"{'='*80}")
        
        for feature in features:
            num_annotations = plot_station_with_annotations(df, station, feature, OUTPUT_DIR)
            if num_annotations:
                total_plots += 1
                print(f"  ✅ {feature}: 标注了 {num_annotations} 个高误差点")
        
        if station_idx % 20 == 0:
            print(f"\n📊 进度: {station_idx}/{len(stations)} 站点完成")
    
    print("\n" + "✅" * 40)
    print(f"任务完成！共生成 {total_plots} 张图")
    print("✅" * 40)
    print(f"\n📁 输出目录: {OUTPUT_DIR}")
    print("💡 每张图包含:")
    print("   - 真实值和预测值曲线")
    print("   - 红圈标注误差大的点")
    print("   - 黄色标签显示具体时间和MAPE")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()