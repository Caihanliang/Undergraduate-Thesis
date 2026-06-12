#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
找出LLM预测比GNN误差更小的站点和时间点，并绘制8窗口对比图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ast import literal_eval
import os
from collections import defaultdict

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
CSV_FILE = '/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/config/cai/inference_results_all_stations_with_predictions518/llm_gnn_prediction_compare.csv'
OUTPUT_DIR = '/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/config/cai/llm_better_than_gnn_plots'

def parse_sequence(seq_str):
    """将字符串序列转换为列表"""
    try:
        if isinstance(seq_str, str):
            return literal_eval(seq_str)
        return seq_str
    except:
        return None

def calculate_mae(true_seq, pred_seq):
    """计算MAE"""
    if true_seq is None or pred_seq is None:
        return float('inf')
    true_arr = np.array(true_seq)
    pred_arr = np.array(pred_seq)
    return np.mean(np.abs(true_arr - pred_arr))

def find_llm_better_samples():
    """找出LLM比GNN误差更小的样本"""
    print("=" * 100)
    print("开始分析LLM vs GNN预测效果...")
    print("=" * 100)
    
    # 读取数据
    print(f"\n正在加载数据: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    print(f"数据加载完成: {len(df)} 条记录")
    
    # 解析序列
    print("\n正在解析预测序列...")
    df['true_list'] = df['true_seq'].apply(parse_sequence)
    df['gnn_list'] = df['gnn_seq'].apply(parse_sequence)
    df['llm_list'] = df['llm_seq'].apply(parse_sequence)
    
    # 计算MAE
    print("正在计算MAE...")
    df['gnn_mae'] = df.apply(lambda row: calculate_mae(row['true_list'], row['gnn_list']), axis=1)
    df['llm_mae'] = df.apply(lambda row: calculate_mae(row['true_list'], row['llm_list']), axis=1)
    
    # 找出LLM更好的样本
    df['llm_better'] = df['llm_mae'] < df['gnn_mae']
    df['mae_improvement'] = df['gnn_mae'] - df['llm_mae']  # 正值表示LLM更好
    
    # 统计信息
    total_samples = len(df)
    llm_better_count = df['llm_better'].sum()
    improvement_rate = llm_better_count / total_samples * 100
    
    print(f"\n{'='*100}")
    print(f"分析结果:")
    print(f"{'='*100}")
    print(f"总样本数: {total_samples}")
    print(f"LLM优于GNN的样本数: {llm_better_count} ({improvement_rate:.2f}%)")
    print(f"GNN优于LLM的样本数: {total_samples - llm_better_count} ({100-improvement_rate:.2f}%)")
    
    # 按站点统计
    station_stats = df.groupby(['station_id', 'station_short_name']).agg({
        'llm_better': ['sum', 'count'],
        'mae_improvement': 'mean'
    }).reset_index()
    station_stats.columns = ['station_id', 'station_name', 'llm_better_count', 'total_count', 'avg_improvement']
    station_stats['improvement_rate'] = station_stats['llm_better_count'] / station_stats['total_count'] * 100
    
    # 找出LLM表现最好的站点（按提升率排序）
    top_stations = station_stats.nlargest(10, 'improvement_rate')
    
    print(f"\n{'='*100}")
    print(f"LLM表现最好的前10个站点（按提升率）:")
    print(f"{'='*100}")
    for idx, row in top_stations.iterrows():
        print(f"站点 {int(row['station_id']):03d} - {row['station_name']}: "
              f"提升率={row['improvement_rate']:.1f}% "
              f"({int(row['llm_better_count'])}/{int(row['total_count'])}个样本)")
    
    return df, top_stations

def plot_comparison(df, station_id, sample_idx, feature_id, output_dir):
    """绘制单个样本的8窗口对比图"""
    
    # 过滤数据
    sample_data = df[
        (df['station_id'] == station_id) & 
        (df['sample_idx'] == sample_idx) & 
        (df['feature_id'] == feature_id)
    ]
    
    if len(sample_data) == 0:
        print(f"  ⚠️ 未找到站点{station_id} 样本{sample_idx} 特征{feature_id}的数据")
        return None
    
    row = sample_data.iloc[0]
    
    true_seq = row['true_list']
    gnn_seq = row['gnn_list']
    llm_seq = row['llm_list']
    
    if true_seq is None or gnn_seq is None or llm_seq is None:
        print(f"  ⚠️ 数据解析失败")
        return None
    
    # 创建时间轴（8个时间点）
    base_time = pd.to_datetime(row['timestamp'])
    timestamps = [base_time + pd.Timedelta(hours=i) for i in range(8)]
    
    # 计算MAE
    gnn_mae = calculate_mae(true_seq, gnn_seq)
    llm_mae = calculate_mae(true_seq, llm_seq)
    improvement = gnn_mae - llm_mae
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 绘制三条线
    ax.plot(timestamps, true_seq, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='True Value', zorder=3)
    ax.plot(timestamps, gnn_seq, 's--', color='#d62728', linewidth=1.5, markersize=5, label=f'GNN Pred (MAE={gnn_mae:.2f})', zorder=2)
    ax.plot(timestamps, llm_seq, '^-', color='#2ca02c', linewidth=1.5, markersize=5, label=f'LLM Pred (MAE={llm_mae:.2f})', zorder=1)
    
    # 设置标题和标签
    feature_names = {
        0: 'Passenger Car Up',
        1: 'Passenger Car Down',
        2: 'Non-Passenger Car Up',
        3: 'Non-Passenger Car Down'
    }
    feature_name = feature_names.get(feature_id, f'Feature {feature_id}')
    
    title = f"Station {int(station_id):03d} - {row['station_short_name']}\n"
    title += f"{feature_name} | Sample {sample_idx} | {base_time.strftime('%Y-%m-%d %H:%M')}\n"
    title += f"LLM Improvement: {improvement:.2f} (GNN MAE={gnn_mae:.2f}, LLM MAE={llm_mae:.2f})"
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Traffic Volume', fontsize=11)
    
    # 设置图例
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # 设置时间轴格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45, ha='right')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    filename = f"station_{int(station_id):03d}_sample_{sample_idx}_feat_{feature_id}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ 已保存: {output_path}")
    return output_path

def plot_top_samples(df, top_stations, num_samples_per_station=3, output_dir=None):
    """为每个顶级站点绘制最佳样本"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    print(f"\n{'='*100}")
    print(f"开始绘制LLM优于GNN的最佳样本...")
    print(f"{'='*100}")
    
    # 筛选LLM更好的样本
    llm_better_df = df[df['llm_better'] == True].copy()
    
    generated_count = 0
    
    for _, station_row in top_stations.iterrows():
        station_id = int(station_row['station_id'])
        station_name = station_row['station_name']
        
        print(f"\n处理站点 {station_id:03d} - {station_name}")
        
        # 获取该站点LLM更好的样本
        station_samples = llm_better_df[llm_better_df['station_id'] == station_id]
        
        if len(station_samples) == 0:
            print(f"  ⚠️ 该站点没有LLM更好的样本")
            continue
        
        # 按改进幅度排序，取前N个
        top_samples = station_samples.nlargest(num_samples_per_station, 'mae_improvement')
        
        for _, sample_row in top_samples.iterrows():
            sample_idx = int(sample_row['sample_idx'])
            feature_id = int(sample_row['feature_id'])
            
            print(f"  绘制样本 {sample_idx}, 特征 {feature_id}, 改进幅度: {sample_row['mae_improvement']:.2f}")
            
            result = plot_comparison(df, station_id, sample_idx, feature_id, output_dir)
            if result:
                generated_count += 1
    
    print(f"\n{'='*100}")
    print(f"绘图完成！共生成 {generated_count} 张图片")
    print(f"输出目录: {output_dir}")
    print(f"{'='*100}")

def generate_summary_report(df, top_stations, output_dir=None):
    """生成汇总报告"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'llm_vs_gnn_summary.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("LLM vs GNN 预测效果对比分析报告\n")
        f.write("=" * 100 + "\n\n")
        
        # 总体统计
        total_samples = len(df)
        llm_better_count = df['llm_better'].sum()
        improvement_rate = llm_better_count / total_samples * 100
        
        f.write("【总体统计】\n")
        f.write(f"总样本数: {total_samples}\n")
        f.write(f"LLM优于GNN的样本数: {llm_better_count} ({improvement_rate:.2f}%)\n")
        f.write(f"GNN优于LLM的样本数: {total_samples - llm_better_count} ({100-improvement_rate:.2f}%)\n\n")
        
        # 平均改进幅度
        avg_improvement = df[df['llm_better']]['mae_improvement'].mean()
        max_improvement = df[df['llm_better']]['mae_improvement'].max()
        f.write(f"【改进幅度统计（仅LLM更好的样本）】\n")
        f.write(f"平均MAE改进: {avg_improvement:.2f}\n")
        f.write(f"最大MAE改进: {max_improvement:.2f}\n\n")
        
        # 按特征统计
        f.write("【按特征类型统计】\n")
        feature_stats = df.groupby('feature_name').agg({
            'llm_better': ['sum', 'count'],
            'mae_improvement': 'mean'
        }).reset_index()
        feature_stats.columns = ['feature', 'llm_better_count', 'total_count', 'avg_improvement']
        feature_stats['improvement_rate'] = feature_stats['llm_better_count'] / feature_stats['total_count'] * 100
        
        for _, row in feature_stats.iterrows():
            f.write(f"{row['feature']}:\n")
            f.write(f"  提升率: {row['improvement_rate']:.2f}% ({int(row['llm_better_count'])}/{int(row['total_count'])})\n")
            f.write(f"  平均改进: {row['avg_improvement']:.2f}\n\n")
        
        # Top站点列表
        f.write("【LLM表现最好的前20个站点】\n")
        f.write(f"{'站点ID':<10} {'站点名称':<30} {'提升率':<12} {'改进样本数':<12} {'平均改进':<12}\n")
        f.write("-" * 100 + "\n")
        
        top_20 = top_stations.nlargest(20, 'improvement_rate')
        for _, row in top_20.iterrows():
            f.write(f"{int(row['station_id']):<10} {row['station_name']:<30} "
                   f"{row['improvement_rate']:<11.1f}% "
                   f"{int(row['llm_better_count'])}/{int(row['total_count']):<11} "
                   f"{row['avg_improvement']:<12.2f}\n")
    
    print(f"\n✅ 汇总报告已保存: {report_path}")
    return report_path

if __name__ == "__main__":
    # 步骤1: 分析数据，找出LLM更好的样本
    df, top_stations = find_llm_better_samples()
    
    # 步骤2: 生成汇总报告
    generate_summary_report(df, top_stations)
    
    # 步骤3: 绘制Top站点的最佳样本
    plot_top_samples(df, top_stations, num_samples_per_station=5)
    
    print("\n" + "=" * 100)
    print("所有任务完成！")
    print("=" * 100)
