"""
预测结果可视化脚本

功能:
1. 加载预测结果CSV文件
2. 为每个站点和特征生成预测vs真实值对比图
3. 包含置信区间展示
4. 保存为PNG图片
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from tqdm import tqdm

# ==================== 配置参数 ====================
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "prediction_results"
VISUALIZATION_DIR = RESULTS_DIR / "visualizations"

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 特征定义
FEATURES = {
    'passenger_up': '小客车上行',
    'passenger_down': '小客车下行',
    'non_passenger_up': '非小客车上行',
    'non_passenger_down': '非小客车下行'
}


def load_prediction_results():
    """加载所有特征的预测结果"""
    print("正在加载预测结果...")
    all_data = {}
    
    for feature_key in FEATURES.keys():
        file_path = RESULTS_DIR / f"{feature_key}_predictions.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            all_data[feature_key] = df
            print(f"  已加载: {feature_key}, 记录数: {len(df)}")
        else:
            print(f"  警告: 文件不存在 - {file_path}")
    
    return all_data


def plot_single_series(ax, df_series, title, show_confidence=True):
    """
    绘制单个序列的预测vs真实值
    
    Args:
        ax: matplotlib轴对象
        df_series: 单个序列的数据
        title: 图表标题
        show_confidence: 是否显示置信区间
    """
    timestamps = df_series['timestamp'].values
    true_values = df_series['true_value'].values
    pred_values = df_series['pred_value'].values
    
    # 绘制真实值
    if not np.all(np.isnan(true_values)):
        ax.plot(timestamps, true_values, 'o-', label='真实值', color='#2E86AB', linewidth=2, markersize=6)
    
    # 绘制预测值
    ax.plot(timestamps, pred_values, 's--', label='预测值', color='#A23B72', linewidth=2, markersize=6)
    
    # 绘制置信区间
    if show_confidence and 'pred_lower_10' in df_series.columns:
        lower = df_series['pred_lower_10'].values
        upper = df_series['pred_upper_90'].values
        ax.fill_between(timestamps, lower, upper, alpha=0.2, color='#A23B72', label='90%置信区间')
    
    # 设置标签和标题
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('时间', fontsize=10)
    ax.set_ylabel('流量', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 格式化x轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.xticks(rotation=45)


def visualize_by_feature(all_data, max_stations=5):
    """
    按特征可视化预测结果
    
    Args:
        all_data: 所有特征的预测数据字典
        max_stations: 每个特征最多可视化的站点数
    """
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    
    for feature_key, feature_name in FEATURES.items():
        if feature_key not in all_data:
            continue
        
        df = all_data[feature_key]
        
        # 获取所有站点
        stations = df['station_id'].unique()[:max_stations]
        
        print(f"\n正在可视化: {feature_name}")
        print(f"  站点数: {len(stations)}")
        
        # 为每个站点创建图表
        for station_id in tqdm(stations, desc=f"处理{feature_name}"):
            station_data = df[df['station_id'] == station_id]
            
            # 按window_idx分组(每个窗口一个子图)
            windows = station_data['window_idx'].unique()
            
            if len(windows) == 0:
                continue
            
            # 只可视化前3个窗口
            windows_to_plot = windows[:3]
            
            fig, axes = plt.subplots(len(windows_to_plot), 1, figsize=(14, 4*len(windows_to_plot)))
            if len(windows_to_plot) == 1:
                axes = [axes]
            
            for idx, window_idx in enumerate(windows_to_plot):
                window_data = station_data[station_data['window_idx'] == window_idx]
                
                if len(window_data) == 0:
                    continue
                
                title = f"{feature_name} - 站点: {station_id}\n窗口: {window_idx}"
                plot_single_series(axes[idx], window_data, title)
            
            plt.tight_layout()
            
            # 保存图片
            output_file = VISUALIZATION_DIR / f"{feature_key}_station_{station_id}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  可视化结果已保存至: {VISUALIZATION_DIR}")


def create_summary_plots(all_data):
    """
    创建汇总图表: 所有站点的平均预测效果
    """
    print("\n正在创建汇总图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (feature_key, feature_name) in enumerate(FEATURES.items()):
        if feature_key not in all_data:
            continue
        
        df = all_data[feature_key]
        
        # 计算每个时间步的平均真实值和预测值
        summary = df.groupby('hour_offset').agg({
            'true_value': 'mean',
            'pred_value': 'mean',
            'pred_lower_10': 'mean',
            'pred_upper_90': 'mean'
        }).reset_index()
        
        ax = axes[idx]
        
        hours = summary['hour_offset'].values
        true_vals = summary['true_value'].values
        pred_vals = summary['pred_value'].values
        lower_vals = summary['pred_lower_10'].values
        upper_vals = summary['pred_upper_90'].values
        
        # 绘制
        ax.plot(hours, true_vals, 'o-', label='平均真实值', color='#2E86AB', linewidth=2)
        ax.plot(hours, pred_vals, 's--', label='平均预测值', color='#A23B72', linewidth=2)
        ax.fill_between(hours, lower_vals, upper_vals, alpha=0.2, color='#A23B72', label='90%置信区间')
        
        ax.set_title(f'{feature_name} - 所有站点平均', fontsize=12, fontweight='bold')
        ax.set_xlabel('预测步长 (小时)', fontsize=10)
        ax.set_ylabel('流量', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(hours)))
    
    plt.tight_layout()
    
    output_file = VISUALIZATION_DIR / "summary_all_features.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"汇总图表已保存至: {output_file}")


def main():
    """主函数"""
    print("=" * 80)
    print("Lag-Llama 预测结果可视化")
    print("=" * 80)
    
    # 检查结果目录
    if not RESULTS_DIR.exists():
        print(f"错误: 结果目录不存在 - {RESULTS_DIR}")
        print("请先运行预测脚本: python scripts/predict_with_lag_llama.py")
        return
    
    # 加载数据
    all_data = load_prediction_results()
    
    if not all_data:
        print("错误: 未找到任何预测结果文件!")
        return
    
    # 按特征可视化
    visualize_by_feature(all_data, max_stations=5)
    
    # 创建汇总图表
    create_summary_plots(all_data)
    
    print("\n" + "=" * 80)
    print("可视化完成!")
    print(f"所有图表已保存至: {VISUALIZATION_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
