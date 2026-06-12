"""
交通流非平稳波动趋势可视化脚本

该脚本用于可视化高速公路观测站数据中四个关键特征的日周期和节假日流量模式。
参考图片风格：深蓝色线条、圆点标记、清晰的时间段标签
所有文本使用拼音/英文，避免中文字符
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置字体 - 使用默认英文字体
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# 定义文件路径
DATA_FILE_9 = '/home/user/Downloads/cai/GMAN/dataset/观测站小时交通量-9.csv'
DATA_FILE_10 = '/home/user/Downloads/cai/GMAN/dataset/观测站小时交通量-10.csv'


def load_and_merge_data():
    """加载并合并9月和10月的数据"""
    print("正在加载数据...")
    
    # 读取两个月份的数据
    df_9 = pd.read_csv(DATA_FILE_9)
    df_10 = pd.read_csv(DATA_FILE_10)
    
    # 合并数据
    df = pd.concat([df_9, df_10], ignore_index=True)
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"时间范围: {df['观测日期'].min()} 至 {df['观测日期'].max()}")
    
    return df


def classify_time_period(row):
    """
    根据日期和时间分类时间段
    
    参数:
        row: DataFrame的一行，包含'观测日期'和'小时'列
    
    返回:
        str: 时间段分类 (使用拼音)
    """
    date_str = row['观测日期']
    hour = row['小时']
    
    # 解析日期
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    except:
        date_obj = pd.to_datetime(date_str)
    
    # 判断是否为周末或节假日
    weekday = date_obj.weekday()  # 0=周一, 6=周日
    
    # 定义2023年9-10月的法定节假日
    holidays = [
        '2023-09-29',  # 中秋节
        '2023-09-30',  # 中秋调休
        '2023-10-01', '2023-10-02', '2023-10-03', 
        '2023-10-04', '2023-10-05', '2023-10-06',  # 国庆节
    ]
    
    date_str_formatted = date_obj.strftime('%Y-%m-%d')
    
    # 判断是否为节假日（周末或法定假日）
    is_holiday = (weekday >= 5) or (date_str_formatted in holidays)
    
    if is_holiday:
        return 'Holiday'  # 节假日 -> Holiday
    
    # 工作日的时间段分类（使用拼音）
    if 7 <= hour < 9:
        return 'Morning_Peak'  # 早高峰
    elif 9 <= hour < 12:
        return 'Off_Peak'  # 平峰
    elif 12 <= hour < 14:
        return 'Noon'  # 午间
    elif 14 <= hour < 17:
        return 'Off_Peak'  # 平峰
    elif 17 <= hour < 19:
        return 'Evening_Peak'  # 晚高峰
    elif 19 <= hour < 23:
        return 'Off_Peak'  # 平峰
    else:  # 23:00-6:00
        return 'Night'  # 夜间


def calculate_feature_averages(df):
    """
    计算四个特征在各个时间段的平均值
    
    参数:
        df: 原始数据DataFrame
    
    返回:
        dict: 包含四个特征在各时间段平均值的字典
    """
    print("\n正在计算各时间段的平均流量...")
    
    # 添加时间段分类列
    df['Time_Period'] = df.apply(classify_time_period, axis=1)
    
    # 计算非小客车数量
    df['Non_Passenger_Car'] = df['汽车自然数'] - df['小客车']
    
    # 定义四个特征（使用拼音命名）
    features = {
        'Passenger_Up': (df['行驶方向'] == '上行') & (df['小客车'] > 0),
        'Passenger_Down': (df['行驶方向'] == '下行') & (df['小客车'] > 0),
        'Non_Passenger_Up': (df['行驶方向'] == '上行') & (df['Non_Passenger_Car'] > 0),
        'Non_Passenger_Down': (df['行驶方向'] == '下行') & (df['Non_Passenger_Car'] > 0),
    }
    
    # 定义时间段顺序（使用拼音）
    time_periods_order = ['Off_Peak', 'Morning_Peak', 'Noon', 'Evening_Peak', 'Night', 'Holiday']
    
    results = {}
    
    for feature_name, condition in features.items():
        # 筛选对应特征的数据
        feature_df = df[condition].copy()
        
        if len(feature_df) == 0:
            print(f"警告: {feature_name} 没有数据")
            continue
        
        # 提取对应的流量值
        if 'Passenger' in feature_name and 'Non' not in feature_name:
            feature_df['Flow'] = feature_df['小客车']
        else:
            feature_df['Flow'] = feature_df['Non_Passenger_Car']
        
        # 按时间段分组计算平均值
        avg_by_period = feature_df.groupby('Time_Period')['Flow'].mean()
        
        # 按照指定顺序重新排列
        avg_ordered = avg_by_period.reindex(time_periods_order)
        
        results[feature_name] = avg_ordered
    
    return results


def plot_traffic_patterns(results):
    """
    绘制交通流非平稳波动趋势图（分特征展示）
    
    参数:
        results: 包含四个特征在各时间段平均值的字典
    """
    print("\n正在生成可视化图表...")
    
    # 定义时间段标签（使用拼音）
    time_periods = ['Off_Peak', 'Morning_Peak', 'Noon', 'Evening_Peak', 'Night', 'Holiday']
    x_labels = ['Off-Peak', 'Morning\nPeak', 'Noon', 'Evening\nPeak', 'Night', 'Holiday']
    
    # 创建图形 - 使用子图分别展示四个特征
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Traffic Flow Non-Stationary Fluctuation Trends', fontsize=18, fontweight='bold', color='#1a5f7a')
    
    # 定义颜色和样式
    colors = ['#1e5f8e', '#2ecc71', '#e74c3c', '#f39c12']
    feature_names = list(results.keys())
    
    # 特征名称映射（用于显示）
    feature_display_names = {
        'Passenger_Up': 'Passenger Car (Upstream)',
        'Passenger_Down': 'Passenger Car (Downstream)',
        'Non_Passenger_Up': 'Non-Passenger Car (Upstream)',
        'Non_Passenger_Down': 'Non-Passenger Car (Downstream)'
    }
    
    for idx, (ax, feature_name) in enumerate(zip(axes.flat, feature_names)):
        if feature_name not in results:
            ax.text(0.5, 0.5, f'{feature_name}\nNo Data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # 获取数据
        data = results[feature_name]
        values = [data.get(period, 0) for period in time_periods]
        
        # 绘制折线图
        x_pos = range(len(time_periods))
        ax.plot(x_pos, values, marker='o', markersize=8, linewidth=2.5, 
               color=colors[idx], markerfacecolor=colors[idx])
        
        # 设置标题和标签
        display_name = feature_display_names.get(feature_name, feature_name)
        ax.set_title(display_name, fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Average Flow (vehicles/hour)', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置Y轴范围（自动调整但留有余量）
        max_val = max(values) if values else 0
        min_val = min(values) if values else 0
        if max_val > 0:
            ax.set_ylim(0, max_val * 1.2)
        
        # 在点上标注数值
        for i, val in enumerate(values):
            ax.annotate(f'{val:.0f}', xy=(x_pos[i], val), 
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/user/Downloads/cai/traffic_flow_patterns.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    print("图表已保存至: /home/user/Downloads/cai/traffic_flow_patterns.png")
    plt.show()


def plot_combined_traffic_patterns(results):
    """
    绘制综合对比图 - 所有特征在同一图中
    
    参数:
        results: 包含四个特征在各时间段平均值的字典
    """
    print("\n正在生成综合对比图...")
    
    time_periods = ['Off_Peak', 'Morning_Peak', 'Noon', 'Evening_Peak', 'Night', 'Holiday']
    x_labels = ['Off-Peak', 'Morning\nPeak', 'Noon', 'Evening\nPeak', 'Night', 'Holiday']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 定义颜色和线型
    styles = [
        {'color': '#1e5f8e', 'linestyle': '-', 'marker': 'o', 'label': 'Passenger Up'},
        {'color': '#2ecc71', 'linestyle': '--', 'marker': 's', 'label': 'Passenger Down'},
        {'color': '#e74c3c', 'linestyle': '-.', 'marker': '^', 'label': 'Non-Passenger Up'},
        {'color': '#f39c12', 'linestyle': ':', 'marker': 'd', 'label': 'Non-Passenger Down'},
    ]
    
    x_pos = range(len(time_periods))
    
    for idx, style in enumerate(styles):
        feature_name = style['label'].replace(' ', '_').replace('-', '_')
        # 映射回实际的feature_name
        mapping = {
            'Passenger_Up': 'Passenger_Up',
            'Passenger_Down': 'Passenger_Down',
            'Non_Passenger_Up': 'Non_Passenger_Up',
            'Non_Passenger_Down': 'Non_Passenger_Down'
        }
        
        actual_feature = None
        for key in mapping:
            if key.replace('_', '') == feature_name.replace('_', ''):
                actual_feature = key
                break
        
        if actual_feature and actual_feature in results:
            data = results[actual_feature]
            values = [data.get(period, 0) for period in time_periods]
            
            ax.plot(x_pos, values, 
                   marker=style['marker'], markersize=8, 
                   linewidth=2.5, linestyle=style['linestyle'],
                   color=style['color'], label=style['label'])
    
    # 设置标题和标签
    ax.set_title('Comprehensive Comparison of Traffic Flow Patterns', fontsize=18, fontweight='bold', 
                color='#1a5f7a', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('Average Flow (vehicles/hour)', fontsize=13)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 自动调整Y轴范围
    all_values = []
    for feature_name in results:
        data = results[feature_name]
        values = [data.get(period, 0) for period in time_periods]
        all_values.extend(values)
    
    if all_values:
        max_val = max(all_values)
        ax.set_ylim(0, max_val * 1.15)
    
    plt.tight_layout()
    plt.savefig('/home/user/Downloads/cai/traffic_flow_patterns_combined.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    print("综合对比图已保存至: /home/user/Downloads/cai/traffic_flow_patterns_combined.png")
    plt.show()


def print_statistics(results):
    """打印统计信息（使用拼音）"""
    print("\n" + "="*80)
    print("Average Traffic Flow by Time Period")
    print("="*80)
    
    time_periods = ['Off_Peak', 'Morning_Peak', 'Noon', 'Evening_Peak', 'Night', 'Holiday']
    period_display = {
        'Off_Peak': 'Off-Peak',
        'Morning_Peak': 'Morning Peak',
        'Noon': 'Noon',
        'Evening_Peak': 'Evening Peak',
        'Night': 'Night',
        'Holiday': 'Holiday'
    }
    
    # 特征名称映射
    feature_display_names = {
        'Passenger_Up': 'Passenger Car (Upstream)',
        'Passenger_Down': 'Passenger Car (Downstream)',
        'Non_Passenger_Up': 'Non-Passenger Car (Upstream)',
        'Non_Passenger_Down': 'Non-Passenger Car (Downstream)'
    }
    
    for feature_name in results:
        print(f"\n[{feature_display_names.get(feature_name, feature_name)}]")
        print("-" * 60)
        data = results[feature_name]
        
        for period in time_periods:
            value = data.get(period, 0)
            display_period = period_display.get(period, period)
            print(f"  {display_period:15s}: {value:10.2f} vehicles/hour")
        
        # 计算峰值比
        peak_value = max(data.dropna().values)  # 修复：使用 .values 而非 .values()
        off_peak_value = data.get('Off_Peak', 0)
        if off_peak_value > 0:
            ratio = peak_value / off_peak_value
            print(f"\n  Peak/Off-Peak Ratio: {ratio:.2f}")
    
    print("\n" + "="*80)


def main():
    """主函数"""
    print("="*80)
    print("Traffic Flow Non-Stationary Fluctuation Visualization")
    print("="*80)
    
    # 1. 加载数据
    df = load_and_merge_data()
    
    # 2. 计算各特征的平均值
    results = calculate_feature_averages(df)
    
    # 3. 打印统计信息
    print_statistics(results)
    
    # 4. 生成可视化图表
    plot_traffic_patterns(results)
    plot_combined_traffic_patterns(results)
    
    print("\n✅ All charts generated successfully!")
    print("Output files:")
    print("  - /home/user/Downloads/cai/traffic_flow_patterns.png (separate features)")
    print("  - /home/user/Downloads/cai/traffic_flow_patterns_combined.png (combined comparison)")


if __name__ == '__main__':
    main()
