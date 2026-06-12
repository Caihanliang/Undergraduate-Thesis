import pandas as pd
import numpy as np
from datetime import datetime, timedelta

"""
创建示例数据模板，展示如何格式化你的原始数据
"""

# 创建模拟的湖南高速流量数据
print("正在生成示例数据模板...")

# 时间范围：2023 年 9 月 1 日 - 10 月 31 日（2 个月）
start_date = datetime(2023, 9, 1, 0, 0, 0)
end_date = datetime(2023, 10, 31, 23, 45, 0)

# 生成 15 分钟间隔的时间序列
time_range = pd.date_range(start=start_date, end=end_date, freq='15T')

# 创建 161 个站点名称（黄兴到芙蓉镇）
station_names = []
for i in range(1, 162):
    station_names.append(f"Station_{i:03d}")

# 生成模拟流量数据（带有日周期和周周期模式）
np.random.seed(42)
data = []

for t_idx, current_time in enumerate(time_range):
    hour = current_time.hour
    day_of_week = current_time.weekday()
    
    row = []
    for station_idx in range(161):
        # 基础流量
        base_flow = 100 + np.random.randn() * 10
        
        # 小时模式（早晚高峰）
        if 7 <= hour <= 9:  # 早高峰
            hour_factor = 2.5
        elif 17 <= hour <= 19:  # 晚高峰
            hour_factor = 2.8
        elif 12 <= hour <= 14:  # 午间
            hour_factor = 1.5
        elif 22 <= hour or hour <= 5:  # 夜间
            hour_factor = 0.3
        else:
            hour_factor = 1.0
        
        # 周末效应
        if day_of_week >= 5:  # 周六周日
            weekend_factor = 1.2
        else:
            weekend_factor = 1.0
        
        # 站点差异
        station_factor = 1.0 + (station_idx % 10) * 0.05
        
        # 计算最终流量
        flow = base_flow * hour_factor * weekend_factor * station_factor
        flow = max(0, flow)  # 流量不能为负
        
        row.append(flow)
    
    data.append(row)

# 创建 DataFrame
df = pd.DataFrame(data, columns=station_names)
df.insert(0, 'Time', time_range)

# 保存为 Excel 文件
output_excel = "hunan_highway_traffic_test.xlsx"
df.to_excel(output_excel, index=False)
print(f"✓ 已生成 Excel 模板：{output_excel}")

# 保存为 CSV 文件
output_csv = "hunan_highway_traffic_template.csv"
df.to_csv(output_csv, index=False)
print(f"✓ 已生成 CSV 模板：{output_csv}")

print(f"\n数据形状：{df.shape}")
print(f"时间范围：{start_date} 到 {end_date}")
print(f"站点数量：{len(station_names)}")
print(f"\n请将此模板作为参考，替换为你的真实数据！")
print("保持第一列为时间，后续列为各站点流量。")
