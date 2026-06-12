"""
合并Lag-Llama四个特征的预测结果

功能:
1. 读取4个CSV文件(小客车上行/下行,非小客车上行/下行)
2. 按station_id和timestamp进行数据对齐
3. 创建宽表格式,每个时间戳一行,包含4个特征的真实值和预测值
4. 只保留前850个小时的数据
5. 输出到all_features_combined.csv
"""

import pandas as pd
import os
from pathlib import Path

# 配置
RESULTS_DIR = Path('/home/user/Downloads/cai/lag-llama-main/prediction_results')
MAX_HOURS = 850  # 只保留前850个小时的数据

# 特征映射
FEATURE_MAP = {
    'passenger_up_predictions.csv': (0, '小客车上行'),
    'passenger_down_predictions.csv': (1, '小客车下行'),
    'non_passenger_up_predictions.csv': (2, '非小客车上行'),
    'non_passenger_down_predictions.csv': (3, '非小客车下行')
}

print("=" * 80)
print("Lag-Llama 预测结果合并")
print("=" * 80)

# 1. 读取所有CSV文件
print("\n正在读取CSV文件...")
feature_dfs = {}

for filename, (feat_id, feat_name) in FEATURE_MAP.items():
    filepath = RESULTS_DIR / filename
    print(f"  读取: {filename} (特征{feat_id}: {feat_name})")
    
    df = pd.read_csv(filepath)
    print(f"    原始数据: {len(df)} 行")
    
    # 只保留需要的列
    df = df[['station_id', 'timestamp', 'hour_offset', 'true_value', 'pred_value']].copy()
    
    # 重命名列
    df = df.rename(columns={
        'true_value': f'true_value_{feat_id}',
        'pred_value': f'pred_value_{feat_id}'
    })
    
    feature_dfs[feat_id] = df

# 2. 合并所有特征数据
print("\n正在合并数据...")

# 以第一个特征为基础
result_df = feature_dfs[0].copy()

# 依次合并其他特征
for feat_id in [1, 2, 3]:
    feat_df = feature_dfs[feat_id][['station_id', 'timestamp', 'hour_offset', 
                                     f'true_value_{feat_id}', f'pred_value_{feat_id}']]
    
    result_df = result_df.merge(
        feat_df,
        on=['station_id', 'timestamp', 'hour_offset'],
        how='outer'
    )

print(f"合并后总行数: {len(result_df)}")

# 3. 按时间排序
print("\n正在排序...")
result_df = result_df.sort_values(['station_id', 'timestamp', 'hour_offset']).reset_index(drop=True)

# 4. 只保留前850个小时的数据
print(f"\n正在筛选前{MAX_HOURS}个小时的数据...")

# 获取所有唯一的时间戳(按时间排序)
unique_timestamps = result_df['timestamp'].unique()
unique_timestamps_sorted = sorted(unique_timestamps)

if len(unique_timestamps_sorted) > MAX_HOURS:
    # 只保留前850个时间戳
    timestamps_to_keep = set(unique_timestamps_sorted[:MAX_HOURS])
    result_df = result_df[result_df['timestamp'].isin(timestamps_to_keep)].copy()
    print(f"  从 {len(unique_timestamps_sorted)} 小时筛选到 {MAX_HOURS} 小时")
else:
    print(f"  总时长 {len(unique_timestamps_sorted)} 小时 <= {MAX_HOURS} 小时,无需筛选")

# 5. 保存结果
output_file = RESULTS_DIR / 'all_features_combined.csv'
print(f"\n正在保存结果到: {output_file}")

result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✓ 合并完成!")
print(f"  输出文件: {output_file}")
print(f"  总行数: {len(result_df)}")
print(f"  站点数: {result_df['station_id'].nunique()}")
print(f"  时间跨度: {result_df['timestamp'].nunique()} 小时")
print(f"\n列结构:")
print(f"  station_id, timestamp, hour_offset")
print(f"  true_value_0, pred_value_0 (小客车上行)")
print(f"  true_value_1, pred_value_1 (小客车下行)")
print(f"  true_value_2, pred_value_2 (非小客车上行)")
print(f"  true_value_3, pred_value_3 (非小客车下行)")

# 显示前几行数据
print(f"\n前5行数据预览:")
print(result_df.head().to_string())