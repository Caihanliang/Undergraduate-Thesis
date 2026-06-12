import pandas as pd
import os

# 文件路径
input_csv = '/home/user/Downloads/cai/moment-main/moment-main/dataset/观测站小时交通量-9.csv'
output_csv = '/home/user/Downloads/cai/moment-main/moment-main/dataset/station_mapping.csv'

# 读取CSV文件
print(f"正在读取文件: {input_csv}")
df = pd.read_csv(input_csv)

# 提取唯一的观测站编号和名称组合
station_df = df[['观测站编号', '观测站名称']].drop_duplicates()

# 重置索引，创建station_id列（从0开始）
station_df = station_df.reset_index(drop=True)
station_df.index.name = 'station_id'
station_df = station_df.reset_index()

# 重命名列
station_df.columns = ['station_id', 'station_code', 'station_name']

# 按station_id排序
station_df = station_df.sort_values('station_id').reset_index(drop=True)

# 保存为CSV文件
station_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"\n✅ 站点映射文件已生成!")
print(f"输出路径: {output_csv}")
print(f"总站点数: {len(station_df)}")
print(f"\n前10个站点:")
print(station_df.head(10).to_string(index=False))
