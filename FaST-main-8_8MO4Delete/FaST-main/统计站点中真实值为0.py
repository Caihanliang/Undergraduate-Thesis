"""
python 统计站点中真实值为0.py

"""
import pandas as pd
import os

# 文件路径
file_path = '/home/user/Downloads/cai/结果汇总/fast/visulization-result-4FEAT-train/all_nodes_prediction.csv'

# 读取CSV文件
print("正在读取CSV文件...")
df = pd.read_csv(file_path)

# 查找真实值为0的记录
zero_records = df[df['真实值'] == 0]

print(f"\n总记录数: {len(df)}")
print(f"真实值为0的记录数: {len(zero_records)}")

if len(zero_records) == 0:
    print("\n没有找到真实值为0的记录！")
else:
    # 按站点分组统计
    print("\n" + "="*80)
    print("真实值为0的站点统计")
    print("="*80)
    
    # 获取有0值的站点列表
    stations_with_zero = zero_records.groupby('站点名称')
    
    print(f"\n共有 {len(stations_with_zero)} 个站点存在真实值为0的情况\n")
    
    # 详细输出每个站点的0值情况
    for station_name, group in stations_with_zero:
        print(f"\n{'-'*80}")
        print(f"站点: {station_name}")
        print(f"{'-'*80}")
        print(f"真实值为0的次数: {len(group)}")
        print(f"\n具体时间点和特征:")
        print(f"{'时间':<25} {'特征':<25} {'真实值':<10} {'预测值':<10}")
        print("-" * 70)
        
        for _, row in group.iterrows():
            print(f"{row['时间']:<25} {row['特征']:<25} {row['真实值']:<10.1f} {row['预测值']:<10.2f}")
    
    # 保存结果到文件
    output_file = '/home/user/Downloads/cai/结果汇总/fast/visulization-result-4FEAT-train/zero_value_stations.csv'
    zero_records.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n{'='*80}")
    print(f"结果已保存到: {output_file}")
    print(f"{'='*80}")
    
    # 统计每个站点的0值次数
    print("\n各站点真实值为0的次数统计:")
    print(f"{'站点名称':<15} {'0值次数':<10}")
    print("-" * 30)
    station_counts = zero_records.groupby('站点名称').size().sort_values(ascending=False)
    for station, count in station_counts.items():
        print(f"{station:<15} {count:<10}")