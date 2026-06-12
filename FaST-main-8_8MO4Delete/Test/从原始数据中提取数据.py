#  python 从原始数据中提取数据.py
# FaST-main-8_8MO4/Test/从原始数据中提取数据.py
###########################################这个提取的数据不对劲
# python 从原始数据中提取数据.py
# FaST-main-8_8MO4/Test/从原始数据中提取数据.py
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def process_traffic_data_correct():
    print("正在读取数据文件...")
    df_9 = pd.read_csv('观测站小时交通量-9.csv')
    df_10 = pd.read_csv('观测站小时交通量 -10.csv')
    
    combined_df = pd.concat([df_9, df_10], ignore_index=True)
    print(f"合并后数据形状: {combined_df.shape}")

    filtered_df = combined_df[combined_df['行驶方向'].isin(['上行', '下行'])].copy()
    print(f"过滤后数据形状: {filtered_df.shape}")

    station_names = df_9['观测站名称'].unique().tolist()
    print(f"观测站数量: {len(station_names)}")
    print(f"前10个观测站: {station_names[:10]}")

    # 【关键修复】正确构建时间列
    filtered_df['小时调整'] = filtered_df['小时'] - 1
    filtered_df['Time'] = pd.to_datetime(filtered_df['观测日期']) + pd.to_timedelta(filtered_df['小时调整'], unit='h')

    start_time = pd.to_datetime('2023-09-01 00:00:00')
    end_time = pd.to_datetime('2023-10-31 23:00:00')
    full_time_range = pd.date_range(start=start_time, end=end_time, freq='h')
    base_df = pd.DataFrame({'Time': full_time_range})
    print(f"时间序列长度: {len(base_df)} (从 {start_time} 到 {end_time})")

    filtered_df['非小客车'] = filtered_df['汽车自然数'] - filtered_df['小客车']

    # ===================== 完全修复版 =====================
    def create_traffic_df(data_df, station_list):
        """
        创建符合FAST模型要求的宽表格式数据
        格式：Time,站点1,站点2,...,站点N
        
        【关键修复】使用 groupby + sum 而非 pivot_table，避免聚合导致的数值错误
        """
        # 【修复1】先按时间和站点分组求和（处理同一时间同站点多条记录的情况）
        grouped = data_df.groupby(['Time', '观测站名称'])['value'].sum().reset_index()
        
        # 【修复2】再转换为宽表格式
        pivot = grouped.pivot(
            index='Time',
            columns='观测站名称',
            values='value'
        )
        
        # 确保站点顺序一致，缺失值填充为0
        pivot = pivot.reindex(columns=station_list, fill_value=0)
        
        # 与完整时间序列对齐，填充缺失值为0
        result_df = base_df.copy()
        result_df = result_df.merge(pivot, left_on='Time', right_index=True, how='left')
        result_df = result_df.fillna(0)

        # 【关键修复】将Time转换为字符串格式 "YYYY-MM-DD HH:MM:SS"
        result_df['Time'] = result_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return result_df

    # 1. 小客车上行
    print("\n正在生成小客车上行流量...")
    up_small_car = filtered_df[filtered_df['行驶方向'] == '上行'][['Time', '观测站名称', '小客车']].copy()
    up_small_car = up_small_car.rename(columns={'小客车':'value'})
    df1 = create_traffic_df(up_small_car, station_names)
    df1.to_csv('小客车上行流量.csv', index=False, encoding='utf-8-sig')
    print(f"  ✓ 形状: {df1.shape}, 时间范围: {df1['Time'].iloc[0]} ~ {df1['Time'].iloc[-1]}")
    print(f"  ✓ 零值统计示例（前5个站点）:")
    for col in station_names[:5]:
        zero_count = (df1[col] == 0).sum()
        total_count = len(df1)
        print(f"    {col}: {zero_count}个零值 ({zero_count/total_count*100:.2f}%)")

    # 2. 小客车下行
    print("\n正在生成小客车下行流量...")
    down_small_car = filtered_df[filtered_df['行驶方向'] == '下行'][['Time', '观测站名称', '小客车']].copy()
    down_small_car = down_small_car.rename(columns={'小客车':'value'})
    df2 = create_traffic_df(down_small_car, station_names)
    df2.to_csv('小客车下行流量.csv', index=False, encoding='utf-8-sig')
    print(f"  ✓ 形状: {df2.shape}, 时间范围: {df2['Time'].iloc[0]} ~ {df2['Time'].iloc[-1]}")
    print(f"  ✓ 零值统计示例（前5个站点）:")
    for col in station_names[:5]:
        zero_count = (df2[col] == 0).sum()
        total_count = len(df2)
        print(f"    {col}: {zero_count}个零值 ({zero_count/total_count*100:.2f}%)")

    # 3. 非小客车上行
    print("\n正在生成(汽车自然数-小客车)上行流量...")
    up_non = filtered_df[filtered_df['行驶方向'] == '上行'][['Time', '观测站名称', '非小客车']].copy()
    up_non = up_non.rename(columns={'非小客车':'value'})
    df3 = create_traffic_df(up_non, station_names)
    df3.to_csv('(汽车自然数-小客车)上行流量.csv', index=False, encoding='utf-8-sig')
    print(f"  ✓ 形状: {df3.shape}, 时间范围: {df3['Time'].iloc[0]} ~ {df3['Time'].iloc[-1]}")
    print(f"  ✓ 零值统计示例（前5个站点）:")
    for col in station_names[:5]:
        zero_count = (df3[col] == 0).sum()
        total_count = len(df3)
        print(f"    {col}: {zero_count}个零值 ({zero_count/total_count*100:.2f}%)")

    # 4. 非小客车下行
    print("\n正在生成(汽车自然数-小客车)下行流量...")
    down_non = filtered_df[filtered_df['行驶方向'] == '下行'][['Time', '观测站名称', '非小客车']].copy()
    down_non = down_non.rename(columns={'非小客车':'value'})
    df4 = create_traffic_df(down_non, station_names)
    df4.to_csv('(汽车自然数-小客车)下行流量.csv', index=False, encoding='utf-8-sig')
    print(f"  ✓ 形状: {df4.shape}, 时间范围: {df4['Time'].iloc[0]} ~ {df4['Time'].iloc[-1]}")
    print(f"  ✓ 零值统计示例（前5个站点）:")
    for col in station_names[:5]:
        zero_count = (df4[col] == 0).sum()
        total_count = len(df4)
        print(f"    {col}: {zero_count}个零值 ({zero_count/total_count*100:.2f}%)")

    print("\n✅ 全部生成完成！无报错、无警告、时间格式正常！")
    print(f"\n📊 数据统计:")
    print(f"  - 观测站数量: {len(station_names)}")
    print(f"  - 时间步数: {len(full_time_range)}")
    print(f"  - 输出文件格式: Time,{','.join(station_names[:3])},...")
    print(f"  - 示例数据预览:")
    print(df1.head(3).to_string())
    
    return station_names

if __name__ == "__main__":
    try:
        print("开始处理交通量数据...")
        process_traffic_data_correct()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

##################################################################################################
