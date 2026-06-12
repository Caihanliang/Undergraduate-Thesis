#!/usr/bin/env python3
"""
诊断微调代码中的站点匹配问题
"""

import os
import pandas as pd

PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/config/cai/"

# ============================== 1. 检查站点列表 ==============================
print("="*80)
print("📋 检查站点列表文件")
print("="*80)

station_file_98 = os.path.join(PROJECT_ROOT, "station_list_hngs_98.txt")
with open(station_file_98, 'r', encoding='utf-8') as f:
    stations_98 = [line.strip() for line in f if line.strip()]

print(f"\n✓ 站点列表文件: {station_file_98}")
print(f"  站点总数: {len(stations_98)}")
print(f"\n前20个站点:")
for i, station in enumerate(stations_98[:20]):
    # 提取第一个词作为站点名称
    s_name = station.split()[0] if station.split() else ""
    print(f"  索引 {i:3d}: 英文名称={s_name:20s} | 完整描述={station[:80]}...")

# ============================== 2. 检查天气文件命名 ==============================
print("\n" + "="*80)
print("🌤️  检查天气文件命名")
print("="*80)

weather_dir = os.path.join(PROJECT_ROOT, "98站点天气信息/")
if os.path.exists(weather_dir):
    weather_files = sorted(os.listdir(weather_dir))
    print(f"\n✓ 天气目录: {weather_dir}")
    print(f"  文件总数: {len(weather_files)}")
    print(f"\n前20个天气文件:")
    for wf in weather_files[:20]:
        # 提取前缀数字
        prefix = wf[:3] if len(wf) >= 3 else "???"
        try:
            idx = int(prefix)
            expected_station = stations_98[idx].split()[0] if idx < len(stations_98) else "N/A"
            print(f"  文件: {wf:50s} | 索引={idx:3d} | 预期站点={expected_station}")
        except:
            print(f"  文件: {wf:50s} | 索引=??? (无法解析)")
else:
    print(f"❌ 天气目录不存在: {weather_dir}")

# ============================== 3. 检查事件文件 ==============================
print("\n" + "="*80)
print("🎯 检查事件文件")
print("="*80)

events_file = os.path.join(PROJECT_ROOT, "events_list_quan.csv")
if os.path.exists(events_file):
    try:
        df_events = pd.read_csv(events_file, encoding='utf-8')
        print(f"\n✓ 事件文件: {events_file}")
        print(f"  列名: {list(df_events.columns)}")
        print(f"  总事件数: {len(df_events)}")
        
        if '站点名称' in df_events.columns:
            unique_stations = df_events['站点名称'].unique()
            print(f"\n  事件中涉及的站点 ({len(unique_stations)}个):")
            for station in sorted(unique_stations)[:20]:
                count = (df_events['站点名称'] == station).sum()
                print(f"    - {station:20s} ({count}条事件)")
            
            # 检查站点名称格式
            print(f"\n  站点名称样例:")
            for station in unique_stations[:10]:
                print(f"    '{station}' (长度={len(station)}, 类型={type(station).__name__})")
    except Exception as e:
        print(f"❌ 读取事件文件失败: {e}")
else:
    print(f"❌ 事件文件不存在: {events_file}")

# ============================== 4. 检查数据集CSV中的站点 ==============================
print("\n" + "="*80)
print("📊 检查数据集中的站点")
print("="*80)

dataset_path = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/DataPipeline/观测站小时交通量-9.csv"
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    stations_in_data = df[['观测站编号', '观测站名称']].drop_duplicates()
    stations_in_data = stations_in_data.sort_values('观测站编号').reset_index(drop=True)
    
    print(f"\n✓ 数据集文件: {dataset_path}")
    print(f"  站点总数: {len(stations_in_data)}")
    print(f"\n前20个站点:")
    for idx, row in stations_in_data.head(20).iterrows():
        print(f"  索引 {idx:3d}: 编号={row['观测站编号']:20s} | 名称={row['观测站名称']}")
    
    # 尝试匹配站点列表和数据集
    print(f"\n  尝试匹配站点列表与数据集:")
    matches = 0
    mismatches = []
    for i, station_desc in enumerate(stations_98[:min(20, len(stations_98))]):
        s_name_en = station_desc.split()[0].lower()
        # 在数据集中查找匹配的站点
        found = False
        for _, row in stations_in_data.iterrows():
            cn_name = row['观测站名称']
            # 简单的中英文映射检查（这里需要实际的映射表）
            if i < len(stations_in_data):
                data_station = stations_in_data.iloc[i]['观测站名称']
                print(f"    索引{i}: 列表='{s_name_en}' vs 数据集='{data_station}'")
                break
else:
    print(f"❌ 数据集文件不存在: {dataset_path}")

# ============================== 5. 总结与建议 ==============================
print("\n" + "="*80)
print("💡 诊断总结与建议")
print("="*80)

print("""
⚠️  关键检查点:

1. 【站点顺序一致性】
   - station_list_hngs_98.txt 的行号顺序
   - 天气文件的前缀数字 (000, 001, ...)
   - 数据集中站点的实际顺序
   → 这三者必须完全一致！

2. 【事件文件站点名称格式】
   - 事件文件中使用的是中文名称（如"黄兴"、"黄兴镇"）
   - 微调代码提取的是英文名称（如"Huangxing"）
   → 需要建立中英文映射，或修改事件文件格式

3. 【天气文件命名规范】
   - 当前格式: "000黄兴镇, 长沙县, _xxx_数据.csv"
   - 代码期望: 文件名以 "000" 开头即可
   → 当前格式应该可以正常工作

🔧 修复建议:

A. 如果站点顺序不一致:
   - 重新生成 station_list_hngs_98.txt，确保顺序与数据集一致
   - 或者重新排序天气文件的前缀数字

B. 如果事件匹配失败:
   - 方案1: 在 events_list_quan.csv 中添加英文站点名称列
   - 方案2: 创建中英文站点名称映射字典
   - 方案3: 修改微调代码，使用中文名称进行事件匹配

C. 验证方法:
   - 运行微调代码时，观察日志输出
   - 检查是否有大量 "Event: None" 的情况
   - 手动抽查几个站点的天气和事件是否正确
""")
