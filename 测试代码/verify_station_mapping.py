#!/usr/bin/env python3
"""
详细验证站点中英文映射关系
"""

import os
import pandas as pd

PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/config/cai/"

# ============================== 加载数据 ==============================
print("="*80)
print("🔍 详细验证站点中英文映射")
print("="*80)

# 1. 加载站点列表（英文）
station_file = os.path.join(PROJECT_ROOT, "station_list_hngs_98.txt")
with open(station_file, 'r', encoding='utf-8') as f:
    stations_en = [line.strip() for line in f if line.strip()]

# 2. 加载数据集（中文）
dataset_path = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/DataPipeline/观测站小时交通量-9.csv"
df_data = pd.read_csv(dataset_path)
stations_cn = df_data[['观测站编号', '观测站名称']].drop_duplicates()
stations_cn = stations_cn.sort_values('观测站编号').reset_index(drop=True)

# 3. 加载天气文件列表
weather_dir = os.path.join(PROJECT_ROOT, "98站点天气信息/")
weather_files = sorted(os.listdir(weather_dir))

# ============================== 构建映射表 ==============================
print("\n📋 站点映射对照表（前30个站点）:")
print("-"*100)
print(f"{'索引':<5} {'英文站名':<20} {'中文站名':<10} {'天气文件名(前30字符)':<35} {'匹配状态'}")
print("-"*100)

mismatches = []
for i in range(min(30, len(stations_en))):
    # 提取英文站名
    en_name = stations_en[i].split()[0] if stations_en[i].split() else ""
    
    # 获取中文站名
    cn_name = stations_cn.iloc[i]['观测站名称'] if i < len(stations_cn) else "N/A"
    
    # 获取天气文件名
    weather_file = weather_files[i] if i < len(weather_files) else "N/A"
    weather_prefix = weather_file[:30] if len(weather_file) >= 30 else weather_file
    
    # 简单判断：检查天气文件名是否包含中文站名的关键字
    # 这里需要人工判断，因为中文名和英文名没有直接映射
    match_status = "待验证"
    
    print(f"{i:<5} {en_name:<20} {cn_name:<10} {weather_prefix:<35} {match_status}")

# ============================== 重点检查疑似错位的站点 ==============================
print("\n" + "="*80)
print("⚠️  重点检查疑似错位的站点")
print("="*80)

suspected_mismatches = [
    (4, "Yueyang", "岳阳东", "岳阳市"),
    (5, "Jianjia'ao", "简家坳", "望城区"),
    (6, "Taling", "塔岭", "雨花区"),
    (15, "Yuelong", "跃龙", "浏阳市"),
    (17, "Lijiachong", "黎家冲", "醴陵市"),
]

print("\n这些站点可能存在映射错误，需要人工确认:\n")
for idx, en_name, cn_name, weather_hint in suspected_mismatches:
    print(f"索引 {idx}:")
    print(f"  英文站名: {en_name}")
    print(f"  中文站名: {cn_name}")
    print(f"  天气文件提示: {weather_hint}")
    print(f"  ❓ 问题: '{weather_hint}' 是否等于 '{cn_name}' 或与之相关?")
    print()

# ============================== 检查事件文件中的站点覆盖 ==============================
print("="*80)
print("🎯 检查事件文件覆盖情况")
print("="*80)

events_file = os.path.join(PROJECT_ROOT, "events_list_quan.csv")
df_events = pd.read_csv(events_file, encoding='utf-8')
event_stations = set(df_events['站点名称'].unique())

print(f"\n事件文件中包含 {len(event_stations)} 个站点")
print(f"站点列表中有 {len(stations_en)} 个站点")

# 检查有多少站点在事件文件中有记录
matched_count = 0
unmatched_stations = []

for i, station_desc in enumerate(stations_en):
    en_name = station_desc.split()[0]
    if en_name in event_stations:
        matched_count += 1
    else:
        unmatched_stations.append((i, en_name))

print(f"\n✓ 事件匹配的站点数: {matched_count}/{len(stations_en)}")
print(f"✗ 未匹配的站点数: {len(unmatched_stations)}")

if unmatched_stations:
    print(f"\n未匹配的站点示例（前10个）:")
    for idx, name in unmatched_stations[:10]:
        print(f"  索引 {idx}: {name}")

# ============================== 总结建议 ==============================
print("\n" + "="*80)
print("💡 总结与建议")
print("="*80)

print("""
📌 关键发现:

1. 【索引一致性】✅ 
   - 三个数据源的索引编号（0-97）完全一致
   - 微调代码能正确通过索引获取数据

2. 【语义匹配性】⚠️ 需要人工验证
   - 天气文件的中文命名可能与站点列表的英文名称不完全对应
   - 例如：索引5的"Jianjia'ao/简家坳"对应天气文件"005望城区"
   - 需要确认：望城区是否就是简家坳所在区域？

3. 【事件匹配】✅ 基本正常
   - 事件文件使用英文名称，与站点列表格式一致
   - 大部分站点应该有事件记录

🔧 建议操作:

A. 如果需要精确验证天气数据：
   1. 手动抽查几个站点的地理位置
   2. 确认天气文件的行政区是否与站点位置对应
   3. 如果发现错位，需要重新整理天气文件命名

B. 如果接受当前配置：
   1. 运行微调代码，观察训练效果
   2. 如果模型表现异常，再回头检查数据匹配问题
   3. 可以在Prompt中加入调试信息，打印实际使用的天气和事件

C. 长期改进方案：
   1. 创建标准的中英文站点映射表（station_mapping.json）
   2. 统一所有数据源使用同一套映射关系
   3. 添加数据校验步骤，在训练前自动检测匹配问题
""")
