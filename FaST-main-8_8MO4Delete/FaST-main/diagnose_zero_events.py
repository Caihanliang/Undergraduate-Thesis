"""
诊断特殊事件样本为0的根本原因
检查事件描述内容和高峰匹配情况
python diagnose_zero_events.py
"""
import pandas as pd
import numpy as np
import json
import re
from chinese_calendar import is_workday

print("="*80)
print("🔍 特殊事件样本为0的深度诊断")
print("="*80)

# 文件路径
EVENTS_CSV_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/events_list_quan_en.csv"
MAPPING_JSON_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/station_name_mapping.json"
HIS_DATA_CSV_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/his_data_with_index.csv"
NATURAL_PATTERN_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/station_natural_list_4feat.txt"

# 1. 加载数据
print("\n1️⃣  加载基础数据...")
with open(MAPPING_JSON_PATH, 'r', encoding='utf-8') as f:
    station_mapping = json.load(f)

his_df = pd.read_csv(HIS_DATA_CSV_PATH)
his_df['时间'] = pd.to_datetime(his_df['时间'])
parking_data = his_df.set_index('时间').sort_index()

events_df = pd.read_csv(EVENTS_CSV_PATH, encoding='utf-8')
print(f"✓ 加载 {len(events_df)} 条事件记录")

with open(NATURAL_PATTERN_PATH, "r", encoding='utf-8') as f:
    natural_pattern_list = [l.strip() for l in f if l.strip()]
print(f"✓ 加载 {len(natural_pattern_list)} 条Pattern")

# 2. 分析事件描述内容
print("\n2️⃣  分析事件描述内容...")
event_descriptions = events_df.iloc[:, 2].astype(str).tolist()

has_time_count = 0
no_time_count = 0
time_samples = []
no_time_samples = []

for desc in event_descriptions[:50]:  # 采样前50条
    has_time = bool(re.search(r"\d{2}:\d{2}", desc))
    if has_time:
        has_time_count += 1
        if len(time_samples) < 5:
            time_samples.append(desc[:80])
    else:
        no_time_count += 1
        if len(no_time_samples) < 5:
            no_time_samples.append(desc[:80])

print(f"  包含时间的描述: {has_time_count}/{len(event_descriptions[:50])}")
print(f"  不包含时间的描述: {no_time_count}/{len(event_descriptions[:50])}")

if time_samples:
    print(f"\n  ✅ 包含时间的样例:")
    for s in time_samples:
        print(f"    - {s}")

if no_time_samples:
    print(f"\n  ❌ 不包含时间的样例:")
    for s in no_time_samples:
        print(f"    - {s}")

# 3. 检查日期重叠
print("\n3️⃣  检查日期范围重叠...")
event_dates = pd.to_datetime(events_df.iloc[:, 0]).dt.strftime('%Y-%m-%d').unique()
data_dates = parking_data.index.strftime('%Y-%m-%d').unique()

overlap_dates = set(event_dates) & set(data_dates)
print(f"  事件文件日期数: {len(event_dates)}")
print(f"  训练数据日期数: {len(data_dates)}")
print(f"  重叠日期数: {len(overlap_dates)}")

if overlap_dates:
    print(f"  重叠日期样例: {list(overlap_dates)[:5]}")
else:
    print(f"  ⚠️  无重叠日期！这是导致Event=0的主要原因")

# 4. 模拟高峰匹配测试
print("\n4️⃣  模拟高峰匹配测试...")
matched_stations = ['Hongqi', 'Langli', 'Dongjing', 'Xingcheng', 'Huangxing']
print(f"  匹配的站点: {matched_stations}")

# 随机选择一个匹配的站点和日期进行测试
test_station = matched_stations[0]
test_date = list(overlap_dates)[0] if overlap_dates else None

if test_date:
    print(f"\n  测试案例: 站点={test_station}, 日期={test_date}")
    
    # 查找该站点的事件
    station_events = events_df[events_df.iloc[:, 1] == test_station]
    date_events = station_events[pd.to_datetime(station_events.iloc[:, 0]).dt.strftime('%Y-%m-%d') == test_date]
    
    if len(date_events) > 0:
        event_desc = str(date_events.iloc[0, 2])
        print(f"  事件描述: {event_desc[:100]}")
        
        has_time = bool(re.search(r"\d{2}:\d{2}", event_desc))
        print(f"  包含时间: {'是' if has_time else '否'}")
        
        # 检查高峰匹配
        pattern_idx = 0  # 简化测试，使用第一个Pattern
        if pattern_idx < len(natural_pattern_list):
            pattern_str = natural_pattern_list[pattern_idx]
            pattern_part = pattern_str.split(';')[0]  # 工作日部分
            
            p_match = re.search(
                r"peak time ([\d:,\s]+), average peak flow ([\d:,\s]+), low-peak time ([\d:,\s]+), average low-peak flow ([\d:,\s]+)",
                pattern_part)
            
            if p_match:
                p_times = [t.strip() for t in p_match.group(1).split(',')]
                print(f"  Pattern中的高峰时间: {p_times[:5]}")
                
                # 检查事件描述中的时间是否在高峰时段
                times_in_event = re.findall(r"(\d{2}:\d{2})", event_desc)
                if times_in_event:
                    print(f"  事件中的时间: {times_in_event}")
                    matched_times = [t for t in times_in_event if t in p_times]
                    print(f"  命中高峰的时间: {matched_times}")
                else:
                    print(f"  事件中无具体时间")
    else:
        print(f"  ⚠️  该日期该站点无事件记录")

# 5. 统计事件类型分布
print("\n5️⃣  事件类型分布统计...")
event_keywords = {
    "演唱会": 0,
    "展览": 0,
    "会议": 0,
    "比赛": 0,
    "节日": 0,
    "其他": 0
}

for desc in event_descriptions:
    if "演唱会" in desc or "音乐会" in desc:
        event_keywords["演唱会"] += 1
    elif "展览" in desc or "展会" in desc:
        event_keywords["展览"] += 1
    elif "会议" in desc or "论坛" in desc:
        event_keywords["会议"] += 1
    elif "比赛" in desc or "赛事" in desc:
        event_keywords["比赛"] += 1
    elif "节日" in desc or "庆典" in desc:
        event_keywords["节日"] += 1
    else:
        event_keywords["其他"] += 1

print("  事件类型分布:")
for k, v in event_keywords.items():
    if v > 0:
        print(f"    - {k}: {v}")

# 6. 给出诊断结论
print("\n" + "="*80)
print("📊 诊断结论")
print("="*80)

issues = []
if len(overlap_dates) == 0:
    issues.append("❌ 严重：事件日期与训练数据日期完全不重叠")
elif len(overlap_dates) < 5:
    issues.append(f"⚠️  警告：重叠日期过少（仅{len(overlap_dates)}天）")

if no_time_count > has_time_count:
    issues.append("⚠️  大部分事件描述不包含具体时间，导致无法通过时间验证")

if len(matched_stations) < 10:
    issues.append(f"⚠️  站点名称匹配率低（仅{len(matched_stations)}/156个站点）")

if issues:
    print("\n发现的问题:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✅ 未发现明显问题")

print("\n💡 建议:")
print("  1. 如果事件描述不含时间，可以放宽判定条件，只要有事件记录就视为Event样本")
print("  2. 或者在事件文件中补充具体时间信息")
print("  3. 检查是否需要扩大站点名称映射范围")
