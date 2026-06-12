"""
诊断事件数据加载和匹配问题
python diagnose_event_matching.py
"""
import pandas as pd
import json
import os

print("="*80)
print("🔍 事件数据匹配诊断工具")
print("="*80)

# 文件路径
EVENTS_CSV_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/events_list_quan.csv"
MAPPING_JSON_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/station_name_mapping.json"

# 1. 加载映射字典
print("\n1️⃣  加载站点名称映射...")
with open(MAPPING_JSON_PATH, 'r', encoding='utf-8') as f:
    station_mapping = json.load(f)
print(f"✓ 已加载 {len(station_mapping)} 个站点映射")
print(f"  示例: {list(station_mapping.items())[:5]}")

# 2. 读取事件文件
print("\n2️⃣  读取事件文件...")
try:
    events_df = pd.read_csv(EVENTS_CSV_PATH, encoding='utf-8')
    print(f"✓ 成功加载 {len(events_df)} 条事件记录")
    print(f"  列名: {list(events_df.columns)}")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit(1)

# 3. 检查事件文件中的站点名称
print("\n3️⃣  事件文件中的站点名称统计:")
event_stations = events_df.iloc[:, 1].unique()  # 假设第2列是站点名
print(f"  事件文件中涉及的站点数: {len(event_stations)}")
print(f"  前10个站点: {list(event_stations)[:10]}")

# 4. 检查映射匹配情况
print("\n4️⃣  中英文映射匹配检查:")
matched_count = 0
unmatched_stations = []

for station in event_stations:
    if station in station_mapping:
        matched_count += 1
    else:
        unmatched_stations.append(station)

print(f"  ✓ 成功匹配: {matched_count}/{len(event_stations)} 个站点")
print(f"  ✗ 未匹配: {len(unmatched_stations)} 个站点")

if unmatched_stations:
    print(f"\n  未匹配的站点 (前20个):")
    for s in unmatched_stations[:20]:
        print(f"    - '{s}'")
    if len(unmatched_stations) > 20:
        print(f"    ... 还有 {len(unmatched_stations) - 20} 个")

# 5. 模拟事件查询
print("\n5️⃣  模拟事件查询测试:")
test_date = "2023-09-29"  # 中秋节
test_chinese_station = "黄兴"
test_english_station = station_mapping.get(test_chinese_station, "NOT_FOUND")

print(f"  测试日期: {test_date}")
print(f"  测试站点 (中文): {test_chinese_station}")
print(f"  测试站点 (英文): {test_english_station}")

# 查找该日期的事件
events_on_date = events_df[events_df.iloc[:, 0] == test_date]
print(f"  该日期的事件数: {len(events_on_date)}")

if len(events_on_date) > 0:
    print(f"  事件样例:")
    for _, row in events_on_date.head(3).iterrows():
        print(f"    - 站点: {row.iloc[1]}, 事件: {row.iloc[2][:50]}")

# 6. 构建事件映射字典（带中英文转换）
print("\n6️⃣  构建事件映射字典...")
events_df['日期'] = pd.to_datetime(events_df.iloc[:, 0]).dt.strftime('%Y-%m-%d')

event_dict_chinese = {}
event_dict_english = {}

for _, row in events_df.iterrows():
    date_key = row['日期']
    chinese_station = str(row.iloc[1]).strip()
    event_desc = str(row.iloc[2]).strip()
    
    # 中文版
    event_dict_chinese[(date_key, chinese_station)] = event_desc
    
    # 英文版（如果映射存在）
    if chinese_station in station_mapping:
        english_station = station_mapping[chinese_station]
        event_dict_english[(date_key, english_station)] = event_desc

print(f"  中文版事件映射: {len(event_dict_chinese)} 条")
print(f"  英文版事件映射: {len(event_dict_english)} 条")

# 7. 测试查询
print("\n7️⃣  测试查询:")
test_key_chinese = (test_date, test_chinese_station)
test_key_english = (test_date, test_english_station)

print(f"  查询 (中文): {test_key_chinese}")
print(f"    结果: {event_dict_chinese.get(test_key_chinese, 'NOT_FOUND')[:60] if test_key_chinese in event_dict_chinese else 'NOT_FOUND'}")

print(f"  查询 (英文): {test_key_english}")
print(f"    结果: {event_dict_english.get(test_key_english, 'NOT_FOUND')[:60] if test_key_english in event_dict_english else 'NOT_FOUND'}")

print("\n" + "="*80)
print("✅ 诊断完成")
print("="*80)
print("\n📝 建议:")
if len(event_dict_english) == 0:
    print("  ⚠️  英文版事件映射为空，需要先生成中英文映射文件")
    print("  运行: python convert_events_to_english.py")
elif len(unmatched_stations) > 0:
    print(f"  ⚠️  有 {len(unmatched_stations)} 个站点未找到映射")
    print("  请检查 station_name_mapping.json 是否包含所有站点")
else:
    print("  ✓ 事件映射正常，问题可能在其他地方")
