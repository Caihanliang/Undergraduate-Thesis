"""
辅助脚本: 从事件文件和站点描述文件中自动提取中英文站点名,生成映射JSON
使用方法: python build_station_mapping.py
"""
import pandas as pd
import json
import os
import re

# 配置路径
EVENTS_CSV_PATH = '/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/events_list_quan中.csv'
STATION_DESC_PATH = '/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/station_list_hngs.txt'
OUTPUT_MAPPING_PATH = '/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/station_name_mapping.json'

print("="*80)
print("🚀 自动构建中英文站点名称映射")
print("="*80)

# 步骤1: 从事件文件中提取中文站点名(保持出现顺序,去重)
print("\n📋 步骤1: 从事件文件中提取中文站点名...")
try:
    # 尝试多种编码
    df_events = None
    for encoding in ['utf-8', 'gb18030', 'gbk']:
        try:
            df_events = pd.read_csv(EVENTS_CSV_PATH, encoding=encoding)
            print(f"✓ 使用 {encoding} 编码成功读取事件文件")
            break
        except UnicodeDecodeError:
            continue
    
    if df_events is None:
        # 容错读取
        with open(EVENTS_CSV_PATH, 'r', encoding='gb18030', errors='ignore') as f:
            import io
            csv_content = f.read()
            df_events = pd.read_csv(io.StringIO(csv_content))
        print(f"✓ 使用容错模式读取事件文件")
    
    # 提取唯一中文站点名(保持首次出现的顺序)
    station_col = df_events.columns[1]  # 假设第2列是站点名
    seen_stations = set()
    chinese_stations_ordered = []
    
    for station in df_events[station_col]:
        station_str = str(station).strip()
        if station_str and station_str != 'nan' and station_str not in seen_stations:
            seen_stations.add(station_str)
            chinese_stations_ordered.append(station_str)
    
    print(f"✓ 提取到 {len(chinese_stations_ordered)} 个唯一中文站点名(按首次出现顺序):")
    for i, station in enumerate(chinese_stations_ordered[:20], 1):
        print(f"  {i:3d}. '{station}'")
    if len(chinese_stations_ordered) > 20:
        print(f"  ... 还有 {len(chinese_stations_ordered) - 20} 个站点")
        
except Exception as e:
    print(f"❌ 读取事件文件失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 步骤2: 从站点描述文件中提取英文站点名(保持顺序)
print("\n📋 步骤2: 从站点描述文件中提取英文站点名...")
try:
    with open(STATION_DESC_PATH, 'r', encoding='utf-8') as f:
        english_stations = []
        for line in f:
            line = line.strip()
            if line:
                # 提取第一个单词作为英文名
                eng_name = line.split(' ')[0]
                english_stations.append(eng_name)
    
    print(f"✓ 提取到 {len(english_stations)} 个英文站点名:")
    for i, station in enumerate(english_stations[:20], 1):
        print(f"  {i:3d}. '{station}'")
    if len(english_stations) > 20:
        print(f"  ... 还有 {len(english_stations) - 20} 个站点")
        
except Exception as e:
    print(f"❌ 读取站点描述文件失败: {e}")
    exit(1)

# 步骤3: 基于位置顺序建立映射
print("\n📋 步骤3: 基于位置顺序建立中英文映射...")

# 确定最小长度,避免越界
min_len = min(len(chinese_stations_ordered), len(english_stations))
print(f"  中文站点数: {len(chinese_stations_ordered)}")
print(f"  英文站点数: {len(english_stations)}")
print(f"  将映射前 {min_len} 个站点")

# 建立映射
mapping = {}
for i in range(min_len):
    cn_name = chinese_stations_ordered[i]
    en_name = english_stations[i]
    mapping[cn_name] = en_name

print(f"\n✓ 成功建立 {len(mapping)} 个映射关系")
print(f"\n映射样例(前10个):")
for i, (cn, en) in enumerate(list(mapping.items())[:10], 1):
    print(f"  {i:2d}. '{cn}' -> '{en}'")

# 处理未匹配的站点
if len(chinese_stations_ordered) > min_len:
    print(f"\n⚠️ 警告: 有 {len(chinese_stations_ordered) - min_len} 个中文站点未在英文列表中找到对应:")
    for cn in chinese_stations_ordered[min_len:min_len+10]:
        print(f"  - '{cn}'")
    if len(chinese_stations_ordered) > min_len + 10:
        print(f"  ... 还有 {len(chinese_stations_ordered) - min_len - 10} 个")

if len(english_stations) > min_len:
    print(f"\n⚠️ 警告: 有 {len(english_stations) - min_len} 个英文站点未在中文列表中找到对应:")
    for en in english_stations[min_len:min_len+10]:
        print(f"  - '{en}'")
    if len(english_stations) > min_len + 10:
        print(f"  ... 还有 {len(english_stations) - min_len - 10} 个")

# 步骤4: 保存映射文件
print(f"\n📋 步骤4: 保存映射文件...")
with open(OUTPUT_MAPPING_PATH, 'w', encoding='utf-8') as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)

print(f"✓ 映射文件已保存到: {OUTPUT_MAPPING_PATH}")
print(f"✓ 共 {len(mapping)} 个映射关系")

# 步骤5: 生成 fine_gnn_vals.py 中的代码片段
print("\n" + "="*80)
print("📝 生成的映射代码(复制到 fine_gnn_vals.py 中):")
print("="*80)
print("\nchinese_to_english_station = {")
for cn, en in list(mapping.items())[:20]:
    print(f'    "{cn}": "{en}",')
if len(mapping) > 20:
    print(f"    # ... 还有 {len(mapping) - 20} 个映射(已从JSON文件加载)")
print("}")

print("\n" + "="*80)
print("✅ 完成!")
print("="*80)
print(f"\n下一步操作:")
print(f"1. 检查 {OUTPUT_MAPPING_PATH} 中的映射是否正确")
print(f"2. 如有错误,手动修正后保存")
print(f"3. 重新运行 fine_gnn_vals.py,将自动加载此JSON文件")
