"""
辅助脚本：从事件文件中提取中文站点名，并生成中英文映射模板
使用方法: python generate_station_mapping.py
"""
import pandas as pd
import json

# 读取事件文件
events_path = '/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/events_list_quan.csv'
station_desc_path = '/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/station_list_hngs.txt'

print("="*60)
print("步骤1: 从事件文件中提取中文站点名")
print("="*60)

try:
    df_events = pd.read_csv(events_path, encoding='gbk')
    print(f"✓ 成功加载事件文件，共 {len(df_events)} 条记录")
    
    # 提取唯一的中文站点名
    chinese_stations = df_events.iloc[:, 1].unique()
    print(f"✓ 提取到 {len(chinese_stations)} 个不同的中文站点名:")
    for i, station in enumerate(chinese_stations[:20], 1):
        print(f"  {i:3d}. '{station}'")
    if len(chinese_stations) > 20:
        print(f"  ... 还有 {len(chinese_stations) - 20} 个站点")
        
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit(1)

print("\n" + "="*60)
print("步骤2: 从站点描述文件中提取英文站点名")
print("="*60)

try:
    with open(station_desc_path, 'r', encoding='utf-8') as f:
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
    print(f"❌ 加载失败: {e}")
    exit(1)

print("\n" + "="*60)
print("步骤3: 生成映射模板")
print("="*60)

# 生成JSON映射模板
mapping_template = {}
for i, (cn, en) in enumerate(zip(chinese_stations, english_stations)):
    mapping_template[cn] = en
    print(f'    "{cn}": "{en}",')

if len(chinese_stations) != len(english_stations):
    print(f"\n⚠️ 警告: 中文站点数({len(chinese_stations)}) != 英文站点数({len(english_stations)})")
    print(f"  请手动检查并补充缺失的映射关系")

# 保存为JSON文件
output_path = '/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/station_name_mapping.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(mapping_template, f, ensure_ascii=False, indent=2)

print(f"\n✓ 映射模板已保存到: {output_path}")
print(f"  共 {len(mapping_template)} 个映射关系")
print(f"\n下一步操作:")
print(f"  1. 打开 {output_path} 检查映射是否正确")
print(f"  2. 如有错误，手动修正后保存")
print(f"  3. 在 fine_gnn_vals.py 中加载此JSON文件替代硬编码映射")
