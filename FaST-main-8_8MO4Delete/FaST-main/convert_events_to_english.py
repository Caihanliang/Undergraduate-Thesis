"""
验证事件文件是否已使用英文站点名称
如果事件文件已是英文，则直接使用；如果是中文，则转换为英文
"""
import pandas as pd
import json
import os

# 文件路径
EVENTS_CSV_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/events_list_quan.csv"
MAPPING_JSON_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/station_name_mapping.json"
OUTPUT_CSV_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/events_list_quan_en.csv"

print("="*80)
print("🔍 事件文件语言检测与转换工具")
print("="*80)

# 1. 加载映射字典
print("\n1️⃣  加载站点名称映射...")
with open(MAPPING_JSON_PATH, 'r', encoding='utf-8') as f:
    station_mapping = json.load(f)

print(f"✓ 已加载 {len(station_mapping)} 个站点映射")

# 2. 读取事件CSV文件
print("\n2️⃣  读取事件文件...")
events_df = None
for encoding in ['utf-8', 'gbk', 'gb18030', 'latin1']:
    try:
        events_df = pd.read_csv(EVENTS_CSV_PATH, encoding=encoding)
        print(f"✓ 使用 {encoding} 编码成功加载")
        break
    except (UnicodeDecodeError, UnicodeError):
        continue

if events_df is None:
    print("❌ 无法读取事件文件")
    exit(1)

print(f"✓ 原始数据形状: {events_df.shape}")
print(f"  列名: {list(events_df.columns)}")

# 3. 检测站点名称语言
station_col = events_df.columns[1]
sample_stations = events_df[station_col].dropna().head(20).astype(str).tolist()

is_chinese = any('\u4e00' <= char <= '\u9fff' for station in sample_stations for char in station)

print(f"\n3️⃣  语言检测结果:")
print(f"  采样站点 (前10个): {sample_stations[:10]}")
print(f"  是否包含中文: {'是' if is_chinese else '否'}")

# 4. 根据检测结果处理
if not is_chinese:
    print("\n✅ 事件文件已使用英文站点名称，无需转换！")
    print(f"  可直接使用: {EVENTS_CSV_PATH}")
    
    # 复制一份作为英文版（方便统一管理）
    import shutil
    shutil.copy2(EVENTS_CSV_PATH, OUTPUT_CSV_PATH)
    print(f"  已复制到: {OUTPUT_CSV_PATH}")
else:
    print("\n🔄 事件文件使用中文站点名称，开始转换...")
    
    success_count = 0
    failed_mappings = []
    
    def replace_station_name(chinese_name):
        global success_count, failed_mappings
        
        if chinese_name in station_mapping:
            success_count += 1
            return station_mapping[chinese_name]
        else:
            failed_mappings.append(chinese_name)
            return chinese_name
    
    events_df[station_col] = events_df[station_col].apply(replace_station_name)
    
    print(f"\n✓ 转换完成:")
    print(f"  - 成功映射: {success_count} 条记录")
    print(f"  - 失败映射: {len(set(failed_mappings))} 个站点")
    
    if failed_mappings:
        print(f"\n⚠️  未找到映射的站点 (前10个):")
        for station in list(set(failed_mappings))[:10]:
            print(f"    - '{station}'")
    
    # 保存结果
    events_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\n✓ 已保存至: {OUTPUT_CSV_PATH}")

# 5. 验证最终结果
print("\n4️⃣  验证最终结果:")
final_df = pd.read_csv(OUTPUT_CSV_PATH, encoding='utf-8')
final_stations = final_df.iloc[:, 1].unique()
print(f"  最终站点数量: {len(final_stations)}")
print(f"  示例站点: {list(final_stations)[:5]}")

# 检查是否有中文字符
has_chinese = any('\u4e00' <= char <= '\u9fff' for station in final_stations for char in str(station))
print(f"  是否仍含中文: {'是 ⚠️' if has_chinese else '否 ✓'}")

print("\n" + "="*80)
print("✅ 处理完成！")
print("="*80)
print(f"\n📝 使用说明:")
print(f"  在 fine_gnn_vals.py 中使用:")
print(f"  EVENTS_CSV_PATH = \"{OUTPUT_CSV_PATH}\"")
