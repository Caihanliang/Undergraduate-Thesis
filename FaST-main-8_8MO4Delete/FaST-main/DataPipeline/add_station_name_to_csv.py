#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 his_data.csv 中添加站点名称列
根据 station_mapping.json 中的映射关系，将站点编号转换为可读的站点名称

注意：station_mapping.json 中只包含站点编号（如 G0401L01C），不包含中文名称
本脚本提供两种模式：
  1. 直接使用站点编号作为名称（默认）
  2. 如果有自定义的名称映射文件，可以加载并替换
"""

import pandas as pd
import json
import os
import sys

# ========== 配置参数 ==========
# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录是 DataPipeline 的父目录
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 数据集目录
DATASET_DIR = os.path.join(PROJECT_ROOT, "main-master", "datasets", "HNGS_4FEAT")

# 输入文件
# CSV_PATH = os.path.join(DATASET_DIR, "his_data.csv")
"""
/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/main-master/datasets/HNGS_4FEAT/his_data.csv
"""
CSV_PATH = "main-master/datasets/HNGS_4FEAT/his_data.csv"
"""
/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/main-master/datasets/HNGS_4FEAT/station_mapping.json
"""
# MAPPING_PATH = os.path.join(DATASET_DIR, "station_mapping.json")
MAPPING_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/main-master/datasets/HNGS_4FEAT/station_mapping.json"



# 可选：自定义站点名称映射文件（如果有的话）
# CUSTOM_NAME_MAPPING = os.path.join(DATASET_DIR, "station_names.json")

# 输出文件
print("DATASET_DIR:",DATASET_DIR)
OUTPUT_CSV = os.path.join(DATASET_DIR, "his_data_with_name.csv")


def load_station_mapping():
    """加载站点映射关系"""
    print("=" * 60)
    print("步骤1: 加载站点映射")
    print("=" * 60)
    
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    # station_mapping.json 中的 stations 列表就是按索引排序的站点编号
    stations = mapping_data['stations']
    station_to_idx = mapping_data.get('station_to_idx', {})
    
    print(f"  ✓ 加载 {len(stations)} 个站点")
    print(f"  站点编号示例（前5个）:")
    for i, station_id in enumerate(stations[:5]):
        idx = station_to_idx.get(station_id, 'N/A')
        print(f"    索引 {idx}: {station_id}")
    
    return stations, station_to_idx


def load_custom_names():
    """加载自定义站点名称（如果存在）"""
    if os.path.exists(CUSTOM_NAME_MAPPING):
        print(f"\n  📂 发现自定义名称映射: {CUSTOM_NAME_MAPPING}")
        with open(CUSTOM_NAME_MAPPING, 'r', encoding='utf-8') as f:
            custom_names = json.load(f)
        print(f"  ✓ 加载 {len(custom_names)} 个自定义名称")
        return custom_names
    else:
        print(f"\n  ℹ️  未找到自定义名称映射文件: {CUSTOM_NAME_MAPPING}")
        print(f"     将使用站点编号作为名称")
        return None


def add_station_name_column():
    """在 CSV 中添加站点名称列"""
    print("\n" + "=" * 60)
    print("步骤2: 添加站点名称列")
    print("=" * 60)
    
    # 加载映射
    stations, station_to_idx = load_station_mapping()
    
    # 尝试加载自定义名称
    custom_names = load_custom_names()
    
    # 构建映射字典
    if custom_names:
        # 使用自定义名称
        station_name_map = custom_names
        print(f"\n  ✓ 使用自定义站点名称映射")
    else:
        # 使用站点编号作为名称（默认）
        station_name_map = {station_id: station_id for station_id in stations}
        print(f"\n  ✓ 使用站点编号作为名称")
    
    # 加载 CSV 数据
    print(f"\n📂 加载数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  ✓ 加载完成: {len(df)} 行, {len(df.columns)} 列")
    print(f"  当前列名: {list(df.columns)}")
    
    # 检查是否已有站点编号列
    if '站点编号' not in df.columns:
        print(f"\n  ❌ 错误: 未找到'站点编号'列")
        print(f"  可用列: {list(df.columns)}")
        raise ValueError("缺少'站点编号'列")
    
    # 添加站点名称列（在站点编号列后面）
    print(f"\n  正在添加'站点名称'列...")
    station_col_idx = list(df.columns).index('站点编号')
    df.insert(station_col_idx + 1, '站点名称', df['站点编号'].map(station_name_map))
    
    print(f"  ✓ 添加完成")
    print(f"  新列名: {list(df.columns)}")
    
    # 显示示例数据
    print(f"\n  数据预览（前5行）:")
    print(df.head().to_string(index=False))
    
    # 保存新的 CSV
    print(f"\n💾 保存文件: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    file_size = os.path.getsize(OUTPUT_CSV) / 1024 / 1024
    print(f"  ✓ 保存成功！文件大小: {file_size:.2f} MB")
    
    # 统计信息
    print(f"\n📊 统计信息:")
    print(f"  总行数: {len(df)}")
    print(f"  站点数: {df['站点编号'].nunique()}")
    print(f"  时间步数: {df['时间'].nunique()}")
    
    return df


def create_custom_name_template():
    """创建自定义名称映射模板文件"""
    print("\n" + "=" * 60)
    print("创建自定义名称映射模板")
    print("=" * 60)
    
    # 加载站点编号
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    stations = mapping_data['stations']
    
    # 创建模板（站点编号 -> 中文名称）
    template = {station: f"站点_{station}" for station in stations}
    
    template_path = CUSTOM_NAME_MAPPING
    with open(template_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 模板已创建: {template_path}")
    print(f"  请编辑该文件，将值替换为实际的中文站点名称")
    print(f"  示例:")
    print(f'    {{')
    print(f'      "G0401L01C": "黄兴站",')
    print(f'      "G0401L02C": "雨花站",')
    print(f'      ...')
    print(f'    }}')


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🚀 站点名称列添加工具")
    print("=" * 60 + "\n")
    
    try:
        # 检查是否需要创建模板
        if not os.path.exists(CSV_PATH):
            print(f"❌ 错误: 找不到输入文件 {CSV_PATH}")
            return
        
        df = add_station_name_column()
        
        print("\n" + "=" * 60)
        print("🎉 处理完成！")
        print("=" * 60)
        print(f"\n输出文件: {OUTPUT_CSV}")
        print(f"\n提示:")
        print(f"  1. 如果需要中文站点名称，请创建 station_names.json 文件")
        print(f"  2. 运行以下命令创建模板:")
        print(f"     python {os.path.basename(__file__)} --create-template")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # 支持命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--create-template':
        create_custom_name_template()
    else:
        main()
