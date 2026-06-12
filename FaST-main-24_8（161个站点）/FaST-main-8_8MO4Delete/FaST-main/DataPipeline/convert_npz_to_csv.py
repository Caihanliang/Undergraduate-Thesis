#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 his.npz 数据转换为 CSV 格式
输出格式：[时间, 站点编号, 小客车上行, 小客车下行, 非小客车上行, 非小客车下行]
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta

# ========== 配置参数 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 修正：项目根目录是 BASE_DIR 的上一级
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_NAME = "HNGS_4FEAT"
DATA_DIR = os.path.join(PROJECT_ROOT, "main-master", "datasets", DATASET_NAME)

# 输入文件
NPZ_PATH = os.path.join(DATA_DIR, "his.npz")
DESC_PATH = os.path.join(DATA_DIR, "desc.json")
STATION_MAPPING_PATH = os.path.join(DATA_DIR, "station_mapping.json")

# 输出文件
OUTPUT_CSV = os.path.join(DATA_DIR, "his_data.csv")


def load_metadata():
    """加载元数据（描述信息和站点映射）"""
    print("=" * 60)
    print("步骤1: 加载元数据")
    print("=" * 60)
    
    # 加载描述信息
    with open(DESC_PATH, 'r', encoding='utf-8') as f:
        desc = json.load(f)
    
    print(f"  数据集名称: {desc['dataset_name']}")
    print(f"  时间范围: {desc['time_range']}")
    print(f"  站点数量: {desc['num_nodes']}")
    print(f"  特征数量: {desc['num_features']}")
    print(f"  特征名称: {', '.join(desc['feature_names'])}")
    
    # 加载站点映射
    with open(STATION_MAPPING_PATH, 'r', encoding='utf-8') as f:
        station_mapping = json.load(f)
    
    print(f"  站点映射已加载: {len(station_mapping['stations'])} 个站点")
    
    return desc, station_mapping


def convert_npz_to_csv():
    """将 npz 文件转换为 CSV"""
    print("\n" + "=" * 60)
    print("步骤2: 转换数据为 CSV 格式")
    print("=" * 60)
    
    # 加载元数据
    desc, station_mapping = load_metadata()
    
    # 加载 npz 数据
    print(f"\n📂 加载数据文件: {NPZ_PATH}")
    data_dict = np.load(NPZ_PATH)
    data_array = data_dict['data']  # 形状: [时间步, 站点数, 4特征]
    
    print(f"  ✓ 数据形状: {data_array.shape}")
    print(f"     - 时间步: {data_array.shape[0]}")
    print(f"     - 站点数: {data_array.shape[1]}")
    print(f"     - 特征数: {data_array.shape[2]}")
    
    # 解析起始时间
    start_date_str = desc['time_range'].split(' to ')[0]
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    
    print(f"\n  基准时间: {start_dt.strftime('%Y-%m-%d %H:%M')}")
    
    # 构建 DataFrame
    print(f"\n  正在构建 DataFrame...")
    rows = []
    
    num_times, num_stations, num_features = data_array.shape
    stations = station_mapping['stations']
    feature_names = desc['feature_names']
    
    for t_idx in range(num_times):
        # 计算当前时间点（每个索引代表1小时）
        current_time = start_dt + timedelta(hours=t_idx)
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        for s_idx in range(num_stations):
            station_id = stations[s_idx]
            
            row = {
                '时间': time_str,
                '站点编号': station_id,
                feature_names[0]: round(float(data_array[t_idx, s_idx, 0]), 2),  # 小客车上行
                feature_names[1]: round(float(data_array[t_idx, s_idx, 1]), 2),  # 小客车下行
                feature_names[2]: round(float(data_array[t_idx, s_idx, 2]), 2),  # 非小客车上行
                feature_names[3]: round(float(data_array[t_idx, s_idx, 3]), 2),  # 非小客车下行
            }
            rows.append(row)
        
        # 进度显示
        if (t_idx + 1) % 100 == 0 or t_idx == num_times - 1:
            print(f"    处理进度: {t_idx + 1}/{num_times} ({(t_idx+1)/num_times*100:.1f}%)")
    
    # 创建 DataFrame
    df = pd.DataFrame(rows)
    
    print(f"\n  ✅ DataFrame 构建完成")
    print(f"     总行数: {len(df)}")
    print(f"     列名: {list(df.columns)}")
    
    # 保存为 CSV
    print(f"\n💾 保存 CSV 文件: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    file_size = os.path.getsize(OUTPUT_CSV) / 1024 / 1024
    print(f"  ✅ 保存成功！文件大小: {file_size:.2f} MB")
    
    # 打印统计信息
    print(f"\n📊 数据统计预览:")
    print(f"  前5行数据:")
    print(df.head().to_string(index=False))
    
    print(f"\n  各特征统计:")
    for feat_name in feature_names:
        print(f"    {feat_name}:")
        print(f"      均值: {df[feat_name].mean():.2f}")
        print(f"      标准差: {df[feat_name].std():.2f}")
        print(f"      最小值: {df[feat_name].min():.2f}")
        print(f"      最大值: {df[feat_name].max():.2f}")
    
    return df


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🚀 开始转换 his.npz 为 CSV 格式")
    print("=" * 60 + "\n")
    
    try:
        df = convert_npz_to_csv()
        
        print("\n" + "=" * 60)
        print("🎉 转换完成！")
        print("=" * 60)
        print(f"\n输出文件: {OUTPUT_CSV}")
        print(f"\n使用示例:")
        print(f"  # 在 Python 中读取")
        print(f"  import pandas as pd")
        print(f"  df = pd.read_csv('{OUTPUT_CSV}')")
        print(f"  print(df.head())")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
""" 
python DataPipeline/convert_npz_to_csv.py
FaST-main-8_8MO4/FaST-main/DataPipeline/convert_npz_to_csv.py
"""