#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理脚本：将原始CSV转换为4特征时序数据
特征定义：
  - Feature 0: 小客车上行
  - Feature 1: 小客车下行
  - Feature 2: 非小客车上行 (汽车自然数-小客车)上行
  - Feature 3: 非小客车下行 (汽车自然数-小客车)下行

输入：观测站小时交通量 -9.csv, 观测站小时交通量 -10.csv
输出：
  - his.npz [时间步, 站点数, 4]
  - his.h5 [时间步, 站点数, 4] (HDF5格式，支持高效部分读取)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("⚠️  警告: h5py 未安装，将跳过 .h5 文件生成")
    print("   安装命令: pip install h5py")

# ========== 配置参数 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "main-master", "datasets", "HNGS_4FEAT")

# 输入文件
INPUT_FILES = [
    "观测站小时交通量-9.csv",
    "观测站小时交通量-10.csv"
]

# 时间配置
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
HOURLY_FREQ = "1H"


def load_and_merge_data():
    """加载并合并两个月的数据"""
    print("=" * 60)
    print("步骤1: 加载原始数据")
    print("=" * 60)
    
    all_data = []
    
    for file_name in INPUT_FILES:
        file_path = os.path.join(BASE_DIR, file_name)
        print(f"\n📂 读取文件: {file_name}")
        
        if not os.path.exists(file_path):
            print(f"  ⚠️  文件不存在，跳过")
            continue
        
        # 读取CSV
        df = pd.read_csv(file_path)
        print(f"  ✓ 加载完成: {len(df)} 行")
        all_data.append(df)
    
    if not all_data:
        raise FileNotFoundError("未找到任何输入文件！")
    
    # 合并数据
    merged_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ 合并完成: {len(merged_df)} 行")
    
    return merged_df


def extract_features(df):
    """从原始数据提取4个特征"""
    print("\n" + "=" * 60)
    print("步骤2: 提取4个特征")
    print("=" * 60)
    
    # 过滤只保留上行和下行数据（排除"断面"）
    df_filtered = df[df['行驶方向'].isin(['上行', '下行'])].copy()
    print(f"  过滤后数据量: {len(df_filtered)} 行")
    
    # 计算非小客车流量
    df_filtered['非小客车'] = df_filtered['汽车自然数'] - df_filtered['小客车']
    
    # 确保非负
    df_filtered['非小客车'] = df_filtered['非小客车'].clip(lower=0)
    
    print(f"\n  特征统计:")
    print(f"    - 小客车: 均值={df_filtered['小客车'].mean():.2f}, 范围=[{df_filtered['小客车'].min()}, {df_filtered['小客车'].max()}]")
    print(f"    - 非小客车: 均值={df_filtered['非小客车'].mean():.2f}, 范围=[{df_filtered['非小客车'].min()}, {df_filtered['非小客车'].max()}]")
    
    return df_filtered


def pivot_to_timeseries(df):
    """将长格式数据转换为宽表格式 [时间, 站点, 特征]"""
    print("\n" + "=" * 60)
    print("步骤3: 转换为时序格式")
    print("=" * 60)
    
    # 处理小时字段：将24点转换为次日的0点
    print(f"\n  ⏰ 处理时间字段...")
    original_hours = df['小时'].copy()
    
    # 找到24点的行
    midnight_mask = df['小时'] == 24
    midnight_count = midnight_mask.sum()
    
    if midnight_count > 0:
        print(f"  发现 {midnight_count} 个24:00时间点，正在转换为次日00:00...")
        
        # 创建datetime列，处理24点
        df['datetime'] = pd.to_datetime(
            df['观测日期'].astype(str) + ' ' + df['小时'].astype(str).str.zfill(2) + ':00:00',
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'  # 将无效的24:00标记为NaT
        )
        
        # 对于24点的行，手动计算为次日00:00
        for idx in df[midnight_mask].index:
            current_date = pd.to_datetime(df.loc[idx, '观测日期'])
            next_day = current_date + pd.Timedelta(days=1)
            df.loc[idx, 'datetime'] = next_day
        
        print(f"  ✅ 时间转换完成")
    else:
        # 没有24点，直接解析
        df['datetime'] = pd.to_datetime(
            df['观测日期'].astype(str) + ' ' + df['小时'].astype(str).str.zfill(2) + ':00:00',
            format='%Y-%m-%d %H:%M:%S'
        )
    
    # 检查是否有NaT值
    nat_count = df['datetime'].isna().sum()
    if nat_count > 0:
        print(f"  ⚠️  警告: 仍有 {nat_count} 个无效时间，已删除")
        df = df.dropna(subset=['datetime'])
    
    # 获取所有站点
    stations = df['观测站编号'].unique()
    print(f"  站点数量: {len(stations)}")
    print(f"  站点示例: {stations[:5]}")
    
    # 创建站点编号到索引的映射
    station_to_idx = {station: idx for idx, station in enumerate(sorted(stations))}
    
    # 初始化数据结构 [时间, 站点, 特征]
    # 特征顺序: [小客车上行, 小客车下行, 非小客车上行, 非小客车下行]
    data = {}
    
    print(f"\n  开始处理每个站点...")
    
    for station in sorted(stations):
        station_data = df[df['观测站编号'] == station]
        
        for direction in ['上行', '下行']:
            direction_data = station_data[station_data['行驶方向'] == direction]
            
            for time_row in direction_data.itertuples():
                time_key = time_row.datetime
                
                if time_key not in data:
                    data[time_key] = np.zeros((len(stations), 4))
                
                station_idx = station_to_idx[station]
                
                if direction == '上行':
                    data[time_key][station_idx, 0] = time_row.小客车  # 小客车上行
                    data[time_key][station_idx, 2] = time_row.非小客车  # 非小客车上行
                else:  # 下行
                    data[time_key][station_idx, 1] = time_row.小客车  # 小客车下行
                    data[time_key][station_idx, 3] = time_row.非小客车  # 非小客车下行
    
    # 按时间排序
    sorted_times = sorted(data.keys())
    
    # 构建最终数组
    num_times = len(sorted_times)
    num_stations = len(stations)
    num_features = 4
    
    result_array = np.zeros((num_times, num_stations, num_features))
    
    for t_idx, time_key in enumerate(sorted_times):
        result_array[t_idx] = data[time_key]
    
    print(f"\n  ✅ 数据形状: {result_array.shape}")
    print(f"     - 时间步: {num_times}")
    print(f"     - 站点数: {num_stations}")
    print(f"     - 特征数: {num_features}")
    
    return result_array, sorted_times, station_to_idx


def save_data(result_array, sorted_times, station_to_idx):
    """保存处理后的数据（NPZ + HDF5）"""
    print("\n" + "=" * 60)
    print("步骤4: 保存数据")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存 npz 文件
    npz_path = os.path.join(OUTPUT_DIR, "his.npz")
    np.savez(npz_path, data=result_array)
    print(f"  ✅ 保存 NPZ 数据: {npz_path}")
    print(f"     形状: {result_array.shape}")
    npz_size = os.path.getsize(npz_path) / 1024 / 1024
    print(f"     大小: {npz_size:.2f} MB")
    
    # 🔥 新增：保存 HDF5 文件
    if H5PY_AVAILABLE:
        h5_path = os.path.join(OUTPUT_DIR, "his.h5")
        try:
            with h5py.File(h5_path, 'w') as f:
                # 创建数据集，支持压缩和分块读取
                f.create_dataset(
                    'data',
                    data=result_array,
                    compression='gzip',
                    compression_opts=4,
                    chunks=True,  # 自动选择最优块大小
                    dtype='float32'
                )
                
                # 保存元数据
                f.attrs['dataset_name'] = 'HNGS_4FEAT'
                f.attrs['num_nodes'] = len(station_to_idx)
                f.attrs['num_features'] = 4
                f.attrs['num_time_steps'] = len(sorted_times)
                f.attrs['time_range_start'] = sorted_times[0].strftime('%Y-%m-%d %H:%M:%S')
                f.attrs['time_range_end'] = sorted_times[-1].strftime('%Y-%m-%d %H:%M:%S')
                f.attrs['feature_names'] = json.dumps(
                    ["小客车上行", "小客车下行", "非小客车上行", "非小客车下行"],
                    ensure_ascii=False
                )
            
            h5_size = os.path.getsize(h5_path) / 1024 / 1024
            print(f"  ✅ 保存 HDF5 数据: {h5_path}")
            print(f"     形状: {result_array.shape}")
            print(f"     大小: {h5_size:.2f} MB")
            print(f"     压缩比: {npz_size/h5_size:.2f}x")
        except Exception as e:
            print(f"  ⚠️  HDF5 保存失败: {e}")
    else:
        print(f"  ⏭️  跳过 HDF5 保存（h5py 未安装）")
    
    # 保存站点映射
    station_list = sorted(station_to_idx.keys())
    station_mapping = {idx: name for name, idx in station_to_idx.items()}
    
    mapping_path = os.path.join(OUTPUT_DIR, "station_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            "num_stations": len(station_list),
            "stations": station_list,
            "station_to_idx": station_to_idx,
            "feature_names": ["小客车上行", "小客车下行", "非小客车上行", "非小客车下行"]
        }, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 保存站点映射: {mapping_path}")
    
    # 保存描述信息
    desc = {
        "dataset_name": "HNGS_4FEAT",
        "time_range": f"{sorted_times[0].strftime('%Y-%m-%d')} to {sorted_times[-1].strftime('%Y-%m-%d')}",
        "num_nodes": len(station_to_idx),
        "num_features": 4,
        "feature_names": ["小客车上行", "小客车下行", "非小客车上行", "非小客车下行"],
        "mean": result_array.mean(axis=(0, 1)).tolist(),
        "std": result_array.std(axis=(0, 1)).tolist(),
        "min": result_array.min(axis=(0, 1)).tolist(),
        "max": result_array.max(axis=(0, 1)).tolist()
    }
    
    desc_path = os.path.join(OUTPUT_DIR, "desc.json")
    with open(desc_path, 'w', encoding='utf-8') as f:
        json.dump(desc, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 保存描述信息: {desc_path}")
    
    # 打印统计信息
    print(f"\n  📊 数据统计:")
    print(f"     时间范围: {sorted_times[0]} ~ {sorted_times[-1]}")
    print(f"     总时间步: {len(sorted_times)}")
    print(f"     站点数量: {len(station_to_idx)}")
    print(f"     特征数量: 4")
    print(f"\n     各特征统计:")
    for i, feat_name in enumerate(desc["feature_names"]):
        print(f"       {feat_name}: 均值={desc['mean'][i]:.2f}, 标准差={desc['std'][i]:.2f}")


def verify_data(result_array):
    """验证数据质量"""
    print("\n" + "=" * 60)
    print("步骤5: 数据验证")
    print("=" * 60)
    
    # 检查零值
    zero_counts = (result_array == 0).sum(axis=(0, 1))
    total_values = result_array.shape[0] * result_array.shape[1]
    
    print(f"\n  零值统计:")
    for i, feat_name in enumerate(["小客车上行", "小客车下行", "非小客车上行", "非小客车下行"]):
        zero_pct = zero_counts[i] / total_values * 100
        print(f"    {feat_name}: {zero_counts[i]} 个零值 ({zero_pct:.2f}%)")
    
    # 检查负值
    negative_mask = result_array < 0
    negative_count = negative_mask.sum()
    if negative_count > 0:
        print(f"\n  ⚠️  警告: 发现 {negative_count} 个负值！")
    else:
        print(f"\n  ✅ 无负值")
    
    # 检查缺失值
    nan_count = np.isnan(result_array).sum()
    if nan_count > 0:
        print(f"  ⚠️  警告: 发现 {nan_count} 个NaN值！")
    else:
        print(f"  ✅ 无缺失值")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🚀 开始处理数据 - 4特征版本")
    print("   特征: 小客车上行、小客车下行、非小客车上行、非小客车下行")
    print("=" * 60 + "\n")
    
    try:
        # 1. 加载数据
        merged_df = load_and_merge_data()
        
        # 2. 提取特征
        feature_df = extract_features(merged_df)
        
        # 3. 转换为时序格式
        result_array, sorted_times, station_to_idx = pivot_to_timeseries(feature_df)
        
        # 4. 保存数据
        save_data(result_array, sorted_times, station_to_idx)
        
        # 5. 验证数据
        verify_data(result_array)
        
        print("\n" + "=" * 60)
        print("🎉 数据处理完成！")
        print("=" * 60)
        print(f"\n输出目录: {OUTPUT_DIR}")
        print(f"文件列表:")
        for file in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file)
            size = os.path.getsize(file_path) / 1024 / 1024
            print(f"  - {file} ({size:.2f} MB)")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
