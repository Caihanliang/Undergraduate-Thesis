"""
从原始 CSV/Excel 生成多变量数据集
适用于你只有原始 Excel 文件的情况

输入：
- 小客车流量 Excel/CSV（如：LittleCar真正名9_10.csv）
- 非小客车流量 Excel/CSV（如：NonLightCar9_10.csv）

输出：
- HNGS_MULTI/his.npz: [T, N, 2]
"""
import os
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_traffic_data_from_csv(csv_path, time_column='Time'):
    """
    从 CSV/Excel 加载交通流量数据
    
    Args:
        csv_path: CSV/Excel 文件路径
        time_column: 时间列名称
    
    Returns:
        data_array: [T, N, 1] 流量数据
        timestamps: 时间戳列表
        station_names: 站点名称列表
    """
    print(f"\n📂 加载数据: {csv_path}")
    
    # 读取数据
    if str(csv_path).endswith('.csv'):
        df = pd.read_csv(csv_path)
    elif str(csv_path).endswith(('.xlsx', '.xls')):
        df = pd.read_excel(csv_path)
    else:
        raise ValueError(f"不支持的文件格式: {csv_path}")
    
    print(f"   原始数据形状: {df.shape}")
    print(f"   列名: {df.columns.tolist()[:5]}...")
    
    # 提取时间列
    if time_column in df.columns:
        timestamps = pd.to_datetime(df[time_column])
        df = df.drop(columns=[time_column])
    else:
        raise ValueError(f"找不到时间列: {time_column}")
    
    # 提取站点名称（所有数值列）
    station_names = df.columns.tolist()
    
    # 转换为 numpy 数组 [T, N]
    data_array = df.values.astype(np.float32)
    
    print(f"   ✅ 提取完成: {data_array.shape[0]} 时间步, {data_array.shape[1]} 站点")
    
    return data_array, timestamps, station_names


def add_time_features(data_array, timestamps):
    """
    添加时间特征（TOD 和 DOW）
    
    Args:
        data_array: [T, N, 1] 流量数据
        timestamps: 时间戳
    
    Returns:
        data_with_time: [T, N, 3] 包含 [流量, TOD, DOW]
    """
    T, N = data_array.shape
    
    # 计算 TOD (Time of Day): 0~23
    tod = np.array([t.hour for t in timestamps], dtype=np.float32) / 24.0
    tod = np.repeat(tod[:, np.newaxis], N, axis=1)  # [T, N]
    
    # 计算 DOW (Day of Week): 0~6
    dow = np.array([t.dayofweek for t in timestamps], dtype=np.float32) / 7.0
    dow = np.repeat(dow[:, np.newaxis], N, axis=1)  # [T, N]
    
    # 拼接: [流量, TOD, DOW]
    data_with_time = np.stack([data_array, tod, dow], axis=2)  # [T, N, 3]
    
    print(f"   ✅ 添加时间特征: {data_with_time.shape}")
    
    return data_with_time


def create_multivariate_from_csv(
    light_vehicle_csv,
    non_light_vehicle_csv,
    output_dir=None,
    time_column='Time'
):
    """
    从两个 CSV 文件创建多变量数据集
    
    Args:
        light_vehicle_csv: 小客车流量 CSV/Excel
        non_light_vehicle_csv: 非小客车流量 CSV/Excel
        output_dir: 输出目录（默认自动创建）
        time_column: 时间列名
    """
    base_path = Path(__file__).parent.parent / "datasets"
    
    print("=" * 70)
    print("从 CSV/Excel 生成多变量数据集")
    print("=" * 70)
    
    # 1. 加载小客车数据
    lc_data, lc_timestamps, lc_stations = load_traffic_data_from_csv(
        light_vehicle_csv, time_column
    )
    
    # 2. 加载非小客车数据
    nlc_data, nlc_timestamps, nlc_stations = load_traffic_data_from_csv(
        non_light_vehicle_csv, time_column
    )
    
    # 3. 验证数据一致性
    assert lc_data.shape[0] == nlc_data.shape[0], "❌ 时间步不一致！"
    assert lc_data.shape[1] == nlc_data.shape[1], "❌ 站点数不一致！"
    assert list(lc_stations) == list(nlc_stations), "❌ 站点顺序不一致！"
    
    T, N = lc_data.shape
    print(f"\n📊 数据验证:")
    print(f"   - 时间步: {T}")
    print(f"   - 站点数: {N}")
    print(f"   - 时间范围: {lc_timestamps[0]} ~ {lc_timestamps[-1]}")
    
    # 4. 添加时间特征
    print("\n🔧 添加时间特征（TOD + DOW）...")
    lc_with_time = add_time_features(lc_data, lc_timestamps)
    nlc_with_time = add_time_features(nlc_data, nlc_timestamps)
    
    # 5. 合并为多变量格式 [T, N, 2]
    print("\n🔗 合并多变量数据...")
    multi_array = np.concatenate(
        [lc_with_time[:, :, 0:1], nlc_with_time[:, :, 0:1]],
        axis=2
    )  # [T, N, 2] 只保留流量，不保留 TOD/DOW
    
    print(f"   ✅ 合并后形状: {multi_array.shape}")
    print(f"   C=2: [小客车流量, 非小客车流量]")
    
    # 6. 创建输出目录
    output_name = "HNGS_MULTI"
    if output_dir is None:
        output_path = base_path / output_name
    else:
        output_path = Path(output_dir) / output_name
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 7. 保存数据（his.npz 格式）
    output_data_file = output_path / "his.npz"
    np.savez(output_data_file, data=multi_array)
    print(f"\n💾 数据已保存: {output_data_file}")
    
    # 8. 创建描述文件
    multi_desc = {
        "name": "hngs_multi",
        "domain": "traffic flow",
        "shape": list(multi_array.shape),
        "num_time_steps": int(T),
        "num_nodes": int(N),
        "num_features": 2,
        "feature_description": [
            "light vehicle flow",
            "non-light vehicle flow"
        ],
        "has_graph": True,
        "frequency (minutes)": 60,
        "description": "Hunan Highway Multi-Vehicle Dataset (from CSV)",
        "time_range": f"{lc_timestamps[0].strftime('%Y-%m-%d')} to {lc_timestamps[-1].strftime('%Y-%m-%d')}",
        "mean": [float(multi_array[:, :, 0].mean()), float(multi_array[:, :, 1].mean())],
        "std": [float(multi_array[:, :, 0].std()), float(multi_array[:, :, 1].std())],
        "regular_settings": {
            "INPUT_LEN": 8,
            "OUTPUT_LEN": 8,
            "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
            "NORM_EACH_CHANNEL": False,
            "RESCALE": False,
            "METRICS": ["MAE", "RMSE", "MAPE"],
            "NULL_VAL": 0.0
        }
    }
    
    output_desc_file = output_path / "desc.json"
    with open(output_desc_file, 'w') as f:
        json.dump(multi_desc, f, indent=2)
    print(f"💾 描述文件已保存: {output_desc_file}")
    
    # 9. 生成索引文件
    print("\n🔢 生成训练/验证/测试索引...")
    idx_train, idx_val, idx_test = generate_indices(T, 8, 8, [0.6, 0.2, 0.2])
    
    # 保存到 24_8 子目录（适配你的项目结构）
    idx_dir = output_path / "24_8"
    idx_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(idx_dir / "idx_train.npy", idx_train)
    np.save(idx_dir / "idx_val.npy", idx_val)
    np.save(idx_dir / "idx_test.npy", idx_test)
    print(f"💾 索引文件已保存: {idx_dir}")
    
    print("\n" + "=" * 70)
    print("✅ 多变量数据集创建完成！")
    print("=" * 70)
    print(f"\n📁 输出路径: {output_path}")
    print(f"📊 数据格式: [T, N, C] = {multi_array.shape}")
    print(f"\n💡 下一步:")
    print(f"   1. 检查数据: python FaST/prepare_multivariate_data.py verify")
    print(f"   2. 训练模型: python experiments/train_seed.py -c FaST/HNGS_8_8MV.py -g 0")
    
    return output_path


def generate_indices(total_steps, input_len, output_len, ratio):
    """
    生成训练/验证/测试索引
    
    Args:
        total_steps: 总时间步数
        input_len: 输入长度
        output_len: 输出长度
        ratio: [train, val, test] 比例
    
    Returns:
        idx_train, idx_val, idx_test
    """
    # 计算样本总数
    sample_len = input_len + output_len
    num_samples = total_steps - sample_len + 1
    
    # 划分数据集
    train_size = int(num_samples * ratio[0])
    val_size = int(num_samples * ratio[1])
    test_size = num_samples - train_size - val_size
    
    # 生成索引（注意：索引是输入序列的结束位置）
    idx_train = np.arange(input_len - 1, input_len - 1 + train_size)
    idx_val = np.arange(
        input_len - 1 + train_size,
        input_len - 1 + train_size + val_size
    )
    idx_test = np.arange(
        input_len - 1 + train_size + val_size,
        input_len - 1 + train_size + val_size + test_size
    )
    
    print(f"   训练集: {len(idx_train)} 样本")
    print(f"   验证集: {len(idx_val)} 样本")
    print(f"   测试集: {len(idx_test)} 样本")
    
    return idx_train, idx_val, idx_test


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='从 CSV/Excel 生成多变量数据集')
    parser.add_argument('--lc', type=str, required=True,
                       help='小客车流量 CSV/Excel 文件路径')
    parser.add_argument('--nlc', type=str, required=True,
                       help='非小客车流量 CSV/Excel 文件路径')
    parser.add_argument('--time-col', type=str, default='Time',
                       help='时间列名称（默认: Time）')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录（默认: datasets/）')
    
    args = parser.parse_args()
    
    # 创建多变量数据集
    create_multivariate_from_csv(
        light_vehicle_csv=args.lc,
        non_light_vehicle_csv=args.nlc,
        output_dir=args.output_dir,
        time_column=args.time_col
    )
"""
        python main-master/FaST/generate_multivariate_from_csv.py \
  --lc /home/user/Downloads/cai/FaST-main-8_8MO/FaST-main/DataPipeline/LittleCar真正名9_10.csv \
  --nlc /home/user/Downloads/cai/FaST-main-8_8MO/FaST-main/DataPipeline/NLittleCar真正名9_10.csv
        """
