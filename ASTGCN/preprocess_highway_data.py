#!/usr/bin/env python
# coding: utf-8
"""
ASTGCN 数据预处理脚本
功能：将原始CSV数据转换为ASTGCN所需的.npz格式
预测目标：4个特征（小客车上/下行、非小客车上/下行）
时序配置：8输入8输出
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse


def load_and_merge_data(dataset_dir):
    """加载并合并9月和10月的数据"""
    print("=== 加载数据 ===")
    
    # 读取两个月份的数据
    df_sep = pd.read_csv(os.path.join(dataset_dir, '观测站小时交通量-9.csv'), encoding='utf-8')
    df_oct = pd.read_csv(os.path.join(dataset_dir, '观测站小时交通量-10.csv'), encoding='utf-8')
    
    # 合并数据
    df = pd.concat([df_sep, df_oct], ignore_index=True)
    print(f"总数据行数: {len(df)}")
    print(f"时间范围: {df['观测日期'].min()} 至 {df['观测日期'].max()}")
    
    return df


def extract_features(df):
    """提取4个目标特征"""
    print("\n=== 提取特征 ===")
    
    # 只保留上行和下行数据（排除断面）
    df_filtered = df[df['行驶方向'].isin(['上行', '下行'])].copy()
    
    # 计算非小客车流量
    df_filtered['非小客车'] = df_filtered['汽车自然数'] - df_filtered['小客车']
    
    # 获取所有站点列表
    stations = df_filtered['观测站编号'].unique()
    print(f"站点数量: {len(stations)}")
    
    # 创建站点到索引的映射
    station_to_idx = {station: idx for idx, station in enumerate(sorted(stations))}
    
    # 保存站点映射文件
    mapping_df = pd.DataFrame({
        'station_id': list(station_to_idx.values()),
        'station_code': list(station_to_idx.keys())
    })
    mapping_df.to_csv(os.path.join(DATASET_DIR, 'station_mapping.csv'), index=False)
    print(f"站点映射已保存到: {os.path.join(DATASET_DIR, 'station_mapping.csv')}")
    
    return df_filtered, station_to_idx, stations


def build_adjacency_matrix(df, stations, method='distance'):
    """构建邻接矩阵
    
    Args:
        df: 数据框
        stations: 站点列表
        method: 构建方法 ('distance' 基于路线编号距离, 'correlation' 基于流量相关性)
    
    Returns:
        adj_matrix: 邻接矩阵 (N, N)
    """
    print(f"\n=== 构建邻接矩阵 (方法: {method}) ===")
    
    n_stations = len(stations)
    adj_matrix = np.zeros((n_stations, n_stations), dtype=np.float32)
    
    if method == 'distance':
        # 基于路线编号的简单距离度量
        # 提取路线编号中的数字部分作为位置标识
        station_positions = {}
        for station in stations:
            # 例如: G0401L010430121 -> 提取中间的数字部分
            try:
                # 简化处理：使用站点编码的哈希值作为相对位置
                pos = hash(station) % 1000
                station_positions[station] = pos
            except:
                station_positions[station] = 0
        
        # 计算站点间的距离
        positions = np.array([station_positions[s] for s in stations])
        distances = np.abs(positions[:, None] - positions[None, :])
        
        # 使用高斯核函数转换为相似度
        sigma = distances.std()
        adj_matrix = np.exp(-(distances ** 2) / (sigma ** 2 + 1e-8))
        
    elif method == 'correlation':
        # 基于流量相关性的邻接矩阵
        # 为每个站点计算平均流量向量
        station_flows = []
        for station in stations:
            station_data = df[df['观测站编号'] == station]
            # 取所有方向的平均流量
            avg_flow = station_data['汽车自然数'].mean()
            station_flows.append(avg_flow)
        
        # 计算相关性矩阵
        flows_array = np.array(station_flows).reshape(-1, 1)
        # 简化的相关性计算（实际应该使用时间序列相关性）
        for i in range(n_stations):
            for j in range(n_stations):
                if i == j:
                    adj_matrix[i, j] = 1.0
                else:
                    # 使用流量差异的倒数作为相似度
                    diff = abs(flows_array[i] - flows_array[j])
                    adj_matrix[i, j] = np.exp(-diff / (flows_array.std() + 1e-8))
    
    # 确保对角线为1（自连接）
    np.fill_diagonal(adj_matrix, 1.0)
    
    print(f"邻接矩阵形状: {adj_matrix.shape}")
    print(f"邻接矩阵均值: {adj_matrix.mean():.4f}")
    print(f"邻接矩阵标准差: {adj_matrix.std():.4f}")
    
    # 保存邻接矩阵
    np.savetxt(os.path.join(DATASET_DIR, 'adj_matrix.csv'), adj_matrix, delimiter=',')
    print(f"邻接矩阵已保存到: {os.path.join(DATASET_DIR, 'adj_matrix.csv')}")
    
    return adj_matrix


def create_spatio_temporal_data(df, station_to_idx, stations):
    """创建时空数据张量
    
    Returns:
        data: np.ndarray, shape=(T, N, F)
              T=时间步数, N=站点数, F=特征数(4)
    """
    print("\n=== 创建时空数据张量 ===")
    
    n_stations = len(stations)
    n_features = 4  # 小客车上/下行, 非小客车上/下行
    
    # 创建时间索引
    df['datetime'] = pd.to_datetime(df['观测日期']) + pd.to_timedelta(df['小时'] - 1, unit='h')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 获取唯一的时间点
    unique_times = sorted(df['datetime'].unique())
    n_timesteps = len(unique_times)
    print(f"总时间步数: {n_timesteps} ({n_timesteps // 24} 天)")
    
    # 初始化数据张量 (T, N, F)
    data = np.zeros((n_timesteps, n_stations, n_features), dtype=np.float32)
    
    # 填充数据
    time_to_idx = {t: idx for idx, t in enumerate(unique_times)}
    
    for _, row in df.iterrows():
        station = row['观测站编号']
        direction = row['行驶方向']
        time_idx = time_to_idx[row['datetime']]
        station_idx = station_to_idx[station]
        
        if direction == '上行':
            data[time_idx, station_idx, 0] = row['小客车']  # 小客车上行
            data[time_idx, station_idx, 2] = row['非小客车']  # 非小客车上行
        elif direction == '下行':
            data[time_idx, station_idx, 1] = row['小客车']  # 小客车下行
            data[time_idx, station_idx, 3] = row['非小客车']  # 非小客车下行
    
    print(f"数据张量形状: {data.shape}")
    print(f"数据范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"零值比例: {(data == 0).sum() / data.size * 100:.2f}%")
    
    return data


def generate_sliding_windows(data, input_len=8, output_len=8):
    """生成滑动窗口样本
    
    Args:
        data: np.ndarray, shape=(T, N, F)
        input_len: 输入序列长度
        output_len: 输出序列长度
    
    Returns:
        samples: list of tuples (input_seq, target_seq)
    """
    print(f"\n=== 生成滑动窗口样本 (输入:{input_len}, 输出:{output_len}) ===")
    
    T, N, F = data.shape
    samples = []
    
    for t in range(input_len, T - output_len + 1):
        input_seq = data[t - input_len:t]  # (input_len, N, F)
        target_seq = data[t:t + output_len]  # (output_len, N, F)
        samples.append((input_seq, target_seq))
    
    print(f"生成样本数量: {len(samples)}")
    
    return samples


def normalize_and_save(samples, save_path, mean=None, std=None, is_train=True):
    """归一化数据并保存为.npz格式 (ASTGCN兼容格式)
    
    Args:
        samples: list of (input, target) tuples
        save_path: 保存路径
        mean: 预计算的均值 (用于验证/测试集)
        std: 预计算的标准差 (用于验证/测试集)
        is_train: 是否为训练集（用于计算统计量）
    
    Returns:
        mean, std: 归一化参数
    """
    print("\n=== 归一化并保存数据 ===")
    
    # 将所有样本堆叠成数组
    all_inputs = np.array([s[0] for s in samples])  # (num_samples, input_len, N, F)
    all_targets = np.array([s[1] for s in samples])  # (num_samples, output_len, N, F)
    
    print(f"原始输入形状: {all_inputs.shape} (B, T_in, N, F)")
    print(f"原始目标形状: {all_targets.shape} (B, T_out, N, F)")
    
    # 计算归一化参数（仅在训练集上）
    if is_train:
        # 在训练集上计算统计量 (B, T, N, F) -> 对 B, T, N 维度求均值和标准差
        mean = all_inputs.mean(axis=(0, 1, 2), keepdims=True)  # (1, 1, 1, F)
        std = all_inputs.std(axis=(0, 1, 2), keepdims=True) + 1e-8  # 避免除零
        print(f"计算得到均值 shape: {mean.shape}, 值: {mean.squeeze()}")
        print(f"计算得到标准差 shape: {std.shape}, 值: {std.squeeze()}")
    
    # 归一化输入和目标
    inputs_norm = (all_inputs - mean) / std
    targets_norm = (all_targets - mean) / std  # ✅ 目标也需要归一化
    
    print(f"归一化后输入范围: [{inputs_norm.min():.4f}, {inputs_norm.max():.4f}]")
    print(f"归一化后目标范围: [{targets_norm.min():.4f}, {targets_norm.max():.4f}]")
    
    # ASTGCN 期望的格式（多变量版本）:
    # x: (B, N, F, T_in) - 输入特征（归一化）
    # y: (B, N, F, T_out) - 预测目标（归一化）
    
    # 转换输入格式: (B, T_in, N, F) -> (B, N, F, T_in)
    x_final = inputs_norm.transpose(0, 2, 3, 1)  # (B, N, F, T_in)
    
    # 转换目标格式: (B, T_out, N, F) -> (B, N, F, T_out)
    # 保持所有4个特征，不进行平均
    y_final = targets_norm.transpose(0, 2, 3, 1)  # (B, N, F, T_out) - 归一化后的目标
    
    print(f"最终 x 形状: {x_final.shape} (B, N, F_in, T_in)")
    print(f"最终 y 形状: {y_final.shape} (B, N, F_out, T_out)")
    print(f"✅ 采用多变量输出模式：同时预测4个特征")
    
    # 保存为 ASTGCN 期望的格式
    np.savez_compressed(
        save_path,
        x=x_final,      # ASTGCN 期望的键名
        y=y_final,      # ASTGCN 期望的键名（现在是多变量）
        mean=mean,
        std=std
    )
    
    print(f"✅ 数据已保存到: {save_path}.npz")
    
    return mean, std


def main():
    parser = argparse.ArgumentParser(description='ASTGCN数据预处理')
    parser.add_argument('--dataset_dir', type=str, default='./dataset', help='数据集目录')
    parser.add_argument('--input_len', type=int, default=8, help='输入序列长度')
    parser.add_argument('--output_len', type=int, default=8, help='输出序列长度')
    parser.add_argument('--adj_method', type=str, default='distance', 
                       choices=['distance', 'correlation'], help='邻接矩阵构建方法')
    args = parser.parse_args()
    
    global DATASET_DIR
    DATASET_DIR = args.dataset_dir
    
    # Step 1: 加载数据
    df = load_and_merge_data(DATASET_DIR)
    
    # Step 2: 提取特征
    df_filtered, station_to_idx, stations = extract_features(df)
    
    # Step 3: 构建邻接矩阵
    adj_matrix = build_adjacency_matrix(df_filtered, stations, method=args.adj_method)
    
    # Step 4: 创建时空数据张量
    data_tensor = create_spatio_temporal_data(df_filtered, station_to_idx, stations)
    
    # Step 5: 生成滑动窗口样本
    samples = generate_sliding_windows(data_tensor, args.input_len, args.output_len)
    
    # Step 6: 划分训练/验证/测试集 (60%/20%/20%)
    total_samples = len(samples)
    train_end = int(total_samples * 0.6)
    val_end = int(total_samples * 0.8)
    
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_samples)} 样本")
    print(f"  验证集: {len(val_samples)} 样本")
    print(f"  测试集: {len(test_samples)} 样本")
    
    # Step 7: 归一化并保存
    # 先在训练集上计算归一化参数
    train_path = os.path.join(DATASET_DIR, 'train')
    mean, std = normalize_and_save(train_samples, train_path)
    
    # 使用相同的参数归一化验证集和测试集
    val_path = os.path.join(DATASET_DIR, 'val')
    normalize_and_save(val_samples, val_path, mean=mean, std=std)
    
    test_path = os.path.join(DATASET_DIR, 'test')
    normalize_and_save(test_samples, test_path, mean=mean, std=std)
    
    # Step 8: 保存配置文件
    config_content = f"""[Data]
adj_filename = {os.path.join(DATASET_DIR, 'adj_matrix.csv')}
graph_signal_matrix_filename = {DATASET_DIR}
num_of_vertices = {len(stations)}
points_per_hour = 1
num_for_predict = {args.output_len}
len_input = {args.input_len}
dataset_name = highway_traffic

[Training]
ctx = 0
in_channels = 4
nb_block = 2
K = 3
nb_chev_filter = 64
nb_time_filter = 64
batch_size = 32
model_name = astgcn_r
dataset_name = highway_traffic
num_of_weeks = 0
num_of_days = 0
num_of_hours = 1
start_epoch = 0
epochs = 100
learning_rate = 0.001
"""
    
    config_path = os.path.join('./configurations', 'highway_traffic.conf')
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    print(f"\n配置文件已保存到: {config_path}")
    
    print("\n=== 数据预处理完成 ===")
    print(f"下一步: 运行训练命令")
    print(f"  python train_ASTGCN_r.py --config configurations/highway_traffic.conf")


if __name__ == '__main__':
    main()
