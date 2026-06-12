"""
GMAN数据预处理脚本 - 高速公路交通流量预测
功能: 将原始CSV数据转换为GMAN所需的格式，支持4维特征预测
特征定义:
1. 小客车上行
2. 小客车下行  
3. 非小客车上行 (汽车自然数 - 小客车)
4. 非小客车下行 (汽车自然数 - 小客车)
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# ==================== 配置参数 ====================
DATA_DIR = '/home/user/Downloads/cai/GMAN/dataset'
OUTPUT_DIR = '/home/user/Downloads/cai/GMAN/data/highway_4feat'

# 时序配置
P = 8  # 输入步长
Q = 8  # 预测步长
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# 站点映射文件
STATION_MAPPING_FILE = os.path.join(DATA_DIR, 'station_mapping.csv')


def parse_datetime_with_24hour(date_str, hour):
    """
    处理包含24:00的时间解析
    24:00表示当日结束时刻，应转换为次日00:00
    """
    try:
        hour = int(hour)
        if hour == 24:
            # 24:00转换为次日00:00
            dt = pd.to_datetime(date_str) + timedelta(days=1)
            return dt.replace(hour=0, minute=0, second=0)
        else:
            # 正常时间
            return pd.to_datetime(f"{date_str} {hour:02d}:00:00")
    except Exception as e:
        print(f"⚠️  时间解析失败: date={date_str}, hour={hour}, error={e}")
        return pd.NaT


def load_and_preprocess_data():
    """加载并预处理原始数据"""
    print("📊 加载原始数据...")
    
    # 读取两个月的数据
    df_sep = pd.read_csv(os.path.join(DATA_DIR, '观测站小时交通量-9.csv'))
    df_oct = pd.read_csv(os.path.join(DATA_DIR, '观测站小时交通量-10.csv'))
    
    df = pd.concat([df_sep, df_oct], ignore_index=True)
    print(f"   总数据量: {len(df)} 条记录")
    
    # 构建时间戳（处理24:00特殊情况）
    print("   解析时间戳（处理24:00特殊情况）...")
    df['datetime'] = df.apply(
        lambda row: parse_datetime_with_24hour(row['观测日期'], row['小时']), 
        axis=1
    )
    
    # 移除解析失败的时间
    initial_count = len(df)
    df = df.dropna(subset=['datetime'])
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"   ⚠️  移除了 {removed_count} 条时间解析失败的记录")
    
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 提取唯一站点列表
    stations = df['观测站编号'].unique()
    num_stations = len(stations)
    print(f"   站点数量: {num_stations}")
    
    # 创建站点索引映射
    station_to_idx = {station: idx for idx, station in enumerate(stations)}
    
    # 获取唯一时间点并建立时间索引映射
    unique_times = sorted(df['datetime'].unique())
    time_to_idx = {t: idx for idx, t in enumerate(unique_times)}
    num_time_steps = len(unique_times)
    
    print(f"   时间步数: {num_time_steps}")
    print(f"   时间范围: {unique_times[0]} ~ {unique_times[-1]}")
    
    # 初始化4维特征数组 [时间步, 站点数, 特征数]
    num_features = 4
    data_array = np.zeros((num_time_steps, num_stations, num_features))
    
    print(f"   特征维度: {num_features}")
    
    # 填充数据（优化版：避免重复查找）
    print("   填充数据...")
    processed_count = 0
    
    for _, row in df.iterrows():
        station = row['观测站编号']
        direction = row['行驶方向']
        dt = row['datetime']
        
        if station not in station_to_idx or dt not in time_to_idx:
            continue
        
        station_idx = station_to_idx[station]
        time_idx = time_to_idx[dt]
        
        passenger = float(row['小客车'])
        total_cars = float(row['汽车自然数'])
        non_passenger = total_cars - passenger
        
        # 根据方向填充对应的特征
        if direction == '上行':
            data_array[time_idx, station_idx, 0] = passenger  # 小客车上行
            data_array[time_idx, station_idx, 2] = non_passenger  # 非小客车上行
        elif direction == '下行':
            data_array[time_idx, station_idx, 1] = passenger  # 小客车下行
            data_array[time_idx, station_idx, 3] = non_passenger  # 非小客车下行
        # 断面数据忽略
        
        processed_count += 1
    
    print(f"   已处理 {processed_count} 条记录")
    print(f"✅ 数据预处理完成!")
    print(f"   数据形状: {data_array.shape}")
    
    # 检查是否有全零的站点或时间点
    zero_stations = np.where(data_array.sum(axis=(0, 2)) == 0)[0]
    if len(zero_stations) > 0:
        print(f"   ⚠️  警告: 发现 {len(zero_stations)} 个站点在所有时间点均为0")
    
    return data_array, stations, station_to_idx, unique_times


def create_sliding_windows(data, P, Q):
    """创建滑动窗口样本"""
    num_time_steps, num_stations, num_features = data.shape
    num_samples = num_time_steps - P - Q + 1
    
    # 展平站点和特征维度: (N*F)
    data_flat = data.reshape(num_time_steps, num_stations * num_features)
    
    X = np.zeros((num_samples, P, num_stations * num_features))
    Y = np.zeros((num_samples, Q, num_stations * num_features))
    
    for i in range(num_samples):
        X[i] = data_flat[i : i + P]
        Y[i] = data_flat[i + P : i + P + Q]
    
    print(f"   样本数量: {num_samples}")
    print(f"   X形状: {X.shape}, Y形状: {Y.shape}")
    
    return X, Y


def generate_temporal_embedding(unique_times, P, Q):
    """生成时间嵌入 (dayofweek, timeofday)"""
    num_samples = len(unique_times) - P - Q + 1
    
    TE_windows = np.zeros((num_samples, P + Q, 2), dtype=np.int32)
    
    for i in range(num_samples):
        window_times = unique_times[i : i + P + Q]
        
        for j, t in enumerate(window_times):
            # day of week (0-6, Monday=0)
            dayofweek = t.weekday()
            # time of day (0-23)
            timeofday = t.hour
            
            TE_windows[i, j, 0] = dayofweek
            TE_windows[i, j, 1] = timeofday
    
    return TE_windows


def save_spatial_embedding(stations, output_dir):
    """保存空间嵌入文件 (使用随机初始化或基于距离)"""
    N = len(stations)
    D = 64  # 嵌入维度
    
    # 简单起见，使用随机初始化
    SE = np.random.randn(N, D).astype(np.float32)
    
    se_file = os.path.join(output_dir, 'SE.txt')
    with open(se_file, 'w') as f:
        f.write(f"{N} {D}\n")
        for i in range(N):
            f.write(f"{i} " + " ".join(map(str, SE[i])) + "\n")
    
    print(f"   空间嵌入已保存到: {se_file}")
    return SE


def main():
    """主函数"""
    print("=" * 60)
    print("GMAN 高速公路交通流量数据预处理")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载和预处理数据
    data_array, stations, station_to_idx, unique_times = load_and_preprocess_data()
    
    # 2. 创建滑动窗口
    print("\n🔄 创建滑动窗口样本...")
    X, Y = create_sliding_windows(data_array, P, Q)
    
    # 3. 划分训练/验证/测试集
    num_samples = X.shape[0]
    train_size = int(num_samples * TRAIN_RATIO)
    val_size = int(num_samples * VAL_RATIO)
    
    trainX = X[:train_size]
    valX = X[train_size : train_size + val_size]
    testX = X[train_size + val_size :]
    
    trainY = Y[:train_size]
    valY = Y[train_size : train_size + val_size]
    testY = Y[train_size + val_size :]
    
    print(f"\n📊 数据集划分:")
    print(f"   训练集: {trainX.shape[0]} 样本")
    print(f"   验证集: {valX.shape[0]} 样本")
    print(f"   测试集: {testX.shape[0]} 样本")
    
    # 4. 归一化 (基于训练集统计)
    mean = np.mean(trainX)
    std = np.std(trainX)
    
    if std == 0:
        std = 1.0
    
    # 对X和Y都进行归一化
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std
    
    trainY = (trainY - mean) / std
    valY = (valY - mean) / std
    testY = (testY - mean) / std
    
    print(f"\n📈 归一化参数:")
    print(f"   Mean: {mean:.4f}")
    print(f"   Std: {std:.4f}")
    
    # 5. 生成时间嵌入
    print("\n⏰ 生成时间嵌入...")
    # TE应该与X/Y有相同的样本数
    trainTE = generate_temporal_embedding(unique_times, P, Q)[:train_size]
    valTE = generate_temporal_embedding(unique_times, P, Q)[train_size : train_size + val_size]
    testTE = generate_temporal_embedding(unique_times, P, Q)[train_size + val_size :]
    
    print(f"   trainTE: {trainTE.shape}")
    print(f"   valTE: {valTE.shape}")
    print(f"   testTE: {testTE.shape}")
    
    # 6. 保存空间嵌入
    print("\n🌐 生成空间嵌入...")
    SE = save_spatial_embedding(stations, OUTPUT_DIR)
    
    # 7. 保存所有数据
    print("\n💾 保存数据...")
    np.savez(
        os.path.join(OUTPUT_DIR, 'highway_data.npz'),
        trainX=trainX,
        trainTE=trainTE,
        trainY=trainY,
        valX=valX,
        valTE=valTE,
        valY=valY,
        testX=testX,
        testTE=testTE,
        testY=testY,
        SE=SE,
        mean=np.array([mean]),
        std=np.array([std])
    )
    
    # 保存站点映射
    station_mapping = pd.DataFrame({
        'station_id': range(len(stations)),
        'station_code': stations
    })
    station_mapping.to_csv(os.path.join(OUTPUT_DIR, 'station_mapping.csv'), index=False)
    
    print(f"\n✅ 数据预处理完成!")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   数据文件: highway_data.npz")
    print("=" * 60)


if __name__ == '__main__':
    main()
