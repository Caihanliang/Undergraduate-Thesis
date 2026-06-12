# Copyright 2024 Arjun Ashok
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
高速公路交通流量数据预处理脚本
将CSV格式的交通流量数据转换为GluonTS的ListDataset格式(JSONL)

数据集说明:
- 输入: 观测站小时交通量-9.csv 和 观测站小时交通量-10.csv
- 预测目标: 
  1. 小客车上行
  2. 小客车下行
  3. 非小客车上行 (汽车自然数 - 小客车)
  4. 非小客车下行 (汽车自然数 - 小客车)
- 时序配置: 8小时输入, 8小时输出 (但Lag-Llama建议context_length从32开始)
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from gluonts.dataset.common import ListDataset
from tqdm import tqdm

# ==================== 配置参数 ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "processed_data"

# 时间序列配置
CONTEXT_LEN = 8   # 修改为8小时输入（实现8输入8输出）
PREDICTION_LEN = 8  # 预测长度保持8小时
FREQ = 'H'  # 小时级频率

# 数据文件
DATA_FILES = [
    DATASET_DIR / "观测站小时交通量-9.csv",
    DATASET_DIR / "观测站小时交通量-10.csv"
]

# 特征定义
FEATURES = {
    'passenger_up': {'name': '小客车上行', 'direction': '上行', 'calc': 'direct'},
    'passenger_down': {'name': '小客车下行', 'direction': '下行', 'calc': 'direct'},
    'non_passenger_up': {'name': '非小客车上行', 'direction': '上行', 'calc': 'derived'},
    'non_passenger_down': {'name': '非小客车下行', 'direction': '下行', 'calc': 'derived'}
}


def load_and_merge_data():
    """加载并合并所有CSV文件"""
    print("正在加载数据文件...")
    dfs = []
    for file_path in DATA_FILES:
        if file_path.exists():
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"  已加载: {file_path.name}, 行数: {len(df)}")
        else:
            print(f"  警告: 文件不存在 - {file_path}")
    
    if not dfs:
        raise FileNotFoundError("未找到任何数据文件!")
    
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"合并后总行数: {len(merged_df)}")
    return merged_df


def preprocess_dataframe(df):
    """
    预处理DataFrame:
    1. 创建完整的时间戳
    2. 计算非小客车流量
    3. 按站点和方向分组
    """
    print("正在预处理数据...")
    
    # 处理"24:00"特殊情况: 将其转换为次日"00:00"
    print("  正在处理特殊时间格式(24:00)...")
    
    # 先复制原始列,避免修改原数据
    df = df.copy()
    
    # 检测并转换24:00的情况
    mask_24h = df['小时'] == 24
    if mask_24h.any():
        print(f"  发现 {mask_24h.sum()} 条24:00的记录,正在转换...")
        # 对于24:00,将观测日期加1天,小时设为0
        df.loc[mask_24h, '观测日期'] = (pd.to_datetime(df.loc[mask_24h, '观测日期'].astype(str)) + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
        df.loc[mask_24h, '小时'] = 0
    
    # 创建时间戳列 - 使用format参数提高解析速度
    time_strs = df['观测日期'].astype(str) + ' ' + df['小时'].astype(int).apply(lambda x: f"{x:02d}:00:00")
    df['timestamp'] = pd.to_datetime(time_strs, format='%Y-%m-%d %H:%M:%S')
    
    # 计算非小客车流量 (汽车自然数 - 小客车)
    df['非小客车'] = df['汽车自然数'] - df['小客车']
    
    # 确保数值列为float类型
    numeric_cols = ['小客车', '非小客车']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"  时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
    print(f"  站点数量: {df['观测站编号'].nunique()}")
    print(f"  方向类型: {df['行驶方向'].unique()}")
    
    return df


def create_time_series_for_station_feature(df, station_id, station_name, feature_key, feature_info):
    """
    为单个站点的单个特征创建时间序列
    
    Returns:
        dict: 包含start, target, item_id的字典,符合GluonTS格式
    """
    # 筛选特定站点和方向的数据
    mask = (df['观测站编号'] == station_id) & (df['行驶方向'] == feature_info['direction'])
    station_data = df[mask].copy()
    
    if len(station_data) == 0:
        return None
    
    # 按时间排序
    station_data = station_data.sort_values('timestamp')
    
    # 提取目标特征
    if feature_info['calc'] == 'direct':
        target_col = '小客车'
    else:  # derived
        target_col = '非小客车'
    
    # 检查是否有足够的连续数据
    if len(station_data) < (CONTEXT_LEN + PREDICTION_LEN):
        return None
    
    # 处理缺失值 (使用前向填充)
    series = station_data[target_col].fillna(method='ffill').fillna(method='bfill').values
    
    # 检查是否全为NaN
    if np.all(np.isnan(series)):
        return None
    
    # 替换剩余的NaN为0
    series = np.nan_to_num(series, nan=0.0)
    
    # 创建GluonTS格式的时间序列
    time_series = {
        'start': station_data['timestamp'].iloc[0],
        'target': series.tolist(),
        'item_id': f"{station_id}_{feature_key}"
    }
    
    return time_series


def create_sliding_windows(time_series_dict, context_len, prediction_len):
    """
    使用滑动窗口从长时间序列中生成多个样本
    
    Args:
        time_series_dict: 原始时间序列字典
        context_len: 上下文长度
        prediction_len: 预测长度
    
    Returns:
        list: 滑动窗口后的样本列表
    """
    samples = []
    target = time_series_dict['target']
    start_time = time_series_dict['start']
    item_id = time_series_dict['item_id']
    
    total_len = len(target)
    window_size = context_len + prediction_len
    
    if total_len < window_size:
        return samples
    
    # 生成滑动窗口
    for i in range(0, total_len - window_size + 1, 1):  # stride=1
        window_start = start_time + pd.Timedelta(hours=i)
        sample = {
            'start': window_start,
            'target': target[i:i+window_size],
            'item_id': f"{item_id}_window_{i}"
        }
        samples.append(sample)
    
    return samples


def main():
    """主函数: 执行完整的数据预处理流程"""
    print("=" * 80)
    print("Lag-Llama 高速公路交通流量数据预处理")
    print("=" * 80)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: 加载数据
    df = load_and_merge_data()
    
    # Step 2: 预处理
    df = preprocess_dataframe(df)
    
    # Step 3: 为每个站点和每个特征创建时间序列
    print("\n正在创建时间序列...")
    all_samples = []
    station_ids = df['观测站编号'].unique()
    
    print(f"  总站点数: {len(station_ids)}")
    print(f"  每个站点4个特征,预计生成约 {len(station_ids) * 4} 个基础序列")
    
    for station_id in tqdm(station_ids, desc="处理站点"):
        station_name = df[df['观测站编号'] == station_id]['观测站名称'].iloc[0]
        
        for feature_key, feature_info in FEATURES.items():
            # 创建基础时间序列
            ts = create_time_series_for_station_feature(
                df, station_id, station_name, feature_key, feature_info
            )
            
            if ts is None:
                continue
            
            # 使用滑动窗口生成多个样本
            windows = create_sliding_windows(ts, CONTEXT_LEN, PREDICTION_LEN)
            all_samples.extend(windows)
    
    print(f"\n总共生成 {len(all_samples)} 个样本")
    
    # Step 4: 划分训练集和测试集 - 关键修复:按时间划分,确保所有站点都有测试数据
    print("\n正在按时间划分训练集和测试集...")
    
    # 策略:对每个站点的每个特征,取最后20%的时间窗口作为测试集
    train_samples = []
    test_samples = []
    
    # 按站点ID和特征分组
    from collections import defaultdict
    grouped_samples = defaultdict(list)
    
    for sample in all_samples:
        # 提取站点ID和特征(从item_id中)
        item_id = sample['item_id']
        # item_id格式: {station_id}_{feature_key}_window_{idx}
        parts = item_id.split('_')
        if len(parts) >= 3:
            station_id = parts[0]
            feature_key = parts[1]
            group_key = f"{station_id}_{feature_key}"
            grouped_samples[group_key].append(sample)
    
    # 对每个组,按window_idx排序,取最后20%作为测试集
    for group_key, samples in grouped_samples.items():
        # 按window_idx排序
        samples.sort(key=lambda x: int(x['item_id'].split('_window_')[1]))
        
        split_idx = int(len(samples) * 0.8)
        train_samples.extend(samples[:split_idx])
        test_samples.extend(samples[split_idx:])
    
    print(f"  训练集样本数: {len(train_samples)}")
    print(f"  测试集样本数: {len(test_samples)}")
    print(f"  测试集涉及站点数: {len(set(s['item_id'].split('_')[0] for s in test_samples))}")
    
    # Step 5: 保存为JSONL格式
    print("\n正在保存数据...")
    
    train_file = OUTPUT_DIR / "train.jsonl"
    test_file = OUTPUT_DIR / "test.jsonl"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            # 将timestamp转换为ISO格式字符串
            sample_copy = sample.copy()
            sample_copy['start'] = sample_copy['start'].isoformat()
            f.write(json.dumps(sample_copy, ensure_ascii=False) + '\n')
    
    with open(test_file, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            sample_copy = sample.copy()
            sample_copy['start'] = sample_copy['start'].isoformat()
            f.write(json.dumps(sample_copy, ensure_ascii=False) + '\n')
    
    print(f"  训练集已保存至: {train_file}")
    print(f"  测试集已保存至: {test_file}")
    
    # Step 6: 保存站点映射文件(用于后续追溯)
    station_mapping = df[['观测站编号', '观测站名称']].drop_duplicates()
    mapping_file = OUTPUT_DIR / "station_mapping.csv"
    station_mapping.to_csv(mapping_file, index=False, encoding='utf-8-sig')
    print(f"  站点映射已保存至: {mapping_file}")
    
    # Step 7: 输出统计信息
    print("\n" + "=" * 80)
    print("数据预处理完成!")
    print("=" * 80)
    print(f"  上下文长度 (context_length): {CONTEXT_LEN}")
    print(f"  预测长度 (prediction_length): {PREDICTION_LEN}")
    print(f"  数据频率: {FREQ}")
    print(f"  总样本数: {len(all_samples)}")
    print(f"  特征列表:")
    for key, info in FEATURES.items():
        print(f"    - {key}: {info['name']} ({info['direction']})")
    print("=" * 80)


if __name__ == "__main__":
    main()
