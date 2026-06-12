#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time-MoE 数据预处理脚本
将原始交通量CSV数据转换为Time-MoE所需的格式
预测目标：小客车上行、小客车下行、(汽车自然数-小客车)上行、(汽车自然数-小客车)下行
时序配置：8输入8输出
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def load_and_process_data(data_dir):
    """加载并处理原始数据"""
    print("📊 正在加载数据...")
    
    # 读取两个月份的数据
    df_sep = pd.read_csv(os.path.join(data_dir, '观测站小时交通量-9.csv'))
    df_oct = pd.read_csv(os.path.join(data_dir, '观测站小时交通量-10.csv'))
    
    # 合并数据
    df = pd.concat([df_sep, df_oct], ignore_index=True)
    
    print(f"✅ 数据加载完成，总行数: {len(df)}")
    print(f"   列名: {df.columns.tolist()}")
    
    return df


def extract_target_features(df):
    """提取目标特征 - 按站点分组处理"""
    print("\n🎯 正在提取目标特征（按站点分组）...")
    
    # 需要的列
    required_cols = ['观测日期', '小时', '观测站编号', '行驶方向', '小客车', '汽车自然数']
    
    # 检查必需的列是否存在
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")
    
    # 创建时间戳
    df['datetime'] = pd.to_datetime(df['观测日期']) + pd.to_timedelta(df['小时'], unit='h')
    
    # 计算 (汽车自然数 - 小客车)
    df['其他车辆'] = df['汽车自然数'] - df['小客车']
    
    # 获取所有站点列表
    stations = df['观测站编号'].unique()
    print(f"   📍 共发现 {len(stations)} 个站点")
    
    # 按站点处理
    all_station_data = []
    
    for station_idx, station_id in enumerate(stations):
        if (station_idx + 1) % 20 == 0:
            print(f"   处理进度: {station_idx + 1}/{len(stations)}")
        
        # 提取该站点的数据
        df_station = df[df['观测站编号'] == station_id].copy()
        
        if len(df_station) == 0:
            continue
        
        # 分别处理上行和下行
        station_features = {'station_id': station_id, 'datetime': None}
        
        for direction in ['上行', '下行']:
            df_dir = df_station[df_station['行驶方向'] == direction].copy()
            
            if len(df_dir) == 0:
                # 如果某个方向缺失，用None标记，后续跳过或处理
                station_features[f'小客车_{direction}'] = None
                station_features[f'其他车辆_{direction}'] = None
                continue
            
            # 按时间排序
            df_dir = df_dir.sort_values('datetime').reset_index(drop=True)
            
            # 记录时间列（以上行为准，或者取并集，这里简化为只要有一个方向有数据就记录时间）
            if station_features['datetime'] is None:
                station_features['datetime'] = df_dir['datetime'].values
            
            # 提取特征
            station_features[f'小客车_{direction}'] = df_dir['小客车'].values
            station_features[f'其他车辆_{direction}'] = df_dir['其他车辆'].values
        
        # 只保留有完整数据的站点 (至少要有时间索引)
        if station_features['datetime'] is not None:
            all_station_data.append(station_features)
    
    print(f"   ✅ 成功处理 {len(all_station_data)} 个站点")
    
    return all_station_data


def create_time_series_sequences_per_station(station_data_list, context_length=8, prediction_length=8, stride=1):
    """为每个站点的每个特征创建独立的时间序列滑动窗口（Channel Independence）"""
    print(f"\n🔄 正在创建滑动窗口 (context={context_length}, prediction={prediction_length}, stride={stride})...")
    print(f"   📊 采用 Channel Independence 策略：每个特征作为独立序列")
    
    sequences = []
    timestamps = []
    window_size = context_length + prediction_length
    
    feature_names = ['小客车_上行', '其他车辆_上行', '小客车_下行', '其他车辆_下行']
    
    total_samples = 0
    total_sequences = 0
    
    for station_idx, station_data in enumerate(station_data_list):
        if (station_idx + 1) % 20 == 0:
            print(f"   站点处理进度: {station_idx + 1}/{len(station_data_list)}, 已生成 {total_samples} 样本")
        
        station_id = station_data['station_id']
        datetimes = station_data['datetime']
        
        # 对每个特征分别处理（Channel Independence）
        for feat_idx, feat_name in enumerate(feature_names):
            try:
                # 提取单个特征的时间序列 [T]
                feature_values = station_data[feat_name]
                
                if feature_values is None or len(feature_values) < window_size:
                    continue
                
                # 注意：Time-MoE 会在训练时自动对每个序列进行 zero normalization
                # 所以我们这里直接使用原始数值，不进行预标准化
                feature_array = np.array(feature_values)
                
                # 创建滑动窗口
                n_points = len(feature_array)
                
                for i in range(0, n_points - window_size + 1, stride):
                    window = feature_array[i:i + window_size]
                    
                    seq = {
                        'station_id': station_id,
                        'feature': feat_name,
                        'input': window[:context_length].tolist(),  # [8]
                        'target': window[context_length:].tolist(),  # [8]
                        'timestamp_start': str(datetimes[i]),
                        'timestamp_end': str(datetimes[i + window_size - 1])
                    }
                    sequences.append(seq)
                    timestamps.append({
                        'station_id': station_id,
                        'feature': feat_name,
                        'start': str(datetimes[i]),
                        'end': str(datetimes[i + window_size - 1])
                    })
                    total_samples += 1
                
                total_sequences += 1
            
            except Exception as e:
                print(f"   ⚠️  警告: 站点 {station_id} 的特征 {feat_name} 处理失败: {e}")
                continue
    
    print(f"✅ 创建了 {len(sequences)} 个序列样本（{total_sequences} 个单变量序列）")
    
    return sequences, timestamps, feature_names


def save_to_jsonl(sequences, output_path):
    """保存为JSONL格式（Time-MoE训练所需 - 单变量序列）"""
    print(f"\n💾 正在保存到 {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for seq in sequences:
            # Time-MoE需要完整的单变量序列 [input + target]
            # 现在是1D数组：[8] + [8] = [16]
            record = {
                'station_id': seq['station_id'],
                'feature': seq['feature'],
                'sequence': seq['input'] + seq['target']  # 总共 16 个值（单变量）
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存 {len(sequences)} 条记录到 {output_path}")


def save_train_test_split(sequences, train_ratio=0.7, val_ratio=0.15, output_dir='./processed_data'):
    """划分训练集、验证集和测试集 - 按站点分组保持时间连续性"""
    print(f"\n📊 正在划分数据集 (train={train_ratio}, val={val_ratio}, test={1-train_ratio-val_ratio})...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 按站点分组
    station_groups = {}
    for seq in sequences:
        station_id = seq['station_id']
        if station_id not in station_groups:
            station_groups[station_id] = []
        station_groups[station_id].append(seq)
    
    train_seqs = []
    val_seqs = []
    test_seqs = []
    
    # 对每个站点按时间顺序划分
    for station_id, station_seqs in station_groups.items():
        # 已经按时间排序，直接按比例划分
        n_total = len(station_seqs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_seqs.extend(station_seqs[:n_train])
        val_seqs.extend(station_seqs[n_train:n_train + n_val])
        test_seqs.extend(station_seqs[n_train + n_val:])
    
    print(f"   训练集: {len(train_seqs)} 样本")
    print(f"   验证集: {len(val_seqs)} 样本")
    print(f"   测试集: {len(test_seqs)} 样本")
    
    # 保存为JSONL格式
    save_to_jsonl(train_seqs, os.path.join(output_dir, 'train.jsonl'))
    save_to_jsonl(val_seqs, os.path.join(output_dir, 'val.jsonl'))
    save_to_jsonl(test_seqs, os.path.join(output_dir, 'test.jsonl'))
    
    # 同时保存为pickle格式（用于快速加载）
    import pickle
    with open(os.path.join(output_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train_seqs, f)
    with open(os.path.join(output_dir, 'val.pkl'), 'wb') as f:
        pickle.dump(val_seqs, f)
    with open(os.path.join(output_dir, 'test.pkl'), 'wb') as f:
        pickle.dump(test_seqs, f)
    
    print(f"✅ 数据集已保存到 {output_dir}")
    
    return train_seqs, val_seqs, test_seqs


def main():
    """主函数"""
    print("="*60)
    print("Time-MoE 数据预处理")
    print("="*60)
    
    # 配置
    config = {
        'context_length': 8,
        'prediction_length': 8,
        'stride': 1,  # 小数据集建议使用1
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'data_dir': './dataset',
        'output_dir': './processed_data'
    }
    
    # 1. 加载数据
    df = load_and_process_data(config['data_dir'])
    
    # 2. 提取目标特征（按站点分组）
    station_data_list = extract_target_features(df)
    
    # 3. 创建滑动窗口序列（按站点分别处理）
    sequences, timestamps, feature_names = create_time_series_sequences_per_station(
        station_data_list,
        context_length=config['context_length'],
        prediction_length=config['prediction_length'],
        stride=config['stride']
    )
    
    if len(sequences) == 0:
        print("\n❌ 错误: 未生成任何序列样本，请检查数据")
        return
    
    # 4. 保存元数据
    metadata = {
        'feature_columns': feature_names,
        'num_stations': len(station_data_list),
        'config': config
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(os.path.join(config['output_dir'], 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\n📝 元数据已保存")
    
    # 5. 划分并保存数据集
    train_seqs, val_seqs, test_seqs = save_train_test_split(
        sequences,
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        output_dir=config['output_dir']
    )
    
    print("\n" + "="*60)
    print("✅ 数据预处理完成！")
    print("="*60)
    print(f"\n📁 输出目录: {config['output_dir']}")
    print(f"   - train.jsonl: 训练集 ({len(train_seqs)} 样本)")
    print(f"   - val.jsonl: 验证集 ({len(val_seqs)} 样本)")
    print(f"   - test.jsonl: 测试集 ({len(test_seqs)} 样本)")
    print(f"   - metadata.json: 元数据")
    print(f"\n🎯 下一步: 使用以下命令开始训练")
    print(f"   python torch_dist_run.py main.py -d {config['output_dir']}/train.jsonl")
    print("="*60)


if __name__ == '__main__':
    main()
