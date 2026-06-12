#!/usr/bin/env python3
"""
数据预处理脚本：从原始交通流量数据中提取目标特征并准备训练数据

目标特征：
1. 小客车上行
2. 小客车下行  
3. 非小客车上行 = (汽车自然数 - 小客车) 上行
4. 非小客车下行 = (汽车自然数 - 小客车) 下行

时序配置：32输入8输出（使用过去32小时预测未来8小时）
支持：自动合并多个CSV文件
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import glob

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入统一配置模块
import config


class Config:
    """配置类（保留用于向后兼容）"""
    HF_ENDPOINT = 'https://huggingface.co'
    PREPROCESSED_DIR = 'dataset/preprocessed'
    CONTEXT_LEN = 32
    HORIZON_LEN = 8
    
    @staticmethod
    def ensure_directories():
        """确保必要目录存在"""
        os.makedirs(Config.PREPROCESSED_DIR, exist_ok=True)
        os.makedirs('dataset', exist_ok=True)
    
    @staticmethod
    def get_raw_data_files():
        """获取所有原始数据文件，支持合并多个文件"""
        dataset_dir = 'dataset'
        
        # 查找所有包含"观测站"的CSV文件
        csv_files = glob.glob(os.path.join(dataset_dir, '*观测站*.csv'))
        csv_files.extend(glob.glob(os.path.join(dataset_dir, '*.csv')))
        
        # 去重
        csv_files = list(set(csv_files))
        
        # 排除预处理目录
        csv_files = [f for f in csv_files if 'preprocessed' not in f]
        
        if not csv_files:
            raise FileNotFoundError(f"未找到数据文件在 {dataset_dir} 目录")
        
        # 按文件名排序
        csv_files.sort()
        
        print(f"找到 {len(csv_files)} 个数据文件:")
        for f in csv_files:
            print(f"  - {os.path.basename(f)}")
        
        return csv_files
    
    @staticmethod
    def merge_data_files(file_list):
        """合并多个CSV文件"""
        print("\n正在合并数据文件...")
        
        dfs = []
        total_rows = 0
        
        for file_path in file_list:
            print(f"  加载: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path, encoding='utf-8')
            rows = len(df)
            total_rows += rows
            print(f"    行数: {rows}")
            dfs.append(df)
        
        # 合并所有数据
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # 去重（基于所有列）
        before_dedup = len(merged_df)
        merged_df = merged_df.drop_duplicates()
        after_dedup = len(merged_df)
        
        print(f"\n  总行数: {total_rows}")
        print(f"  去重后: {after_dedup} (移除 {before_dedup - after_dedup} 条重复)")
        
        return merged_df


# 设置 HuggingFace 镜像加速
os.environ['HF_ENDPOINT'] = config.HF_ENDPOINT


def load_and_process_data(csv_path=None):
    """加载并合并所有原始交通流量数据"""
    # 如果没有提供路径，自动检测并合并所有CSV文件
    if csv_path is None:
        import glob
        
        dataset_dir = config.DATASET_DIR
        csv_files = glob.glob(os.path.join(dataset_dir, '*.csv'))
        
        # 排除预处理目录
        csv_files = [f for f in csv_files if 'preprocessed' not in f]
        
        if not csv_files:
            raise FileNotFoundError(f"在 {dataset_dir} 目录下未找到CSV文件")
        
        # 按文件名排序
        csv_files.sort()
        
        print(f"\n发现 {len(csv_files)} 个数据文件:")
        for f in csv_files:
            print(f"  - {os.path.basename(f)}")
        
        # 如果只有一个文件，直接读取
        if len(csv_files) == 1:
            csv_path = csv_files[0]
            print(f"\n正在加载数据: {csv_path}")
            df = pd.read_csv(csv_path, encoding='utf-8')
        else:
            # 合并多个文件
            print("\n正在合并多个数据文件...")
            dfs = []
            total_rows = 0
            
            for file_path in csv_files:
                print(f"  加载: {os.path.basename(file_path)}")
                try:
                    df_temp = pd.read_csv(file_path, encoding='utf-8')
                    rows = len(df_temp)
                    total_rows += rows
                    print(f"    行数: {rows}")
                    dfs.append(df_temp)
                except Exception as e:
                    print(f"    ⚠️  警告: 加载失败 - {e}")
            
            if not dfs:
                raise ValueError("所有文件都加载失败")
            
            # 合并所有数据
            df = pd.concat(dfs, ignore_index=True)
            
            # 去重
            before_dedup = len(df)
            df = df.drop_duplicates()
            after_dedup = len(df)
            
            print(f"\n  合并前总行数: {total_rows}")
            print(f"  去重后行数: {after_dedup} (移除 {before_dedup - after_dedup} 条重复)")
            
            csv_path = f"{len(csv_files)}个文件合并"
    else:
        # 使用指定的单个文件
        print(f"正在加载数据: {csv_path}")
        df = pd.read_csv(csv_path, encoding='utf-8')
    
    print(f"\n原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查必要的列是否存在
    required_columns = ['观测日期', '小时', '观测站编号', '行驶方向', '小客车', '汽车自然数']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")
    
    return df


def create_timestamp_with_hour24_handling(df: pd.DataFrame) -> pd.Series:
    """
    创建时间戳，正确处理小时=24的情况
    将小时24转换为第二天的00:00
    """
    # 复制数据避免警告
    df = df.copy()
    
    # 确保小时是整数
    hours = df['小时'].astype(int)
    
    # 创建日期时间列
    dates = pd.to_datetime(df['观测日期'].astype(str))
    
    # 处理小时=24的情况
    mask_hour_24 = hours == 24
    
    if mask_hour_24.any():
        print(f"  检测到 {mask_hour_24.sum()} 行小时=24的数据，正在转换为次日00:00...")
        # 将小时24改为0，日期加1天
        hours_processed = hours.copy()
        hours_processed[mask_hour_24] = 0
        dates_processed = dates.copy()
        dates_processed[mask_hour_24] = dates_processed[mask_hour_24] + pd.Timedelta(days=1)
    else:
        hours_processed = hours
        dates_processed = dates
    
    # 创建时间戳
    timestamps = pd.to_datetime(
        dates_processed.dt.strftime('%Y-%m-%d') + ' ' + 
        hours_processed.astype(str).str.zfill(2) + ':00:00'
    )
    
    return timestamps


def extract_target_features(df: pd.DataFrame):
    """
    提取4维目标特征
    
    Returns:
        dict: 按观测站分组的特征字典
              {station_id: DataFrame with columns ['timestamp', 'passenger_car_up', 'passenger_car_down', 
                                                    'non_passenger_car_up', 'non_passenger_car_down']}
    """
    print("\n正在提取目标特征...")
    
    # 复制数据避免修改原数据
    df = df.copy()
    
    # 创建时间戳列（处理小时24问题）
    print("  创建时间戳...")
    df['timestamp'] = create_timestamp_with_hour24_handling(df)
    
    # 计算非小客车数量
    print("  计算非小客车流量...")
    df['非小客车'] = df['汽车自然数'] - df['小客车']
    
    # 确保非负
    df['非小客车'] = df['非小客车'].clip(lower=0)
    
    print(f"  时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"  总数据点: {len(df)}")
    
    # 按观测站分组
    station_data = {}
    
    # 获取所有站点
    stations = df['观测站编号'].unique()
    print(f"  发现 {len(stations)} 个观测站")
    
    skipped_stations = 0
    
    for station_id in stations:
        group = df[df['观测站编号'] == station_id].copy()
        station_name = group['观测站名称'].iloc[0] if len(group) > 0 else str(station_id)
        
        # 分离上行和下行数据
        up_data = group[group['行驶方向'] == '上行'].copy()
        down_data = group[group['行驶方向'] == '下行'].copy()
        
        if len(up_data) == 0 or len(down_data) == 0:
            skipped_stations += 1
            continue
        
        # 按时间排序
        up_data = up_data.sort_values('timestamp')
        down_data = down_data.sort_values('timestamp')
        
        # 获取所有唯一时间戳
        all_timestamps = sorted(set(up_data['timestamp'].tolist() + down_data['timestamp'].tolist()))
        
        # 创建完整的时间序列DataFrame
        result_df = pd.DataFrame({'timestamp': all_timestamps})
        
        # 合并各项数据
        up_passenger = up_data[['timestamp', '小客车']].rename(columns={'小客车': 'passenger_car_up'})
        result_df = result_df.merge(up_passenger, on='timestamp', how='left')
        
        down_passenger = down_data[['timestamp', '小客车']].rename(columns={'小客车': 'passenger_car_down'})
        result_df = result_df.merge(down_passenger, on='timestamp', how='left')
        
        up_non_passenger = up_data[['timestamp', '非小客车']].rename(columns={'非小客车': 'non_passenger_car_up'})
        result_df = result_df.merge(up_non_passenger, on='timestamp', how='left')
        
        down_non_passenger = down_data[['timestamp', '非小客车']].rename(columns={'非小客车': 'non_passenger_car_down'})
        result_df = result_df.merge(down_non_passenger, on='timestamp', how='left')
        
        # 按时间排序
        result_df = result_df.sort_values('timestamp').reset_index(drop=True)
        
        # 填充缺失值（使用前向填充，然后后向填充，最后填0）
        # 使用新的方法避免FutureWarning
        result_df = result_df.ffill().bfill().fillna(0)
        
        # 只保留数值列
        feature_cols = ['passenger_car_up', 'passenger_car_down', 'non_passenger_car_up', 'non_passenger_car_down']
        result_df = result_df[['timestamp'] + feature_cols]
        
        station_data[station_id] = {
            'data': result_df,
            'name': station_name,
            'total_hours': len(result_df)
        }
    
    print(f"✓ 成功处理 {len(station_data)} 个观测站 (跳过 {skipped_stations} 个数据不完整的站点)")
    
    # 显示数据统计
    if station_data:
        lengths = [info['total_hours'] for info in station_data.values()]
        print(f"  站点数据时长范围: {min(lengths)} - {max(lengths)} 小时")
        print(f"  平均时长: {sum(lengths)/len(lengths):.0f} 小时")
    
    return station_data


def prepare_series_for_timesfm(station_data: dict, context_len: int = 32, horizon_len: int = 8):
    """
    为TimesFM准备时间序列数据
    
    Args:
        station_data: 站点数据字典，格式为 {station_id: {'data': DataFrame, 'name': str, 'total_hours': int}}
        context_len: 上下文长度（输入窗口）- TimesFM 2.5 要求 >= 32
        horizon_len: 预测 horizon
        
    Returns:
        train_series: 训练序列列表，每个元素是 (seq_len,) 的数组
        val_series: 验证序列列表
        test_series: 测试序列列表
        metadata: 元数据信息
    """
    print(f"\n准备时间序列数据 (context_len={context_len}, horizon_len={horizon_len})...")
    print(f"配置: {context_len}输入{horizon_len}输出")
    print(f"注意: TimesFM 2.5 要求 context_len >= 32 且是 32 的倍数")
    
    train_series = []
    val_series = []
    test_series = []
    station_ids = []
    
    min_length = context_len + horizon_len  # 最少需要 40 个时间点
    
    for station_id, station_info in station_data.items():
        # 提取实际的 DataFrame
        df = station_info['data']
        
        # 确保列名是字符串类型
        df.columns = [str(col) for col in df.columns]
        
        # 提取4维特征
        feature_cols = ['passenger_car_up', 'passenger_car_down', 
                       'non_passenger_car_up', 'non_passenger_car_down']
        
        # 检查列是否存在
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            print(f"警告: 站点 {station_id} 缺少列 {missing_cols}，跳过")
            continue
        
        features = df[feature_cols].values
        
        if len(features) < min_length:
            print(f"警告: 站点 {station_id} 数据长度 {len(features)} < {min_length}，跳过")
            continue
        
        # 将多变量序列拆分为4个单变量序列（TimesFM采用Channel Independence策略）
        for feat_idx, feat_name in enumerate(feature_cols):
            series = features[:, feat_idx]
            
            # 划分训练/验证/测试集 (70%/15%/15%)
            total_len = len(series)
            train_end = int(total_len * 0.7)
            val_end = int(total_len * 0.85)
            
            train_data = series[:train_end]
            val_data = series[train_end:val_end]
            test_data = series[val_end:]
            
            # 只保留足够长的序列
            if len(train_data) >= min_length:
                train_series.append(train_data)
                station_ids.append(f"{station_id}_{feat_name}_train")
            
            if len(val_data) >= min_length:
                val_series.append(val_data)
                station_ids.append(f"{station_id}_{feat_name}_val")
            
            if len(test_data) >= min_length:
                test_series.append(test_data)
                station_ids.append(f"{station_id}_{feat_name}_test")
    
    print(f"训练序列数: {len(train_series)}")
    print(f"验证序列数: {len(val_series)}")
    print(f"测试序列数: {len(test_series)}")
    
    metadata = {
        'station_ids': station_ids,
        'context_len': context_len,
        'horizon_len': horizon_len,
        'feature_names': ['passenger_car_up', 'passenger_car_down', 
                         'non_passenger_car_up', 'non_passenger_car_down'],
        'config': f'{context_len}_input_{horizon_len}_output'
    }
    
    return train_series, val_series, test_series, metadata


def save_preprocessed_data(output_dir: str, train_series, val_series, test_series, metadata):
    """保存预处理后的数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存序列数据
    np.savez(os.path.join(output_dir, 'train_series.npz'), 
             data=np.array(train_series, dtype=object))
    np.savez(os.path.join(output_dir, 'val_series.npz'), 
             data=np.array(val_series, dtype=object))
    np.savez(os.path.join(output_dir, 'test_series.npz'), 
             data=np.array(test_series, dtype=object))
    
    # 保存元数据
    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 数据已保存到: {output_dir}")


def main():
    # 从配置文件获取路径和参数
    output_dir = config.PREPROCESSED_DIR
    context_len = config.CONTEXT_LEN  # 32
    horizon_len = config.HORIZON_LEN  # 8
    
    print("=" * 60)
    print("TimesFM 数据预处理")
    print("=" * 60)
    print(f"配置: {context_len}输入{horizon_len}输出")
    print(f"注意: TimesFM 2.5 要求 context_len >= 32 且是 32 的倍数")
    
    # 确保目录存在
    config.ensure_directories()
    
    # 步骤1: 加载并合并数据（自动检测所有CSV文件）
    df = load_and_process_data()
    
    # 步骤2: 提取目标特征
    station_data = extract_target_features(df)
    
    # 步骤3: 准备TimesFM格式的数据
    train_series, val_series, test_series, metadata = prepare_series_for_timesfm(
        station_data, context_len=context_len, horizon_len=horizon_len
    )
    
    # 步骤4: 保存数据
    save_preprocessed_data(output_dir, train_series, val_series, test_series, metadata)
    
    print("\n✅ 数据预处理完成！")
    print(f"   - 训练序列: {len(train_series)} 个")
    print(f"   - 验证序列: {len(val_series)} 个")
    print(f"   - 测试序列: {len(test_series)} 个")
    print(f"   - 配置: {context_len}输入{horizon_len}输出")
    print(f"   - 输出目录: {output_dir}")


if __name__ == '__main__':
    main()
