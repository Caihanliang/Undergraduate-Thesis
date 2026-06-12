"""
python preprocess_data.py
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
import glob

def load_and_merge_datasets(dataset_dir):
    """
    Load and merge all CSV files from the dataset directory
    """
    print(f"Loading datasets from {dataset_dir}")
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(dataset_dir, '*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in sorted(csv_files):
        print(f"  - {os.path.basename(f)}")
    
    # Load and concatenate all CSV files
    dfs = []
    for csv_file in sorted(csv_files):
        print(f"\nLoading {os.path.basename(csv_file)}...")
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)[:8]}...")  # Show first 8 columns
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠️  Error loading {csv_file}: {e}")
    
    if not dfs:
        raise ValueError("No valid dataframes loaded")
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Merged dataset shape: {merged_df.shape}")
    
    return merged_df

def preprocess_traffic_data(df):
    """
    Preprocess traffic flow data for MOMENT model
    Features: 
    1. Small passenger car up (小客车上行)
    2. Small passenger car down (小客车下行)
    3. Non-passenger car up (汽车自然数-小客车)上行
    4. Non-passenger car down (汽车自然数-小客车)下行
    """
    print("\n" + "="*60)
    print("Preprocessing traffic data...")
    print("="*60)
    
    # Check required columns
    required_cols = ['观测日期', '小时', '观测站编号', '观测站名称', '行驶方向', 
                     '小客车', '汽车自然数']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}\nAvailable columns: {list(df.columns)}")
    
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Unique stations: {df['观测站编号'].nunique()}")
    print(f"Time range: {df['观测日期'].min()} to {df['观测日期'].max()}")
    
    # Create datetime column from date and hour
    # Handle hour 24 by converting it to next day 00:00:00
    def create_datetime(row):
        date_str = str(row['观测日期'])
        hour = int(row['小时'])
        
        if hour == 24:
            # Convert 24:00 to next day 00:00
            next_day = pd.to_datetime(date_str) + pd.Timedelta(days=1)
            return next_day.replace(hour=0, minute=0, second=0)
        else:
            return pd.to_datetime(f"{date_str} {hour:02d}:00:00")
    
    print("Creating datetime column (handling hour=24)...")
    df['时间'] = df.apply(create_datetime, axis=1)
    
    # Calculate non-passenger cars (汽车自然数 - 小客车)
    df['非小客车'] = df['汽车自然数'] - df['小客车']
    
    # Ensure no negative values
    df['非小客车'] = df['非小客车'].clip(lower=0)
    
    # Filter for up direction (上行)
    df_up = df[df['行驶方向'] == '上行'][[
        '时间', '观测站编号', '观测站名称', 
        '小客车', '非小客车'
    ]].copy()
    df_up.rename(columns={
        '小客车': '小客车上行',
        '非小客车': '非小客车上行'
    }, inplace=True)
    
    # Filter for down direction (下行)
    df_down = df[df['行驶方向'] == '下行'][[
        '时间', '观测站编号', '观测站名称',
        '小客车', '非小客车'
    ]].copy()
    df_down.rename(columns={
        '小客车': '小客车下行',
        '非小客车': '非小客车下行'
    }, inplace=True)
    
    print(f"\nUp direction samples: {len(df_up)}")
    print(f"Down direction samples: {len(df_down)}")
    
    # Merge up and down data by time and station
    df_merged = pd.merge(
        df_up, 
        df_down, 
        on=['时间', '观测站编号', '观测站名称'], 
        how='outer'
    )
    
    # Fill NaN values with 0 (no traffic)
    feature_cols = ['小客车上行', '小客车下行', '非小客车上行', '非小客车下行']
    for col in feature_cols:
        df_merged[col] = df_merged[col].fillna(0)
    
    print(f"Merged dataset shape: {df_merged.shape}")
    print(f"Features: {feature_cols}")
    
    # Sort by time and station
    df_merged = df_merged.sort_values(['观测站编号', '时间']).reset_index(drop=True)
    
    # Create station index mapping
    unique_stations = sorted(df_merged['观测站编号'].unique())
    station_to_idx = {station: idx for idx, station in enumerate(unique_stations)}
    df_merged['站点索引'] = df_merged['观测站编号'].map(station_to_idx)
    
    # Create station name mapping
    station_mapping = {}
    for station_code, station_idx in station_to_idx.items():
        station_name = df_merged[df_merged['观测站编号'] == station_code]['观测站名称'].iloc[0]
        station_mapping[int(station_idx)] = {
            'station_code': station_code,
            'station_name': station_name
        }
    
    # Save station mapping
    with open('station_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(station_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Unique stations: {df_merged['站点索引'].nunique()}")
    print(f"✓ Time range: {df_merged['时间'].min()} to {df_merged['时间'].max()}")
    print(f"✓ Station mapping saved to station_mapping.json")
    
    return df_merged, station_to_idx

def pivot_and_normalize(full_df):
    """
    Pivot data to have features as columns
    NOTE: No normalization is applied here - MOMENT model has built-in RevIN normalization
    Returns pivoted dataframe (raw values) and empty normalization params
    """
    print("\n" + "="*60)
    print("Pivoting data (NO normalization - using MOMENT's built-in RevIN)...")
    print("="*60)
    
    # Get unique stations - use the correct column name from preprocess_traffic_data
    if '站点索引' in full_df.columns:
        station_col = '站点索引'
    elif 'station_id' in full_df.columns:
        station_col = 'station_id'
    else:
        raise KeyError(f"Cannot find station column. Available columns: {full_df.columns.tolist()}")
    
    stations = sorted(full_df[station_col].unique())
    print(f"Number of stations: {len(stations)}")
    
    # Create feature columns for each station
    # Features: 小客车上行, 小客车下行, 非小客车上行, 非小客车下行
    feature_cols = []
    for station in stations:
        for feat in ['小客车上行', '小客车下行', '非小客车上行', '非小客车下行']:
            col_name = f"{feat}_station{station}"
            feature_cols.append(col_name)
    
    # Pivot the data - create a wide format table
    print("Pivoting data to wide format...")
    pivoted_data = {}
    
    # Group by time and station
    for _, row in full_df.iterrows():
        timestamp = row['时间']
        station = row[station_col]
        
        if timestamp not in pivoted_data:
            pivoted_data[timestamp] = {}
        
        # Add features for this station (RAW VALUES, NO NORMALIZATION)
        pivoted_data[timestamp][f'小客车上行_station{station}'] = row.get('小客车上行', 0)
        pivoted_data[timestamp][f'小客车下行_station{station}'] = row.get('小客车下行', 0)
        pivoted_data[timestamp][f'非小客车上行_station{station}'] = row.get('非小客车上行', 0)
        pivoted_data[timestamp][f'非小客车下行_station{station}'] = row.get('非小客车下行', 0)
    
    # Convert to DataFrame
    pivoted_df = pd.DataFrame.from_dict(pivoted_data, orient='index')
    pivoted_df.index.name = 'timestamp'
    pivoted_df = pivoted_df.sort_index()
    
    # Fill any missing values with 0
    pivoted_df = pivoted_df.fillna(0)
    
    print(f"Pivoted data shape: {pivoted_df.shape}")
    print(f"Features per station: 4 (小客车上/下行, 非小客车上/下行)")
    print(f"Total features: {len(feature_cols)}")
    
    # Verify data statistics (for debugging)
    print(f"\nData statistics (RAW VALUES):")
    print(f"  Min: {pivoted_df.min().min():.2f}")
    print(f"  Max: {pivoted_df.max().max():.2f}")
    print(f"  Mean: {pivoted_df.mean().mean():.2f}")
    print(f"  Std: {pivoted_df.std().mean():.2f}")
    
    # Save empty normalization params (not used, but kept for compatibility)
    normalization_params = {}
    with open('normalization_params.json', 'w') as f:
        json.dump(normalization_params, f)
    
    print("\n✓ Data saved WITHOUT normalization (MOMENT will handle it internally via RevIN)")
    
    return pivoted_df, normalization_params

def split_and_save_data(normalized_df):
    """
    Split data into train/val/test and save
    Note: Data is NOT normalized - MOMENT model handles normalization internally via RevIN
    """
    print("\n" + "="*60)
    print("Splitting and saving raw data (no normalization applied)...")
    print("="*60)
    
    # Split data into train/val/test (70%, 15%, 15%)
    n_time_points = len(normalized_df)
    train_end = int(n_time_points * 0.7)
    val_end = int(n_time_points * 0.85)
    
    train_data = normalized_df[:train_end]
    val_data = normalized_df[train_end:val_end]
    test_data = normalized_df[val_end:]
    
    print(f"\nData split:")
    print(f"  Train: {train_data.shape[0]} time points ({train_data.shape[1]} features)")
    print(f"  Val:   {val_data.shape[0]} time points ({val_data.shape[1]} features)")
    print(f"  Test:  {test_data.shape[0]} time points ({test_data.shape[1]} features)")
    
    # Verify no NaN values
    for name, data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            print(f"  ⚠️  WARNING: {name} set contains {nan_count} NaN values!")
        else:
            print(f"  ✓ {name} set: No NaN values")
    
    # Save processed data
    os.makedirs('processed_data', exist_ok=True)
    
    train_data.to_csv('processed_data/train.csv')
    val_data.to_csv('processed_data/val.csv')
    test_data.to_csv('processed_data/test.csv')
    
    print("\n✓ Raw data saved to processed_data/")
    
    return train_data, val_data, test_data

def create_moment_dataset_format(data, seq_len=512, pred_len=96):
    """
    Convert data to the format required by MOMENT model for forecasting
    Creates sliding window sequences
    """
    print(f"\nCreating MOMENT dataset format (seq_len={seq_len}, pred_len={pred_len})...")
    
    # Convert to numpy array
    values = data.values.astype(np.float32)
    
    print(f"  Data shape: {values.shape}")
    print(f"  Time points: {len(values)}")
    print(f"  Features: {values.shape[1]}")
    
    # Calculate total samples
    total_samples = len(values) - seq_len - pred_len + 1
    
    print(f"  Required minimum length: {seq_len + pred_len}")
    print(f"  Total samples to generate: {total_samples}")
    
    # Check if we have enough data
    if total_samples <= 0:
        raise ValueError(
            f"❌ Insufficient data! Need at least {seq_len + pred_len} time points, "
            f"but only have {len(values)}. "
            f"Please either:\n"
            f"  1. Add more data files to dataset/ directory\n"
            f"  2. Reduce seq_len (currently {seq_len})\n"
            f"  3. Reduce pred_len (currently {pred_len})"
        )
    
    # Prepare sequences
    sequences = []
    targets = []
    
    # Create sliding window sequences
    print(f"  Generating {total_samples} samples...")
    for i in range(total_samples):
        seq = values[i:i+seq_len]
        target = values[i+seq_len:i+seq_len+pred_len]
        sequences.append(seq)
        targets.append(target)
    
    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    print(f"  ✓ Input shape: {sequences.shape}  [samples, seq_len, n_features]")
    print(f"  ✓ Target shape: {targets.shape}  [samples, pred_len, n_features]")
    
    # Verify data is not empty
    if sequences.size == 0 or targets.size == 0:
        raise RuntimeError("Generated sequences are empty! Check your data and parameters.")
    
    return sequences, targets

if __name__ == "__main__":
    # Define the path to your dataset directory
    dataset_dir = 'dataset'
    
    # Configuration - Adjust these parameters based on your data size
    # Option 1: Quick test configuration (8 hours in, 8 hours out)
    SEQ_LEN = 8    # Input sequence length in hours
    PRED_LEN = 8   # Prediction horizon in hours
    
    # Option 2: Short-term prediction (1 day in, 1 day out)
    # SEQ_LEN = 24
    # PRED_LEN = 24
    
    # Option 3: Medium-term prediction (7 days in, 1 day out)
    # SEQ_LEN = 168
    # PRED_LEN = 24
    
    # Option 4: Standard configuration (21 days in, 4 days out)
    # SEQ_LEN = 512
    # PRED_LEN = 96
    
    # Minimum required: SEQ_LEN + PRED_LEN time points
    
    print("="*60)
    print("MOMENT Traffic Flow Data Preprocessing Pipeline")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Sequence length: {SEQ_LEN} hours")
    print(f"  Prediction length: {PRED_LEN} hours")
    print(f"  Minimum data required: {SEQ_LEN + PRED_LEN} time points")
    print()
    
    # Step 1: Load and merge datasets
    merged_df = load_and_merge_datasets(dataset_dir)
    
    # Step 2: Preprocess traffic data
    processed_df, station_mapping = preprocess_traffic_data(merged_df)
    
    # Step 3: Pivot data (No normalization, MOMENT uses RevIN)
    normalized_df, norm_params = pivot_and_normalize(processed_df)
    
    # Step 4: Split and save
    train_data, val_data, test_data = split_and_save_data(normalized_df)
    
    # Step 5: Create MOMENT dataset format
    print("\n" + "="*60)
    print("Creating MOMENT-compatible datasets...")
    print("="*60)
    
    # Generate sliding window samples for each split
    train_inputs, train_targets = create_moment_dataset_format(train_data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    val_inputs, val_targets = create_moment_dataset_format(val_data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    test_inputs, test_targets = create_moment_dataset_format(test_data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    
    # Save datasets - 分别保存训练集、验证集、测试集
    os.makedirs('moment_data', exist_ok=True)
    
    # 保存训练集
    np.savez_compressed(
        'moment_data/train_dataset.npz',
        input=train_inputs,
        target=train_targets
    )
    print(f"\n  ✓ Training set saved to moment_data/train_dataset.npz")
    print(f"    Shape: {train_inputs.shape}")
    
    # 保存验证集
    np.savez_compressed(
        'moment_data/val_dataset.npz',
        input=val_inputs,
        target=val_targets
    )
    print(f"  ✓ Validation set saved to moment_data/val_dataset.npz")
    print(f"    Shape: {val_inputs.shape}")
    
    # 保存测试集
    np.savez_compressed(
        'moment_data/test_dataset.npz',
        input=test_inputs,
        target=test_targets
    )
    print(f"  ✓ Test set saved to moment_data/test_dataset.npz")
    print(f"    Shape: {test_inputs.shape}")
    
    # 为了向后兼容，也保存一个包含所有数据集的联合文件
    np.savez_compressed(
        'moment_data/dataset.npz',
        train_input=train_inputs,
        train_target=train_targets,
        val_input=val_inputs,
        val_target=val_targets,
        test_input=test_inputs,
        test_target=test_targets
    )
    print(f"  ✓ Combined dataset saved to moment_data/dataset.npz")
    
    print("\n" + "="*60)
    print("✓ All preprocessing completed successfully!")
    print("="*60)
    print("\nOutput files:")
    print("  - station_mapping.json              : Station index mapping")
    print("  - normalization_params.json         : Normalization parameters")
    print("  - processed_data/train.csv          : Training set (CSV)")
    print("  - processed_data/val.csv            : Validation set (CSV)")
    print("  - processed_data/test.csv           : Test set (CSV)")
    print("  - moment_data/train_dataset.npz     : Training set (NPZ)")
    print("  - moment_data/val_dataset.npz       : Validation set (NPZ)")
    print("  - moment_data/test_dataset.npz      : Test set (NPZ)")
    print("  - moment_data/dataset.npz           : Combined dataset (NPZ)")
    print("="*60)