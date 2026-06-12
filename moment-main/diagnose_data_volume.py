#!/usr/bin/env python
"""
诊断数据量问题
"""
import numpy as np
import pandas as pd

print("="*70)
print("Data Volume Diagnosis")
print("="*70)

# 1. 检查NPZ文件
data = np.load('moment_data/dataset.npz')
print(f"\n1. NPZ Dataset:")
print(f"   Input shape:  {data['input'].shape}")
print(f"   Target shape: {data['target'].shape}")
print(f"   Total samples: {data['input'].shape[0]}")

# 2. 检查CSV文件
train_csv = pd.read_csv('processed_data/train.csv', index_col=0)
val_csv = pd.read_csv('processed_data/val.csv', index_col=0)
test_csv = pd.read_csv('processed_data/test.csv', index_col=0)

print(f"\n2. CSV Files:")
print(f"   Train: {train_csv.shape[0]} time points, {train_csv.shape[1]} features")
print(f"   Val:   {val_csv.shape[0]} time points, {val_csv.shape[1]} features")
print(f"   Test:  {test_csv.shape[0]} time points, {test_csv.shape[1]} features")

# 3. 计算理论样本数
seq_len = 8
pred_len = 8

for name, df in [('Train', train_csv), ('Val', val_csv), ('Test', test_csv)]:
    theoretical_samples = len(df) - seq_len - pred_len + 1
    print(f"\n3. {name} Set:")
    print(f"   Time points: {len(df)}")
    print(f"   Theoretical samples (with seq={seq_len}, pred={pred_len}): {theoretical_samples}")

# 4. 检查原始数据
his_df = pd.read_csv('dataset/his_data_with_names.csv')
his_df['时间'] = pd.to_datetime(his_df['时间'])
print(f"\n4. Original Data:")
print(f"   Total records: {len(his_df)}")
print(f"   Time range: {his_df['时间'].min()} to {his_df['时间'].max()}")
print(f"   Unique stations: {his_df['观测站编号'].nunique()}")
print(f"   Time span: {(his_df['时间'].max() - his_df['时间'].min()).days} days")

# 5. 建议
print("\n" + "="*70)
if data['input'].shape[0] < 1000:
    print("⚠️  WARNING: Training samples are TOO FEW!")
    print("\nPossible reasons:")
    print("  1. Original data time span is too short")
    print("  2. Too many stations causing insufficient data per station")
    print("  3. Data preprocessing filter removed too many samples")
    print("\nSuggestions:")
    print("  - Increase original data collection period")
    print("  - Reduce number of stations (select key stations only)")
    print("  - Use longer sequence length if data allows")
else:
    print("✓ Sample size is acceptable")
print("="*70)
