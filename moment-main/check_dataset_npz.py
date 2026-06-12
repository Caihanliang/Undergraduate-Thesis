#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 dataset.npz 文件内容
Usage: python check_dataset_npz.py
"""

import numpy as np
import os

# 文件路径
npz_path = 'moment_data/dataset.npz'

print("="*70)
print("MOMENT Dataset NPZ File Inspector")
print("="*70)

if not os.path.exists(npz_path):
    print(f"\n❌ Error: File not found at {npz_path}")
    exit(1)

print(f"\n📂 Loading: {npz_path}")
data = np.load(npz_path, allow_pickle=True)

print(f"\n📊 NPZ File Contents:")
print(f"  Keys in NPZ: {list(data.keys())}")
print(f"  Total arrays: {len(data.keys())}")

print("\n" + "="*70)
print("Array Details:")
print("="*70)

for key in data.keys():
    array = data[key]
    print(f"\n📌 {key}:")
    print(f"  Shape: {array.shape}")
    print(f"  Dtype: {array.dtype}")
    print(f"  Size: {array.size}")
    print(f"  Dimensions: {array.ndim}D")
    
    # 统计信息
    if array.dtype in [np.float32, np.float64, np.int32, np.int64]:
        print(f"  Min: {array.min()}")
        print(f"  Max: {array.max()}")
        print(f"  Mean: {array.mean():.4f}")
        print(f"  Std: {array.std():.4f}")
        print(f"  NaN count: {np.isnan(array).sum()}")
        print(f"  Zero count: {(array == 0).sum()}")
        print(f"  Zero ratio: {(array == 0).mean() * 100:.2f}%")
    
    # 前几个值（如果是1D或2D）
    if array.ndim == 1:
        print(f"  First 10 values: {array[:10]}")
    elif array.ndim == 2:
        print(f"  First row: {array[0, :10]}")
    elif array.ndim == 3:
        print(f"  First sample, first timestep: {array[0, 0, :10]}")

print("\n" + "="*70)
print("Data Interpretation:")
print("="*70)

# 根据shape推断数据结构
if 'input' in data and 'target' in data:
    input_data = data['input']
    target_data = data['target']
    
    print(f"\n📈 Input (X): {input_data.shape}")
    print(f"  Format: [num_samples, seq_len, n_features]")
    print(f"  num_samples: {input_data.shape[0]}")
    print(f"  seq_len (input): {input_data.shape[1]}")
    print(f"  n_features: {input_data.shape[2]}")
    
    print(f"\n🎯 Target (Y): {target_data.shape}")
    print(f"  Format: [num_samples, pred_len, n_features]")
    print(f"  num_samples: {target_data.shape[0]}")
    print(f"  pred_len (output): {target_data.shape[1]}")
    print(f"  n_features: {target_data.shape[2]}")
    
    if input_data.shape[0] == target_data.shape[0]:
        print(f"\n✓ Samples match: {input_data.shape[0]}")
    
    if input_data.shape[2] == target_data.shape[2]:
        n_features = input_data.shape[2]
        print(f"✓ Features match: {n_features}")
        print(f"  Features per station: 4 (Passenger Up/Down, Non-Passenger Up/Down)")
        print(f"  Estimated stations: {n_features // 4}")

print("\n" + "="*70)
print("Memory Size:")
print("="*70)

total_size = sum(data[key].nbytes for key in data.keys())
print(f"\n💾 Total size: {total_size / (1024*1024):.2f} MB")

for key in data.keys():
    size = data[key].nbytes
    print(f"  {key}: {size / (1024*1024):.2f} MB")

print("\n" + "="*70)
print("✅ Inspection Complete!")
print("="*70)