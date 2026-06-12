#!/usr/bin/env python
"""
检查预处理数据的统计特性
"""
import numpy as np
import json

print("="*70)
print("Data Quality Check")
print("="*70)

# 加载数据
data = np.load('moment_data/dataset.npz')
input_data = data['input']
target_data = data['target']

print(f"\n1. Dataset Shape:")
print(f"   Input shape:  {input_data.shape}")
print(f"   Target shape: {target_data.shape}")

print(f"\n2. NaN Detection:")
print(f"   Input NaN count:  {np.isnan(input_data).sum()}")
print(f"   Target NaN count: {np.isnan(target_data).sum()}")

if np.isnan(input_data).sum() > 0 or np.isnan(target_data).sum() > 0:
    print("\n   ⚠️  WARNING: Data contains NaN values!")
    print("   This is likely due to incorrect normalization.")
else:
    print("   ✓ No NaN values detected")

print(f"\n3. Statistical Properties (Input):")
print(f"   Min:  {np.nanmin(input_data):.6f}")
print(f"   Max:  {np.nanmax(input_data):.6f}")
print(f"   Mean: {np.nanmean(input_data):.6f}")
print(f"   Std:  {np.nanstd(input_data):.6f}")

print(f"\n4. Statistical Properties (Target):")
print(f"   Min:  {np.nanmin(target_data):.6f}")
print(f"   Max:  {np.nanmax(target_data):.6f}")
print(f"   Mean: {np.nanmean(target_data):.6f}")
print(f"   Std:  {np.nanstd(target_data):.6f}")

print(f"\n5. Sample Values (first sample, first time step):")
sample_input = input_data[0, 0, :10]
sample_target = target_data[0, 0, :10]
print(f"   Input[0,0,:10]:  {sample_input}")
print(f"   Target[0,0,:10]: {sample_target}")

print(f"\n6. Check for Infinite Values:")
print(f"   Input Inf count:  {np.isinf(input_data).sum()}")
print(f"   Target Inf count: {np.isinf(target_data).sum()}")

print("\n" + "="*70)
if np.isnan(input_data).sum() > 0 or np.isnan(target_data).sum() > 0:
    print("❌ DATA QUALITY ISSUE DETECTED!")
    print("   The dataset contains NaN values which will cause training failure.")
    print("   Please re-run preprocessing with proper normalization.")
else:
    print("✓ Data quality check passed!")
print("="*70)
