#!/usr/bin/env python
# coding: utf-8
"""
快速检查训练数据的数值范围
"""

import numpy as np

print("=" * 60)
print("检查训练数据数值范围")
print("=" * 60)

# 加载训练数据
train_data = np.load('./dataset/train.npz')
x = train_data['x']  # (B, N, F, T_in) - 归一化后的输入
y = train_data['y']  # (B, N, F, T_out) - 原始值目标
mean = train_data['mean']
std = train_data['std']

print(f"\n📊 数据形状:")
print(f"   x: {x.shape}")
print(f"   y: {y.shape}")
print(f"   mean: {mean.shape}, 值: {mean.squeeze()}")
print(f"   std: {std.shape}, 值: {std.squeeze()}")

print(f"\n🔢 x (归一化输入) 统计:")
print(f"   均值: {x.mean():.4f}")
print(f"   标准差: {x.std():.4f}")
print(f"   最小值: {x.min():.4f}")
print(f"   最大值: {x.max():.4f}")

print(f"\n🔢 y (原始值目标) 统计:")
print(f"   均值: {y.mean():.2f}")
print(f"   标准差: {y.std():.2f}")
print(f"   最小值: {y.min():.2f}")
print(f"   最大值: {y.max():.2f}")

# 按特征分别统计
feature_names = ['小客车上行', '小客车下行', '非小客车上行', '非小客车下行']
print(f"\n🔢 y 按特征统计:")
for i in range(4):
    feat_y = y[:, :, i, :]
    print(f"   {feature_names[i]}:")
    print(f"     均值: {feat_y.mean():.2f}, 标准差: {feat_y.std():.2f}")
    print(f"     范围: [{feat_y.min():.2f}, {feat_y.max():.2f}]")

print("\n" + "=" * 60)
print("✅ 检查完成")
print("=" * 60)
