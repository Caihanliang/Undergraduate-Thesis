#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据加载是否正常工作
"""

import os
import sys
import numpy as np

# 添加项目路径
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, "main-master"))

# 模拟数据加载过程
dataset_name = "HNGS_4FEAT"
data_file_path = f"datasets/{dataset_name}"
input_len = 8
output_len = 8
sample_path = f"/{input_len}_{output_len}"

print("=" * 80)
print(" 测试数据加载")
print("=" * 80)

# 尝试加载数据
possible_paths = [
    data_file_path,
    f"main-master/{data_file_path}",
    os.path.join(os.getcwd(), data_file_path),
]

print("\n【1】搜索数据文件")
data = None
for base_path in possible_paths:
    his_file = base_path + "/his.npz"
    print(f"  检查: {his_file}")
    if os.path.exists(his_file):
        print(f"    ✅ 找到文件")
        data = np.load(his_file)["data"]
        print(f"    形状: {data.shape}")
        break
else:
    print(f"\n❌ 未找到数据文件")
    sys.exit(1)

# 尝试加载索引
print("\n【2】搜索索引文件")
idx_found = False
for base_path in possible_paths:
    train_idx_file = base_path + sample_path + "/idx_train.npy"
    print(f"  检查: {train_idx_file}")
    if os.path.exists(train_idx_file):
        print(f"    ✅ 找到索引文件")
        train_idx = np.load(base_path + sample_path + "/idx_train.npy")
        val_idx = np.load(base_path + sample_path + "/idx_val.npy")
        test_idx = np.load(base_path + sample_path + "/idx_test.npy")
        idx_found = True
        print(f"    train_idx: [{train_idx[0]}, {train_idx[-1]}], count={len(train_idx)}")
        print(f"    val_idx: [{val_idx[0]}, {val_idx[-1]}], count={len(val_idx)}")
        print(f"    test_idx: [{test_idx[0]}, {test_idx[-1]}], count={len(test_idx)}")
        break

if not idx_found:
    print(f"\n❌ 未找到索引文件")
    sys.exit(1)

# 测试切片
print("\n【3】测试数据切片")
total_length = len(data)

for mode, idx in [("train", train_idx), ("valid", val_idx), ("test", test_idx)]:
    start_idx = max(0, idx[0] - output_len - input_len + 1)
    if mode == "test":
        end_idx = total_length
    else:
        end_idx = min(total_length, idx[-1] + 1)
    
    sliced = data[start_idx:end_idx]
    num_samples = len(sliced) - input_len - output_len + 1
    
    print(f"\n{mode.upper()}:")
    print(f"  索引范围: [{idx[0]}, {idx[-1]}]")
    print(f"  数据切片: [{start_idx}:{end_idx}]")
    print(f"  切片长度: {len(sliced)}")
    print(f"  预期样本数: {num_samples}")
    
    if num_samples < 0:
        print(f"  ❌ 样本数为负数！")
    else:
        print(f"  ✅ 正常")

# 模拟数据集实例化
print("\n【4】模拟数据集实例化")
try:
    from basicts.data.simple_tsf_dataset import MyTimeSeries
    
    for mode in ["train", "valid", "test"]:
        print(f"\n实例化 {mode} 数据集...")
        dataset = MyTimeSeries(
            dataset_name=dataset_name,
            train_val_test_ratio=[0.6, 0.2, 0.2],
            mode=mode,
            input_len=input_len,
            output_len=output_len,
        )
        print(f"  ✅ 成功")
        print(f"  数据形状: {dataset.data.shape}")
        print(f"  样本数: {len(dataset)}")
        
        if len(dataset) < 0:
            print(f"  ❌ 样本数为负数！")
        else:
            print(f"  ✅ 样本数正常")
    
    print("\n" + "=" * 80)
    print("🎉 所有测试通过！数据加载正常")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
