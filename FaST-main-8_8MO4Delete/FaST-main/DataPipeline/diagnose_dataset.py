#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细检查 HNGS_4FEAT 数据集的文件和数据
帮助诊断 __len__() 返回负数的问题
"""

import numpy as np
import os
import sys

# 数据集路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
DATASET_DIR = os.path.join(PROJECT_DIR, "main-master", "datasets", "HNGS_4FEAT")
SAMPLE_DIR = os.path.join(DATASET_DIR, "8_8")

INPUT_LEN = 8
OUTPUT_LEN = 8

print("=" * 80)
print("🔍 HNGS_4FEAT 数据集详细诊断")
print("=" * 80)

# 1. 检查 his.npz
print("\n【1】检查 his.npz")
his_file = os.path.join(DATASET_DIR, "his.npz")
if os.path.exists(his_file):
    data = np.load(his_file)['data']
    print(f"✅ 文件存在")
    print(f"   形状: {data.shape}")
    print(f"   数据类型: {data.dtype}")
    print(f"   时间步数: {data.shape[0]}")
    print(f"   站点数: {data.shape[1]}")
    print(f"   特征数: {data.shape[2]}")
    print(f"   数值范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"   均值: {data.mean():.2f}")
    print(f"   标准差: {data.std():.2f}")
    print(f"   零值占比: {(data == 0).sum() / data.size * 100:.2f}%")
    total_timesteps = data.shape[0]
else:
    print(f"❌ 文件不存在: {his_file}")
    print("   需要运行: python process_4features_data.py")
    sys.exit(1)

# 2. 检查索引文件
print("\n【2】检查索引文件")
for idx_file in ["idx_train.npy", "idx_val.npy", "idx_test.npy"]:
    idx_path = os.path.join(SAMPLE_DIR, idx_file)
    if os.path.exists(idx_path):
        idx = np.load(idx_path)
        print(f"✅ {idx_file}")
        print(f"   样本数: {len(idx)}")
        print(f"   索引范围: [{idx.min()}, {idx.max()}]")
        
        # 验证索引有效性
        if idx.min() < INPUT_LEN + OUTPUT_LEN - 1:
            print(f"   ⚠️ 警告: 最小索引 {idx.min()} 小于 {INPUT_LEN + OUTPUT_LEN - 1}")
        if idx.max() >= total_timesteps:
            print(f"   ⚠️ 警告: 最大索引 {idx.max()} 超过时间步数 {total_timesteps}")
    else:
        print(f"❌ 文件不存在: {idx_path}")
        print("   需要运行: python generate_4feat_indices.py")

# 3. 模拟数据集切片
print("\n【3】模拟数据集切片（关键诊断）")
if all(os.path.exists(os.path.join(SAMPLE_DIR, f)) for f in ["idx_train.npy", "idx_val.npy", "idx_test.npy"]):
    train_idx = np.load(os.path.join(SAMPLE_DIR, "idx_train.npy"))
    val_idx = np.load(os.path.join(SAMPLE_DIR, "idx_val.npy"))
    test_idx = np.load(os.path.join(SAMPLE_DIR, "idx_test.npy"))
    
    for mode, idx in [("train", train_idx), ("valid", val_idx), ("test", test_idx)]:
        # 计算切片
        start_idx = max(0, idx[0] - OUTPUT_LEN - INPUT_LEN + 1)
        if mode == "test":
            end_idx = total_timesteps
        else:
            end_idx = min(total_timesteps, idx[-1] + 1)
        
        slice_length = end_idx - start_idx
        expected_samples = slice_length - INPUT_LEN - OUTPUT_LEN + 1
        
        print(f"\n{mode.upper()} 集:")
        print(f"   索引范围: [{idx[0]}, {idx[-1]}]")
        print(f"   数据切片: [{start_idx}:{end_idx}]")
        print(f"   切片长度: {slice_length}")
        print(f"   预期样本数: {expected_samples}")
        
        if expected_samples < 0:
            print(f"   ❌ 错误: 预期样本数为负数！")
            print(f"      切片长度({slice_length}) < INPUT_LEN({INPUT_LEN}) + OUTPUT_LEN({OUTPUT_LEN}) - 1")
        elif expected_samples == 0:
            print(f"   ⚠️ 警告: 预期样本数为0")
        else:
            print(f"   ✅ 样本数正常")

# 4. 检查 desc.json
print("\n【4】检查 desc.json")
desc_file = os.path.join(DATASET_DIR, "desc.json")
if os.path.exists(desc_file):
    import json
    with open(desc_file, 'r') as f:
        desc = json.load(f)
    print(f"✅ 文件存在")
    print(f"   数据集名称: {desc.get('dataset_name', 'N/A')}")
    print(f"   训练/验证/测试比例: {desc.get('train_val_test_ratio', 'N/A')}")
else:
    print(f"⚠️ 文件不存在（可选）: {desc_file}")

# 5. 总结
print("\n" + "=" * 80)
print("📊 诊断总结")
print("=" * 80)

if os.path.exists(his_file) and all(os.path.exists(os.path.join(SAMPLE_DIR, f)) for f in ["idx_train.npy", "idx_val.npy", "idx_test.npy"]):
    print("✅ 所有必需文件都存在")
    print("✅ 如果还有 __len__() 错误，请检查上面的切片诊断结果")
else:
    print("❌ 缺少必需文件，请按顺序运行:")
    if not os.path.exists(his_file):
        print("   1. python process_4features_data.py")
    if not all(os.path.exists(os.path.join(SAMPLE_DIR, f)) for f in ["idx_train.npy", "idx_val.npy", "idx_test.npy"]):
        print("   2. python generate_4feat_indices.py")

print("\n💡 下一步:")
print("   运行修复脚本: python fix_and_verify_dataset.py")
print("   或直接运行训练: python main-master/experiments/train_seed.py -c FaST/HNGS_8_8_4FEAT.py -g 0")
