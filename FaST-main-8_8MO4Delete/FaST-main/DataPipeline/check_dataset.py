#!/usr/bin/env python3
"""
检查 HNGS_4FEAT 数据集文件是否完整
"""

import os
import numpy as np

# 数据集路径
dataset_name = "HNGS_4FEAT"
base_path = "/home/user/Downloads/cai/FaST-main-8_8MO4/FaST-main/main-master/datasets"
data_path = os.path.join(base_path, dataset_name)
sample_path = os.path.join(data_path, "8_8")

print("=" * 60)
print(f"📊 检查数据集: {dataset_name}")
print("=" * 60)

# 1. 检查 his.npz
his_file = os.path.join(data_path, "his.npz")
if os.path.exists(his_file):
    print(f"\n✅ 找到 his.npz")
    data = np.load(his_file)
    print(f"   数据形状: {data['data'].shape}")
    print(f"   数据类型: {data['data'].dtype}")
    print(f"   数据范围: [{data['data'].min():.2f}, {data['data'].max():.2f}]")
    print(f"   数据均值: {data['data'].mean():.2f}")
    print(f"   零值占比: {(data['data'] == 0).sum() / data['data'].size * 100:.2f}%")
else:
    print(f"\n❌ 未找到 his.npz: {his_file}")
    print("   请先运行 DataPipeline/process_4features_data.py")

# 2. 检查索引文件
print(f"\n📂 检查索引文件 (8_8):")
for idx_file in ["idx_train.npy", "idx_val.npy", "idx_test.npy"]:
    idx_path = os.path.join(sample_path, idx_file)
    if os.path.exists(idx_path):
        idx = np.load(idx_path)
        print(f"   ✅ {idx_file}")
        print(f"      形状: {idx.shape}")
        print(f"      范围: [{idx.min()}, {idx.max()}]")
        print(f"      样本数: {len(idx)}")
    else:
        print(f"   ❌ 未找到 {idx_file}: {idx_path}")
        print("      请先运行 DataPipeline/generate_4feat_indices.py")

# 3. 检查 desc.json
desc_file = os.path.join(data_path, "desc.json")
if os.path.exists(desc_file):
    import json
    with open(desc_file, 'r') as f:
        desc = json.load(f)
    print(f"\n✅ 找到 desc.json")
    print(f"   数据集名称: {desc.get('dataset_name', 'N/A')}")
    print(f"   训练比例: {desc.get('train_val_test_ratio', 'N/A')}")
else:
    print(f"\n❌ 未找到 desc.json: {desc_file}")

# 4. 计算预期样本数
print(f"\n📈 预期样本数计算:")
if os.path.exists(his_file):
    data = np.load(his_file)
    total_steps = data['data'].shape[0]
    input_len = 8
    output_len = 8
    
    print(f"   总时间步: {total_steps}")
    print(f"   输入长度: {input_len}")
    print(f"   输出长度: {output_len}")
    print(f"   可用样本数: {total_steps - input_len - output_len + 1}")
    print(f"   训练集 (60%): {int((total_steps - input_len - output_len + 1) * 0.6)}")
    print(f"   验证集 (20%): {int((total_steps - input_len - output_len + 1) * 0.2)}")
    print(f"   测试集 (20%): {int((total_steps - input_len - output_len + 1) * 0.2)}")

print("\n" + "=" * 60)
print("💡 如果有任何 ❌ 标记，请先运行对应的数据处理脚本")
print("=" * 60)
