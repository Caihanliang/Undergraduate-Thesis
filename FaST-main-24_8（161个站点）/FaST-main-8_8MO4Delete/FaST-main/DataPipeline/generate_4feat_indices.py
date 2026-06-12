#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成4特征数据集的训练/验证/测试集索引
输入：his.npz (由 process_4features_data.py 生成)
输出：idx_train.npy, idx_val.npy, idx_test.npy
"""

import numpy as np
import os
import json

# ========== 配置参数 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "main-master", "datasets", "HNGS_4FEAT")

# 数据集划分比例
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# 序列配置
INPUT_LEN = 8   # 输入序列长度
OUTPUT_LEN = 8  # 预测序列长度


def generate_indices():
    """生成数据集索引"""
    print("=" * 60)
    print("生成训练/验证/测试集索引")
    print("=" * 60)
    
    # 加载数据
    npz_path = os.path.join(DATASET_DIR, "his.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"未找到数据文件: {npz_path}")
    
    data = np.load(npz_path)['data']
    num_timesteps = data.shape[0]
    
    print(f"\n📊 数据信息:")
    print(f"  总时间步: {num_timesteps}")
    print(f"  站点数: {data.shape[1]}")
    print(f"  特征数: {data.shape[2]}")
    
    # 计算可生成的样本数
    # 每个样本需要 INPUT_LEN + OUTPUT_LEN 个时间步
    window_size = INPUT_LEN + OUTPUT_LEN
    num_samples = num_timesteps - window_size + 1
    
    print(f"\n📈 样本信息:")
    print(f"  窗口大小: {window_size} (输入{INPUT_LEN} + 输出{OUTPUT_LEN})")
    print(f"  总样本数: {num_samples}")
    
    # 划分数据集
    train_size = int(num_samples * TRAIN_RATIO)
    val_size = int(num_samples * VAL_RATIO)
    test_size = num_samples - train_size - val_size
    
    print(f"\n📋 数据集划分:")
    print(f"  训练集: {train_size} 样本 ({TRAIN_RATIO*100:.0f}%)")
    print(f"  验证集: {val_size} 样本 ({VAL_RATIO*100:.0f}%)")
    print(f"  测试集: {test_size} 样本 ({TEST_RATIO*100:.0f}%)")
    
    # 生成索引
    # 索引表示输入序列的结束位置
    all_indices = np.arange(num_samples)
    
    # 注意：idx[i] 表示第i个样本输入序列的结束时间索引
    # 第i个样本的输入范围: [idx[i]-INPUT_LEN+1, idx[i]]
    # 第i个样本的输出范围: [idx[i]+1, idx[i]+OUTPUT_LEN]
    
    idx_train = all_indices[:train_size]
    idx_val = all_indices[train_size:train_size + val_size]
    idx_test = all_indices[train_size + val_size:]
    
    # 保存索引
    sample_dir = os.path.join(DATASET_DIR, f"{INPUT_LEN}_{OUTPUT_LEN}")
    os.makedirs(sample_dir, exist_ok=True)
    
    np.save(os.path.join(sample_dir, "idx_train.npy"), idx_train)
    np.save(os.path.join(sample_dir, "idx_val.npy"), idx_val)
    np.save(os.path.join(sample_dir, "idx_test.npy"), idx_test)
    
    print(f"\n✅ 索引文件已保存:")
    print(f"  {os.path.join(sample_dir, 'idx_train.npy')}: {len(idx_train)} 个索引")
    print(f"  {os.path.join(sample_dir, 'idx_val.npy')}: {len(idx_val)} 个索引")
    print(f"  {os.path.join(sample_dir, 'idx_test.npy')}: {len(idx_test)} 个索引")
    
    # 验证索引
    print(f"\n🔍 索引验证:")
    print(f"  idx_train[0] = {idx_train[0]} (输入结束索引)")
    print(f"  idx_train[-1] = {idx_train[-1]}")
    print(f"  idx_val[0] = {idx_val[0]}")
    print(f"  idx_val[-1] = {idx_val[-1]}")
    print(f"  idx_test[0] = {idx_test[0]}")
    print(f"  idx_test[-1] = {idx_test[-1]}")
    
    # 更新时间范围到desc.json
    desc_path = os.path.join(DATASET_DIR, "desc.json")
    if os.path.exists(desc_path):
        with open(desc_path, 'r', encoding='utf-8') as f:
            desc = json.load(f)
        
        desc["train_val_test_ratio"] = [TRAIN_RATIO, VAL_RATIO, TEST_RATIO]
        desc["input_len"] = INPUT_LEN
        desc["output_len"] = OUTPUT_LEN
        desc["num_samples"] = num_samples
        
        with open(desc_path, 'w', encoding='utf-8') as f:
            json.dump(desc, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 已更新 {desc_path}")
    
    print("\n" + "=" * 60)
    print("🎉 索引生成完成！")
    print("=" * 60)


def main():
    try:
        generate_indices()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
