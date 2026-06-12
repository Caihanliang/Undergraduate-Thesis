#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复并重新生成 HNGS_4FEAT 数据集
这个脚本会：
1. 检查并修复数据文件
2. 生成索引文件
3. 验证数据集完整性
"""

import numpy as np
import os
import json
import sys

# 添加项目路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
sys.path.insert(0, PROJECT_DIR)

from process_4features_data import process_data
from generate_4feat_indices import generate_indices

def verify_dataset():
    """验证数据集完整性"""
    print("=" * 60)
    print("📊 验证数据集")
    print("=" * 60)
    
    dataset_dir = os.path.join(PROJECT_DIR, "main-master", "datasets", "HNGS_4FEAT")
    sample_dir = os.path.join(dataset_dir, "8_8")
    
    # 检查文件
    files_to_check = {
        "his.npz": dataset_dir,
        "desc.json": dataset_dir,
        "idx_train.npy": sample_dir,
        "idx_val.npy": sample_dir,
        "idx_test.npy": sample_dir,
    }
    
    all_ok = True
    for filename, dir_path in files_to_check.items():
        file_path = os.path.join(dir_path, filename)
        if os.path.exists(file_path):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} (缺失)")
            all_ok = False
    
    if all_ok:
        # 加载并验证数据
        print(f"\n📈 数据详情:")
        data = np.load(os.path.join(dataset_dir, "his.npz"))['data']
        print(f"  形状: {data.shape}")
        print(f"  范围: [{data.min():.2f}, {data.max():.2f}]")
        print(f"  均值: {data.mean():.2f}")
        print(f"  标准差: {data.std():.2f}")
        
        # 验证索引
        idx_train = np.load(os.path.join(sample_dir, "idx_train.npy"))
        idx_val = np.load(os.path.join(sample_dir, "idx_val.npy"))
        idx_test = np.load(os.path.join(sample_dir, "idx_test.npy"))
        
        print(f"\n📋 索引详情:")
        print(f"  训练集: {len(idx_train)} 样本 (范围: {idx_train[0]}-{idx_train[-1]})")
        print(f"  验证集: {len(idx_val)} 样本 (范围: {idx_val[0]}-{idx_val[-1]})")
        print(f"  测试集: {len(idx_test)} 样本 (范围: {idx_test[0]}-{idx_test[-1]})")
        
        # 验证索引不会导致负长度
        total_steps = data.shape[0]
        input_len = 8
        output_len = 8
        
        train_data_slice = data[
            idx_train[0] - output_len - input_len + 1 : idx_train[-1] + 1
        ]
        val_data_slice = data[
            idx_val[0] - output_len - input_len + 1 : idx_val[-1] + 1
        ]
        test_data_slice = data[
            idx_test[0] - output_len - input_len + 1 :
        ]
        
        print(f"\n✅ 数据切片验证:")
        print(f"  训练集切片形状: {train_data_slice.shape}")
        print(f"  验证集切片形状: {val_data_slice.shape}")
        print(f"  测试集切片形状: {test_data_slice.shape}")
        
        if train_data_slice.shape[0] > 0 and val_data_slice.shape[0] > 0 and test_data_slice.shape[0] > 0:
            print("\n✅ 数据集验证通过！")
            return True
        else:
            print("\n❌ 数据切片有问题！")
            return False
    else:
        print("\n❌ 文件缺失，需要重新生成")
        return False

def main():
    print("=" * 60)
    print("🔧 HNGS_4FEAT 数据集修复与验证")
    print("=" * 60)
    
    # 检查是否需要重新生成
    need_regenerate = not verify_dataset()
    
    if need_regenerate:
        print("\n" + "=" * 60)
        print("🔄 开始重新生成数据集")
        print("=" * 60)
        
        # 步骤1: 生成数据文件
        print("\n步骤1: 处理原始数据生成 his.npz")
        try:
            process_data()
        except Exception as e:
            print(f"❌ 数据处理失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 步骤2: 生成索引
        print("\n步骤2: 生成训练/验证/测试集索引")
        try:
            generate_indices()
        except Exception as e:
            print(f"❌ 索引生成失败: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 最终验证
    print("\n" + "=" * 60)
    print("✅ 最终验证")
    print("=" * 60)
    if verify_dataset():
        print("\n🎉 数据集准备完成！可以开始训练了。")
    else:
        print("\n❌ 数据集仍有问题，请检查上述错误信息。")

if __name__ == "__main__":
    main()
