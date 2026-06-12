#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成数据索引文件（train/val/test）
适配删除站点后的新数据集
"""

import numpy as np
import os
import json

# ========== 配置参数 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = "HNGS_4FEAT"
DATA_DIR = os.path.join(BASE_DIR, "main-master", "datasets", DATASET_NAME)

# 输入输出配置
NPZ_PATH = os.path.join(DATA_DIR, "his.npz")
DESC_PATH = os.path.join(DATA_DIR, "desc.json")
OUTPUT_DIR = os.path.join(DATA_DIR, "8_8")

# 时间划分参数
INPUT_LEN = 8
OUTPUT_LEN = 8
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2


def regenerate_indices():
    """重新生成索引文件"""
    print("=" * 80)
    print("🔄 重新生成数据索引文件")
    print("=" * 80)
    
    # 1. 加载数据
    print(f"\n📂 加载数据: {NPZ_PATH}")
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"数据文件不存在: {NPZ_PATH}")
    
    data_dict = np.load(NPZ_PATH)
    data = data_dict['data']
    print(f"  ✓ 数据形状: {data.shape}")
    print(f"     - 时间步: {data.shape[0]}")
    print(f"     - 站点数: {data.shape[1]}")
    print(f"     - 特征数: {data.shape[2]}")
    
    num_samples = data.shape[0]
    
    # 2. 加载描述文件
    print(f"\n📂 加载描述文件: {DESC_PATH}")
    if os.path.exists(DESC_PATH):
        with open(DESC_PATH, 'r', encoding='utf-8') as f:
            desc = json.load(f)
        print(f"  ✓ 数据集名称: {desc.get('dataset_name', 'N/A')}")
        print(f"  ✓ 时间范围: {desc.get('time_range', 'N/A')}")
    else:
        print(f"  ⚠️  描述文件不存在，使用默认配置")
        desc = {}
    
    # 3. 计算划分点
    print(f"\n📊 计算数据划分...")
    train_end = int(num_samples * TRAIN_RATIO)
    val_end = train_end + int(num_samples * VAL_RATIO)
    
    # 考虑输入输出长度，确保每个集合都有足够的样本
    min_samples = INPUT_LEN + OUTPUT_LEN
    
    train_end = max(min_samples, train_end - OUTPUT_LEN)
    val_end = max(train_end + min_samples, val_end - OUTPUT_LEN)
    
    print(f"  总时间步: {num_samples}")
    print(f"  训练集: [0 : {train_end}] ({train_end} 步)")
    print(f"  验证集: [{train_end} : {val_end}] ({val_end - train_end} 步)")
    print(f"  测试集: [{val_end} : {num_samples}] ({num_samples - val_end} 步)")
    
    # 4. 生成索引
    print(f"\n🔨 生成索引文件...")
    
    # 训练集索引：所有可以生成完整样本的起始位置
    idx_train = np.arange(0, train_end - INPUT_LEN - OUTPUT_LEN + 1)
    
    # 验证集索引
    idx_val = np.arange(train_end, val_end - INPUT_LEN - OUTPUT_LEN + 1)
    
    # 测试集索引
    idx_test = np.arange(val_end, num_samples - INPUT_LEN - OUTPUT_LEN + 1)
    
    print(f"  ✓ 训练集样本数: {len(idx_train)}")
    print(f"  ✓ 验证集样本数: {len(idx_val)}")
    print(f"  ✓ 测试集样本数: {len(idx_test)}")
    print(f"  ✓ 总样本数: {len(idx_train) + len(idx_val) + len(idx_test)}")
    
    # 5. 保存索引文件
    print(f"\n💾 保存索引文件到: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    np.save(os.path.join(OUTPUT_DIR, "idx_train.npy"), idx_train)
    np.save(os.path.join(OUTPUT_DIR, "idx_val.npy"), idx_val)
    np.save(os.path.join(OUTPUT_DIR, "idx_test.npy"), idx_test)
    
    print(f"  ✓ idx_train.npy: {idx_train.shape}")
    print(f"  ✓ idx_val.npy: {idx_val.shape}")
    print(f"  ✓ idx_test.npy: {idx_test.shape}")
    
    # 6. 更新描述文件
    print(f"\n📝 更新描述文件...")
    desc.update({
        "num_nodes": int(data.shape[1]),
        "num_features": int(data.shape[2]),
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        "num_samples": num_samples,
        "train_val_test_ratio": [TRAIN_RATIO, VAL_RATIO, TEST_RATIO]
    })
    
    with open(DESC_PATH, 'w', encoding='utf-8') as f:
        json.dump(desc, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 已更新 {DESC_PATH}")
    
    # 7. 验证索引
    print(f"\n✅ 验证索引有效性...")
    
    # 检查索引范围
    assert idx_train[-1] + INPUT_LEN + OUTPUT_LEN <= train_end, "训练集索引越界"
    assert idx_val[-1] + INPUT_LEN + OUTPUT_LEN <= val_end, "验证集索引越界"
    assert idx_test[-1] + INPUT_LEN + OUTPUT_LEN <= num_samples, "测试集索引越界"
    
    print(f"  ✓ 所有索引范围正确")
    
    # 打印示例索引
    print(f"\n📋 索引示例:")
    print(f"  训练集前5个: {idx_train[:5]}")
    print(f"  验证集前5个: {idx_val[:5]}")
    print(f"  测试集前5个: {idx_test[:5]}")
    
    return idx_train, idx_val, idx_test


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🚀 开始重新生成索引文件")
    print("=" * 80 + "\n")
    
    try:
        idx_train, idx_val, idx_test = regenerate_indices()
        
        print("\n" + "=" * 80)
        print("🎉 索引文件生成完成！")
        print("=" * 80)
        print(f"\n下一步:")
        print(f"  1. 重新训练模型:")
        print(f"     python main-master/experiments/train_seed.py \\")
        print(f"       -c FaST/HNGS_8_8_4FEAT.py \\")
        print(f"       -g 0")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
