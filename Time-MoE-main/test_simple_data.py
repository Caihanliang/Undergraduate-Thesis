#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本：生成符合 Time-MoE 要求的简单测试数据
"""

import json
import numpy as np
import os

def generate_test_data(output_dir='./test_data', num_samples=1000):
    """生成简单的正弦波测试数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔄 正在生成 {num_samples} 个测试样本...")
    
    samples = []
    for i in range(num_samples):
        # 生成16个时间步的正弦波数据
        t = np.linspace(0, 2*np.pi, 16)
        sequence = np.sin(t + i * 0.1).tolist()
        
        sample = {
            'station_id': f'test_station_{i % 10}',
            'feature': 'test_feature',
            'sequence': sequence
        }
        samples.append(sample)
    
    # 保存为JSONL
    output_path = os.path.join(output_dir, 'train.jsonl')
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✅ 测试数据已保存到: {output_path}")
    print(f"   样本数: {len(samples)}")
    print(f"   每个序列长度: {len(samples[0]['sequence'])}")
    
    return output_path


if __name__ == '__main__':
    data_path = generate_test_data()
    print(f"\n🎯 现在可以使用以下命令测试训练:")
    print(f"python torch_dist_run.py main.py \\")
    print(f"    -d {data_path} \\")
    print(f"    -m Maple728/TimeMoE-50M \\")
    print(f"    -o ./logs/time_moe_test \\")
    print(f"    --max_length 16 \\")
    print(f"    --micro_batch_size 8 \\")
    print(f"    --global_batch_size 32 \\")
    print(f"    --learning_rate 1e-5 \\")
    print(f"    --num_train_epochs 2 \\")
    print(f"    --precision fp32")
