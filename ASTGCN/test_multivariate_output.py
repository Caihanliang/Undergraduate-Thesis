#!/usr/bin/env python
# coding: utf-8
"""
测试ASTGCN多变量输出的维度转换
"""

import torch
import sys
sys.path.insert(0, './model')

from ASTGCN_r import make_model
import numpy as np

print("=" * 60)
print("ASTGCN 多变量输出维度测试")
print("=" * 60)

# 模拟配置参数
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
nb_block = 2
in_channels = 4
K = 3
nb_chev_filter = 64
nb_time_filter = 64
time_strides = 1
num_for_predict = 8
len_input = 8
num_of_vertices = 98
num_features_out = 4

# 创建随机邻接矩阵（仅用于测试）
adj_mx = np.eye(num_of_vertices, dtype=np.float32)

print(f"\n📋 模型配置:")
print(f"   输入: (B={32}, N={num_of_vertices}, F_in={in_channels}, T_in={len_input})")
print(f"   输出: (B={32}, N={num_of_vertices}, F_out={num_features_out}, T_out={num_for_predict})")

try:
    # 创建模型
    print(f"\n🔧 创建模型...")
    net = make_model(
        DEVICE, nb_block, in_channels, K, nb_chev_filter, 
        nb_time_filter, time_strides, adj_mx,
        num_for_predict, len_input, num_of_vertices, 
        num_features_out=num_features_out
    )
    
    print(f"✅ 模型创建成功")
    print(f"   总参数量: {sum(p.numel() for p in net.parameters()):,}")
    
    # 创建测试数据
    print(f"\n🧪 创建测试数据...")
    batch_size = 32
    test_input = torch.randn(batch_size, num_of_vertices, in_channels, len_input).to(DEVICE)
    print(f"   输入形状: {test_input.shape}")
    
    # 前向传播
    print(f"\n🚀 执行前向传播...")
    with torch.no_grad():
        output = net(test_input)
    
    print(f"✅ 前向传播成功")
    print(f"   输出形状: {output.shape}")
    
    # 验证输出形状
    expected_shape = (batch_size, num_of_vertices, num_features_out, num_for_predict)
    if output.shape == torch.Size(expected_shape):
        print(f"\n🎉 维度验证通过!")
        print(f"   期望: {expected_shape}")
        print(f"   实际: {tuple(output.shape)}")
    else:
        print(f"\n❌ 维度验证失败!")
        print(f"   期望: {expected_shape}")
        print(f"   实际: {tuple(output.shape)}")
        sys.exit(1)
    
    # 检查数值范围
    print(f"\n📊 输出统计:")
    print(f"   均值: {output.mean().item():.4f}")
    print(f"   标准差: {output.std().item():.4f}")
    print(f"   最小值: {output.min().item():.4f}")
    print(f"   最大值: {output.max().item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过! 可以开始训练了。")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 测试失败: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
