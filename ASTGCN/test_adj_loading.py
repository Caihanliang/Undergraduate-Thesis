#!/usr/bin/env python
# coding: utf-8
"""
快速测试邻接矩阵加载（独立版本，不依赖 utils.py）
"""

import numpy as np


def load_adj_standalone(adj_filename, num_of_vertices=None):
    """独立的邻接矩阵加载函数
    
    Args:
        adj_filename: 邻接矩阵文件路径 (CSV格式)
        num_of_vertices: 站点数量（可选，如果不提供则从文件推断）
    
    Returns:
        adj: 邻接矩阵 (N, N)
    """
    # 读取邻接矩阵文件
    if adj_filename.endswith('.csv'):
        # CSV格式：直接读取为矩阵
        adj = np.loadtxt(adj_filename, delimiter=',')
        
        # 验证矩阵是否为方阵
        if len(adj.shape) != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"邻接矩阵必须是方阵，当前形状: {adj.shape}")
        
        # 如果提供了num_of_vertices，验证一致性
        if num_of_vertices is not None:
            if adj.shape[0] != num_of_vertices:
                print(f"⚠️  警告: 配置文件中的num_of_vertices={num_of_vertices}与邻接矩阵大小{adj.shape[0]}不一致")
                print(f"   将使用邻接矩阵的实际大小: {adj.shape[0]}")
        
        print(f"✅ 邻接矩阵加载成功: shape={adj.shape}")
        return adj
    
    else:
        raise ValueError(f"不支持的邻接矩阵文件格式: {adj_filename}")


# 测试加载
adj_filename = './dataset/adj_matrix.csv'
num_of_vertices = 98

print("=" * 60)
print("测试邻接矩阵加载")
print("=" * 60)

try:
    adj = load_adj_standalone(adj_filename, num_of_vertices=num_of_vertices)
    print(f"\n✅ 加载成功!")
    print(f"   形状: {adj.shape}")
    print(f"   类型: {type(adj)}")
    print(f"   数据类型: {adj.dtype}")
    print(f"   对角线全为1: {all(adj[i][i] == 1.0 for i in range(adj.shape[0]))}")
    print(f"   数值范围: [{adj.min():.4f}, {adj.max():.4f}]")
    print(f"   均值: {adj.mean():.4f}")
    print(f"   标准差: {adj.std():.4f}")
    
    # 验证对称性
    is_symmetric = (adj == adj.T).all()
    print(f"   是否对称: {is_symmetric}")
    
    print("\n" + "=" * 60)
    print("测试通过! 可以开始训练了。")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 加载失败: {str(e)}")
    import traceback
    traceback.print_exc()
