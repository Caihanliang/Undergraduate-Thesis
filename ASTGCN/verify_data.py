#!/usr/bin/env python
# coding: utf-8
"""
数据格式验证脚本
检查预处理后的数据是否符合ASTGCN要求
"""

import numpy as np
import os

def verify_data_format(dataset_dir='./dataset'):
    """验证数据格式"""
    print("=" * 60)
    print("ASTGCN 数据格式验证")
    print("=" * 60)
    
    required_files = ['train.npz', 'val.npz', 'test.npz']
    
    for filename in required_files:
        filepath = os.path.join(dataset_dir, filename)
        print(f"\n📁 检查文件: {filepath}")
        
        if not os.path.exists(filepath):
            print(f"  ❌ 文件不存在!")
            continue
        
        try:
            data = np.load(filepath)
            print(f"  ✅ 文件加载成功")
            print(f"  📋 可用键: {list(data.keys())}")
            
            # 检查必需的键
            required_keys = ['x', 'y', 'mean', 'std']
            for key in required_keys:
                if key in data:
                    print(f"  ✅ {key}: shape={data[key].shape}, dtype={data[key].dtype}")
                else:
                    print(f"  ❌ {key}: 缺失!")
            
            # 验证数据形状
            if 'x' in data and 'y' in data:
                x = data['x']
                y = data['y']
                
                # ASTGCN 期望: x=(B, N, F, T_in), y=(B, N, T_out)
                if len(x.shape) == 4:
                    B, N, F, T_in = x.shape
                    print(f"\n  📊 输入数据维度:")
                    print(f"     - Batch size (B): {B}")
                    print(f"     - Num stations (N): {N}")
                    print(f"     - Num features (F): {F}")
                    print(f"     - Input length (T_in): {T_in}")
                    
                    if len(y.shape) == 3:
                        B_y, N_y, T_out = y.shape
                        print(f"\n  📊 目标数据维度:")
                        print(f"     - Batch size (B): {B_y}")
                        print(f"     - Num stations (N): {N_y}")
                        print(f"     - Output length (T_out): {T_out}")
                        
                        # 验证一致性
                        if B == B_y and N == N_y:
                            print(f"\n  ✅ 批次和站点数一致")
                        else:
                            print(f"\n  ❌ 批次或站点数不一致!")
                        
                        if T_in == 8 and T_out == 8:
                            print(f"  ✅ 时序长度符合8输入8输出要求")
                        else:
                            print(f"  ⚠️  时序长度不符合预期 (期望8, 实际{T_in}/{T_out})")
                    else:
                        print(f"  ❌ y的维度不正确! 期望3维 (B, N, T_out), 实际{len(y.shape)}维")
                else:
                    print(f"  ❌ x的维度不正确! 期望4维 (B, N, F, T_in), 实际{len(x.shape)}维")
            
            # 检查数值范围
            if 'x' in data:
                x = data['x']
                print(f"\n  📈 输入数据统计:")
                print(f"     - 均值: {x.mean():.4f}")
                print(f"     - 标准差: {x.std():.4f}")
                print(f"     - 最小值: {x.min():.4f}")
                print(f"     - 最大值: {x.max():.4f}")
                print(f"     - 零值比例: {(x == 0).sum() / x.size * 100:.2f}%")
            
            if 'y' in data:
                y = data['y']
                print(f"\n  📈 目标数据统计:")
                print(f"     - 均值: {y.mean():.4f}")
                print(f"     - 标准差: {y.std():.4f}")
                print(f"     - 最小值: {y.min():.4f}")
                print(f"     - 最大值: {y.max():.4f}")
            
        except Exception as e:
            print(f"  ❌ 加载失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 检查邻接矩阵
    adj_path = os.path.join(dataset_dir, 'adj_matrix.csv')
    print(f"\n📁 检查邻接矩阵: {adj_path}")
    if os.path.exists(adj_path):
        adj = np.loadtxt(adj_path, delimiter=',')
        print(f"  ✅ 邻接矩阵 shape: {adj.shape}")
        print(f"  ✅ 对角线全为1: {np.all(np.diag(adj) == 1)}")
        print(f"  📊 邻接矩阵统计:")
        print(f"     - 均值: {adj.mean():.4f}")
        print(f"     - 标准差: {adj.std():.4f}")
        print(f"     - 最小值: {adj.min():.4f}")
        print(f"     - 最大值: {adj.max():.4f}")
    else:
        print(f"  ❌ 邻接矩阵不存在!")
    
    # 检查站点映射
    mapping_path = os.path.join(dataset_dir, 'station_mapping.csv')
    print(f"\n📁 检查站点映射: {mapping_path}")
    if os.path.exists(mapping_path):
        import pandas as pd
        mapping = pd.read_csv(mapping_path)
        print(f"  ✅ 站点数量: {len(mapping)}")
        print(f"  📋 列名: {list(mapping.columns)}")
        print(f"  前5个站点:")
        print(mapping.head())
    else:
        print(f"  ❌ 站点映射文件不存在!")
    
    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)


if __name__ == '__main__':
    verify_data_format()
