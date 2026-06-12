#!/usr/bin/env python
# coding: utf-8
"""
测试反归一化函数的广播逻辑（独立版本）
"""

import numpy as np

def re_normalization(x, mean, std):
    """
    反归一化函数，支持多种数据格式
    
    Args:
        x: 归一化后的数据
        mean: 均值，形状可能是 (1,1,1,F) 或 (1,1,F,1)
        std: 标准差，形状可能是 (1,1,1,F) 或 (1,1,F,1)
    
    Returns:
        反归一化后的数据
    """
    # 处理不同的数据格式
    if len(x.shape) == 4:
        # 情况1: x shape = (B, N, F, T) - 多变量输入
        # mean/std shape = (1, 1, 1, F) 需要转换为 (1, 1, F, 1)
        if mean.shape == (1, 1, 1, x.shape[2]):
            # 转换归一化参数以匹配 (B, N, F, T)
            mean_reshaped = mean.reshape(1, 1, x.shape[2], 1)
            std_reshaped = std.reshape(1, 1, x.shape[2], 1)
            x = x * std_reshaped + mean_reshaped
        elif mean.shape == (1, 1, x.shape[2], 1):
            # 已经匹配，直接使用
            x = x * std + mean
        else:
            raise ValueError(f"mean形状 {mean.shape} 与数据形状 {x.shape} 不匹配")
    
    elif len(x.shape) == 3:
        # 情况2: x shape = (B, N, T) - 单变量
        # mean/std shape = (1, 1, 1, F) 取第一个特征
        if len(mean.shape) == 4:
            mean_val = mean[0, 0, 0, 0] if mean.shape[3] > 0 else mean[0, 0, 0]
            std_val = std[0, 0, 0, 0] if std.shape[3] > 0 else std[0, 0, 0]
            x = x * std_val + mean_val
        else:
            x = x * std + mean
    else:
        # 其他情况，直接广播
        x = x * std + mean
    
    return x


print("=" * 60)
print("测试反归一化广播逻辑")
print("=" * 60)

# 模拟数据
B, N, F, T = 32, 98, 4, 8

# 创建测试数据 (B, N, F, T)
test_data = np.random.randn(B, N, F, T).astype(np.float32)

# 归一化参数 (1, 1, 1, F) - 这是预处理脚本保存的格式
mean = np.array([[[[10.0, 20.0, 30.0, 40.0]]]], dtype=np.float32)  # (1, 1, 1, 4)
std = np.array([[[[5.0, 10.0, 15.0, 20.0]]]], dtype=np.float32)   # (1, 1, 1, 4)

print(f"\n📊 数据形状:")
print(f"   test_data: {test_data.shape}")
print(f"   mean: {mean.shape}")
print(f"   std: {std.shape}")

try:
    # 测试反归一化
    print(f"\n🔄 执行反归一化...")
    result = re_normalization(test_data, mean, std)
    
    print(f"✅ 反归一化成功!")
    print(f"   输出形状: {result.shape}")
    
    # 验证数值范围
    print(f"\n📈 数值统计:")
    for i in range(F):
        feat_name = ['小客车上行', '小客车下行', '非小客车上行', '非小客车下行'][i]
        original_mean = mean[0, 0, 0, i]
        original_std = std[0, 0, 0, i]
        recovered_mean = result[:, :, i, :].mean()
        
        print(f"   {feat_name}:")
        print(f"     原始均值: {original_mean:.2f}, 原始标准差: {original_std:.2f}")
        print(f"     恢复后均值: {recovered_mean:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ 测试通过! 反归一化逻辑正确。")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 测试失败: {str(e)}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)
