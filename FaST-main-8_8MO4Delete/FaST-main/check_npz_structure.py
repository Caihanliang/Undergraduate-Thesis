"""
检查NPZ文件结构对比脚本
用于验证微调输出文件是否符合模型输入要求
python check_npz_structure.py
"""
import numpy as np
import os

print("="*80)
print("📊 NPZ文件结构对比分析")
print("="*80)

# 定义要检查的文件
files_info = [
    {
        'path': '/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/visulization-result-4FEAT-train/predictions/finetune_data.npz',
        'name': 'LLM微调预测(归一化)',
        'expected_keys': ['prediction', 'target']
    },
    {
        'path': '/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/visulization-result-4FEAT-train/predictions/finetune_real_traffic.npz',
        'name': 'LLM微调真实值(原始尺度)',
        'expected_keys': ['prediction', 'target']
    },
    {
        'path': '/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/Li/ypred_train.npz',
        'name': 'GNN原始预测(模板)',
        'expected_keys': ['arr_0']
    },
    {
        'path': '/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/Li/ytrue_train.npz',
        'name': 'GNN真实标签(模板)',
        'expected_keys': ['arr_0']
    }
]

results = {}

for file_info in files_info:
    path = file_info['path']
    name = file_info['name']
    
    print(f"\n{'='*80}")
    print(f"📁 文件: {name}")
    print(f"路径: {path}")
    print(f"{'='*80}")
    
    if not os.path.exists(path):
        print(f"❌ 文件不存在!")
        continue
    
    try:
        data = np.load(path)
        keys = list(data.keys())
        
        print(f"✓ 键名: {keys}")
        print(f"  期望键名: {file_info['expected_keys']}")
        
        # 检查键名是否匹配
        if set(keys) == set(file_info['expected_keys']):
            print(f"  ✅ 键名匹配")
        else:
            print(f"  ⚠️  键名不匹配!")
        
        results[name] = {'keys': keys, 'shapes': {}}
        
        for key in keys:
            arr = data[key]
            shape = arr.shape
            results[name]['shapes'][key] = shape
            
            print(f"\n  键 '{key}':")
            print(f"    - 形状: {shape}")
            print(f"    - 维度数: {arr.ndim}")
            print(f"    - 数据类型: {arr.dtype}")
            
            if arr.ndim == 4:
                print(f"    - 第1维 (样本数): {shape[0]}")
                print(f"    - 第2维 (序列长度): {shape[1]}")
                print(f"    - 第3维 (站点数): {shape[2]}")
                print(f"    - 第4维 (特征数): {shape[3]}")
                
                # 判断是否是8步输出
                if shape[1] == 8:
                    print(f"    ✅ 包含8个时间步的输出")
                elif shape[1] == 1:
                    print(f"    ❌ 只包含1个时间步的输出（需要修改!）")
                else:
                    print(f"    ⚠️  时间步数为 {shape[1]}")
            
            # 显示数值范围
            print(f"    - 数值范围: [{arr.min():.4f}, {arr.max():.4f}]")
            print(f"    - 均值: {arr.mean():.4f}")
            print(f"    - 标准差: {arr.std():.4f}")
            
            # 显示样例数据
            sample = arr.flatten()[:5]
            print(f"    - 前5个元素: {sample}")
    
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()

# 总结对比
print("\n" + "="*80)
print("📋 结构对比总结")
print("="*80)

print("\n1️⃣  键名对比:")
for name, info in results.items():
    print(f"  {name}: {info['keys']}")

print("\n2️⃣  形状对比:")
for name, info in results.items():
    for key, shape in info['shapes'].items():
        print(f"  {name}['{key}']: {shape}")

print("\n3️⃣  关键问题检查:")

# 检查是否有8步输出
has_8_steps = False
has_1_step = False
for name, info in results.items():
    for key, shape in info['shapes'].items():
        if len(shape) == 4:
            if shape[1] == 8:
                has_8_steps = True
                print(f"  ✅ {name} 包含8步输出: {shape}")
            elif shape[1] == 1:
                has_1_step = True
                print(f"  ❌ {name} 只有1步输出: {shape}")

if has_1_step and not has_8_steps:
    print("\n  ⚠️  警告: 所有文件都只有1步输出，需要修改为8步!")
elif has_8_steps:
    print("\n  ✅ 存在8步输出的文件")

print("\n4️⃣  兼容性建议:")
print("  - 如果微调文件是 (N, 8, S, 4)，可以直接用于 fine_gnn_vals.py")
print("  - 如果模板文件是 (N, 1, S, C)，需要调整代码以支持多步预测")
print("  - 键名不同不影响使用，fine_gnn_vals.py 已支持动态检测键名")

print("\n" + "="*80)
print("✅ 分析完成")
print("="*80)
