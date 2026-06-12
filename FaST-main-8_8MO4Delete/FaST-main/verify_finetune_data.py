"""
验证微调数据文件的正确性
检查预测值和真实值是否不同，以及数据范围是否合理
"""
import numpy as np
import os

print("="*80)
print("🔍 微调数据文件验证工具")
print("="*80)

# 文件路径
PRED_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/visulization-result-4FEAT/predictions/finetune_data.npz"
TARGET_PATH = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/visulization-result-4FEAT/predictions/finetune_real_traffic.npz"
"""
/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/visulization-result-4FEAT-train/predictions/finetune_data.npz
/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/visulization-result-4FEAT-train/predictions/finetune_real_traffic.npz
"""
# 1. 检查文件是否存在
print("\n1️⃣  检查文件存在性...")
if not os.path.exists(PRED_PATH):
    print(f"❌ 预测文件不存在: {PRED_PATH}")
    exit(1)
else:
    print(f"✓ 预测文件存在: {PRED_PATH}")

if not os.path.exists(TARGET_PATH):
    print(f"❌ 真实值文件不存在: {TARGET_PATH}")
    exit(1)
else:
    print(f"✓ 真实值文件存在: {TARGET_PATH}")

# 2. 加载数据
print("\n2️⃣  加载数据...")
try:
    pred_data = np.load(PRED_PATH)
    target_data = np.load(TARGET_PATH)
    print(f"✓ 成功加载两个文件")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit(1)

# 3. 检查键名
print("\n3️⃣  检查数据结构...")
print(f"  finetune_data.npz 键名: {list(pred_data.keys())}")
print(f"  finetune_real_traffic.npz 键名: {list(target_data.keys())}")

# 提取数据
if 'prediction' in pred_data:
    pred = pred_data['prediction']
elif 'arr_0' in pred_data:
    pred = pred_data['arr_0']
else:
    print(f"❌ 预测文件中未找到预期键名")
    exit(1)

if 'target' in target_data:
    target = target_data['target']
elif 'arr_0' in target_data:
    target = target_data['arr_0']
else:
    print(f"❌ 真实值文件中未找到预期键名")
    exit(1)

print(f"\n✓ 数据形状:")
print(f"  - Prediction: {pred.shape}")
print(f"  - Target: {target.shape}")

# 4. 验证形状一致性
print("\n4️⃣  验证形状一致性...")
if pred.shape != target.shape:
    print(f"❌ 形状不匹配!")
    print(f"  Prediction: {pred.shape}")
    print(f"  Target: {target.shape}")
    exit(1)
else:
    print(f"✓ 形状一致: {pred.shape}")
    print(f"  解释: (samples={pred.shape[0]}, steps={pred.shape[1]}, nodes={pred.shape[2]}, features={pred.shape[3]})")

# 5. 验证数据是否不同
print("\n5️⃣  验证预测值与真实值是否不同...")
if np.allclose(pred, target):
    print(f"❌ ⚠️  严重错误：预测值和真实值完全相同！")
    print(f"  这意味着LLM无法学习修正能力")
    print(f"  可能原因:")
    print(f"    1. 可视化脚本保存逻辑错误")
    print(f"    2. GNN预测被同时用作预测值和标签")
    exit(1)
else:
    diff = np.abs(pred - target)
    print(f"✓ 预测值与真实值不同（符合预期）")
    print(f"  - 平均绝对误差 (MAE): {diff.mean():.2f}")
    print(f"  - 最大绝对误差: {diff.max():.2f}")
    print(f"  - 误差标准差: {diff.std():.2f}")
    print(f"  - 相对误差 (%): {(diff.mean() / (np.abs(target).mean() + 1e-8)) * 100:.2f}%")

# 6. 验证数据范围
print("\n6️⃣  验证数据范围（是否为原始尺度）...")
print(f"  Prediction:")
print(f"    - Min: {pred.min():.2f}")
print(f"    - Max: {pred.max():.2f}")
print(f"    - Mean: {pred.mean():.2f}")
print(f"    - Std: {pred.std():.2f}")

print(f"  Target:")
print(f"    - Min: {target.min():.2f}")
print(f"    - Max: {target.max():.2f}")
print(f"    - Mean: {target.mean():.2f}")
print(f"    - Std: {target.std():.2f}")

# 判断是否为归一化数据
if pred.max() < 5 and pred.min() > -5:
    print(f"\n⚠️  警告：数据范围较小，可能是归一化后的数据")
    print(f"  建议检查可视化脚本是否正确反归一化")
else:
    print(f"\n✓ 数据范围合理，应为原始尺度数据")

# 7. 按特征维度分析
print("\n7️⃣  按特征维度分析...")
feature_names = ["小客车上行", "小客车下行", "非小客车上行", "非小客车下行"]
for feat_idx, feat_name in enumerate(feature_names):
    pred_feat = pred[:, :, :, feat_idx].flatten()
    target_feat = target[:, :, :, feat_idx].flatten()
    mae = np.abs(pred_feat - target_feat).mean()
    
    print(f"  [{feat_name}]")
    print(f"    - Pred range: [{pred_feat.min():.2f}, {pred_feat.max():.2f}]")
    print(f"    - Target range: [{target_feat.min():.2f}, {target_feat.max():.2f}]")
    print(f"    - MAE: {mae:.2f}")

# 8. 总结
print("\n" + "="*80)
print("✅ 验证完成！")
print("="*80)

if not np.allclose(pred, target):
    print("\n🎉 数据文件正确，可以用于LLM微调！")
    print(f"  - 预测值包含GNN模型的预测误差")
    print(f"  - LLM可以学习如何修正这些误差")
else:
    print("\n❌ 数据文件有误，需要重新生成！")
    print(f"  请运行: python 可视化测试4特征.py")
