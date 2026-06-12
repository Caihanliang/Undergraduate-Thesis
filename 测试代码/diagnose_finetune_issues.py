#!/usr/bin/env python3
"""
诊断微调效果变差的原因
"""

import os
import pandas as pd
import numpy as np

PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/config/cai/"

print("="*80)
print("🔍 诊断微调效果变差的原因")
print("="*80)

# ============================== 1. 检查阈值设置是否合理 ==============================
print("\n1️⃣ 检查阈值设置:")
print("-"*50)

thresholds = {
    0: ("小客车上行", 100),
    1: ("小客车下行", 90),
    2: ("非小客车上行", 27),
    3: ("非小客车下行", 27)
}

print("当前阈值配置:")
for feat_id, (name, threshold) in thresholds.items():
    print(f"  特征{feat_id} ({name}): 阈值={threshold}")

print("\n问题分析:")
print("  - 阈值过高会导致只有极少数样本被选中进行微调")
print("  - 如果GNN预测已经很准确（MAE<阈值），这些样本不会被用于微调")
print("  - 模型只学习了如何修正大误差样本，可能破坏了原有的良好预测能力")

# ============================== 2. 检查天气数据匹配问题 ==============================
print("\n2️⃣ 检查天气数据匹配:")
print("-"*50)

weather_dir = os.path.join(PROJECT_ROOT, "98站点天气信息/")
if os.path.exists(weather_dir):
    weather_files = sorted(os.listdir(weather_dir))
    print(f"天气文件数量: {len(weather_files)}")
    
    # 检查几个示例文件的命名
    print("\n天气文件命名示例:")
    for i, wf in enumerate(weather_files[:5]):
        print(f"  索引{i}: {wf}")
    
    print("\n潜在问题:")
    print("  - 天气文件按地区命名（如'005望城区'）")
    print("  - 但站点索引5对应的是'Jianjia'ao/简家坳'")
    print("  - 如果地区和站点不对应，会导致错误的天气-流量关联")
else:
    print("❌ 天气目录不存在")

# ============================== 3. 检查事件数据匹配 ==============================
print("\n3️⃣ 检查事件数据匹配:")
print("-"*50)

events_file = os.path.join(PROJECT_ROOT, "events_list_quan.csv")
if os.path.exists(events_file):
    df_events = pd.read_csv(events_file, encoding='utf-8')
    print(f"事件总数: {len(df_events)}")
    print(f"涉及站点数: {df_events['站点名称'].nunique()}")
    
    # 检查事件分布
    event_counts = df_events['站点名称'].value_counts()
    print("\n事件最多的前5个站点:")
    for station, count in event_counts.head(5).items():
        print(f"  {station}: {count}条事件")
    
    print("\n潜在问题:")
    print("  - 如果事件站点与训练站点不完全对应，会导致错误的事件-流量关联")
    print("  - 某些站点可能没有事件记录，影响模型学习")
else:
    print("❌ 事件文件不存在")

# ============================== 4. 检查训练数据质量 ==============================
print("\n4️⃣ 检查训练数据质量:")
print("-"*50)

# 加载NPZ文件
ytrue_path = os.path.join(PROJECT_ROOT, "finetune_real_traffic.npz")
ypred_path = os.path.join(PROJECT_ROOT, "finetune_data.npz")

if os.path.exists(ytrue_path) and os.path.exists(ypred_path):
    ytrue = np.load(ytrue_path)
    ypred = np.load(ypred_path)
    
    print(f"真实值形状: {ytrue['target'].shape if 'target' in ytrue else ytrue['arr_0'].shape}")
    print(f"预测值形状: {ypred['prediction'].shape if 'prediction' in ypred else ypred['arr_0'].shape}")
    
    # 计算MAE分布
    true_data = ytrue['target'] if 'target' in ytrue else ytrue['arr_0']
    pred_data = ypred['prediction'] if 'prediction' in ypred else ypred['arr_0']
    
    mae_per_sample = np.abs(true_data - pred_data).mean(axis=(1,2,3))  # 平均每个样本的MAE
    
    print(f"\nMAE统计:")
    print(f"  最小MAE: {mae_per_sample.min():.2f}")
    print(f"  最大MAE: {mae_per_sample.max():.2f}")
    print(f"  平均MAE: {mae_per_sample.mean():.2f}")
    print(f"  中位数MAE: {np.median(mae_per_sample):.2f}")
    
    # 检查有多少样本超过阈值
    for feat_id, (name, threshold) in thresholds.items():
        # 这里需要更复杂的逻辑来计算每个特征的MAE
        pass
    
    print("\n潜在问题:")
    print("  - 如果大部分样本的MAE都低于阈值，微调数据会很少")
    print("  - 模型可能过拟合到少数大误差样本上")
else:
    print("❌ NPZ文件不存在")

# ============================== 5. 检查微调策略 ==============================
print("\n5️⃣ 检查微调策略:")
print("-"*50)

print("当前策略:")
print("  - 阈值触发修正: 只有MAE>阈值的样本才进行微调")
print("  - 事件触发: 有事件的样本也进行微调")
print("  - 随机采样: 10%的正常样本也加入训练")
print("  - 修正目标: 直接让模型输出真实值")

print("\n潜在问题:")
print("  - 这种策略假设GNN在大误差时完全错误，需要LLM完全修正")
print("  - 但实际上GNN可能只是略有偏差，LLM过度修正反而会更差")
print("  - 模型学会了'推翻'GNN的预测，而不是'优化'它")

# ============================== 6. 建议的改进方案 ==============================
print("\n" + "="*80)
print("💡 改进建议:")
print("="*80)

print("""
1. 【降低阈值】
   - 将阈值从100/90/27/27降低到更合理的值（如50/45/15/15）
   - 让更多样本参与微调，避免过拟合到大误差样本

2. 【修复天气匹配】
   - 建立准确的中英文站点映射表
   - 确保天气文件与站点正确对应
   - 或者暂时移除天气特征，专注于时间和事件特征

3. 【调整微调目标】
   - 不要直接让模型输出真实值
   - 而是让模型输出修正量（delta = true - pred）
   - 这样模型学习的是如何优化GNN，而不是取代GNN

4. 【增加正常样本比例】
   - 将NORMAL_SAMPLE_RATIO从0.1提高到0.3-0.5
   - 让模型也学习何时不需要修正

5. 【验证数据质量】
   - 在微调前，手动检查几个样本的Prompt内容
   - 确认天气、事件等信息是否正确
   - 如果发现错位，先修复数据再重新微调

6. 【对比实验】
   - 尝试不同的阈值组合
   - 尝试有无天气/事件特征的对比
   - 找到最优的微调配置
""")
