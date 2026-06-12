#!/usr/bin/env python3
"""
深入分析为什么MAE和MAPE改善但RMSE恶化
"""

import pandas as pd
import numpy as np
import os

# 文件路径
csv_path = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/config/cai/inference_results_all_stations_with_predictions/all_station_feature_mean_metrics.csv"

print("="*100)
print("🔍 深入分析指标不一致的原因")
print("="*100)

# 读取数据
df = pd.read_csv(csv_path)

# ============================== 1. 全局统计 ==============================
print("\n1️⃣ 全局性能对比:")
print("-"*80)

gnn_mae = df['GNN_MAE'].mean()
llm_mae = df['LLM_MAE'].mean()
gnn_rmse = df['GNN_RMAE'].mean()
llm_rmse = df['LLM_RMAE'].mean()
gnn_mape = df['GNN_MAPE'].mean()
llm_mape = df['LLM_MAPE'].mean()

print(f"MAE:  GNN={gnn_mae:.4f}, LLM={llm_mae:.4f}, 变化={llm_mae-gnn_mae:+.4f} ({(llm_mae-gnn_mae)/gnn_mae*100:+.2f}%)")
print(f"RMSE: GNN={gnn_rmse:.4f}, LLM={llm_rmse:.4f}, 变化={llm_rmse-gnn_rmse:+.4f} ({(llm_rmse-gnn_rmse)/gnn_rmse*100:+.2f}%)")
print(f"MAPE: GNN={gnn_mape:.4f}, LLM={llm_mape:.4f}, 变化={llm_mape-gnn_mape:+.4f} ({(llm_mape-gnn_mape)/gnn_mape*100:+.2f}%)")

# ============================== 2. 按特征分析 ==============================
print("\n2️⃣ 按特征分析指标变化:")
print("-"*80)

feature_names = {
    0: "小客车上行",
    1: "小客车下行",
    2: "非小客车上行",
    3: "非小客车下行"
}

for feat_id in sorted(df['feature_id'].unique()):
    feat_data = df[df['feature_id'] == feat_id]
    feat_name = feature_names.get(feat_id, f"Feature {feat_id}")
    
    gnn_mae_f = feat_data['GNN_MAE'].mean()
    llm_mae_f = feat_data['LLM_MAE'].mean()
    gnn_rmse_f = feat_data['GNN_RMAE'].mean()
    llm_rmse_f = feat_data['LLM_RMAE'].mean()
    gnn_mape_f = feat_data['GNN_MAPE'].mean()
    llm_mape_f = feat_data['LLM_MAPE'].mean()
    
    mae_change = llm_mae_f - gnn_mae_f
    rmse_change = llm_rmse_f - gnn_rmse_f
    mape_change = llm_mape_f - gnn_mape_f
    
    print(f"\n{feat_name}:")
    print(f"  MAE:  {gnn_mae_f:.2f} → {llm_mae_f:.2f} ({mae_change:+.2f})")
    print(f"  RMSE: {gnn_rmse_f:.2f} → {llm_rmse_f:.2f} ({rmse_change:+.2f})")
    print(f"  MAPE: {gnn_mape_f:.2f} → {llm_mape_f:.2f} ({mape_change:+.2f})")
    
    # 判断是否一致
    if mae_change < 0 and rmse_change < 0 and mape_change < 0:
        print(f"  ✅ 三个指标同时改善")
    elif mae_change > 0 and rmse_change > 0 and mape_change > 0:
        print(f"  ❌ 三个指标同时恶化")
    else:
        print(f"  ⚠️ 指标变化不一致")

# ============================== 3. 分析RMSE恶化的原因 ==============================
print("\n3️⃣ 分析RMSE恶化的可能原因:")
print("-"*80)

# 计算每个站点-特征组合的RMSE变化
df['RMSE_change'] = df['LLM_RMAE'] - df['GNN_RMAE']
df['MAE_change'] = df['LLM_MAE'] - df['GNN_MAE']

# 找出RMSE恶化但MAE改善的案例
worse_rmse_better_mae = df[(df['RMSE_change'] > 0) & (df['MAE_change'] < 0)]
better_both = df[(df['RMSE_change'] < 0) & (df['MAE_change'] < 0)]
worse_both = df[(df['RMSE_change'] > 0) & (df['MAE_change'] > 0)]

print(f"\n统计结果:")
print(f"  RMSE恶化但MAE改善: {len(worse_rmse_better_mae)} 个样本 ({len(worse_rmse_better_mae)/len(df)*100:.1f}%)")
print(f"  两个指标都改善: {len(better_both)} 个样本 ({len(better_both)/len(df)*100:.1f}%)")
print(f"  两个指标都恶化: {len(worse_both)} 个样本 ({len(worse_both)/len(df)*100:.1f}%)")

if len(worse_rmse_better_mae) > 0:
    print(f"\n典型案例分析（RMSE恶化但MAE改善的前5个）:")
    print(f"{'─'*90}")
    print(f"{'站点':<15} {'特征':<25} {'GNN_MAE':<10} {'LLM_MAE':<10} {'GNN_RMSE':<10} {'LLM_RMSE':<10}")
    print(f"{'─'*90}")
    
    for _, row in worse_rmse_better_mae.head(5).iterrows():
        print(f"{row['station_short_name']:<15} {row['feature_name']:<25} {row['GNN_MAE']:<10.2f} {row['LLM_MAE']:<10.2f} {row['GNN_RMAE']:<10.2f} {row['LLM_RMAE']:<10.2f}")

# ============================== 4. 解释现象 ==============================
print("\n" + "="*100)
print("💡 现象解释")
print("="*100)

print("""
📌 为什么会出现 MAE/MAPE 改善但 RMSE 恶化？

1. 【误差分布变化】
   - LLM减少了大多数样本的小误差（MAE改善）
   - 但在少数样本上产生了更大的极端误差（RMSE恶化）
   - RMSE对大误差更敏感（平方放大效应）

2. 【具体表现】
   - 假设GNN: 90%样本误差=10, 10%样本误差=100
   - 假设LLM: 95%样本误差=8, 5%样本误差=200
   
   计算结果:
   - MAE_GNN = 0.9*10 + 0.1*100 = 19
   - MAE_LLM = 0.95*8 + 0.05*200 = 17.6 ✅ 改善
   - RMSE_GNN = sqrt(0.9*100 + 0.1*10000) = 33.2
   - RMSE_LLM = sqrt(0.95*64 + 0.05*40000) = 44.9 ❌ 恶化

3. 【实际意义】
   - MAE改善说明整体预测精度提高
   - RMSE恶化说明预测稳定性下降，存在极端错误
   - MAPE改善说明在低值数据上表现更好

4. 【改进建议】
   A. 检查LLM输出是否有异常大的预测值
   B. 添加后处理步骤，限制预测值的合理范围
   C. 调整微调策略，减少极端误差的产生
   D. 考虑使用Huber Loss等鲁棒损失函数
""")

# ============================== 5. 建议的后续分析 ==============================
print("\n" + "="*100)
print("🔧 建议的后续分析")
print("="*100)

print("""
1. 【检查极端误差样本】
   - 找出LLM预测误差最大的前10%样本
   - 分析这些样本的特征（时间、站点、天气、事件等）
   - 确定是否存在系统性问题

2. 【调整微调策略】
   - 降低学习率，避免过拟合到极端样本
   - 增加正则化强度
   - 使用梯度裁剪防止梯度爆炸

3. 【后处理优化】
   - 添加预测值范围限制（如不超过历史最大值的2倍）
   - 使用 ensemble 方法结合GNN和LLM的优势
   - 对极端预测进行平滑处理

4. 【评估指标选择】
   - 如果关注整体精度，优先看MAE
   - 如果关注稳定性，优先看RMSE
   - 如果关注相对误差，优先看MAPE或sMAPE
   - 建议综合使用多个指标，不要只看单一指标
""")

print("\n" + "="*100)
