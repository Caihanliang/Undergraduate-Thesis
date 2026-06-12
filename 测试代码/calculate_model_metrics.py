#!/usr/bin/env python3
"""
计算两个模型（GNN和LLM）在四个特征上的三个评估指标（MAE、RMSE、MAPE）
并生成详细的统计报告
排除站点18和55后重新计算所有指标
python calculate_model_metrics.py
"""

import pandas as pd
import numpy as np
import os

# 文件路径
csv_path = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/config/cai/inference_results_all_stations_with_predictions/all_station_feature_mean_metrics.csv"

print("="*100)
print("📊 两模型四特征三指标评估报告（排除站点18、55）")
print("="*100)

# 读取数据
df = pd.read_csv(csv_path)

print(f"\n✓ 原始数据加载成功")
print(f"  原始总行数: {len(df)}")
print(f"  原始站点数: {df['station_id'].nunique()}")
print(f"  特征数: {df['feature_id'].nunique()}")

# ============================== 核心修改：排除站点18和55 ==============================
# 定义需要排除的站点ID
exclude_stations = [18, 55]
# 过滤数据，排除指定站点
df_filtered = df[~df['station_id'].isin(exclude_stations)]

# 打印过滤后的基本信息
print(f"\n✓ 已排除站点: {exclude_stations}")
print(f"  过滤后总行数: {len(df_filtered)}")
print(f"  过滤后站点数: {df_filtered['station_id'].nunique()}")
print(f"  排除的记录数: {len(df) - len(df_filtered)}")

# 后续所有分析都使用过滤后的数据集 df_filtered
df = df_filtered

# ============================== 1. 按特征分组统计 ==============================
print("\n" + "="*100)
print("📈 按特征分组的模型性能对比（排除站点18、55）")
print("="*100)

feature_names = {
    0: "Passenger Car Up (小客车上行)",
    1: "Passenger Car Down (小客车下行)",
    2: "Non-Passenger Car Up (非小客车上行)",
    3: "Non-Passenger Car Down (非小客车下行)"
}

for feat_id in sorted(df['feature_id'].unique()):
    feat_data = df[df['feature_id'] == feat_id]
    feat_name = feature_names.get(feat_id, f"Feature {feat_id}")
    
    print(f"\n{'─'*100}")
    print(f"特征 {feat_id}: {feat_name}")
    print(f"{'─'*100}")
    
    # GNN模型指标
    gnn_mae = feat_data['GNN_MAE'].mean()
    gnn_rmse = feat_data['GNN_RMAE'].mean()  # RMAE在这里作为RMSE的相对值
    gnn_mape = feat_data['GNN_MAPE'].mean()
    
    # LLM模型指标
    llm_mae = feat_data['LLM_MAE'].mean()
    llm_rmse = feat_data['LLM_RMAE'].mean()
    llm_mape = feat_data['LLM_MAPE'].mean()
    
    # 计算提升
    mae_improvement = gnn_mae - llm_mae
    rmse_improvement = gnn_rmse - llm_rmse
    mape_improvement = gnn_mape - llm_mape
    
    # 计算提升百分比
    mae_improvement_pct = (mae_improvement / gnn_mae * 100) if gnn_mae > 0 else 0
    rmse_improvement_pct = (rmse_improvement / gnn_rmse * 100) if gnn_rmse > 0 else 0
    mape_improvement_pct = (mape_improvement / gnn_mape * 100) if gnn_mape > 0 else 0
    
    print(f"\n{'指标':<15} {'GNN模型':<15} {'LLM模型':<15} {'绝对提升':<15} {'提升百分比':<15}")
    print(f"{'─'*75}")
    print(f"{'MAE':<15} {gnn_mae:<15.4f} {llm_mae:<15.4f} {mae_improvement:<15.4f} {mae_improvement_pct:<14.2f}%")
    print(f"{'RMSE (RMAE)':<15} {gnn_rmse:<15.4f} {llm_rmse:<15.4f} {rmse_improvement:<15.4f} {rmse_improvement_pct:<14.2f}%")
    print(f"{'MAPE':<15} {gnn_mape:<15.4f} {llm_mape:<15.4f} {mape_improvement:<15.4f} {mape_improvement_pct:<14.2f}%")
    
    # 统计显著改善的站点数
    improved_sites_mae = (feat_data['MAE_improvement'] > 0).sum()
    improved_sites_rmse = (feat_data['RMAE_improvement'] > 0).sum()
    improved_sites_mape = (feat_data['MAPE_improvement'] > 0).sum()
    total_sites = len(feat_data)
    
    print(f"\n📊 站点级别统计:")
    print(f"  MAE改善的站点数: {improved_sites_mae}/{total_sites} ({improved_sites_mae/total_sites*100:.1f}%)")
    print(f"  RMSE改善的站点数: {improved_sites_rmse}/{total_sites} ({improved_sites_rmse/total_sites*100:.1f}%)")
    print(f"  MAPE改善的站点数: {improved_sites_mape}/{total_sites} ({improved_sites_mape/total_sites*100:.1f}%)")

# ============================== 2. 全局平均统计 ==============================
print("\n" + "="*100)
print("🌍 全局平均性能对比（所有剩余站点和特征，排除18、55）")
print("="*100)

# GNN全局指标
gnn_global_mae = df['GNN_MAE'].mean()
gnn_global_rmse = df['GNN_RMAE'].mean()
gnn_global_mape = df['GNN_MAPE'].mean()

# LLM全局指标
llm_global_mae = df['LLM_MAE'].mean()
llm_global_rmse = df['LLM_RMAE'].mean()
llm_global_mape = df['LLM_MAPE'].mean()

# 全局提升
global_mae_improvement = gnn_global_mae - llm_global_mae
global_rmse_improvement = gnn_global_rmse - llm_global_rmse
global_mape_improvement = gnn_global_mape - llm_global_mape

global_mae_improvement_pct = (global_mae_improvement / gnn_global_mae * 100) if gnn_global_mae > 0 else 0
global_rmse_improvement_pct = (global_rmse_improvement / gnn_global_rmse * 100) if gnn_global_rmse > 0 else 0
global_mape_improvement_pct = (global_mape_improvement / gnn_global_mape * 100) if gnn_global_mape > 0 else 0

print(f"\n{'指标':<15} {'GNN模型':<15} {'LLM模型':<15} {'绝对提升':<15} {'提升百分比':<15}")
print(f"{'─'*75}")
print(f"{'MAE':<15} {gnn_global_mae:<15.4f} {llm_global_mae:<15.4f} {global_mae_improvement:<15.4f} {global_mae_improvement_pct:<14.2f}%")
print(f"{'RMSE (RMAE)':<15} {gnn_global_rmse:<15.4f} {llm_global_rmse:<15.4f} {global_rmse_improvement:<15.4f} {global_rmse_improvement_pct:<14.2f}%")
print(f"{'MAPE':<15} {gnn_global_mape:<15.4f} {llm_global_mape:<15.4f} {global_mape_improvement:<15.4f} {global_mape_improvement_pct:<14.2f}%")

# ============================== 3. 按站点统计 ==============================
print("\n" + "="*100)
print("🏢 按站点统计的模型性能（前10个剩余站点示例，排除18、55）")
print("="*100)

station_ids = df['station_id'].unique()[:10]

for station_id in station_ids:
    station_data = df[df['station_id'] == station_id]
    station_name = station_data.iloc[0]['station_short_name']
    
    print(f"\n站点 {station_id}: {station_name}")
    print(f"{'─'*80}")
    print(f"{'特征':<25} {'GNN_MAE':<10} {'LLM_MAE':<10} {'提升':<10} {'GNN_MAPE':<10} {'LLM_MAPE':<10} {'提升':<10}")
    print(f"{'─'*80}")
    
    for _, row in station_data.iterrows():
        feat_name = row['feature_name']
        gnn_mae = row['GNN_MAE']
        llm_mae = row['LLM_MAE']
        mae_imp = row['MAE_improvement']
        gnn_mape = row['GNN_MAPE']
        llm_mape = row['LLM_MAPE']
        mape_imp = row['MAPE_improvement']
        
        print(f"{feat_name:<25} {gnn_mae:<10.2f} {llm_mae:<10.2f} {mae_imp:<10.2f} {gnn_mape:<10.2f} {llm_mape:<10.2f} {mape_imp:<10.2f}")

# ============================== 4. 最佳和最差表现 ==============================
print("\n" + "="*100)
print("🏆 LLM相比GNN的最佳和最差表现（排除站点18、55）")
print("="*100)

# 按MAE提升排序
best_mae = df.nlargest(5, 'MAE_improvement')
worst_mae = df.nsmallest(5, 'MAE_improvement')

print("\n📈 MAE提升最大的前5个站点-特征组合:")
print(f"{'─'*90}")
print(f"{'排名':<5} {'站点':<15} {'特征':<25} {'GNN_MAE':<10} {'LLM_MAE':<10} {'提升':<10} {'提升%':<10}")
print(f"{'─'*90}")

for idx, (_, row) in enumerate(best_mae.iterrows(), 1):
    improvement_pct = (row['MAE_improvement'] / row['GNN_MAE'] * 100) if row['GNN_MAE'] > 0 else 0
    print(f"{idx:<5} {row['station_short_name']:<15} {row['feature_name']:<25} {row['GNN_MAE']:<10.2f} {row['LLM_MAE']:<10.2f} {row['MAE_improvement']:<10.2f} {improvement_pct:<9.2f}%")

print("\n📉 MAE提升最小（或下降）的前5个站点-特征组合:")
print(f"{'─'*90}")
print(f"{'排名':<5} {'站点':<15} {'特征':<25} {'GNN_MAE':<10} {'LLM_MAE':<10} {'提升':<10} {'提升%':<10}")
print(f"{'─'*90}")

for idx, (_, row) in enumerate(worst_mae.iterrows(), 1):
    improvement_pct = (row['MAE_improvement'] / row['GNN_MAE'] * 100) if row['GNN_MAE'] > 0 else 0
    print(f"{idx:<5} {row['station_short_name']:<15} {row['feature_name']:<25} {row['GNN_MAE']:<10.2f} {row['LLM_MAE']:<10.2f} {row['MAE_improvement']:<10.2f} {improvement_pct:<9.2f}%")

# ============================== 5. 保存结果 ==============================
output_dir = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/config/cai/inference_results_all_stations_with_predictions"
# 文件名增加排除站点标识，方便区分
output_path = os.path.join(output_dir, "model_comparison_summary_exclude_18_55.csv")

# 创建汇总表
summary_data = []

for feat_id in sorted(df['feature_id'].unique()):
    feat_data = df[df['feature_id'] == feat_id]
    feat_name = feature_names.get(feat_id, f"Feature {feat_id}")
    
    summary_data.append({
        'feature_id': feat_id,
        'feature_name': feat_name,
        'GNN_MAE': feat_data['GNN_MAE'].mean(),
        'LLM_MAE': feat_data['LLM_MAE'].mean(),
        'MAE_Improvement': feat_data['GNN_MAE'].mean() - feat_data['LLM_MAE'].mean(),
        'MAE_Improvement_Pct': ((feat_data['GNN_MAE'].mean() - feat_data['LLM_MAE'].mean()) / feat_data['GNN_MAE'].mean() * 100),
        'GNN_RMSE': feat_data['GNN_RMAE'].mean(),
        'LLM_RMSE': feat_data['LLM_RMAE'].mean(),
        'RMSE_Improvement': feat_data['GNN_RMAE'].mean() - feat_data['LLM_RMAE'].mean(),
        'RMSE_Improvement_Pct': ((feat_data['GNN_RMAE'].mean() - feat_data['LLM_RMAE'].mean()) / feat_data['GNN_RMAE'].mean() * 100),
        'GNN_MAPE': feat_data['GNN_MAPE'].mean(),
        'LLM_MAPE': feat_data['LLM_MAPE'].mean(),
        'MAPE_Improvement': feat_data['GNN_MAPE'].mean() - feat_data['LLM_MAPE'].mean(),
        'MAPE_Improvement_Pct': ((feat_data['GNN_MAPE'].mean() - feat_data['LLM_MAPE'].mean()) / feat_data['GNN_MAPE'].mean() * 100),
        'Sites_Improved_MAE': (feat_data['MAE_improvement'] > 0).sum(),
        'Total_Sites': len(feat_data),
        'Improvement_Rate_MAE': (feat_data['MAE_improvement'] > 0).sum() / len(feat_data) * 100
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 汇总结果已保存到: {output_path}")

# ============================== 6. 总结 ==============================
print("\n" + "="*100)
print("📝 总结（排除站点18、55）")
print("="*100)

print(f"\n全局性能对比:")
print(f"  MAE:  GNN={gnn_global_mae:.4f}, LLM={llm_global_mae:.4f}, 提升={global_mae_improvement:.4f} ({global_mae_improvement_pct:.2f}%)")
print(f"  RMSE: GNN={gnn_global_rmse:.4f}, LLM={llm_global_rmse:.4f}, 提升={global_rmse_improvement:.4f} ({global_rmse_improvement_pct:.2f}%)")
print(f"  MAPE: GNN={gnn_global_mape:.4f}, LLM={llm_global_mape:.4f}, 提升={global_mape_improvement:.4f} ({global_mape_improvement_pct:.2f}%)")

# 统计整体改善情况
overall_improved_mae = (df['MAE_improvement'] > 0).sum()
overall_total = len(df)
overall_rate_mae = overall_improved_mae / overall_total * 100

print(f"\n整体改善率:")
print(f"  MAE改善的样本比例: {overall_improved_mae}/{overall_total} ({overall_rate_mae:.2f}%)")

if global_mae_improvement > 0:
    print(f"\n✅ LLM模型整体优于GNN模型，MAE平均降低{global_mae_improvement:.4f} ({global_mae_improvement_pct:.2f}%)")
else:
    print(f"\n⚠️ LLM模型整体劣于GNN模型，MAE平均增加{-global_mae_improvement:.4f} ({-global_mae_improvement_pct:.2f}%)")

print("\n" + "="*100)