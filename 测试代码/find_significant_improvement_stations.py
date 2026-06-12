import pandas as pd
import numpy as np

# 读取CSV文件
file_path = '/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/config/cai/inference_results_all_stations_with_predictions/all_station_feature_mean_metrics.csv'
df = pd.read_csv(file_path)

print("=" * 100)
print("分析四个特征都显著提升的站点")
print("=" * 100)

# 检查数据基本信息
print(f"\n总记录数: {len(df)}")
print(f"唯一站点数: {df['station_id'].nunique()}")
print(f"特征列表: {df['feature_name'].unique()}")

# 定义显著提升的阈值（可以根据需要调整）
# MAE_improvement > 0 表示LLM比GNN好，数值越大提升越显著
MAE_THRESHOLD = 0  # MAE提升阈值
MAPE_THRESHOLD = 0  # MAPE提升阈值
RMAE_THRESHOLD = 0  # RMAE提升阈值

print(f"\n判断标准:")
print(f"  - MAE提升 > {MAE_THRESHOLD}")
print(f"  - MAPE提升 > {MAPE_THRESHOLD}")
print(f"  - RMAE提升 > {RMAE_THRESHOLD}")
print(f"  - 四个特征都需要满足上述条件")

# 按站点和特征分组，找出每个站点四个特征都显著提升的记录
def check_station_improvement(station_df):
    """检查一个站点的所有四个特征是否都有显著提升"""
    # 确保该站点有4个特征的数据
    if len(station_df) != 4:
        return False, None
    
    # 检查每个特征是否都有显著提升
    for _, row in station_df.iterrows():
        if (row['MAE_improvement'] <= MAE_THRESHOLD or 
            row['MAPE_improvement'] <= MAPE_THRESHOLD or 
            row['RMAE_improvement'] <= RMAE_THRESHOLD):
            return False, None
    
    return True, station_df

# 按站点分组并检查
results = []
for station_id, station_data in df.groupby('station_id'):
    is_significant, data = check_station_improvement(station_data)
    if is_significant:
        results.append({
            'station_id': station_id,
            'station_short_name': station_data.iloc[0]['station_short_name'],
            'station_name': station_data.iloc[0]['station_name'],
            'avg_mae_improvement': station_data['MAE_improvement'].mean(),
            'avg_mape_improvement': station_data['MAPE_improvement'].mean(),
            'avg_rmae_improvement': station_data['RMAE_improvement'].mean(),
            'min_mae_improvement': station_data['MAE_improvement'].min(),
            'min_mape_improvement': station_data['MAPE_improvement'].min(),
            'min_rmae_improvement': station_data['RMAE_improvement'].min(),
        })

# 转换为DataFrame并按平均MAE提升排序
if results:
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('avg_mae_improvement', ascending=False)
    
    print(f"\n{'=' * 100}")
    print(f"找到 {len(result_df)} 个四个特征都显著提升的站点")
    print(f"{'=' * 100}\n")
    
    # 显示详细结果
    print("站点详细信息（按平均MAE提升降序排列）:\n")
    for idx, row in result_df.iterrows():
        print(f"站点ID: {int(row['station_id'])}")
        print(f"站点简称: {row['station_short_name']}")
        print(f"站点全称: {row['station_name']}")
        print(f"  平均MAE提升: {row['avg_mae_improvement']:.4f}")
        print(f"  平均MAPE提升: {row['avg_mape_improvement']:.4f}")
        print(f"  平均RMAE提升: {row['avg_rmae_improvement']:.4f}")
        print(f"  最小MAE提升: {row['min_mae_improvement']:.4f}")
        print(f"  最小MAPE提升: {row['min_mape_improvement']:.4f}")
        print(f"  最小RMAE提升: {row['min_rmae_improvement']:.4f}")
        print("-" * 100)
    
    # 保存结果到CSV
    output_path = '/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main/config/cai/inference_results_all_stations_with_predictions/significant_improvement_stations.csv'
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_path}")
    
    # 统计信息
    print(f"\n{'=' * 100}")
    print("统计摘要")
    print(f"{'=' * 100}")
    print(f"总站点数: {df['station_id'].nunique()}")
    print(f"显著提升站点数: {len(result_df)}")
    print(f"占比: {len(result_df)/df['station_id'].nunique()*100:.2f}%")
    print(f"\n平均MAE提升: {result_df['avg_mae_improvement'].mean():.4f}")
    print(f"平均MAPE提升: {result_df['avg_mape_improvement'].mean():.4f}")
    print(f"平均RMAE提升: {result_df['avg_rmae_improvement'].mean():.4f}")
    
else:
    print("\n未找到四个特征都显著提升的站点")
    print("建议降低阈值或检查数据质量")

# 可选：显示所有站点的四个特征的详细对比
print(f"\n{'=' * 100}")
print("所有站点各特征的MAE提升情况（前20个站点）")
print(f"{'=' * 100}\n")

sample_stations = df['station_id'].unique()[:20]
for station_id in sample_stations:
    station_data = df[df['station_id'] == station_id]
    print(f"站点ID {int(station_id)} ({station_data.iloc[0]['station_short_name']}):")
    for _, row in station_data.iterrows():
        marker = "✓" if (row['MAE_improvement'] > 0 and row['MAPE_improvement'] > 0 and row['RMAE_improvement'] > 0) else "✗"
        print(f"  {marker} {row['feature_name']:30s} | MAE提升: {row['MAE_improvement']:8.4f} | MAPE提升: {row['MAPE_improvement']:8.4f} | RMAE提升: {row['RMAE_improvement']:8.4f}")
    print()
