"""
python LLM改进程度统计.py
"""
import pandas as pd

# 读取过滤后的站点数据文件
file_path = "config/cai/inference_results_skip_14stations/filtered_stations.csv"
df = pd.read_csv(file_path)

# 按feature_id、feature_name分组，计算所有需求指标
feature_summary = df.groupby(["feature_id", "feature_name"]).agg(
    station_count=("station_id", "nunique"),
    total_samples=("num_samples", "sum"),
    total_points=("num_points", "sum"),
    fallback_count=("fallback_count", "sum"),
    GNN_MAE=("GNN_MAE", "mean"),
    GNN_MAPE=("GNN_MAPE", "mean"),
    GNN_RMAE=("GNN_RMAE", "mean"),
    LLM_MAE=("LLM_MAE", "mean"),
    LLM_MAPE=("LLM_MAPE", "mean"),
    LLM_RMAE=("LLM_RMAE", "mean"),
    MAE_improvement=("MAE_improvement", "mean"),
    MAPE_improvement=("MAPE_improvement", "mean"),
    RMAE_improvement=("RMAE_improvement", "mean")
).reset_index()

# 按feature_id升序排序
feature_summary = feature_summary.sort_values("feature_id").reset_index(drop=True)

# 保存结果
output_path = "config/cai/inference_results_skip_14stations/feature_summary_metrics.csv"
feature_summary.to_csv(output_path, index=False, encoding="utf-8")

# 打印结果
print("="*100)
print("特征维度汇总指标生成完成！")
print("="*100)
print(feature_summary.to_string(index=False))
print("="*100)
print(f"总特征数量：{len(feature_summary)}")
print(f"文件已保存至：{output_path}")