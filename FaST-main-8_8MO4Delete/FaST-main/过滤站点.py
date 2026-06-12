import pandas as pd

# ===================== 你的文件路径（直接使用）=====================
file_path = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/inference_results_skip_14stations/all_station_feature_mean_metrics.csx"

# 1. 读取文件（.csx 实际就是 CSV 格式）
df = pd.read_csv(file_path)

# 2. 筛选条件：淘汰 MAE_improvement 减幅超过 10 的站点
# 减幅超过10 = 数值 < -10
淘汰条件 = df["MAE_improvement"] < -10
淘汰站点 = df[淘汰条件]["station_id"].tolist()

# 3. 打印被淘汰的站点编号
print("=" * 60)
print("被淘汰的站点编号（MAE_improvement 减幅 > 10）：")
for sid in 淘汰站点:
    print(f"站点 ID: {sid}")
print("=" * 60)

# 4. 保留合格数据（剔除不合格站点）
df_clean = df[~淘汰条件].copy()

# 5. 保存为新的 CSV 文件
output_path = "/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/inference_results_skip_14stations/filtered_stations.csv"
df_clean.to_csv(output_path, index=False, encoding="utf-8")

print(f"\n处理完成！")
print(f"原始数据行数: {len(df)}")
print(f"过滤后行数: {len(df_clean)}")
print(f"淘汰站点数量: {len(淘汰站点)}")
print(f"过滤后文件已保存到: {output_path}")