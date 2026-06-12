import pandas as pd

# 最最简化的版本
df = pd.read_csv("outputscientific8_W_161.csv", header=None)
df.insert(0, 'Time', pd.date_range("2023-09-01", periods=len(df), freq="60T"))

# 重命名列
new_columns = ['Time'] + [f'Station_{i:03d}' for i in range(1, len(df.columns))]
df.columns = new_columns

# 保存
df.to_excel("hunan_highway_traffic.xlsx", index=False)
print("✅ 完成！")
