import pandas as pd
import numpy as np

# ---------------------- 1. 模型参数（和你训练完全一致） ----------------------
DATA_NAME = 'HNGS'
num_nodes = 161
INPUT_LEN = 24    # 输入24步
OUTPUT_LEN = 8    # 预测8步
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
NULL_VAL = 0

# ---------------------- 2. 读取数据 ----------------------
df = pd.read_excel('DataPipeline/hunan_highway_traffic.xlsx')
df['Time'] = pd.to_datetime(df['Time'])
df = df.sort_values('Time').reset_index(drop=True)

station_cols = [col for col in df.columns if col.startswith('Station_')]
df[station_cols] = df[station_cols].fillna(method='ffill').fillna(method='bfill')

# ---------------------- 3. 【重要】对原始流量做归一化 ----------------------
data = df[station_cols].values
mean = data.mean()
std = data.std()
data_normalized = (data - mean) / std  # 和模型训练一致！

print(f"✅ 数据归一化完成 | mean={mean:.2f}, std={std:.2f}")

# ---------------------- 4. 生成时序样本 ----------------------
total_samples = len(data_normalized) - INPUT_LEN - OUTPUT_LEN + 1
y_true = np.zeros((total_samples, OUTPUT_LEN, num_nodes, 1), dtype=np.float32)

for t in range(OUTPUT_LEN):
    y_true[:, t, :, 0] = data_normalized[INPUT_LEN + t : INPUT_LEN + t + total_samples]

# ---------------------- 5. 划分训练/验证/测试 ----------------------
train_size = int(total_samples * TRAIN_VAL_TEST_RATIO[0])
val_size = int(total_samples * TRAIN_VAL_TEST_RATIO[1])

ytrue_train = y_true[:train_size]
ytrue_val   = y_true[train_size:train_size+val_size]
ytrue_test  = y_true[train_size+val_size:]

print(f"\n✅ 标签形状：")
print(f"训练集 {ytrue_train.shape}")
print(f"验证集 {ytrue_val.shape}")
print(f"测试集 {ytrue_test.shape}")

# ---------------------- 6. 保存文件 ----------------------
np.savez('cai-config/ture/ytrue_train.npz', ytrue_train=ytrue_train)
np.savez('cai-config/ture/ytrue_val.npz',   ytrue_val=ytrue_val)
np.savez('cai-config/ture/ytrue_test.npz',  ytrue_test=ytrue_test)

print(f"\n✅ 全部标签生成完成！")