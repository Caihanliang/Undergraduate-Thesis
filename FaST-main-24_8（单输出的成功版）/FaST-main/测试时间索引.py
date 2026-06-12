import numpy as np

# 加载三个索引文件
idx_train = np.load('main-master/datasets/HNGS/24_8/idx_train.npy')
idx_val = np.load('main-master/datasets/HNGS/24_8/idx_val.npy')
idx_test = np.load('main-master/datasets/HNGS/24_8/idx_test.npy')

print("=" * 70)
print("📊 HNGS 数据集索引详细分析")
print("=" * 70)

print(f"\n【训练集】")
print(f"   索引范围：{idx_train[0]} - {idx_train[-1]}")
print(f"   样本数量：{len(idx_train)}")
print(f"   第一个样本索引：{idx_train[0]}")
print(f"   最后一个样本索引：{idx_train[-1]}")

print(f"\n【验证集】")
print(f"   索引范围：{idx_val[0]} - {idx_val[-1]}")
print(f"   样本数量：{len(idx_val)}")
print(f"   第一个样本索引：{idx_val[0]}")
print(f"   最后一个样本索引：{idx_val[-1]}")

print(f"\n【测试集】")
print(f"   索引范围：{idx_test[0]} - {idx_test[-1]}")
print(f"   样本数量：{len(idx_test)}")
print(f"   第一个样本索引：{idx_test[0]}")
print(f"   最后一个样本索引：{idx_test[-1]}")

print("\n" + "=" * 70)
print("🔍 关键问题检查")
print("=" * 70)

# 检查是否连续
print(f"\n连续性检查:")
print(f"   训练集结束索引：{idx_train[-1]}")
print(f"   验证集开始索引：{idx_val[0]}")
print(f"   间隔：{idx_val[0] - idx_train[-1]}")

print(f"\n   验证集结束索引：{idx_val[-1]}")
print(f"   测试集开始索引：{idx_test[0]}")
print(f"   间隔：{idx_test[0] - idx_val[-1]}")

# 您的理解是否正确
print("\n" + "=" * 70)
print("❓ 验证您的理解")
print("=" * 70)

total_samples = len(idx_train) + len(idx_val) + len(idx_test)
print(f"\n总样本数：{total_samples}")
print(f"\n您的理解:")
print(f"   训练集：859 个小时 ❌ (实际是 {len(idx_train)} 个样本)")
print(f"   验证集：286 个 ❌ (实际是 {len(idx_val)} 个样本)")
print(f"   测试集：288 个 ❌ (实际是 {len(idx_test)} 个样本)")

print(f"\n实际情况:")
print(f"   训练集：{len(idx_train)} 个样本 ({len(idx_train)/total_samples*100:.1f}%)")
print(f"   验证集：{len(idx_val)} 个样本 ({len(idx_val)/total_samples*100:.1f}%)")
print(f"   测试集：{len(idx_test)} 个样本 ({len(idx_test)/total_samples*100:.1f}%)")

# 计算对应的时间范围
data = np.load('main-master/datasets/HNGS/his.npz')['data']
T, N, D = data.shape
print(f"\n原始数据总时间步数：{T}")

from datetime import datetime, timedelta
import json

with open('main-master/datasets/HNGS/desc.json', 'r') as f:
    desc = json.load(f)

start_date = desc['time_range'].split(' to ')[0]
start_dt = datetime.strptime(start_date, "%Y-%m-%d")

print(f"数据集起始时间：{start_date}")

train_start_time = start_dt + timedelta(hours=int(idx_train[0]))
train_end_time = start_dt + timedelta(hours=int(idx_train[-1]))
val_start_time = start_dt + timedelta(hours=int(idx_val[0]))
val_end_time = start_dt + timedelta(hours=int(idx_val[-1]))
test_start_time = start_dt + timedelta(hours=int(idx_test[0]))
test_end_time = start_dt + timedelta(hours=int(idx_test[-1]))

print(f"\n📅 实际时间范围（时间粒度：1 小时）:")
print(f"   训练集：{train_start_time.strftime('%Y-%m-%d %H:%M')} - {train_end_time.strftime('%Y-%m-%d %H:%M')}")
print(f"   验证集：{val_start_time.strftime('%Y-%m-%d %H:%M')} - {val_end_time.strftime('%Y-%m-%d %H:%M')}")
print(f"   测试集：{test_start_time.strftime('%Y-%m-%d %H:%M')} - {test_end_time.strftime('%Y-%m-%d %H:%M')}")
# python 测试时间索引.py
"""
📅 实际时间范围（时间粒度：1 小时）:
   训练集：2023-09-02 07:00 - 2023-10-08 01:00
   验证集：2023-10-08 02:00 - 2023-10-19 23:00
   测试集：2023-10-20 00:00 - 2023-10-31 23:00
你的模型要用 前 24 小时 + 未来 8 小时 来构造样本！
所以前面 31 个小时必须被吃掉，不能当样本  
"""