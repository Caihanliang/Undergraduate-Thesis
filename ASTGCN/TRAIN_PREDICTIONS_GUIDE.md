# ASTGCN 训练集预测结果生成指南

## 📋 功能说明

此脚本用于**预测训练集样本**并将结果保存为CSV格式，便于后续分析和可视化。

### 输出格式

参考 `/home/user/Downloads/cai/结果汇总98/moment/train_predictions.csv` 的格式：

```csv
站点名称,站点编号,时间,特征,真实值,预测值
Station_000,000,Sample_0,Passenger_Car_Up,688.0,673.96
Station_000,000,Sample_0,Passenger_Car_Down,577.0,578.47
Station_000,000,Sample_0,Non_Passenger_Car_Up,256.0,332.42
Station_000,000,Sample_0,Non_Passenger_Car_Down,330.0,447.09
```

### 关键特性

✅ **长格式（Long Format）**: 每个样本-站点-特征-时间步组合占一行  
✅ **自动反归一化**: 使用训练时的归一化参数还原真实数值  
✅ **多变量支持**: 同时处理4个特征（小客车上/下行、非小客车上/下行）  
✅ **站点映射**: 自动加载站点名称和编号（如果存在 `station_mapping.csv`）  
✅ **统计报告**: 输出分特征的 MAE、RMSE、SMAPE 指标  

---

## 🚀 使用方法

### **步骤1: 确保已完成训练**

```bash
# 确认最佳模型已保存
ls experiments/highway_traffic/astgcn_r_h1d0w0_channel4_1.000000e-03/epoch_*.params
```

应该看到类似：
```
epoch_89.params  # 最佳模型
```

### **步骤2: 运行预测脚本**

```bash
cd /home/user/Downloads/cai/ASTGCN

python predict_and_save_train.py --config configurations/highway_traffic.conf
```

### **步骤3: 查看输出结果**

脚本会生成文件：
```
experiments/highway_traffic/astgcn_r_h1d0w0_channel4_1.000000e-03/train_predictions.csv
```

---

## 📊 输出示例

### **控制台输出**

```
============================================================
ASTGCN 训练集预测结果生成
============================================================
Read configuration file: configurations/highway_traffic.conf
CUDA: True cuda:0

📂 加载训练数据...
   训练集 x: (865, 98, 4, 8)
   训练集 y: (865, 98, 4, 8)
✅ 邻接矩阵加载成功: shape=(98, 98)

🔧 创建模型...
📦 加载模型: experiments/.../epoch_89.params
✅ 模型加载完成 (Epoch 89)

🔄 准备训练集数据...
   Batch数量: 28

🚀 开始预测训练集...
   进度: 10/28 batches
   进度: 20/28 batches
   进度: 28/28 batches

✅ 预测完成!
   预测值形状: (865, 98, 4, 8)
   真实值形状: (865, 98, 4, 8)

📊 反归一化...

💾 转换为CSV格式...
   正在处理 865 个样本 × 98 个站点 × 4 个特征 × 8 个时间步...
✅ DataFrame创建完成: 2712320 行

💾 保存到: experiments/.../train_predictions.csv

✅ 保存成功!
   文件路径: experiments/.../train_predictions.csv
   文件大小: XX.XX MB

============================================================
📊 预测结果统计
============================================================

总体统计:
  样本数: 865
  站点数: 98
  特征数: 4
  时间步: 8
  总行数: 2712320

按特征分组统计:
  Passenger_Car_Up:
    MAE: XX.XX
    RMSE: XX.XX
    SMAPE: XX.XX%
  ...

前5行预览:
     站点名称  站点编号      时间              特征  真实值   预测值
0  Station_000  000  Sample_0_T0  Passenger_Car_Up  688.0  673.96
1  Station_000  000  Sample_0_T0  Passenger_Car_Down  577.0  578.47
...

============================================================
✅ 全部完成!
============================================================
```

---

## 🔍 CSV文件结构

### **列说明**

| 列名 | 说明 | 示例 |
|------|------|------|
| **站点名称** | 站点的中文名称 | `Station_000` |
| **站点编号** | 站点的数字编号 | `000` |
| **时间** | 样本索引和时间步 | `Sample_0_T0`, `Sample_0_T1`, ... |
| **特征** | 交通流量类型 | `Passenger_Car_Up`（小客车上行）<br>`Passenger_Car_Down`（小客车下行）<br>`Non_Passenger_Car_Up`（非小客车上行）<br>`Non_Passenger_Car_Down`（非小客车下行） |
| **真实值** | 实际观测的交通流量 | `688.0` |
| **预测值** | 模型预测的交通流量 | `673.96` |

### **数据量计算**

```
总行数 = 样本数 × 站点数 × 特征数 × 时间步
       = 865 × 98 × 4 × 8
       = 2,712,320 行
```

---

## 💡 应用场景

### **1. 误差分析**

```python
import pandas as pd
import numpy as np

# 加载预测结果
df = pd.read_csv('experiments/.../train_predictions.csv')

# 计算每个特征的MAE
for feature in df['特征'].unique():
    feat_data = df[df['特征'] == feature]
    mae = np.mean(np.abs(feat_data['真实值'] - feat_data['预测值']))
    print(f"{feature}: MAE = {mae:.2f}")
```

### **2. 可视化特定站点**

```python
# 筛选特定站点和特征
station_0 = df[(df['站点编号'] == '000') & (df['特征'] == 'Passenger_Car_Up')]

# 绘制真实值vs预测值
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(station_0['时间'], station_0['真实值'], label='真实值', marker='o')
plt.plot(station_0['时间'], station_0['预测值'], label='预测值', marker='s')
plt.legend()
plt.title('Station 000 - Passenger Car Up')
plt.show()
```

### **3. 导出到Excel进行进一步分析**

```python
# 转换为宽格式便于Excel查看
pivot_df = df.pivot_table(
    index=['站点名称', '站点编号', '时间'],
    columns='特征',
    values=['真实值', '预测值']
)

pivot_df.to_excel('train_predictions_wide.xlsx')
```

---

## ⚠️ 注意事项

### **1. 内存占用**

- 对于大规模数据集（如本例的270万行），CSV文件可能较大（~100MB）
- 如果内存不足，可以考虑：
  - 只保存部分样本（修改脚本中的循环范围）
  - 使用 Parquet 格式替代 CSV（更紧凑）

### **2. 站点映射**

- 如果存在 `dataset/station_mapping.csv`，会自动加载站点名称
- 否则使用默认命名：`Station_000`, `Station_001`, ...

### **3. 时间标识**

- 当前使用 `Sample_{b}_T{t}` 格式
- 如果需要真实时间戳，可以修改脚本加载 `timestamps.csv`

### **4. 模型选择**

- 脚本自动选择最后一个保存的 epoch 作为最佳模型
- 如需指定特定 epoch，可以修改代码中的 `best_epoch` 变量

---

## 🔧 自定义选项

### **修改输出路径**

```python
# 在脚本中修改
output_dir = './my_results'  # 自定义目录
output_file = os.path.join(output_dir, 'my_predictions.csv')
```

### **只保存特定特征**

```python
# 在循环中添加过滤条件
if f_idx != 0:  # 只保存第一个特征
    continue
```

### **添加额外列**

```python
row = {
    '站点名称': station_name,
    '站点编号': station_code,
    '时间': f'Sample_{b}_T{t}',
    '特征': feature_names[f_idx],
    '真实值': round(targets_denorm[b, n, f_idx, t], 2),
    '预测值': round(predictions_denorm[b, n, f_idx, t], 2),
    '绝对误差': round(abs(targets_denorm[b, n, f_idx, t] - predictions_denorm[b, n, f_idx, t]), 2),
    '相对误差': round(abs(targets_denorm[b, n, f_idx, t] - predictions_denorm[b, n, f_idx, t]) / (targets_denorm[b, n, f_idx, t] + 1e-8) * 100, 2)
}
```

---

## 📈 后续分析建议

1. **误差分布分析**: 绘制误差直方图，识别异常样本
2. **时间模式分析**: 对比不同时间段的预测精度
3. **站点聚类**: 根据预测误差对站点进行聚类
4. **特征相关性**: 分析4个特征之间的预测误差相关性

---

**祝分析顺利！** 🎯
