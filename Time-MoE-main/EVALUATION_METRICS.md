# Time-MoE 评估指标说明

## 📊 评估指标定义

### 1. MAE (Mean Absolute Error) - 平均绝对误差
```
MAE = (1/n) * Σ|y_true - y_pred|
```
- **含义**：预测值与真实值之间绝对误差的平均值
- **单位**：与原始数据相同（车辆数）
- **优点**：直观易懂，对异常值不敏感
- **缺点**：不能反映误差的方向

### 2. RMSE (Root Mean Square Error) - 均方根误差
```
RMSE = sqrt((1/n) * Σ(y_true - y_pred)²)
```
- **含义**：预测误差平方的平均值的平方根
- **单位**：与原始数据相同（车辆数）
- **优点**：对大误差更敏感，能突出极端预测错误
- **缺点**：对异常值敏感

### 3. MAPE (Mean Absolute Percentage Error) - 平均绝对百分比误差
```
MAPE = (100%/n) * Σ|(y_true - y_pred) / y_true|
```
- **含义**：预测误差占真实值的百分比的平均值
- **单位**：百分比 (%)
- **优点**：无量纲，便于不同量级数据间的比较
- **缺点**：当真实值接近0时会失真

---

## 📁 输出文件说明

### 1. `evaluation_metrics.json`
包含整体评估指标的JSON文件：

```json
{
  "MAE": 15.234,
  "RMSE": 22.567,
  "MAPE": 8.45,
  "timestamp": "2026-04-29 15:30:00"
}
```

**字段说明**：
- `MAE`: 平均绝对误差
- `RMSE`: 均方根误差
- `MAPE`: 平均绝对百分比误差（%）
- `timestamp`: 评估时间戳

### 2. `prediction_results.csv`
包含每个样本、每个时间点、每个特征的详细预测结果：

| 列名 | 说明 | 示例 |
|------|------|------|
| `sample_index` | 样本索引 | 0, 1, 2... |
| `prediction_time` | 预测时间 | 2023-09-01 09:00:00 |
| `feature` | 特征名称 | 小客车_上行 |
| `true_value` | 真实值 | 120.5 |
| `predicted_value` | 预测值 | 118.3 |
| `error` | 绝对误差 | 2.2 |
| `absolute_percentage_error` | 绝对百分比误差(%) | 1.83 |

**用途**：
- 分析模型在不同时间段的表现
- 识别预测误差较大的特定场景
- 进行细粒度的误差分析

### 3. `visualization_results/sample_X.png`
可视化图表，每个样本一张图：
- 展示真实值 vs 预测值的对比曲线
- 标题中包含该样本的 MAE 和 RMSE
- 多特征采用子图展示

---

## 🔍 如何使用评估结果

### 1. 快速查看整体性能
```bash
cat evaluation_metrics.json
```

### 2. 分析特定时间段的误差
```python
import pandas as pd

df = pd.read_csv('prediction_results.csv')

# 查看早高峰时段（7-9点）的误差
morning_peak = df[df['prediction_time'].str.contains(' 0[7-9]:')]
print(f"早高峰 MAE: {morning_peak['error'].mean():.2f}")

# 查看哪个特征误差最大
feature_errors = df.groupby('feature')['error'].mean()
print(feature_errors.sort_values(ascending=False))
```

### 3. 绘制误差分布
```python
import matplotlib.pyplot as plt

df = pd.read_csv('prediction_results.csv')

# 误差直方图
plt.figure(figsize=(10, 6))
plt.hist(df['error'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.title('Prediction Error Distribution')
plt.savefig('error_distribution.png', dpi=150)
plt.close()
```

---

## 💡 指标解读建议

### 交通流量预测的典型指标范围

| 指标 | 优秀 | 良好 | 一般 | 较差 |
|------|------|------|------|------|
| MAE | < 10 | 10-20 | 20-30 | > 30 |
| RMSE | < 15 | 15-25 | 25-40 | > 40 |
| MAPE | < 5% | 5-10% | 10-15% | > 15% |

**注意**：具体阈值取决于数据特性和业务需求。

### 常见问题诊断

1. **MAE 低但 RMSE 高**
   - 说明大部分预测准确，但存在少量极端错误
   - 建议：检查异常值，考虑使用鲁棒损失函数

2. **MAPE 异常高**
   - 可能存在真实值接近0的情况
   - 建议：过滤低值样本或使用 SMAPE

3. **不同特征间指标差异大**
   - 某些特征可能更难预测
   - 建议：分别调优各特征的预测策略

---

## 📝 引用

如果在论文或报告中使用这些指标，建议引用：

```bibtex
@article{hyndman2006another,
  title={Another look at measures of forecast accuracy},
  author={Hyndman, Rob J and Koehler, Anne B},
  journal={International journal of forecasting},
  volume={22},
  number={4},
  pages={679--688},
  year={2006},
  publisher={Elsevier}
}
```
