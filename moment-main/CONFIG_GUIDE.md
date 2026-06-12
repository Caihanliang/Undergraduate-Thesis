# 参数配置指南

## 🔧 关键参数说明

### 数据预处理参数 (`preprocess_data.py`)

```python
SEQ_LEN = 512   # 输入序列长度（小时）
PRED_LEN = 96   # 预测步长（小时）
```

**含义:**
- **SEQ_LEN**: 模型看到的历史数据长度
- **PRED_LEN**: 模型需要预测的未来数据长度
- **最小数据需求**: `SEQ_LEN + PRED_LEN` 个连续时间点

### 当前数据集情况

根据你的数据输出：
- **站点数**: 157个
- **时间范围**: 2023-09-01 到 2023-10-31（约61天 = 1464小时）
- **每个 split 的数据量**:
  - Train: ~1024 时间点 (70%)
  - Val: ~220 时间点 (15%)
  - Test: ~220 时间点 (15%)

### 推荐配置

#### 配置1：标准配置（当前设置）
```python
SEQ_LEN = 512  # ~21天历史
PRED_LEN = 96  # 4天预测
# 需要: 608个时间点
# 训练集: 1024 - 608 + 1 = 417个样本 ✓
```

#### 配置2：快速测试配置
```python
SEQ_LEN = 168  # 7天历史
PRED_LEN = 24  # 1天预测
# 需要: 192个时间点
# 训练集: 1024 - 192 + 1 = 833个样本 ✓✓✓
```

#### 配置3：中等配置
```python
SEQ_LEN = 336  # 14天历史
PRED_LEN = 48  # 2天预测
# 需要: 384个时间点
# 训练集: 1024 - 384 + 1 = 641个样本 ✓✓
```

## 📊 如何选择合适的参数

### 检查你的数据量

运行预处理后，查看输出：
```
Train: XXX time points (YYY features)
```

**计算公式:**
```
可生成样本数 = 时间点数 - SEQ_LEN - PRED_LEN + 1
```

**要求:**
- 至少生成 **10个样本** 才能训练
- 推荐 **100+ 样本** 获得较好效果
- 理想 **500+ 样本** 用于生产环境

### 调整步骤

1. **打开 `preprocess_data.py`**
2. **修改主函数中的参数**:
   ```python
   SEQ_LEN = 168  # 改为更小的值
   PRED_LEN = 24
   ```
3. **同时修改 `train_moment.py`**:
   ```python
   SEQ_LEN = 168  # 必须与预处理一致
   PRED_LEN = 24
   ```
4. **重新运行**:
   ```bash
   python preprocess_data.py
   python train_moment.py
   ```

## ⚠️ 常见错误及解决

### 错误1: `total_samples <= 0`
```
ValueError: ❌ Insufficient data! Need at least 608 time points, but only have 500.
```

**解决:**
- 减小 `SEQ_LEN` 或 `PRED_LEN`
- 或添加更多数据文件到 `dataset/` 目录

### 错误2: `Input shape: (0,)`
```
IndexError: tuple index out of range
```

**原因:** 预处理生成的数据为空

**解决:**
1. 检查预处理是否成功完成
2. 确认 `moment_data/dataset.npz` 文件大小 > 0
3. 重新运行 `python preprocess_data.py`
4. 查看详细错误信息

### 错误3: CUDA不可用
```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
Device: cpu
```

**影响:** CPU训练会非常慢（可能几天）

**解决:**
1. **更新NVIDIA驱动**（推荐）
   ```bash
   # 访问 http://www.nvidia.com/Download/index.aspx
   # 下载并安装最新驱动
   ```
2. **或使用较小的配置快速验证**
   ```python
   EPOCHS = 2  # 只训练2轮测试
   BATCH_SIZE = 8
   ```

## 🎯 快速开始建议

### 第一次运行（验证流程）

```python
# preprocess_data.py
SEQ_LEN = 168
PRED_LEN = 24

# train_moment.py
EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
```

**预期时间:**
- 预处理: 1-2分钟
- 训练 (CPU): 10-30分钟
- 训练 (GPU): 2-5分钟

### 正式训练

```python
# preprocess_data.py
SEQ_LEN = 512
PRED_LEN = 96

# train_moment.py
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
```

**预期时间:**
- 预处理: 2-3分钟
- 训练 (GPU): 30-60分钟
- 训练 (CPU): 不建议（可能需要数天）

## 📈 性能对比

| 配置 | SEQ_LEN | PRED_LEN | 训练样本数 | 训练难度 | 预测能力 |
|------|---------|----------|-----------|---------|---------|
| 快速测试 | 168 | 24 | 833 | ⭐ | 短期预测 |
| 中等 | 336 | 48 | 641 | ⭐⭐ | 中期预测 |
| 标准 | 512 | 96 | 417 | ⭐⭐⭐ | 长期预测 |

**注意:** 样本数越多，模型训练越稳定，但SEQ_LEN越长能捕捉更长期的模式。

## 🔍 调试技巧

### 检查生成的数据

```python
import numpy as np

# 加载数据
data = np.load('moment_data/dataset.npz')
print("Input shape:", data['input'].shape)
print("Target shape:", data['target'].shape)
print("Sample count:", len(data['input']))

# 检查是否有空值
print("Has NaN:", np.isnan(data['input']).any())
print("Has Inf:", np.isinf(data['input']).any())
```

### 可视化数据分布

```python
import matplotlib.pyplot as plt

data = np.load('moment_data/dataset.npz')
sample = data['input'][0]  # 第一个样本

plt.figure(figsize=(12, 6))
plt.plot(sample[:, 0], label='Feature 0')  # 第一个特征
plt.title('Sample Input Sequence')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid(True)
plt.show()
```

---

**最后更新**: 2026-04-22  
**适用版本**: MOMENT Traffic Flow v2.0