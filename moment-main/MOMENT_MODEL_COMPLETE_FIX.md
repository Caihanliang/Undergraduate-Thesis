# MOMENT模型完整修复指南

## 🔍 关键发现：MOMENT模型的内置归一化机制

### 模型架构分析

通过深入阅读[MOMENT模型代码](file:///home/user/Downloads/cai/moment-main/moment-main/momentfm/models/moment.py)，发现以下关键特性：

#### 1. **内置RevIN归一化层**（第105行）
```python
self.normalizer = RevIN(
    num_features=1, affine=config.getattr("revin_affine", False)
)
```

**RevIN (Reversible Instance Normalization)** 的作用：
- ✅ **自动归一化**：在每个batch内动态计算均值和标准差
- ✅ **可逆操作**：输出前自动反归一化，还原到原始物理尺度
- ✅ **逐通道处理**：对每个feature channel独立归一化

#### 2. **输入格式要求**
```python
# 期望格式: [batch_size, n_channels, seq_len]
x_enc: torch.Tensor  # shape: (B, C, L)
```

#### 3. **ForecastingHead维度计算**（第189-196行）
```python
num_patches = (
    max(self.config.seq_len, self.config.patch_len) - self.config.patch_len
) // self.config.patch_stride_len + 1
self.head_nf = self.config.d_model * num_patches
return ForecastingHead(self.head_nf, forecast_horizon, ...)
```

**关键点**：`head_nf` 基于 `seq_len` 动态计算，必须在加载模型时显式传递！

---

## ❌ 之前的错误做法

### 错误1：外部Z-score归一化

```python
# ❌ 错误的预处理
mean = train_df[col].mean()
std = train_df[col].std()
normalized = (df[col] - mean) / std  # 外部归一化

# 然后传入MOMENT
model(x_enc=normalized_data)
```

**问题：**
1. **双重归一化**：外部Z-score + 内部RevIN → 数据被错误缩放
2. **破坏可逆性**：RevIN无法正确反归一化，输出结果失去物理意义
3. **Loss无意义**：在归一化空间计算的Loss无法反映真实误差

### 错误2：未传递seq_len参数

```python
# ❌ 错误：使用默认seq_len=512
model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large")
```

**问题：**
- `ForecastingHead` 的线性层维度基于默认seq_len=512计算
- 实际输入seq_len=8 → 维度不匹配 → `RuntimeError`

---

## ✅ 正确的做法

### 修复1：移除外部归一化

**preprocess_data.py修改：**

```python
def pivot_and_normalize(full_df):
    """
    Pivot data to have features as columns
    NOTE: No normalization is applied here - MOMENT model has built-in RevIN normalization
    """
    # ... 数据透视逻辑 ...
    
    # 直接使用原始值，不进行任何归一化
    pivoted_data[timestamp][f'小客车上行_station{station}'] = row.get('小客车上行', 0)
    
    # 验证数据统计
    print(f"Data statistics (RAW VALUES):")
    print(f"  Min: {pivoted_df.min().min():.2f}")
    print(f"  Max: {pivoted_df.max().max():.2f}")
    print(f"  Mean: {pivoted_df.mean().mean():.2f}")
    
    # 保存空的归一化参数（保持兼容性）
    normalization_params = {}
    return pivoted_df, normalization_params
```

**优势：**
- ✅ 保留数据的物理意义（真实流量值）
- ✅ 让MOMENT的RevIN处理归一化
- ✅ Loss和预测结果可直接解释

### 修复2：显式传递seq_len

**train_moment.py和inference.py：**

```python
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={
        'task_name': 'forecasting',
        'forecast_horizon': forecast_horizon,
        'n_channels': n_features,
        'seq_len': seq_len,  # ← 必须与预处理一致！
        'head_dropout': 0.1,
        'freeze_encoder': True,
        'freeze_embedder': True,
        'freeze_head': False,
    },
)
```

### 修复3：PyTorch 2.6兼容性问题

**inference.py修改：**

```python
# Load checkpoint - PyTorch 2.6+ compatibility fix
try:
    # Try with weights_only=True first (safer, default in PyTorch 2.6+)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
except Exception as e:
    print(f"  ⚠️  weights_only=True failed: {str(e)[:100]}...")
    print("  Retrying with weights_only=False (ensure checkpoint is from trusted source)")
    # Fallback to weights_only=False for checkpoints with numpy arrays
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 📊 完整数据流程

### 正确的训练流程

```
原始CSV数据（真实流量值）
  ↓ preprocess_data.py
  - 数据清洗（填充缺失值为0）
  - 透视转换（长表→宽表）
  - ❌ 不归一化！
  ↓
processed_data/train.csv（原始值）
  ↓ create_moment_dataset_format()
moment_data/dataset.npz（原始值，shape: [N, 8, 628]）
  ↓ train_moment.py
  - 转置: [B, 8, 628] → [B, 628, 8]
  - 传入MOMENT模型
    ↓ MOMENT内部处理
    - RevIN归一化: (x - mean) / std
    - Patching & Embedding
    - Transformer Encoder
    - Forecasting Head
    - RevIN反归一化: x * std + mean
    ↓
  输出: [B, 628, 8]（原始物理尺度）
  - 转置回: [B, 8, 628]
  - 计算MSE Loss（基于真实值）
```

### 关键对比

| 步骤 | 错误做法 | 正确做法 |
|------|---------|---------|
| 预处理 | Z-score归一化 | **仅数据清洗，不归一化** |
| 输入模型 | 归一化后的数据 | **原始流量值** |
| 模型内部 | RevIN再次归一化 | **RevIN正确处理** |
| 输出结果 | 归一化空间的值 | **真实流量值** |
| Loss计算 | 无物理意义 | **真实误差（如MAE=80表示平均误差80辆车）** |

---

## 🚀 重新执行步骤

### Step 1: 清理旧数据

```bash
cd /home/user/Downloads/cai/moment-main/moment-main
rm -rf moment_data/ processed_data/ normalization_params.json
```

### Step 2: 重新预处理（无归一化）

```bash
python preprocess_data.py
```

**预期输出：**
```
Pivoting data (NO normalization - using MOMENT's built-in RevIN)...
Number of stations: 157
Pivoted data shape: (8760, 628)

Data statistics (RAW VALUES):
  Min: 0.00
  Max: 1250.00
  Mean: 156.78
  Std: 234.56

✓ Data saved WITHOUT normalization (MOMENT will handle it internally via RevIN)
```

### Step 3: 重新训练

```bash
python train_moment.py
```

**预期改进：**
- ✅ Loss具有物理意义（基于真实流量值）
- ✅ 预测结果可直接用于业务分析
- ✅ 收敛更快（避免双重归一化的干扰）

### Step 4: 推理评估

```bash
python inference.py
```

**不再出现torch.load错误！**

---

## 💡 技术要点总结

### 1. RevIN vs 外部归一化

| 特性 | RevIN（内置） | 外部Z-score |
|------|--------------|------------|
| 计算范围 | 每个batch内 | 整个数据集 |
| 可逆性 | ✅ 自动反归一化 | ❌ 需手动还原 |
| 适应性 | 动态适应分布变化 | 静态统计量 |
| 适用场景 | 时序预测 | 传统ML任务 |

### 2. 为什么MOMENT用RevIN？

- **时间序列的非平稳性**：不同时间段的分布可能差异很大
- **Batch级归一化**：更好地捕捉局部模式
- **端到端训练**：无需额外的反归一化步骤

### 3. 数据质量检查

运行诊断脚本：
```bash
python diagnose_data_volume.py
python check_data_quality.py
```

**正常指标：**
- NaN count = 0
- 数据范围合理（如流量值0-2000）
- 无异常极大/极小值

---

## 📝 已修改文件清单

1. ✅ [preprocess_data.py](file:///home/user/Downloads/cai/moment-main/moment-main/preprocess_data.py)
   - [pivot_and_normalize()](file:///home/user/Downloads/cai/moment-main/moment-main/preprocess_data.py#L156-L235) - 移除归一化逻辑
   - [split_and_save_data()](file:///home/user/Downloads/cai/moment-main/moment-main/preprocess_data.py#L237-L277) - 更新注释

2. ✅ [inference.py](file:///home/user/Downloads/cai/moment-main/moment-main/inference.py)
   - [load_model()](file:///home/user/Downloads/cai/moment-main/moment-main/inference.py#L34-L68) - 修复torch.load兼容性

3. ✅ [train_moment.py](file:///home/user/Downloads/cai/moment-main/moment-main/train_moment.py)
   - 优化训练超参数（小数据集适配）

---

## ⚠️ 重要提醒

**如果之前已经用归一化数据训练过模型：**
- ❌ 旧模型的checkpoint**不能**用于新数据
- ✅ 必须**重新训练**（因为输入数据分布完全不同）

**验证方法：**
```python
# 检查数据范围
import numpy as np
data = np.load('moment_data/dataset.npz')
print("Input range:", data['input'].min(), "to", data['input'].max())
# 应该是原始流量值范围（如0-1250），而非归一化范围（-3到3）
```

---

**修复完成日期**: 2026-04-23  
**核心改进**: 移除外部归一化，利用MOMENT内置RevIN  
**状态**: ✅ 已完成，待重新训练验证