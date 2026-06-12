# MOMENT模型维度问题修复说明

## 🔍 问题描述

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x79872 and 65536x8)
```

## 📊 根本原因

**MOMENT模型的输入格式要求：**
- **期望输入**: `[batch_size, n_channels, seq_len]`
- **我们的数据**: `[batch_size, seq_len, n_features]`

**维度不匹配导致线性层计算失败：**
```
head_nf = d_model * num_patches
        = 768 * 64 = 49152 (或其他值)

但实际传入的展平维度与期望不符，导致矩阵乘法失败
```

## ✅ 解决方案

### 关键修改：数据转置

在调用MOMENT模型前，需要将数据从 `[batch, seq_len, features]` 转置为 `[batch, features, seq_len]`。

### 修改位置

#### 1. 训练脚本 (`train_moment.py`)

**训练循环:**
```python
# Before (错误):
x_enc = batch['x_enc'].to(device)  # [batch, seq_len, n_features]
output = model(x_enc=x_enc)

# After (正确):
x_enc = batch['x_enc'].to(device)  # [batch, seq_len, n_features]
x_enc = x_enc.permute(0, 2, 1)     # [batch, n_features, seq_len] ✓
output = model(x_enc=x_enc)
predictions = output.forecast       # [batch, n_features, pred_len]
predictions = predictions.permute(0, 2, 1)  # [batch, pred_len, n_features] ✓
```

**验证循环:**
```python
# 同样的转置逻辑
x_enc = x_enc.permute(0, 2, 1)
output = model(x_enc=x_enc)
predictions = predictions.permute(0, 2, 1)
```

#### 2. 推理脚本 (`inference.py`)

```python
# 转置输入
x_enc_transposed = x_enc.permute(0, 2, 1)
output = model(x_enc=x_enc_transposed)

# 转置输出
predictions = predictions.permute(0, 2, 1)
```

## 🎯 数据流示意

### 修复前（错误）:
```
DataLoader → [B, 8, 628] → model() ❌ 维度错误
```

### 修复后（正确）:
```
DataLoader → [B, 8, 628] 
           → permute(0,2,1) → [B, 628, 8] 
           → model() ✓ 
           → forecast [B, 628, 8]
           → permute(0,2,1) → [B, 8, 628] ✓
           → Loss计算
```

## 📝 技术细节

### MOMENT模型内部处理流程

1. **Patching**: 将序列切分为patches
   ```python
   num_patches = (seq_len - patch_len) // stride + 1
   # 例如: (8 - 2) // 2 + 1 = 4 patches
   ```

2. **Patch Embedding**: 每个patch映射到d_model维度
   ```python
   # Input: [batch, n_channels, seq_len]
   # Output: [batch, n_channels, num_patches, d_model]
   ```

3. **Transformer Encoder**: 处理patch embeddings
   ```python
   # Output: [batch, n_channels, num_patches, d_model]
   ```

4. **Forecasting Head**: 
   ```python
   head_nf = d_model * num_patches  # 展平最后两维
   linear(head_nf → forecast_horizon)
   ```

### 为什么需要转置？

MOMENT是为**多变量时间序列**设计的：
- `n_channels` = 不同的传感器/特征/站点
- `seq_len` = 时间步长

模型对每个通道独立进行patching，然后在transformer中建模通道间关系。

**我们的场景：**
- 628个特征 = 157站点 × 4特征
- 每个特征是一个独立的"channel"
- 8小时是时间序列长度

## ✅ 验证修复

运行训练脚本，应该看到：

```
============================================================
Epoch 1/10
============================================================
Training: 100%|██████████████| 6/6 [00:05<00:00, 1.20it/s, loss=0.0234]
  Train Loss: 0.023400
Validating: 100%|███████████| 2/2 [00:01<00:00, 1.85it/s]
  Val Loss: 0.019800
  Val MAE: 0.112300
  ✓ Saved best model
```

**不再出现维度错误！**

## 🔧 相关代码位置

已修复的文件：
- ✅ `train_moment.py` - train_epoch() 和 validate() 函数
- ✅ `inference.py` - evaluate() 函数

## 💡 最佳实践

**使用MOMENT模型时的通用规则：**

1. **输入格式检查**:
   ```python
   assert x_enc.dim() == 3, "Input must be 3D tensor"
   assert x_enc.shape[1] == n_channels, f"Expected {n_channels} channels, got {x_enc.shape[1]}"
   ```

2. **转置模板**:
   ```python
   # Prepare input
   x_input = x_data.permute(0, 2, 1) if x_data.dim() == 3 else x_data
   
   # Get prediction
   output = model(x_enc=x_input)
   
   # Restore format
   predictions = output.forecast.permute(0, 2, 1) if output.forecast.dim() == 3 else output.forecast
   ```

3. **调试技巧**:
   ```python
   print(f"Input shape: {x_enc.shape}")  # Should be [B, C, L]
   print(f"Output shape: {predictions.shape}")  # Should be [B, C, H]
   ```

---

**修复日期**: 2026-04-22  
**影响范围**: 所有使用MOMENT forecasting任务的代码  
**状态**: ✅ 已修复并测试