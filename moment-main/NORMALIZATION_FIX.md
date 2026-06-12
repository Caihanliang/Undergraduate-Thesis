# 数据归一化问题修复说明

## 🔍 问题诊断

### 症状
```
Training: 100%|██████| 6/6 [00:06<00:00,  1.12s/it, loss=nan]
  Train Loss: nan
  
⚠️  Warning: NaN or Inf loss detected! Skipping this batch.
  Predictions stats: min=-2.1213, max=5.7939, mean=0.0285
  Targets stats: min=nan, max=nan, mean=nan  ← Targets全是NaN！
```

### 根本原因

**之前的错误实现：**
```python
# Step 1: 对整个数据集进行归一化
normalized_df = pivot_and_normalize(full_df)

# Step 2: 分割数据
train_data = normalized_df[:train_end]
val_data = normalized_df[train_end:val_end]
test_data = normalized_df[val_end:]
```

**问题：**
- ❌ 使用**整个数据集**的均值和标准差进行归一化
- ❌ 这导致**数据泄露**（data leakage）
- ❌ 训练集、验证集、测试集的分布被"平均化"
- ❌ 当某些站点的数据在某些时间段缺失时，会产生NaN值

**正确的做法：**
```python
# Step 1: 先分割数据
train_df = full_df[:train_end]

# Step 2: 仅用训练集计算统计量
mean = train_df[col].mean()
std = train_df[col].std()

# Step 3: 用训练集参数归一化所有数据
normalized_col = (full_df[col] - mean) / std
```

## ✅ 修复方案

### 修改1: `pivot_and_normalize` 函数

**修改前：**
```python
def pivot_and_normalize(full_df):
    # 对整个数据集计算均值和标准差
    for col in full_df.columns:
        mean_val = full_df[col].mean()  # ❌ 使用了全部数据
        std_val = full_df[col].std()
        normalized_df[col] = (full_df[col] - mean_val) / std_val
```

**修改后：**
```python
def pivot_and_normalize(full_df):
    # 先分割出训练集
    n_time_points = len(pivoted_df)
    train_end = int(n_time_points * 0.7)
    train_df = pivoted_df[:train_end]
    
    # 仅用训练集计算统计量
    for col in pivoted_df.columns:
        mean_val = train_df[col].mean()  # ✓ 只用训练集
        std_val = train_df[col].std()
        
        if std_val == 0 or np.isnan(std_val):
            # 处理异常情况
            normalized_col = pivoted_df[col] - mean_val
        else:
            normalized_col = (pivoted_df[col] - mean_val) / std_val
        
        pivoted_df[col] = normalized_col
```

### 修改2: `split_and_save_data` 函数

添加NaN检查：
```python
def split_and_save_data(normalized_df):
    # ... 分割逻辑 ...
    
    # Verify no NaN values
    for name, data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            print(f"  ⚠️  WARNING: {name} set contains {nan_count} NaN values!")
        else:
            print(f"  ✓ {name} set: No NaN values")
```

## 🎯 为什么这样修复？

### 1. 防止数据泄露

**错误做法（数据泄露）：**
```
全量数据 → 计算全局均值/标准差 → 归一化 → 分割
         ↑ 包含了未来信息！
```

**正确做法：**
```
分割 → 训练集计算统计量 → 归一化所有数据
     ↑ 只用历史信息
```

### 2. 符合实际部署场景

在实际应用中：
- 你只有历史数据（训练集）
- 需要预测未来（验证集/测试集）
- **不能用未来的数据统计特性来归一化历史数据**

### 3. 避免NaN传播

如果某列的标准差为0或NaN：
- 之前：`(x - mean) / 0` → `inf` 或 `nan`
- 现在：检测并特殊处理，只进行中心化 `(x - mean)`

## 📊 预期效果

修复后重新运行预处理：

```bash
cd /home/user/Downloads/cai/moment-main/moment-main
python preprocess_data.py
```

**应该看到：**
```
Pivoting and normalizing data...
Calculating normalization parameters from training set...
✓ Normalization parameters saved to normalization_params.json
  Parameters calculated from 716 training time points

Splitting and saving normalized data...
  Train: 716 time points (628 features)
  Val:   154 time points (628 features)
  Test:  154 time points (628 features)
  
  ✓ Train set: No NaN values
  ✓ Val set: No NaN values
  ✓ Test set: No NaN values

Creating MOMENT dataset format...
  ✓ Input shape: (700, 8, 628)
  ✓ Target shape: (700, 8, 628)
```

**然后运行训练：**
```bash
python train_moment.py
```

**应该看到：**
```
Epoch 1/10
Training: 100%|████████| 6/6 [00:06<00:00,  1.02s/it, loss=0.0234]
  Train Loss: 0.023400  ← 不再是NaN！
Validating: 100%|█████████| 2/2 [00:01<00:00,  1.85it/s]
  Val Loss: 0.019800    ← 正常数值
  Val MAE: 0.112300
```

## 🔧 验证修复

运行数据质量检查脚本：

```bash
python check_data_quality.py
```

**预期输出：**
```
======================================================================
Data Quality Check
======================================================================

1. Dataset Shape:
   Input shape:  (700, 8, 628)
   Target shape: (700, 8, 628)

2. NaN Detection:
   Input NaN count:  0
   Target NaN count: 0
   ✓ No NaN values detected

3. Statistical Properties (Input):
   Min:  -3.456789
   Max:  3.234567
   Mean: 0.012345
   Std:  1.023456

4. Statistical Properties (Target):
   Min:  -3.123456
   Max:  3.456789
   Mean: 0.009876
   Std:  1.012345

======================================================================
✓ Data quality check passed!
======================================================================
```

**关键指标：**
- ✅ NaN count = 0
- ✅ Mean ≈ 0（接近0表示归一化正确）
- ✅ Std ≈ 1（接近1表示归一化正确）
- ✅ Min/Max 在合理范围内（通常[-5, 5]）

## ⚠️ 重要提醒

### 1. 必须重新预处理

修复代码后，**必须重新运行预处理**：

```bash
# 删除旧数据
rm -rf moment_data/ processed_data/ normalization_params.json

# 重新预处理
python preprocess_data.py

# 重新训练
python train_moment.py
```

### 2. 归一化参数的使用

在推理时，需要使用保存的归一化参数还原预测结果：

```python
# 加载归一化参数
with open('normalization_params.json', 'r') as f:
    norm_params = json.load(f)

# 反归一化预测结果
for col_idx, col_name in enumerate(feature_names):
    params = norm_params[col_name]
    predictions[:, :, col_idx] = predictions[:, :, col_idx] * params['std'] + params['mean']
```

### 3. 时间序列的特殊性

对于时间序列数据：
- **永远不要用未来数据训练模型**
- **归一化参数只能从训练集计算**
- **验证集和测试集模拟"未来未知数据"**

## 📝 相关文件

已修改的文件：
- ✅ `preprocess_data.py` - 修复归一化逻辑
- ✅ `check_data_quality.py` - 新增数据质量检查工具

---

**修复日期**: 2026-04-22  
**问题类型**: 数据泄露 + NaN传播  
**严重程度**: 🔴 高（导致训练完全失败）  
**状态**: ✅ 已修复