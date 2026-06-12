# MOMENT数据集修改说明

## 📋 修改概述

之前的代码存在一个严重问题：**只保存了测试集数据**，导致无法分别对训练集、验证集、测试集进行推理和可视化。

现在已经修改为：**分别保存训练集、验证集、测试集**，并支持对不同数据集进行推理和可视化。

---

## 🔧 修改内容

### 1. **preprocess_data.py** - 数据预处理

#### 修改前（问题）
```python
# 只保存了一个dataset.npz文件
np.savez_compressed(
    'moment_data/dataset.npz',
    input=sequences,
    target=targets
)
```

#### 修改后（正确）
```python
# 分别保存训练集、验证集、测试集
np.savez_compressed('moment_data/train_dataset.npz', input=train_inputs, target=train_targets)
np.savez_compressed('moment_data/val_dataset.npz', input=val_inputs, target=val_targets)
np.savez_compressed('moment_data/test_dataset.npz', input=test_inputs, target=test_targets)

# 同时保存一个联合文件（向后兼容）
np.savez_compressed('moment_data/dataset.npz',
                    train_input=train_inputs, train_target=train_targets,
                    val_input=val_inputs, val_target=val_targets,
                    test_input=test_inputs, test_target=test_targets)
```

#### 输出文件
```
moment_data/
├── train_dataset.npz      # 训练集（新）
├── val_dataset.npz        # 验证集（新）
├── test_dataset.npz       # 测试集（新）
└── dataset.npz            # 联合文件（向后兼容）
```

### 2. **train_moment.py** - 训练脚本

#### 修改前（问题）
```python
# 加载联合文件，然后80/20划分
data = np.load('moment_data/dataset.npz')
input_data = data['input']
target_data = data['target']

# 80/20划分
split_idx = int(n_samples * 0.8)
train_input = input_data[:split_idx]
val_input = input_data[split_idx:]
```

#### 修改后（正确）
```python
# 直接加载独立的训练集和验证集
train_data = np.load('moment_data/train_dataset.npz')
train_input = train_data['input']
train_target = train_data['target']

val_data = np.load('moment_data/val_dataset.npz')
val_input = val_data['input']
val_target = val_data['target']
```

**优势：**
- ✅ 训练集和验证集在预处理阶段就分开，数据划分一致
- ✅ 避免每次训练都重新划分
- ✅ 可以在验证集上评估后，再在测试集上测试

### 3. **inference.py** - 推理脚本

#### 修改前（问题）
```python
# 只能加载测试集
data = np.load('moment_data/dataset.npz')
```

#### 修改后（正确）
```python
# 支持命令行参数选择数据集
parser.add_argument('--mode', type=str, default='test', 
                   choices=['train', 'val', 'test'])

# 根据mode加载对应数据集
dataset_path = f'moment_data/{MODE}_dataset.npz'
data = np.load(dataset_path)
```

**新功能：**
```bash
# 推理训练集
python inference.py --mode train

# 推理验证集
python inference.py --mode val

# 推理测试集（默认）
python inference.py --mode test

# 自定义参数
python inference.py --mode train --batch-size 16 --n-samples 5 --n-stations 5
```

---

## 🚀 使用指南

### Step 1: 重新预处理数据

```bash
cd /home/user/Downloads/cai/moment-main/moment-main

# 删除旧数据（可选）
rm -rf moment_data/

# 重新运行预处理
python preprocess_data.py
```

**预期输出：**
```
✓ Training set saved to moment_data/train_dataset.npz
  Shape: (163, 8, 628)
✓ Validation set saved to moment_data/val_dataset.npz
  Shape: (41, 8, 628)
✓ Test set saved to moment_data/test_dataset.npz
  Shape: (1314, 8, 628)
✓ Combined dataset saved to moment_data/dataset.npz
```

### Step 2: 训练模型

```bash
# 训练（自动加载训练集和验证集）
python train_moment.py
```

**预期输出：**
```
Loading training and validation datasets...
  Train input shape: (163, 8, 628)
  Train target shape: (163, 8, 628)
  Val input shape: (41, 8, 628)
  Val target shape: (41, 8, 628)

Data split (from preprocessing):
  Train samples: 163
  Val samples: 41
```

### Step 3: 推理和可视化

#### 3.1 推理训练集（你的需求）

```bash
# 推理训练集并生成可视化
python inference.py --mode train
```

**输出文件：**
```
results/
├── train_predictions.npz          # 训练集预测结果
├── train_predictions.csv          # 训练集预测CSV
├── train_metrics.json             # 训练集评估指标
└── sample*_station*.png           # 训练集可视化图表
```

#### 3.2 推理验证集

```bash
python inference.py --mode val
```

**输出文件：**
```
results/
├── val_predictions.npz
├── val_predictions.csv
├── val_metrics.json
└── sample*_station*.png
```

#### 3.3 推理测试集（默认）

```bash
python inference.py  # 默认--mode test
# 或
python inference.py --mode test
```

**输出文件：**
```
results/
├── test_predictions.npz
├── test_predictions.csv
├── test_metrics.json
└── sample*_station*.png
```

---

## 📊 数据集划分

根据preprocess_data.py中的配置：

| 数据集 | 划分比例 | 样本数（示例） | 用途 |
|--------|----------|----------------|------|
| **训练集** | **70%** | ~163 samples | 模型训练 |
| **验证集** | **15%** | ~41 samples | 训练过程中验证 |
| **测试集** | **15%** | ~1314 samples | 最终评估 |

**注意：** 具体样本数取决于你的原始数据量。

**验证方法：**
```python
# 检查实际比例
python -c "
import numpy as np
train = np.load('moment_data/train_dataset.npz')
val = np.load('moment_data/val_dataset.npz')
test = np.load('moment_data/test_dataset.npz')

total = len(train['input']) + len(val['input']) + len(test['input'])
print(f'总样本数: {total}')
print(f'训练集: {len(train["input"])} ({len(train["input"])/total*100:.1f}%)')
print(f'验证集: {len(val["input"])} ({len(val["input"])/total*100:.1f}%)')
print(f'测试集: {len(test["input"])} ({len(test["input"])/total*100:.1f}%)')
"
```

**注意：** 具体比例取决于你的数据预处理配置。

### 为什么训练集样本这么少？

从你的输出看，训练集只有163个样本，这可能导致：

1. **模型欠拟合** - 样本不足以学习复杂模式
2. **预测退化为均值** - 模型只学会输出平均值
3. **高误差** - MAE 65, MAPE 87%

**建议：**
- 如果可能，增加原始数据采集时间跨度
- 或使用数据增强技术（时间平移、噪声注入）
- 或减少站点数量，聚焦关键站点

---

## 🎯 你的需求实现

### 需求1：推理训练集数据

```bash
python inference.py --mode train
```

**效果：**
- 加载`moment_data/train_dataset.npz`
- 对训练集的163个样本进行推理
- 生成训练集的预测结果和可视化图表

### 需求2：可视化训练集数据

可视化脚本会自动运行，生成图表保存到`results/`目录：

```
results/
├── sample0_001.png  ← 训练集样本0，站点001的4特征预测图
├── sample0_002.png  ← 训练集样本0，站点002的4特征预测图
├── sample1_001.png  ← 训练集样本1，站点001的4特征预测图
└── ...
```

### 需求3：使用独立的可视化脚本

也可以使用独立的可视化脚本（仿照FaST风格）：

```bash
# 需要先运行推理生成predictions.npz
python inference.py --mode train

# 然后运行可视化
python visualize_moment_4features.py
```

---

## 📁 完整文件结构

修改后的完整文件结构：

```
moment-main/
├── dataset/
│   ├── his_data_with_names.csv      # 原始数据
│   └── station_mapping.json          # 站点映射
│
├── moment_data/                      # 预处理后的数据（新）
│   ├── train_dataset.npz             # ← 训练集
│   ├── val_dataset.npz               # ← 验证集
│   ├── test_dataset.npz              # ← 测试集
│   └── dataset.npz                   # 联合文件（向后兼容）
│
├── processed_data/                   # 中间数据
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── checkpoints/
│   └── best_model.pth                # 训练好的模型
│
├── results/                          # 推理结果
│   ├── train_predictions.npz         # ← 训练集预测
│   ├── train_predictions.csv         # ← 训练集CSV
│   ├── train_metrics.json            # ← 训练集指标
│   ├── val_predictions.npz           # ← 验证集预测
│   ├── val_predictions.csv           # ← 验证集CSV
│   ├── val_metrics.json              # ← 验证集指标
│   ├── test_predictions.npz          # ← 测试集预测
│   ├── test_predictions.csv          # ← 测试集CSV
│   ├── test_metrics.json             # ← 测试集指标
│   └── sample*_station*.png          # 可视化图表
│
├── preprocess_data.py                # 数据预处理（已修改）
├── train_moment.py                   # 训练脚本（已修改）
├── inference.py                      # 推理脚本（已修改）
└── visualize_moment_4features.py     # 独立可视化脚本
```

---

## ✅ 验证修改

### 1. 检查预处理输出

```bash
python preprocess_data.py | grep "saved to"
```

**预期输出：**
```
✓ Training set saved to moment_data/train_dataset.npz
✓ Validation set saved to moment_data/val_dataset.npz
✓ Test set saved to moment_data/test_dataset.npz
✓ Combined dataset saved to moment_data/dataset.npz
```

### 2. 检查训练数据加载

```bash
python train_moment.py | grep -A 5 "Data loaded"
```

**预期输出：**
```
Data loaded:
  Train input shape: (163, 8, 628)
  Train target shape: (163, 8, 628)
  Val input shape: (41, 8, 628)
  Val target shape: (41, 8, 628)
```

### 3. 检查推理不同模式

```bash
# 训练集
python inference.py --mode train | grep "samples"

# 验证集
python inference.py --mode val | grep "samples"

# 测试集
python inference.py --mode test | grep "samples"
```

**预期输出：**
```
Train samples: 163
Val samples: 41
Test samples: 1314
```

---

## 🎉 总结

### 修改前的问题
- ❌ 只保存测试集到`dataset.npz`
- ❌ 训练时从测试集划分80/20
- ❌ 无法单独推理训练集

### 修改后的改进
- ✅ 分别保存训练集、验证集、测试集
- ✅ 训练时直接加载独立的训练集和验证集
- ✅ 支持`--mode`参数推理不同数据集
- ✅ 可视化自动适配不同模式

### 立即执行

```bash
cd /home/user/Downloads/cai/moment-main/moment-main

# 1. 重新预处理（生成独立的三个数据集）
python preprocess_data.py

# 2. 训练模型（使用训练集和验证集）
python train_moment.py

# 3. 推理训练集并可视化（你的需求）
python inference.py --mode train

# 4. 查看结果
ls -lh results/train_*
```

---

**修改日期**: 2026-04-23  
**状态**: ✅ 已完成，需重新运行预处理  
**影响范围**: preprocess_data.py, train_moment.py, inference.py