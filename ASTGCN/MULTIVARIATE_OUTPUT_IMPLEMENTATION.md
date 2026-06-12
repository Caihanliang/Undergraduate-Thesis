# ASTGCN 多变量输出实现方案（Channel Independence）

## 📋 方案概述

本方案借鉴 **PatchTST** 的 **Channel Independence (CI)** 思想，修改 ASTGCN 模型以支持真正的多变量输出（同时预测4个特征）。

### 核心改进
- ✅ **输入**: (B, N=98, F=4, T=8) - 4个特征通道
- ✅ **输出**: (B, N=98, F=4, T=8) - 4个特征分别预测
- ✅ **架构**: 保持时空注意力机制，仅修改输出层

---

## 🔧 修改内容

### 1. 模型架构修改 (`model/ASTGCN_r.py`)

#### 修改 `ASTGCN_submodule` 类

**关键变化**:
```python
# 原始设计（单变量输出）
self.final_conv = nn.Conv2d(T_in, T_out, kernel_size=(1, nb_time_filter))
# 输出: (B, N, T_out)

# 新设计（多变量输出）
self.final_conv = nn.Conv2d(T_in, F_out * T_out, kernel_size=(1, nb_time_filter))
# 输出: (B, N, F_out * T_out) -> reshape -> (B, N, F_out, T_out)
```

**新增参数**:
- `num_features_out`: 输出特征通道数（默认等于 `in_channels`）

**Forward 逻辑**:
```python
def forward(self, x):
    # x: (B, N, F_in, T_in)
    for block in self.BlockList:
        x = block(x)
    
    # 卷积输出: (B, F_out*T_out, N, 1)
    output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1]
    
    # 重塑为: (B, N, F_out, T_out)
    output = output.view(B, N, F_out, T_out)
    
    return output
```

### 2. 数据预处理修改 (`preprocess_highway_data.py`)

**关键变化**:
```python
# 原始方案（平均值聚合）
y_final = all_targets.mean(axis=3).transpose(0, 2, 1)  # (B, N, T_out)

# 新方案（保留所有特征）
y_final = all_targets.transpose(0, 2, 3, 1)  # (B, N, F, T_out)
```

**数据格式**:
- `x`: (B, N, F=4, T_in=8) - 归一化后的输入
- `y`: (B, N, F=4, T_out=8) - 未归一化的真实值（4个特征）

### 3. 数据加载修改 (`lib/utils.py`)

#### 修改 `generate_data` 函数
- 更新注释说明现在支持多变量目标
- `y` 的形状从 `(B, N, T_out)` 变为 `(B, N, F, T_out)`

#### 重写 `predict_and_save_results_mstgcn` 函数
**新增功能**:
1. **按特征分类评估**: 分别计算4个特征的 MAE、RMSE、MAPE
2. **按时间步评估**: 对每个预测时间点单独计算指标
3. **全局汇总**: 计算所有特征和时间步的平均指标

**输出示例**:
```
按特征分类评估结果
============================================================

当前epoch: 79, 预测第 1 个时间点
  小客车上行 - MAE: 15.23, RMSE: 22.45, MAPE: 8.5%
  小客车下行 - MAE: 14.87, RMSE: 21.92, MAPE: 8.2%
  非小客车上行 - MAE: 18.34, RMSE: 26.78, MAPE: 10.1%
  非小客车下行 - MAE: 17.92, RMSE: 25.43, MAPE: 9.8%

...

全局评估结果（所有特征和时间步）
============================================================
全局 MAE: 16.59
全局 RMSE: 24.15
全局 MAPE: 9.2%
```

### 4. 训练脚本修改 (`train_ASTGCN_r.py`)

**关键变化**:
```python
# 创建模型时指定输出特征数
net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, 
                nb_time_filter, time_strides, adj_mx,
                num_for_predict, len_input, num_of_vertices, 
                num_features_out=in_channels)  # 新增参数
```

---

## 🚀 使用流程

### 步骤1: 重新预处理数据

```bash
cd /home/user/Downloads/cai/ASTGCN

# 清理旧数据
rm -f dataset/train.npz dataset/val.npz dataset/test.npz

# 重新运行预处理（生成多变量目标）
python preprocess_highway_data.py \
    --dataset_dir ./dataset \
    --input_len 8 \
    --output_len 8 \
    --adj_method distance
```

### 步骤2: 验证数据格式

```bash
python verify_data.py
```

应该看到：
```
✅ x: shape=(865, 98, 4, 8), dtype=float32
✅ y: shape=(865, 98, 4, 8), dtype=float32  # 注意：现在是4维
```

### 步骤3: 删除旧模型并重新训练

```bash
# 删除旧的实验目录（因为模型架构已改变）
rm -rf experiments/highway_traffic/astgcn_r_h1d0w0_channel4_1.000000e-03

# 开始训练
python train_ASTGCN_r.py --config configurations/highway_traffic.conf
```

---

## 📊 预期效果

### 优势
1. **细粒度预测**: 可以分别查看每个车型的预测精度
2. **业务价值**: 不同车型可能有不同的预测难度，分开评估更有意义
3. **灵活性**: 未来可以轻松扩展到更多特征

### 对比

| 维度 | 方案A（平均值） | 方案C（多变量） |
|------|----------------|----------------|
| 输出维度 | (B, N, T_out) | (B, N, F, T_out) |
| 特征独立性 | ❌ 混合 | ✅ 独立 |
| 评估粒度 | 综合指标 | 分特征指标 |
| 模型参数量 | 较少 | 略多（输出层） |
| 训练时间 | 快 | 相近 |

---

## ⚠️ 注意事项

### 1. 损失函数
PyTorch 的 `MSELoss` 会自动处理多维张量，无需修改：
```python
# outputs: (B, N, F, T_out)
# labels: (B, N, F, T_out)
loss = criterion(outputs, labels)  # 自动计算所有元素的MSE
```

### 2. 内存占用
- 输出张量从 `(B, N, T_out)` 增加到 `(B, N, F, T_out)`
- 对于当前配置（B=32, N=98, F=4, T=8），增加约 4倍
- 仍在可接受范围内（约 1MB/batch）

### 3. 反归一化
评估时需要使用对应的特征通道进行反归一化：
```python
# _mean/_std: (1, 1, 1, F)
# prediction: (B, N, F, T_out)
# 广播机制自动处理
```

---

## 🔬 技术细节

### Channel Independence vs Channel Dependence

**Channel Independence (CI)** - PatchTST 采用的策略:
- 每个特征通道独立处理
- 共享模型权重
- 优点：降低复杂度，提高泛化能力

**Channel Dependence (CD)** - 传统方法:
- 所有特征联合建模
- 捕捉特征间相关性
- 缺点：参数多，易过拟合

我们的实现采用了 **CI 思想**，但通过图卷积保留了空间依赖性。

### 输出层设计原理

```
输入: (B, N, F_in=4, T_in=8)
  ↓ [ASTGCN Blocks]
中间表示: (B, N, F_mid=64, T_mid=8)
  ↓ [Final Conv: kernel=(1, 64)]
输出: (B, F_out*T_out=32, N, 1)
  ↓ [Reshape]
最终输出: (B, N, F_out=4, T_out=8)
```

关键公式:
```
output_channels = num_features_out * num_for_predict
                = 4 * 8 = 32
```

---

## 📈 后续优化方向

1. **加权损失函数**: 根据不同特征的重要性设置权重
2. **特征间注意力**: 在输出层前添加特征维度的注意力机制
3. **多任务学习**: 将4个特征视为相关任务，共享部分参数
4. **不确定性估计**: 为每个特征输出置信区间

---

**最后更新**: 2024年
**参考**: PatchTST (ICLR 2023) - "A Time Series is Worth 64 Words"
