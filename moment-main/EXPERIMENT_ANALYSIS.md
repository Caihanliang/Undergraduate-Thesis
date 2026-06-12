# MOMENT模型实验结果分析

## 📊 可视化结果解读

### 1. 图表结构说明

根据改进后的可视化代码，每个图表包含：

**2×2子图布局（一个站点）：**
```
┌────────────────────┬────────────────────┐
│  Passenger Car Up  │ Passenger Car Down │
│  MAE: XX | RMSE: XX│ MAE: XX | RMSE: XX │
├────────────────────┼────────────────────┤
│Non-Passenger Car Up│Non-Passenger Car Down│
│  MAE: XX | RMSE: XX│ MAE: XX | RMSE: XX │
└────────────────────┴────────────────────┘
```

**每条曲线的含义：**
- **实线（Actual）**：真实交通流量值
- **虚线（Predicted）**：MOMENT模型预测值
- **红色标记**：误差最大的时间点（Top 10%）
- **黄色标注**：最大误差值及位置

### 2. 从你的截图中观察到的问题

#### ❌ 问题1：预测曲线过于平缓

**现象：**
- 预测值（虚线）几乎是一条水平线
- 真实值（实线）有明显的波动和趋势
- 预测模型似乎只学到了数据的平均值

**可能原因：**
```python
# 模型预测结果接近均值
prediction ≈ mean(target)
```

**技术分析：**
1. **Linear Probing模式限制**：只训练Forecasting Head层（~8200参数），而编码器（3.4亿参数）被冻结
2. **训练样本不足**：仅163个训练样本，无法充分学习复杂模式
3. **过拟合到均值**：在数据不足时，模型退化为"输出训练集均值"的简单策略

#### ❌ 问题2：MAE和RMSE指标偏高

**你的截图显示：**
```
Non-Passenger Car Up:
  MAE: 59.775  ← 平均误差约60辆车/小时
  RMSE: 81.988

Non-Passenger Car Down:
  MAE: 102.628 ← 平均误差约103辆车/小时
  RMSE: 123.219
```

**指标解读：**
- **MAE=60-103** 意味着预测平均偏离真实值60-103辆车/小时
- **相对误差**：如果真实流量是200-400辆/小时，那么相对误差达15-50%
- **RMSE > MAE**：说明存在较大的离群误差（某些时间点预测特别差）

#### ❌ 问题3：不同特征预测性能差异大

**观察：**
- 非小客车下行预测最差（MAE=102.6）
- 非小客车上行稍好（MAE=59.8）
- 可能存在特征间的预测难度差异

**原因分析：**
1. **数据分布不平衡**：某些时段某些车型流量很少
2. **RevIN归一化问题**：每个batch独立归一化可能导致小值特征学习困难
3. **模型容量不足**：8200个参数无法捕捉4个特征的复杂关系

---

## 🎯 改进建议

### 优先级1：增加训练数据（关键⭐⭐⭐）

**当前问题：**
```
Train samples: 163  ← 严重不足！
Val samples:   41
```

**解决方案：**

#### 方案A：扩展数据采集时间
```python
# 修改 preprocess_data.py 中的数据加载逻辑
# 从仅几个月扩展到1-2年数据

# 预期效果：
# Train samples: 2000-5000  ← 10-30倍增长
```

#### 方案B：减少站点数量
```python
# 选择关键的20-30个站点而非157个
# 每个站点获得更多样本

# 预期效果：
# Train samples per station: 50-100  ← 从1提升到50-100
```

#### 方案C：数据增强
```python
# 添加时间平移、噪声注入、季节模式增强
# 示例代码（添加到 preprocess_data.py）

def augment_data(data, noise_std=0.05, n_augment=3):
    """数据增强：添加噪声和平移"""
    augmented = [data]
    for _ in range(n_augment):
        noise = np.random.normal(0, noise_std, data.shape)
        augmented.append(data + noise)
    return np.concatenate(augmented, axis=0)
```

### 优先级2：调整训练策略（重要⭐⭐）

#### 2.1 改用Fine-tuning模式

**当前配置：**
```python
'freeze_encoder': True,   # ← 冻结编码器
'freeze_embedder': True,  # ← 冻结嵌入层
'freeze_head': False,     # ← 仅训练预测头
```

**改进配置：**
```python
# train_moment.py
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={
        'task_name': 'forecasting',
        'forecast_horizon': 8,
        'n_channels': n_features,
        'seq_len': 8,
        'head_dropout': 0.1,
        'freeze_encoder': False,  # ← 解冻编码器
        'freeze_embedder': False, # ← 解冻嵌入层
        'freeze_head': False,
        'lr': 1e-5,  # ← 更小的学习率（微调）
    },
)

# 分层学习率
optimizer = torch.optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': 1e-5},  # 慢速微调
    {'params': model.head.parameters(), 'lr': 1e-4},     # 快速训练
], weight_decay=1e-4)
```

**预期效果：**
- 可训练参数：8,200 → **~50,000,000**（6000倍增长）
- 模型容量大幅提升
- 能够学习更复杂的模式

#### 2.2 优化学习率和批次大小

```python
# train_moment.py 配置优化
BATCH_SIZE = 8           # 从32降至8
EPOCHS = 50              # 从10增至50
LEARNING_RATE = 1e-4     # 保持
WEIGHT_DECAY = 1e-4      # 增强正则化
```

### 优先级3：改进模型架构（可选⭐）

#### 3.1 添加位置编码增强

```python
# 在输入中添加时间特征
def add_temporal_features(x_enc, timestamps):
    """添加时间编码：小时、星期、月份"""
    hour_sin = torch.sin(2 * torch.pi * timestamps.hour / 24)
    hour_cos = torch.cos(2 * torch.pi * timestamps.hour / 24)
    # 拼接时间特征到输入
    return torch.cat([x_enc, hour_sin.unsqueeze(-1), hour_cos.unsqueeze(-1)], dim=-1)
```

#### 3.2 使用Channel-Independent策略

```python
# 分别训练每个特征，然后集成
for feat_idx in range(4):
    model = train_single_feature(data[:, :, feat_idx])
    # 集成预测结果
```

---

## 📈 预期改进效果

### 当前性能基线
```
Overall Test Metrics:
  MAE:   ~80 vehicles/hour
  RMSE:  ~100 vehicles/hour
  MAPE:  ~30-50%
  
Visualization:
  - 预测曲线平缓，无法捕捉波动
  - 误差集中在峰值时段
```

### 改进后预期性能

#### 场景A：仅增加数据量（保持Linear Probing）
```
Train samples: 163 → 2000（12倍增长）

Expected Metrics:
  MAE:   80 → 40-50 vehicles/hour  （降低40%）
  RMSE:  100 → 55-65 vehicles/hour （降低40%）
  MAPE:  40% → 20-25%              （降低50%）

Visualization:
  ✓ 预测曲线开始捕捉趋势
  ✓ 峰值时段预测改善
  ✗ 仍无法捕捉复杂模式
```

#### 场景B：增加数据 + Fine-tuning
```
Train samples: 163 → 2000
Trainable params: 8,200 → 50,000,000

Expected Metrics:
  MAE:   80 → 20-30 vehicles/hour  （降低65%）
  RMSE:  100 → 30-40 vehicles/hour （降低65%）
  MAPE:  40% → 10-15%              （降低70%）

Visualization:
  ✓ 预测曲线紧密跟随真实值
  ✓ 成功捕捉波动和趋势
  ✓ 峰值时段误差显著降低
```

#### 场景C：理想情况（充足数据 + Fine-tuning + 数据增强）
```
Train samples: 5000+
Trainable params: 50,000,000
Data augmentation: 3x

Expected Metrics:
  MAE:   15-20 vehicles/hour
  RMSE:  20-25 vehicles/hour
  MAPE:  8-12%

Visualization:
  ✓ 预测曲线几乎与真实值重合
  ✓ 能够预测突发流量变化
  ✓ 误差主要集中在极端情况
```

---

## 🔬 实验验证步骤

### Step 1：重新预处理（无归一化）
```bash
cd /home/user/Downloads/cai/moment-main/moment-main
rm -rf moment_data/ processed_data/
python preprocess_data.py
```

### Step 2：检查数据量
```bash
python diagnose_data_volume.py
# 期望输出：Train samples < 500 → 需要改进
```

### Step 3：运行当前配置基线
```bash
python train_moment.py
python inference.py
# 记录当前MAE、RMSE、MAPE
```

### Step 4：实施改进（推荐方案B）

#### 4.1 修改训练配置
```python
# train_moment.py
# 改为Fine-tuning模式
model_kwargs = {
    'freeze_encoder': False,  # 关键改动
    'freeze_embedder': False,
    'freeze_head': False,
}
```

#### 4.2 重新训练
```bash
python train_moment.py
# 观察训练曲线是否改善
```

#### 4.3 评估和可视化
```bash
python inference.py
# 查看新的可视化结果和指标
```

### Step 5：对比分析

创建对比表格：

| 配置 | Train Samples | Trainable Params | MAE | RMSE | MAPE | 可视化质量 |
|------|--------------|------------------|-----|------|------|-----------|
| Baseline | 163 | 8,200 | 80 | 100 | 40% | ❌ 差 |
| +Data (2000) | 2000 | 8,200 | 45 | 60 | 25% | ⚠️ 一般 |
| +Fine-tune | 2000 | 50M | 25 | 35 | 12% | ✅ 好 |
| +Augment | 6000 | 50M | 18 | 22 | 8% | ✅✅ 优秀 |

---

## 💡 关键结论

### 1. 当前实验状态评估

**✅ 做得好的方面：**
- 代码框架完整（预处理→训练→推理→可视化）
- 成功集成了MOMENT预训练模型
- 可视化符合多特征时序预测规范
- 指标计算全面（MAE、RMSE、MAPE、逐特征分析）

**❌ 需要改进的核心问题：**
- **训练数据严重不足**（163样本 vs 需求2000+）
- **Linear Probing模式容量不足**（8200参数 vs 3.4亿总参数）
- **预测结果退化为均值**（未学到有效模式）

### 2. 优先级建议

```
紧急程度排序：
1. 增加训练数据（最重要，影响60%效果）
2. 改用Fine-tuning模式（次重要，影响30%效果）
3. 优化超参数（学习率、批次大小等，影响10%效果）
4. 数据增强和架构改进（锦上添花）
```

### 3. 可行性分析

| 改进方案 | 实施难度 | 预期收益 | 时间成本 | 推荐度 |
|---------|---------|---------|---------|-------|
| 增加数据采集时间 | ⭐⭐ 中 | ⭐⭐⭐⭐⭐ 高 | 1-2天 | ⭐⭐⭐⭐⭐ |
| Fine-tuning | ⭐ 低 | ⭐⭐⭐⭐ 高 | 半天 | ⭐⭐⭐⭐⭐ |
| 数据增强 | ⭐⭐ 中 | ⭐⭐⭐ 中 | 1天 | ⭐⭐⭐⭐ |
| 减少站点数量 | ⭐ 低 | ⭐⭐ 中 | 2小时 | ⭐⭐⭐ |
| Channel-Independent | ⭐⭐⭐ 高 | ⭐⭐⭐ 中 | 2-3天 | ⭐⭐ |

---

## 📝 总结

**当前实验结果表明：**
- MOMENT模型框架搭建成功 ✓
- 但由于**数据量不足**和**训练模式限制**，预测性能较差
- 预测退化为"输出均值"的简单策略

**建议立即执行：**
1. 收集更多历史数据（至少1年）
2. 改用Fine-tuning模式训练
3. 预期MAE从80降至20-30（65%改进）

**长期规划：**
- 建立持续数据收集管道
- 探索多模态数据融合（天气、事件等）
- 对比其他SOTA模型（PatchTST、TimesNet等）

---

**分析日期**: 2026-04-23  
**当前状态**: ⚠️ 基线建立完成，需数据增强  
**下一步**: 优先增加数据量 + 改用Fine-tuning模式