# 训练稳定性问题修复指南

## 🔍 问题诊断

### 问题1: Loss为NaN
```
Training: 100%|██████| 6/6 [00:06<00:00,  1.12s/it, loss=nan]
  Train Loss: nan
```

**原因分析：**
1. **学习率过高**: 初始学习率1e-3对于微调任务可能过大
2. **梯度爆炸**: 没有梯度裁剪机制
3. **数据异常**: 可能存在未归一化的异常值
4. **缺少正则化**: 没有weight decay

### 问题2: JSON序列化错误
```
TypeError: Object of type float32 is not JSON serializable
```

**原因：** numpy的float32类型不能直接被json.dump序列化

### 问题3: 中文字体警告
```
UserWarning: Glyph 23567 (\N{CJK UNIFIED IDEOGRAPH-5C0F}) missing from font(s) DejaVu Sans.
```

**原因：** matplotlib默认字体不支持中文

## ✅ 完整修复方案

### 修复1: 降低学习率并添加调度器

**修改前：**
```python
LEARNING_RATE = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

**修改后：**
```python
LEARNING_RATE = 1e-4  # 降低10倍
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=1e-5  # 添加L2正则化
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=EPOCHS, 
    eta_min=1e-6  # 最小学习率
)

# 每个epoch后更新
scheduler.step()
```

**优势：**
- ✅ 更稳定的收敛
- ✅ 防止过拟合（weight decay）
- ✅ 自动调整学习率（Cosine退火）

### 修复2: 添加梯度裁剪

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    # ... forward pass ...
    
    # Check for NaN loss
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"⚠️  Warning: NaN or Inf loss detected! Skipping this batch.")
        continue
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

**作用：**
- ✅ 限制梯度范数不超过1.0
- ✅ 防止梯度爆炸
- ✅ 跳过异常batch继续训练

### 修复3: JSON序列化修复

```python
# Save training history - convert numpy types to Python native types
training_history_serializable = {
    'train_loss': [float(x) for x in training_history['train_loss']],
    'val_loss': [float(x) for x in training_history['val_loss']],
    'val_mae': [float(x) for x in training_history['val_mae']]
}

with open('results/training_history.json', 'w') as f:
    json.dump(training_history_serializable, f, indent=2)
```

### 修复4: 字体警告修复

```python
def plot_feature_predictions(...):
    # Set font to support Chinese characters
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # Use English feature names to avoid font issues
    feature_names = ['Passenger Car Up', 'Passenger Car Down', 
                     'Non-Passenger Car Up', 'Non-Passenger Car Down']
```

## 🎯 预期效果

修复后运行训练，应该看到：

```
============================================================
Epoch 1/10
============================================================
Training: 100%|████████| 6/6 [00:06<00:00,  1.02s/it, loss=0.0234]
  Train Loss: 0.023400
Validating: 100%|█████████| 2/2 [00:01<00:00,  1.85it/s]
  Val Loss: 0.019800
  Val MAE: 0.112300
  Current Learning Rate: 0.000098  ← 学习率自动调整
  ✓ Saved best model (val_loss: 0.019800, val_mae: 0.112300)

...

Epoch 10/10
Training: 100%|████████| 6/6 [00:05<00:00,  1.01it/s, loss=0.0156]
  Train Loss: 0.015600
  Val Loss: 0.014200
  Val MAE: 0.098700
  Current Learning Rate: 0.000010  ← 逐渐减小

Training completed!
Best validation loss: 0.014200
Results saved to results/
```

**关键改进：**
- ✅ Loss不再为NaN
- ✅ 训练稳定收敛
- ✅ 学习率动态调整
- ✅ 无JSON错误
- ✅ 无字体警告

## 📊 超参数调优建议

如果仍然出现NaN或不收敛，尝试以下调整：

### 方案A: 进一步降低学习率
```python
LEARNING_RATE = 5e-5  # 更保守的学习率
```

### 方案B: 增强梯度裁剪
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 更严格
```

### 方案C: 增加Warmup
```python
from torch.optim.lr_scheduler import SequentialLR, LinearLR

warmup_scheduler = LinearLR(
    optimizer, 
    start_factor=0.1, 
    end_factor=1.0, 
    total_iters=EPOCHS // 5  # 前20% epoch warmup
)

main_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * 4 // 5)

scheduler = SequentialLR(
    optimizer, 
    schedulers=[warmup_scheduler, main_scheduler],
    milestones=[EPOCHS // 5]
)
```

### 方案D: 检查数据归一化
```python
# 在preprocess_data.py中确认
print(f"Input data range: [{input_data.min():.4f}, {input_data.max():.4f}]")
print(f"Input data mean: {input_data.mean():.4f}, std: {input_data.std():.4f}")

# 理想情况：mean≈0, std≈1, range≈[-3, 3]
```

## 🔧 调试技巧

### 监控训练过程

在train_epoch中添加详细日志：

```python
if n_batches % 2 == 0:  # 每2个batch打印一次
    print(f"\nBatch {n_batches}:")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Predictions - min: {predictions.min():.4f}, max: {predictions.max():.4f}")
    print(f"  Targets - min: {y.min():.4f}, max: {y.max():.4f}")
    
    # 检查梯度
    total_grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item()
    print(f"  Total gradient norm: {total_grad_norm:.6f}")
```

### 可视化Loss曲线

训练完成后查看：

```bash
python -c "
import json
import matplotlib.pyplot as plt

with open('results/training_history.json') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(history['val_mae'])
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Validation MAE')

plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=150)
plt.show()
"
```

## 📝 已修改文件清单

- ✅ `train_moment.py`
  - 降低学习率: 1e-3 → 1e-4
  - 添加weight decay: 1e-5
  - 添加CosineAnnealingLR调度器
  - 添加梯度裁剪: max_norm=1.0
  - 添加NaN检测和跳过逻辑
  - 修复JSON序列化问题
  - 修复字体警告

---

**修复日期**: 2026-04-22  
**状态**: ✅ 已完成  
**下一步**: 重新运行训练验证修复效果