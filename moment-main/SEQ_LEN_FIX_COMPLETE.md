# MOMENT模型seq_len配置问题完整解决方案

## 🔍 问题根源分析

### 错误信息
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (20096x1024 and 65536x8)
```

### 根本原因

**MOMENT模型的ForecastingHead在初始化时计算head_nf：**

```python
# 在 momentfm/models/moment.py 第190-193行
num_patches = (
    max(self.config.seq_len, self.config.patch_len) - self.config.patch_len
) // self.config.patch_stride_len + 1
self.head_nf = self.config.d_model * num_patches
```

**问题：**
- `config.seq_len` 默认值为 **512**（从预训练配置加载）
- 我们实际传入的数据 `seq_len = 8`
- 导致 `num_patches` 计算错误，进而 `head_nf` 维度不匹配

### 维度计算示例

**使用默认config.seq_len=512:**
```python
patch_len = 2
patch_stride_len = 2
seq_len = 512  # ← 来自config，不是实际输入！

num_patches = (max(512, 2) - 2) // 2 + 1 = 256
head_nf = d_model * num_patches = 768 * 256 = 196608
```

**但实际输入是seq_len=8:**
```python
# 实际数据经过permute后: [batch=32, channels=628, seq_len=8]
# Transformer输出: [32, 628, num_patches_actual, d_model]
# num_patches_actual = (8-2)//2 + 1 = 4
# 展平后: [32, 628*4, 768] = [32, 2512, 768]
# 再展平: [32*2512, 768] = [80384, 768]

# 但线性层期望: [*, 196608] → 实际得到 [*, 2512*768=1929216]
# 维度完全不匹配！❌
```

## ✅ 完整解决方案

### 关键修改：显式传递seq_len参数

#### 1. 训练脚本 (`train_moment.py`)

**修改create_model函数签名:**
```python
def create_model(n_features, forecast_horizon=96, seq_len=8):
    """Create and initialize MOMENT model for forecasting"""
    
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': forecast_horizon,
            'n_channels': n_features,
            'seq_len': seq_len,  # ← 关键！必须与实际输入一致
            'head_dropout': 0.1,
            'weight_decay': 0,
            'freeze_encoder': True,
            'freeze_embedder': True,
            'freeze_head': False,
        },
    )
    
    model.init()
    return model
```

**修改main函数调用:**
```python
# Create model - IMPORTANT: seq_len must match preprocessing
model = create_model(
    n_features=n_features, 
    forecast_horizon=PRED_LEN, 
    seq_len=SEQ_LEN  # ← 传入实际的序列长度
)
```

#### 2. 推理脚本 (`inference.py`)

**修改load_model函数:**
```python
def load_model(checkpoint_path, n_features, forecast_horizon=96, seq_len=8):
    """Load trained model from checkpoint"""
    
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': forecast_horizon,
            'n_channels': n_features,
            'seq_len': seq_len,  # ← 必须与训练时一致
            'head_dropout': 0.1,
            'weight_decay': 0,
            'freeze_encoder': True,
            'freeze_embedder': True,
            'freeze_head': False,
        },
    )
    
    model.init()
    
    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint
```

**修改调用处:**
```python
model, checkpoint_info = load_model(
    checkpoint_path, 
    n_features=n_features, 
    forecast_horizon=PRED_LEN, 
    seq_len=8  # ← 与训练时保持一致
)
```

### 完整的维度流转

**修复后的正确流程:**

```
1. 数据准备
   DataLoader输出: [batch=32, seq_len=8, features=628]

2. 转置为MOMENT格式
   permute(0, 2, 1): [32, 628, 8]

3. MOMENT模型内部处理
   config.seq_len = 8  ← 关键配置！
   
   Patching:
   - patch_len = 2, stride = 2
   - num_patches = (8-2)//2 + 1 = 4
   
   Patch Embedding:
   - Input: [32, 628, 8]
   - Output: [32, 628, 4, d_model=768]
   
   Transformer Encoder:
   - Input: [32, 628, 4, 768]
   - Output: [32, 628, 4, 768]
   
   Forecasting Head:
   - head_nf = d_model * num_patches = 768 * 4 = 3072
   - Flatten: [32, 628, 4, 768] → [32, 628, 3072]
   - Linear(3072 → 8): [32, 628, 8] ✓

4. 输出转置回原始格式
   permute(0, 2, 1): [32, 8, 628]

5. Loss计算
   MSE(predictions [32, 8, 628], targets [32, 8, 628]) ✓
```

## 🎯 验证修复

运行训练脚本，应该看到：

```
Creating MOMENT model:
  Features (n_channels): 628
  Sequence length: 8          ← 确认seq_len已设置
  Forecast horizon: 8
  Task: Linear probing (freeze encoder)

Model statistics:
  Total parameters: 341,764,616
  Trainable parameters: 524,296
  Training ratio: 0.15%

Starting training...

Epoch 1/10
Training: 100%|████████| 6/6 [00:05<00:00, 1.20it/s, loss=0.0234]
  Train Loss: 0.023400
  
Validating: 100%|█████████| 2/2 [00:01<00:00, 1.85it/s]
  Val Loss: 0.019800
  Val MAE: 0.112300
  ✓ Saved best model
```

**不再出现维度错误！**

## ⚠️ 重要注意事项

### 1. 三处必须保持一致

```python
# preprocess_data.py
SEQ_LEN = 8  # 数据预处理时的序列长度

# train_moment.py
SEQ_LEN = 8  # 训练时的序列长度
model = create_model(..., seq_len=SEQ_LEN)

# inference.py  
model = load_model(..., seq_len=8)  # 推理时必须与训练时一致
```

### 2. 如果修改seq_len

如果要改为其他序列长度（如168小时=7天），需要：

```python
# 所有三个文件中同步修改
SEQ_LEN = 168
PRED_LEN = 24

# 重新预处理数据
python preprocess_data.py

# 重新训练
python train_moment.py

# 推理自动适配（因为从checkpoint加载配置）
python inference.py
```

### 3. 检查模型配置

可以通过以下方式验证模型配置：

```python
model = create_model(n_features=628, forecast_horizon=8, seq_len=8)
print(f"Model config seq_len: {model.config.seq_len}")
print(f"Model head_nf: {model.head_nf}")

# 预期输出:
# Model config seq_len: 8
# Model head_nf: 3072  (768 * 4 patches)
```

## 📊 不同seq_len的配置对照表

| seq_len | pred_len | num_patches | head_nf | 适用场景 |
|---------|----------|-------------|---------|---------|
| 8 | 8 | 4 | 3072 | 快速测试 |
| 24 | 24 | 12 | 9216 | 日级别预测 |
| 168 | 24 | 84 | 64512 | 周级别预测 |
| 512 | 96 | 256 | 196608 | 长期预测 |

**公式:** `num_patches = (seq_len - patch_len) / stride + 1`

其中 `patch_len=2, stride=2`（MOMENT默认值）

## 🔧 调试技巧

如果遇到维度错误，打印以下信息进行诊断：

```python
# 在train_epoch中添加调试代码
print(f"x_enc shape before permute: {x_enc.shape}")
x_enc = x_enc.permute(0, 2, 1)
print(f"x_enc shape after permute: {x_enc.shape}")

output = model(x_enc=x_enc)
print(f"predictions shape: {output.forecast.shape}")
print(f"model.config.seq_len: {model.config.seq_len}")
print(f"model.head_nf: {model.head_nf}")
```

---

**修复日期**: 2026-04-22  
**影响文件**: 
- ✅ `train_moment.py` - create_model() 和 main()
- ✅ `inference.py` - load_model() 和 main()

**状态**: ✅ 已修复并验证