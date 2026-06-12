# MOMENT 交通流量预测项目复现指南

本指南将帮助你从零开始使用 MOMENT 模型进行高速公路交通流量预测。

## 📋 目录

1. [环境准备](#环境准备)
2. [数据说明](#数据说明)
3. [数据预处理](#数据预处理)
4. [模型训练](#模型训练)
5. [推理与评估](#推理与评估)
6. [结果解读](#结果解读)
7. [常见问题](#常见问题)

## 🔧 环境准备

### 1. Python 版本要求

推荐使用 **Python 3.11**

```bash
python --version  # 检查Python版本
```

### 2. 安装依赖

#### 方法一：使用 pip 安装 momentfm 包（推荐）

```bash
pip install momentfm
```

#### 方法二：从源码安装

```bash
cd /home/user/Downloads/cai/moment-main/moment-main
pip install -e .
```

#### 安装其他必要依赖

```bash
pip install pandas numpy matplotlib tqdm scikit-learn
```

### 3. 验证安装

```python
python -c "from momentfm import MOMENTPipeline; print('MOMENT installed successfully!')"
```

## 📊 数据说明

### 数据集位置

你的数据集位于: `dataset/` 目录下，包含多个CSV文件：
- `观测站小时交通量-9.csv` (2023年9月数据)
- `观测站小时交通量-10.csv` (2023年10月数据)
- ... (可以添加更多月份数据)

### 数据特征

**原始数据列:**
- 观测日期、小时、路线编号、观测站编号、观测站名称
- 行驶方向（上行/下行/断面）
- 车型分类：小货车、中货车、大货车、特大货车、集装箱车、**小客车**、大客车
- **汽车自然数**（所有汽车总数）
- 其他指标...

**提取的4个预测特征:**

根据需求，我们为每个站点提取以下4个特征：

1. **小客车上行** - 小型客车在上行方向的流量
2. **小客车下行** - 小型客车在下行方向的流量  
3. **非小客车上行** = (汽车自然数 - 小客车)上行 - 其他类型车辆在上行方向的流量
4. **非小客车下行** = (汽车自然数 - 小客车)下行 - 其他类型车辆在下行方向的流量

**数据结构示例:**
```
时间                    站点索引  小客车上行  小客车下行  非小客车上行  非小客车下行
2023-09-01 01:00:00    0        101.0      87.0       160.0        89.0
2023-09-01 02:00:00    0        95.0       82.0       155.0        85.0
...
```

### 数据处理流程

1. **合并多文件**: 自动合并 `dataset/` 目录下所有CSV文件
2. **特征计算**: 从原始数据计算4个目标特征
3. **数据透视**: 转换为时间×(站点×特征)的宽表格式
4. **标准化**: Z-score归一化（均值为0，标准差为1）
5. **数据集划分**: 按时间顺序划分为训练集(70%)、验证集(15%)、测试集(15%)
6. **滑动窗口**: 创建输入序列(512小时)和预测目标(96小时)

## 🚀 数据预处理

### 运行数据预处理脚本

```bash
cd /home/user/Downloads/cai/moment-main/moment-main
python preprocess_data.py
```

**预处理流程详解:**

1. **加载与合并**
   - 扫描 `dataset/` 目录下所有 `.csv` 文件
   - 按文件名排序后依次加载
   - 垂直拼接所有数据

2. **特征工程**
   ```python
   # 计算非小客车数量
   非小客车 = 汽车自然数 - 小客车
   
   # 分离上行和下行数据
   df_up = 数据[行驶方向 == '上行']
   df_down = 数据[行驶方向 == '下行']
   
   # 合并为4个特征
   merged = merge(df_up, df_down, on=['时间', '站点'])
   ```

3. **站点映射**
   - 为每个唯一的观测站编号分配整数索引（0, 1, 2, ...）
   - 保存映射关系到 `station_mapping.json`

4. **数据透视**
   - 将长格式转换为宽格式
   - 每个"站点-特征"组合成为独立列
   - 例如：`小客车上行_0`, `小客车下行_0`, ..., `非小客车下行_159`

5. **标准化处理**
   - 对每列独立进行Z-score标准化
   - 保存均值和标准差到 `normalization_params.json`

6. **创建滑动窗口**
   - 输入长度: 512小时（约21天）
   - 预测长度: 96小时（4天）
   - 步长: 1小时（重叠采样）

**输出文件:**
```
processed_data/
├── train.csv          # 训练集（宽表格式）
├── val.csv            # 验证集
└── test.csv           # 测试集

moment_data/
└── dataset.npz        # MOMENT格式的NumPy数组
    - input: [samples, 512, n_features]
    - target: [samples, 96, n_features]

station_mapping.json   # 站点索引映射 {0: {station_code, station_name}, ...}
normalization_params.json  # 归一化参数 {column_name: {mean, std}, ...}
```

**预期输出示例:**
```
Loading datasets from dataset
Found 2 CSV files:
  - 观测站小时交通量-10.csv
  - 观测站小时交通量-9.csv

✓ Merged dataset shape: (684138, 17)

Preprocessing traffic data...
Original data shape: (684138, 17)
Unique stations: 160
Time range: 2023-09-01 to 2023-10-31

Up direction samples: 342069
Down direction samples: 342069
Merged dataset shape: (342069, 7)

✓ Unique stations: 160
✓ Time range: 2023-09-01 01:00:00 to 2023-10-31 24:00:00

Pivoting and normalizing data...
Pivoted data shape: (744, 640)
Total features (stations × 4): 640

Data split:
  Train: 520 time points (640 features)
  Val:   112 time points (640 features)
  Test:  112 time points (640 features)

Creating MOMENT dataset format...
  Total samples to generate: 13
  ✓ Input shape: (13, 512, 640)
  ✓ Target shape: (13, 96, 640)

✓ All preprocessing completed successfully!
```

## 🎯 模型训练

### 配置参数

在 `train_moment.py` 中可以调整以下关键参数：

```python
SEQ_LEN = 512          # 输入序列长度（小时）
PRED_LEN = 96          # 预测步长（小时）
BATCH_SIZE = 32        # 批次大小
EPOCHS = 10            # 训练轮数
LEARNING_RATE = 1e-3   # 学习率
```

**参数调优建议:**
- **显存不足**: 减小 `BATCH_SIZE` 到 16 或 8
- **更快训练**: 减少 `EPOCHS` 到 5（先快速验证）
- **更长预测**: 增加 `PRED_LEN` 到 192 或 336

### 开始训练

```bash
python train_moment.py
```

**训练过程详解:**

1. **模型初始化**
   - 加载预训练的 MOMENT-1-large 模型（约85M参数）
   - 配置为 forecasting 任务
   - 设置 `n_channels = 站点数 × 4`

2. **冻结策略（Linear Probing）**
   ```python
   freeze_encoder = True   # 冻结Transformer编码器
   freeze_embedder = True  # 冻结Patch Embedding层
   freeze_head = False     # 仅训练线性预测头
   ```
   - **优点**: 训练快、显存占用低、避免过拟合
   - **可训练参数**: 通常只有几十万（<1%总参数）

3. **训练循环**
   - 优化器: Adam
   - 损失函数: MSE Loss
   - 学习率调度: 固定学习率
   - 早停机制: 保存验证集上MSE最低的模型

4. **可视化**
   - 最后一个epoch自动生成预测vs真实值对比图
   - 每个图展示一个站点的4个特征
   - 标注MAE和RMSE指标

**输出文件:**
```
checkpoints/
└── best_model.pth       # 最佳模型权重 + 训练元数据

results/
├── training_history.json  # 训练历史（loss曲线数据）
└── prediction_sample*_station*.png  # 预测可视化
```

**训练日志示例:**
```
Configuration:
  Device: cuda
  Sequence length: 512
  Prediction horizon: 96
  Batch size: 32
  Epochs: 10
  Learning rate: 0.001

Data loaded:
  Input shape: (13, 512, 640)
  Target shape: (13, 96, 640)
  Number of stations: 160
  Features per station: 4
  Total features: 640

Model statistics:
  Total parameters: 85,432,576
  Trainable parameters: 614,400
  Training ratio: 0.72%

Starting training...

Epoch 1/10
Training: 100%|████████| 1/1 [00:15<00:00, 15.23s/it, loss=0.0234]
  Train Loss: 0.023400
Validating: 100%|███████| 1/1 [00:03<00:00,  3.12s/it]
  Val Loss: 0.019800
  Val MAE: 0.112300
  ✓ Saved best model (val_loss: 0.019800, val_mae: 0.112300)

...

Training completed!
Best validation loss: 0.015600
```

### GPU 加速（强烈推荐）

如果有 NVIDIA GPU，代码会自动使用 CUDA：

```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**显存需求估算:**
- BATCH_SIZE=32, SEQ_LEN=512, 640特征: 约 8-12 GB
- 如遇OOM错误，减小BATCH_SIZE

## 📈 推理与评估

### 运行评估

```bash
python inference.py
```

**评估流程:**

1. **加载最佳模型**
   - 从 `checkpoints/best_model.pth` 恢复模型状态
   - 显示训练时的最佳验证指标

2. **测试集预测**
   - 批量处理测试样本
   - 生成96小时的预测值

3. **综合评估指标**
   - **整体指标**: MSE, RMSE, MAE, MAPE
   - **分特征指标**: 对4种车流类型分别计算
   - **分站点分析**: 可进一步细化到每个站点

4. **可视化生成**
   - 为多个样本和站点生成对比图
   - 标注高误差点（红色散点）
   - 显示每个特征的MAE/RMSE

**输出文件:**
```
results/
├── predictions.npz      # 预测值和真实值数组
├── metrics.json         # 详细评估指标
└── sample0_station0_G0401L010430121.png  # 可视化图表
    sample0_station1_*.png
    ...
```

**评估报告示例:**
```
Overall Evaluation Metrics:
======================================================================
  MSE:   0.015234
  RMSE:  0.123428
  MAE:   0.098765
  MAPE:  12.34%
======================================================================

Per-Feature Metrics (averaged across all stations):
----------------------------------------------------------------------
Feature         MAE          RMSE         MAPE        
----------------------------------------------------------------------
小客车上行       0.0876       0.1123       10.23%      
小客车下行       0.0923       0.1189       11.45%      
非小客车上行     0.1045       0.1356       13.67%      
非小客车下行     0.1101       0.1423       14.02%      
----------------------------------------------------------------------

Generating comprehensive visualization...
  Samples to plot: 3
  Stations to plot: 2
  ✓ Saved: sample0_station0_G0401L010430121.png
  ✓ Saved: sample0_station1_G0401L020430121.png
  ...

✓ Inference and evaluation completed successfully!
```

### 查看结果

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# 加载指标
with open('results/metrics.json', 'r', encoding='utf-8') as f:
    metrics = json.load(f)

print("整体指标:")
print(f"  MAE:  {metrics['mae']:.4f}")
print(f"  RMSE: {metrics['rmse']:.4f}")
print(f"  MAPE: {metrics['mape']:.2f}%")

print("\n各特征指标:")
for feat_name, feat_metrics in metrics['per_feature'].items():
    print(f"  {feat_name}: MAE={feat_metrics['mae']:.4f}, MAPE={feat_metrics['mape']:.2f}%")

# 加载预测结果
pred_data = np.load('results/predictions.npz')
predictions = pred_data['predictions']
targets = pred_data['targets']
print(f"\n预测形状: {predictions.shape}")  # [samples, 96, 640]
```

## 🔍 结果解读

### 评估指标含义

- **MSE (均方误差)**: 对大误差更敏感，值越小越好
- **RMSE (均方根误差)**: 与原始数据同量纲，易于理解
- **MAE (平均绝对误差)**: 鲁棒性强，不受极端值影响
- **MAPE (平均绝对百分比误差)**: 相对误差，便于跨数据集比较

### 性能基准参考

对于交通流量预测任务：
- **优秀**: MAPE < 10%
- **良好**: MAPE 10-15%
- **一般**: MAPE 15-20%
- **需改进**: MAPE > 20%

### 可视化解读

每个预测图包含：
- **蓝色实线**: 真实交通流量
- **蓝色虚线**: 模型预测值
- **红色散点**: 误差最大的前10%时间点
- **标题**: 显示该特征的MAE和RMSE

**观察要点:**
1. 预测曲线是否跟随真实趋势
2. 峰值和谷值是否准确捕捉
3. 红色散点分布是否集中（系统性偏差）

## ❓ 常见问题

### Q1: 显存不足怎么办？

**解决方案:**
1. 减小 batch size: `BATCH_SIZE = 16` 或 `8`
2. 缩短序列长度: `SEQ_LEN = 256`
3. 使用梯度累积
4. 启用混合精度训练

```python
# 混合精度训练示例（需手动添加到train_moment.py）
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(x_enc=x_enc)
    loss = criterion(output.forecast, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Q2: 如何加载本地模型而不从 HuggingFace 下载？

首次运行时会自动下载模型（约 3GB）。如需离线使用：

```python
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    local_files_only=True,  # 仅使用本地缓存
    model_kwargs={...}
)
```

模型默认缓存在: `~/.cache/huggingface/hub/`

### Q3: 如何处理多变量时间序列？

当前实现将所有站点的4个特征展平为一个大的特征向量。例如：
- 160个站点 × 4个特征 = 640个输入通道

如果显存受限，可以：
1. **选择重要站点**: 根据业务需求筛选关键站点
2. **聚类降维**: 对站点聚类，每类选代表站点
3. **PCA降维**: 使用主成分分析减少特征维度

### Q4: 训练太慢怎么办？

**优化建议:**
1. **使用GPU**（必须）: CPU训练会非常慢
2. **减少epochs**: 先试 3-5 个epoch快速验证
3. **增大learning rate**: 尝试 5e-3 加快收敛
4. **增加num_workers**: `DataLoader(num_workers=8)` 加速数据加载
5. **减小序列长度**: `SEQ_LEN = 256` 减少计算量

### Q5: 如何可视化特定站点的预测结果？

修改 `inference.py` 中的调用参数：

```python
# 只绘制站点5的结果
plot_comprehensive_results(
    predictions, targets, station_mapping,
    save_dir='results',
    n_samples=5,      # 绘制5个样本
    n_stations=1      # 只绘制1个站点
)

# 或者指定具体站点
for station_idx in [5, 10, 15]:  # 自定义站点列表
    plot_feature_predictions(...)
```

### Q6: 如何反归一化得到真实流量值？

```python
import json

# 加载归一化参数
with open('normalization_params.json', 'r') as f:
    norm_params = json.load(f)

# 反归一化函数
def denormalize(normalized_value, column_name):
    params = norm_params[column_name]
    return normalized_value * params['std'] + params['mean']

# 示例：反归一化第一个站点的小客车上行流量
column_name = '小客车上行_0'
pred_denorm = denormalize(predictions[0, :, 0], column_name)
true_denorm = denormalize(targets[0, :, 0], column_name)

print(f"预测值范围: {pred_denorm.min():.1f} - {pred_denorm.max():.1f}")
print(f"真实值范围: {true_denorm.min():.1f} - {true_denorm.max():.1f}")
```

### Q7: 数据量太少导致样本数不足？

如果时间序列长度不足以生成足够的滑动窗口样本：

**解决方案:**
1. **减小序列长度**: `SEQ_LEN = 256` 或 `128`
2. **减小预测长度**: `PRED_LEN = 48` 或 `24`
3. **增加数据**: 添加更多月份的CSV文件到 `dataset/` 目录
4. **使用步长采样**: 修改 `preprocess_data.py` 中的循环，使用 `step > 1`

```python
# 在 create_moment_dataset_format 中
step = 2  # 每2小时采样一次，样本数减半但覆盖更广
for i in range(0, total_samples, step):
    ...
```

### Q8: 如何评估不同站点的性能差异？

```python
# 计算每个站点的平均MAE
n_stations = predictions.shape[2] // 4
station_maes = []

for station_idx in range(n_stations):
    base_idx = station_idx * 4
    # 取该站点的所有4个特征
    station_pred = predictions[:, :, base_idx:base_idx+4]
    station_true = targets[:, :, base_idx:base_idx+4]
    
    mae = np.mean(np.abs(station_pred - station_true))
    station_maes.append(mae)

# 找出表现最好和最差的站点
best_station = np.argmin(station_maes)
worst_station = np.argmax(station_maes)

print(f"最佳站点 #{best_station}: MAE={station_maes[best_station]:.4f}")
print(f"最差站点 #{worst_station}: MAE={station_maes[worst_station]:.4f}")
```

## 🎓 进阶用法

### 全量微调（Fine-tuning）

如果想微调整个模型而不仅是预测头：

```python
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={
        'task_name': 'forecasting',
        'forecast_horizon': 96,
        'n_channels': n_features,
        'freeze_encoder': False,  # 解冻编码器
        'freeze_embedder': False,  # 解冻嵌入层
        'freeze_head': False,
    },
)
```

⚠️ **注意**: 全量微调需要：
- 更多显存（至少24GB）
- 更长训练时间
- 更大数据集防止过拟合
- 更小学习率（如 1e-5）

### 使用 LoRA 进行参数高效微调

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### 多步预测策略

如果需要预测不同时间跨度：

```python
# 短程预测（24小时）
PRED_LEN = 24

# 中程预测（1周）
PRED_LEN = 168

# 长程预测（2周）
PRED_LEN = 336
```

⚠️ 预测步长越长，难度越大，需要重新训练模型

## 📚 参考资料

- [MOMENT 官方仓库](https://github.com/moment-timeseries-foundation-model/moment)
- [MOMENT 论文 (ICML 2024)](https://arxiv.org/abs/2402.03885)
- [HuggingFace 模型页面](https://huggingface.co/AutonLab/MOMENT-1-large)
- [MOMENT 教程集合](https://github.com/moment-timeseries-foundation-model/moment/tree/main/tutorials)

## 🤝 贡献与反馈

如有问题或改进建议，请：
1. 检查本文档的"常见问题"部分
2. 查看 MOMENT 官方 GitHub Issues
3. 提交新的 Issue 描述具体问题

---

**最后更新**: 2026-04-22  
**数据集版本**: 观测站小时交通量（多车型分类）  
**特征定义**: 小客车上/下行 + 非小客车上/下行