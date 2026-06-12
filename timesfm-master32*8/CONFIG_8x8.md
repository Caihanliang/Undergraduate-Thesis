# TimesFM 8输入8输出配置说明

## 📋 配置概览

本项目已配置为 **8输入8输出** 的时序预测任务：
- **输入**: 过去 8 小时的交通流量数据
- **输出**: 预测未来 8 小时的交通流量
- **时间粒度**: 小时级
- **适用场景**: 短期交通预测、实时调度、拥堵预警

---

## 🔧 主要修改点

### 1. 数据预处理脚本 (`prepare_traffic_data.py`)

**修改内容**:
```python
# 原配置
context_len = 64   # 64小时输入
horizon_len = 24   # 24小时预测

# 新配置（8输入8输出）
context_len = 8    # 8小时输入
horizon_len = 8    # 8小时预测
```

**影响**:
- 最小序列长度要求从 88 降低到 16
- 更多站点和特征可以满足训练要求
- 生成的样本数量增加

### 2. 微调训练脚本 (`finetune_timesfm.py`)

**修改内容**:
```python
# 默认参数
parser.add_argument('--context_len', type=int, default=8)
parser.add_argument('--horizon_len', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=32)  # 增大batch size

# 输出目录
--output_dir checkpoints/traffic-lora-8x8
```

**优势**:
- 更短的训练序列，减少显存占用
- 可以使用更大的 batch_size（32 vs 16）
- 更快的训练速度

### 3. 预测可视化脚本 (`predict_and_visualize.py`)

**修改内容**:
```python
# 适配器路径
--adapter_path checkpoints/traffic-lora-8x8

# 输出目录
--output_dir prediction_results_8x8

# 图表标题添加配置信息
f'Config: {context_len}-in {horizon_len}-out'
```

**改进**:
- 图表标题明确显示配置
- 纯英文标签（符合Matplotlib规范）
- 避免中文字体问题

### 4. 一键运行脚本 (`run_complete_pipeline.sh`)

**修改内容**:
```bash
python finetune_timesfm.py \
    --context_len 8 \
    --horizon_len 8 \
    --batch_size 32 \
    --output_dir checkpoints/traffic-lora-8x8

python predict_and_visualize.py \
    --adapter_path checkpoints/traffic-lora-8x8 \
    --output_dir prediction_results_8x8
```

---

## 📊 配置对比

| 配置项 | 原配置 | 新配置（8x8） | 说明 |
|--------|--------|---------------|------|
| 输入窗口 | 64 小时 | **8 小时** | 更短的上下文 |
| 预测窗口 | 24 小时 | **8 小时** | 短期预测 |
| 最小序列长度 | 88 | **16** | 更容易满足 |
| Batch Size | 16 | **32** | 可使用更大batch |
| 训练速度 | 较慢 | **更快** | 序列更短 |
| 显存占用 | 较高 | **较低** | 计算量减少 |
| 适用场景 | 中长期预测 | **短期预测** | 实时调度 |

---

## 🚀 使用方法

### 快速开始

```bash
cd /home/user/Downloads/cai/timesfm-master

# 一键运行（8输入8输出）
./run_complete_pipeline.sh
```

### 分步执行

```bash
# 1. 数据预处理
python prepare_traffic_data.py

# 2. 微调训练
python finetune_timesfm.py \
    --context_len 8 \
    --horizon_len 8 \
    --epochs 10 \
    --batch_size 32

# 3. 预测评估
python predict_and_visualize.py \
    --adapter_path checkpoints/traffic-lora-8x8 \
    --output_dir prediction_results_8x8
```

---

## 🎯 性能预期

### 训练指标（参考）
```
Epoch 1/10 — Train Loss: 0.2345, Val Loss: 0.1987
...
Epoch 10/10 — Train Loss: 0.1234, Val Loss: 0.1123

Average MAE:  ~25-35 (取决于站点和特征)
Average RMSE: ~35-45
```

### 推理速度
- **单样本预测**: < 10ms (GPU)
- **批量预测**: ~100 样本/秒
- **相比64输入**: 速度提升约 3-5 倍

---

## 🔄 切换其他配置

如果需要尝试不同的输入输出配置，只需修改参数：

### 改为 24输入24输出
```bash
python prepare_traffic_data.py  # 需修改脚本中的 context_len 和 horizon_len

python finetune_timesfm.py \
    --context_len 24 \
    --horizon_len 24 \
    --batch_size 16 \
    --output_dir checkpoints/traffic-lora-24x24
```

### 改为 16输入48输出
```bash
python finetune_timesfm.py \
    --context_len 16 \
    --horizon_len 48 \
    --batch_size 16 \
    --output_dir checkpoints/traffic-lora-16x48
```

---

## ⚠️ 注意事项

### 1. 数据要求
- 每个序列至少需要 `context_len + horizon_len = 16` 个时间点
- 8输入8输出对数据长度要求很低，几乎所有站点都满足

### 2. 预测精度
- 短期预测（8小时）通常比长期预测（24+小时）更准确
- MAE 通常在 20-40 范围内（取决于流量大小）

### 3. 显存优化
- 8输入8输出配置显存占用很低
- 可以在普通 GPU（如 RTX 3060）上运行
- Batch size 可以设为 32-64

### 4. 训练时间
- 每 epoch 约 5-10 分钟（取决于样本数）
- 10 epochs 总计约 1-2 小时

---

## 📝 文件结构

```
timesfm-master/
├── prepare_traffic_data.py          # 数据预处理（8x8）
├── finetune_timesfm.py              # 微调训练（8x8）
├── predict_and_visualize.py         # 预测可视化（8x8）
├── setup_env.sh                     # 环境配置
├── run_complete_pipeline.sh         # 一键运行（8x8）
├── README_TRAFFIC.md                # 项目说明
├── CONFIG_8x8.md                    # 本配置文件
├── dataset/
│   ├── 观测站小时交通量-9.csv       # 原始数据
│   └── preprocessed/                # 预处理数据（8x8）
├── checkpoints/
│   └── traffic-lora-8x8/            # 模型检查点
└── prediction_results_8x8/          # 预测结果
```

---

## 💡 优化建议

### 提高预测精度
1. **增加训练轮数**: `--epochs 20`
2. **增加采样数**: `--num_samples 10000`
3. **调整学习率**: `--lr 5e-5`（更小）
4. **增大 LoRA rank**: `--lora_r 8`

### 加速训练
1. **减少采样数**: `--num_samples 2000`
2. **减少轮数**: `--epochs 5`
3. **增大 batch size**: `--batch_size 64`

### 平衡配置
```bash
python finetune_timesfm.py \
    --context_len 8 \
    --horizon_len 8 \
    --epochs 15 \
    --batch_size 32 \
    --lr 8e-5 \
    --lora_r 4 \
    --num_samples 8000
```

---

## 🎉 总结

✅ **已完成**: 将 TimesFM 项目配置为 8输入8输出
✅ **优势**: 更快的训练速度、更低的显存占用、适合短期预测
✅ **易用性**: 一键运行脚本，开箱即用
✅ **灵活性**: 可轻松切换到其他配置

**开始使用**: `./run_complete_pipeline.sh`
