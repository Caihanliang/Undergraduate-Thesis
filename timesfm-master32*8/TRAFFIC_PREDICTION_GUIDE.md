# TimesFM 交通流量预测 - 完整复现指南

## 📋 项目概述

本项目使用 **TimesFM 2.5**（Google Research 的时间序列基础模型）对高速公路交通流量进行预测。

### 预测目标（4维特征）
1. **小客车上行** (`passenger_car_up`)
2. **小客车下行** (`passenger_car_down`)
3. **非小客车上行** (`non_passenger_car_up`) = 汽车自然数 - 小客车（上行）
4. **非小客车下行** (`non_passenger_car_down`) = 汽车自然数 - 小客车（下行）

### 技术栈
- **模型**: TimesFM 2.5 (200M 参数)
- **微调方法**: LoRA (参数高效微调)
- **框架**: HuggingFace Transformers + PEFT + PyTorch
- **数据**: 小时级交通流量数据

---

## 🚀 快速开始

### 步骤 1: 环境配置

```bash
cd /home/user/Downloads/cai/timesfm-master

# 运行环境配置脚本
chmod +x setup_env.sh
./setup_env.sh
```

或者手动安装：

```bash
# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft pandas numpy matplotlib scikit-learn
pip install -e .
```

### 步骤 2: 数据预处理

```bash
python prepare_traffic_data.py
```

**功能**：
- 从原始 CSV 提取 4 维交通流量特征
- 按观测站和方向分离数据
- 计算非小客车流量（汽车自然数 - 小客车）
- 划分训练/验证/测试集（70%/15%/15%）
- 保存为 NPZ 格式

**输出**：
```
dataset/preprocessed/
├── train_series.npz      # 训练序列
├── val_series.npz        # 验证序列
├── test_series.npz       # 测试序列
└── metadata.json         # 元数据
```

### 步骤 3: 微调训练

```bash
python finetune_timesfm.py \
    --context_len 64 \
    --horizon_len 24 \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-4 \
    --lora_r 4 \
    --lora_alpha 8 \
    --num_samples 5000 \
    --output_dir checkpoints/traffic-lora
```

**关键参数说明**：
- `--context_len`: 输入窗口长度（必须是 32 的倍数，默认 64）
- `--horizon_len`: 预测未来小时数（默认 24）
- `--epochs`: 训练轮数（建议 5-10）
- `--batch_size`: 批次大小（根据显存调整，默认 16）
- `--lr`: 学习率（默认 1e-4）
- `--lora_r`: LoRA rank（默认 4，参数量约 0.6%）
- `--num_samples`: 每 epoch 采样的训练窗口数

**训练过程**：
- 使用 Cosine Annealing 学习率调度器
- 梯度裁剪防止爆炸（max_norm=1.0）
- 自动保存最佳模型（基于验证损失）
- 记录训练历史到 `training_history.json`

### 步骤 4: 预测和可视化

```bash
python predict_and_visualize.py \
    --adapter_path checkpoints/traffic-lora \
    --output_dir prediction_results
```

**输出**：
```
prediction_results/
├── G0401L010430121_passenger_car_up.png    # 可视化图表
├── G0401L010430121_passenger_car_down.png
├── ...
└── prediction_results.json                  # 所有预测结果和指标
```

每个图表包含：
- 真实值 vs 预测值对比
- MAE、RMSE、MAPE 指标

---

## 📊 数据处理详解

### 特征工程

```python
# 原始数据列
观测日期, 小时, 观测站编号, 行驶方向, 小客车, 汽车自然数, ...

# 计算非小客车
非小客车 = 汽车自然数 - 小客车

# 提取 4 维特征
1. passenger_car_up     = 小客车[上行]
2. passenger_car_down   = 小客车[下行]
3. non_passenger_car_up = 非小客车[上行]
4. non_passenger_car_down = 非小客车[下行]
```

### Channel Independence 策略

TimesFM 采用**通道独立**策略，将多变量时间序列拆分为多个单变量序列：

```
站点 A:
  - passenger_car_up      → 序列 1
  - passenger_car_down    → 序列 2
  - non_passenger_car_up  → 序列 3
  - non_passenger_car_down → 序列 4

站点 B:
  - passenger_car_up      → 序列 5
  - ...
```

**优势**：
- 简化模型复杂度
- 避免变量间的相关性干扰
- 更容易捕捉每个特征的独立模式

---

## 🔧 高级配置

### 调整预测 Horizon

```bash
# 预测未来 48 小时
python finetune_timesfm.py --horizon_len 48

# 预测未来 168 小时（1周）
python finetune_timesfm.py --horizon_len 168
```

### 调整上下文窗口

```bash
# 使用更长的历史窗口（需要更多数据）
python finetune_timesfm.py --context_len 128
```

### 仅评估（不训练）

```bash
python finetune_timesfm.py --eval_only --output_dir checkpoints/traffic-lora
```

### 自定义 LoRA 配置

```bash
# 更大的 LoRA rank（更多可训练参数）
python finetune_timesfm.py --lora_r 8 --lora_alpha 16

# 更小的 dropout
python finetune_timesfm.py --lora_dropout 0.01
```

---

## 📈 性能优化建议

### 1. 显存优化
```bash
# 减小 batch_size
python finetune_timesfm.py --batch_size 8

# 启用梯度检查点（在代码中添加）
model.gradient_checkpointing_enable()
```

### 2. 训练加速
```bash
# 减少采样窗口数（牺牲精度换速度）
python finetune_timesfm.py --num_samples 2000

# 减少 epochs
python finetune_timesfm.py --epochs 5
```

### 3. 提高精度
```bash
# 增加采样窗口数
python finetune_timesfm.py --num_samples 10000

# 增加 epochs
python finetune_timesfm.py --epochs 20

# 使用更大的 LoRA rank
python finetune_timesfm.py --lora_r 8
```

---

## 🐛 常见问题

### Q1: 下载模型时连接超时
**解决**：已配置 HuggingFace 镜像，确保环境变量生效：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q2: CUDA out of memory
**解决**：减小 batch_size 或 context_len：
```bash
python finetune_timesfm.py --batch_size 8 --context_len 32
```

### Q3: Loss 为 NaN
**解决**：
- 降低学习率：`--lr 5e-5`
- 检查数据是否有异常值
- 确保数据预处理正确

### Q4: 预测效果不佳
**排查步骤**：
1. 检查数据质量（是否有缺失值）
2. 增加训练 epochs
3. 增加 num_samples
4. 调整 horizon_len（ shorter horizon 通常更容易）

---

## 📝 文件结构

```
timesfm-master/
├── prepare_traffic_data.py          # 数据预处理脚本
├── finetune_timesfm.py              # 微调训练脚本
├── predict_and_visualize.py         # 预测和可视化脚本
├── setup_env.sh                     # 环境配置脚本
├── dataset/
│   ├── 观测站小时交通量-9.csv       # 原始数据
│   └── preprocessed/                # 预处理后的数据
├── checkpoints/
│   └── traffic-lora/                # 保存的 LoRA 适配器
├── prediction_results/              # 预测结果和可视化
└── training.log                     # 训练日志
```

---

## 🎯 预期结果

### 训练指标示例
```
Epoch 1/10 — Train Loss: 0.2345, Val Loss: 0.1987
Epoch 2/10 — Train Loss: 0.1876, Val Loss: 0.1654
...
Epoch 10/10 — Train Loss: 0.1234, Val Loss: 0.1123

平均 Zero-shot MAE: 45.67
平均 Fine-tuned MAE: 32.45
改进幅度: 28.9%
```

### 可视化输出
- 每个站点每个特征生成一张对比图
- 显示真实值和预测值曲线
- 标注 MAE、RMSE、MAPE 指标

---

## 📚 参考资料

- [TimesFM 官方仓库](https://github.com/google-research/timesfm)
- [TimesFM 论文](https://arxiv.org/abs/2310.10688)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT 库](https://huggingface.co/docs/peft)

---

## 💡 下一步改进方向

1. **多步预测优化**：尝试不同的 horizon 长度
2. **特征工程**：添加天气、节假日等外部特征
3. **集成学习**：结合多个模型的预测结果
4. **实时预测**：部署为 API 服务
5. **异常检测**：识别交通流量异常模式

---

## 📞 支持

如有问题，请检查：
1. `training.log` 查看训练日志
2. `prediction_results/prediction_results.json` 查看详细指标
3. 确保数据预处理正确执行

祝使用愉快！🎉
