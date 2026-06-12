# Time-MoE 快速开始指南

## 📋 项目概述

本项目使用 Time-MoE（Time Series Mixture of Experts）模型进行交通流量预测。

**预测目标：**
- 小客车上行流量
- 小客车下行流量
- (汽车自然数-小客车)上行流量
- (汽车自然数-小客车)下行流量

**时序配置：** 8小时输入 → 8小时输出

---

## 🚀 快速开始

### 1️⃣ 环境搭建

```bash
# 赋予执行权限
chmod +x setup_env.sh

# 运行环境搭建脚本
bash setup_env.sh

# 激活环境
conda activate time-moe
```

### 2️⃣ 数据预处理

```bash
python preprocess_data.py
```

这将：
- 读取 `dataset/` 目录下的CSV文件
- 提取4个目标特征
- 创建滑动窗口序列（8输入8输出）
- 划分训练集(70%)、验证集(15%)、测试集(15%)
- 保存到 `processed_data/` 目录

### 3️⃣ 模型训练

#### 选项A: 使用预训练模型微调（推荐）

```bash
python torch_dist_run.py main.py \
    -d ./processed_data/train.jsonl \
    -m Maple728/TimeMoE-50M \
    -o ./logs/time_moe_traffic \
    --max_length 64 \
    --micro_batch_size 16 \
    --global_batch_size 64 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --stride 1 \
    --normalization_method zero \
    --precision fp32 \
    --save_strategy epoch \
    --logging_steps 10
```

#### 选项B: 从头训练

```bash
python torch_dist_run.py main.py \
    -d ./processed_data/train.jsonl \
    -o ./logs/time_moe_from_scratch \
    --from_scratch \
    --max_length 64 \
    --micro_batch_size 16 \
    --global_batch_size 64 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --stride 1
```

#### 选项C: 使用GPU加速训练

```bash
# 确保CUDA可用
python -c "import torch; print(torch.cuda.is_available())"

# 使用多GPU训练
python torch_dist_run.py main.py \
    -d ./processed_data/train.jsonl \
    -m Maple728/TimeMoE-50M \
    -o ./logs/time_moe_gpu \
    --max_length 64 \
    --micro_batch_size 32 \
    --global_batch_size 128 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --precision fp16  # 使用混合精度
```

### 4️⃣ 推理和可视化

```bash
python inference_and_visualize.py
```

这将：
- 加载训练好的模型
- 在测试集上进行预测
- 计算评估指标（MAE, RMSE, MAPE）
- 生成可视化图表
- 保存预测结果到CSV

---

## 📊 一键运行完整流程

```bash
chmod +x run_complete_pipeline.sh
bash run_complete_pipeline.sh
```

---

## 📁 项目结构

```
Time-MoE-main/
├── dataset/                      # 原始数据
│   ├── 观测站小时交通量-9.csv
│   └── 观测站小时交通量-10.csv
├── processed_data/               # 预处理后的数据（自动生成）
│   ├── train.jsonl              # 训练集
│   ├── val.jsonl                # 验证集
│   ├── test.jsonl               # 测试集
│   └── metadata.json            # 元数据
├── logs/                         # 训练日志和模型检查点（自动生成）
│   └── time_moe_traffic/
├── visualization_results/        # 可视化结果（自动生成）
│   ├── sample_0.png
│   ├── sample_1.png
│   └── ...
├── prediction_results.csv        # 预测结果（自动生成）
├── preprocess_data.py           # 数据预处理脚本
├── inference_and_visualize.py   # 推理和可视化脚本
├── setup_env.sh                 # 环境搭建脚本
├── run_complete_pipeline.sh     # 一键运行脚本
├── main.py                      # 训练入口
├── run_eval.py                  # 评估脚本
└── README_QUICKSTART.md         # 本文件
```

---

## ⚙️ 配置说明

### 数据预处理配置

编辑 `preprocess_data.py` 中的 `config` 字典：

```python
config = {
    'context_length': 8,          # 输入长度
    'prediction_length': 8,       # 输出长度
    'stride': 1,                  # 滑动步长（小数据集建议用1）
    'train_ratio': 0.7,          # 训练集比例
    'val_ratio': 0.15,           # 验证集比例
    'data_dir': './dataset',     # 数据目录
    'output_dir': './processed_data'  # 输出目录
}
```

### 训练参数配置

主要参数说明：

| 参数 | 说明 | 默认值 | 建议 |
|------|------|--------|------|
| `--max_length` | 最大序列长度 | 1024 | context+pred的倍数，如64 |
| `--micro_batch_size` | 单设备batch size | 16 | 根据显存调整 |
| `--global_batch_size` | 全局batch size | 64 | micro_batch * GPU数 * grad_accum |
| `--learning_rate` | 学习率 | 1e-4 | 1e-4 ~ 5e-5 |
| `--num_train_epochs` | 训练轮数 | 1.0 | 小数据集建议10-20 |
| `--stride` | 滑动步长 | None | 小数据集设为1 |
| `--precision` | 精度模式 | fp32 | GPU可用fp16/bf16 |
| `--save_strategy` | 保存策略 | no | epoch/steps |

---

## 🔧 常见问题

### Q1: CUDA out of memory

**解决方案：**
```bash
# 减小batch size
--micro_batch_size 8

# 使用梯度累积
--global_batch_size 64  # 保持全局batch不变

# 启用梯度检查点
--gradient_checkpointing

# 使用混合精度
--precision fp16
```

### Q2: 训练速度太慢

**解决方案：**
```bash
# 安装flash-attn（需要GPU）
pip install flash-attn==2.6.3 --no-build-isolation

# 使用混合精度
--precision fp16

# 增加batch size（如果显存允许）
--micro_batch_size 32

# 减少dataloader workers
--dataloader_num_workers 2
```

### Q3: 如何评估模型？

```bash
# 使用run_eval.py进行评估
python run_eval.py \
    -d ./processed_data/test.jsonl \
    -m ./logs/time_moe_traffic/checkpoint-xxx \
    -p 8 \
    -c 8
```

### Q4: 如何使用自定义模型？

```python
# 在inference_and_visualize.py中修改模型路径
config = {
    'model_path': './logs/time_moe_traffic/checkpoint-xxx',  # 你的模型路径
    'device': 'cuda',  # 如果有GPU
    ...
}
```

---

## 📈 模型版本选择

Time-MoE 提供两个预训练模型：

| 模型 | 参数量 | HuggingFace ID | 适用场景 |
|------|--------|----------------|----------|
| Time-MoE Base | 50M | `Maple728/TimeMoE-50M` | 快速实验，资源有限 |
| Time-MoE Large | 200M | `Maple728/TimeMoE-200M` | 更高精度，资源充足 |

**推荐：** 从50M开始，如果效果不满意再尝试200M。

---

## 📝 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{shi2024timemoe,
    title={Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts}, 
    author={Xiaoming Shi and Shiyu Wang and Yuqi Nie and Dianqi Li and Zhou Ye and Qingsong Wen and Ming Jin},
    year={2024},
    eprint={2409.16040},
    archivePrefix={arXiv},
    url={https://arxiv.org/abs/2409.16040}, 
}
```

---

## 💡 下一步

1. **调整超参数**：尝试不同的学习率、batch size等
2. **增加训练轮数**：如果验证集指标还在下降
3. **尝试更大模型**：从50M升级到200M
4. **数据增强**：增加更多月份的数据
5. **特征工程**：添加时间特征（小时、星期等）

---

## 📞 支持

如有问题，请查看：
- Time-MoE 官方文档：https://github.com/Time-MoE/Time-MoE
- 论文：https://arxiv.org/abs/2409.16040
