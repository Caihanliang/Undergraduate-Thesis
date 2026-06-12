# TimesFM 交通流量预测项目（8输入8输出）

## 📌 快速开始

### 方式一：一键运行（推荐）

```bash
cd /home/user/Downloads/cai/timesfm-master

# 1. 配置环境（首次运行）
chmod +x setup_env.sh
./setup_env.sh

# 2. 运行完整流程（8输入8输出）
chmod +x run_complete_pipeline.sh
./run_complete_pipeline.sh
```

### 方式二：分步执行

```bash
# 激活环境
source venv/bin/activate

# 步骤 1: 数据预处理（8输入8输出）
python prepare_traffic_data.py

# 步骤 2: 微调训练（8输入8输出）
python finetune_timesfm.py \
    --context_len 8 \
    --horizon_len 8 \
    --epochs 10 \
    --batch_size 32

# 步骤 3: 预测和可视化
python predict_and_visualize.py \
    --adapter_path checkpoints/traffic-lora-8x8 \
    --output_dir prediction_results_8x8
```

---

## 🎯 项目说明

本项目从零开始复现 **TimesFM 2.5** 时间序列预测模型，应用于高速公路交通流量预测。

### ⚙️ 时序配置：**8输入8输出**
- **输入窗口**: 过去 8 小时的交通流量数据
- **输出窗口**: 预测未来 8 小时的交通流量
- **时间粒度**: 小时级

### 预测目标（4维特征）
- ✅ 小客车上行流量 (`passenger_car_up`)
- ✅ 小客车下行流量 (`passenger_car_down`)
- ✅ 非小客车上行流量 (`non_passenger_car_up`) = 汽车自然数 - 小客车
- ✅ 非小客车下行流量 (`non_passenger_car_down`) = 汽车自然数 - 小客车

### 技术特点
- 🚀 使用 Google Research 的 TimesFM 2.5 基础模型（200M 参数）
- 🔧 LoRA 参数高效微调（仅训练 0.6% 参数）
- 📊 Channel Independence 策略处理多变量时序
- 🎨 自动生成预测可视化图表
- ⏱️ **8小时短期预测**：适合实时交通管理和调度

---

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `prepare_traffic_data.py` | 数据预处理脚本（8输入8输出） |
| `finetune_timesfm.py` | 微调训练脚本（8输入8输出） |
| `predict_and_visualize.py` | 预测和可视化脚本（8输入8输出） |
| `setup_env.sh` | 环境配置脚本 |
| `run_complete_pipeline.sh` | 一键运行脚本（8输入8输出） |
| `TRAFFIC_PREDICTION_GUIDE.md` | 详细使用指南 |

---

## 📊 数据集

- **原始数据**: `dataset/观测站小时交通量-9.csv`
- **预处理后**: `dataset/preprocessed/`
- **特征维度**: 4 维（小客车上下行 + 非小客车上下行）
- **时间粒度**: 小时级
- **预测配置**: **8输入8输出**

---

## ⚙️ 核心参数

### 训练参数（8输入8输出）
```bash
--context_len 8       # 输入窗口长度：8小时
--horizon_len 8       # 预测 horizon：8小时
--epochs 10           # 训练轮数
--batch_size 32       # 批次大小（8小时窗口可使用更大batch）
--lr 1e-4            # 学习率
--lora_r 4           # LoRA rank
```

### 可调参数建议
- **显存不足**: 减小 `--batch_size 16`
- **提高精度**: 增加 `--epochs 20` 或 `--num_samples 10000`
- **加速训练**: 减少 `--num_samples 2000` 或 `--epochs 5`
- **更长预测**: 修改为 `--horizon_len 24`（24小时预测）

---

## 📈 预期输出

### 训练完成后
```
checkpoints/traffic-lora-8x8/
├── adapter_model.bin      # LoRA 适配器权重
├── adapter_config.json    # LoRA 配置
└── training_history.json  # 训练历史
```

### 预测完成后
```
prediction_results_8x8/
├── G0401L010430121_passenger_car_up.png     # 可视化图表
├── G0401L010430121_passenger_car_down.png
├── ...
└── prediction_results.json                   # 预测结果和指标
```

---

## 🔍 常见问题

### Q: 为什么选择8输入8输出？
A: 8小时短期预测适合：
- 实时交通调度和管理
- 短时拥堵预警
- 收费站人员排班
- 更快的推理速度

### Q: 如何改为其他配置（如24输入24输出）？
A: 修改参数即可：
```bash
python finetune_timesfm.py --context_len 24 --horizon_len 24
```

### Q: 如何查看训练进度？
A: 实时查看日志：
```bash
tail -f training.log
```

### Q: 如何使用已训练的模型进行预测？
A: 
```bash
python predict_and_visualize.py \
    --adapter_path checkpoints/traffic-lora-8x8 \
    --output_dir my_predictions
```

### Q: 下载模型太慢怎么办？
A: 已配置国内镜像，确保环境变量生效：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 📚 详细文档

完整的使用指南、参数说明、性能优化建议请查看：
👉 [TRAFFIC_PREDICTION_GUIDE.md](TRAFFIC_PREDICTION_GUIDE.md)

---

## 🎓 技术背景

- **TimesFM**: Google Research 开发的时间序列基础模型
- **LoRA**: Low-Rank Adaptation，参数高效微调方法
- **Channel Independence**: 将多变量时序拆分为单变量独立预测
- **8输入8输出**: 短期预测配置，平衡精度和效率

参考论文：[A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688)

---

## ✨ 下一步

1. ✅ 完成环境配置
2. ✅ 运行数据预处理（8输入8输出）
3. ✅ 执行微调训练（8输入8输出）
4. ✅ 生成预测结果
5. 🔜 分析预测效果
6. 🔜 调整超参数优化
7. 🔜 尝试其他配置（如24输入24输出）
8. 🔜 部署为 API 服务

---

**祝使用愉快！** 🚀
