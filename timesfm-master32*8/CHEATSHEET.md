# TimesFM 8输入8输出 - 快速参考卡片

## 📍 数据集路径
```
/home/user/Downloads/cai/timesfm-master/dataset/
├── 观测站小时交通量-9.csv
└── 观测站小时交通量-10.csv
```

## ⚡ 一键运行
```bash
cd /home/user/Downloads/cai/timesfm-master
./run_complete_pipeline.sh
```

## 🔧 配置文件
```python
# config.py - 所有配置都在这里
DATASET_DIR = '/home/user/Downloads/cai/timesfm-master/dataset'
CONTEXT_LEN = 8    # 8小时输入
HORIZON_LEN = 8    # 8小时预测
```

## 📊 预测目标
1. 小客车上行
2. 小客车下行
3. 非小客车上行（汽车自然数 - 小客车）
4. 非小客车下行（汽车自然数 - 小客车）

## 🎯 输出目录
- 预处理数据: `dataset/preprocessed/`
- 模型检查点: `checkpoints/traffic-lora-8x8/`
- 预测结果: `prediction_results_8x8/`
- 训练日志: `training.log`

## 💻 常用命令

### 查看配置
```bash
python config.py
```

### 仅数据预处理
```bash
python prepare_traffic_data.py
```

### 仅训练
```bash
python finetune_timesfm.py
```

### 仅预测
```bash
python predict_and_visualize.py
```

### 监控训练
```bash
tail -f training.log
```

## ⚙️ 修改配置

### 切换数据文件
```python
# config.py
RAW_DATA_FILE = '.../观测站小时交通量-9.csv'
```

### 更改时序配置
```python
# config.py
CONTEXT_LEN = 24
HORIZON_LEN = 24
```

### 调整训练参数
```python
# config.py
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
```

## 📁 项目结构
```
timesfm-master/
├── config.py                    # ⭐ 核心配置
├── prepare_traffic_data.py      # 数据预处理
├── finetune_timesfm.py          # 微调训练
├── predict_and_visualize.py     # 预测可视化
├── run_complete_pipeline.sh     # 一键运行
├── dataset/                     # 数据集
│   └── preprocessed/            # 预处理数据
├── checkpoints/                 # 模型检查点
└── prediction_results_8x8/      # 预测结果
```

## 🆘 常见问题

**Q: 如何知道使用的是哪个数据文件？**
```bash
python config.py  # 查看"原始数据文件"行
```

**Q: 显存不足怎么办？**
```python
# config.py
BATCH_SIZE = 16  # 减小batch size
```

**Q: 如何提高精度？**
```python
# config.py
EPOCHS = 20
NUM_SAMPLES = 10000
```

## 📚 文档索引
- `README_TRAFFIC.md` - 项目总览
- `QUICKSTART_8x8.md` - 快速启动
- `CONFIG_8x8.md` - 配置说明
- `PATH_CONFIG.md` - 路径配置
- `DATASET_PATH_SETUP.md` - 数据集设置

---
**版本**: 8输入8输出 | **更新时间**: 2026-04-23
