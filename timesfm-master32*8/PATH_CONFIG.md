# TimesFM 项目路径配置说明

## 📂 目录结构

```
/home/user/Downloads/cai/timesfm-master/
├── config.py                          # 统一配置文件（核心）
├── dataset/                           # 数据集目录
│   ├── 观测站小时交通量-9.csv         # 原始数据文件1
│   ├── 观测站小时交通量-10.csv        # 原始数据文件2
│   └── preprocessed/                  # 预处理后的数据
│       ├── train_series.npz
│       ├── val_series.npz
│       ├── test_series.npz
│       └── metadata.json
├── checkpoints/                       # 模型检查点
│   └── traffic-lora-8x8/
├── prediction_results_8x8/            # 预测结果
└── training.log                       # 训练日志
```

---

## 🔧 配置文件说明

### `config.py` - 统一管理所有路径和参数

这是项目的**核心配置文件**，集中管理所有路径、超参数和设置。

#### 主要配置项

```python
# 路径配置
PROJECT_ROOT = '/home/user/Downloads/cai/timesfm-master'
DATASET_DIR = '/home/user/Downloads/cai/timesfm-master/dataset'
PREPROCESSED_DIR = '/home/user/Downloads/cai/timesfm-master/dataset/preprocessed'
CHECKPOINT_DIR = '/home/user/Downloads/cai/timesfm-master/checkpoints/traffic-lora-8x8'
PREDICTION_DIR = '/home/user/Downloads/cai/timesfm-master/prediction_results_8x8'

# 模型配置
MODEL_ID = 'google/timesfm-2.5-200m-transformers'
CONTEXT_LEN = 8    # 8小时输入
HORIZON_LEN = 8    # 8小时预测

# 训练配置
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
```

---

## 📝 如何修改数据集路径

### 方法1：自动检测（推荐）

脚本会**自动检测** `dataset/` 目录下的第一个 CSV 文件：

```python
# config.py 中保持默认设置
RAW_DATA_FILE = None  # 自动检测
```

**优点**：无需手动配置，开箱即用

---

### 方法2：指定特定文件

如果需要指定使用哪个数据文件，在 `config.py` 中修改：

```python
# 使用观测站小时交通量-9.csv
RAW_DATA_FILE = '/home/user/Downloads/cai/timesfm-master/dataset/观测站小时交通量-9.csv'

# 或使用观测站小时交通量-10.csv
RAW_DATA_FILE = '/home/user/Downloads/cai/timesfm-master/dataset/观测站小时交通量-10.csv'
```

---

### 方法3：命令行参数

运行脚本时通过命令行参数指定：

```bash
# 数据预处理时指定
python prepare_traffic_data.py

# 脚本会自动检测并显示使用的文件
```

---

## 🎯 常用路径修改场景

### 场景1：更换数据集

```python
# 在 config.py 中修改
DATASET_DIR = '/path/to/your/new/dataset'
RAW_DATA_FILE = '/path/to/your/new/dataset/your_file.csv'
```

### 场景2：更改输出目录

```python
# 在 config.py 中修改
CHECKPOINT_DIR = '/path/to/save/checkpoints'
PREDICTION_DIR = '/path/to/save/predictions'
```

### 场景3：调整时序配置

```python
# 在 config.py 中修改
CONTEXT_LEN = 24   # 改为24小时输入
HORIZON_LEN = 24   # 改为24小时预测
```

---

## ✅ 验证配置

运行以下命令验证配置是否正确：

```bash
cd /home/user/Downloads/cai/timesfm-master
python config.py
```

**预期输出**：
```
============================================================
TimesFM 配置信息
============================================================
项目根目录: /home/user/Downloads/cai/timesfm-master
数据集目录: /home/user/Downloads/cai/timesfm-master/dataset
原始数据文件: 观测站小时交通量-9.csv
预处理目录: /home/user/Downloads/cai/timesfm-master/dataset/preprocessed
模型检查点: /home/user/Downloads/cai/timesfm-master/checkpoints/traffic-lora-8x8
预测结果: /home/user/Downloads/cai/timesfm-master/prediction_results_8x8
------------------------------------------------------------
模型ID: google/timesfm-2.5-200m-transformers
时序配置: 8输入8输出
训练轮数: 10
批次大小: 32
学习率: 0.0001
LoRA配置: r=4, alpha=8
============================================================
✓ 所有必要目录已创建/确认
```

---

## 🔄 配置文件优势

### 1. 集中管理
- 所有路径在一个地方配置
- 避免硬编码散落在各个脚本中
- 修改一处，全局生效

### 2. 易于维护
- 清晰的配置结构
- 详细的注释说明
- 类型提示和默认值

### 3. 灵活扩展
- 支持环境变量覆盖
- 支持命令行参数覆盖
- 支持多套配置切换

### 4. 自动检测
- 自动查找数据文件
- 自动创建必要目录
- 智能错误提示

---

## 🛠️ 高级用法

### 使用不同的配置集

```python
# 创建多个配置文件
config_dev.py      # 开发环境配置
config_prod.py     # 生产环境配置
config_test.py     # 测试环境配置

# 运行时指定
export CONFIG_FILE=config_dev.py
python prepare_traffic_data.py
```

### 环境变量覆盖

```bash
# 临时修改配置
export CONTEXT_LEN=24
export HORIZON_LEN=24
python finetune_timesfm.py
```

---

## ⚠️ 注意事项

1. **路径格式**：使用绝对路径，避免相对路径问题
2. **目录存在性**：脚本会自动创建不存在的目录
3. **文件权限**：确保有读写权限
4. **磁盘空间**：预留足够空间（至少10GB）

---

## 📞 常见问题

### Q: 如何查看当前使用的数据文件？
A: 运行 `python config.py` 或查看预处理脚本的输出

### Q: 可以同时使用多个数据文件吗？
A: 目前仅支持单个文件，如需合并请预先处理

### Q: 修改配置后需要重新运行哪些步骤？
A: 
- 修改数据文件 → 重新运行所有步骤
- 修改时序配置 → 重新运行所有步骤
- 修改训练参数 → 仅需重新训练

---

**配置完成！开始使用吧！** 🚀

```bash
./run_complete_pipeline.sh
```
