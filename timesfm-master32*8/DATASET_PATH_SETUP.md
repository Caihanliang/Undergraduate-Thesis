# TimesFM 项目 - 数据集路径配置完成

## ✅ 已完成的修改

### 1. 创建统一配置文件 `config.py`

**功能**：
- 集中管理所有路径配置
- 自动检测 dataset 目录下的 CSV 文件
- 提供配置验证和打印功能
- 支持灵活的路径自定义

**关键配置**：
```python
PROJECT_ROOT = '/home/user/Downloads/cai/timesfm-master'
DATASET_DIR = '/home/user/Downloads/cai/timesfm-master/dataset'
RAW_DATA_FILE = None  # 自动检测第一个CSV文件
PREPROCESSED_DIR = os.path.join(DATASET_DIR, 'preprocessed')
```

---

### 2. 更新所有脚本使用 config 模块

#### 📄 `prepare_traffic_data.py`
- ✅ 导入 config 模块
- ✅ 使用 `config.get_raw_data_file()` 自动检测数据文件
- ✅ 使用 `config.PREPROCESSED_DIR` 作为输出目录
- ✅ 使用 `config.CONTEXT_LEN` 和 `config.HORIZON_LEN`

#### 📄 `finetune_timesfm.py`
- ✅ 导入 config 模块
- ✅ 使用 `config.PREPROCESSED_DIR` 作为数据目录
- ✅ 使用 `config.CHECKPOINT_DIR` 作为输出目录
- ✅ 所有默认参数从 config 读取

#### 📄 `predict_and_visualize.py`
- ✅ 导入 config 模块
- ✅ 使用 `config.PREPROCESSED_DIR` 作为数据目录
- ✅ 使用 `config.CHECKPOINT_DIR` 作为适配器路径
- ✅ 使用 `config.PREDICTION_DIR` 作为输出目录

#### 📄 `run_complete_pipeline.sh`
- ✅ 设置 PROJECT_ROOT 变量
- ✅ cd 到项目根目录
- ✅ 确保路径一致性

---

## 📂 当前数据集配置

### 数据集目录
```
/home/user/Downloads/cai/timesfm-master/dataset/
├── 观测站小时交通量-9.csv      (29.6 MB)
└── 观测站小时交通量-10.csv     (30.1 MB)
```

### 自动检测逻辑
脚本会**自动选择第一个找到的 CSV 文件**（按字母顺序）：
- 当前会使用：`观测站小时交通量-10.csv`（因为 "10" < "9" 在字符串比较中）

### 如何指定特定文件

如果需要明确使用某个文件，在 `config.py` 中修改：

```python
# 使用观测站小时交通量-9.csv
RAW_DATA_FILE = '/home/user/Downloads/cai/timesfm-master/dataset/观测站小时交通量-9.csv'

# 或使用观测站小时交通量-10.csv
RAW_DATA_FILE = '/home/user/Downloads/cai/timesfm-master/dataset/观测站小时交通量-10.csv'
```

---

## 🚀 使用方法

### 方式一：一键运行（推荐）

```bash
cd /home/user/Downloads/cai/timesfm-master

# 首次运行需要配置环境
./setup_env.sh

# 运行完整流程
./run_complete_pipeline.sh
```

### 方式二：分步执行

```bash
cd /home/user/Downloads/cai/timesfm-master
source venv/bin/activate

# 步骤 1: 查看当前配置
python config.py

# 步骤 2: 数据预处理（自动检测数据文件）
python prepare_traffic_data.py

# 步骤 3: 微调训练
python finetune_timesfm.py

# 步骤 4: 预测评估
python predict_and_visualize.py
```

---

## 🔍 验证配置

### 检查当前使用的数据文件

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
原始数据文件: 观测站小时交通量-10.csv  <-- 自动检测到的文件
预处理目录: /home/user/Downloads/cai/timesfm-master/dataset/preprocessed
模型检查点: /home/user/Downloads/cai/timesfm-master/checkpoints/traffic-lora-8x8
预测结果: /home/user/Downloads/cai/timesfm-master/prediction_results_8x8
------------------------------------------------------------
时序配置: 8输入8输出
...
✓ 所有必要目录已创建/确认
```

---

## 📝 文件清单

### 核心脚本（4个）
1. ✅ `config.py` - 统一配置文件（新建）
2. ✅ `prepare_traffic_data.py` - 数据预处理（已更新）
3. ✅ `finetune_timesfm.py` - 微调训练（已更新）
4. ✅ `predict_and_visualize.py` - 预测可视化（已更新）

### 自动化脚本（2个）
5. ✅ `setup_env.sh` - 环境配置
6. ✅ `run_complete_pipeline.sh` - 一键运行（已更新）

### 文档（7个）
7. ✅ `README_TRAFFIC.md` - 项目说明
8. ✅ `CONFIG_8x8.md` - 8x8配置说明
9. ✅ `QUICKSTART_8x8.md` - 快速启动
10. ✅ `TRAFFIC_PREDICTION_GUIDE.md` - 完整指南
11. ✅ `SUMMARY.md` - 项目总结
12. ✅ `PATH_CONFIG.md` - 路径配置说明（新建）
13. ✅ `DATASET_PATH_SETUP.md` - 本文档（新建）

---

## 🎯 关键改进

### 1. 路径集中管理
- ❌ 之前：路径硬编码在各个脚本中
- ✅ 现在：所有路径在 `config.py` 统一管理

### 2. 自动数据检测
- ❌ 之前：需要手动指定文件名
- ✅ 现在：自动检测 dataset 目录下的 CSV 文件

### 3. 灵活配置
- ❌ 之前：修改路径需要改多个文件
- ✅ 现在：只需修改 `config.py` 一处

### 4. 配置验证
- ❌ 之前：运行时才发现路径错误
- ✅ 现在：可提前运行 `python config.py` 验证

---

## ⚙️ 配置项总览

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `PROJECT_ROOT` | `/home/user/Downloads/cai/timesfm-master` | 项目根目录 |
| `DATASET_DIR` | `{PROJECT_ROOT}/dataset` | 数据集目录 |
| `RAW_DATA_FILE` | `None` (自动检测) | 原始数据文件 |
| `PREPROCESSED_DIR` | `{DATASET_DIR}/preprocessed` | 预处理输出 |
| `CHECKPOINT_DIR` | `{PROJECT_ROOT}/checkpoints/traffic-lora-8x8` | 模型检查点 |
| `PREDICTION_DIR` | `{PROJECT_ROOT}/prediction_results_8x8` | 预测结果 |
| `CONTEXT_LEN` | `8` | 输入窗口长度 |
| `HORIZON_LEN` | `8` | 预测窗口长度 |
| `MODEL_ID` | `google/timesfm-2.5-200m-transformers` | 模型ID |

---

## 💡 使用建议

### 1. 首次使用
```bash
# 1. 查看配置
python config.py

# 2. 确认数据文件是否正确
#    如果不正确，修改 config.py 中的 RAW_DATA_FILE

# 3. 运行完整流程
./run_complete_pipeline.sh
```

### 2. 切换数据文件
```python
# 在 config.py 中修改
RAW_DATA_FILE = '/home/user/Downloads/cai/timesfm-master/dataset/观测站小时交通量-9.csv'
```

### 3. 更改预测配置
```python
# 在 config.py 中修改
CONTEXT_LEN = 24   # 24小时输入
HORIZON_LEN = 24   # 24小时预测
```

---

## 🎉 总结

✅ **数据集路径已正确配置**
- 位置：`/home/user/Downloads/cai/timesfm-master/dataset`
- 文件：自动检测 CSV 文件
- 配置：集中在 `config.py` 管理

✅ **所有脚本已更新**
- 使用统一的 config 模块
- 路径一致性强
- 易于维护和扩展

✅ **文档完善**
- 路径配置说明
- 使用指南
- 常见问题解答

**现在可以开始使用了！** 🚀

```bash
cd /home/user/Downloads/cai/timesfm-master
./run_complete_pipeline.sh
```
