# 数据更新说明

## 📊 数据集变更

### 旧数据集
- **路径**: `dataset/his_data_with_names.csv`
- **格式**: 已预处理的宽表格式
- **特征**: 小客车上行、小客车下行、非小客车上行、非小客车下行（直接提供）

### 新数据集
- **路径**: `dataset/观测站小时交通量-*.csv` (多个CSV文件)
- **格式**: 原始观测站数据，长表格式
- **原始列**: 观测日期、小时、路线编号、观测站编号、观测站名称、行驶方向、各车型数量...
- **提取特征**: 
  1. **小客车上行** - 直接从"小客车"列（上行方向）
  2. **小客车下行** - 直接从"小客车"列（下行方向）
  3. **非小客车上行** = 汽车自然数(上行) - 小客车(上行)
  4. **非小客车下行** = 汽车自然数(下行) - 小客车(下行)

## 🔧 代码修改清单

### 1. `preprocess_data.py` - 完全重写

**主要变更:**
- ✅ 支持加载多个CSV文件并自动合并
- ✅ 从原始数据计算4个目标特征
- ✅ 处理"上行/下行"方向分离
- ✅ 增强的错误处理和日志输出
- ✅ 更详细的数据统计信息

**关键函数:**
```python
load_and_merge_datasets()     # 合并多个CSV文件
preprocess_traffic_data()      # 提取4个特征
pivot_and_normalize()          # 数据透视和标准化
create_moment_dataset_format() # 创建滑动窗口
```

### 2. `train_moment.py` - 增强版

**主要变更:**
- ✅ 支持4特征格式的可视化
- ✅ 按站点和特征分别绘图
- ✅ 显示每个特征的MAE/RMSE
- ✅ 保存训练历史到JSON
- ✅ 更详细的模型统计信息

**新增功能:**
```python
plot_feature_predictions()  # 绘制单站点4特征对比图
```

### 3. `inference.py` - 全面升级

**主要变更:**
- ✅ 综合评估指标（整体 + 分特征）
- ✅ 智能防重叠标注
- ✅ 高误差点标记（红色散点）
- ✅ 中文字体支持
- ✅ 多站点批量可视化

**新增功能:**
```python
plot_comprehensive_results()  # 生成 comprehensive 可视化
evaluate_model()              # 返回分特征指标
```

### 4. `REPRODUCTION_GUIDE.md` - 完整重写

**新增章节:**
- 📊 数据说明（详细解释特征计算）
- 🔍 结果解读（性能基准参考）
- 🎓 进阶用法（全量微调、LoRA等）
- ❓ 常见问题（8个新问题）

**改进内容:**
- 更详细的预处理流程说明
- 完整的代码示例
- 预期输出示例
- 故障排查指南

### 5. `run_all.sh` - 更新

**改进:**
- ✅ 检查多个CSV文件
- ✅ 显示数据处理进度
- ✅ 更友好的提示信息
- ✅ 自动显示评估指标摘要

## 🚀 快速开始

### 方式一：一键运行（推荐）

```bash
cd /home/user/Downloads/cai/moment-main/moment-main
chmod +x run_all.sh
./run_all.sh
```

### 方式二：分步执行

```bash
# Step 1: 数据预处理
python preprocess_data.py

# Step 2: 模型训练
python train_moment.py

# Step 3: 推理评估
python inference.py
```

## 📁 输出文件结构

```
moment-main/
├── dataset/                          # 输入数据
│   ├── 观测站小时交通量-9.csv
│   └── 观测站小时交通量-10.csv
│
├── processed_data/                   # 中间处理结果
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── moment_data/                      # MOMENT格式数据
│   └── dataset.npz
│
├── checkpoints/                      # 模型权重
│   └── best_model.pth
│
├── results/                          # 最终结果
│   ├── predictions.npz               # 预测值
│   ├── metrics.json                  # 评估指标
│   ├── training_history.json         # 训练历史
│   └── sample*_station*.png          # 可视化图表
│
├── station_mapping.json              # 站点映射
├── normalization_params.json         # 归一化参数
│
├── preprocess_data.py                # 数据预处理脚本 ⭐
├── train_moment.py                   # 训练脚本 ⭐
├── inference.py                      # 推理脚本 ⭐
├── run_all.sh                        # 一键运行脚本
└── REPRODUCTION_GUIDE.md             # 完整文档 ⭐
```

## 🎯 核心特性

### 数据处理
- ✅ 自动合并多个CSV文件
- ✅ 智能特征工程（计算非小客车流量）
- ✅ Z-score标准化
- ✅ 时间序列滑动窗口采样

### 模型训练
- ✅ Linear Probing（冻结编码器）
- ✅ 自动GPU检测
- ✅ 最佳模型保存
- ✅ 实时进度显示

### 评估可视化
- ✅ 多维度指标（MSE/RMSE/MAE/MAPE）
- ✅ 分特征性能分析
- ✅ 高质量对比图
- ✅ 高误差点标注

## 📊 数据流程图

```
原始CSV文件 (多个)
    ↓
[load_and_merge_datasets]
    ↓
合并的DataFrame (长表)
    ↓
[preprocess_traffic_data]
    ↓
提取4个特征 + 站点映射
    ↓
[pivot_and_normalize]
    ↓
宽表格式 + 标准化
    ↓
[split_and_save_data]
    ↓
训练/验证/测试集
    ↓
[create_moment_dataset_format]
    ↓
滑动窗口序列 (NPZ格式)
    ↓
[MOMENT Model Training]
    ↓
预测结果 + 评估指标
```

## ⚙️ 配置参数

### 数据预处理 (`preprocess_data.py`)
```python
seq_len = 512      # 输入序列长度（小时）
pred_len = 96      # 预测步长（小时）
train_ratio = 0.7  # 训练集比例
val_ratio = 0.15   # 验证集比例
test_ratio = 0.15  # 测试集比例
```

### 模型训练 (`train_moment.py`)
```python
BATCH_SIZE = 32        # 批次大小
EPOCHS = 10            # 训练轮数
LEARNING_RATE = 1e-3   # 学习率
```

### 推理评估 (`inference.py`)
```python
BATCH_SIZE = 32        # 批次大小
n_samples_to_plot = 3  # 可视化样本数
n_stations_to_plot = 2 # 可视化站点数
```

## 🔍 故障排查

### 问题1: 找不到CSV文件
```
错误: dataset/ 目录下没有找到CSV文件
解决: 确保CSV文件在 dataset/ 目录下，文件名包含"观测站小时交通量"
```

### 问题2: 显存不足
```
错误: CUDA out of memory
解决: 减小 BATCH_SIZE (32 → 16 → 8)
```

### 问题3: 中文显示乱码
```
现象: 图表中文显示为方框
解决: 安装中文字体或修改 matplotlib 配置
```

### 问题4: 样本数太少
```
现象: Total samples to generate: 0
解决: 减小 seq_len 或 pred_len，或增加数据文件
```

## 📈 性能优化建议

1. **使用GPU**: 训练速度提升10-50倍
2. **增加num_workers**: DataLoader并行加载数据
3. **混合精度训练**: 减少显存占用，加速训练
4. **梯度累积**: 模拟更大的batch size

## 📝 版本历史

- **v2.0** (2026-04-22): 适配新的观测站数据集，支持4特征提取
- **v1.0** (2026-04-22): 初始版本，基于预处理好的数据

## 🤝 技术支持

如有问题，请查阅：
1. `REPRODUCTION_GUIDE.md` - 完整使用文档
2. MOMENT官方文档: https://github.com/moment-timeseries-foundation-model/moment
3. 提交Issue时附上错误日志和数据集信息

---

**更新日期**: 2026-04-22  
**数据集**: 观测站小时交通量（多月份）  
**特征**: 4维（小客车上/下行 + 非小客车上/下行）