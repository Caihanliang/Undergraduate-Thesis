# ASTGCN 高速公路交通流量预测使用指南

## 📋 项目概述

本项目使用 **ASTGCN** (Attention Based Spatial-Temporal Graph Convolutional Networks) 模型进行高速公路交通流量预测。

### 预测任务
- **站点数量**: 98个观测站
- **预测特征**: 4维
  1. 小客车上行
  2. 小客车下行
  3. 非小客车上行 (汽车自然数 - 小客车)
  4. 非小客车下行 (汽车自然数 - 小客车)
- **时序配置**: 8小时输入 → 8小时输出
- **数据粒度**: 小时级

---

## 🚀 快速开始

### 步骤1: 环境准备

```bash
# 安装依赖
pip install torch numpy pandas scikit-learn scipy configparser tensorboardX

# 验证PyTorch GPU支持
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 步骤2: 数据预处理

```bash
cd /home/user/Downloads/cai/ASTGCN

# 运行数据预处理脚本
python preprocess_highway_data.py \
    --dataset_dir ./dataset \
    --input_len 8 \
    --output_len 8 \
    --adj_method distance
```

**预处理脚本会生成以下文件**:
- `dataset/train.npz` - 训练集 (60%)
- `dataset/val.npz` - 验证集 (20%)
- `dataset/test.npz` - 测试集 (20%)
- `dataset/adj_matrix.csv` - 邻接矩阵 (98×98)
- `dataset/station_mapping.csv` - 站点映射文件
- `configurations/highway_traffic.conf` - 配置文件

### 步骤3: 模型训练

```bash
# 使用配置文件启动训练
python train_ASTGCN_r.py --config configurations/highway_traffic.conf
```

**训练过程监控**:
- TensorBoard日志保存在 `experiments/highway_traffic/` 目录
- 模型checkpoint自动保存在最佳验证损失时
- 每1000步打印一次训练损失

### 步骤4: 查看训练结果

```bash
# 启动TensorBoard
tensorboard --logdir experiments/highway_traffic/

# 在浏览器中访问 http://localhost:6006
```

---

## 📊 数据处理流程详解

### 1. 数据加载与合并
```python
# 读取9月和10月数据
df_sep = pd.read_csv('dataset/观测站小时交通量-9.csv')
df_oct = pd.read_csv('dataset/观测站小时交通量-10.csv')
df = pd.concat([df_sep, df_oct])
```

### 2. 特征工程
```python
# 计算非小客车流量
df['非小客车'] = df['汽车自然数'] - df['小客车']

# 提取4个目标特征
- 小客车上行 (direction='上行', feature='小客车')
- 小客车下行 (direction='下行', feature='小客车')
- 非小客车上行 (direction='上行', feature='非小客车')
- 非小客车下行 (direction='下行', feature='非小客车')
```

### 3. 时空张量构建
```
原始数据形状: (T=1464小时, N=98站点, F=4特征)
滑动窗口采样后: 
  - 训练集: ~878样本
  - 验证集: ~293样本
  - 测试集: ~293样本

最终输入格式: (B, N=98, F=4, T=8)
最终输出格式: (B, N=98, T=8)
```

### 4. 归一化策略
- **方法**: Z-Score标准化
- **统计量**: 仅在训练集上计算均值和标准差
- **应用**: 训练/验证/测试集使用相同的归一化参数

---

## 🔧 关键配置说明

### 配置文件参数 (`highway_traffic.conf`)

```ini
[Data]
num_of_vertices = 98          # 站点数量
points_per_hour = 1           # 每小时1个数据点
num_for_predict = 8           # 预测未来8小时
len_input = 8                 # 输入历史8小时
in_channels = 4               # 4个特征通道

[Training]
nb_block = 2                  # ASTGCN块数量
K = 3                         # 切比雪夫多项式阶数
nb_chev_filter = 64           # 图卷积滤波器数量
nb_time_filter = 64           # 时间卷积滤波器数量
batch_size = 32               # 批次大小
epochs = 100                  # 训练轮数
learning_rate = 0.001         # 学习率
```

### 模型架构

```
ASTGCN Model:
├── ASTGCN_Block × 2
│   ├── Temporal Attention Layer
│   ├── Spatial Attention Layer
│   ├── Chebyshev Graph Conv (K=3)
│   └── Time Convolution (kernel=1×3)
└── Final Conv Layer (输出8小时预测)
```

---

## 📈 评估指标

训练完成后，模型会自动在测试集上计算以下指标：

- **MAE** (Mean Absolute Error) - 平均绝对误差
- **RMSE** (Root Mean Square Error) - 均方根误差
- **MAPE** (Mean Absolute Percentage Error) - 平均绝对百分比误差

**注意**: MAPE在低流量值时可能失真，建议结合MAE/RMSE综合评估。

---

## ⚠️ 常见问题

### Q1: CUDA内存不足
```bash
# 减小batch_size
# 修改 configurations/highway_traffic.conf
batch_size = 16  # 或更小
```

### Q2: 训练损失不下降
- 检查数据归一化是否正确
- 尝试降低学习率 (0.0005 或 0.0001)
- 增加训练轮数 (epochs = 200)

### Q3: 预测结果异常
- 验证邻接矩阵是否合理
- 检查是否有大量零值或缺失值
- 确认时间序列连续性

### Q4: 如何自定义邻接矩阵
```python
# 修改 preprocess_highway_data.py 中的 build_adjacency_matrix 函数
# 可选方法:
# - 'distance': 基于路线编号距离
# - 'correlation': 基于流量相关性
# - 自定义: 导入真实的地理距离数据
```

---

## 📁 输出文件结构

```
ASTGCN/
├── dataset/
│   ├── train.npz              # 训练集
│   ├── val.npz                # 验证集
│   ├── test.npz               # 测试集
│   ├── adj_matrix.csv         # 邻接矩阵
│   └── station_mapping.csv    # 站点映射
├── experiments/
│   └── highway_traffic/
│       └── astgcn_r_h1d0w0_channel4_0.001000/
│           ├── epoch_0.params      # 模型权重
│           ├── epoch_1.params
│           └── ...
│           ├── output_epoch_X_test.npz  # 测试结果
│           └── events.out.tfevents.*  # TensorBoard日志
└── configurations/
    └── highway_traffic.conf     # 配置文件
```

---

## 🔬 进阶用法

### 调整预测 horizon
```bash
# 修改为 12输入12输出
python preprocess_highway_data.py --input_len 12 --output_len 12

# 更新配置文件
sed -i 's/num_for_predict = 8/num_for_predict = 12/' configurations/highway_traffic.conf
sed -i 's/len_input = 8/len_input = 12/' configurations/highway_traffic.conf
```

### 使用不同的注意力机制
```python
# 修改 model/ASTGCN_r.py
# 可调整:
# - K (切比雪夫阶数): 控制空间感受野
# - nb_block: 控制网络深度
# - nb_chev_filter/nb_time_filter: 控制模型容量
```

### 多GPU训练
```python
# 修改 train_ASTGCN_r.py
DEVICE = torch.device('cuda:0')  # 指定GPU
# 或使用 DataParallel
net = nn.DataParallel(net)
```

---

## 📚 参考文献

```bibtex
@inproceedings{guo2019attention,
  title={Attention based spatial-temporal graph convolutional networks for traffic flow forecasting},
  author={Guo, Shengnan and Lin, Youfang and Feng, Ning and Song, Chao and Wan, Huaiyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={922--929},
  year={2019}
}
```

---

## 💡 下一步建议

1. **可视化预测结果**: 创建脚本来绘制真实值vs预测值曲线
2. **站点级别分析**: 按站点分别计算评估指标，识别难预测站点
3. **超参数调优**: 使用网格搜索优化学习率、batch_size等
4. **对比实验**: 与其他基线模型 (如LSTM、GRU、TCN) 对比

---

**最后更新**: 2024年
**维护者**: ASTGCN Team
