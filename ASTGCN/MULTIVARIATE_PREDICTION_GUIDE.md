# ASTGCN 多变量预测适配说明

## ⚠️ 重要说明

### 当前实现的限制

原始 **ASTGCN** 模型设计为**单变量时间序列预测**，即：
- **输入**: 多站点 × 单特征 × 时间步
- **输出**: 多站点 × 单时间步预测

但你的需求是**4个特征同时预测**：
1. 小客车上行
2. 小客车下行
3. 非小客车上行
4. 非小客车下行

---

## 🔧 当前解决方案

### 方案A: 特征平均值聚合（已实现）

**策略**: 将4个特征的平均值作为综合流量指标进行预测

**优点**:
- ✅ 无需修改模型架构
- ✅ 快速实现和训练
- ✅ 捕捉整体流量趋势

**缺点**:
- ❌ 丢失各特征的独立信息
- ❌ 无法分别评估每个特征的预测精度
- ❌ 如果某些特征量级差异大，平均值可能被主导

**适用场景**: 
- 关注整体交通流量趋势
- 不需要细粒度的车型分类预测

---

## 🚀 推荐方案B: Channel Independence（独立通道）

**策略**: 为每个特征训练独立的 ASTGCN 模型

**实现步骤**:

### 1. 数据预处理（为每个特征单独处理）

```python
# 修改 preprocess_highway_data.py
FEATURES = [
    ('passenger_up', 0),      # 小客车上行
    ('passenger_down', 1),    # 小客车下行
    ('non_passenger_up', 2),  # 非小客车上行
    ('non_passenger_down', 3) # 非小客车下行
]

for feat_name, feat_idx in FEATURES:
    # 提取单个特征
    single_feat_data = data_tensor[:, :, feat_idx:feat_idx+1]  # (T, N, 1)
    
    # 生成滑动窗口并保存
    samples = generate_sliding_windows(single_feat_data, 8, 8)
    save_to_npz(f'./dataset/{feat_name}', samples)
```

### 2. 训练4个独立模型

```bash
# 为每个特征创建配置文件
cp configurations/highway_traffic.conf configurations/passenger_up.conf
sed -i 's/dataset_name = highway_traffic/dataset_name = passenger_up/' configurations/passenger_up.conf

# 分别训练
python train_ASTGCN_r.py --config configurations/passenger_up.conf
python train_ASTGCN_r.py --config configurations/passenger_down.conf
python train_ASTGCN_r.py --config configurations/non_passenger_up.conf
python train_ASTGCN_r.py --config configurations/non_passenger_down.conf
```

**优点**:
- ✅ 每个特征独立优化
- ✅ 可以分别评估每个特征的预测性能
- ✅ 符合 TimesFM/Lag-Llama 等现代模型的 Channel Independence 理念

**缺点**:
- ❌ 需要训练4次，时间成本增加
- ❌ 忽略了特征间的相关性

---

## 🎯 推荐方案C: 修改模型支持多变量输出（高级）

**策略**: 修改 ASTGCN 的输出层以支持多变量预测

### 修改步骤

#### 1. 修改 `model/ASTGCN_r.py`

```python
class ASTGCN_submodule(nn.Module):
    def __init__(self, ..., num_for_predict, len_input, num_of_vertices, num_features=4):
        super(ASTGCN_submodule, self).__init__()
        
        
        # 修改最终卷积层以输出多变量
        self.final_conv = nn.Conv2d(
            int(len_input/time_strides), 
            num_for_predict * num_features,  # 输出 T_out * F
            kernel_size=(1, nb_time_filter)
        )
        self.num_features = num_features
        self.num_for_predict = num_for_predict
    
    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, F_out, T_out)
        '''
        for block in self.BlockList:
            x = block(x)
        
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1]
        # (b, c_out*T_out*F, N, 1) -> (b, N, c_out*T_out*F)
        
        # 重塑为 (B, N, F, T_out)
        batch_size, num_nodes, _ = output.shape
        output = output.view(batch_size, num_nodes, self.num_features, self.num_for_predict)
        
        return output
```

#### 2. 修改 `lib/utils.py` 中的评估函数

```python
def predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, ...):
    # 修改目标张量形状以支持多变量
    # data_target_tensor: (B, N, F, T_out)
    
    # 计算每个特征的独立指标
    for feat_idx in range(num_features):
        feat_pred = prediction[:, :, feat_idx, :]
        feat_true = data_target_tensor[:, :, feat_idx, :]
        
        mae = mean_absolute_error(feat_true, feat_pred)
        rmse = mean_squared_error(feat_true, feat_pred) ** 0.5
        
        print(f'Feature {feat_idx} - MAE: {mae:.2f}, RMSE: {rmse:.2f}')
```

#### 3. 修改数据预处理

```python
# 保持4个特征完整，不取平均值
y_final = all_targets.transpose(0, 2, 3, 1)  # (B, N, F, T_out)

np.savez_compressed(
    save_path,
    x=x_final,      # (B, N, F, T_in)
    y=y_final,      # (B, N, F, T_out) - 多变量目标
    mean=mean,
    std=std
)
```

**优点**:
- ✅ 真正的多变量预测
- ✅ 捕捉特征间的相关性
- ✅ 一次训练获得所有特征的预测

**缺点**:
- ❌ 需要深入修改模型代码
- ❌ 可能需要调整损失函数（如加权多任务学习）
- ❌ 调试复杂度增加

---

## 📊 方案对比

| 特性 | 方案A (平均值) | 方案B (独立通道) | 方案C (多变量输出) |
|------|---------------|-----------------|-------------------|
| 实现难度 | ⭐ 简单 | ⭐⭐ 中等 | ⭐⭐⭐ 复杂 |
| 训练时间 | ⭐ 快 | ⭐⭐⭐ 慢 (4x) | ⭐⭐ 中等 |
| 预测精度 | ⭐⭐ 一般 | ⭐⭐⭐ 好 | ⭐⭐⭐⭐ 最好 |
| 特征独立性 | ❌ 丢失 | ✅ 完全独立 | ⚠️ 部分相关 |
| 代码修改量 | 少 | 少 | 多 |

---

## 💡 建议

### 短期方案（快速验证）
使用**方案A**（当前已实现），快速验证 ASTGCN 在你的数据集上的基本效果。

### 中期方案（生产环境）
采用**方案B**（Channel Independence），这是目前时序预测的主流做法，被 TimesFM、Lag-Llama 等采用。

### 长期方案（研究探索）
实现**方案C**（多变量输出），如果你需要深入研究特征间的时空相关性。

---

## 🔍 下一步行动

### 如果使用方案A（当前）

```bash
# 1. 重新运行数据预处理
python preprocess_highway_data.py --dataset_dir ./dataset --input_len 8 --output_len 8

# 2. 开始训练
python train_ASTGCN_r.py --config configurations/highway_traffic.conf
```

### 如果切换到方案B

我可以为你创建自动化脚本，一次性处理4个特征的训练流程。请告诉我是否需要。

### 如果尝试方案C

我需要修改多个文件（模型、数据加载、评估），这是一个较大的工程。建议先完成方案A或B的验证后再考虑。

---

**你希望我帮你实现哪个方案？**
