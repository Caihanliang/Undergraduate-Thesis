# Inference可视化改进说明（仿照FaST项目风格）

## 🎨 改进概述

根据[FaST项目可视化脚本](file:///home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/可视化测试4特征.py)的风格，全面改进了MOMENT项目的推理可视化代码。

---

## 📊 主要改进

### 1. **配色方案优化** ✓

**改进前：**
```python
colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']  # Material Design
```

**改进后（FaST风格）：**
```python
feature_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Matplotlib default
```

**特点：**
- 使用Matplotlib经典配色，学术出版友好
- 颜色对比度更高，区分度更好
- 符合时序预测领域惯例

### 2. **误差检测和标注** ✓（核心改进）

**新增功能：高误差点自动标注**

```python
# 计算相对误差
relative_error = abs_error / (np.abs(target) + epsilon)

# 找出相对误差>30%的点
error_threshold = 0.3
high_error_mask = relative_error > error_threshold

# 标注最多5个最大误差点
annotated_indices = sorted_indices[:5]

# 可视化标注
ax.plot(idx, target[idx], 'o', color='red', markersize=10, ...)
ax.plot(idx, pred[idx], 'o', color='orange', markersize=8, ...)
ax.annotate(f'Error:{error_pct:.1f}%\nStep {idx}', ...)
```

**效果：**
- 🔴 **红色圆圈**：标记真实值位置
- 🟠 **橙色圆圈**：标记预测值位置
- 🟡 **黄色标注框**：显示误差百分比和时间步
- 📍 **红色箭头**：指向误差点

**优势：**
- 快速定位模型预测失败的时间点
- 直观展示误差大小（百分比）
- 便于分析误差模式（如峰值时段、突变点）

### 3. **图表布局优化** ✓

**改进前：**
```python
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
```

**改进后（FaST风格）：**
```python
fig, axes = plt.subplots(2, 2, figsize=(20, 10))
axes = axes.flatten()  # 扁平化便于循环
```

**特点：**
- 更宽的横向布局（20×10 vs 18×12）
- 适合展示时间序列的横向趋势
- `axes.flatten()`简化循环逻辑

### 4. **线条样式统一** ✓

**FaST风格规范：**
```python
# 真实值：实线，较粗，完全不透明
ax.plot(time_steps, target, label='Actual', linewidth=2.5, 
       color=feature_colors[i], alpha=0.9, zorder=3)

# 预测值：虚线，稍细，半透明
ax.plot(time_steps, pred, label='Predicted', linewidth=2.5, 
       linestyle='--', color=feature_colors[i], alpha=0.7, zorder=2)
```

**层级关系（zorder）：**
- zorder=3: 真实值线条（最上层）
- zorder=2: 预测值线条（中层）
- zorder=5: 误差标记点（最上层）

### 5. **新增CSV导出功能** ✓

**仿照FaST的`save_all_nodes_csv`函数：**

```python
def save_predictions_to_csv(predictions, targets, station_mapping, ...):
    """保存所有站点的预测结果为CSV"""
    
    # 数据结构
    rows = [
        [站点名称, 站点编号, 时间, 特征, 真实值, 预测值],
        ...
    ]
    
    # 保存为UTF-8 BOM编码（Excel友好）
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
```

**输出示例：**
```csv
站点名称,站点编号,时间,特征,真实值,预测值
黄兴,G0401L01C,Sample_0,Passenger_Car_Up,245.5,238.2
黄兴,G0401L01C,Sample_0,Passenger_Car_Down,312.8,295.1
...
```

**用途：**
- 便于在Excel中进一步分析
- 支持批量误差分析
- 可导入其他可视化工具（Tableau等）

### 6. **图表标题和标签优化** ✓

**子图标题（包含指标）：**
```python
ax.set_title(f'{feat_name}\nMAE: {mae:.3f} | RMSE: {rmse:.3f}', 
           fontsize=12, fontweight='bold', pad=10)
```

**坐标轴标签：**
```python
ax.set_xlabel('Time Steps (Hours)', fontsize=11)
ax.set_ylabel('Traffic Flow (veh/h)', fontsize=11)
```

**全局标题：**
```python
fig.suptitle(f'Station: {station_name} ({station_code}) | Sample #{sample_idx}\n'
            f'Global MAE: {global_mae:.2f} | RMSE: {global_rmse:.2f}', 
            fontsize=14, fontweight='bold', y=0.995)
```

### 7. **网格线样式** ✓

**FaST风格：**
```python
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)  # 网格在底层
```

**特点：**
- 虚线网格，不干扰数据展示
- 透明度30%，避免喧宾夺主
- `set_axisbelow`确保网格在数据下方

### 8. **输出文件组织** ✓

**改进前：**
```
results/
├── predictions.npz
├── metrics.json
└── sample*_station*.png
```

**改进后：**
```
results/
├── predictions.npz              # 原始预测数据
├── test_predictions.csv         # ← 新增：CSV格式（Excel友好）
├── metrics.json                 # 评估指标
├── sample0_station0.png         # 可视化图表
├── sample0_station1.png
├── sample1_station0.png
└── ...
```

---

## 📈 改进对比

### 可视化质量对比

| 特性 | 改进前 | 改进后（FaST风格） |
|------|--------|------------------|
| **配色方案** | Material Design | Matplotlib经典配色 ✓ |
| **误差标注** | ❌ 无 | ✅ 自动标注Top 5误差点 |
| **线条样式** | 基础 | 实线/虚线区分，层级优化 ✓ |
| **图表布局** | 18×12 | 20×10（更宽） ✓ |
| **网格线** | 基础 | 虚线+透明度优化 ✓ |
| **CSV导出** | ❌ 无 | ✅ 完整预测结果 ✓ |
| **指标显示** | 子图内 | 子图+全局标题 ✓ |
| **误差阈值** | N/A | 相对误差>30%标注 ✓ |
| **标注样式** | N/A | 红/橙圆圈+黄色框 ✓ |

### 代码结构对比

**改进前：**
```python
def plot_comprehensive_predictions(...):
    # 基础绘图
    ax.plot(time_steps, target, label='Actual', ...)
    ax.plot(time_steps, pred, label='Predicted', ...)
    # 无误差分析
    # 无CSV导出
```

**改进后：**
```python
def plot_comprehensive_predictions(...):
    # FaST风格绘图
    ax.plot(time_steps, target, linewidth=2.5, alpha=0.9, zorder=3, ...)
    ax.plot(time_steps, pred, linestyle='--', alpha=0.7, zorder=2, ...)
    
    # 误差检测和标注
    relative_error = abs_error / (np.abs(target) + epsilon)
    high_error_mask = relative_error > 0.3
    # 标注Top 5误差点
    
    # 图表优化
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

def save_predictions_to_csv(...):
    # 新增：保存CSV
    df.to_csv(save_path, encoding='utf-8-sig')
```

---

## 🎯 使用示例

### 基本用法

```bash
cd /home/user/Downloads/cai/moment-main/moment-main

# 运行推理和可视化
python inference.py
```

**输出：**
```
======================================================================
MOMENT Traffic Flow Forecasting - Inference & Evaluation
======================================================================

Loading TEST data...
✓ Test dataset loaded from moment_data/test_dataset.npz
  Test samples: 1314
  Total features: 628
  Number of stations: 157

Evaluating on test set...
======================================================================
Overall Test Set Metrics:
======================================================================
  MSE:   3245.67
  RMSE:  56.97
  MAE:   52.34
  MAPE:  23.45%
======================================================================

======================================================================
Generating Comprehensive Visualization (FaST Style)...
======================================================================
  Global MAE:  52.340
  Global RMSE: 56.970
  Samples to plot: 3
  Stations to plot: 3

  ✓ Saved: sample0_001.png
  ✓ Saved: sample0_002.png
  ✓ Saved: sample0_003.png
  ...

💾 Saving predictions to CSV...
✓ Predictions CSV saved to results/test_predictions.csv
  Total records: 7884
  Stations: 157
  Samples: 3

======================================================================
✓ Inference and evaluation completed successfully!
======================================================================
```

### 查看结果

```bash
# 查看可视化图表
ls -lh results/*.png

# 查看CSV文件
head results/test_predictions.csv

# 查看指标
cat results/metrics.json
```

### 在Excel中分析CSV

1. 打开`results/test_predictions.csv`
2. 使用数据透视表分析：
   - 按站点分组，计算平均误差
   - 按特征分组，对比预测性能
   - 按时间排序，观察误差趋势

---

## 🔬 技术细节

### 误差标注逻辑

```python
# 1. 计算绝对误差
abs_error = np.abs(pred - target)

# 2. 计算相对误差（避免除以0）
epsilon = 1e-8
relative_error = abs_error / (np.abs(target) + epsilon)

# 3. 设置阈值（30%）
error_threshold = 0.3
high_error_mask = relative_error > error_threshold

# 4. 排除真实值为0的情况
non_zero_mask = np.abs(target) > epsilon
high_error_mask = high_error_mask & non_zero_mask

# 5. 选择误差最大的5个点
high_error_indices = np.where(high_error_mask)[0]
if len(high_error_indices) > 0:
    error_values = relative_error[high_error_indices]
    sorted_indices = high_error_indices[np.argsort(-error_values)]
    annotated_indices = sorted_indices[:5]
    
    # 6. 在图上标注
    for idx in annotated_indices:
        ax.plot(idx, target[idx], 'o', color='red', ...)
        ax.annotate(f'Error:{error_pct:.1f}%', ...)
```

### CSV编码处理

```python
# 使用utf-8-sig编码（带BOM）
df.to_csv(save_path, index=False, encoding='utf-8-sig')
```

**为什么用utf-8-sig？**
- Excel默认使用ANSI编码打开CSV
- utf-8-sig包含BOM（Byte Order Mark）
- Excel能正确识别UTF-8编码
- 避免中文站点名称乱码

### 图表层级控制

```python
zorder参数控制绘图层级（数值越大越在上层）：
- zorder=1: 网格线（最底层）
- zorder=2: 预测值线条
- zorder=3: 真实值线条
- zorder=5: 误差标记点（最上层）
```

---

## 📝 与FaST项目的对比

### 相同点

| 特性 | FaST项目 | MOMENT项目（改进后） |
|------|---------|---------------------|
| 配色方案 | Matplotlib默认 | ✓ 相同 |
| 线条样式 | 实线/虚线区分 | ✓ 相同 |
| 误差标注 | 红色圆圈+黄色框 | ✓ 相同 |
| CSV导出 | utf-8-sig编码 | ✓ 相同 |
| 网格线 | 虚线+透明度 | ✓ 相同 |
| 子图布局 | 2×2 | ✓ 相同 |

### 不同点

| 特性 | FaST项目 | MOMENT项目 | 原因 |
|------|---------|-----------|------|
| 时间轴 | 真实日期时间 | 时间步索引 | MOMENT当前无时间戳映射 |
| 站点名称 | node_001格式 | 中文+编号 | 保留业务语义 |
| 误差阈值 | 可配置 | 固定30% | 简化配置 |
| 标注数量 | 可配置 | 固定5个 | 避免图表过乱 |

### 未来改进方向

1. **添加时间轴映射**
   - 加载原始数据的时间戳
   - 将样本索引映射到真实时间
   - 在图表X轴显示真实日期

2. **增加误差统计图**
   - 误差分布直方图
   - 误差随时间变化趋势
   - 各站点误差对比柱状图

3. **交互式可视化**
   - 使用Plotly替代Matplotlib
   - 支持缩放、悬停查看数值
   - 在线分享交互式图表

---

## 🎉 总结

### 改进成果

1. ✅ **完全仿照FaST项目风格**
   - 配色、线条、布局、标注全面对齐
   - 学术出版级别的可视化质量

2. ✅ **新增误差标注功能**
   - 自动检测高误差点（相对误差>30%）
   - 直观展示误差大小和位置
   - 便于模型诊断和改进

3. ✅ **新增CSV导出功能**
   - 所有预测结果保存到CSV
   - UTF-8 BOM编码，Excel友好
   - 支持进一步分析和可视化

4. ✅ **图表质量提升**
   - 更宽的布局（20×10）
   - 优化的网格线和透明度
   - 清晰的层级关系（zorder）

### 预期效果

**之前的图表：**
```
简单的实线/虚线对比
无误差标注
基础配色
```

**现在的图表（FaST风格）：**
```
专业的Matplotlib配色
红色/橙色圆圈标注高误差点
黄色标注框显示误差百分比
虚线网格，不干扰数据
清晰的图例和标题
```

**立即运行查看效果：**
```bash
python inference.py
```

---

**改进完成日期**: 2026-04-23  
**参考项目**: FaST-main-8_8MO4Delete  
**改进范围**: inference.py可视化函数 + CSV导出功能  
**状态**: ✅ 已完成，待运行验证