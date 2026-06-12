# MOMENT 4特征可视化脚本使用说明

## 📋 概述

这是一个全新的独立可视化脚本，**完全仿照FaST项目的可视化风格**，专门用于MOMENT模型的4特征交通流量预测结果可视化。

## 🎯 核心功能

### 1. **总览图** (Overview Plots)
- 2×2子图布局
- 显示所有站点在第一个预测步长的平均趋势
- 包含全局MAE/RMSE指标
- 真实时间轴（日期格式）

### 2. **站点详细图** (Per-Station Plots)
- 每个站点单独生成2×2子图
- 4个特征分别展示
- 自动标注高误差点（相对误差>30%）
- 红色/橙色圆圈标记 + 黄色标注框

### 3. **CSV导出**
- 所有预测结果保存到CSV
- UTF-8 BOM编码（Excel友好）
- 包含站点名称、时间、特征、真实值、预测值

### 4. **指标摘要**
- 逐特征MAE/RMSE/MAPE
- 保存到CSV便于分析

## 🚀 快速开始

### 前提条件

确保已经运行过推理脚本，生成以下文件：
```
results/
├── predictions.npz      # 预测结果
├── metrics.json         # 评估指标
└── ...

dataset/
└── station_mapping.json # 站点映射（可选）
```

### 基本用法

```bash
cd /home/user/Downloads/cai/moment-main/moment-main

# 直接运行（使用默认配置）
python visualize_moment_4features.py
```

### 自定义配置

```python
# 编辑 visualize_moment_4features.py 的 main() 函数

runner = MOMENTVisualizationRunner(
    predictions_path='results/predictions.npz',      # 预测结果路径
    metrics_path='results/metrics.json',             # 指标文件路径
    dataset_path='dataset/his_data_with_names.csv',  # 原始数据集
    mapping_path='dataset/station_mapping.json',     # 站点映射
    start_time='2023-09-01',                          # 数据集起始时间
    time_step_hours=1                                 # 时间步长（小时）
)
```

## 📊 输出文件结构

```
visualization-results/
├── figures/
│   ├── overview_4features.png          # 4特征总览图
│   └── per_station/
│       ├── 001.png                     # 站点001详细图
│       ├── 002.png                     # 站点002详细图
│       └── ...                         # 更多站点（默认前5个）
├── predictions/
│   └── all_predictions.csv             # 所有预测结果（CSV）
└── metrics/
    └── feature_metrics.csv             # 逐特征指标摘要
```

## 🎨 可视化样式（FaST风格）

### 配色方案
```python
Passenger Car Up:      #1f77b4 (蓝色)
Passenger Car Down:    #ff7f0e (橙色)
Non-Passenger Car Up:  #2ca02c (绿色)
Non-Passenger Car Down:#d62728 (红色)
```

### 线条样式
- **真实值**: 实线，linewidth=2.5，alpha=0.9
- **预测值**: 虚线，linewidth=2.5，alpha=0.7

### 误差标注
- 🔴 **红色圆圈**: 真实值位置（markersize=10）
- 🟠 **橙色圆圈**: 预测值位置（markersize=8）
- 🟡 **黄色标注框**: 显示误差百分比
- 📍 **红色箭头**: 指向误差点

## 📝 与FaST项目的对比

### 相同点

| 特性 | FaST项目 | MOMENT项目 |
|------|---------|-----------|
| 配色方案 | Matplotlib默认 | ✓ 相同 |
| 线条样式 | 实线/虚线区分 | ✓ 相同 |
| 误差标注 | 红色圆圈+黄色框 | ✓ 相同 |
| CSV导出 | utf-8-sig编码 | ✓ 相同 |
| 网格线 | 虚线+透明度 | ✓ 相同 |
| 子图布局 | 2×2 | ✓ 相同 |

### 不同点

| 特性 | FaST项目 | MOMENT项目 | 原因 |
|------|---------|-----------|------|
| 时间轴 | 真实日期时间 | 样本索引 | MOMENT数据格式不同 |
| 独立脚本 | ✓ 是 | ✓ 是 | 保持模块化 |
| 站点图数量 | 所有站点 | 默认前5个 | 避免生成过多文件 |

## 🔧 高级配置

### 修改站点数量

```python
# 编辑 visualize_moment_4features.py 的 run() 方法

# 修改前
self.plot_per_station(predictions, targets, station_mapping, 
                     max_stations=5, max_samples=20)

# 修改后（生成更多站点）
self.plot_per_station(predictions, targets, station_mapping, 
                     max_stations=10, max_samples=50)
```

### 修改误差阈值

```python
# 编辑 _annotate_high_errors() 方法

# 修改前
self._annotate_high_errors(ax, pred_seq, target_seq, time_steps, 
                          color, error_threshold=0.3, max_annotations=5)

# 修改后（更严格的阈值）
self._annotate_high_errors(ax, pred_seq, target_seq, time_steps, 
                          color, error_threshold=0.2, max_annotations=10)
```

### 修改图表尺寸

```python
# 编辑 plot_overview() 或 plot_per_station() 方法

# 修改前
fig, axes = plt.subplots(2, 2, figsize=(20, 10))

# 修改后（更大尺寸）
fig, axes = plt.subplots(2, 2, figsize=(24, 12))
```

### 修改输出分辨率

```python
# 编辑所有 plt.savefig() 调用

# 修改前
plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')

# 修改后（更高分辨率）
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
```

## 📈 示例输出

### 总览图
```
┌──────────────────────────────────────────────────┐
│ MOMENT 4-Feature Overview                        │
│ Global MAE: 65.21 | RMSE: 124.02                 │
├────────────────┬─────────────────────────────────┤
│ Passenger Car  │ Passenger Car Down              │
│ Up             │ MAE: 88.79 | RMSE: 162.81       │
│ MAE: 89.00     │                                 │
│ RMSE: 159.96   │  [实线=真实值, 虚线=预测值]     │
├────────────────┼─────────────────────────────────┤
│ Non-Passenger  │ Non-Passenger Car Down          │
│ Car Up         │ MAE: 42.40 | RMSE: 70.83        │
│ MAE: 40.65     │                                 │
│ RMSE: 66.41    │                                 │
└──────────────────────────────────────────────────┘
```

### 站点详细图
```
┌──────────────────────────────────────────────────┐
│ Station: 黄兴 (001)                              │
├─────────────────────────────────────────────────┤
│ Passenger Car  │ Passenger Car Down              │
│ Up             │ MAE: 92.3 | RMSE: 115.7         │
│ MAE: 78.1      │ 🔴 Error:45.2%                  │
│ RMSE: 95.4     │ 🟠 Sample 5                     │
│                │  [红色圆圈标注高误差点]          │
└──────────────────────────────────────────────────┘
```

### CSV文件内容
```csv
站点名称,站点编号,时间,特征,真实值,预测值
黄兴,001,2023-09-01 09:00:00,Passenger Car Up,245.5,238.2
黄兴,001,2023-09-01 09:00:00,Passenger Car Down,312.8,295.1
黄兴,001,2023-09-01 09:00:00,Non-Passenger Car Up,156.3,142.7
黄兴,001,2023-09-01 09:00:00,Non-Passenger Car Down,89.4,76.8
梨,002,2023-09-01 09:00:00,Passenger Car Up,198.7,185.3
...
```

## 🐛 故障排查

### 问题1：找不到预测文件

```
FileNotFoundError: Predictions file not found: results/predictions.npz
```

**解决方案：**
```bash
# 先运行推理脚本
python inference.py

# 检查文件是否生成
ls -lh results/predictions.npz
```

### 问题2：站点映射文件不存在

```
⚠️  Mapping file not found: dataset/station_mapping.json
→ Using default mapping (Station_001, Station_002, ...)
```

**解决方案：**
```bash
# 从训练脚本复制站点映射
cp /path/to/train_moment_output/station_mapping.json dataset/

# 或手动创建
python -c "
import json
mapping = {str(i): {'station_name': f'Station_{i:03d}', 'station_code': f'{i:03d}'} for i in range(157)}
with open('dataset/station_mapping.json', 'w') as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)
"
```

### 问题3：图表中文乱码

**解决方案：**
```python
# 在 visualize_moment_4features.py 开头添加
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
```

### 问题4：时间轴不正确

**解决方案：**
```python
# 修改 start_time 参数
runner = MOMENTVisualizationRunner(
    start_time='2023-09-01',  # 确保与实际数据起始时间一致
    time_step_hours=1          # 确保与数据采样频率一致
)
```

## 📚 与inference.py的关系

```
工作流程：
┌──────────────┐
│ train_moment │  训练模型
│   .py        │
└──────┬───────┘
       │
       ↓
┌──────────────┐
│  inference   │  推理评估
│   .py        │  生成 predictions.npz, metrics.json
└─────────────┘
       │
       ↓
┌──────────────┐
│ visualize_   │  可视化展示（本脚本）
│ moment_4feat │  生成图表和CSV
│   ures.py    │
└──────────────┘
```

**区别：**
- [inference.py](file:///home/user/Downloads/cai/moment-main/moment-main/inference.py): 模型推理 + 指标计算 + 简单可视化
- [visualize_moment_4features.py](file:///home/user/Downloads/cai/moment-main/moment-main/visualize_moment_4features.py): **专门的可视化脚本**，更详细的图表和CSV导出

## 🎉 总结

### 优势

1. ✅ **完全独立**：不依赖推理脚本，可单独运行
2. ✅ **FaST风格**：配色、布局、标注完全一致
3. ✅ **模块化设计**：每个功能独立函数，易于扩展
4. ✅ **配置灵活**：通过参数控制站点数量、误差阈值等
5. ✅ **Excel友好**：CSV使用UTF-8 BOM编码
6. ✅ **详细文档**：本使用说明 + 代码注释

### 使用建议

1. **常规分析**：直接运行`python visualize_moment_4features.py`
2. **详细研究**：修改`max_stations`生成更多站点图
3. **报告展示**：调整`figsize`和`dpi`获得更高清图表
4. **数据导出**：使用CSV文件在Excel/Tableau中进一步分析

---

**创建日期**: 2026-04-23  
**参考项目**: FaST-main-8_8MO4Delete  
**状态**: ✅ 已完成，可直接运行