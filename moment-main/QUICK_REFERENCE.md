# MOMENT 项目快速参考卡片

## 🚀 一键运行
```bash
cd /home/user/Downloads/cai/moment-main/moment-main
./run_all.sh
```

## 📊 数据集
- **位置**: `dataset/观测站小时交通量-*.csv`
- **特征**: 
  1. 小客车上行
  2. 小客车下行  
  3. 非小客车上行 (= 汽车自然数 - 小客车)
  4. 非小客车下行 (= 汽车自然数 - 小客车)

## 🔧 核心命令

### 数据预处理
```bash
python preprocess_data.py
```
输出:
- `moment_data/dataset.npz` - MOMENT格式数据
- `station_mapping.json` - 站点映射
- `normalization_params.json` - 归一化参数

### 模型训练
```bash
python train_moment.py
```
输出:
- `checkpoints/best_model.pth` - 最佳模型

### 推理评估
```bash
python inference.py
```
输出:
- `results/metrics.json` - 评估指标
- `results/predictions.npz` - 预测结果
- `results/sample*_station*.png` - 可视化图表

## ⚙️ 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| SEQ_LEN | 512 | 输入序列长度(小时) |
| PRED_LEN | 96 | 预测步长(小时) |
| BATCH_SIZE | 32 | 批次大小 |
| EPOCHS | 10 | 训练轮数 |
| LEARNING_RATE | 1e-3 | 学习率 |

## 📁 文件结构
```
moment-main/
├── dataset/              # 输入数据 (CSV)
├── processed_data/       # 中间数据
├── moment_data/          # NPZ格式数据
├── checkpoints/          # 模型权重
├── results/              # 最终结果
├── preprocess_data.py    # 预处理 ⭐
├── train_moment.py       # 训练 ⭐
├── inference.py          # 推理 ⭐
└── REPRODUCTION_GUIDE.md # 完整文档
```

## 🎯 评估指标
- **MSE**: 均方误差
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **MAPE**: 平均绝对百分比误差

**性能基准:**
- 优秀: MAPE < 10%
- 良好: MAPE 10-15%
- 一般: MAPE 15-20%

## ❓ 常见问题

**Q: 显存不足?**  
A: 减小 BATCH_SIZE (32 → 16 → 8)

**Q: 训练太慢?**  
A: 使用GPU，减少EPOCHS到5先验证

**Q: 样本数太少?**  
A: 减小 SEQ_LEN (512 → 256) 或添加更多CSV文件

**Q: 中文乱码?**  
A: 安装中文字体或修改matplotlib配置

## 📖 详细文档
- 完整指南: `REPRODUCTION_GUIDE.md`
- 更新说明: `DATA_UPDATE_NOTES.md`
- 官方文档: https://github.com/moment-timeseries-foundation-model/moment

---
**提示**: 首次运行会自动下载MOMENT模型 (~3GB)