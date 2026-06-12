# 🚀 TimesFM 8输入8输出 - 快速启动指南

## ⚡ 3步完成复现

### 第1步：环境配置（仅首次运行）

```bash
cd /home/user/Downloads/cai/timesfm-master

# 运行环境配置脚本
chmod +x setup_env.sh
./setup_env.sh
```

**预计时间**: 5-10 分钟（下载依赖）

---

### 第2步：一键运行完整流程

```bash
# 运行一键脚本（8输入8输出）
chmod +x run_complete_pipeline.sh
./run_complete_pipeline.sh
```

**预计时间**: 
- 数据预处理: 1-2 分钟
- 模型训练: 1-2 小时（10 epochs）
- 预测评估: 5-10 分钟

**总计**: 约 1.5-2.5 小时

---

### 第3步：查看结果

```bash
# 查看训练日志
tail -f training.log

# 查看预测结果
ls -lh prediction_results_8x8/

# 打开可视化图表（如果有图形界面）
xdg-open prediction_results_8x8/*.png
```

---

## 📊 预期输出示例

### 训练日志
```
============================================================
TimesFM Fine-tuning Training - 8-Input 8-Output Configuration
============================================================
Input window: 8 hours
Prediction window: 8 hours
============================================================
Using device: cuda
Loading model: google/timesfm-2.5-200m-transformers
Actual context_len: 8, horizon_len: 8
Configuration: 8-input 8-output

Trainable parameters: 1,423,872 (0.6% of total)

Epoch 1/10 — Train Loss: 0.2345, Val Loss: 0.1987
Epoch 2/10 — Train Loss: 0.1876, Val Loss: 0.1654
...
Epoch 10/10 — Train Loss: 0.1234, Val Loss: 0.1123

Training completed! Best validation loss: 0.1123
```

### 评估结果
```
============================================================
Average Zero-shot MAE: 45.67
Average Fine-tuned MAE: 32.45
Improvement: 28.9%
============================================================
```

### 预测结果文件
```
prediction_results_8x8/
├── G0401L010430121_passenger_car_up.png
├── G0401L010430121_passenger_car_down.png
├── G0401L010430121_non_passenger_car_up.png
├── G0401L010430121_non_passenger_car_down.png
├── ... (更多站点)
└── prediction_results.json
```

---

## 🔍 故障排查

### 问题1：虚拟环境不存在
```bash
✗ Virtual environment not found
```
**解决**: 先运行 `./setup_env.sh`

### 问题2：CUDA out of memory
```bash
RuntimeError: CUDA out of memory
```
**解决**: 减小 batch size
```bash
python finetune_timesfm.py --batch_size 16
```

### 问题3：模型下载超时
```bash
Connection timeout when downloading model
```
**解决**: 确保已设置镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题4：数据预处理失败
```bash
ValueError: No series long enough
```
**解决**: 检查原始数据是否有足够的时间点
- 8输入8输出仅需 16 个时间点
- 应该很少有站点不满足

---

## 💡 常用命令

### 仅重新训练（跳过数据处理）
```bash
python finetune_timesfm.py \
    --context_len 8 \
    --horizon_len 8 \
    --epochs 10 \
    --output_dir checkpoints/traffic-lora-8x8
```

### 仅预测（使用已有模型）
```bash
python predict_and_visualize.py \
    --adapter_path checkpoints/traffic-lora-8x8 \
    --output_dir prediction_results_8x8
```

### 查看训练历史
```bash
cat checkpoints/traffic-lora-8x8/training_history.json | python -m json.tool
```

### 监控GPU使用
```bash
watch -n 1 nvidia-smi
```

---

## 📈 性能基准

### 硬件要求
- **最低**: CPU（慢，不推荐）
- **推荐**: GPU with 8GB+ VRAM (RTX 3060, RTX 4060, etc.)
- **理想**: GPU with 16GB+ VRAM (RTX 4080, A100, etc.)

### 训练速度参考
| GPU | Batch Size | 每 Epoch | 10 Epochs |
|-----|-----------|----------|-----------|
| RTX 3060 (12GB) | 32 | ~8 min | ~80 min |
| RTX 4080 (16GB) | 64 | ~4 min | ~40 min |
| A100 (40GB) | 128 | ~2 min | ~20 min |

### 推理速度
- **单样本**: < 10ms
- **批量 (batch=32)**: ~100 样本/秒

---

## 🎯 下一步优化

### 1. 调整超参数
```bash
# 更精确的预测
python finetune_timesfm.py \
    --epochs 20 \
    --num_samples 10000 \
    --lr 5e-5
```

### 2. 尝试不同配置
```bash
# 24小时预测
python finetune_timesfm.py \
    --context_len 24 \
    --horizon_len 24 \
    --output_dir checkpoints/traffic-lora-24x24
```

### 3. 分析预测误差
```python
import json
with open('prediction_results_8x8/prediction_results.json') as f:
    results = json.load(f)
    
# 找出误差最大的站点
results.sort(key=lambda x: x['mae'], reverse=True)
print("Top 5 worst predictions:")
for r in results[:5]:
    print(f"{r['station_id']}_{r['feature_name']}: MAE={r['mae']:.2f}")
```

---

## 📞 获取帮助

### 查看详细文档
- [README_TRAFFIC.md](README_TRAFFIC.md) - 项目总览
- [CONFIG_8x8.md](CONFIG_8x8.md) - 配置说明
- [TRAFFIC_PREDICTION_GUIDE.md](TRAFFIC_PREDICTION_GUIDE.md) - 完整指南

### 检查日志
```bash
# 实时查看训练日志
tail -f training.log

# 查看错误信息
grep "ERROR" training.log
```

---

## ✅ 验证清单

运行前确认：
- [ ] Python 3.10+ 已安装
- [ ] GPU 可用（`nvidia-smi` 能显示）
- [ ] 磁盘空间充足（至少 10GB）
- [ ] 数据集存在：`dataset/观测站小时交通量-9.csv`
- [ ] 网络连接正常（下载模型需要）

运行后确认：
- [ ] `dataset/preprocessed/` 目录已创建
- [ ] `checkpoints/traffic-lora-8x8/` 包含模型文件
- [ ] `prediction_results_8x8/` 包含 PNG 图表
- [ ] `training.log` 显示训练完成
- [ ] 最佳验证损失 < 0.2（合理范围）

---

**准备好了吗？开始吧！** 🚀

```bash
./run_complete_pipeline.sh
```
