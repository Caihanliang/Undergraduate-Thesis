# CUDA/GPU 加速配置完整指南

## 🔍 当前问题诊断

你的系统显示：
```
Device: cpu
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12080)
```

**原因分析：**
- NVIDIA驱动版本：12080（对应CUDA 12.0）
- 当前PyTorch：可能编译自更新版本的CUDA
- 结果：CUDA初始化失败，回退到CPU

## ✅ 解决方案（按推荐顺序）

### 方案1：快速修复 - 安装兼容的PyTorch（推荐⭐）

**最简单的方法**，无需更新驱动：

```bash
cd /home/user/Downloads/cai/moment-main/moment-main
chmod +x quick_fix_cuda.sh
./quick_fix_cuda.sh
```

**这个脚本会：**
1. 卸载当前PyTorch
2. 安装CUDA 12.1版本的PyTorch（向后兼容你的12.0驱动）
3. 自动验证安装

**预期输出：**
```
CUDA available: True
GPU 0: NVIDIA GeForce XXXX
Speedup: 15.3x
✓ CUDA is properly configured and ready to use!
```

### 方案2：交互式配置脚本

提供三个选项的完整配置向导：

```bash
chmod +x setup_cuda.sh
./setup_cuda.sh
```

**可选方案：**
- **A**: 更新NVIDIA驱动（需要重启）
- **B**: 安装兼容的PyTorch（同方案1）
- **C**: 继续使用CPU（慢但可用）

### 方案3：手动安装（如果脚本失败）

```bash
# Step 1: 卸载当前PyTorch
pip uninstall -y torch torchvision torchaudio

# Step 2: 安装CUDA 12.1版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 3: 验证
python check_cuda.py
```

### 方案4：更新NVIDIA驱动（最彻底）

```bash
# Ubuntu/Debian系统
sudo apt update
sudo ubuntu-drivers autoinstall
sudo reboot

# 重启后验证
nvidia-smi
python check_cuda.py
```

**或从官网下载：**
访问 http://www.nvidia.com/Download/index.aspx
根据你的GPU型号下载最新驱动

## 🚀 验证CUDA是否启用

### 方法1：运行检测脚本
```bash
python check_cuda.py
```

查看输出中的：
- `CUDA available: True` ✓
- `Device: cuda` ✓
- `Speedup: XXx` （越大越好）

### 方法2：训练时观察
```bash
python train_moment.py
```

第一行应显示：
```
Configuration:
  Device: cuda    ← 成功！
```

而不是：
```
  Device: cpu     ← 仍在用CPU
```

### 方法3：单行命令
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"
```

## ⚡ 性能对比

| 配置 | 训练时间（10 epochs） | 相对速度 |
|------|---------------------|---------|
| CPU | ~30-60分钟 | 1x |
| GPU (RTX 3090) | ~2-5分钟 | 15-30x |
| GPU (A100) | ~1-2分钟 | 30-60x |

**对于你的8-8配置：**
- CPU: 约5-10分钟
- GPU: 约30秒-2分钟

## 🔧 如果仍然无法使用GPU

### 检查清单

1. **确认有NVIDIA GPU**
   ```bash
   lspci | grep -i nvidia
   ```

2. **检查驱动是否正确加载**
   ```bash
   nvidia-smi
   ```
   如果报错，说明驱动未正确安装

3. **检查PyTorch版本**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```
   应该包含 `+cu121` 后缀

4. **检查CUDA版本匹配**
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```
   应该显示 `12.1`

### 常见问题

**Q1: 安装后仍显示 `CUDA available: False`**

A: 尝试重启Python环境或重新打开终端

**Q2: 出现 `libcudart.so.xx: cannot open shared object file`**

A: CUDA库路径问题，执行：
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Q3: 显存不足**

A: 减小batch size：
```python
# 在 train_moment.py 中修改
BATCH_SIZE = 8  # 从32改为8
```

**Q4: 多GPU如何指定？**

A: 设置环境变量：
```bash
export CUDA_VISIBLE_DEVICES=0  # 只使用第0个GPU
python train_moment.py
```

## 💡 临时方案：优化CPU训练

如果暂时无法使用GPU，可以优化CPU训练速度：

```python
# 在 train_moment.py 中修改
EPOCHS = 3           # 减少epoch数（测试用）
BATCH_SIZE = 8       # 减小batch size
SEQ_LEN = 8          # 保持短序列
PRED_LEN = 8

# 在 DataLoader 中增加并行
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=8  # 增加数据加载线程
)
```

## 📊 预期效果

**成功启用GPU后，你应该看到：**

```
============================================================
MOMENT Traffic Flow Forecasting - Training
============================================================

Configuration:
  Device: cuda                      ← GPU已启用！
  Sequence length: 8
  Prediction horizon: 8
  Batch size: 32
  Epochs: 10
  Learning rate: 0.001

Creating MOMENT model:
  Features (n_channels): 628
  Forecast horizon: 8
  Task: Linear probing (freeze encoder)

Model statistics:
  Total parameters: 85,432,576
  Trainable parameters: 5,024
  Training ratio: 0.01%

Starting training...

Epoch 1/10
Training: 100%|████████| 26/26 [00:08<00:00,  3.12it/s, loss=0.0234]
  Train Loss: 0.023400
Validating: 100%|███████| 7/7 [00:02<00:00,  3.45it/s]
  Val Loss: 0.019800
  Val MAE: 0.112300
  ✓ Saved best model

...

Training completed!
Best validation loss: 0.015600
```

**每个epoch只需几秒到十几秒！**

## 🎯 推荐操作流程

1. **立即执行**（5分钟）：
   ```bash
   cd /home/user/Downloads/cai/moment-main/moment-main
   chmod +x quick_fix_cuda.sh
   ./quick_fix_cuda.sh
   ```

2. **验证安装**：
   ```bash
   python check_cuda.py
   ```

3. **开始训练**：
   ```bash
   python train_moment.py
   ```

4. **查看结果**：
   ```bash
   ls -lh checkpoints/best_model.pth
   ls -lh results/*.png
   ```

---

**最后更新**: 2026-04-22  
**适用系统**: Ubuntu 24.04  
**Python版本**: 3.11  
**PyTorch版本**: 2.x with CUDA 12.1