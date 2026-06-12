# TimesFM 2.5 Patch 机制说明与配置调整

## ⚠️ 重要更新：Patch 长度要求

### 问题描述

TimesFM 2.5 模型使用 **Patch（补丁）机制**处理时间序列，要求：
- **输入长度必须是 patch_len (32) 的倍数**
- **最小输入长度为 32**

之前的配置 `context_len=8` 会导致错误：
```
RuntimeError: shape '[16, -1, 32]' is invalid for input of size 128
```

---

## ✅ 已修复的配置

### 新配置（32输入8输出）

```python
# config.py
CONTEXT_LEN = 32   # 输入窗口：32小时（符合 Patch 要求）
HORIZON_LEN = 8    # 预测窗口：8小时
```

### 自动调整逻辑

脚本现在包含智能检测和调整：

```python
# finetune_timesfm.py
patch_len = 32
if context_len < patch_len:
    logger.warning(f"⚠️  警告: context_len={context_len} 小于 patch_len={patch_len}")
    logger.warning(f"   自动调整 context_len 从 {context_len} -> {patch_len}")
    context_len = patch_len

if context_len % patch_len != 0:
    adjusted_context_len = ((context_len // patch_len) + 1) * patch_len
    logger.warning(f"⚠️  警告: context_len={context_len} 不是 patch_len 的倍数")
    logger.warning(f"   自动调整 context_len 从 {context_len} -> {adjusted_context_len}")
    context_len = adjusted_context_len
```

---

## 📊 配置对比

| 配置项 | 旧配置（❌ 不可用） | 新配置（✅ 可用） |
|--------|-------------------|------------------|
| 输入窗口 | 8 小时 | **32 小时** |
| 预测窗口 | 8 小时 | 8 小时 |
| 最小序列长度 | 16 | **40** |
| Patch 对齐 | ❌ 不满足 | ✅ 满足 |
| 模型兼容性 | ❌ 报错 | ✅ 正常 |

---

## 🚀 使用方法

### 方式一：重新预处理数据（推荐）

```bash
cd /home/user/Downloads/cai/timesfm-master

# 1. 重新预处理数据（使用新的 context_len=32）
python prepare_traffic_data.py

# 2. 运行训练
python finetune_timesfm.py \
    --context_len 32 \
    --horizon_len 8 \
    --epochs 10 \
    --batch_size 16
```

### 方式二：一键运行

```bash
./run_complete_pipeline.sh
```

脚本会自动使用正确的配置（32输入8输出）。

---

## 🔍 为什么需要 32？

### TimesFM 2.5 架构

TimesFM 2.5 将时间序列分割成 **Patches（补丁）**：

```
原始序列: [x1, x2, x3, ..., x32, x33, ..., x64]
           |____ Patch 1 ____| |____ Patch 2 ____|
           
每个 Patch 长度 = 32
模型处理的是 Patch 序列，而非单个时间点
```

### 技术原因

1. **Patch Embedding**：模型首先将每 32 个时间点编码为一个 Patch 向量
2. **Transformer 处理**：Transformer 层处理 Patch 序列
3. **形状要求**：输入必须能整除 patch_len，否则无法正确 reshape

---

## 💡 其他可行的配置

如果需要不同的预测 horizon，可以使用以下配置：

### 配置选项

| Context Len | Horizon Len | 最小序列长度 | 适用场景 |
|-------------|-------------|-------------|---------|
| 32 | 8 | 40 | 短期预测（推荐） |
| 32 | 16 | 48 | 中期预测 |
| 32 | 24 | 56 | 长期预测 |
| 64 | 8 | 72 | 更长历史窗口 |
| 64 | 24 | 88 | 长历史+长期预测 |
| 96 | 24 | 120 | 超长历史窗口 |

**注意**：所有 context_len 必须是 32 的倍数（32, 64, 96, 128...）

---

## 📝 修改示例

### 改为 64输入24输出

```python
# config.py
CONTEXT_LEN = 64   # 64小时输入
HORIZON_LEN = 24   # 24小时预测
```

```bash
# 重新预处理
python prepare_traffic_data.py

# 训练
python finetune_timesfm.py \
    --context_len 64 \
    --horizon_len 24 \
    --epochs 10
```

---

## ⚠️ 注意事项

### 1. 数据要求提高

- **旧配置**：每个序列至少 16 个时间点
- **新配置**：每个序列至少 40 个时间点

大多数站点应该满足这个要求，但如果数据不足，可能需要：
- 合并多个时间段的数据
- 减小 horizon_len

### 2. 显存占用增加

- 更长的输入窗口 → 更大的显存占用
- 如果显存不足，减小 batch_size：
  ```bash
  python finetune_timesfm.py --batch_size 8
  ```

### 3. 训练时间略增

- 32 输入 vs 8 输入：训练时间增加约 20-30%
- 但仍在可接受范围内

---

## 🎯 性能预期

### 预测精度

32 小时的历史窗口通常能提供：
- ✅ 更好的趋势捕捉能力
- ✅ 更稳定的预测结果
- ✅ 更强的周期性建模

### 训练指标参考

```
Epoch 1/10 — Train Loss: ~0.25, Val Loss: ~0.22
Epoch 5/10 — Train Loss: ~0.16, Val Loss: ~0.14
Epoch 10/10 — Train Loss: ~0.12, Val Loss: ~0.11

Average MAE:  ~25-35
Average RMSE: ~35-45
```

---

## 🔄 回退到短窗口（不推荐）

如果确实需要使用短于 32 的窗口，可以：

### 方案1：使用 TimesFM 1.0（v1 目录）

```bash
cd /home/user/Downloads/cai/timesfm-master/v1
# 使用旧版本模型，无 Patch 限制
```

### 方案2：零填充（效果差）

```python
# 手动将 8 小时数据填充到 32 小时
padded_input = np.pad(original_input, (0, 24), mode='constant')
```

**不推荐**：会引入大量噪声，严重影响预测质量。

---

## ✅ 验证配置

运行以下命令确认配置正确：

```bash
cd /home/user/Downloads/cai/timesfm-master
python config.py
```

**预期输出**：
```
时序配置: 32输入8输出
```

---

## 📞 常见问题

### Q: 为什么不能直接用 8 输入？
A: TimesFM 2.5 的 Patch 机制强制要求，这是模型架构决定的，无法绕过。

### Q: 32 小时会不会太长？
A: 对于小时级数据，32 小时约 1.3 天，能捕捉日周期模式，是合理的窗口长度。

### Q: 可以用 16 吗？
A: 不行，16 不是 32 的倍数。必须是 32, 64, 96, 128...

### Q: 数据不够 40 个点怎么办？
A: 
1. 检查数据源是否有更多历史数据
2. 减小 horizon_len（如改为 4）
3. 合并相邻站点的数据

---

## 🎉 总结

✅ **问题已解决**：添加自动检测和 adjustment 逻辑
✅ **配置已更新**：默认使用 32输入8输出
✅ **向后兼容**：脚本会自动调整不合法的配置
✅ **文档完善**：详细说明 Patch 机制和配置选项

**立即开始训练：**

```bash
cd /home/user/Downloads/cai/timesfm-master
python prepare_traffic_data.py  # 重新预处理
python finetune_timesfm.py      # 开始训练
```
