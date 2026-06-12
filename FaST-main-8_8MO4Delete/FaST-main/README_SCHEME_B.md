# 方案B：智能样本匹配评估方案

## 📋 方案说明

**核心思路**：不重新生成完整的53万条训练数据，而是通过智能索引映射机制，动态匹配`quick.json`中实际存在的样本进行评怗。

**优势**：
- ✅ 无需重新运行耗时的数据生成过程（节省10-30分钟）
- ✅ 自动处理稀疏采样导致的样本缺失问题
- ✅ 对缺失样本自动回退到GNN预测值，保证评估完整性
- ✅ 提供详细的成功/失败统计，便于分析模型覆盖度

---

## 🚀 快速开始

### **方法1：直接运行（推荐）**

```bash
cd /home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main

# 赋予执行权限
chmod +x run_evaluation_scheme_b.sh manage_screen_scheme_b.sh

# 直接在后台运行（推荐）
bash manage_screen_scheme_b.sh start
```

### **方法2：Screen会话管理**

```bash
# 启动评估任务
bash manage_screen_scheme_b.sh start

# 查看任务状态和最近日志
bash manage_screen_scheme_b.sh status

# 附加到运行中的任务（实时查看进度）
bash manage_screen_scheme_b.sh attach

# 在screen中分离会话：按 Ctrl+A, 然后按 D

# 停止任务（如需中断）
bash manage_screen_scheme_b.sh stop
```

### **方法3：前台直接运行**

```bash
cd /home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main
conda activate FST_unsloth
python evaluate2_8.py --json_file config/cai/quick.json
```

---

## 📊 评估流程

### **步骤1：前置条件检查**

脚本会自动检查以下文件是否存在：
- ✅ `config/cai/quick.json` - LLM微调训练数据
- ✅ `config/cai/finetune_real_traffic.npz` - 真实交通流数据
- ✅ `config/cai/finetune_data.npz` - GNN预测值数据
- ✅ `/home/user/Llama-3.1-8B` - Tokenizer模型路径

### **步骤2：构建智能索引映射**

评估脚本会：
1. 遍历`quick.json`中的所有样本
2. 从每条样本的`labels`中解码文本
3. 提取站点名称和特征类型
4. 建立映射表：`(station_idx, feat_idx, sample_idx) -> answer_list_index`

**预期输出**：
```
🔍 正在构建样本索引映射表...
解析样本: 100%|██████████| 30000/30000 [02:30<00:00, 199.45it/s]
✓ 索引映射构建完成:
   成功匹配: 28500 条
   未匹配: 1500 条
   映射表大小: 28500 条
```

### **步骤3：执行站点评估**

对每个站点（共157个）：
- 遍历4个特征
- 对每个特征的850个样本：
  - 如果样本在映射表中 → 使用LLM预测值
  - 如果样本不在映射表中 → 回退到GNN预测值
- 计算MAE、RMSE、MAPE指标

**预期输出**：
```
==========================================================================================
🔍 正在评估站点 0: Huangxing is located inChangsha City...
==========================================================================================
   有效样本: 180/850, 缺失/失败: 670/850

特征 0: Passenger Car Up
  MAE:  GNN=147.3515, LLM=135.2341, 改进=+8.23%
  RMSE: GNN=226.2717, LLM=210.4562, 改进=+6.99%
  MAPE: GNN=0.5121, LLM=0.4856, 改进=+5.17%
```

### **步骤4：生成汇总报告**

评估完成后生成：
- 📄 `config/cai/evaluation_results_all_stations.json` - 完整评估结果
- 📝 `logs/evaluation_scheme_b.log` - 运行日志

---

## 🔍 关键参数说明

### **evaluate2_8.py 参数**

```bash
python evaluate2_8.py \
    --json_file config/cai/quick.json \      # LLM推理结果文件
    --tokenizer /home/user/Llama-3.1-8B \    # Tokenizer路径
    --station -1                              # -1=全站点, 0-156=单站点
```

### **核心逻辑**

```python
# 对于每个样本 (station_idx, feat_idx, sample_idx)
key = (target_station_idx, feat_idx, i)

if key in index_map:
    # 从quick.json中提取LLM预测值
    ans_idx = index_map[key]
    pred_nums = extract_numbers_from_labels(labels, tokenizer)
else:
    # 样本不存在，回退到GNN
    pred_nums = None

if pred_nums is not None and len(pred_nums) == 8:
    llm_preds.append(pred_nums)
else:
    llm_preds.append(gnn_nums)  # 回退
```

---

## 📈 结果解读

### **评估指标**

| 指标 | 含义 | 改进率计算 |
|------|------|-----------|
| MAE | 平均绝对误差 | `(GNN_MAE - LLM_MAE) / GNN_MAE × 100%` |
| RMSE | 均方根误差 | `(GNN_RMSE - LLM_RMSE) / GNN_RMSE × 100%` |
| MAPE | 平均绝对百分比误差 | `(GNN_MAPE - LLM_MAPE) / GNN_MAPE × 100%` |

**改进率解读**：
- ✅ **正值**（如 +8.23%）：LLM优于GNN，误差降低
- ❌ **负值**（如 -15.42%）：LLM劣于GNN，误差增加
- ⚠️ **0.00%**：LLM与GNN相同（通常是回退导致）

### **成功样本率**

```
有效样本: 180/850, 缺失/失败: 670/850
```

- **有效样本**：成功从quick.json提取到8个数值的样本
- **缺失/失败**：样本不在quick.json中或提取失败，回退到GNN

**注意**：如果你的quick.json只有30,000条（而非533,800条），则每个站点的平均有效样本数约为：
```
30000 / (157站点 × 4特征) ≈ 48条/特征
```

这意味着大部分样本会回退到GNN，可能导致整体改进率接近0%。

---

## ⚠️ 常见问题

### **Q1: 为什么所有改进率都是0%？**

**原因**：quick.json中样本数量不足，导致几乎所有样本都回退到GNN。

**解决方案**：
1. 检查quick.json样本数：
   ```bash
   python3 -c "import json; print(len(json.load(open('config/cai/quick.json'))))"
   ```
2. 如果样本数远小于533,800，需要重新运行微调代码生成完整数据：
   ```bash
   # 修改 fin_fast_vals_quick.py 中的 NORMAL_SAMPLE_RATIO = 1.0
   python fin_fast_vals_quick.py
   ```

### **Q2: 提取失败率很高怎么办？**

**可能原因**：
- Tokenizer路径错误
- labels格式不正确
- 模型生成的文本中没有`Final Correction: [...]`格式

**调试方法**：
```bash
python3 << 'EOF'
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/home/user/Llama-3.1-8B')
with open('config/cai/quick.json', 'r') as f:
    data = json.load(f)

# 检查前3条样本
for i in range(3):
    labels = data[i]['labels']
    valid_tokens = [t for t in labels if t != -100]
    text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
    print(f"\n样本 {i}:")
    print(text[-200:])  # 打印最后200字符
EOF
```

### **Q3: 如何只评估单个站点？**

```bash
python evaluate2_8.py --station 0  # 评估站点0
python evaluate2_8.py --station 50 # 评估站点50
```

### **Q4: Screen会话意外断开怎么办？**

```bash
# 查看所有screen会话
screen -list

# 重新附加
screen -r highway_llm_eval_scheme_b

# 如果显示"There is a screen on..."，使用：
screen -d -r highway_llm_eval_scheme_b
```

---

## 📁 文件清单

| 文件 | 用途 |
|------|------|
| `evaluate2_8.py` | 主评估脚本（已修改支持智能匹配） |
| `run_evaluation_scheme_b.sh` | 自动化运行脚本 |
| `manage_screen_scheme_b.sh` | Screen会话管理工具 |
| `config/cai/quick.json` | LLM微调训练数据（输入） |
| `config/cai/evaluation_results_all_stations.json` | 评估结果（输出） |
| `logs/evaluation_scheme_b.log` | 运行日志 |

---

## 💡 优化建议

### **短期优化**
1. **增加样本覆盖率**：修改`fin_fast_vals_quick.py`中的`NORMAL_SAMPLE_RATIO = 1.0`，重新生成完整数据
2. **调整ERROR_THRESHOLD**：降低阈值以包含更多困难样本

### **长期优化**
1. **改进Prompt工程**：确保模型始终输出完整的8个数值
2. **引入验证集**：在微调时加入早停机制，防止过拟合
3. **多轮迭代**：基于评估结果筛选失败样本，进行针对性微调

---

## 🎯 下一步行动

1. **立即执行**：
   ```bash
   bash manage_screen_scheme_b.sh start
   ```

2. **监控进度**：
   ```bash
   bash manage_screen_scheme_b.sh status
   ```

3. **查看结果**：
   ```bash
   cat config/cai/evaluation_results_all_stations.json | python -m json.tool | head -100
   ```

4. **根据结果决策**：
   - 如果改进率显著为正 → ✅ 微调成功，可部署
   - 如果改进率接近0% → ⚠️ 需要重新生成完整训练数据
   - 如果改进率为负 → ❌ 需要调整微调策略
