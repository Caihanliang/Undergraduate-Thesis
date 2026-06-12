# 方案B实施总结

## 📝 修改文件清单

### **1. evaluate2_8.py** （核心修改）

#### **新增功能：智能样本索引映射**

**新增函数 `build_sample_index_map()`**（第52-130行）
```python
def build_sample_index_map(answer_list, tokenizer, num_stations, num_samples):
    """
    从quick.json中解析每条样本的(station_idx, feat_idx, sample_idx)
    建立映射表用于后续评估时的快速查找
    """
```

**工作原理**：
1. 遍历`answer_list`中的所有样本
2. 解码`labels`中的token为文本
3. 使用正则表达式提取站点名称和特征类型
4. 模糊匹配站点名称到索引
5. 按顺序分配sample_idx（每个station+feature组合独立计数）
6. 返回映射字典：`{(station, feat, sample): answer_idx}`

**修改函数 `evaluate_single_station()`**（第133-210行）
```python
def evaluate_single_station(..., index_map):
    # 新增参数: index_map
    
    for i in range(num_samples):
        key = (target_station_idx, feat_idx, i)
        
        if key in index_map:
            # 从quick.json提取LLM预测
            ans_idx = index_map[key]
            pred_nums = extract_numbers_from_labels(...)
        else:
            # 样本不存在，标记为缺失
            pred_nums = None
        
        if pred_nums is not None and len(pred_nums) == 8:
            llm_preds.append(pred_nums)
        else:
            llm_preds.append(gnn_nums)  # 回退到GNN
```

**main函数修改**（第375-380行）
```python
# 3.5. 构建智能样本索引映射
station_list_global = station_list
index_map = build_sample_index_map(answer_list, tokenizer, num_stations, num_samples)

# 5. 执行评估（传递index_map）
station_results = evaluate_single_station(
    station_idx, station_name, answer_list, 
    true_array, pred_array, num_samples, FEATURE_NAMES, tokenizer, index_map
)
```

---

### **2. fin_fast_vals_quick.py** （可选修改）

**修改1：NORMAL_SAMPLE_RATIO**（第382行）
```python
# 原值: NORMAL_SAMPLE_RATIO = 0.1
# 新值: 
NORMAL_SAMPLE_RATIO = 1.0  # 使用100%样本
```

**修改2：MAX_SAMPLES限制**（第517-522行）
```python
# 原值: MAX_SAMPLES = 30000
# 新值:
MAX_SAMPLES = None  # 不限制样本数量
if MAX_SAMPLES is not None and len(generated_dataset) > MAX_SAMPLES:
    generated_dataset = generated_dataset[:MAX_SAMPLES]
else:
    print(f"✓ 使用全部生成的样本: {len(generated_dataset)} 条")
```

**注意**：这些修改仅在重新生成数据时需要。如果直接使用现有quick.json，可以不修改此文件。

---

### **3. 新增脚本文件**

#### **run_evaluation_scheme_b.sh**
- 自动化评估流程
- 前置条件检查
- 日志记录
- 结果摘要显示

#### **manage_screen_scheme_b.sh**
- Screen会话管理
- 支持start/status/attach/stop命令
- 后台运行支持

#### **quick_start.sh**
- 交互式一键启动
- 提供3种运行模式选择
- 自动权限设置

#### **README_SCHEME_B.md**
- 完整的使用文档
- 常见问题解答
- 结果解读指南

---

## 🎯 核心优势

### **1. 无需重新生成数据**
- 直接使用现有的`quick.json`（即使只有30,000条）
- 节省10-30分钟的数据生成时间

### **2. 智能容错机制**
- 自动识别缺失样本
- 优雅回退到GNN预测值
- 提供详细的成功率统计

### **3. 灵活可扩展**
- 支持全站点/单站点评估
- 可调整Tokenizer路径
- 易于集成到CI/CD流程

---

## ⚠️ 局限性

### **1. 样本覆盖率问题**

如果`quick.json`只有30,000条样本：
```
覆盖率 = 30000 / (157 × 4 × 850) ≈ 5.6%
```

这意味着：
- 每个特征平均只有约48个有效样本
- 95%的样本会回退到GNN
- 整体改进率可能接近0%

### **2. 模糊匹配的准确性**

站点名称匹配使用简单的字符串包含检查：
```python
if station_name.lower() in s_name.lower() or s_name.lower() in station_name.lower():
```

**潜在问题**：
- 相似站点名可能误匹配
- 缩写/别名可能无法匹配

**改进建议**：
- 在微调代码中添加明确的元数据字段（如`"station_idx": 0`）
- 使用编辑距离算法提高匹配精度

---

## 📊 预期输出示例

### **索引映射构建阶段**
```
🔍 正在构建样本索引映射表...
解析样本: 100%|██████████| 30000/30000 [02:30<00:00, 199.45it/s]
✓ 索引映射构建完成:
   成功匹配: 28500 条
   未匹配: 1500 条
   映射表大小: 28500 条
```

### **单站点评估输出**
```
==========================================================================================
🔍 正在评估站点 0: Huangxing is located inChangsha City...
==========================================================================================
   有效样本: 45/850, 缺失/失败: 805/850

特征 0: Passenger Car Up
  MAE:  GNN=147.3515, LLM=142.1234, 改进=+3.55%
  RMSE: GNN=226.2717, LLM=218.4562, 改进=+3.45%
  MAPE: GNN=0.5121, LLM=0.4956, 改进=+3.22%
```

### **汇总报告**
```
========================================================================================================================
📊 评估结果汇总 (ALL_STATIONS)
========================================================================================================================

站点         | 特征                             | Metric | GNN          | LLM          | 改进          
------------------------------------------------------------------------------------------------------------------------
AVG        | Passenger Car Up               | MAE    | 90.7304      | 89.5421      |     +1.31%
           |                                | RMSE   | 163.4690     | 161.2345     |     +1.37%
           |                                | MAPE   | 0.4615       | 0.4523       |     +2.00%
...
```

---

## 🚀 立即执行

```bash
cd /home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main

# 方法1：交互式启动（推荐新手）
bash quick_start.sh

# 方法2：直接后台运行
bash manage_screen_scheme_b.sh start

# 方法3：查看进度
bash manage_screen_scheme_b.sh status

# 方法4：进入screen实时查看
bash manage_screen_scheme_b.sh attach
```

---

## 🔧 故障排查

### **问题1：索引映射匹配率低**

**症状**：
```
成功匹配: 5000 条
未匹配: 25000 条
```

**原因**：站点名称格式不一致

**解决**：
```bash
# 调试：查看前10条样本的文本
python3 << 'EOF'
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/home/user/Llama-3.1-8B')
with open('config/cai/quick.json', 'r') as f:
    data = json.load(f)

for i in range(10):
    labels = data[i]['labels']
    valid_tokens = [t for t in labels if t != -100]
    text = tokenizer.decode(valid_tokens[:50], skip_special_tokens=True)
    print(f"\n样本 {i}:")
    print(text[:200])
EOF
```

### **问题2：评估速度过慢**

**原因**：每次都要解码labels

**优化**：预计算所有样本的预测值
```python
# 在build_sample_index_map时同时提取数值
pred_cache = {}
for ans_idx, item in enumerate(answer_list):
    labels = item['labels']
    pred_nums = extract_numbers_from_labels(labels, tokenizer)
    pred_cache[ans_idx] = pred_nums

# 评估时直接使用缓存
pred_nums = pred_cache.get(ans_idx)
```

### **问题3：内存不足**

**症状**：`CUDA out of memory`

**解决**：
```bash
# 减少batch size或启用CPU模式
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python evaluate2_8.py --json_file config/cai/quick.json
```

---

## 📈 后续优化方向

### **短期（1-2天）**
1. ✅ 运行当前方案B评估
2. 📊 分析覆盖率和改进率
3. 🔍 识别失败样本的特征

### **中期（1周）**
1. 🔄 重新生成完整训练数据（NORMAL_SAMPLE_RATIO=1.0）
2. 🎯 针对性微调困难样本
3. 📝 改进Prompt工程

### **长期（1月）**
1. 🏗️ 引入验证集和早停机制
2. 🧪 A/B测试不同微调策略
3. 🚀 部署到生产环境

---

## 💡 关键洞察

**方案B的本质**：
- 不是完美的解决方案
- 而是在**时间成本**和**评估准确性**之间的权衡
- 适合快速验证微调效果，但不适合最终性能评估

**建议决策树**：
```
quick.json样本数 > 100,000？
├─ 是 → 使用方案B评估，结果可信度高
└─ 否 → 先重新生成完整数据，再评估
```
