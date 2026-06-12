# 🚀 方案B快速参考卡片

## ⚡ 30秒启动

```bash
cd /home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main
bash manage_screen_scheme_b.sh start
```

## 📊 监控进度

```bash
# 查看状态和最近日志
bash manage_screen_scheme_b.sh status

# 实时查看日志尾部
tail -f logs/screen_eval_scheme_b.log

# 进入screen会话
bash manage_screen_scheme_b.sh attach
# 分离: Ctrl+A, D
```

## 🔍 检查结果

```bash
# 查看评估结果摘要
python3 << 'EOF'
import json
with open("config/cai/evaluation_results_all_stations.json") as f:
    r = json.load(f)
for k in ["0","1","2","3"]:
    if k in r.get("summary",{}):
        s = r["summary"][k]
        print(f"{s['name']}: MAE={s['overall']['MAE']['imp']:+.2f}%")
EOF
```

## ⚠️ 常见问题速查

| 问题 | 命令 |
|------|------|
| 停止任务 | `bash manage_screen_scheme_b.sh stop` |
| 检查样本数 | `python3 -c "import json; print(len(json.load(open('config/cai/quick.json'))))"` |
| 调试提取 | `python evaluate2_8.py --station 0 2>&1 \| head -50` |
| 查看完整日志 | `cat logs/evaluation_scheme_b.log` |

## 📁 关键文件

- **输入**: `config/cai/quick.json` (30K样本)
- **输出**: `config/cai/evaluation_results_all_stations.json`
- **日志**: `logs/evaluation_scheme_b.log`

## 💡 预期结果

- **样本覆盖率**: ~5-10% (因quick.json只有30K条)
- **改进率**: 可能接近0% (大部分回退到GNN)
- **运行时间**: 5-15分钟

## 🎯 下一步

如果改进率≈0%：
```bash
# 1. 修改 fin_fast_vals_quick.py
# NORMAL_SAMPLE_RATIO = 1.0
# MAX_SAMPLES = None

# 2. 重新生成数据
python fin_fast_vals_quick.py

# 3. 重新评估
bash manage_screen_scheme_b.sh start
```

---

**详细文档**: 参见 `README_SCHEME_B.md` 和 `IMPLEMENTATION_SUMMARY.md`
