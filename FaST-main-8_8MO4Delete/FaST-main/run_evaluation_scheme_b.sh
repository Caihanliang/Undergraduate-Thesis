#!/bin/bash
# ============================================================================
# 高速公路4特征LLM评估脚本（方案B：智能样本匹配）
# 使用方法: bash run_evaluation_scheme_b.sh
# 说明：此脚本直接使用现有的quick.json进行智能评估，无需重新生成数据
# ============================================================================

set -e  # 遇到错误立即退出

PROJECT_ROOT="/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main"
CONFIG_DIR="$PROJECT_ROOT/config/cai"
LOG_DIR="$PROJECT_ROOT/logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "🚀 开始高速公路4特征LLM评估（方案B）"
echo "=========================================="
echo "项目根目录: $PROJECT_ROOT"
echo "配置目录: $CONFIG_DIR"
echo "日志目录: $LOG_DIR"
echo ""

# ============================================================================
# 检查前置条件
# ============================================================================
echo "=========================================="
echo "📋 检查前置条件"
echo "=========================================="

# 检查quick.json是否存在
if [ ! -f "$CONFIG_DIR/quick.json" ]; then
    echo "❌ 错误: quick.json不存在: $CONFIG_DIR/quick.json"
    echo "💡 请先运行微调代码生成训练数据"
    exit 1
fi

# 检查finetune_real_traffic.npz和finetune_data.npz
if [ ! -f "$CONFIG_DIR/finetune_real_traffic.npz" ] || [ ! -f "$CONFIG_DIR/finetune_data.npz" ]; then
    echo "❌ 错误: GNN预测值或真实值文件缺失"
    echo "   需要文件:"
    echo "   - $CONFIG_DIR/finetune_real_traffic.npz"
    echo "   - $CONFIG_DIR/finetune_data.npz"
    exit 1
fi

# 检查Tokenizer路径
TOKENIZER_PATH="/home/user/Llama-3.1-8B"
if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "⚠️  警告: Tokenizer路径不存在: $TOKENIZER_PATH"
    echo "💡 如需修改，请编辑本脚本的TOKENIZER_PATH变量"
fi

echo "✅ 所有前置条件检查通过"
echo ""

# 显示quick.json信息
QUICK_JSON_SIZE=$(du -h "$CONFIG_DIR/quick.json" | cut -f1)
QUICK_JSON_LINES=$(python3 -c "import json; data=json.load(open('$CONFIG_DIR/quick.json')); print(len(data))")
echo "📊 quick.json信息:"
echo "   文件大小: $QUICK_JSON_SIZE"
echo "   样本数量: $QUICK_JSON_LINES 条"
echo ""

# ============================================================================
# 执行评估
# ============================================================================
echo "=========================================="
echo "📈 执行模型评估"
echo "=========================================="
echo "评估模式: 全站点评估（157个站点）"
echo "预计耗时: 5-15分钟（取决于Tokenizer加载速度）"
echo ""

cd "$PROJECT_ROOT"
python evaluate2_8.py --json_file "$CONFIG_DIR/quick.json" --tokenizer "$TOKENIZER_PATH" 2>&1 | tee "$LOG_DIR/evaluation_scheme_b.log"

if [ $? -ne 0 ]; then
    echo "❌ 模型评估失败！请检查日志: $LOG_DIR/evaluation_scheme_b.log"
    exit 1
fi

echo ""
echo "✅ 模型评估完成！"
echo "   结果文件: $CONFIG_DIR/evaluation_results_all_stations.json"
echo ""

# ============================================================================
# 显示关键指标摘要
# ============================================================================
echo "=========================================="
echo "📊 评估结果摘要"
echo "=========================================="

if [ -f "$CONFIG_DIR/evaluation_results_all_stations.json" ]; then
    python3 << 'EOF'
import json

with open("/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main/config/cai/evaluation_results_all_stations.json", "r") as f:
    results = json.load(f)

print("\n🏆 平均改进率（所有站点）:")
print("-" * 60)

for feat_key in ["0", "1", "2", "3"]:
    if feat_key in results.get("summary", {}):
        summary = results["summary"][feat_key]
        feat_name = summary.get("name", f"Feature {feat_key}")
        mae_imp = summary.get("overall", {}).get("MAE", {}).get("imp", 0)
        rmse_imp = summary.get("overall", {}).get("RMSE", {}).get("imp", 0)
        mape_imp = summary.get("overall", {}).get("MAPE", {}).get("imp", 0)
        
        print(f"{feat_name}:")
        print(f"  MAE改进:  {mae_imp:+.2f}%")
        print(f"  RMSE改进: {rmse_imp:+.2f}%")
        print(f"  MAPE改进: {mape_imp:+.2f}%")
        print()
else:
    print("⚠️  无法解析评估结果文件")

EOF
fi

# ============================================================================
# 完成
# ============================================================================
echo "=========================================="
echo "🎉 评估流程完成！"
echo "=========================================="
echo ""
echo "📁 关键文件位置:"
echo "   1. 评估结果: $CONFIG_DIR/evaluation_results_all_stations.json"
echo "   2. 运行日志: $LOG_DIR/evaluation_scheme_b.log"
echo ""
echo "💡 提示:"
echo "   - 查看详细结果: cat $CONFIG_DIR/evaluation_results_all_stations.json | python -m json.tool"
echo "   - 重新评估单站点: python evaluate2_8.py --station <idx>"
echo "   - 查看日志尾部: tail -f $LOG_DIR/evaluation_scheme_b.log"
echo ""
