#!/bin/bash
# Lag-Llama 8输入8输出完整流程自动化脚本
# 用法: bash run_8x8_pipeline.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Lag-Llama 8输入8输出完整流程"
echo "=========================================="
echo ""

# 步骤1: 重新预处理数据
echo "【步骤1/3】正在重新预处理数据（8输入8输出）..."
python scripts/preprocess_traffic_data.py

if [ $? -ne 0 ]; then
    echo "❌ 数据预处理失败！"
    exit 1
fi

echo ""
echo "✅ 数据预处理完成"
echo ""

# 步骤2: 预测训练集所有时间
echo "【步骤2/3】正在预测训练集所有时间窗口..."
python scripts/predict_train_all.py

if [ $? -ne 0 ]; then
    echo "❌ 预测失败！"
    exit 1
fi

echo ""
echo "✅ 预测完成"
echo ""

# 步骤3: 合并和评估结果
echo "【步骤3/3】正在合并预测结果并计算评估指标..."
python scripts/combine_predictions.py
python scripts/evaluate_combined_predictions.py

if [ $? -ne 0 ]; then
    echo "⚠️  评估步骤出现警告，但预测结果已保存"
fi

echo ""
echo "=========================================="
echo "✅ 全部流程完成！"
echo "=========================================="
echo ""
echo "📊 结果文件位置:"
echo "  - 预测结果: prediction_results/*_predictions.csv"
echo "  - 合并结果: prediction_results/all_features_combined.csv"
echo "  - 评估指标: prediction_results/evaluation_metrics.{csv,json}"
echo ""
echo "📈 查看评估结果:"
echo "  cat prediction_results/evaluation_metrics.json"
echo ""
