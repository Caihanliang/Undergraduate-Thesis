#!/bin/bash
# GMAN 高速公路交通流量预测 - 完整流程脚本
# 功能: 数据预处理 -> 模型训练 -> 结果可视化

set -e  # 遇到错误立即退出

echo "============================================================"
echo "GMAN 高速公路交通流量预测 - 4特征 (8输入8输出)"
echo "============================================================"
echo ""

# 项目根目录
PROJECT_ROOT="/home/user/Downloads/cai/GMAN"
cd $PROJECT_ROOT

# ==================== Step 1: 数据预处理 ====================
echo "Step 1/3: 数据预处理..."
echo "------------------------------------------------------------"
python preprocess_highway_data.py

if [ $? -ne 0 ]; then
    echo "❌ 数据预处理失败!"
    exit 1
fi

echo "✅ 数据预处理完成!"
echo ""

# ==================== Step 2: 模型训练 ====================
echo "Step 2/3: 模型训练..."
echo "------------------------------------------------------------"
python train_highway_4feat.py \
    --P 8 \
    --Q 8 \
    --L 5 \
    --K 8 \
    --d 8 \
    --batch_size 32 \
    --max_epoch 100 \
    --patience 10 \
    --learning_rate 0.001 \
    --decay_epoch 5

if [ $? -ne 0 ]; then
    echo "❌ 模型训练失败!"
    exit 1
fi

echo "✅ 模型训练完成!"
echo ""

# ==================== Step 3: 结果可视化与评估 ====================
echo "Step 3/3: 结果可视化与评估..."
echo "------------------------------------------------------------"
python visualize_and_evaluate.py

if [ $? -ne 0 ]; then
    echo "❌ 结果分析失败!"
    exit 1
fi

echo "✅ 结果分析完成!"
echo ""

# ==================== 完成 ====================
echo "============================================================"
echo "🎉 全部流程完成!"
echo "============================================================"
echo ""
echo "📁 输出文件位置:"
echo "   - 预处理数据: $PROJECT_ROOT/data/highway_4feat/"
echo "   - 训练模型:   $PROJECT_ROOT/models/highway_4feat/"
echo "   - 训练日志:   $PROJECT_ROOT/logs/highway_4feat.log"
echo "   - 预测结果:   $PROJECT_ROOT/results/highway_4feat/"
echo ""
echo "📊 关键文件:"
echo "   - all_predictions.csv: 所有站点的预测结果"
echo "   - sample_*.png: 可视化图表"
echo ""
