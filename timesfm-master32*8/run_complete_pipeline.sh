#!/bin/bash
# 一键运行完整流程：数据预处理 → 训练 → 预测（8输入8输出）

set -e  # 遇到错误立即退出

echo "=========================================="
echo "TimesFM Traffic Prediction - 8-Input 8-Output"
echo "=========================================="
echo ""

# 设置项目根目录
PROJECT_ROOT="/home/user/Downloads/cai/timesfm-master"
cd "$PROJECT_ROOT"

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "✗ Virtual environment not found. Please run: ./setup_env.sh first"
    exit 1
fi

# 步骤 1: 数据预处理
echo ""
echo "=========================================="
echo "Step 1/3: Data Preprocessing (8-in 8-out)"
echo "=========================================="
python prepare_traffic_data.py

if [ $? -ne 0 ]; then
    echo "✗ Data preprocessing failed"
    exit 1
fi

echo "✓ Data preprocessing completed"

# 步骤 2: 微调训练
echo ""
echo "=========================================="
echo "Step 2/3: Fine-tuning Training"
echo "=========================================="
python finetune_timesfm.py \
    --context_len 32 \
    --horizon_len 8 \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-4 \
    --lora_r 4 \
    --lora_alpha 8 \
    --num_samples 5000 \
    --output_dir checkpoints/traffic-lora-32x8

if [ $? -ne 0 ]; then
    echo "✗ Training failed"
    exit 1
fi

echo "✓ Training completed"

# 步骤 3: 预测和可视化
echo ""
echo "=========================================="
echo "Step 3/3: Prediction and Visualization"
echo "=========================================="
python predict_and_visualize.py \
    --adapter_path checkpoints/traffic-lora-8x8 \
    --output_dir prediction_results_8x8

if [ $? -ne 0 ]; then
    echo "✗ Prediction failed"
    exit 1
fi

echo "✓ Prediction completed"

echo ""
echo "=========================================="
echo "🎉 All steps completed!"
echo "=========================================="
echo ""
echo "View results:"
echo "  - Training log: training.log"
echo "  - Model checkpoint: checkpoints/traffic-lora-8x8/"
echo "  - Prediction results: prediction_results_8x8/"
echo "  - Visualizations: prediction_results_8x8/*.png"
echo ""
