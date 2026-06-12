#!/bin/bash
# Time-MoE 完整流程运行脚本
# 包含：数据预处理 -> 训练 -> 推理 -> 可视化

set -e

echo "=========================================="
echo "Time-MoE 完整流程"
echo "=========================================="

# 检查conda环境
ENV_NAME="time-moe"
if ! conda info --envs | grep -q $ENV_NAME; then
    echo "❌ 错误: 未找到conda环境 '$ENV_NAME'"
    echo "请先运行: bash setup_env.sh"
    exit 1
fi

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo ""
echo "📊 步骤1: 数据预处理..."
python preprocess_data.py

if [ $? -ne 0 ]; then
    echo "❌ 数据预处理失败"
    exit 1
fi

echo ""
echo "🔥 步骤2: 训练模型..."
echo "   提示: 这将需要较长时间，可以使用Ctrl+C中断"
echo "   如果使用GPU，请确保CUDA可用"

# 训练参数配置
DATA_PATH="./processed_data/train.jsonl"
MODEL_PATH="Maple728/TimeMoE-50M"  # 从头训练可以改为 --from_scratch
OUTPUT_PATH="./logs/time_moe_traffic"
MAX_LENGTH=64  # context_length(8) + prediction_length(8) = 16，设置稍大一些
BATCH_SIZE=16  # 根据显存调整
LEARNING_RATE=1e-4
EPOCHS=10

python torch_dist_run.py main.py \
    -d $DATA_PATH \
    -m $MODEL_PATH \
    -o $OUTPUT_PATH \
    --max_length $MAX_LENGTH \
    --micro_batch_size $BATCH_SIZE \
    --global_batch_size 64 \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --stride 1 \
    --normalization_method zero \
    --precision fp32 \
    --save_strategy epoch \
    --logging_steps 10

if [ $? -ne 0 ]; then
    echo "❌ 训练失败"
    exit 1
fi

echo ""
echo "🔮 步骤3: 推理和可视化..."
python inference_and_visualize.py

if [ $? -ne 0 ]; then
    echo "❌ 推理失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 完整流程执行完成！"
echo "=========================================="
echo ""
echo "📁 结果位置:"
echo "   - 训练日志: $OUTPUT_PATH"
echo "   - 可视化结果: ./visualization_results/"
echo "   - 预测结果CSV: ./prediction_results.csv"
echo ""
