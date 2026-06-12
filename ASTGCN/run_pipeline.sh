#!/bin/bash
# ASTGCN 高速公路交通流量预测 - 一键运行脚本

set -e  # 遇到错误立即退出

echo "========================================="
echo "ASTGCN 高速公路交通流量预测"
echo "========================================="
echo ""

# 配置变量
DATASET_DIR="./dataset"
CONFIG_FILE="./configurations/highway_traffic.conf"
INPUT_LEN=8
OUTPUT_LEN=8

# 步骤1: 检查数据文件
echo "[步骤1/4] 检查数据文件..."
if [ ! -f "$DATASET_DIR/观测站小时交通量-9.csv" ]; then
    echo "❌ 错误: 找不到 9月数据文件"
    exit 1
fi

if [ ! -f "$DATASET_DIR/观测站小时交通量-10.csv" ]; then
    echo "❌ 错误: 找不到 10月数据文件"
    exit 1
fi

echo "✅ 数据文件检查通过"
echo ""

# 步骤2: 数据预处理
echo "[步骤2/4] 开始数据预处理..."
echo "  - 输入序列长度: $INPUT_LEN 小时"
echo "  - 输出序列长度: $OUTPUT_LEN 小时"
echo "  - 邻接矩阵方法: distance"
echo ""

python preprocess_highway_data.py \
    --dataset_dir $DATASET_DIR \
    --input_len $INPUT_LEN \
    --output_len $OUTPUT_LEN \
    --adj_method distance

if [ $? -ne 0 ]; then
    echo "❌ 数据预处理失败"
    exit 1
fi

echo "✅ 数据预处理完成"
echo ""

# 步骤3: 验证生成的文件
echo "[步骤3/4] 验证生成的文件..."
required_files=(
    "$DATASET_DIR/train.npz"
    "$DATASET_DIR/val.npz"
    "$DATASET_DIR/test.npz"
    "$DATASET_DIR/adj_matrix.csv"
    "$DATASET_DIR/station_mapping.csv"
    "$CONFIG_FILE"
)

all_exist=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (缺失)"
        all_exist=false
    fi
done

if [ "$all_exist" = false ]; then
    echo "❌ 部分文件生成失败"
    exit 1
fi

echo ""
echo "✅ 所有文件验证通过"
echo ""

# 步骤4: 询问是否开始训练
echo "[步骤4/4] 准备开始模型训练"
echo ""
echo "训练配置:"
echo "  - 配置文件: $CONFIG_FILE"
echo "  - 站点数量: 98"
echo "  - 特征数量: 4"
echo "  - 批次大小: 32"
echo "  - 训练轮数: 100"
echo "  - 学习率: 0.001"
echo ""
read -p "是否开始训练? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 启动训练..."
    echo "提示: 可以使用 TensorBoard 监控训练过程"
    echo "      tensorboard --logdir experiments/highway_traffic/"
    echo ""
    
    python train_ASTGCN_r.py --config $CONFIG_FILE
    
    echo ""
    echo "✅ 训练完成!"
    echo ""
    echo "查看结果:"
    echo "  1. TensorBoard: tensorboard --logdir experiments/highway_traffic/"
    echo "  2. 模型权重: ls experiments/highway_traffic/*/epoch_*.params"
    echo "  3. 测试结果: ls experiments/highway_traffic/*/output_epoch_*_test.npz"
else
    echo ""
    echo "⏸️  训练已跳过"
    echo ""
    echo "稍后手动启动训练:"
    echo "  python train_ASTGCN_r.py --config $CONFIG_FILE"
fi

echo ""
echo "========================================="
echo "流程完成!"
echo "========================================="
