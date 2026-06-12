#!/bin/bash
# Time-MoE 环境搭建脚本
# 使用conda创建虚拟环境

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Time-MoE 环境搭建"
echo "=========================================="

# 配置
ENV_NAME="time-moe"
PYTHON_VERSION="3.10"

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

echo ""
echo "📦 步骤1: 创建conda虚拟环境..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

conda create -n time-moe python=3.10 -y


echo ""
echo "📦 步骤2: 激活环境并安装依赖..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo ""
echo "📦 步骤3: 安装基础依赖..."
pip install pyyaml numpy pandas torch scikit-learn matplotlib

echo ""
echo "📦 步骤4: 安装transformers和相关库..."
pip install transformers==4.40.1 datasets>=2.18.0 accelerate>=0.28.0

echo ""
echo "📦 步骤5: [可选] 安装flash-attn加速（需要GPU）..."
echo "   如果有GPU且想要更快的训练速度，可以取消下面注释："
# pip install flash-attn==2.6.3 --no-build-isolation

echo ""
echo "✅ 环境搭建完成！"
echo ""
echo "=========================================="
echo "使用说明："
echo "=========================================="
echo "1. 激活环境: conda activate $ENV_NAME"
echo "2. 数据预处理: python preprocess_data.py"
echo "3. 训练模型: python torch_dist_run.py main.py -d ./processed_data/train.jsonl"
echo "4. 推理预测: python inference_and_visualize.py"
echo "=========================================="
echo ""
echo "💡 提示: 如果要使用GPU训练，请确保已安装CUDA和对应的PyTorch版本"
echo ""
