#!/bin/bash

# Lag-Llama 高速公路交通流量预测 - 完整流程脚本
# 功能: 数据预处理 -> 模型预测 -> 结果可视化

set -e  # 遇到错误立即退出

echo "========================================================================"
echo "Lag-Llama 高速公路交通流量预测 - 完整流程"
echo "========================================================================"

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "项目根目录: $PROJECT_ROOT"
echo ""

# Step 1: 检查Python环境
echo "Step 1: 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3,请先安装Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "  Python版本: $PYTHON_VERSION"

# Step 2: 检查依赖包
echo ""
echo "Step 2: 检查依赖包..."
REQUIRED_PACKAGES=("torch" "gluonts" "pandas" "numpy" "matplotlib" "tqdm" "huggingface_hub")

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo "  ✓ $pkg 已安装"
    else
        echo "  ✗ $pkg 未安装"
        MISSING+=("$pkg")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo "警告: 以下包未安装: ${MISSING[*]}"
    echo "正在安装缺失的包..."
    pip install "${MISSING[@]}"
fi

# Step 3: 设置HuggingFace镜像(加速国内访问)
echo ""
echo "Step 3: 配置HuggingFace镜像..."
export HF_ENDPOINT=https://hf-mirror.com
echo "  HF_ENDPOINT=$HF_ENDPOINT"

# Step 4: 数据预处理
echo ""
echo "========================================================================"
echo "Step 4: 数据预处理"
echo "========================================================================"
python3 scripts/preprocess_traffic_data.py

if [ $? -ne 0 ]; then
    echo "错误: 数据预处理失败!"
    exit 1
fi

# Step 5: 检查预处理结果
echo ""
echo "Step 5: 检查预处理结果..."
if [ ! -f "processed_data/train.jsonl" ] || [ ! -f "processed_data/test.jsonl" ]; then
    echo "错误: 未找到预处理后的数据文件!"
    exit 1
fi

TRAIN_SAMPLES=$(wc -l < processed_data/train.jsonl)
TEST_SAMPLES=$(wc -l < processed_data/test.jsonl)
echo "  训练集样本数: $TRAIN_SAMPLES"
echo "  测试集样本数: $TEST_SAMPLES"

# Step 6: 模型预测
echo ""
echo "========================================================================"
echo "Step 6: 模型预测 (这可能需要较长时间)"
echo "========================================================================"
python3 scripts/predict_with_lag_llama.py

if [ $? -ne 0 ]; then
    echo "错误: 模型预测失败!"
    exit 1
fi

# Step 7: 检查预测结果
echo ""
echo "Step 7: 检查预测结果..."
PREDICTION_FILES=(
    "prediction_results/passenger_up_predictions.csv"
    "prediction_results/passenger_down_predictions.csv"
    "prediction_results/non_passenger_up_predictions.csv"
    "prediction_results/non_passenger_down_predictions.csv"
)

ALL_EXISTS=true
for file in "${PREDICTION_FILES[@]}"; do
    if [ -f "$file" ]; then
        RECORDS=$(tail -n +2 "$file" | wc -l)
        echo "  ✓ $file ($RECORDS 条记录)"
    else
        echo "  ✗ $file 不存在"
        ALL_EXISTS=false
    fi
done

if [ "$ALL_EXISTS" = false ]; then
    echo "警告: 部分预测结果文件缺失"
fi

# Step 8: 可视化
echo ""
echo "========================================================================"
echo "Step 8: 生成可视化图表"
echo "========================================================================"
python3 scripts/visualize_results.py

if [ $? -ne 0 ]; then
    echo "警告: 可视化失败,但预测结果已保存"
fi

# Step 9: 输出总结
echo ""
echo "========================================================================"
echo "流程完成! 总结"
echo "========================================================================"
echo ""
echo "生成的文件:"
echo "  1. 预处理数据:"
echo "     - processed_data/train.jsonl"
echo "     - processed_data/test.jsonl"
echo "     - processed_data/station_mapping.csv"
echo ""
echo "  2. 预测结果:"
echo "     - prediction_results/passenger_up_predictions.csv"
echo "     - prediction_results/passenger_down_predictions.csv"
echo "     - prediction_results/non_passenger_up_predictions.csv"
echo "     - prediction_results/non_passenger_down_predictions.csv"
echo ""
echo "  3. 可视化图表:"
echo "     - prediction_results/visualizations/*.png"
echo ""
echo "下一步建议:"
echo "  - 查看 prediction_results/visualizations/summary_all_features.png 了解整体效果"
echo "  - 如需微调模型,可参考 scripts/finetune.sh"
echo ""
echo "========================================================================"
