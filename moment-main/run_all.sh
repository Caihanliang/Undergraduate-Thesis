#!/bin/bash

# MOMENT 交通流量预测 - 一键运行脚本
# 本脚本将自动执行数据预处理、模型训练和评估的完整流程

set -e  # 遇到错误立即退出

echo "=========================================="
echo "MOMENT 交通流量预测项目"
echo "=========================================="
echo ""
echo "数据集: dataset/观测站小时交通量-*.csv"
echo "特征: 小客车上/下行 + 非小客车上/下行 (4特征)"
echo ""

# 检查Python版本
echo "1. 检查Python环境..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Python版本: $python_version"

# 检查必要的包
echo ""
echo "2. 检查依赖包..."
python -c "import momentfm" 2>/dev/null || {
    echo "   ❌ momentfm 未安装，正在安装..."
    pip install momentfm
}
python -c "import pandas, numpy, torch, matplotlib" 2>/dev/null || {
    echo "   ❌ 必要依赖未安装，正在安装..."
    pip install pandas numpy torch matplotlib tqdm
}
echo "   ✓ 所有依赖已就绪"

# 检查数据文件
echo ""
echo "3. 检查数据文件..."
if [ ! -d "dataset" ]; then
    echo "   ❌ 错误: 找不到数据集目录 dataset/"
    exit 1
fi

csv_count=$(ls dataset/*.csv 2>/dev/null | wc -l)
if [ "$csv_count" -eq 0 ]; then
    echo "   ❌ 错误: dataset/ 目录下没有找到CSV文件"
    exit 1
fi
echo "   ✓ 找到 $csv_count 个CSV文件"

# 创建必要的目录
echo ""
echo "4. 创建输出目录..."
mkdir -p processed_data moment_data checkpoints results
echo "   ✓ 目录创建完成"

# 步骤1: 数据预处理
echo ""
echo "=========================================="
echo "步骤 1/3: 数据预处理"
echo "=========================================="
echo "提示: 此步骤将合并所有CSV文件并提取4个特征"
echo ""
python preprocess_data.py

if [ $? -ne 0 ]; then
    echo "❌ 数据预处理失败"
    exit 1
fi
echo "✓ 数据预处理完成"

# 步骤2: 模型训练
echo ""
echo "=========================================="
echo "步骤 2/3: 模型训练"
echo "=========================================="
echo "提示: 训练可能需要较长时间，建议使用GPU加速"
echo "      如需中断，按 Ctrl+C 即可"
echo ""

python train_moment.py

if [ $? -ne 0 ]; then
    echo "❌ 模型训练失败"
    exit 1
fi
echo "✓ 模型训练完成"

# 步骤3: 推理与评估
echo ""
echo "=========================================="
echo "步骤 3/3: 推理与评估"
echo "=========================================="
python inference.py

if [ $? -ne 0 ]; then
    echo "❌ 推理评估失败"
    exit 1
fi
echo "✓ 推理评估完成"

# 总结
echo ""
echo "=========================================="
echo "🎉 所有步骤完成！"
echo "=========================================="
echo ""
echo "结果文件位置:"
echo "  - 模型权重:     checkpoints/best_model.pth"
echo "  - 预测结果:     results/predictions.npz"
echo "  - 评估指标:     results/metrics.json"
echo "  - 可视化图表:   results/sample*_station*.png"
echo ""
echo "查看评估指标:"
if [ -f "results/metrics.json" ]; then
    python -c "
import json
with open('results/metrics.json', 'r', encoding='utf-8') as f:
    m = json.load(f)
print(f\"  MAE:  {m['mae']:.4f}\")
print(f\"  RMSE: {m['rmse']:.4f}\")
print(f\"  MAPE: {m['mape']:.2f}%\")
"
else
    echo "（指标文件不存在）"
fi
echo ""
echo "=========================================="