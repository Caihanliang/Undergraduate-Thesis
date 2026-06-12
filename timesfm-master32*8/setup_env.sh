#!/bin/bash
# TimesFM 项目环境配置和快速启动脚本

echo "=========================================="
echo "TimesFM 交通流量预测 - 环境配置"
echo "=========================================="

# 设置 HuggingFace 镜像加速
export HF_ENDPOINT=https://hf-mirror.com
echo "✓ 已设置 HuggingFace 镜像: $HF_ENDPOINT"

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate
echo "✓ 虚拟环境已激活"

# 升级 pip
pip install --upgrade pip

# 安装核心依赖
echo ""
echo "安装核心依赖包..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft pandas numpy matplotlib scikit-learn

# 安装 timesfm 包
echo ""
echo "安装 timesfm 包..."
pip install -e .

echo ""
echo "=========================================="
echo "✓ 环境配置完成！"
echo "=========================================="
echo ""
echo "下一步操作："
echo "1. 数据预处理: python prepare_traffic_data.py"
echo "2. 微调训练:   python finetune_timesfm.py"
echo "3. 预测评估:   python predict_and_visualize.py"
echo ""
