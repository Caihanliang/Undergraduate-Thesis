#!/bin/bash

# 配置 HuggingFace 镜像加速
# 解决国内访问 HuggingFace 超时问题

echo "=========================================="
echo "Configure HuggingFace Mirror"
echo "=========================================="
echo ""

# 设置环境变量（当前会话）
export HF_ENDPOINT=https://hf-mirror.com
echo "✓ Set HF_ENDPOINT for current session"

# 永久配置
if ! grep -q "HF_ENDPOINT" ~/.bashrc; then
    echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
    echo "✓ Added to ~/.bashrc (permanent)"
else
    echo "✓ Already configured in ~/.bashrc"
fi

# 验证配置
echo ""
echo "Current configuration:"
echo "  HF_ENDPOINT=$HF_ENDPOINT"
echo ""

# 测试连接
echo "Testing connection to HuggingFace mirror..."
python -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import try_to_load_from_cache
print('✓ Mirror endpoint configured successfully')
" 2>/dev/null || echo "⚠️  Test skipped (huggingface_hub not installed)"

echo ""
echo "=========================================="
echo "Configuration complete!"
echo "=========================================="
echo ""
echo "Now you can run training with faster download:"
echo "  python train_moment.py"
echo ""