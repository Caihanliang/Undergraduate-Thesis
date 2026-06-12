#!/bin/bash

# 快速修复：安装支持CUDA的PyTorch
# 适用于驱动版本较旧但想使用GPU的情况

set -e

echo "=========================================="
echo "Quick Fix: Install CUDA-enabled PyTorch"
echo "=========================================="
echo ""

echo "Current PyTorch:"
python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"
echo ""

echo "This script will:"
echo "  1. Uninstall current PyTorch"
echo "  2. Install PyTorch with CUDA 12.1 support"
echo "  3. Verify installation"
echo ""

read -p "Continue? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Step 1/3: Uninstalling current PyTorch..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo ""
echo "Step 2/3: Installing PyTorch with CUDA 12.1..."
echo "(Compatible with your CUDA 12.0 driver)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Step 3/3: Verifying installation..."
python check_cuda.py

echo ""
echo "=========================================="
echo "✓ Installation complete!"
echo "=========================================="
echo ""
echo "Now you can run training with GPU acceleration:"
echo "  python train_moment.py"
echo ""