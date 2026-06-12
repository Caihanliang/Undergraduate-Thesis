#!/bin/bash

# CUDA环境修复脚本
# 本脚本将帮助你配置CUDA环境以启用GPU加速

set -e

echo "=========================================="
echo "CUDA Environment Setup Script"
echo "=========================================="
echo ""

# 检查当前状态
echo "1. Checking current status..."
echo "   Python version: $(python --version 2>&1)"
echo "   PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "   CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"

if command -v nvidia-smi &> /dev/null; then
    echo "   NVIDIA Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
    echo "   ⚠️  nvidia-smi not found - driver may not be installed"
fi

echo ""
echo "2. Available solutions:"
echo ""
echo "   Option A: Update NVIDIA Driver (Recommended)"
echo "   Option B: Install compatible PyTorch version"
echo "   Option C: Continue with CPU (slow)"
echo ""
read -p "Choose option (A/B/C): " choice

case $choice in
    [Aa])
        echo ""
        echo "=========================================="
        echo "Option A: Updating NVIDIA Driver"
        echo "=========================================="
        echo ""
        echo "Step 1: Check your GPU model"
        lspci | grep -i nvidia || echo "No NVIDIA GPU detected!"
        
        echo ""
        echo "Step 2: Install latest driver"
        echo ""
        echo "For Ubuntu/Debian:"
        echo "  sudo apt update"
        echo "  sudo ubuntu-drivers autoinstall"
        echo "  sudo reboot"
        echo ""
        echo "Or download from NVIDIA website:"
        echo "  http://www.nvidia.com/Download/index.aspx"
        echo ""
        read -p "Have you updated the driver? (y/n): " updated
        
        if [ "$updated" = "y" ] || [ "$updated" = "Y" ]; then
            echo ""
            echo "After reboot, verify installation:"
            echo "  nvidia-smi"
            echo "  python check_cuda.py"
            echo ""
            echo "✓ Please reboot your system and run this script again"
        else
            echo ""
            echo "Please update your driver first, then run this script again."
        fi
        ;;
        
    [Bb])
        echo ""
        echo "=========================================="
        echo "Option B: Installing Compatible PyTorch"
        echo "=========================================="
        echo ""
        echo "Current PyTorch version:"
        python -c "import torch; print(torch.__version__)"
        
        echo ""
        echo "Installing PyTorch with CUDA 12.1 support..."
        echo "(This is backward compatible with CUDA 12.0)"
        echo ""
        
        # Uninstall current PyTorch
        echo "Uninstalling current PyTorch..."
        pip uninstall -y torch torchvision torchaudio
        
        # Install CUDA-enabled PyTorch
        echo "Installing PyTorch with CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        
        echo ""
        echo "Verifying installation..."
        python check_cuda.py
        
        echo ""
        echo "✓ Installation complete! Try running training again."
        ;;
        
    [Cc])
        echo ""
        echo "=========================================="
        echo "Option C: Continue with CPU"
        echo "=========================================="
        echo ""
        echo "Training will be significantly slower on CPU."
        echo ""
        echo "Recommendations for faster CPU training:"
        echo "  1. Use smaller sequence lengths (already set to 8)"
        echo "  2. Reduce number of epochs (try 2-3 for testing)"
        echo "  3. Reduce batch size if memory issues occur"
        echo ""
        echo "To modify training parameters, edit train_moment.py:"
        echo "  EPOCHS = 2  # Instead of 10"
        echo "  BATCH_SIZE = 8  # Instead of 32"
        echo ""
        read -p "Continue with CPU training? (y/n): " continue_cpu
        
        if [ "$continue_cpu" = "y" ] || [ "$continue_cpu" = "Y" ]; then
            echo ""
            echo "Starting training on CPU..."
            echo "This may take a while..."
            python train_moment.py
        else
            echo "Aborted. Please update your drivers or PyTorch first."
        fi
        ;;
        
    *)
        echo "Invalid option. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="