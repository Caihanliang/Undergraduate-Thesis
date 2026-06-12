#!/bin/bash

# 手动下载 MOMENT 模型脚本
# 使用新的 hf 命令替代已废弃的 huggingface-cli

set -e

echo "=========================================="
echo "Manual MOMENT Model Download"
echo "=========================================="
echo ""

MODEL_NAME="AutonLab/MOMENT-1-large"
CACHE_DIR="${HOME}/.cache/huggingface/hub"

echo "Model: $MODEL_NAME"
echo "Cache directory: $CACHE_DIR"
echo ""

# 检查是否已存在
if [ -d "$CACHE_DIR/models--AutonLab--MOMENT-1-large" ]; then
    echo "✓ Model already exists in cache!"
    echo "  Location: $CACHE_DIR/models--AutonLab--MOMENT-1-large"
    echo ""
    echo "You can now run training:"
    echo "  python train_moment.py"
    exit 0
fi

# 设置镜像加速（国内用户）
export HF_ENDPOINT=https://hf-mirror.com
echo "Using HuggingFace mirror: $HF_ENDPOINT"
echo ""

echo "Downloading model using 'hf' command..."
echo "(This may take several minutes depending on your network)"
echo ""

# 方法1: 使用 hf download（新命令）
if command -v hf &> /dev/null; then
    echo "Using hf download command..."
    hf download AutonLab/MOMENT-1-large --local-dir "$CACHE_DIR/models--AutonLab--MOMENT-1-large/snapshots/main"
else
    echo "hf command not found, trying Python method..."
    
    # 方法2: 使用 Python huggingface_hub
    python << 'EOF'
import os
from huggingface_hub import snapshot_download

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("Downloading MOMENT-1-large model...")
print("This may take 5-15 minutes depending on your network speed.")
print("")

try:
    local_path = snapshot_download(
        repo_id="AutonLab/MOMENT-1-large",
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        resume_download=True
    )
    print(f"\n✓ Model downloaded successfully!")
    print(f"  Location: {local_path}")
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    print("\nAlternative solution:")
    print("  1. Check your network connection")
    print("  2. Try setting proxy if needed")
    print("  3. Or just run 'python train_moment.py' directly")
    print("     (it will auto-download with retry logic)")
    exit(1)
EOF
fi

echo ""
echo "=========================================="
echo "✓ Download complete!"
echo "=========================================="
echo ""
echo "Model cached at: $CACHE_DIR"
echo ""
echo "Now you can run training (will use local cache):"
echo "  python train_moment.py"