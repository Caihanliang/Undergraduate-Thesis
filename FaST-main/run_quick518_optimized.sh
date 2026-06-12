#!/bin/bash
# 快速启动优化版微调脚本

echo "=========================================="
echo "🚀 启动 quick518_optimized 微调"
echo "=========================================="
echo ""

# 进入项目目录
cd /home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main

# 检查Python环境
echo "✓ 检查Python环境..."
python --version

# 检查必要文件
echo ""
echo "✓ 检查必要数据文件..."
FILES=(
    "config/cai/finetune_real_traffic.npz"
    "config/cai/finetune_data.npz"
    "config/cai/station_list_hngs_98.txt"
    "config/cai/station_natural_list_4feat_98.txt"
    "config/cai/his_data_with_index_98.csv"
    "config/cai/events_list_quan.csv"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (缺失)"
        exit 1
    fi
done

# 检查天气目录
if [ -d "config/cai/98站点天气信息/" ]; then
    WEATHER_COUNT=$(ls config/cai/98站点天气信息/ | wc -l)
    echo "  ✅ 98站点天气信息/ ($WEATHER_COUNT 个文件)"
else
    echo "  ⚠️  98站点天气信息/ (目录不存在，将跳过天气特征)"
fi

echo ""
echo "=========================================="
echo "⚙️  配置信息"
echo "=========================================="
echo "  数据集缓存: config/cai/quick518_optimized.json"
echo "  训练输出: config/cai/results-quick518-optimized/"
echo "  模型保存: config/cai/llama-3-1-8b-highway-finetuned-quick518-optimized/"
echo "  日志文件: config/cai/quick518_optimized_*.txt"
echo ""

# 询问是否强制重建数据集
read -p "是否强制重建数据集? (y/n, 默认n): " REBUILD
if [ "$REBUILD" = "y" ] || [ "$REBUILD" = "Y" ]; then
    echo "⚠️  将强制重建数据集（可能需要30-60分钟）"
    # 临时修改代码中的FORCE_REBUILD_DATASET
    sed -i 's/FORCE_REBUILD_DATASET = False/FORCE_REBUILD_DATASET = True/' 微调518_optimized.py
else
    echo "✅ 使用已有数据集缓存（若存在）"
fi

echo ""
echo "=========================================="
echo "🚀 开始执行微调..."
echo "=========================================="
echo ""

# 运行微调脚本
python 微调518_optimized.py

# 恢复FORCE_REBUILD_DATASET设置
sed -i 's/FORCE_REBUILD_DATASET = True/FORCE_REBUILD_DATASET = False/' 微调518_optimized.py

echo ""
echo "=========================================="
echo "✅ 微调完成！"
echo "=========================================="
echo ""
echo "📊 查看结果:"
echo "  日志文件: ls config/cai/quick518_optimized_*.txt"
echo "  对比日志: cat config/cai/comparison_log_quick518_optimized.csv"
echo "  模型目录: ls config/cai/llama-3-1-8b-highway-finetuned-quick518-optimized/"
echo ""
