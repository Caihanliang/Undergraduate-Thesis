#!/bin/bash
# ============================================================================
# 一键启动脚本 - 方案B智能评估
# 使用方法: bash quick_start.sh
# ============================================================================

echo "=========================================="
echo "🚀 高速公路4特征LLM评估 - 一键启动"
echo "=========================================="
echo ""

PROJECT_ROOT="/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main"
cd "$PROJECT_ROOT"

# 检查conda环境
if ! conda info --envs | grep -q "FST_unsloth"; then
    echo "❌ 错误: 未找到conda环境 'FST_unsloth'"
    exit 1
fi

# 赋予执行权限
chmod +x run_evaluation_scheme_b.sh manage_screen_scheme_b.sh

echo "✅ 权限设置完成"
echo ""

# 显示选项
echo "请选择运行模式:"
echo "  1) 后台运行（推荐）- 使用Screen会话"
echo "  2) 前台运行 - 直接查看输出"
echo "  3) 仅检查前置条件"
echo ""
read -p "请输入选项 (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "📦 启动后台Screen会话..."
        echo "=========================================="
        bash manage_screen_scheme_b.sh start
        
        echo ""
        echo "💡 提示:"
        echo "   - 查看进度: bash manage_screen_scheme_b.sh status"
        echo "   - 进入会话: bash manage_screen_scheme_b.sh attach"
        echo "   - 查看日志: tail -f $PROJECT_ROOT/logs/screen_eval_scheme_b.log"
        ;;
    
    2)
        echo ""
        echo "=========================================="
        echo "📦 启动前台评估任务..."
        echo "=========================================="
        conda activate FST_unsloth
        bash run_evaluation_scheme_b.sh
        ;;
    
    3)
        echo ""
        echo "=========================================="
        echo "🔍 检查前置条件..."
        echo "=========================================="
        
        # 检查必要文件
        FILES=(
            "config/cai/quick.json"
            "config/cai/finetune_real_traffic.npz"
            "config/cai/finetune_data.npz"
        )
        
        ALL_EXIST=true
        for file in "${FILES[@]}"; do
            if [ -f "$file" ]; then
                SIZE=$(du -h "$file" | cut -f1)
                echo "✅ $file ($SIZE)"
            else
                echo "❌ $file - 不存在"
                ALL_EXIST=false
            fi
        done
        
        echo ""
        if [ "$ALL_EXIST" = true ]; then
            echo "✅ 所有前置条件满足，可以开始评估"
            
            # 显示quick.json信息
            SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('config/cai/quick.json'))))")
            echo ""
            echo "📊 quick.json样本数: $SAMPLE_COUNT"
            
            if [ "$SAMPLE_COUNT" -lt 100000 ]; then
                echo "⚠️  警告: 样本数较少，可能导致覆盖率不足"
                echo "   建议: 重新运行 fin_fast_vals_quick.py 生成完整数据"
            fi
        else
            echo "❌ 缺少必要文件，请先运行微调代码"
        fi
        ;;
    
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✨ 操作完成"
echo "=========================================="
