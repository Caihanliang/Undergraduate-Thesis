#!/bin/bash
# ============================================================================
# Screen会话管理脚本（方案B专用）
# 使用方法:
#   bash manage_screen_scheme_b.sh start    # 启动评估任务
#   bash manage_screen_scheme_b.sh status   # 查看状态
#   bash manage_screen_scheme_b.sh attach   # 附加到会话
#   bash manage_screen_scheme_b.sh stop     # 停止会话
# ============================================================================

SESSION_NAME="highway_llm_eval_scheme_b"
PROJECT_ROOT="/home/user/Downloads/cai/FaST-main-8_8MO4Delete/FaST-main"
SCRIPT_PATH="$PROJECT_ROOT/run_evaluation_scheme_b.sh"

case "$1" in
    start)
        echo "=========================================="
        echo "🚀 启动Screen会话: $SESSION_NAME"
        echo "=========================================="
        
        # 检查是否已存在同名session
        if screen -list | grep -q "$SESSION_NAME"; then
            echo "⚠️  Session '$SESSION_NAME' 已存在！"
            echo "   如需重新启动，请先执行: bash manage_screen_scheme_b.sh stop"
            exit 1
        fi
        
        # 创建detached screen session
        screen -dmS "$SESSION_NAME" bash -c "
            cd $PROJECT_ROOT && \
            source ~/.bashrc && \
            conda activate FST_unsloth && \
            bash $SCRIPT_PATH 2>&1 | tee logs/screen_eval_scheme_b.log; \
            echo ''; \
            echo '========================================'; \
            echo '✅ 评估完成！按任意键退出...'; \
            read -n 1
        "
        
        echo "✅ Screen会话已启动！"
        echo ""
        echo "📋 常用命令:"
        echo "   查看状态:   bash manage_screen_scheme_b.sh status"
        echo "   附加会话:   bash manage_screen_scheme_b.sh attach"
        echo "   查看日志:   tail -f $PROJECT_ROOT/logs/screen_eval_scheme_b.log"
        echo "   分离会话:   在screen中按 Ctrl+A, D"
        echo "   停止会话:   bash manage_screen_scheme_b.sh stop"
        echo ""
        ;;
    
    status)
        echo "=========================================="
        echo "📊 Screen会话状态"
        echo "=========================================="
        screen -list | grep "$SESSION_NAME" || echo "❌ Session '$SESSION_NAME' 不存在"
        echo ""
        
        # 显示最近的日志
        if [ -f "$PROJECT_ROOT/logs/screen_eval_scheme_b.log" ]; then
            echo "📝 最近日志（最后30行）:"
            tail -30 "$PROJECT_ROOT/logs/screen_eval_scheme_b.log"
        else
            echo "📝 日志文件尚未生成"
        fi
        ;;
    
    attach)
        echo "=========================================="
        echo "🔗 附加到Screen会话: $SESSION_NAME"
        echo "=========================================="
        
        if screen -list | grep -q "$SESSION_NAME"; then
            screen -r "$SESSION_NAME"
        else
            echo "❌ Session '$SESSION_NAME' 不存在或已结束"
            exit 1
        fi
        ;;
    
    stop)
        echo "=========================================="
        echo "🛑 停止Screen会话: $SESSION_NAME"
        echo "=========================================="
        
        if screen -list | grep -q "$SESSION_NAME"; then
            screen -S "$SESSION_NAME" -X quit
            echo "✅ Session '$SESSION_NAME' 已停止"
        else
            echo "⚠️  Session '$SESSION_NAME' 不存在"
        fi
        ;;
    
    *)
        echo "=========================================="
        echo "📖 Screen会话管理工具（方案B）"
        echo "=========================================="
        echo ""
        echo "使用方法: bash manage_screen_scheme_b.sh <command>"
        echo ""
        echo "可用命令:"
        echo "   start   - 启动LLM评估任务（使用现有quick.json）"
        echo "   status  - 查看任务状态和最近日志"
        echo "   attach  - 附加到运行中的任务"
        echo "   stop    - 停止当前任务"
        echo ""
        echo "示例:"
        echo "   bash manage_screen_scheme_b.sh start    # 开始评估"
        echo "   bash manage_screen_scheme_b.sh status   # 查看进度"
        echo "   bash manage_screen_scheme_b.sh attach   # 进入screen查看"
        echo "   bash manage_screen_scheme_b.sh stop     # 终止评估"
        echo ""
        ;;
esac
