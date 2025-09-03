#!/bin/bash

# Seeed Wiki 监控服务管理脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/wiki_monitor.pid"
LOG_FILE="$SCRIPT_DIR/wiki_monitor.log"
DAEMON_SCRIPT="$SCRIPT_DIR/monitor_daemon.py"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查进程是否运行
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
        fi
    fi
    return 1
}

# 启动监控服务
start() {
    print_info "启动 Seeed Wiki 监控服务..."
    
    if is_running; then
        print_warning "监控服务已经在运行中"
        return 1
    fi
    
    # 检查环境
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        return 1
    fi
    
    if ! command -v ollama &> /dev/null; then
        print_error "Ollama 未安装"
        return 1
    fi
    
    # 检查Ollama服务
    if ! curl -s http://localhost:11434/api/tags > /dev/null; then
        print_warning "Ollama 服务未运行，正在启动..."
        ollama serve &
        sleep 5
    fi
    
    # 启动守护进程
    nohup python3 "$DAEMON_SCRIPT" > "$LOG_FILE" 2>&1 &
    local pid=$!
    
    # 等待启动
    sleep 2
    
    if is_running; then
        print_success "监控服务启动成功 (PID: $pid)"
        print_info "日志文件: $LOG_FILE"
        print_info "使用 'tail -f $LOG_FILE' 查看实时日志"
    else
        print_error "监控服务启动失败"
        return 1
    fi
}

# 停止监控服务
stop() {
    print_info "停止 Seeed Wiki 监控服务..."
    
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            kill "$pid"
            sleep 2
            
            if ps -p "$pid" > /dev/null 2>&1; then
                print_warning "进程未响应，强制终止..."
                kill -9 "$pid"
            fi
            
            rm -f "$PID_FILE"
            print_success "监控服务已停止"
        else
            print_warning "监控服务未运行"
            rm -f "$PID_FILE"
        fi
    else
        print_warning "未找到PID文件，监控服务可能未运行"
    fi
}

# 重启监控服务
restart() {
    print_info "重启 Seeed Wiki 监控服务..."
    stop
    sleep 2
    start
}

# 查看状态
status() {
    print_info "Seeed Wiki 监控服务状态:"
    
    if is_running; then
        local pid=$(cat "$PID_FILE")
        print_success "运行中 (PID: $pid)"
        
        # 显示进程信息
        echo "进程信息:"
        ps -p "$pid" -o pid,ppid,cmd,etime,pcpu,pmem
        
        # 显示日志文件大小
        if [ -f "$LOG_FILE" ]; then
            local size=$(du -h "$LOG_FILE" | cut -f1)
            echo "日志文件: $LOG_FILE ($size)"
        fi
        
        # 显示最后几行日志
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "最近日志:"
            tail -n 10 "$LOG_FILE"
        fi
    else
        print_error "未运行"
    fi
}

# 查看日志
logs() {
    if [ -f "$LOG_FILE" ]; then
        if [ "$1" = "-f" ]; then
            print_info "实时查看日志 (按 Ctrl+C 退出)..."
            tail -f "$LOG_FILE"
        else
            print_info "查看日志:"
            cat "$LOG_FILE"
        fi
    else
        print_warning "日志文件不存在"
    fi
}

# 手动运行一次更新
update() {
    print_info "手动执行一次更新..."
    
    if [ -f "$SCRIPT_DIR/scrape_with_embeddings.py" ]; then
        cd "$SCRIPT_DIR"
        python3 scrape_with_embeddings.py --mode incremental
    else
        print_error "未找到 scrape_with_embeddings.py"
        return 1
    fi
}

# 显示帮助信息
show_help() {
    echo "Seeed Wiki 监控服务管理脚本"
    echo ""
    echo "用法: $0 {start|stop|restart|status|logs|update|help}"
    echo ""
    echo "命令:"
    echo "  start    启动监控服务"
    echo "  stop     停止监控服务"
    echo "  restart  重启监控服务"
    echo "  status   查看服务状态"
    echo "  logs     查看日志文件"
    echo "  logs -f  实时查看日志"
    echo "  update   手动执行一次更新"
    echo "  help     显示此帮助信息"
    echo ""
    echo "功能说明:"
    echo "  - 每30分钟检查新页面并更新到本地"
    echo "  - 每天凌晨12点自动进行完整数据库更新"
    echo "  - 支持后台运行和自动重启"
    echo "  - 详细的日志记录"
}

# 主函数
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs "$2"
        ;;
    update)
        update
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "未知命令: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

exit 0
