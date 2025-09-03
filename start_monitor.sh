#!/bin/bash

# Seeed Wiki 爬虫监控模式启动脚本

echo "🚀 启动 Seeed Wiki 爬虫监控模式"
echo "📊 功能说明:"
echo "   - 实时检查新页面并更新到本地"
echo "   - 每天凌晨12点自动进行完整数据库更新"
echo "   - 每30分钟快速检查一次新页面"
echo "   - 按 Ctrl+C 停止监控"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python3"
    exit 1
fi

# 检查依赖
echo "🔍 检查依赖..."
python3 -c "import requests, bs4, numpy, faiss, ollama, schedule" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  缺少依赖，正在安装..."
    pip3 install requests beautifulsoup4 numpy faiss-cpu ollama schedule
fi

# 检查Ollama服务
echo "🔍 检查 Ollama 服务..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️  Ollama 服务未运行，正在启动..."
    ollama serve &
    sleep 5
fi

# 检查模型
echo "🔍 检查 nomic-embed-text 模型..."
if ! ollama list | grep -q "nomic-embed-text"; then
    echo "📥 安装 nomic-embed-text 模型..."
    ollama pull nomic-embed-text
fi

echo ""
echo "✅ 环境检查完成"
echo "🔄 启动监控模式..."
echo ""

# 启动监控模式
python3 scrape_with_embeddings.py --mode monitor
