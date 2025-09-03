#!/bin/bash

# 产品展示系统启动脚本

echo "🎯 产品展示系统启动脚本"
echo "================================"

# 激活conda环境
echo "🔧 激活conda环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate chatbot

if [ $? -ne 0 ]; then
    echo "❌ 激活conda环境失败，请检查环境名称"
    exit 1
fi

echo "✅ conda环境已激活: $(conda info --envs | grep '*' | awk '{print $1}')"

# 检查Python是否安装
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装或不在PATH中"
    exit 1
fi

# 检查依赖是否安装
echo "🔍 检查依赖..."
if ! python -c "import paho.mqtt.client, flask, ollama" 2>/dev/null; then
    echo "❌ 缺少依赖，正在安装..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ 依赖安装失败"
        exit 1
    fi
    echo "✅ 依赖安装完成"
else
    echo "✅ 依赖检查通过"
fi

# 检查MQTT代理服务器
echo "🔍 检查MQTT代理服务器..."
if ! systemctl is-active --quiet mosquitto; then
    echo "⚠️  Mosquitto服务未运行，尝试启动..."
    sudo systemctl start mosquitto
    if [ $? -ne 0 ]; then
        echo "❌ 无法启动Mosquitto服务"
        echo "请手动安装并启动MQTT代理服务器："
        echo "sudo apt-get install mosquitto mosquitto-clients"
        echo "sudo systemctl start mosquitto"
        echo "sudo systemctl enable mosquitto"
        exit 1
    fi
    echo "✅ Mosquitto服务已启动"
else
    echo "✅ Mosquitto服务运行中"
fi

# 启动产品展示系统
echo "🚀 启动产品展示系统..."
python start_server.py

echo "👋 系统已停止"
