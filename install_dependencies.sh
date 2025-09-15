#!/bin/bash

echo "🚀 正在安装Seeed智能问答系统依赖..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "📋 Python版本: $python_version"

# 安装基础依赖
echo "📦 安装基础依赖..."
pip3 install numpy faiss-cpu ollama torch torchaudio

# 安装TTS依赖
echo "🎤 安装TTS依赖..."
pip3 install melo-tts pygame

# 检查系统依赖
echo "🔍 检查系统依赖..."

# 检查音频系统
if command -v aplay &> /dev/null; then
    echo "✅ ALSA音频系统已安装"
else
    echo "⚠️  建议安装ALSA音频系统: sudo apt-get install alsa-utils"
fi

# 检查PulseAudio
if command -v pulseaudio &> /dev/null; then
    echo "✅ PulseAudio已安装"
else
    echo "⚠️  建议安装PulseAudio: sudo apt-get install pulseaudio"
fi

echo "🎉 依赖安装完成！"
echo ""
echo "💡 使用说明:"
echo "   1. 确保Ollama服务正在运行"
echo "   2. 运行: python3 optimized_qa.py"
echo "   3. 输入 'help' 查看所有命令"
echo "   4. 输入 'tts' 切换语音功能"
echo ""
echo "🔧 如果遇到音频问题，请检查:"
echo "   - 音频设备是否正常工作"
echo "   - 音量是否开启"
echo "   - 是否有音频权限"


