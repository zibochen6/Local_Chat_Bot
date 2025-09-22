#!/bin/bash

# Kokoro TTS 模型下载脚本
# 参考: https://blog.csdn.net/u010522887/article/details/146720024

echo "🚀 开始下载 Kokoro TTS 模型文件..."

# 设置 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 创建模型目录
mkdir -p ckpts/kokoro-v1.1

echo "📥 下载 Kokoro-82M-v1.1-zh 模型..."
huggingface-cli download --resume-download hexgrad/Kokoro-82M-v1.1-zh --local-dir ./ckpts/kokoro-v1.1

echo "✅ 模型下载完成！"
echo ""
echo "📁 模型文件位置:"
echo "   - 模型权重: ckpts/kokoro-v1.1/kokoro-v1_1-zh.pth"
echo "   - 配置文件: ckpts/kokoro-v1.1/config.json"
echo "   - 音色文件: ckpts/kokoro-v1.1/voices/"
echo ""
echo "🎵 可用的音色文件:"
ls -la ckpts/kokoro-v1.1/voices/ 2>/dev/null || echo "   音色文件下载中..."
echo ""
echo "🚀 现在可以运行 python tts/test.py 来测试中英文混合语音合成！"
