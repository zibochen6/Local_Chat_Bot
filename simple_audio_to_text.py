#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版音频转文字脚本
使用 OpenAI Whisper 处理本地 demo.wav 文件
支持 GPU 加速推理
"""

import os
import whisper
import torch

def check_gpu_support():
    """检查 GPU 支持情况"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU 加速可用: {gpu_name}")
        print(f"💾 GPU 显存: {gpu_memory:.1f} GB")
        return True
    else:
        print("⚠️  GPU 不可用，将使用 CPU 模式")
        return False

def main():
    # 指定本地测试音频文件
    filename = "demo.wav"
    
    # 检查文件是否存在
    if not os.path.exists(filename):
        print(f"❌ 错误: 找不到音频文件 '{filename}'")
        print("请确保 demo.wav 文件存在于当前目录中")
        return
    
    print(f"📁 正在处理音频文件: {filename}")
    
    # 检查 GPU 支持
    use_gpu = check_gpu_support()
    
    # 加载模型
    print("🤖 正在加载 Whisper 模型...")
    try:
        if use_gpu:
            # GPU 模式：使用 medium 模型以获得更好的性能
            print("🔥 使用 GPU 加速模式")
            #medium
            model = whisper.load_model("base")
            # 将模型移动到 GPU
            model = model.to("cuda")
        else:
            # CPU 模式：使用 base 模型以节省内存
            print("💻 使用 CPU 模式")
            model = whisper.load_model("base")
        
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return
    
    # 转换音频
    print("🔄 正在转换语音...")
    try:
        # 设置推理参数
        if use_gpu:
            # GPU 模式：使用半精度浮点数以提高性能
            result = model.transcribe(filename, fp16=True)
        else:
            # CPU 模式：使用默认设置
            result = model.transcribe(filename)
        
        print(f"🔍 检测语言: {result['language']}")
        print(f"📝 转换结果:")
        print("-" * 50)
        print(result["text"])
        print("-" * 50)
        print("✅ 语音转文字完成")
        
        # 如果有分段信息，显示详细时间戳
        if 'segments' in result:
            print("\n📊 详细分段信息:")
            for segment in result['segments']:
                start = segment['start']
                end = segment['end']
                text = segment['text']
                print(f"[{start:.2f}s -> {end:.2f}s] {text}")
        
        # 显示性能信息
        if use_gpu:
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"\n⚡ GPU 性能信息:")
            print(f"   显存使用: {gpu_memory_used:.2f} GB")
            print(f"   显存缓存: {gpu_memory_cached:.2f} GB")
        
    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
    finally:
        # 清理 GPU 内存
        if use_gpu:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
