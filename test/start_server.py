#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动脚本 - 同时启动MQTT服务器和Web服务器
"""

import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import paho.mqtt.client
        import flask
        import ollama
        print("✅ 依赖检查通过")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请确保已激活conda环境: conda activate chatbot")
        print("然后运行: pip install -r requirements.txt")
        return False

def start_mqtt_server():
    """启动MQTT服务器"""
    print("🚀 启动MQTT服务器...")
    try:
        process = subprocess.Popen([
            sys.executable, "mqtt_server.py"
        ], cwd=os.path.dirname(os.path.abspath(__file__)))
        return process
    except Exception as e:
        print(f"❌ 启动MQTT服务器失败: {e}")
        return None

def start_web_server():
    """启动Web服务器"""
    print("🌐 启动Web服务器...")
    try:
        process = subprocess.Popen([
            sys.executable, "web_server.py"
        ], cwd=os.path.dirname(os.path.abspath(__file__)))
        return process
    except Exception as e:
        print(f"❌ 启动Web服务器失败: {e}")
        return None

def signal_handler(signum, frame):
    """信号处理函数"""
    print("\n🛑 收到停止信号，正在关闭服务...")
    sys.exit(0)

def main():
    """主函数"""
    print("🎯 产品展示系统启动器")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动MQTT服务器
    mqtt_process = start_mqtt_server()
    if not mqtt_process:
        return
    
    # 等待MQTT服务器启动
    time.sleep(3)
    
    # 启动Web服务器
    web_process = start_web_server()
    if not web_process:
        print("❌ 启动Web服务器失败，停止MQTT服务器")
        mqtt_process.terminate()
        return
    
    print("\n✅ 所有服务已启动!")
    print("=" * 50)
    print("📱 Web界面: http://localhost:5000")
    print("🔌 MQTT服务器: localhost:1883")
    print("📋 产品列表:")
    print("   - 智能音箱 (ID: 001)")
    print("   - 智能手表 (ID: 002)")
    print("   - 智能摄像头 (ID: 003)")
    print("\n💡 使用说明:")
    print("   1. 用手机触碰产品NFC卡片")
    print("   2. 手机自动打开产品页面")
    print("   3. 点击'获取讲解'按钮")
    print("   4. 系统生成AI产品讲解")
    print("\n按 Ctrl+C 停止所有服务")
    print("=" * 50)
    
    try:
        # 等待进程结束
        while True:
            if mqtt_process.poll() is not None:
                print("❌ MQTT服务器已停止")
                break
            if web_process.poll() is not None:
                print("❌ Web服务器已停止")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 收到停止信号")
    finally:
        # 停止所有进程
        print("🔄 正在停止服务...")
        
        if mqtt_process and mqtt_process.poll() is None:
            mqtt_process.terminate()
            mqtt_process.wait()
            print("✅ MQTT服务器已停止")
            
        if web_process and web_process.poll() is None:
            web_process.terminate()
            web_process.wait()
            print("✅ Web服务器已停止")
            
        print("👋 所有服务已停止")

if __name__ == "__main__":
    main()
