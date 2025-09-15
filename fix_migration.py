#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迁移问题修复脚本
解决新设备上运行 optimized_qa.py 的问题
"""

import os
import subprocess
import sys
import json

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n🔧 {description}...")
    print(f"   执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ 成功: {result.stdout.strip()}")
            return True
        else:
            print(f"   ❌ 失败: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"   ❌ 执行异常: {str(e)}")
        return False

def check_ollama_status():
    """检查 Ollama 状态"""
    print("🔍 检查 Ollama 状态...")
    
    # 检查 Ollama 是否安装
    if not run_command("which ollama", "检查 Ollama 安装"):
        print("❌ Ollama 未安装")
        return False
    
    # 检查 Ollama 版本
    if not run_command("ollama --version", "检查 Ollama 版本"):
        print("❌ Ollama 版本检查失败")
        return False
    
    # 检查 Ollama 服务状态
    if not run_command("ollama list", "检查 Ollama 服务"):
        print("❌ Ollama 服务未运行")
        return False
    
    return True

def install_embedding_models():
    """安装 embedding 模型"""
    print("\n🔧 安装 embedding 模型...")
    
    models_to_install = [
        "nomic-embed-text",
        "all-MiniLM-L6-v2", 
        "text-embedding-ada-002"
    ]
    
    installed_models = []
    
    for model in models_to_install:
        print(f"\n🔍 安装模型: {model}")
        if run_command(f"ollama pull {model}", f"安装 {model}"):
            installed_models.append(model)
        else:
            print(f"⚠️  {model} 安装失败，继续下一个...")
    
    return installed_models

def verify_models():
    """验证模型可用性"""
    print("\n🔍 验证模型可用性...")
    
    try:
        import ollama
        import numpy as np
        
        models = ollama.list()
        model_names = [model['name'] for model in models['models']]
        
        print(f"   可用模型: {model_names}")
        
        # 测试 embedding 模型
        working_models = []
        for model_name in model_names:
            if 'embed' in model_name.lower() or 'nomic' in model_name.lower():
                try:
                    response = ollama.embeddings(model=model_name, prompt="test")
                    if "embedding" in response and len(response["embedding"]) > 0:
                        embedding = np.array(response["embedding"], dtype=np.float32)
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            working_models.append(model_name)
                            print(f"   ✅ {model_name}: 范数 {norm:.6f}")
                        else:
                            print(f"   ❌ {model_name}: 范数为 0")
                    else:
                        print(f"   ❌ {model_name}: 响应格式错误")
                except Exception as e:
                    print(f"   ❌ {model_name}: {str(e)}")
        
        return working_models
        
    except ImportError as e:
        print(f"❌ 缺少必要库: {str(e)}")
        return []
    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")
        return []

def check_data_files():
    """检查数据文件"""
    print("\n🔍 检查数据文件...")
    
    required_files = [
        "./data_base/faiss_index.bin",
        "./data_base/faiss_metadata.pkl",
        "./data_base/seeed_wiki_embeddings_db.json"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"   ✅ {file}: {size} 字节")
        else:
            print(f"   ❌ {file}: 文件不存在")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  缺少文件: {missing_files}")
        return False
    
    return True

def create_backup_config():
    """创建备用配置"""
    print("\n🔧 创建备用配置...")
    
    config = {
        "embedding_model": "nomic-embed-text:latest",
        "fallback_models": [
            "all-MiniLM-L6-v2",
            "text-embedding-ada-002"
        ],
        "max_retries": 3,
        "debug_mode": True
    }
    
    try:
        with open("migration_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print("   ✅ 配置文件创建成功: migration_config.json")
        return True
    except Exception as e:
        print(f"   ❌ 配置文件创建失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("🚀 迁移问题修复工具")
    print("=" * 50)
    
    # 检查 Ollama 状态
    if not check_ollama_status():
        print("\n💡 请先安装并启动 Ollama:")
        print("   1. 安装: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   2. 启动: ollama serve")
        return
    
    # 安装 embedding 模型
    installed_models = install_embedding_models()
    
    if not installed_models:
        print("❌ 没有成功安装任何 embedding 模型")
        return
    
    # 验证模型
    working_models = verify_models()
    
    if not working_models:
        print("❌ 没有可用的 embedding 模型")
        return
    
    # 检查数据文件
    if not check_data_files():
        print("❌ 数据文件不完整")
        return
    
    # 创建配置
    create_backup_config()
    
    print(f"\n🎉 修复完成！")
    print(f"✅ 可用模型: {working_models}")
    print(f"\n💡 现在可以尝试运行:")
    print(f"   python optimized_qa.py")
    print(f"\n💡 如果仍有问题，请运行诊断工具:")
    print(f"   python test_ollama.py")

if __name__ == "__main__":
    main()
