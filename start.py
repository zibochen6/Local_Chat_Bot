#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seeed Wiki 系统启动脚本
提供爬虫和问答系统的选择菜单
"""

import os
import sys
import subprocess

def show_menu():
    """显示主菜单"""
    print("🚀 Seeed Wiki 智能问答系统")
    print("=" * 50)
    print("请选择要运行的功能:")
    print("1. 🕷️  爬取 Wiki 内容并生成向量")
    print("2. 🤖 启动问答系统")
    print("3. 📊 查看系统状态")
    print("4. ❓ 查看帮助")
    print("5. 🚪 退出")
    print("=" * 50)

def check_system_status():
    """检查系统状态"""
    print("📊 系统状态检查")
    print("-" * 30)
    
    # 检查必要文件
    required_files = [
        ("faiss_index.bin", "FAISS 向量索引"),
        ("faiss_metadata.pkl", "向量元数据"),
        ("seeed_wiki_embeddings_db.json", "Wiki 页面数据")
    ]
    
    all_files_exist = True
    for filename, description in required_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            size_mb = size / (1024 * 1024)
            print(f"✅ {description}: {filename} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {description}: {filename} (缺失)")
            all_files_exist = False
    
    # 检查 Python 脚本
    scripts = [
        ("scrape_with_embeddings.py", "爬虫脚本"),
        ("optimized_qa.py", "问答系统")
    ]
    
    for script, description in scripts:
        if os.path.exists(script):
            print(f"✅ {description}: {script}")
        else:
            print(f"❌ {description}: {script} (缺失)")
            all_files_exist = False
    
    if all_files_exist:
        print("\n🎉 系统状态正常，可以运行所有功能")
    else:
        print("\n⚠️  系统状态异常，请先运行爬虫脚本获取数据")
    
    return all_files_exist

def run_crawler():
    """运行爬虫"""
    print("🕷️  启动爬虫...")
    print("💡 提示: 按 Ctrl+C 可以安全停止爬虫")
    print("-" * 30)
    
    try:
        subprocess.run([sys.executable, "scrape_with_embeddings.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 爬虫运行失败: {e}")
    except KeyboardInterrupt:
        print("\n⚠️  爬虫被用户中断")

def run_qa_system():
    """运行问答系统"""
    print("🤖 启动问答系统...")
    print("-" * 30)
    
    try:
        subprocess.run([sys.executable, "optimized_qa.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 问答系统启动失败: {e}")
    except KeyboardInterrupt:
        print("\n⚠️  问答系统被用户中断")

def show_help():
    """显示帮助信息"""
    print("❓ 帮助信息")
    print("=" * 50)
    print("🚀 系统功能:")
    print("   1. 爬虫功能: 爬取 Seeed Studio Wiki 内容并生成向量")
    print("   2. 问答功能: 基于向量索引的智能问答系统")
    print()
    print("📁 必要文件:")
    print("   - faiss_index.bin: FAISS 向量索引")
    print("   - faiss_metadata.pkl: 向量元数据")
    print("   - seeed_wiki_embeddings_db.json: Wiki 页面数据")
    print()
    print("🔧 系统要求:")
    print("   - Python 3.8+")
    print("   - Ollama 服务运行中")
    print("   - nomic-embed-text 模型已安装")
    print()
    print("💡 使用建议:")
    print("   - 首次使用请先运行爬虫获取数据")
    print("   - 数据获取完成后即可使用问答系统")
    print("   - 爬虫支持中断恢复，可随时停止")

def main():
    """主函数"""
    while True:
        try:
            show_menu()
            choice = input("\n请选择功能 (1-5): ").strip()
            
            if choice == "1":
                run_crawler()
            elif choice == "2":
                if check_system_status():
                    run_qa_system()
                else:
                    print("\n⚠️  请先运行爬虫脚本获取数据")
            elif choice == "3":
                check_system_status()
            elif choice == "4":
                show_help()
            elif choice == "5":
                print("👋 感谢使用，再见！")
                break
            else:
                print("❌ 无效选择，请输入 1-5")
            
            if choice in ["1", "2", "3", "4"]:
                input("\n按回车键继续...")
                os.system('clear' if os.name == 'posix' else 'cls')
                
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")
            input("按回车键继续...")

if __name__ == "__main__":
    main()
