#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试输入功能的简单脚本
"""

import sys
import os

def test_input():
    """测试输入功能"""
    print("🧪 输入功能测试")
    print("=" * 30)
    print("请尝试以下操作:")
    print("1. 输入一些文字")
    print("2. 使用退格键删除")
    print("3. 使用方向键移动光标")
    print("4. 使用 Ctrl+U 删除整行")
    print("5. 输入 'quit' 退出")
    print("=" * 30)
    
    while True:
        try:
            user_input = input("\n请输入测试文字: ")
            
            if user_input.lower() == 'quit':
                print("👋 测试结束")
                break
            
            if user_input.strip():
                print(f"✅ 您输入了: '{user_input}'")
                print(f"📏 长度: {len(user_input)} 字符")
            else:
                print("⚠️  输入为空")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ 用户中断")
            break
        except EOFError:
            print("\n\n⚠️ 输入结束")
            break
        except Exception as e:
            print(f"\n❌ 输入错误: {str(e)}")

if __name__ == "__main__":
    test_input()
