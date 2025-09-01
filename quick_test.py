#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试修复后的问答功能
"""

from optimized_qa import OptimizedQASystem

def quick_test():
    """快速测试"""
    try:
        print("🚀 启动问答系统...")
        qa_system = OptimizedQASystem()
        
        # 测试中文问题
        print("\n🧪 测试中文问题:")
        test_question = "什么是gemini2"
        print(f"问题: {test_question}")
        qa_system.ask_question(test_question)
        
        # 测试英文问题
        print("\n🧪 测试英文问题:")
        test_question_en = "What is Gemini 2?"
        print(f"问题: {test_question_en}")
        qa_system.ask_question(test_question_en)
        
        print("\n✅ 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")

if __name__ == "__main__":
    quick_test()
