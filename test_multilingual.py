#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多语言问答功能
"""

from optimized_qa import OptimizedQASystem

def test_multilingual():
    """测试多语言问答功能"""
    print("🧪 测试多语言问答功能")
    print("=" * 50)
    
    try:
        # 初始化系统
        qa_system = OptimizedQASystem()
        
        # 测试问题列表
        test_questions = [
            "介绍一下XIAO系列产品",  # 中文问题
            "What is Grove sensor system?",  # 英文问题
            "SenseCAP有什么功能？",  # 中文问题
            "Tell me about Edge Computing",  # 英文问题
            "reComputer robotics介绍",  # 中英混合问题
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔍 测试 {i}: {question}")
            print("-" * 40)
            
            # 检测语言
            detected_lang = qa_system.detect_language(question)
            print(f"检测语言: {detected_lang}")
            
            # 搜索知识库
            search_results = qa_system.search_knowledge_base(question, top_k=2)
            
            if search_results:
                print(f"找到 {len(search_results)} 个相关文档")
                
                # 生成回答
                answer = qa_system.generate_answer(question, search_results)
                print(f"\n💬 回答 ({detected_lang}):")
                print(answer)
            else:
                print("❌ 未找到相关信息")
            
            print("\n" + "="*50)
        
        print("✅ 多语言测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")

if __name__ == "__main__":
    test_multilingual()


