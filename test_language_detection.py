#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试语言检测功能
"""

import re

def detect_language(text):
    """检测文本语言 - 改进版本"""
    # 检测中文字符
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    english_chars = re.findall(r'[a-zA-Z]', text)
    
    # 计算中英文比例
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    if total_chars == 0:
        return 'en'  # 默认为英文
        
    chinese_ratio = len(chinese_chars) / total_chars
    english_ratio = len(english_chars) / total_chars
    
    # 如果中文字符超过20%，或者中文比例大于英文比例，则认为是中文
    if chinese_ratio > 0.2 or (chinese_ratio > 0 and chinese_ratio > english_ratio):
        return 'zh'
    elif english_ratio > 0.5:
        return 'en'
    else:
        # 如果都不明显，检查是否有中文标点符号
        chinese_punctuation = re.findall(r'[，。！？；：""''（）【】]', text)
        if chinese_punctuation:
            return 'zh'
        return 'en'

def test_language_detection():
    """测试语言检测"""
    test_cases = [
        "什么是gemini2",
        "What is Gemini 2?",
        "Gemini 2是什么？",
        "介绍一下XIAO系列产品",
        "Tell me about XIAO series",
        "XIAO series introduction",
        "边缘计算是什么？",
        "What is edge computing?",
        "Edge computing explanation",
        "你好，世界！Hello World!",
        "Hello 世界！",
        "世界 Hello!",
        "test",
        "测试",
        "123",
        ""
    ]
    
    print("🧪 语言检测测试")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        language = detect_language(text)
        print(f"{i:2d}. '{text}' -> {language}")
    
    print("=" * 50)

if __name__ == "__main__":
    test_language_detection()
