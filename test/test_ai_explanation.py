#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试AI讲解功能
"""

import requests
import json

def test_ai_explanation():
    """测试AI讲解API"""
    base_url = "http://192.168.6.236:5000"
    
    print("🧪 测试AI讲解功能")
    print("=" * 50)
    
    # 测试中文讲解
    print("\n🇨🇳 测试中文AI讲解...")
    try:
        response = requests.post(
            f"{base_url}/api/ai_explanation",
            json={
                "product_id": "001",
                "language": "zh"
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 中文AI讲解生成成功!")
            print(f"📝 讲解内容预览:")
            print(data['ai_explanation'][:200] + "...")
        else:
            print(f"❌ 中文AI讲解失败: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ 中文AI讲解请求异常: {e}")
    
    # 测试英文讲解
    print("\n🇺🇸 测试英文AI讲解...")
    try:
        response = requests.post(
            f"{base_url}/api/ai_explanation",
            json={
                "product_id": "001",
                "language": "en"
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 英文AI讲解生成成功!")
            print(f"📝 讲解内容预览:")
            print(data['ai_explanation'][:200] + "...")
        else:
            print(f"❌ 英文AI讲解失败: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ 英文AI讲解请求异常: {e}")
    
    # 测试产品页面访问
    print("\n🌐 测试产品页面访问...")
    try:
        response = requests.get(f"{base_url}/product/001")
        if response.status_code == 200:
            print("✅ 产品页面访问成功!")
        else:
            print(f"❌ 产品页面访问失败: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 产品页面访问异常: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 测试完成!")
    print(f"💡 访问地址: {base_url}")
    print(f"📱 产品页面: {base_url}/product/001")

if __name__ == "__main__":
    test_ai_explanation()
