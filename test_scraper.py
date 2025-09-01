#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Seeed Wiki 爬虫功能
"""

import os
import json
from scrape_with_embeddings import OptimizedWikiScraper

def test_scraper():
    """测试爬虫功能"""
    print("🧪 开始测试 Seeed Wiki 爬虫...")
    
    # 创建爬虫实例
    scraper = OptimizedWikiScraper()
    
    # 测试数据目录创建
    print(f"📁 数据目录: {scraper.data_dir}")
    print(f"📄 数据库文件: {scraper.db_file}")
    print(f"🔍 FAISS索引文件: {scraper.faiss_index_file}")
    
    # 检查现有数据
    if os.path.exists(scraper.db_file):
        with open(scraper.db_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            pages = data.get('pages', [])
            metadata = data.get('metadata', {})
            print(f"📊 现有数据统计:")
            print(f"   - 总页面数: {len(pages)}")
            print(f"   - 向量维度: {metadata.get('vector_dimension', 'N/A')}")
            print(f"   - 最后更新: {metadata.get('last_update', 'N/A')}")
            print(f"   - 支持语言: {metadata.get('languages', [])}")
            
            # 语言统计
            chinese_pages = len([p for p in pages if p.get('language') == '中文'])
            english_pages = len([p for p in pages if p.get('language') == 'English'])
            print(f"   - 中文页面: {chinese_pages}")
            print(f"   - 英文页面: {english_pages}")
    else:
        print("📂 没有现有数据，将进行首次爬取")
    
    # 测试URL验证
    test_urls = [
        "https://wiki.seeedstudio.com/Getting_Started/",
        "https://wiki.seeedstudio.com/zh/Getting_Started/",
        "https://wiki.seeedstudio.com/XIAO/",
        "https://wiki.seeedstudio.com/zh/XIAO/",
        "https://wiki.seeedstudio.com/test.pdf",  # 应该被排除
        "https://wiki.seeedstudio.com/api/test",  # 应该被排除
    ]
    
    print("\n🔗 测试URL验证:")
    for url in test_urls:
        is_valid = scraper.is_valid_wiki_url(url)
        print(f"   {url}: {'✅' if is_valid else '❌'}")
    
    # 测试embedding生成
    print("\n🧠 测试Embedding生成:")
    test_texts = [
        "Hello, this is a test in English.",
        "你好，这是一个中文测试。",
        "This is a mixed text with 中文 and English content."
    ]
    
    for text in test_texts:
        embedding = scraper.generate_embedding(text)
        if embedding is not None:
            print(f"   ✅ '{text[:30]}...': {embedding.shape}")
        else:
            print(f"   ❌ '{text[:30]}...': 生成失败")
    
    print("\n✅ 测试完成！")
    print("\n💡 使用方法:")
    print("   python scrape_with_embeddings.py --mode incremental  # 增量更新")
    print("   python scrape_with_embeddings.py --mode full         # 完整爬取")
    print("   python scrape_with_embeddings.py --mode schedule     # 定时更新")
    print("   python scrape_with_embeddings.py --mode monitor      # 持续监控模式")
    print("\n📊 监控模式功能:")
    print("   - 实时检查新页面并更新到本地")
    print("   - 每天凌晨12点自动进行完整数据库更新")
    print("   - 每30分钟快速检查一次新页面")

if __name__ == "__main__":
    test_scraper()
