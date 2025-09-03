#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强制检查脚本 - 检查所有页面并更新缺失的内容
用于处理中断后的数据完整性检查
"""

import os
import json
import time
from datetime import datetime
from scrape_with_embeddings import OptimizedWikiScraper

def force_check_all_pages():
    """强制检查所有页面并更新缺失的内容"""
    print("🔧 开始强制检查所有页面...")
    print("📊 功能说明:")
    print("   - 忽略24小时更新限制")
    print("   - 检查所有页面的完整性")
    print("   - 更新缺失或损坏的页面")
    print("   - 重新生成缺失的向量")
    
    # 创建爬虫实例
    scraper = OptimizedWikiScraper()
    
    # 检查现有数据
    print(f"\n📂 现有数据统计:")
    print(f"   - 页面数量: {len(scraper.all_content)}")
    print(f"   - 向量数量: {len([v for v in scraper.faiss_vectors if v is not None])}")
    print(f"   - URL哈希数量: {len(scraper.url_hashes)}")
    
    # 检查数据完整性
    print(f"\n🔍 检查数据完整性...")
    
    # 检查向量和页面数量是否匹配
    valid_vectors = [v for v in scraper.faiss_vectors if v is not None]
    if len(valid_vectors) != len(scraper.all_content):
        print(f"⚠️  向量数量不匹配: 页面 {len(scraper.all_content)}, 向量 {len(valid_vectors)}")
        print("🔄 需要重新生成缺失的向量")
        need_rebuild = True
    else:
        print("✅ 向量数量匹配")
        need_rebuild = False
    
    # 检查URL哈希完整性
    missing_hashes = []
    for page in scraper.all_content:
        url = page.get('url')
        if url and url not in scraper.url_hashes:
            missing_hashes.append(url)
    
    if missing_hashes:
        print(f"⚠️  发现 {len(missing_hashes)} 个页面缺少哈希值")
        print("🔄 需要重新计算哈希值")
        need_rebuild = True
    else:
        print("✅ URL哈希值完整")
    
    # 检查页面内容完整性
    incomplete_pages = []
    for page in scraper.all_content:
        if not page.get('content') or len(page.get('content', '')) < 50:
            incomplete_pages.append(page.get('url'))
    
    if incomplete_pages:
        print(f"⚠️  发现 {len(incomplete_pages)} 个页面内容不完整")
        print("🔄 需要重新爬取这些页面")
        need_rebuild = True
    else:
        print("✅ 页面内容完整")
    
    if not need_rebuild:
        print("\n✅ 数据完整性检查通过，无需重建")
        print("💡 如需强制更新，请使用 --force-check 参数")
        return
    
    print(f"\n🔄 开始强制更新...")
    
    # 强制运行增量更新
    try:
        scraper.run_incremental_update(force_check=True)
        print("\n✅ 强制更新完成")
    except Exception as e:
        print(f"\n❌ 强制更新失败: {str(e)}")
        return
    
    # 更新后的统计
    print(f"\n📊 更新后统计:")
    print(f"   - 页面数量: {len(scraper.all_content)}")
    print(f"   - 向量数量: {len([v for v in scraper.faiss_vectors if v is not None])}")
    print(f"   - URL哈希数量: {len(scraper.url_hashes)}")
    
    # 语言统计
    chinese_pages = len([p for p in scraper.all_content if p.get('language') == '中文'])
    english_pages = len([p for p in scraper.all_content if p.get('language') == 'English'])
    
    print(f"   - 中文页面: {chinese_pages}")
    print(f"   - 英文页面: {english_pages}")
    
    print(f"\n🎉 强制检查完成！")
    print(f"💡 现在可以使用优化版问答系统了")

def check_specific_pages():
    """检查特定页面的完整性"""
    print("🔍 检查特定页面完整性...")
    
    scraper = OptimizedWikiScraper()
    
    # 检查重要页面是否存在
    important_urls = [
        "https://wiki.seeedstudio.com/Getting_Started/",
        "https://wiki.seeedstudio.com/XIAO/",
        "https://wiki.seeedstudio.com/Grove/",
        "https://wiki.seeedstudio.com/SenseCAP/",
        "https://wiki.seeedstudio.com/reComputer/",
        "https://wiki.seeedstudio.com/zh/Getting_Started/",
        "https://wiki.seeedstudio.com/zh/XIAO/",
    ]
    
    missing_pages = []
    for url in important_urls:
        found = False
        for page in scraper.all_content:
            if page.get('url') == url:
                found = True
                content_length = len(page.get('content', ''))
                if content_length < 50:
                    print(f"⚠️  {url}: 内容过短 ({content_length} 字符)")
                else:
                    print(f"✅ {url}: 正常 ({content_length} 字符)")
                break
        
        if not found:
            missing_pages.append(url)
            print(f"❌ {url}: 缺失")
    
    if missing_pages:
        print(f"\n⚠️  发现 {len(missing_pages)} 个重要页面缺失")
        print("建议运行强制检查来补充这些页面")
    else:
        print(f"\n✅ 所有重要页面都存在")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='强制检查脚本')
    parser.add_argument('--check-only', action='store_true', 
                       help='仅检查，不执行更新')
    parser.add_argument('--check-specific', action='store_true',
                       help='检查特定重要页面')
    
    args = parser.parse_args()
    
    if args.check_specific:
        check_specific_pages()
    elif args.check_only:
        print("🔍 仅检查模式...")
        scraper = OptimizedWikiScraper()
        print(f"页面数量: {len(scraper.all_content)}")
        print(f"向量数量: {len([v for v in scraper.faiss_vectors if v is not None])}")
        print(f"URL哈希数量: {len(scraper.url_hashes)}")
    else:
        force_check_all_pages()

if __name__ == "__main__":
    main()
