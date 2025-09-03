#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析前几个高排名页面的内容
"""

import json
import pickle
import numpy as np
import faiss
import ollama

def analyze_top_results():
    """分析前几个高排名页面的内容"""
    print("🔍 分析前几个高排名页面的内容...")
    
    # 加载数据
    print("📂 加载数据...")
    
    # 加载FAISS索引
    faiss_index = faiss.read_index("./data_base/faiss_index.bin")
    print(f"✅ FAISS索引加载完成: {faiss_index.ntotal} 个向量")
    
    # 加载元数据
    with open("./data_base/faiss_metadata.pkl", 'rb') as f:
        faiss_metadata = pickle.load(f)
    print(f"✅ 元数据加载完成: {len(faiss_metadata)} 条记录")
    
    # 加载页面数据
    with open("./data_base/seeed_wiki_embeddings_db.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        wiki_pages = data.get('pages', [])
    print(f"✅ 页面数据加载完成: {len(wiki_pages)} 个页面")
    
    # 找到矽递科技页面
    target_url = 'https://wiki.seeedstudio.com/cn/Getting_Started/'
    target_idx = None
    for i, page in enumerate(wiki_pages):
        if page.get('url') == target_url:
            target_idx = i
            break
    
    if target_idx is None:
        print("❌ 未找到矽递科技页面")
        return
    
    print(f"✅ 找到矽递科技页面，索引: {target_idx}")
    
    # 测试查询
    query = "介绍一下矽递科技"
    print(f"\n🔍 测试查询: '{query}'")
    
    try:
        # 生成查询向量
        response = ollama.embeddings(model='nomic-embed-text', prompt=query)
        query_embedding = np.array(response['embedding'], dtype=np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # 搜索前100个结果
        query_embedding_reshaped = query_embedding.reshape(1, -1)
        scores, indices = faiss_index.search(query_embedding_reshaped, 100)
        
        # 查找目标页面在搜索结果中的位置
        target_rank = None
        for i, idx in enumerate(indices[0]):
            if idx == target_idx:
                target_rank = i + 1
                break
        
        if target_rank:
            print(f"✅ 目标页面排名: {target_rank}")
            print(f"✅ 目标页面相关度: {scores[0][target_rank-1]:.6f}")
        else:
            print("❌ 目标页面在前100个结果中未找到")
            return
        
        # 分析前10个结果
        print(f"\n📊 分析前10个结果:")
        for i in range(10):
            idx = indices[0][i]
            score = scores[0][i]
            if idx < len(wiki_pages):
                page = wiki_pages[idx]
                title = page.get('title', 'N/A')
                url = page.get('url', 'N/A')
                content = page.get('content', '')
                
                print(f"\n{i+1}. 相关度: {score:.6f}")
                print(f"   标题: {title}")
                print(f"   URL: {url}")
                print(f"   内容预览: {content[:300]}...")
                
                # 检查是否包含关键词
                keywords = ['矽递', '科技', '公司', '介绍', '简介', '关于']
                found_keywords = []
                for keyword in keywords:
                    if keyword in content:
                        found_keywords.append(keyword)
                
                if found_keywords:
                    print(f"   包含关键词: {', '.join(found_keywords)}")
                else:
                    print(f"   未包含常见关键词")
        
        # 分析矽递科技页面
        print(f"\n📄 分析矽递科技页面:")
        target_page = wiki_pages[target_idx]
        target_title = target_page.get('title', 'N/A')
        target_url = target_page.get('url', 'N/A')
        target_content = target_page.get('content', '')
        
        print(f"   标题: {target_title}")
        print(f"   URL: {target_url}")
        print(f"   内容预览: {target_content[:500]}...")
        
        # 检查是否包含关键词
        keywords = ['矽递', '科技', '公司', '介绍', '简介', '关于']
        found_keywords = []
        for keyword in keywords:
            if keyword in target_content:
                found_keywords.append(keyword)
        
        if found_keywords:
            print(f"   包含关键词: {', '.join(found_keywords)}")
        else:
            print(f"   未包含常见关键词")
        
        # 分析查询向量
        print(f"\n🔍 分析查询向量:")
        print(f"   查询: '{query}'")
        print(f"   查询向量范数: {np.linalg.norm(query_embedding):.6f}")
        
        # 检查查询是否包含关键词
        query_keywords = []
        for keyword in keywords:
            if keyword in query:
                query_keywords.append(keyword)
        
        if query_keywords:
            print(f"   查询包含关键词: {', '.join(query_keywords)}")
        else:
            print(f"   查询未包含常见关键词")
        
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 分析完成！")

if __name__ == "__main__":
    analyze_top_results()
