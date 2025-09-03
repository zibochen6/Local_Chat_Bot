#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重建向量数据脚本
专门用于修复损坏的向量数据
"""

import os
import json
import pickle
import numpy as np
import faiss
import ollama
from datetime import datetime

def rebuild_vectors():
    """重建所有向量数据"""
    print("🔧 开始重建向量数据...")
    
    # 数据文件路径
    data_dir = "./data_base"
    db_file = f"{data_dir}/seeed_wiki_embeddings_db.json"
    faiss_index_file = f"{data_dir}/faiss_index.bin"
    faiss_metadata_file = f"{data_dir}/faiss_metadata.pkl"
    
    # 检查数据库文件
    if not os.path.exists(db_file):
        print("❌ 数据库文件不存在")
        return
    
    # 加载页面数据
    print("📂 加载页面数据...")
    with open(db_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        pages = data.get('pages', [])
    
    print(f"📊 找到 {len(pages)} 个页面")
    
    # 检查 Ollama 服务
    print("🔍 检查 Ollama 服务...")
    try:
        models = ollama.list()
        if 'nomic-embed-text' not in [m['name'] for m in models['models']]:
            print("📥 安装 nomic-embed-text 模型...")
            ollama.pull('nomic-embed-text')
        print("✅ Ollama 服务正常")
    except Exception as e:
        print(f"❌ Ollama 服务检查失败: {e}")
        return
    
    # 测试 Embedding 生成
    print("🧠 测试 Embedding 生成...")
    try:
        test_response = ollama.embeddings(model='nomic-embed-text', prompt='test')
        dimension = len(test_response['embedding'])
        print(f"✅ Embedding 测试成功，维度: {dimension}")
    except Exception as e:
        print(f"❌ Embedding 测试失败: {e}")
        return
    
    # 重建向量数据
    print("🔄 开始重建向量数据...")
    vectors = []
    metadata = []
    failed_pages = []
    
    for i, page in enumerate(pages):
        try:
            content = page.get('content', '')
            if not content or len(content) < 10:
                print(f"⚠️  页面 {i+1}/{len(pages)}: 内容过短，跳过")
                failed_pages.append(page.get('url', f'page_{i}'))
                continue
            
            # 生成 Embedding
            response = ollama.embeddings(model='nomic-embed-text', prompt=content)
            embedding = np.array(response['embedding'], dtype=np.float32)
            
            # 归一化向量
            embedding = embedding / np.linalg.norm(embedding)
            
            vectors.append(embedding)
            metadata.append({
                'title': page.get('title', ''),
                'url': page.get('url', ''),
                'content_length': len(content),
                'timestamp': datetime.now().isoformat(),
                'language': page.get('language', 'Unknown')
            })
            
            if (i + 1) % 100 == 0:
                print(f"✅ 已处理 {i+1}/{len(pages)} 个页面")
                
        except Exception as e:
            print(f"❌ 页面 {i+1}/{len(pages)} 处理失败: {e}")
            failed_pages.append(page.get('url', f'page_{i}'))
    
    print(f"\n📊 向量重建完成:")
    print(f"   - 成功: {len(vectors)} 个")
    print(f"   - 失败: {len(failed_pages)} 个")
    
    if not vectors:
        print("❌ 没有成功生成任何向量")
        return
    
    # 构建 FAISS 索引
    print("🔍 构建 FAISS 索引...")
    try:
        vectors_array = np.array(vectors, dtype=np.float32)
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(vectors_array)
        
        print(f"✅ FAISS 索引构建完成: {faiss_index.ntotal} 个向量")
    except Exception as e:
        print(f"❌ FAISS 索引构建失败: {e}")
        return
    
    # 保存数据
    print("💾 保存数据...")
    try:
        # 保存 FAISS 索引
        faiss.write_index(faiss_index, faiss_index_file)
        print(f"✅ FAISS 索引已保存: {faiss_index_file}")
        
        # 保存元数据
        with open(faiss_metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ 元数据已保存: {faiss_metadata_file}")
        
        # 更新数据库文件
        data['metadata'].update({
            'total_vectors': len(vectors),
            'vector_dimension': dimension,
            'last_update': datetime.now().isoformat(),
            'rebuild_time': datetime.now().isoformat()
        })
        
        with open(db_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ 数据库已更新: {db_file}")
        
    except Exception as e:
        print(f"❌ 保存数据失败: {e}")
        return
    
    # 统计信息
    print(f"\n📈 重建统计:")
    print(f"   - 总页面数: {len(pages)}")
    print(f"   - 成功向量: {len(vectors)}")
    print(f"   - 失败页面: {len(failed_pages)}")
    print(f"   - 向量维度: {dimension}")
    
    if failed_pages:
        print(f"\n⚠️  失败的页面:")
        for url in failed_pages[:10]:  # 只显示前10个
            print(f"   - {url}")
        if len(failed_pages) > 10:
            print(f"   ... 还有 {len(failed_pages) - 10} 个")
    
    print(f"\n🎉 向量重建完成！")
    print(f"💡 现在可以使用优化版问答系统了")

def check_vectors():
    """检查向量数据状态"""
    print("🔍 检查向量数据状态...")
    
    data_dir = "./data_base"
    db_file = f"{data_dir}/seeed_wiki_embeddings_db.json"
    faiss_index_file = f"{data_dir}/faiss_index.bin"
    faiss_metadata_file = f"{data_dir}/faiss_metadata.pkl"
    
    # 检查文件存在
    print("📁 文件检查:")
    print(f"   数据库文件: {'✅' if os.path.exists(db_file) else '❌'}")
    print(f"   FAISS索引: {'✅' if os.path.exists(faiss_index_file) else '❌'}")
    print(f"   元数据文件: {'✅' if os.path.exists(faiss_metadata_file) else '❌'}")
    
    if not os.path.exists(db_file):
        print("❌ 数据库文件不存在")
        return
    
    # 加载数据
    try:
        with open(db_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            pages = data.get('pages', [])
            metadata = data.get('metadata', {})
        
        print(f"\n📊 数据统计:")
        print(f"   页面数量: {len(pages)}")
        print(f"   向量数量: {metadata.get('total_vectors', 'N/A')}")
        print(f"   向量维度: {metadata.get('vector_dimension', 'N/A')}")
        print(f"   最后更新: {metadata.get('last_update', 'N/A')}")
        
        # 检查 FAISS 索引
        if os.path.exists(faiss_index_file):
            try:
                faiss_index = faiss.read_index(faiss_index_file)
                print(f"   FAISS索引: {faiss_index.ntotal} 个向量")
            except Exception as e:
                print(f"   FAISS索引: ❌ 损坏 ({e})")
        else:
            print("   FAISS索引: ❌ 不存在")
        
        # 检查元数据
        if os.path.exists(faiss_metadata_file):
            try:
                with open(faiss_metadata_file, 'rb') as f:
                    metadata_list = pickle.load(f)
                print(f"   元数据: {len(metadata_list)} 条记录")
            except Exception as e:
                print(f"   元数据: ❌ 损坏 ({e})")
        else:
            print("   元数据: ❌ 不存在")
            
    except Exception as e:
        print(f"❌ 数据检查失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='重建向量数据')
    parser.add_argument('--check', action='store_true', help='仅检查状态')
    parser.add_argument('--rebuild', action='store_true', help='重建向量数据')
    
    args = parser.parse_args()
    
    if args.check:
        check_vectors()
    elif args.rebuild:
        rebuild_vectors()
    else:
        print("请指定操作:")
        print("  --check   检查向量数据状态")
        print("  --rebuild 重建向量数据")

if __name__ == "__main__":
    main()
