#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本功能测试脚本
"""

import json
import os
import pickle
import numpy as np
import faiss
import ollama
import time
import re
import sys
import readline
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading

class BasicQASystem:
    def __init__(self):
        self.faiss_index = None
        self.faiss_metadata = None
        self.wiki_pages = []
        self.embedding_model = "nomic-embed-text"
        
        # 性能优化相关
        self.embedding_cache = {}
        self.cache_lock = threading.Lock()
        
        print("🚀 正在初始化基本问答系统...")
        
        # 检查数据文件
        self.check_data_files()
        
        # 检查 Ollama 服务
        self.check_ollama_service()
        
        # 初始化系统
        self.initialize_system()
    
    def check_data_files(self):
        """检查必要的数据文件"""
        required_files = [
            "./data_base/faiss_index.bin",
            "./data_base/faiss_metadata.pkl",
            "./data_base/seeed_wiki_embeddings_db.json"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print("❌ 缺少必要的数据文件:")
            for file in missing_files:
                print(f"   - {file}")
            print("\n💡 请先运行爬虫脚本获取数据:")
            print("   python scrape_with_embeddings.py")
            raise FileNotFoundError(f"缺少数据文件: {', '.join(missing_files)}")
        
        print("✅ 所有必要的数据文件已找到")
    
    def check_ollama_service(self):
        """检查 Ollama 服务状态"""
        try:
            models = ollama.list()
            print(f"✅ Ollama 服务正常，可用模型: {len(models['models'])} 个")
            
            model_names = [model['name'] for model in models['models']]
            if 'nomic-embed-text:latest' not in model_names:
                print("⚠️  未找到 nomic-embed-text 模型，正在安装...")
                ollama.pull('nomic-embed-text')
                print("✅ nomic-embed-text 模型安装完成")
            else:
                print("✅ nomic-embed-text 模型已安装")
                
        except Exception as e:
            print(f"❌ Ollama 服务检查失败: {str(e)}")
            raise
    
    def initialize_system(self):
        """初始化系统"""
        print("🚀 正在初始化系统...")
        
        try:
            # 加载 FAISS 索引
            print("🔍 加载 FAISS 索引...")
            self.faiss_index = faiss.read_index("./data_base/faiss_index.bin")
            print(f"✅ FAISS 索引加载完成: {self.faiss_index.ntotal} 个向量")
            
            # 加载向量元数据
            print("📊 加载向量元数据...")
            with open("./data_base/faiss_metadata.pkl", 'rb') as f:
                self.faiss_metadata = pickle.load(f)
            print(f"✅ 元数据加载完成: {len(self.faiss_metadata)} 条记录")
            
            # 加载 Wiki 页面数据
            print("📚 加载 Wiki 页面数据...")
            with open("./data_base/seeed_wiki_embeddings_db.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.wiki_pages = data['pages']
                self.metadata = data['metadata']
            print(f"✅ 页面数据加载完成: {len(self.wiki_pages)} 个页面")
            
            # 测试 Embedding 模型
            print("🤖 测试 Embedding 模型...")
            test_embedding = self.generate_embedding("test")
            if test_embedding is None:
                raise Exception("Embedding 生成失败")
            print(f"✅ Embedding 模型测试成功: {len(test_embedding)} 维")
            
            print("🎉 系统初始化完成！")
            
        except Exception as e:
            print(f"❌ 系统初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_embedding(self, text):
        """使用 Ollama 生成文本的 embedding 向量"""
        if not text or not text.strip():
            return None
            
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            embedding = response["embedding"]
            embedding = np.array(embedding, dtype=np.float32)
            
            # 归一化
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return None
            embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            print(f"❌ Embedding 生成失败: {str(e)}")
            return None
    
    def search_knowledge_base(self, query, top_k=10):
        """在知识库中搜索相关内容"""
        try:
            query_embedding = self.generate_embedding(query)
            if query_embedding is None:
                return []
            
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.faiss_metadata):
                    metadata = self.faiss_metadata[idx]
                    page_data = self.wiki_pages[idx]
                    
                    results.append({
                        'rank': i + 1,
                        'score': float(score),
                        'title': metadata['title'],
                        'url': metadata['url'],
                        'content': page_data['content'],
                        'content_length': metadata['content_length'],
                        'timestamp': metadata['timestamp']
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ 搜索失败: {str(e)}")
            return []
    
    def ask_question(self, question):
        """提问并获取回答"""
        print(f"\n🤔 用户问题: {question}")
        
        # 搜索知识库
        print("🔍 正在搜索知识库...")
        start_time = time.time()
        search_results = self.search_knowledge_base(question, top_k=5)
        search_time = time.time() - start_time
        
        if not search_results:
            print("❌ 未找到相关信息")
            return
        
        print(f"✅ 搜索完成，耗时: {search_time:.3f} 秒")
        print(f"📊 找到 {len(search_results)} 个相关文档")
        
        # 显示搜索结果
        print(f"🔍 搜索结果:")
        for i, result in enumerate(search_results[:3]):
            print(f"  {i+1}. {result['title']}")
            print(f"     相关度: {result['score']:.3f}")
            print(f"     URL: {result['url']}")
            print()
        
        # 简单回答
        top_result = search_results[0]
        content = top_result['content']
        if content.startswith('[Introduction] '):
            content = content[16:]
        
        answer = f"根据搜索结果，{top_result['title']} 提供了相关信息：{content[:200]}..."
        print(f"\n💬 回答:")
        print(f"{answer}")
    
    def run(self):
        """运行问答系统"""
        print("🤖 Seeed Studio（矽递科技）基本智能助手")
        print("=" * 50)
        print("欢迎使用基本问答系统！")
        print("=" * 50)
        
        sample_questions = [
            "介绍一下XIAO系列产品",
            "Grove传感器模块有什么特点？",
            "SenseCAP的功能是什么？"
        ]
        
        print(f"\n💡 示例问题:")
        for i, question in enumerate(sample_questions, 1):
            print(f"   {i}. {question}")
        
        print(f"\n💬 现在可以开始提问了！")
        print("💡 输入 'quit' 退出")
        print("-" * 50)
        
        try:
            while True:
                try:
                    query = input("\n🤔 请输入您的问题: ").strip()
                    
                    if not query:
                        continue
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        print("👋 感谢使用，再见！")
                        break
                    
                    if query.isdigit() and 1 <= int(query) <= len(sample_questions):
                        query = sample_questions[int(query) - 1]
                        print(f"🔍 选择的问题: {query}")
                    
                    self.ask_question(query)
                    
                except KeyboardInterrupt:
                    print("\n\n⚠️ 用户中断")
                    break
                except Exception as e:
                    print(f"\n❌ 发生错误: {str(e)}")
                    continue
                    
        finally:
            print("👋 程序结束")

def main():
    """主函数"""
    try:
        qa_system = BasicQASystem()
        qa_system.run()
    except Exception as e:
        print(f"\n❌ 系统启动失败: {str(e)}")
        print("请检查数据文件和 Ollama 服务")

if __name__ == "__main__":
    main()


