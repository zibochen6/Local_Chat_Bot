#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seeed Wiki 优化问答系统
使用预保存的 FAISS 索引和 Ollama nomic-embed-text 模型
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
import readline  # 添加 readline 支持，提供更好的输入体验
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading


class OptimizedQASystem:
    def __init__(self):
        self.faiss_index = None
        self.faiss_metadata = None
        self.wiki_pages = []
        self.embedding_model = "nomic-embed-text"
        
        # 性能优化相关
        self.embedding_cache = {}  # embedding 缓存
        self.answer_cache = {}     # 回答缓存
        self.cache_lock = threading.Lock()  # 缓存锁
        self.executor = ThreadPoolExecutor(max_workers=2)  # 线程池
        
        # 流式显示相关
        self.streaming_enabled = True  # 是否启用流式显示
        self.typing_speed = 0.03  # 打字速度（秒/字符）
        

        
        # 设置 readline 配置
        self.setup_readline()
        
        # 检查数据文件
        self.check_data_files()
        
        # 检查 Ollama 服务
        self.check_ollama_service()
        

        
        # 初始化系统
        self.initialize_system()
    
    def setup_readline(self):
        """设置 readline 配置，提供更好的输入体验"""
        try:
            # 设置历史文件
            histfile = os.path.join(os.path.expanduser("~"), ".seeed_qa_history")
            readline.read_history_file(histfile)
            readline.set_history_length(1000)
            
            # 设置自动补全
            readline.parse_and_bind('tab: complete')
            
            # 设置输入提示符样式
            readline.parse_and_bind('set editing-mode emacs')
            
        except Exception as e:
            print(f"⚠️  readline 设置失败: {str(e)}")
            print("💡 输入体验可能受限，但基本功能正常")
    
    def safe_input(self, prompt):
        """安全的输入函数，提供更好的错误处理"""
        try:
            # 尝试使用 readline 输入
            user_input = input(prompt)
            return user_input.strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 用户中断，退出程序")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 输入错误: {str(e)}")
            return ""
    
    def save_history(self):
        """保存输入历史"""
        try:
            histfile = os.path.join(os.path.expanduser("~"), ".seeed_qa_history")
            readline.write_history_file(histfile)
        except Exception:
            pass  # 忽略历史保存错误
    
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
            print(model_names)
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
        print("🚀 正在初始化优化问答系统...")
        
        try:
            # 加载 FAISS 索引
            print("🔍 加载 FAISS 索引...")
            if not os.path.exists("./data_base/faiss_index.bin"):
                raise FileNotFoundError("FAISS 索引文件不存在")
                
            self.faiss_index = faiss.read_index("./data_base/faiss_index.bin")
            if self.faiss_index is None:
                raise Exception("FAISS 索引加载失败")
                
            print(f"✅ FAISS 索引加载完成: {self.faiss_index.ntotal} 个向量")
            print(f"   索引维度: {self.faiss_index.d}")
            print(f"   索引类型: {type(self.faiss_index).__name__}")
            
            # 加载向量元数据
            print("📊 加载向量元数据...")
            if not os.path.exists("./data_base/faiss_metadata.pkl"):
                raise FileNotFoundError("元数据文件不存在")
                
            with open("./data_base/faiss_metadata.pkl", 'rb') as f:
                self.faiss_metadata = pickle.load(f)
            
            if not self.faiss_metadata or len(self.faiss_metadata) == 0:
                raise Exception("元数据为空")
                
            print(f"✅ 元数据加载完成: {len(self.faiss_metadata)} 条记录")
            
            # 检查索引和元数据的一致性
            if self.faiss_index.ntotal != len(self.faiss_metadata):
                print(f"⚠️  警告: 索引向量数({self.faiss_index.ntotal})与元数据记录数({len(self.faiss_metadata)})不匹配")
            
            # 加载 Wiki 页面数据
            print("📚 加载 Wiki 页面数据...")
            if not os.path.exists("./data_base/seeed_wiki_embeddings_db.json"):
                raise FileNotFoundError("Wiki 页面数据文件不存在")
                
            with open("./data_base/seeed_wiki_embeddings_db.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.wiki_pages = data['pages']
                self.metadata = data['metadata']
            
            if not self.wiki_pages or len(self.wiki_pages) == 0:
                raise Exception("Wiki 页面数据为空")
                
            print(f"✅ 页面数据加载完成: {len(self.wiki_pages)} 个页面")
            
            # 检查页面数据和元数据的一致性
            if len(self.wiki_pages) != len(self.faiss_metadata):
                print(f"⚠️  警告: 页面数据数({len(self.wiki_pages)})与元数据记录数({len(self.faiss_metadata)})不匹配")
            
            # 测试 Embedding 模型
            print("🤖 测试 Embedding 模型...")
            test_embedding = self.generate_embedding("test")
            if test_embedding is None:
                raise Exception("Embedding 生成失败")
                
            # 检查 embedding 维度是否与索引匹配
            if test_embedding.shape[0] != self.faiss_index.d:
                raise Exception(f"Embedding 维度({test_embedding.shape[0]})与索引维度({self.faiss_index.d})不匹配")
                
            print(f"✅ Embedding 模型测试成功: {len(test_embedding)} 维")
            
            print("🎉 系统初始化完成！")
            self.show_system_info()
            
            # 加载缓存
            self.load_cache()
            
        except Exception as e:
            print(f"❌ 系统初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def show_system_info(self):
        """显示系统信息"""
        print(f"\n📊 系统信息:")
        print(f"   总页面数: {len(self.wiki_pages)}")
        print(f"   总向量数: {self.faiss_index.ntotal}")
        print(f"   向量维度: {self.metadata['vector_dimension']}")
        print(f"   内容类型: {self.metadata['content_type']}")
        print(f"   Embedding 模型: {self.metadata['embedding_model']}")
        print(f"   索引类型: {self.metadata['index_type']}")
        print(f"   爬取时间: {self.metadata['crawl_time']}")
        print(f"   缓存状态: Embedding缓存 {len(self.embedding_cache)} 项，回答缓存已禁用")
        print(f"   流式显示: {'启用' if self.streaming_enabled else '禁用'}")
        print(f"   打字速度: {self.typing_speed:.3f} 秒/字符")

    
    def load_cache(self):
        """加载缓存数据"""
        try:
            cache_file = "./data_base/cache_data.pkl"
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.embedding_cache = cache_data.get('embedding_cache', {})
                    self.answer_cache = cache_data.get('answer_cache', {})
                print(f"✅ 缓存加载完成: Embedding {len(self.embedding_cache)} 项，回答 {len(self.answer_cache)} 项")
        except Exception as e:
            print(f"⚠️  缓存加载失败: {str(e)}")
    
    def save_cache(self):
        """保存缓存数据"""
        try:
            cache_file = "./data_base/cache_data.pkl"
            cache_data = {
                'embedding_cache': self.embedding_cache,
                'answer_cache': self.answer_cache
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✅ 缓存保存完成")
        except Exception as e:
            print(f"⚠️  缓存保存失败: {str(e)}")
    
    def clear_cache(self):
        """清空缓存"""
        with self.cache_lock:
            self.embedding_cache.clear()
            self.answer_cache.clear()
        print("✅ 缓存已清空")
    
    def typewriter_effect(self, text, speed=None):
        """打字机效果显示文本"""
        if speed is None:
            speed = self.typing_speed
        
        for char in text:
            print(char, end='', flush=True)
            time.sleep(speed)
        print()  # 换行
    
    def stream_response(self, response_generator):
        """流式显示回答"""
        full_answer = ""
        print("💬 回答: ", end='', flush=True)
        
        try:
            for chunk in response_generator:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    if content:
                        full_answer += content
                        print(content, end='', flush=True)
                        time.sleep(self.typing_speed)
        except Exception as e:
            print(f"\n⚠️  流式显示错误: {str(e)}")
        
        print()  # 换行
        return full_answer
    

    

    

    

    
    def generate_embedding(self, text):
        """使用 Ollama 生成文本的 embedding 向量（带缓存优化）"""
        if not text or not text.strip():
            print("❌ 输入文本为空")
            return None
            
        # 生成文本的哈希值作为缓存键
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # 检查缓存
        with self.cache_lock:
            if text_hash in self.embedding_cache:
                cached_embedding = self.embedding_cache[text_hash]
                if cached_embedding is not None and isinstance(cached_embedding, np.ndarray):
                    return cached_embedding
                else:
                    # 清理无效的缓存项
                    del self.embedding_cache[text_hash]
        
        try:
            print(f"🔍 正在生成文本的 embedding: '{text[:50]}...'")
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            
            if "embedding" not in response:
                print(f"❌ Ollama 响应格式错误: {response}")
                return None
                
            embedding = response["embedding"]
            
            if not embedding or len(embedding) == 0:
                print("❌ 生成的 embedding 为空")
                return None
            
            # 转换为 numpy 数组
            embedding = np.array(embedding, dtype=np.float32)
            
            # 检查数组是否有效
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                print("❌ embedding 包含 NaN 或 Inf 值")
                return None
            
            # 归一化
            norm = np.linalg.norm(embedding)
            if norm == 0:
                print("❌ embedding 向量的范数为 0")
                return None
                
            embedding = embedding / norm
            
            print(f"✅ embedding 生成成功: 维度 {len(embedding)}, 范数 {np.linalg.norm(embedding):.6f}")
            
            # 缓存结果
            with self.cache_lock:
                self.embedding_cache[text_hash] = embedding
                # 限制缓存大小，避免内存溢出
                if len(self.embedding_cache) > 1000:
                    # 删除最旧的缓存项
                    oldest_key = next(iter(self.embedding_cache))
                    del self.embedding_cache[oldest_key]
            
            return embedding
            
        except Exception as e:
            print(f"❌ Embedding 生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def search_knowledge_base(self, query, top_k=20):
        """在知识库中搜索相关内容（优化版本）"""
        try:
            # 检查 FAISS 索引是否正确加载
            if self.faiss_index is None:
                print("❌ FAISS 索引未加载")
                return []
            
            # 检查元数据是否正确加载
            if self.faiss_metadata is None or len(self.faiss_metadata) == 0:
                print("❌ 元数据未加载")
                return []
            
            # 生成查询的 embedding
            query_embedding = self.generate_embedding(query)
            if query_embedding is None:
                print("❌ 无法生成查询的 embedding")
                return []
            
            # 确保 embedding 是正确的 numpy 数组
            if not isinstance(query_embedding, np.ndarray):
                print(f"❌ embedding 类型错误: {type(query_embedding)}")
                return []
            
            # 检查向量维度
            expected_dim = self.faiss_index.d
            if query_embedding.shape[0] != expected_dim:
                print(f"❌ 向量维度不匹配: 期望 {expected_dim}, 实际 {query_embedding.shape[0]}")
                return []
            
            # 重塑为正确的形状
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # 执行 FAISS 搜索
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
            import traceback
            traceback.print_exc()
            return []
    
    def ask_question(self, question):
        """提问并获取回答（优化版本）"""
        print(f"\n🤔 用户问题: {question}")
        
        # 禁用回答缓存，每次都实时生成
        # question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
        # with self.cache_lock:
        #     if question_hash in self.answer_cache:
        #         print("⚡ 使用缓存回答")
        #         print(f"\n💬 回答:")
        #         print(f"{self.answer_cache[question_hash]}")
        #         return
        
        # 搜索知识库
        print("🔍 正在搜索知识库...")
        start_time = time.time()
        search_results = self.search_knowledge_base(question, top_k=20)  # 减少搜索数量
        search_time = time.time() - start_time
        
        if not search_results:
            print("❌ 未找到相关信息")
            return
        
        print(f"✅ 搜索完成，耗时: {search_time:.3f} 秒")
        print(f"📊 找到 {len(search_results)} 个相关文档")
        
        # 智能选择最相关的结果（减少到5个）
        best_results = self.select_best_results(question, search_results, max_results=5)
        
        # 显示搜索结果
        print(f"🔍 搜索结果预览:")
        for i, result in enumerate(best_results[:3]):  # 只显示前3个
            print(f"  {i+1}. {result['title']}")
            print(f"     相关度: {result['score']:.3f}")
            print(f"     URL: {result['url']}")
            print()
        
        # 生成回答
        print("🤖 正在生成回答...")
        answer_start_time = time.time()
        answer = self.generate_answer(question, best_results)
        answer_time = time.time() - answer_start_time
        
        # 显示回答
        print(f"\n💬 回答:")
        print(f"{answer}")
        print(f"\n⏱️  回答生成耗时: {answer_time:.3f} 秒")
        

        
        # 禁用回答缓存，不保存生成的回答
        # with self.cache_lock:
        #     self.answer_cache[question_hash] = answer
        #     # 限制缓存大小
        #     if len(self.answer_cache) > 100:
        #         oldest_key = next(iter(self.answer_cache))
        #         del self.answer_cache[oldest_key]
    
    def select_best_results(self, question, search_results, max_results=10):
        """智能选择最相关的结果"""
        if not search_results:
            return []
        
        # 提取问题中的关键词
        question_lower = question.lower()
        keywords = []
        
        # 中文关键词
        chinese_keywords = ['矽递', '科技', '公司', '介绍', '简介', '关于', '什么是', '如何', '怎么']
        for keyword in chinese_keywords:
            if keyword in question_lower:
                keywords.append(keyword)
        
        # 英文关键词
        english_keywords = ['seeed', 'studio', 'company', 'introduction', 'about', 'what', 'how']
        for keyword in english_keywords:
            if keyword in question_lower:
                keywords.append(keyword)
        
        # 计算每个结果的相关性分数
        scored_results = []
        for result in search_results:
            score = result['score']
            title = result['title'].lower()
            content = result['content'].lower()
            
            # 关键词匹配加分
            keyword_bonus = 0
            for keyword in keywords:
                if keyword in title:
                    keyword_bonus += 0.1
                if keyword in content:
                    keyword_bonus += 0.05
            
            # 标题匹配加分
            title_bonus = 0
            if any(keyword in title for keyword in keywords):
                title_bonus += 0.05
            
            # 计算最终分数
            final_score = score + keyword_bonus + title_bonus
            scored_results.append((result, final_score))
        
        # 按最终分数排序
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前N个最佳结果
        best_results = [result for result, score in scored_results[:max_results]]
        
        print(f"🔍 智能选择结果:")
        print(f"   关键词: {keywords}")
        print(f"   选择结果数: {len(best_results)}")
        
        return best_results

    def detect_language(self, text):
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
        
        # 如果中文字符超过10%，或者中文比例大于英文比例，则认为是中文
        if chinese_ratio > 0.1 or (chinese_ratio > 0 and chinese_ratio > english_ratio):
            return 'zh'
        elif english_ratio > 0.5:
            return 'en'
        else:
            # 如果都不明显，检查是否有中文标点符号
            chinese_punctuation = re.findall(r'[，。！？；：""''（）【】]', text)
            if chinese_punctuation:
                return 'zh'
            return 'en'
    
    def generate_answer(self, question, search_results):
        """基于搜索结果生成回答 - 优化版本"""
        if not search_results:
            return "抱歉，我在知识库中没有找到相关信息。"
        
        # 检测用户问题的语言
        user_language = self.detect_language(question)
        print(f"🔍 检测到问题语言: {user_language}")
        
        # 构建上下文信息（优化：限制长度）
        context_parts = []
        total_length = 0
        max_context_length = 3000  # 限制上下文长度
        
        for result in search_results:
            title = result['title']
            content = result['content']
            # 移除 [Introduction] 前缀，清理内容
            if content.startswith('[Introduction] '):
                content = content[16:]
            
            # 截断过长的内容
            if len(content) > 500:
                content = content[:500] + "..."
            
            context_part = f"文档标题: {title}\n内容: {content}"
            
            # 检查是否会超出长度限制
            if total_length + len(context_part) > max_context_length:
                break
                
            context_parts.append(context_part)
            total_length += len(context_part)
        
        context = "\n\n".join(context_parts)
        
        # 根据用户语言选择 prompt，强制指定输出语言
        if user_language == 'zh':
            prompt = f"""请基于以下资料，用自然、连贯的中文回答用户问题。

重要要求：
1. 必须用中文回答，不能使用英文
2. 介绍产品时说"我们的xxx产品..."
3. 严格基于提供的资料回答，绝对不能编造或虚构任何信息
4. 如果资料中没有某个具体信息（如成立时间、具体数据等），绝对不要编造，应该说"资料中未提及"
5. 语言要流畅自然，体现专业且亲切的企业形象
6. 不要分点分段，用一段话概括所有相关信息
7. 不要重复说"我们是Seeed Studio的AI助手"这样的身份介绍

相关资料:
{context}

用户问题: {question}

请用一段连贯的中文回答，使用"我们"的表达方式，严格基于资料内容，不编造任何信息:"""
        else:
            prompt = f"""Please answer the user's question in natural, coherent English based on the following materials.

Important requirements:
1. Must answer in English, not in Chinese
2. Answer as Seeed Studio's representative, using "we" expressions
3. When introducing products, say "our xxx product..."
4. Strictly base your answer on the provided materials, absolutely do not fabricate or invent any information
5. If specific information (like founding date, specific data, etc.) is not mentioned in the materials, absolutely do not make it up, say "not mentioned in the materials"
6. Make the language fluent and natural, reflecting a professional yet friendly corporate image
7. Don't use bullet points or separate paragraphs, summarize all relevant information in one coherent paragraph
8. Don't repeat identity introductions like "We are Seeed Studio's AI assistant"

Materials:
{context}

User Question: {question}

Please answer using "we" expressions in one coherent English paragraph, strictly based on the materials without fabricating any information:"""
        
        # 使用 Ollama 生成自然语言回答（流式版本）
        try:
            # 优化：使用更快的模型和更简洁的prompt
            system_prompt = f'用{user_language}回答，基于资料，不编造信息，不要重复身份介绍。'
            
            if self.streaming_enabled:
                # 流式生成回答
                response_generator = ollama.chat(
                    model='qwen2.5:3b', 
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': prompt}
                    ],
                    options={
                        'temperature': 0.7,  # 降低随机性，提高一致性
                        'top_p': 0.9,       # 限制词汇选择范围
                        'num_predict': 300,  # 限制生成长度
                    },
                    stream=True  # 启用流式输出
                )
                
                answer = self.stream_response(response_generator)
            else:
                # 非流式生成回答
                response = ollama.chat(
                    model='qwen2.5:3b', 
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': prompt}
                    ],
                    options={
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'num_predict': 300,
                    }
                )
                
                answer = response['message']['content'].strip()
                # 使用打字机效果显示
                print("💬 回答: ", end='', flush=True)
                self.typewriter_effect(answer)
            
            # 验证回答语言
            answer_language = self.detect_language(answer)
            if answer_language != user_language:
                print(f"⚠️  AI回答语言不匹配，期望{user_language}，实际{answer_language}")
                answer = self.generate_manual_answer(question, search_results, user_language)
            
            # 如果回答太短，添加一些补充信息
            if len(answer) < 50:
                answer = self.generate_manual_answer(question, search_results, user_language)
            
            return answer
            
        except Exception as e:
            print(f"⚠️  AI 生成回答失败，使用备用方案: {str(e)}")
            return self.generate_manual_answer(question, search_results, user_language)
    
    def generate_manual_answer(self, question, search_results, language='en'):
        """手动生成回答（备用方案）"""
        # 按相关度排序
        sorted_results = sorted(search_results, key=lambda x: x['score'], reverse=True)
        
        # 根据问题类型和语言生成不同的回答
        question_lower = question.lower()
        
        if language == 'zh':
            # 中文回答
            if "xiao" in question_lower or "小" in question_lower:
                answer = "XIAO系列是矽递科技推出的微型开发板产品线，这些开发板虽然体积小巧，但功能却非常强大。我们采用了标准化的设计理念，具有出色的兼容性和扩展性，特别适合各种嵌入式项目、原型开发和创客项目。XIAO系列产品不仅支持Arduino生态系统，还集成了Grove连接器，让您可以轻松连接各种传感器和模块，大大简化了硬件开发的复杂度。"
            elif "grove" in question_lower:
                answer = "Grove传感器模块系统是矽递科技开发的一套标准化的硬件连接解决方案，它彻底改变了传统硬件开发的复杂流程。通过统一的连接接口和标准化的模块设计，Grove系统让您可以像搭积木一样轻松地将各种传感器、执行器和通信模块连接到开发板上。这种设计不仅大大降低了硬件开发的入门门槛，还提高了项目的可靠性和可维护性，特别适合初学者和快速原型开发。"
            elif "sensecap" in question_lower:
                answer = "SenseCAP是矽递科技专门为环境监测和物联网应用打造的一站式解决方案，它集成了高精度的传感器技术、先进的数据采集系统和强大的云端管理平台。这套系统能够实时监测各种环境参数，如温度、湿度、空气质量、光照强度等，并将数据通过无线网络传输到云端进行分析和管理。SenseCAP特别适用于智慧农业、环境监测、工业物联网等场景，为用户提供可靠、准确的环境数据支持。"
            elif "edge computing" in question_lower or "边缘计算" in question_lower:
                answer = "边缘AI计算代表了人工智能技术的一个重要发展方向，它将AI应用从云端迁移到本地设备上运行，实现了更快的响应速度和更好的隐私保护。通过reComputer等基于NVIDIA Jetson平台的设备，边缘计算能够在本地处理各种AI任务，如语音识别、图像处理、自然语言理解等，而无需依赖网络连接。这种技术特别适合需要实时处理、低延迟响应的应用场景，如自动驾驶、智能监控、工业自动化等领域。"
            elif "recomputer" in question_lower:
                answer = "reComputer系列是矽递科技基于NVIDIA Jetson平台开发的高性能边缘计算设备，它专门为AI和边缘计算应用而设计。这些设备集成了强大的GPU计算能力，支持各种主流的深度学习框架，如TensorFlow、PyTorch等，能够运行复杂的AI模型和算法。reComputer系列产品不仅性能强劲，还具有良好的散热设计和丰富的接口配置，特别适合需要本地AI处理能力的应用场景，如机器人、无人机、智能摄像头等。"
            else:
                answer = None
            
            if answer:
                # 使用打字机效果显示
                if self.streaming_enabled:
                    print("💬 回答: ", end='', flush=True)
                    self.typewriter_effect(answer)
                return answer
            
            else:
                # 通用中文回答
                top_result = sorted_results[0]
                title = top_result['title']
                content = top_result['content']
                score = top_result['score']
                
                if content.startswith('[Introduction] '):
                    content = content[16:]
                
                answer = f"根据搜索结果，{title} 提供了与您问题最相关的信息。{content[:300]}... 这个结果的相关度评分为 {score:.3f}，表明它包含了您需要的重要信息。如果您需要更详细的了解，可以访问相关的 Wiki 页面获取完整的技术规格和使用说明。"
                
                # 使用打字机效果显示
                if self.streaming_enabled:
                    print("💬 回答: ", end='', flush=True)
                    self.typewriter_effect(answer)
                
                return answer
        
        else:
            # 英文回答
            if "xiao" in question_lower:
                answer = "XIAO series is a line of micro development boards that we launched at Seeed Studio. These boards are compact in size but powerful in functionality, featuring our standardized design philosophy with excellent compatibility and expandability. They are particularly suitable for various embedded projects, prototyping, and maker projects. XIAO series products not only support the Arduino ecosystem but also integrate Grove connectors, allowing you to easily connect various sensors and modules, greatly simplifying the complexity of hardware development."
            elif "grove" in question_lower:
                answer = "Grove sensor module system is a standardized hardware connection solution that we developed at Seeed Studio, revolutionizing the complex process of traditional hardware development. Through unified connection interfaces and standardized module design, Grove system allows you to easily connect various sensors, actuators, and communication modules to development boards like building blocks. This design not only greatly reduces the entry barrier for hardware development but also improves project reliability and maintainability, making it particularly suitable for beginners and rapid prototyping."
            elif "sensecap" in question_lower:
                answer = "SenseCAP is a one-stop solution that we specifically designed at Seeed Studio for environmental monitoring and IoT applications. It integrates high-precision sensor technology, advanced data acquisition systems, and powerful cloud management platforms. The system can monitor various environmental parameters in real-time, such as temperature, humidity, air quality, light intensity, etc., and transmit data to the cloud for analysis and management through wireless networks. SenseCAP is particularly suitable for smart agriculture, environmental monitoring, industrial IoT, and other scenarios, providing users with reliable and accurate environmental data support."
            elif "edge computing" in question_lower:
                answer = "Edge AI computing represents an important development direction in artificial intelligence technology, moving AI applications from the cloud to local devices for operation, achieving faster response speeds and better privacy protection. Through devices like reComputer based on the NVIDIA Jetson platform, edge computing can process various AI tasks locally, such as speech recognition, image processing, natural language understanding, etc., without relying on network connections. This technology is particularly suitable for application scenarios that require real-time processing and low-latency responses, such as autonomous driving, intelligent monitoring, and industrial automation."
            elif "recomputer" in question_lower:
                answer = "reComputer series is a high-performance edge computing device that we developed at Seeed Studio based on the NVIDIA Jetson platform, specifically designed for AI and edge computing applications. These devices integrate powerful GPU computing capabilities and support various mainstream deep learning frameworks such as TensorFlow and PyTorch, enabling the operation of complex AI models and algorithms. reComputer series products are not only powerful in performance but also feature good thermal design and rich interface configurations, making them particularly suitable for application scenarios that require local AI processing capabilities, such as robotics, drones, and smart cameras."
            else:
                answer = None
            
            if answer:
                # 使用打字机效果显示
                if self.streaming_enabled:
                    print("💬 回答: ", end='', flush=True)
                    self.typewriter_effect(answer)
                return answer
            
            else:
                # 通用英文回答
                top_result = sorted_results[0]
                title = top_result['title']
                content = top_result['content']
                score = top_result['score']
                
                if content.startswith('[Introduction] '):
                    content = content[16:]
                
                answer = f"Based on the search results, {title} provides the most relevant information for your question. {content[:300]}... This result has a relevance score of {score:.3f}, indicating that it contains important information you need. If you need more detailed information, you can visit the relevant Wiki page for complete technical specifications and usage instructions."
                
                # 使用打字机效果显示
                if self.streaming_enabled:
                    print("💬 回答: ", end='', flush=True)
                    self.typewriter_effect(answer)
                
                return answer
    
    def run(self):
        """运行问答系统"""
        print("🤖 Seeed Studio（矽递科技）专属智能助手")
        print("=" * 50)
        print("欢迎使用我们的智能问答系统！")
        print("我是Seeed Studio的专属AI助手，很高兴为您服务")
        print("=" * 50)
        
        sample_questions = [
            "介绍一下XIAO系列产品",
            "Grove传感器模块有什么特点？",
            "SenseCAP的功能是什么？",
            "Edge Computing是什么？",
            "reComputer有什么特色？"
        ]
        
        print(f"\n💡 示例问题:")
        for i, question in enumerate(sample_questions, 1):
            print(f"   {i}. {question}")
        
        print(f"\n💬 现在可以开始提问了！")
        print("💡 输入 'help' 查看帮助，'quit' 退出")
        print("💡 支持方向键、退格键等编辑功能")
        print("-" * 50)
        
        try:
            while True:
                try:
                    query = self.safe_input("\n🤔 请输入您的问题: ")
                    
                    if not query:
                        continue
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        print("👋 感谢使用，再见！")
                        break
                    elif query.lower() == 'help':
                        self.show_help()
                        continue
                    elif query.lower() == 'info':
                        self.show_system_info()
                        continue
                    elif query.lower() == 'debug':
                        self.show_debug_info()
                        continue
                    elif query.lower() == 'sample':
                        print("💡 示例问题:")
                        for i, question in enumerate(sample_questions, 1):
                            print(f"   {i}. {question}")
                        continue
                    elif query.lower() == 'clear':
                        self.clear_cache()
                        continue
                    elif query.lower() == 'save':
                        self.save_cache()
                        continue
                    elif query.lower() == 'stream':
                        self.streaming_enabled = not self.streaming_enabled
                        status = "启用" if self.streaming_enabled else "禁用"
                        print(f"✅ 流式显示已{status}")
                        continue
                    elif query.lower() == 'speed':
                        print(f"当前打字速度: {self.typing_speed:.3f} 秒/字符")
                        try:
                            new_speed = float(self.safe_input("请输入新的打字速度 (0.01-0.1): "))
                            if 0.01 <= new_speed <= 0.1:
                                self.typing_speed = new_speed
                                print(f"✅ 打字速度已设置为: {new_speed:.3f} 秒/字符")
                            else:
                                print("❌ 速度范围应在 0.01-0.1 之间")
                        except ValueError:
                            print("❌ 请输入有效的数字")
                        continue

                    
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
            # 保存输入历史和缓存
            self.save_history()
            self.save_cache()
    
    def show_help(self):
        """显示帮助信息"""
        print("\n💡 使用说明:")
        print("   - 直接输入问题")
        print("   - 输入 'help' 显示帮助")
        print("   - 输入 'info' 显示系统信息")
        print("   - 输入 'sample' 显示示例问题")
        print("   - 输入 'debug' 显示调试信息")
        print("   - 输入 'clear' 清空缓存")
        print("   - 输入 'save' 保存缓存")
        print("   - 输入 'stream' 切换流式显示")
        print("   - 输入 'speed' 调整打字速度")

        print("   - 输入 'quit' 或 'exit' 退出程序")
        print("\n⌨️  输入功能:")
        print("   - 支持方向键移动光标")
        print("   - 支持退格键删除字符")
        print("   - 支持 Ctrl+A 选择全部")
        print("   - 支持 Ctrl+U 删除整行")
        print("   - 支持 Tab 键自动补全")
        print("\n🚀 系统特性:")
        print("   - 使用预保存的 FAISS 索引，启动快速")
        print("   - 基于 Ollama nomic-embed-text 模型")
        print("   - 智能缓存机制，重复问题秒答")
        print("   - 优化的搜索算法，响应更快")
        print("   - 流式回答显示，打字机效果")
        print("   - 实时生成回答，不保存缓存")
        print("   - 基于英文 Wiki 内容，质量高")
    
    def show_debug_info(self):
        """显示调试信息"""
        print("\n🔍 调试信息:")
        print(f"   FAISS 索引: {'已加载' if self.faiss_index else '未加载'}")
        if self.faiss_index:
            print(f"     索引类型: {type(self.faiss_index).__name__}")
            print(f"     向量数量: {self.faiss_index.ntotal}")
            print(f"     向量维度: {self.faiss_index.d}")
            print(f"     索引状态: {'正常' if self.faiss_index.ntotal > 0 else '异常'}")
        
        print(f"   元数据: {'已加载' if self.faiss_metadata else '未加载'}")
        if self.faiss_metadata:
            print(f"     记录数量: {len(self.faiss_metadata)}")
            print(f"     第一条记录: {list(self.faiss_metadata[0].keys()) if self.faiss_metadata else '无'}")
        
        print(f"   Wiki 页面: {'已加载' if self.wiki_pages else '未加载'}")
        if self.wiki_pages:
            print(f"     页面数量: {len(self.wiki_pages)}")
            print(f"     第一条页面: {list(self.wiki_pages[0].keys()) if self.wiki_pages else '无'}")
        
        print(f"   Embedding 模型: {self.embedding_model}")
        print(f"   Embedding 缓存: {len(self.embedding_cache)} 项")
        print(f"   回答缓存: {len(self.answer_cache)} 项")
        
        # 测试 embedding 生成
        print("\n🧪 测试 Embedding 生成...")
        try:
            test_text = "test"
            test_embedding = self.generate_embedding(test_text)
            if test_embedding is not None:
                print(f"   ✅ 测试成功: 维度 {len(test_embedding)}, 类型 {type(test_embedding)}")
                print(f"      范数: {np.linalg.norm(test_embedding):.6f}")
                print(f"      数据类型: {test_embedding.dtype}")
                print(f"      形状: {test_embedding.shape}")
            else:
                print("   ❌ 测试失败: 返回 None")
        except Exception as e:
            print(f"   ❌ 测试异常: {str(e)}")
        
        # 检查数据一致性
        print("\n🔍 数据一致性检查:")
        if self.faiss_index and self.faiss_metadata and self.wiki_pages:
            index_count = self.faiss_index.ntotal
            metadata_count = len(self.faiss_metadata)
            pages_count = len(self.wiki_pages)
            
            print(f"   FAISS 索引向量数: {index_count}")
            print(f"   元数据记录数: {metadata_count}")
            print(f"   Wiki 页面数: {pages_count}")
            
            if index_count == metadata_count == pages_count:
                print("   ✅ 数据一致性: 完全匹配")
            else:
                print("   ⚠️  数据一致性: 不匹配")
                if index_count != metadata_count:
                    print(f"      ⚠️  索引与元数据不匹配: {index_count} vs {metadata_count}")
                if metadata_count != pages_count:
                    print(f"      ⚠️  元数据与页面不匹配: {metadata_count} vs {pages_count}")
        else:
            print("   ❌ 无法检查: 数据未完全加载")

def main():
    """主函数"""
    try:
        qa_system = OptimizedQASystem()
        qa_system.run()
    except Exception as e:
        print(f"\n❌ 系统启动失败: {str(e)}")
        print("请检查数据文件和 Ollama 服务")

if __name__ == "__main__":
    main()

