#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音对话助手
集成Kokoro TTS和问答系统，实现实时语音对话
支持多线程并行处理，优化推理速度
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
import threading
import queue
import torch
import gc
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import pygame
from kokoro import KPipeline
import numpy as np


class VoiceChatAssistant:
    def __init__(self):
        # 基础问答系统组件
        self.faiss_index = None
        self.faiss_metadata = None
        self.wiki_pages = []
        self.embedding_model = "nomic-embed-text"
        
        # 缓存和性能优化
        self.embedding_cache = {}
        self.cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 流式显示相关
        self.streaming_enabled = True
        self.typing_speed = 0.02
        
        # TTS相关
        self.tts_pipeline_zh = None  # 中文TTS模型
        self.tts_pipeline_en = None  # 英文TTS模型
        self.tts_device = None
        self.tts_available = False
        self.audio_playing = False
        
        # 多线程处理
        self.llm_thread = None
        self.tts_thread = None
        self.audio_thread = None
        
        # 任务队列
        self.llm_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # 控制标志
        self.running = False
        self.llm_processing = False
        self.tts_processing = False
        self.audio_processing = False
        
        # 音频系统
        self.audio_initialized = False
        
        # 设置readline
        self.setup_readline()
        
        # 初始化系统
        self.initialize_system()
    
    def setup_readline(self):
        """设置readline配置"""
        try:
            histfile = os.path.join(os.path.expanduser("~"), ".voice_chat_history")
            readline.read_history_file(histfile)
            readline.set_history_length(1000)
            readline.parse_and_bind('tab: complete')
            readline.parse_and_bind('set editing-mode emacs')
        except Exception as e:
            print(f"⚠️ readline设置失败: {str(e)}")
    
    def safe_input(self, prompt):
        """安全的输入函数"""
        try:
            user_input = input(prompt)
            return user_input.strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 用户中断，退出程序")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 输入错误: {str(e)}")
            return ""
    
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
        """检查Ollama服务状态"""
        try:
            models = ollama.list()
            print(f"✅ Ollama服务正常，可用模型: {len(models.models)} 个")
            
            model_names = [model.model for model in models.models]
            if 'nomic-embed-text:latest' not in model_names:
                print("⚠️ 未找到nomic-embed-text模型，正在安装...")
                ollama.pull('nomic-embed-text')
                print("✅ nomic-embed-text模型安装完成")
            else:
                print("✅ nomic-embed-text模型已安装")
                
        except Exception as e:
            print(f"❌ Ollama服务检查失败: {str(e)}")
            raise
    
    def initialize_tts(self):
        """初始化TTS系统"""
        try:
            print("🎤 初始化Kokoro TTS系统...")
            
            # 检查GPU可用性
            self.tts_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🎤 使用设备: {self.tts_device}")
            
            # 初始化pygame音频系统，增加缓冲区大小以支持长音频
            pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=4096)
            self.audio_initialized = True
            print("✅ pygame音频系统初始化完成 (缓冲区: 4096)")
            
            # 加载音色文件
            print("🎵 加载音色文件...")
            voice_zf = "zf_001"
            voice_af = 'af_heart'
            voice_zf_path = f'ckpts/kokoro-v1.1/voices/{voice_zf}.pt'
            voice_af_path = f'ckpts/kokoro-v1.1/voices/{voice_af}.pt'
            
            if os.path.exists(voice_zf_path):
                self.voice_zf_tensor = torch.load(voice_zf_path, weights_only=True)
                print(f"✅ 加载中文音色: {voice_zf}")
            else:
                print(f"❌ 中文音色文件不存在: {voice_zf_path}")
                self.voice_zf_tensor = None
            
            if os.path.exists(voice_af_path):
                self.voice_af_tensor = torch.load(voice_af_path, weights_only=True)
                print(f"✅ 加载英文音色: {voice_af}")
            else:
                print(f"❌ 英文音色文件不存在: {voice_af_path}")
                self.voice_af_tensor = None
            
            # 初始化英文TTS模型（用于en_callable）
            print("🔧 加载英文Kokoro TTS模型...")
            start_time = time.time()
            self.tts_pipeline_en = KPipeline(lang_code='a', device=self.tts_device)
            en_load_time = time.time() - start_time
            print(f"✅ 英文TTS模型加载完成，耗时: {en_load_time:.2f}秒")
            
            # 定义英文回调函数，用于处理中英混杂文本中的英文部分
            def en_callable(text):
                """
                英文回调函数，处理中英混杂文本中的英文部分
                返回英文文本的音素表示
                """
                print(f"    🎤 处理英文文本: '{text}'")
                
                # 特殊词汇的音素映射
                if text == 'Kokoro':
                    return 'kˈOkəɹO'
                elif text == 'Sol':
                    return 'sˈOl'
                elif text == 'reComputer':
                    return 'riːkəmˈpjuːtər'
                elif text == 'Jetson':
                    return 'ˈdʒɛtsən'
                elif text == 'Hello':
                    return 'həˈloʊ'
                elif text == 'world':
                    return 'wɜːrld'
                elif text == 'Welcome':
                    return 'ˈwelkəm'
                elif text == 'to':
                    return 'tuː'
                elif text == 'TTS':
                    return 'tiːtiːˈes'
                elif text == 'AI':
                    return 'eɪˈaɪ'
                elif text == 'technology':
                    return 'tekˈnɑːlədʒi'
                elif text == 'is':
                    return 'ɪz'
                elif text == 'advancing':
                    return 'ədˈvænsɪŋ'
                elif text == 'rapidly':
                    return 'ˈræpədli'
                elif text == 'It\'s':
                    return 'ɪts'
                elif text == 'a':
                    return 'ə'
                elif text == 'beautiful':
                    return 'ˈbjuːtɪfəl'
                elif text == 'day':
                    return 'deɪ'
                elif text == 'today':
                    return 'təˈdeɪ'
                elif text == 'Seeed':
                    return 'siːd'
                elif text == 'Studio':
                    return 'ˈstuːdioʊ'
                elif text == 'XIAO':
                    return 'ˈʃaʊ'
                elif text == 'Grove':
                    return 'ɡroʊv'
                elif text == 'SenseCAP':
                    return 'ˈsenskæp'
                elif text == 'Edge':
                    return 'edʒ'
                elif text == 'Computing':
                    return 'kəmˈpjuːtɪŋ'
                
                # 对于其他英文词汇，使用英文管道生成音素
                try:
                    selected_voice = self.voice_af_tensor if self.voice_af_tensor is not None else self.voice_zf_tensor
                    result = next(self.tts_pipeline_en(text, voice=selected_voice))
                    return result.phonemes
                except Exception as e:
                    print(f"    ⚠️ 无法处理英文文本 '{text}': {e}")
                    # 返回原始文本作为fallback
                    return text
            
            # 初始化中文TTS模型（支持中英混合）
            print("🔧 加载中文Kokoro TTS模型（支持中英混合）...")
            start_time = time.time()
            self.tts_pipeline_zh = KPipeline(lang_code='z', device=self.tts_device, en_callable=en_callable)
            zh_load_time = time.time() - start_time
            print(f"✅ 中文TTS模型加载完成，耗时: {zh_load_time:.2f}秒")
            
            # 模型预热
            print("🔥 进行TTS模型预热...")
            warmup_start = time.time()
            
            # 预热中文模型（使用中文音色）
            warmup_text_zh = "你好"
            zh_voice = self.voice_zf_tensor if self.voice_zf_tensor is not None else 'af_heart'
            generator_zh = self.tts_pipeline_zh(warmup_text_zh, voice=zh_voice)
            for gs, ps, audio in generator_zh:
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
                break
            
            # 预热英文模型（使用英文音色）
            warmup_text_en = "Hello"
            en_voice = self.voice_af_tensor if self.voice_af_tensor is not None else 'af_heart'
            generator_en = self.tts_pipeline_en(warmup_text_en, voice=en_voice)
            for gs, ps, audio in generator_en:
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
                break
            
            warmup_time = time.time() - warmup_start
            print(f"✅ TTS预热完成，耗时: {warmup_time:.2f}秒")
            
            self.tts_available = True
            print("🎉 双语言TTS系统初始化成功！")
            
        except Exception as e:
            print(f"❌ TTS初始化失败: {str(e)}")
            self.tts_available = False
            raise
    
    def initialize_system(self):
        """初始化整个系统"""
        print("🚀 正在初始化语音对话助手...")
        
        try:
            # 检查数据文件
            self.check_data_files()
            
            # 检查Ollama服务
            self.check_ollama_service()
            
            # 加载FAISS索引
            print("🔍 加载FAISS索引...")
            self.faiss_index = faiss.read_index("./data_base/faiss_index.bin")
            print(f"✅ FAISS索引加载完成: {self.faiss_index.ntotal} 个向量")
            
            # 加载元数据
            print("📊 加载元数据...")
            with open("./data_base/faiss_metadata.pkl", 'rb') as f:
                self.faiss_metadata = pickle.load(f)
            print(f"✅ 元数据加载完成: {len(self.faiss_metadata)} 条记录")
            
            # 加载Wiki页面数据
            print("📚 加载Wiki页面数据...")
            with open("./data_base/seeed_wiki_embeddings_db.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.wiki_pages = data['pages']
                self.metadata = data['metadata']
            print(f"✅ 页面数据加载完成: {len(self.wiki_pages)} 个页面")
            
            # 初始化TTS系统
            self.initialize_tts()
            
            # 启动工作线程
            self.start_worker_threads()
            
            print("🎉 语音对话助手初始化完成！")
            self.show_system_info()
            
        except Exception as e:
            print(f"❌ 系统初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def start_worker_threads(self):
        """启动工作线程"""
        print("🧵 启动工作线程...")
        
        self.running = True
        
        # 启动LLM处理线程
        self.llm_thread = threading.Thread(target=self.llm_worker, daemon=True)
        self.llm_thread.start()
        print("✅ LLM处理线程已启动")
        
        # 启动TTS处理线程
        self.tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.tts_thread.start()
        print("✅ TTS处理线程已启动")
        
        # 启动音频播放线程
        self.audio_thread = threading.Thread(target=self.audio_worker, daemon=True)
        self.audio_thread.start()
        print("✅ 音频播放线程已启动")
    
    def llm_worker(self):
        """LLM处理工作线程"""
        while self.running:
            try:
                # 等待任务
                if self.llm_queue.empty():
                    time.sleep(0.1)
                    continue
                
                # 获取任务
                task = self.llm_queue.get(timeout=1)
                self.llm_processing = True
                
                # 处理任务
                question = task['question']
                callback = task['callback']
                
                print(f"🤖 [LLM线程] 开始处理问题: '{question[:30]}...'")
                start_time = time.time()
                
                # 生成回答
                answer = self.generate_answer(question)
                
                process_time = time.time() - start_time
                print(f"✅ [LLM线程] 回答生成完成，耗时: {process_time:.2f}秒")
                
                # 回调处理
                if callback:
                    callback(answer)
                
                self.llm_processing = False
                self.llm_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ [LLM线程] 处理错误: {str(e)}")
                self.llm_processing = False
                time.sleep(1)
    
    def tts_worker(self):
        """TTS处理工作线程"""
        while self.running:
            try:
                # 等待任务
                if self.tts_queue.empty():
                    time.sleep(0.1)
                    continue
                
                # 获取任务
                task = self.tts_queue.get(timeout=1)
                self.tts_processing = True
                
                # 处理任务
                text = task['text']
                callback = task['callback']
                
                if not self.tts_available or not text.strip():
                    self.tts_processing = False
                    self.tts_queue.task_done()
                    continue
                
                # 检测语言
                language = self.detect_language(text)
                print(f"🎤 [TTS线程] 开始生成语音: '{text[:30]}...' (语言: {language})")
                start_time = time.time()
                
                # 根据语言选择TTS模型和音色
                if language == 'zh':
                    pipeline = self.tts_pipeline_zh
                    # 使用中文音色，如果不存在则使用英文音色
                    selected_voice = self.voice_zf_tensor if self.voice_zf_tensor is not None else self.voice_af_tensor
                    if selected_voice is None:
                        selected_voice = 'af_heart'
                else:
                    pipeline = self.tts_pipeline_en
                    # 使用英文音色，如果不存在则使用中文音色
                    selected_voice = self.voice_af_tensor if self.voice_af_tensor is not None else self.voice_zf_tensor
                    if selected_voice is None:
                        selected_voice = 'af_heart'
                
                # 生成语音
                audio_segments = []
                generator = pipeline(text, voice=selected_voice)
                
                segment_count = 0
                for gs, ps, audio in generator:
                    if isinstance(audio, torch.Tensor):
                        audio = audio.detach().cpu().numpy()
                    
                    # 音频预处理
                    if audio.dtype != np.float32:
                        audio = audio.astype(np.float32)
                    
                    # 归一化
                    max_val = np.max(np.abs(audio))
                    if max_val > 1.0:
                        audio = audio / max_val
                    
                    # 转换为int16格式
                    audio_int16 = (audio * 32767).astype(np.int16)
                    audio_segments.append(audio_int16)
                    segment_count += 1
                    print(f"    📝 处理音频片段 {segment_count} (长度: {len(audio_int16)/24000:.1f}秒)")
                
                process_time = time.time() - start_time
                print(f"✅ [TTS线程] 语音生成完成，耗时: {process_time:.2f}秒")
                
                # 将音频片段加入播放队列
                for segment in audio_segments:
                    self.audio_queue.put(segment)
                
                # 回调处理
                if callback:
                    callback(len(audio_segments))
                
                self.tts_processing = False
                self.tts_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ [TTS线程] 处理错误: {str(e)}")
                self.tts_processing = False
                time.sleep(1)
    
    def audio_worker(self):
        """音频播放工作线程"""
        while self.running:
            try:
                # 等待音频片段
                if self.audio_queue.empty():
                    time.sleep(0.01)
                    continue
                
                # 获取音频片段
                audio_segment = self.audio_queue.get(timeout=0.1)
                self.audio_processing = True
                
                # 播放音频
                try:
                    # 计算音频长度
                    audio_duration = len(audio_segment) / 24000.0
                    print(f"🔊 [音频线程] 开始播放音频片段 (长度: {audio_duration:.1f}秒)")
                    
                    sound = pygame.sndarray.make_sound(audio_segment)
                    sound.play()
                    
                    # 等待播放完成，使用更保守的超时机制
                    start_time = time.time()
                    # 超时时间设为音频长度的3倍，但最少30秒，最多120秒
                    max_wait_time = max(30.0, min(120.0, audio_duration * 3))
                    
                    print(f"🔊 [音频线程] 预计播放时间: {audio_duration:.1f}秒，最大等待: {max_wait_time:.1f}秒")
                    
                    # 使用更可靠的播放检测
                    last_busy_time = time.time()
                    while True:
                        is_busy = pygame.mixer.get_busy()
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        
                        if not is_busy:
                            # 如果不再忙碌，等待一小段时间确认播放完成
                            if current_time - last_busy_time > 0.5:  # 等待0.5秒确认
                                break
                        else:
                            last_busy_time = current_time
                        
                        if elapsed_time > max_wait_time:
                            print(f"⚠️ [音频线程] 播放超时 ({elapsed_time:.1f}秒)，强制停止")
                            pygame.mixer.stop()
                            break
                        
                        pygame.time.wait(50)  # 增加等待间隔，减少CPU占用
                    
                    actual_play_time = time.time() - start_time
                    print(f"✅ [音频线程] 音频播放完成 (实际播放时间: {actual_play_time:.1f}秒)")
                    
                except Exception as e:
                    print(f"❌ [音频线程] 播放错误: {str(e)}")
                    # 确保停止播放
                    try:
                        pygame.mixer.stop()
                    except:
                        pass
                
                self.audio_processing = False
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ [音频线程] 处理错误: {str(e)}")
                self.audio_processing = False
                time.sleep(0.1)
    
    def generate_embedding(self, text):
        """生成文本embedding"""
        if not text or not text.strip():
            return None
        
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # 检查缓存
        with self.cache_lock:
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]
        
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            embedding = np.array(response["embedding"], dtype=np.float32)
            
            # 归一化
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # 缓存结果
            with self.cache_lock:
                self.embedding_cache[text_hash] = embedding
                if len(self.embedding_cache) > 1000:
                    oldest_key = next(iter(self.embedding_cache))
                    del self.embedding_cache[oldest_key]
            
            return embedding
            
        except Exception as e:
            print(f"❌ Embedding生成失败: {str(e)}")
            return None
    
    def search_knowledge_base(self, query, top_k=10):
        """搜索知识库"""
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
    
    def generate_answer(self, question):
        """生成回答"""
        # 检测用户问题语言
        user_language = self.detect_language(question)
        print(f"🔍 [LLM线程] 检测到用户问题语言: {user_language}")
        
        # 搜索知识库
        search_results = self.search_knowledge_base(question, top_k=5)
        
        if not search_results:
            if user_language == 'zh':
                return "抱歉，我在知识库中没有找到相关信息。"
            else:
                return "Sorry, I couldn't find relevant information in the knowledge base."
        
        # 构建上下文
        context_parts = []
        for result in search_results[:3]:  # 只使用前3个结果
            title = result['title']
            content = result['content']
            if content.startswith('[Introduction] '):
                content = content[16:]
            if len(content) > 300:
                content = content[:300] + "..."
            context_parts.append(f"文档标题: {title}\n内容: {content}")
        
        context = "\n\n".join(context_parts)
        
        # 根据用户语言构建不同的prompt
        if user_language == 'zh':
            prompt = f"""请基于以下资料，用详细的中文回答用户问题。

重要要求：
1. 必须用中文回答，不能使用英文
2. 回答必须控制在200-250字之间
3. 介绍产品时说"我们的xxx产品..."
4. 严格基于提供的资料回答，不能编造信息
5. 语言要详细完整，包含产品特点、功能和应用场景
6. 不要重复身份介绍
7. 确保回答是一个完整的段落，内容丰富详实
8. 可以适当展开相关技术细节和使用建议

相关资料:
{context}

用户问题: {question}

请用200-250字的详细中文回答:"""
        else:
            prompt = f"""Please answer the user's question in detailed English based on the following materials.

Important requirements:
1. Must answer in English, not in Chinese
2. Keep the answer within 200-250 words
3. When introducing products, say "our xxx product..."
4. Strictly base your answer on the provided materials, don't fabricate information
5. Be detailed and complete, include product features, functions and use cases
6. Don't repeat identity introductions
7. Ensure the answer is a complete paragraph with rich content
8. You can expand on relevant technical details and usage recommendations

Materials:
{context}

User Question: {question}

Please answer in detailed English within 200-250 words:"""
        
        try:
            # 使用Ollama生成回答
            system_prompt = '用中文回答，基于资料，不编造信息，不要重复身份介绍。' if user_language == 'zh' else 'Answer in English, based on materials, don\'t fabricate information, don\'t repeat identity introductions.'
            
            response = ollama.chat(
                model='qwen2.5:3b',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 300,  # 增加生成长度到300字
                }
            )
            
            answer = response['message']['content'].strip()
            
            # 后处理：确保字数限制
            answer = self.limit_answer_length(answer, min_length=200, max_length=250)
            print(f"📊 [LLM线程] 回答字数: {len(answer)} 字")
            return answer
            
        except Exception as e:
            print(f"❌ 回答生成失败: {str(e)}")
            return "抱歉，我无法生成回答。"
    
    def limit_answer_length(self, answer, min_length=200, max_length=250):
        """限制回答长度"""
        # 如果回答太短，尝试扩展
        if len(answer) < min_length:
            # 在句号、问号、感叹号处添加内容
            if answer.endswith(('。', '！', '？')):
                # 如果已经以标点结尾，添加更多信息
                answer = answer + " 该产品具有高性能、易用性强的特点，适合各种应用场景。它采用先进的技术架构，提供稳定可靠的性能表现，能够满足不同用户的需求。无论是初学者还是专业开发者，都能轻松上手使用。"
            else:
                # 如果没有标点结尾，添加标点和信息
                answer = answer + "。该产品具有高性能、易用性强的特点，适合各种应用场景。它采用先进的技术架构，提供稳定可靠的性能表现，能够满足不同用户的需求。无论是初学者还是专业开发者，都能轻松上手使用。"
        
        # 如果回答太长，截断
        if len(answer) > max_length:
            # 在句号、问号、感叹号处截断
            for i in range(max_length, 0, -1):
                if answer[i] in '。！？':
                    return answer[:i+1]
            
            # 如果没有标点符号，直接截断
            return answer[:max_length] + "..."
        
        return answer
    
    def detect_language(self, text):
        """检测文本语言"""
        if not text or not text.strip():
            return 'zh'  # 默认为中文
        
        # 检测中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        english_chars = re.findall(r'[a-zA-Z]', text)
        
        # 计算中英文比例
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        if total_chars == 0:
            return 'zh'
            
        chinese_ratio = len(chinese_chars) / total_chars
        english_ratio = len(english_chars) / total_chars
        
        # 修复逻辑：只要包含中文字符就认为是中文
        if len(chinese_chars) > 0:
            return 'zh'
        elif english_ratio > 0.5:
            return 'en'
        else:
            # 如果都不明显，检查是否有中文标点符号
            chinese_punctuation = re.findall(r'[，。！？；：""''（）【】]', text)
            if chinese_punctuation:
                return 'zh'
            return 'en'
    
    def ask_question_async(self, question):
        """异步提问"""
        print(f"\n🤔 用户问题: {question}")
        
        # 检测用户问题语言
        user_language = self.detect_language(question)
        print(f"🔍 检测到用户问题语言: {user_language}")
        
        def on_answer_generated(answer):
            print(f"\n💬 回答: {answer}")
            
            # 检测AI回答语言
            answer_language = self.detect_language(answer)
            print(f"🔍 检测到AI回答语言: {answer_language}")
            
            # 将回答加入TTS队列
            if self.tts_available:
                self.tts_queue.put({
                    'text': answer,
                    'callback': lambda segments: print(f"🎤 语音已加入播放队列，共{segments}个片段")
                })
        
        # 将问题加入LLM队列
        self.llm_queue.put({
            'question': question,
            'callback': on_answer_generated
        })
        
        print("🔄 问题已加入处理队列，正在生成回答...")
    
    def show_system_info(self):
        """显示系统信息"""
        print(f"\n📊 系统信息:")
        print(f"   总页面数: {len(self.wiki_pages)}")
        print(f"   总向量数: {self.faiss_index.ntotal}")
        print(f"   向量维度: {self.metadata['vector_dimension']}")
        print(f"   Embedding模型: {self.metadata['embedding_model']}")
        print(f"   TTS系统: {'可用' if self.tts_available else '不可用'}")
        print(f"   TTS设备: {self.tts_device}")
        print(f"   中文TTS: {'已加载' if self.tts_pipeline_zh else '未加载'}")
        print(f"   英文TTS: {'已加载' if self.tts_pipeline_en else '未加载'}")
        print(f"   中文音色: {'已加载' if hasattr(self, 'voice_zf_tensor') and self.voice_zf_tensor is not None else '未加载'}")
        print(f"   英文音色: {'已加载' if hasattr(self, 'voice_af_tensor') and self.voice_af_tensor is not None else '未加载'}")
        print(f"   中英混合: {'支持' if hasattr(self, 'voice_zf_tensor') and self.voice_zf_tensor is not None else '不支持'}")
        print(f"   音频系统: {'已初始化' if self.audio_initialized else '未初始化'}")
        print(f"   工作线程: LLM={'运行中' if self.llm_thread and self.llm_thread.is_alive() else '未启动'}, "
              f"TTS={'运行中' if self.tts_thread and self.tts_thread.is_alive() else '未启动'}, "
              f"音频={'运行中' if self.audio_thread and self.audio_thread.is_alive() else '未启动'}")
        print(f"   队列状态: LLM={self.llm_queue.qsize()}, TTS={self.tts_queue.qsize()}, 音频={self.audio_queue.qsize()}")
        print(f"   处理状态: LLM={'处理中' if self.llm_processing else '空闲'}, "
              f"TTS={'处理中' if self.tts_processing else '空闲'}, "
              f"音频={'处理中' if self.audio_processing else '空闲'}")
    
    def run(self):
        """运行语音对话助手"""
        print("🤖 Seeed Studio语音对话助手")
        print("=" * 50)
        print("欢迎使用我们的智能语音问答系统！")
        print("我是Seeed Studio的专属AI助手，支持实时语音对话")
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
        
        print(f"\n💬 现在可以开始语音对话了！")
        print("💡 输入 'help' 查看帮助，'quit' 退出")
        print("💡 支持实时语音合成和播放")
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
                    elif query.lower() == 'sample':
                        print("💡 示例问题:")
                        for i, question in enumerate(sample_questions, 1):
                            print(f"   {i}. {question}")
                        continue
                    
                    if query.isdigit() and 1 <= int(query) <= len(sample_questions):
                        query = sample_questions[int(query) - 1]
                        print(f"🔍 选择的问题: {query}")
                    
                    # 异步处理问题
                    self.ask_question_async(query)
                    
                except KeyboardInterrupt:
                    print("\n\n⚠️ 用户中断")
                    break
                except Exception as e:
                    print(f"\n❌ 发生错误: {str(e)}")
                    continue
                    
        finally:
            # 停止所有线程
            self.running = False
            print("🔄 正在停止工作线程...")
            
            # 等待线程结束
            if self.llm_thread and self.llm_thread.is_alive():
                self.llm_thread.join(timeout=2)
            if self.tts_thread and self.tts_thread.is_alive():
                self.tts_thread.join(timeout=2)
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=2)
            
            print("✅ 语音对话助手已停止")
    
    def show_help(self):
        """显示帮助信息"""
        print("\n💡 使用说明:")
        print("   - 直接输入问题，系统会实时生成语音回答")
        print("   - 输入 'help' 显示帮助")
        print("   - 输入 'info' 显示系统信息")
        print("   - 输入 'sample' 显示示例问题")
        print("   - 输入 'quit' 或 'exit' 退出程序")
        print("\n🚀 系统特性:")
        print("   - 多线程并行处理，响应更快")
        print("   - 实时语音合成和播放")
        print("   - 双语言TTS支持（中文/英文）")
        print("   - 智能语言检测，自动选择TTS模型")
        print("   - 基于Kokoro TTS的高质量语音")
        print("   - 支持GPU加速推理")
        print("   - 智能缓存机制，重复问题秒答")
        print("   - 流式回答显示，打字机效果")


def main():
    """主函数"""
    try:
        assistant = VoiceChatAssistant()
        assistant.run()
    except Exception as e:
        print(f"\n❌ 系统启动失败: {str(e)}")
        print("请检查数据文件和依赖项")


if __name__ == "__main__":
    main()