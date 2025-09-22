#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成语音识别的智能对话助手
结合sherpa-ncnn语音识别和Kokoro TTS，实现完整的语音对话功能
支持唤醒词检测、连续对话、智能结束判断和低延迟响应
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
import threading
import queue
import torch
import gc
import subprocess
import wave
import tempfile
import signal
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from kokoro import KPipeline
import sherpa_ncnn


class IntegratedVoiceAssistant:
    def __init__(self):
        # 语音识别组件
        self.voice_recognizer = None
        self.wake_word = "你好"
        self.conversation_buffer = deque(maxlen=20)
        self.silence_count = 0
        self.max_silence = 3
        self.last_speech_time = 0
        self.has_meaningful_content = False
        self.is_listening_for_wake = True
        self.is_in_conversation = False
        
        # 问答系统组件
        self.faiss_index = None
        self.faiss_metadata = None
        self.wiki_pages = []
        self.embedding_model = "nomic-embed-text"
        self.embedding_cache = {}
        self.cache_lock = threading.Lock()
        
        # TTS组件
        self.tts_pipeline_zh = None
        self.tts_pipeline_en = None
        self.tts_device = None
        self.tts_available = False
        self.voice_zf_tensor = None  # 中文音色
        self.voice_af_tensor = None  # 英文音色
        
        # 线程管理
        self.executor = ThreadPoolExecutor(max_workers=6)
        self.running = False
        
        # 任务队列
        self.voice_queue = queue.Queue()
        self.llm_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # 工作线程
        self.voice_thread = None
        self.llm_thread = None
        self.tts_thread = None
        self.audio_thread = None
        
        # 状态标志
        self.voice_processing = False
        self.llm_processing = False
        self.tts_processing = False
        self.audio_processing = False
        self.is_speaking = False  # 机器人是否正在说话
        
        # 音频系统
        self.audio_initialized = False
        self.temp_audio_files = []  # 临时音频文件管理
        
        # 信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # 初始化系统
        self.initialize_system()
    
    def signal_handler(self, sig, frame):
        """信号处理器"""
        print('\n🛑 正在退出...')
        self.running = False
        sys.exit(0)
    
    def initialize_system(self):
        """初始化整个系统"""
        print("🚀 正在初始化集成语音对话助手...")
        
        try:
            # 初始化各个组件
            self.initialize_knowledge_base()
            self.initialize_voice_recognition()
            self.initialize_tts()
            
            # 启动工作线程
            self.start_worker_threads()
            
            print("🎉 集成语音对话助手初始化完成！")
            self.show_system_info()
            
        except Exception as e:
            print(f"❌ 系统初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def initialize_knowledge_base(self):
        """初始化知识库"""
        print("📚 初始化知识库...")
        
        # 检查数据文件
        required_files = [
            "./data_base/faiss_index.bin",
            "./data_base/faiss_metadata.pkl",
            "./data_base/seeed_wiki_embeddings_db.json"
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                print(f"❌ 缺少数据文件: {file}")
                raise FileNotFoundError(f"缺少数据文件: {file}")
        
        # 检查Ollama服务
        try:
            models = ollama.list()
            print(f"✅ Ollama服务正常，可用模型: {len(models.models)} 个")
        except Exception as e:
            print(f"❌ Ollama服务检查失败: {str(e)}")
            raise
        
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
        print("📖 加载Wiki页面数据...")
        with open("./data_base/seeed_wiki_embeddings_db.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.wiki_pages = data['pages']
            self.metadata = data['metadata']
        print(f"✅ 页面数据加载完成: {len(self.wiki_pages)} 个页面")
    
    def initialize_voice_recognition(self):
        """初始化语音识别"""
        print("🎤 初始化语音识别系统...")
        
        model_dir = "/home/seeed/Local_Chat_Bot/audio/sherpa-ncnn/models"
        
        try:
            self.voice_recognizer = sherpa_ncnn.Recognizer(
                tokens=f"{model_dir}/tokens.txt",
                encoder_param=f"{model_dir}/encoder_jit_trace-pnnx.ncnn.param",
                encoder_bin=f"{model_dir}/encoder_jit_trace-pnnx.ncnn.bin",
                decoder_param=f"{model_dir}/decoder_jit_trace-pnnx.ncnn.param",
                decoder_bin=f"{model_dir}/decoder_jit_trace-pnnx.ncnn.bin",
                joiner_param=f"{model_dir}/joiner_jit_trace-pnnx.ncnn.param",
                joiner_bin=f"{model_dir}/joiner_jit_trace-pnnx.ncnn.bin",
                num_threads=4,
            )
            print("✅ 语音识别模型加载完成")
            print(f"📱 模型采样率: {self.voice_recognizer.sample_rate} Hz")
        except Exception as e:
            print(f"❌ 语音识别初始化失败: {str(e)}")
            raise
    
    def initialize_tts(self):
        """初始化TTS系统"""
        print("🎵 初始化TTS系统...")
        
        try:
            # 检查GPU可用性
            self.tts_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🎵 使用设备: {self.tts_device}")
            
            # 不再使用pygame，使用系统音频播放器
            self.audio_initialized = True
            print("✅ 音频系统初始化完成 (使用系统播放器)")
            
            # 加载音色文件
            print("🎶 加载音色文件...")
            voice_zf_path = 'ckpts/kokoro-v1.1/voices/zf_001.pt'
            voice_af_path = 'ckpts/kokoro-v1.1/voices/af_heart.pt'
            
            if os.path.exists(voice_zf_path):
                self.voice_zf_tensor = torch.load(voice_zf_path, weights_only=True)
                print("✅ 加载中文音色: zf_001")
            
            if os.path.exists(voice_af_path):
                self.voice_af_tensor = torch.load(voice_af_path, weights_only=True)
                print("✅ 加载英文音色: af_heart")
            
            # 定义英文回调函数
            def en_callable(text):
                """英文回调函数"""
                try:
                    selected_voice = self.voice_af_tensor if self.voice_af_tensor is not None else self.voice_zf_tensor
                    result = next(self.tts_pipeline_en(text, voice=selected_voice))
                    return result.phonemes
                except Exception as e:
                    return text
            
            # 初始化TTS模型
            print("🔧 加载TTS模型...")
            self.tts_pipeline_en = KPipeline(lang_code='a', device=self.tts_device)
            self.tts_pipeline_zh = KPipeline(lang_code='z', device=self.tts_device, en_callable=en_callable)
            
            # 模型预热
            print("🔥 TTS模型预热...")
            warmup_voice = self.voice_zf_tensor if self.voice_zf_tensor is not None else 'af_heart'
            generator = self.tts_pipeline_zh("你好", voice=warmup_voice)
            for gs, ps, audio in generator:
                break
            
            self.tts_available = True
            print("✅ TTS系统初始化完成")
            
        except Exception as e:
            print(f"❌ TTS初始化失败: {str(e)}")
            self.tts_available = False
            raise
    
    def start_worker_threads(self):
        """启动工作线程"""
        print("🧵 启动工作线程...")
        
        self.running = True
        
        # 语音识别线程
        self.voice_thread = threading.Thread(target=self.voice_worker, daemon=True)
        self.voice_thread.start()
        print("✅ 语音识别线程已启动")
        
        # LLM处理线程
        self.llm_thread = threading.Thread(target=self.llm_worker, daemon=True)
        self.llm_thread.start()
        print("✅ LLM处理线程已启动")
        
        # TTS处理线程
        self.tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.tts_thread.start()
        print("✅ TTS处理线程已启动")
        
        # 音频播放线程
        self.audio_thread = threading.Thread(target=self.audio_worker, daemon=True)
        self.audio_thread.start()
        print("✅ 音频播放线程已启动")
    
    def voice_worker(self):
        """语音识别工作线程"""
        print("🎤 语音识别线程开始运行...")
        
        # 启动连续监听
        self.continuous_voice_listen()
    
    def continuous_voice_listen(self):
        """连续语音监听"""
        print("🔄 开始监听唤醒词...")
        
        while self.running:
            try:
                if self.is_listening_for_wake:
                    # 监听唤醒词模式
                    self.listen_for_wake_word()
                elif self.is_in_conversation:
                    # 对话模式
                    self.listen_for_conversation()
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"❌ 语音监听错误: {str(e)}")
                time.sleep(1)
    
    def listen_for_wake_word(self):
        """监听唤醒词"""
        # 如果机器人正在说话或播放音频，不监听唤醒词
        if self.is_speaking or self.audio_processing:
            time.sleep(0.2)
            return
            
        audio_data, audio_level = self.record_chunk(duration=1)
        
        if audio_data is not None and audio_level > 0.03:  # 提高阈值，减少噪音干扰
            text = self.recognize_audio(audio_data)
            
            if text and self.wake_word in text:
                print(f"🎯 检测到唤醒词: '{text}'")
                self.handle_wake_up()
        
        time.sleep(0.1)
    
    def handle_wake_up(self):
        """处理唤醒"""
        print("👋 机器人被唤醒，准备回复...")
        
        # 切换状态
        self.is_listening_for_wake = False
        self.is_in_conversation = False
        self.is_speaking = True  # 标记正在说话
        
        # 生成欢迎语音
        welcome_text = "你好呀，有什么问题可以问我。"
        self.tts_queue.put({
            'text': welcome_text,
            'callback': self.on_welcome_complete
        })
    
    def on_welcome_complete(self, segments):
        """欢迎语音播放完成回调"""
        print("✅ 欢迎语音播放完成，开始监听问题...")
        
        # 切换到对话模式
        self.is_speaking = False  # 停止说话状态
        self.is_in_conversation = True
        self.conversation_buffer.clear()
        self.silence_count = 0
        self.last_speech_time = 0
        self.has_meaningful_content = False
    
    def listen_for_conversation(self):
        """监听对话内容"""
        continuous_audio = []
        speech_detected = False
        
        while self.is_in_conversation and self.running:
            # 如果正在播放音频，暂停监听
            if self.is_speaking or self.audio_processing:
                time.sleep(0.2)
                continue
                
            audio_data, audio_level = self.record_chunk(duration=1)
            
            if audio_data is not None:
                if audio_level > 0.03:  # 提高阈值，减少噪音和回音干扰
                    continuous_audio.append(audio_data)
                    speech_detected = True
                    self.silence_count = 0
                    self.last_speech_time = time.time()
                    print("🎵", end="", flush=True)
                    
                else:
                    # 静默处理
                    if speech_detected and len(continuous_audio) > 1:
                        # 处理累积的音频
                        combined_audio = np.concatenate(continuous_audio)
                        text = self.recognize_audio(combined_audio)
                        
                        if text:
                            print(f"\n🗣️  '{text}'")
                            self.conversation_buffer.append({
                                "text": text,
                                "timestamp": time.time()
                            })
                            
                            # 检查是否是有意义的内容
                            if self.is_meaningful_content(text):
                                self.has_meaningful_content = True
                                print("✨ 检测到完整问题")
                        
                        continuous_audio = []
                        speech_detected = False
                    
                    self.silence_count += 1
                    print(".", end="", flush=True)
                    
                    # 检查是否结束对话
                    if self.should_end_conversation():
                        self.end_conversation()
                        break
            
            time.sleep(0.1)
    
    def should_end_conversation(self):
        """判断是否应该结束对话"""
        current_time = time.time()
        
        # 如果已经有有意义的内容，1秒静默就结束（提高实时性）
        if self.has_meaningful_content and self.silence_count >= 1:
            print(f"\n🎯 已获得完整内容，{self.silence_count}秒静默后结束对话")
            return True
        
        # 没有内容时，等待2秒（减少等待时间）
        if not self.has_meaningful_content and self.silence_count >= 2:
            print(f"\n🔇 检测到连续{self.silence_count}次静默，准备结束对话...")
            return True
        
        # 检查结束词
        if len(self.conversation_buffer) > 0:
            recent_text = " ".join([item["text"] for item in list(self.conversation_buffer)[-2:]])
            end_phrases = ["谢谢", "再见", "好的", "知道了", "明白了", "完毕", "结束"]
            
            for phrase in end_phrases:
                if phrase in recent_text:
                    print(f"\n🔚 检测到结束词 '{phrase}'")
                    return True
        
        # 超时保护（减少到8秒）
        if self.last_speech_time > 0 and (current_time - self.last_speech_time) > 8:
            print("\n⏰ 超时强制结束对话")
            return True
        
        return False
    
    def end_conversation(self):
        """结束对话"""
        # 先切换状态，防止重复触发
        self.is_in_conversation = False
        
        print("\n" + "-" * 40)
        print("🔚 对话结束")
        
        # 收集完整问题
        if self.conversation_buffer:
            complete_question = " ".join([item["text"] for item in self.conversation_buffer])
            print(f"📝 完整问题: '{complete_question}'")
            
            # 清空对话缓冲区，防止重复处理
            self.conversation_buffer.clear()
            
            # 将问题发送给LLM处理
            self.llm_queue.put({
                'question': complete_question,
                'callback': self.on_answer_generated
            })
        else:
            print("😅 没有检测到完整问题，返回监听状态")
            self.return_to_wake_listening()
    
    def on_answer_generated(self, answer):
        """LLM回答生成完成回调"""
        print(f"\n💬 AI回答: {answer}")
        
        # 标记开始说话
        self.is_speaking = True
        
        # 将回答发送给TTS
        self.tts_queue.put({
            'text': answer,
            'callback': self.on_answer_speech_complete
        })
    
    def on_answer_speech_complete(self, segments):
        """回答语音播放完成回调"""
        print("✅ 回答播放完成，返回监听状态")
        self.return_to_wake_listening()
    
    def return_to_wake_listening(self):
        """返回唤醒词监听状态"""
        self.is_listening_for_wake = True
        self.is_in_conversation = False
        self.is_speaking = False  # 停止说话状态
        print("🔄 重新监听唤醒词中...")
    
    def cleanup_temp_audio_files(self):
        """清理临时音频文件"""
        for temp_file in self.temp_audio_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"⚠️ 清理临时文件失败: {temp_file} - {e}")
        self.temp_audio_files.clear()
    
    def record_chunk(self, duration=1):
        """录制音频片段"""
        temp_file = f"/tmp/voice_chunk_{int(time.time() * 1000)}.wav"
        
        try:
            cmd = [
                "arecord", 
                "-D", "sysdefault",
                "-f", "S16_LE", 
                "-r", "16000",
                "-c", "1",
                "-d", str(duration),
                temp_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 1)
            
            if result.returncode != 0:
                return None, 0
                
            with wave.open(temp_file, 'rb') as f:
                frames = f.readframes(f.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            audio_level = np.abs(audio_data).max()
            return audio_data, audio_level
            
        except Exception as e:
            return None, 0
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def recognize_audio(self, audio_data):
        """识别音频"""
        try:
            if len(audio_data) < 16000:  # 少于1秒
                return ""
            
            audio_max = np.abs(audio_data).max()
            if audio_max < 0.01:
                return ""
            
            # 音频归一化
            if audio_max > 0:
                if audio_max < 0.1:
                    audio_data = audio_data / audio_max * 0.1
                elif audio_max > 0.8:
                    audio_data = audio_data / audio_max * 0.8
            
            # 创建新的识别器实例
            model_dir = "/home/seeed/Local_Chat_Bot/audio/sherpa-ncnn/models"
            recognizer = sherpa_ncnn.Recognizer(
                tokens=f"{model_dir}/tokens.txt",
                encoder_param=f"{model_dir}/encoder_jit_trace-pnnx.ncnn.param",
                encoder_bin=f"{model_dir}/encoder_jit_trace-pnnx.ncnn.bin",
                decoder_param=f"{model_dir}/decoder_jit_trace-pnnx.ncnn.param",
                decoder_bin=f"{model_dir}/decoder_jit_trace-pnnx.ncnn.bin",
                joiner_param=f"{model_dir}/joiner_jit_trace-pnnx.ncnn.param",
                joiner_bin=f"{model_dir}/joiner_jit_trace-pnnx.ncnn.bin",
                num_threads=4,
            )
            
            recognizer.accept_waveform(16000, audio_data)
            silence = np.zeros(int(0.8 * 16000), dtype=np.float32)
            recognizer.accept_waveform(16000, silence)
            recognizer.input_finished()
            
            result = recognizer.text.strip()
            return result
            
        except Exception as e:
            return ""
    
    def is_meaningful_content(self, text):
        """判断是否是有意义的完整内容"""
        question_words = ["什么", "怎么", "为什么", "哪里", "哪个", "几", "多少", "谁", "吗", "呢", "?", "？"]
        for word in question_words:
            if word in text:
                return True
        
        if len(text.strip()) >= 3:
            return True
            
        return False
    
    def llm_worker(self):
        """LLM处理工作线程"""
        last_processed_question = ""  # 记录上次处理的问题，避免重复
        
        while self.running:
            try:
                if self.llm_queue.empty():
                    time.sleep(0.1)
                    continue
                
                task = self.llm_queue.get(timeout=1)
                self.llm_processing = True
                
                question = task['question']
                callback = task['callback']
                
                # 检查是否是重复问题
                if question == last_processed_question:
                    print(f"⚠️ [LLM线程] 跳过重复问题: '{question[:30]}...'")
                    self.llm_processing = False
                    self.llm_queue.task_done()
                    continue
                
                print(f"🤖 [LLM线程] 开始处理问题: '{question[:30]}...'")
                start_time = time.time()
                
                answer = self.generate_answer(question)
                last_processed_question = question  # 记录已处理的问题
                
                process_time = time.time() - start_time
                print(f"✅ [LLM线程] 回答生成完成，耗时: {process_time:.2f}秒")
                
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
    
    def generate_answer(self, question):
        """生成回答"""
        try:
            # 搜索知识库
            search_results = self.search_knowledge_base(question, top_k=3)
            
            if not search_results:
                return "抱歉，我在知识库中没有找到相关信息。"
            
            # 构建上下文
            context_parts = []
            for result in search_results:
                title = result['title']
                content = result['content']
                if content.startswith('[Introduction] '):
                    content = content[16:]
                if len(content) > 200:
                    content = content[:200] + "..."
                context_parts.append(f"文档: {title}\n内容: {content}")
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""请基于以下资料，用简洁的中文回答用户问题。

重要要求：
1. 必须用中文回答
2. 回答控制在100-150字之间
3. 语言简洁明了，便于语音播放
4. 严格基于提供的资料回答
5. 介绍产品时说"我们的xxx产品"

相关资料:
{context}

用户问题: {question}

请简洁回答:"""
            
            response = ollama.chat(
                model='qwen2.5:3b',
                messages=[
                    {'role': 'system', 'content': '用中文简洁回答，基于资料，适合语音播放。'},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 200,
                }
            )
            
            answer = response['message']['content'].strip()
            
            # 限制长度
            if len(answer) > 150:
                sentences = answer.split('。')
                result = ""
                for sentence in sentences:
                    if len(result + sentence + '。') <= 150:
                        result += sentence + '。'
                    else:
                        break
                answer = result
            
            return answer
            
        except Exception as e:
            print(f"❌ 回答生成失败: {str(e)}")
            return "抱歉，我无法生成回答。"
    
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
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ 搜索失败: {str(e)}")
            return []
    
    def generate_embedding(self, text):
        """生成embedding"""
        if not text or not text.strip():
            return None
        
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            embedding = np.array(response["embedding"], dtype=np.float32)
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            print(f"❌ Embedding生成失败: {str(e)}")
            return None
    
    def tts_worker(self):
        """TTS处理工作线程"""
        while self.running:
            try:
                if self.tts_queue.empty():
                    time.sleep(0.1)
                    continue
                
                task = self.tts_queue.get(timeout=1)
                self.tts_processing = True
                
                text = task['text']
                callback = task['callback']
                
                if not self.tts_available or not text.strip():
                    self.tts_processing = False
                    self.tts_queue.task_done()
                    continue
                
                print(f"🎤 [TTS线程] 开始生成语音: '{text[:30]}...'")
                
                # 清理之前的临时音频文件
                self.cleanup_temp_audio_files()
                
                start_time = time.time()
                
                # 使用中文TTS
                selected_voice = self.voice_zf_tensor if self.voice_zf_tensor is not None else 'af_heart'
                
                audio_segments = []
                generator = self.tts_pipeline_zh(text, voice=selected_voice)
                
                # 收集所有音频片段
                for gs, ps, audio in generator:
                    if isinstance(audio, torch.Tensor):
                        audio = audio.detach().cpu().numpy()
                    
                    if audio.dtype != np.float32:
                        audio = audio.astype(np.float32)
                    
                    audio_segments.append(audio)
                
                # 合并所有音频片段为一个完整的音频，避免分段播放导致的回音
                if audio_segments:
                    combined_audio = np.concatenate(audio_segments)
                    
                    # 音频后处理
                    max_val = np.max(np.abs(combined_audio))
                    if max_val > 1.0:
                        combined_audio = combined_audio / max_val
                    
                    # 转换为int16格式
                    audio_int16 = (combined_audio * 32767).astype(np.int16)
                    
                    process_time = time.time() - start_time
                    print(f"✅ [TTS线程] 语音生成完成，耗时: {process_time:.2f}秒，音频长度: {len(audio_int16)/24000:.1f}秒")
                    
                    # 保存为WAV文件
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_filename = temp_file.name
                    temp_file.close()
                    
                    # 写入WAV文件
                    with wave.open(temp_filename, 'wb') as wav_file:
                        wav_file.setnchannels(1)  # 单声道
                        wav_file.setsampwidth(2)  # 16位
                        wav_file.setframerate(24000)  # 采样率
                        wav_file.writeframes(audio_int16.tobytes())
                    
                    # 添加到临时文件列表
                    self.temp_audio_files.append(temp_filename)
                    
                    # 清空音频队列，防止重叠播放
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    # 添加文件路径到播放队列
                    self.audio_queue.put(temp_filename)
                else:
                    print("⚠️ [TTS线程] 没有生成音频内容")
                
                if callback:
                    callback(1)  # 现在只有一个合并的音频片段
                
                self.tts_processing = False
                self.tts_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ [TTS线程] 处理错误: {str(e)}")
                self.tts_processing = False
                time.sleep(1)
    
    def audio_worker(self):
        """音频播放工作线程 - 使用系统播放器"""
        while self.running:
            try:
                if self.audio_queue.empty():
                    time.sleep(0.01)
                    continue
                
                audio_filename = self.audio_queue.get(timeout=0.1)
                self.audio_processing = True
                
                try:
                    if os.path.exists(audio_filename):
                        print(f"🔊 [音频线程] 播放音频文件")
                        
                        # 使用aplay播放音频文件，降低音量避免回音
                        start_time = time.time()
                        result = subprocess.run(['aplay', '-q', audio_filename], 
                                              capture_output=True, text=True, 
                                              timeout=30)  # 30秒超时，-q安静模式
                        
                        play_time = time.time() - start_time
                        
                        if result.returncode == 0:
                            print(f"✅ [音频线程] 音频播放完成，播放时间: {play_time:.1f}秒")
                            
                            # 播放完成后等待2秒，让扬声器声音完全消失，避免被麦克风捕获
                            print("⏱️ [音频线程] 等待回音消散...")
                            time.sleep(2.0)
                            print("✅ [音频线程] 静音缓冲完成")
                        else:
                            print(f"❌ [音频线程] 播放失败: {result.stderr}")
                            
                        # 清理临时音频文件
                        try:
                            os.remove(audio_filename)
                            if audio_filename in self.temp_audio_files:
                                self.temp_audio_files.remove(audio_filename)
                        except:
                            pass
                            
                    else:
                        print(f"❌ [音频线程] 音频文件不存在: {audio_filename}")
                    
                except subprocess.TimeoutExpired:
                    print("⚠️ [音频线程] 播放超时")
                except Exception as e:
                    print(f"❌ [音频线程] 播放错误: {str(e)}")
                
                self.audio_processing = False
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ [音频线程] 处理错误: {str(e)}")
                self.audio_processing = False
                time.sleep(0.1)
    
    def show_system_info(self):
        """显示系统信息"""
        print(f"\n📊 系统信息:")
        print(f"   知识库页面: {len(self.wiki_pages)}")
        print(f"   语音识别: {'可用' if self.voice_recognizer else '不可用'}")
        print(f"   TTS系统: {'可用' if self.tts_available else '不可用'}")
        print(f"   唤醒词: '{self.wake_word}'")
        print(f"   音频播放: 系统播放器 (防回音模式)")
        print(f"   工作线程: 语音={'运行中' if self.voice_thread and self.voice_thread.is_alive() else '未启动'}")
        print(f"            LLM={'运行中' if self.llm_thread and self.llm_thread.is_alive() else '未启动'}")
        print(f"            TTS={'运行中' if self.tts_thread and self.tts_thread.is_alive() else '未启动'}")
        print(f"            音频={'运行中' if self.audio_thread and self.audio_thread.is_alive() else '未启动'}")
        print(f"\n💡 防回音建议:")
        print(f"   - 使用耳机可完全避免回音问题")
        print(f"   - 降低扬声器音量")
        print(f"   - 保持麦克风与扬声器距离")
        print(f"   - 当前已启用音频暂停监听功能")
    
    def run(self):
        """运行语音对话助手"""
        print("🤖 Seeed Studio 集成语音对话助手 (防回音版)")
        print("=" * 60)
        print("欢迎使用我们的智能语音问答系统！")
        print("🎯 功能: 唤醒词检测 → 语音问答 → 智能回复")
        print(f"🔑 唤醒词: '{self.wake_word}'")
        print("💡 说话流程: 说唤醒词 → 等待回复 → 提问 → 听回答")
        print("🎧 最佳体验: 建议使用耳机避免回音干扰")
        print("🔇 防回音: 播放时自动暂停监听 + 2秒静音缓冲")
        print("=" * 60)
        
        try:
            while True:
                # 显示状态
                if self.is_listening_for_wake:
                    status = "🔄 监听唤醒词中..."
                elif self.is_in_conversation:
                    status = "💬 对话进行中..."
                else:
                    status = "🤖 系统处理中..."
                
                print(f"\r{status}", end="", flush=True)
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n👋 感谢使用，再见！")
        finally:
            self.running = False
            print("🔄 正在停止工作线程...")
            
            # 清理临时音频文件
            self.cleanup_temp_audio_files()
            
            # 等待线程结束
            for thread in [self.voice_thread, self.llm_thread, self.tts_thread, self.audio_thread]:
                if thread and thread.is_alive():
                    thread.join(timeout=2)
            
            print("✅ 语音对话助手已停止")


def main():
    """主函数"""
    try:
        assistant = IntegratedVoiceAssistant()
        assistant.run()
    except Exception as e:
        print(f"\n❌ 系统启动失败: {str(e)}")
        print("请检查数据文件和依赖项")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
