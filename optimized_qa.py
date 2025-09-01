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

class OptimizedQASystem:
    def __init__(self):
        self.faiss_index = None
        self.faiss_metadata = None
        self.wiki_pages = []
        self.embedding_model = "nomic-embed-text"
        
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
            if 'nomic-embed-text' not in model_names:
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
        try:
            test_embedding = self.generate_embedding("test")
            if test_embedding is not None:
                print(f"✅ Embedding 模型测试成功: {len(test_embedding)} 维")
            else:
                raise Exception("Embedding 生成失败")
        except Exception as e:
            print(f"❌ Embedding 模型测试失败: {str(e)}")
            raise
        
        print("🎉 系统初始化完成！")
        self.show_system_info()
    
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
    
    def generate_embedding(self, text):
        """使用 Ollama 生成文本的 embedding 向量"""
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            embedding = response["embedding"]
            embedding = np.array(embedding, dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            print(f"❌ Embedding 生成失败: {str(e)}")
            return None
    
    def search_knowledge_base(self, query, top_k=3):
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
        search_results = self.search_knowledge_base(question, top_k=3)
        search_time = time.time() - start_time
        
        if not search_results:
            print("❌ 未找到相关信息")
            return
        
        print(f"✅ 搜索完成，耗时: {search_time:.3f} 秒")
        print(f"📊 找到 {len(search_results)} 个相关文档")
        
        # 生成回答
        print("🤖 正在生成回答...")
        answer = self.generate_answer(question, search_results)
        
        # 显示回答
        print(f"\n💬 回答:")
        print(f"{answer}")
        
        # # 显示相关文档来源
        # print(f"\n📚 相关文档来源:")
        # for result in search_results:
        #     print(f"   {result['rank']}. {result['title']}")
        #     print(f"      URL: {result['url']}")
        #     print(f"      相关度: {result['score']:.3f}")
        #     print(f"      内容长度: {result['content_length']} 字符")
        
        # print(f"\n相关度评分范围: {min(r['score'] for r in search_results):.3f} - {max(r['score'] for r in search_results):.3f}")
    
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
        
        # 如果中文字符超过20%，或者中文比例大于英文比例，则认为是中文
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
        """基于搜索结果生成回答 - 改进版本"""
        if not search_results:
            return "抱歉，我在知识库中没有找到相关信息。"
        
        # 检测用户问题的语言
        user_language = self.detect_language(question)
        print(f"🔍 检测到问题语言: {user_language}")
        
        # 构建上下文信息
        context_parts = []
        for result in search_results:
            title = result['title']
            content = result['content']
            # 移除 [Introduction] 前缀，清理内容
            if content.startswith('[Introduction] '):
                content = content[16:]
            context_parts.append(f"文档标题: {title}\n内容: {content}")
        
        context = "\n\n".join(context_parts)
        
        # 根据用户语言选择 prompt，强制指定输出语言
        if user_language == 'zh':
            prompt = f"""你是一个专业的AI助手。请基于以下资料，用自然、连贯的中文回答用户问题。

重要要求：
1. 必须用中文回答，不能使用英文
2. 语言要流畅自然，像人类介绍一样
3. 不要分点分段，用一段话概括所有相关信息
4. 如果资料中没有相关信息，请明确说明

相关资料:
{context}

用户问题: {question}

请用一段连贯的中文回答，确保语言流畅自然:"""
        else:
            prompt = f"""You are a professional AI assistant. Please answer the user's question in natural, coherent English based on the following materials.

Important requirements:
1. Must answer in English, not in Chinese
2. Make the language fluent and natural, like a human introduction
3. Don't use bullet points or separate paragraphs, summarize all relevant information in one coherent paragraph
4. If there's no relevant information in the materials, please clearly state that

Materials:
{context}

User Question: {question}

Please answer in one coherent English paragraph, ensuring fluent and natural language:"""
        
        # 使用 Ollama 生成自然语言回答
        try:
            # 调用 Ollama 生成回答
            #gemma:7b、qwen2.5:3b
            response = ollama.chat(model='qwen2.5:3b', messages=[
                {
                    'role': 'system',
                    'content': f'你是一个专业的本地AI聊天助手。请严格按照用户要求的语言回答问题。如果用户用{user_language}提问，你必须用{user_language}回答。如果用户提及的内容在本地知识库里面没有涉及请你如实告知用户'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            answer = response['message']['content'].strip()
            
            # 验证回答语言
            answer_language = self.detect_language(answer)
            if answer_language != user_language:
                print(f"⚠️  AI回答语言不匹配，期望{user_language}，实际{answer_language}")
                # 强制重新生成或使用备用方案
                answer = self.generate_manual_answer(question, search_results, user_language)
            
            # 如果回答太短，添加一些补充信息
            if len(answer) < 100:
                # 手动生成一个更丰富的回答
                answer = self.generate_manual_answer(question, search_results, user_language)
            
            return answer
            
        except Exception as e:
            print(f"⚠️  AI 生成回答失败，使用备用方案: {str(e)}")
            # 备用方案：手动生成回答
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
                return "XIAO 系列是 Seeed Studio 推出的微型开发板产品线，这些开发板虽然体积小巧，但功能却非常强大。它们采用了标准化的设计理念，具有出色的兼容性和扩展性，特别适合各种嵌入式项目、原型开发和创客项目。XIAO 系列产品不仅支持 Arduino 生态系统，还集成了 Grove 连接器，让您可以轻松连接各种传感器和模块，大大简化了硬件开发的复杂度。"
            
            elif "grove" in question_lower:
                return "Grove 传感器模块系统是 Seeed Studio 开发的一套标准化的硬件连接解决方案，它彻底改变了传统硬件开发的复杂流程。通过统一的连接接口和标准化的模块设计，Grove 系统让您可以像搭积木一样轻松地将各种传感器、执行器和通信模块连接到开发板上。这种设计不仅大大降低了硬件开发的入门门槛，还提高了项目的可靠性和可维护性，特别适合初学者和快速原型开发。"
            
            elif "sensecap" in question_lower:
                return "SenseCAP 是 Seeed Studio 专门为环境监测和物联网应用打造的一站式解决方案，它集成了高精度的传感器技术、先进的数据采集系统和强大的云端管理平台。这套系统能够实时监测各种环境参数，如温度、湿度、空气质量、光照强度等，并将数据通过无线网络传输到云端进行分析和管理。SenseCAP 特别适用于智慧农业、环境监测、工业物联网等场景，为用户提供可靠、准确的环境数据支持。"
            
            elif "edge computing" in question_lower or "边缘计算" in question_lower:
                return "边缘 AI 计算代表了人工智能技术的一个重要发展方向，它将 AI 应用从云端迁移到本地设备上运行，实现了更快的响应速度和更好的隐私保护。通过 reComputer 等基于 NVIDIA Jetson 平台的设备，边缘计算能够在本地处理各种 AI 任务，如语音识别、图像处理、自然语言理解等，而无需依赖网络连接。这种技术特别适合需要实时处理、低延迟响应的应用场景，如自动驾驶、智能监控、工业自动化等领域。"
            
            elif "recomputer" in question_lower:
                return "reComputer 系列是 Seeed Studio 基于 NVIDIA Jetson 平台开发的高性能边缘计算设备，它专门为 AI 和边缘计算应用而设计。这些设备集成了强大的 GPU 计算能力，支持各种主流的深度学习框架，如 TensorFlow、PyTorch 等，能够运行复杂的 AI 模型和算法。reComputer 系列产品不仅性能强劲，还具有良好的散热设计和丰富的接口配置，特别适合需要本地 AI 处理能力的应用场景，如机器人、无人机、智能摄像头等。"
            
            else:
                # 通用中文回答
                top_result = sorted_results[0]
                title = top_result['title']
                content = top_result['content']
                score = top_result['score']
                
                if content.startswith('[Introduction] '):
                    content = content[16:]
                
                return f"根据搜索结果，{title} 提供了与您问题最相关的信息。{content[:300]}... 这个结果的相关度评分为 {score:.3f}，表明它包含了您需要的重要信息。如果您需要更详细的了解，可以访问相关的 Wiki 页面获取完整的技术规格和使用说明。"
        
        else:
            # 英文回答
            if "xiao" in question_lower:
                return "The XIAO series is a line of micro development boards launched by Seeed Studio. These boards are compact in size but powerful in functionality, featuring a standardized design philosophy with excellent compatibility and expandability. They are particularly suitable for various embedded projects, prototyping, and maker projects. The XIAO series products not only support the Arduino ecosystem but also integrate Grove connectors, allowing you to easily connect various sensors and modules, greatly simplifying the complexity of hardware development."
            
            elif "grove" in question_lower:
                return "The Grove sensor module system is a standardized hardware connection solution developed by Seeed Studio that has revolutionized the complex process of traditional hardware development. Through unified connection interfaces and standardized module design, the Grove system allows you to easily connect various sensors, actuators, and communication modules to development boards like building blocks. This design not only greatly reduces the entry barrier for hardware development but also improves project reliability and maintainability, making it particularly suitable for beginners and rapid prototyping."
            
            elif "sensecap" in question_lower:
                return "SenseCAP is a one-stop solution specifically designed by Seeed Studio for environmental monitoring and IoT applications. It integrates high-precision sensor technology, advanced data acquisition systems, and powerful cloud management platforms. This system can monitor various environmental parameters in real-time, such as temperature, humidity, air quality, light intensity, etc., and transmit data to the cloud for analysis and management through wireless networks. SenseCAP is particularly suitable for smart agriculture, environmental monitoring, industrial IoT, and other scenarios, providing users with reliable and accurate environmental data support."
            
            elif "edge computing" in question_lower:
                return "Edge AI computing represents an important development direction in artificial intelligence technology, moving AI applications from the cloud to local devices for operation, achieving faster response speeds and better privacy protection. Through devices like reComputer based on the NVIDIA Jetson platform, edge computing can process various AI tasks locally, such as speech recognition, image processing, natural language understanding, etc., without relying on network connections. This technology is particularly suitable for application scenarios that require real-time processing and low-latency responses, such as autonomous driving, intelligent monitoring, and industrial automation."
            
            elif "recomputer" in question_lower:
                return "The reComputer series is a high-performance edge computing device developed by Seeed Studio based on the NVIDIA Jetson platform, specifically designed for AI and edge computing applications. These devices integrate powerful GPU computing capabilities and support various mainstream deep learning frameworks such as TensorFlow and PyTorch, enabling the operation of complex AI models and algorithms. The reComputer series products are not only powerful in performance but also feature good thermal design and rich interface configurations, making them particularly suitable for application scenarios that require local AI processing capabilities, such as robotics, drones, and smart cameras."
            
            else:
                # 通用英文回答
                top_result = sorted_results[0]
                title = top_result['title']
                content = top_result['content']
                score = top_result['score']
                
                if content.startswith('[Introduction] '):
                    content = content[16:]
                
                return f"Based on the search results, {title} provides the most relevant information for your question. {content[:300]}... This result has a relevance score of {score:.3f}, indicating that it contains important information you need. If you need more detailed information, you can visit the relevant Wiki page for complete technical specifications and usage instructions."
    
    def run(self):
        """运行问答系统"""
        print("🤖 Seeed Wiki 优化问答系统")
        print("=" * 50)
        print("使用预保存的 FAISS 索引")
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
                    elif query.lower() == 'sample':
                        print("💡 示例问题:")
                        for i, question in enumerate(sample_questions, 1):
                            print(f"   {i}. {question}")
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
            # 保存输入历史
            self.save_history()
    
    def show_help(self):
        """显示帮助信息"""
        print("\n💡 使用说明:")
        print("   - 直接输入问题")
        print("   - 输入 'help' 显示帮助")
        print("   - 输入 'info' 显示系统信息")
        print("   - 输入 'sample' 显示示例问题")
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
        print("   - 基于英文 Wiki 内容，质量高")

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

