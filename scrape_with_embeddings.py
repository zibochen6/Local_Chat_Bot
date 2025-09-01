#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seeed Wiki 优化爬虫 - 爬取时生成并保存 Embedding 向量
支持中英文内容爬取，增量更新，定时检测新页面
使用 Ollama nomic-embed-text 模型 + FAISS 索引，加速启动和检索速度
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
from urllib.parse import urljoin, urlparse
from collections import deque
import re
import numpy as np
import faiss
import pickle
import ollama
import hashlib
from datetime import datetime, timedelta
import threading
import schedule

class OptimizedWikiScraper:
    def __init__(self, base_url="https://wiki.seeedstudio.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.visited_urls = set()
        self.all_content = []
        self.url_queue = deque()
        self.max_depth = 4  # 减少深度，专注于主要页面
        
        # 数据文件路径
        self.data_dir = "./data_base"
        self.db_file = f"{self.data_dir}/seeed_wiki_embeddings_db.json"
        self.faiss_index_file = f"{self.data_dir}/faiss_index.bin"
        self.faiss_metadata_file = f"{self.data_dir}/faiss_metadata.pkl"
        self.url_hash_file = f"{self.data_dir}/url_hashes.json"
        self.last_update_file = f"{self.data_dir}/last_update.json"
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 加载已存在的数据
        self.load_existing_data()
        
        # 初始化 Ollama Embedding 模型
        print("🚀 正在初始化 Ollama Embedding 模型...")
        try:
            # 检查 Ollama 服务
            self.check_ollama_service()
            
            # 使用 nomic-embed-text 模型
            self.embedding_model = "nomic-embed-text"
            
            # 测试模型
            test_embedding = self.generate_embedding("test")
            if test_embedding is not None:
                self.dimension = len(test_embedding)
                print(f"✅ Ollama Embedding 模型初始化完成: {self.dimension} 维")
                print(f"   模型名称: {self.embedding_model}")
            else:
                raise Exception("模型测试失败")
            
        except Exception as e:
            print(f"❌ Ollama Embedding 模型初始化失败: {str(e)}")
            print("请确保 Ollama 服务正在运行，并已安装 nomic-embed-text 模型")
            print("安装命令: ollama pull nomic-embed-text")
            raise
        
        # 初始化 FAISS 索引
        if os.path.exists(self.faiss_index_file) and self.faiss_vectors:
            # 加载现有索引
            self.faiss_index = faiss.read_index(self.faiss_index_file)
            print(f"✅ 已加载现有 FAISS 索引: {self.faiss_index.ntotal} 个向量")
        else:
            # 创建新索引
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
            print("✅ 创建新的 FAISS 索引")
        
        print("✅ FAISS 索引初始化完成")
    
    def load_existing_data(self):
        """加载已存在的数据"""
        # 加载数据库
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.all_content = data.get('pages', [])
                    print(f"📂 已加载 {len(self.all_content)} 个现有页面")
            except Exception as e:
                print(f"⚠️ 加载数据库失败: {str(e)}")
                self.all_content = []
        
        # 加载向量元数据
        if os.path.exists(self.faiss_metadata_file):
            try:
                with open(self.faiss_metadata_file, 'rb') as f:
                    self.faiss_metadata = pickle.load(f)
                    self.faiss_vectors = [None] * len(self.faiss_metadata)  # 占位符
                    print(f"📊 已加载 {len(self.faiss_metadata)} 个向量元数据")
            except Exception as e:
                print(f"⚠️ 加载向量元数据失败: {str(e)}")
                self.faiss_metadata = []
                self.faiss_vectors = []
        
        # 加载URL哈希值
        self.url_hashes = {}
        if os.path.exists(self.url_hash_file):
            try:
                with open(self.url_hash_file, 'r', encoding='utf-8') as f:
                    self.url_hashes = json.load(f)
                    print(f"🔗 已加载 {len(self.url_hashes)} 个URL哈希值")
            except Exception as e:
                print(f"⚠️ 加载URL哈希值失败: {str(e)}")
                self.url_hashes = {}
    
    def save_url_hashes(self):
        """保存URL哈希值"""
        with open(self.url_hash_file, 'w', encoding='utf-8') as f:
            json.dump(self.url_hashes, f, ensure_ascii=False, indent=2)
    
    def update_last_update_time(self):
        """更新最后更新时间"""
        update_info = {
            'last_update': datetime.now().isoformat(),
            'total_pages': len(self.all_content),
            'total_vectors': len(self.faiss_vectors)
        }
        with open(self.last_update_file, 'w', encoding='utf-8') as f:
            json.dump(update_info, f, ensure_ascii=False, indent=2)
    
    def should_update(self):
        """检查是否需要更新（24小时检查一次）"""
        if not os.path.exists(self.last_update_file):
            return True
        
        try:
            with open(self.last_update_file, 'r', encoding='utf-8') as f:
                last_update_info = json.load(f)
                last_update_str = last_update_info.get('last_update')
                if last_update_str:
                    last_update = datetime.fromisoformat(last_update_str)
                    time_diff = datetime.now() - last_update
                    return time_diff.total_seconds() >= 24 * 3600  # 24小时
        except Exception as e:
            print(f"⚠️ 检查更新时间失败: {str(e)}")
        
        return True
    
    def check_ollama_service(self):
        """检查 Ollama 服务状态"""
        try:
            # 尝试获取可用模型列表
            models = ollama.list()
            print(f"✅ Ollama 服务正常，可用模型: {len(models['models'])} 个")
            
            # 检查是否有 nomic-embed-text 模型
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
    
    def is_valid_wiki_url(self, url):
        """检查是否为有效的 Wiki URL - 支持中英文页面"""
        if not url or 'wiki.seeedstudio.com' not in url:
            return False
        
        # 排除文件类型
        exclude_patterns = [
            r'\.(pdf|doc|docx|xls|xlsx|zip|rar|jpg|jpeg|png|gif|svg|ico|css|js)$',
            r'#.*$', r'\?.*$', r'/api/', r'/admin/',
        ]
        
        for pattern in exclude_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        return True
    
    def get_page_hash(self, url, content):
        """获取页面内容的哈希值"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return content_hash
    
    def is_page_updated(self, url, content):
        """检查页面是否有更新"""
        current_hash = self.get_page_hash(url, content)
        old_hash = self.url_hashes.get(url)
        
        if old_hash != current_hash:
            self.url_hashes[url] = current_hash
            return True
        
        return False
    
    def normalize_url(self, url):
        """标准化 URL"""
        if not url.startswith('http'):
            url = urljoin(self.base_url, url)
        url = url.split('#')[0].split('?')[0]
        if not re.search(r'\.[a-zA-Z0-9]+$', url) and not url.endswith('/'):
            url += '/'
        return url
    
    def extract_links_from_page(self, soup, current_url):
        """从页面中提取所有有效的链接"""
        links = set()
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href:
                full_url = self.normalize_url(href)
                if self.is_valid_wiki_url(full_url):
                    links.add(full_url)
        return links
    
    def extract_page_content(self, soup, url):
        """提取页面内容 - 支持中英文"""
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No Title"
        
        content = ""
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', class_='content') or 
            soup.find('div', class_='main') or
            soup.find('div', class_='theme-doc-markdown') or
            soup.find('div', class_='markdown') or
            soup.find('div', {'role': 'main'})
        )
        
        if main_content:
            # 移除脚本、样式、导航等元素
            for element in main_content(["script", "style", "nav", "header", "footer"]):
                element.decompose()
            
            # 获取内容（中英文都获取）
            paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            if paragraphs:
                intro_content = []
                char_count = 0
                max_chars = 800  # 增加字符数限制
                
                for p in paragraphs[:8]:  # 最多取前8个段落
                    p_text = p.get_text().strip()
                    if p_text and len(p_text) > 10:  # 过滤太短的段落
                        intro_content.append(p_text)
                        char_count += len(p_text)
                        
                        # 如果已经达到字符限制，停止收集
                        if char_count >= max_chars:
                            break
                
                content = ' '.join(intro_content)
                
                # 如果内容太短，尝试获取更多内容
                if len(content) < 200:
                    full_text = main_content.get_text(separator=' ', strip=True)
                    content = full_text[:1000]
                    
                    # 在句号处截断
                    if '.' in content:
                        last_period = content.rfind('.')
                        if last_period > 600:
                            content = content[:last_period + 1]
            else:
                # 如果没有找到段落标签，获取前1000字符
                full_text = main_content.get_text(separator=' ', strip=True)
                content = full_text[:1000]
                
                # 在句号处截断
                if '.' in content:
                    last_period = content.rfind('.')
                    if last_period > 600:
                        content = content[:last_period + 1]
        else:
            # 如果没有找到主要内容区域，获取 body 内容的前部分
            body = soup.find('body')
            if body:
                for element in body(["script", "style", "nav", "header", "footer", "aside"]):
                    element.decompose()
                
                full_text = body.get_text(separator=' ', strip=True)
                content = full_text[:800]  # 限制为800字符
                
                # 在句号处截断
                if '.' in content:
                    last_period = content.rfind('.')
                    if last_period > 400:
                        content = content[:last_period + 1]
        
        # 清理内容
        content = re.sub(r'\s+', ' ', content).strip()
        
        # 添加内容摘要标记
        if content and len(content) > 0:
            # 检测语言
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
            english_chars = len(re.findall(r'[a-zA-Z]', content))
            
            if chinese_chars > english_chars:
                content = f"[中文介绍] {content}"
            else:
                content = f"[English Introduction] {content}"
            
            # 如果内容被截断，添加省略号
            if len(content) > 900:
                content = content[:900] + "..."
        
        return title_text, content
    
    def generate_embedding(self, text):
        """使用 Ollama 生成文本的 embedding 向量"""
        try:
            # 使用 Ollama nomic-embed-text 模型生成 embedding
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            embedding = response["embedding"]
            
            # 转换为 numpy 数组
            embedding = np.array(embedding, dtype=np.float32)
            
            # 归一化向量（用于余弦相似度计算）
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            print(f"❌ Embedding 生成失败: {str(e)}")
            return None
    
    def scrape_page(self, url, depth=0):
        """爬取单个页面并生成 embedding"""
        if url in self.visited_urls or depth > self.max_depth:
            return set()
        
        print(f"正在爬取 (深度 {depth}): {url}")
        self.visited_urls.add(url)
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            title, content = self.extract_page_content(soup, url)
            
            # 内容过滤
            if content and len(content) > 50:
                # 检查页面是否有更新
                if not self.is_page_updated(url, content):
                    print(f"  ⏭️ 页面无更新，跳过: {title}")
                    return set()
                
                # 生成 embedding
                print(f"  🔍 生成 Embedding...")
                embedding = self.generate_embedding(content)
                
                if embedding is not None:
                    # 检查是否已存在该页面
                    existing_page = None
                    for i, page in enumerate(self.all_content):
                        if page['url'] == url:
                            existing_page = i
                            break
                    
                    # 保存页面数据
                    page_data = {
                        'url': url,
                        'title': title,
                        'content': content,
                        'depth': depth,
                        'content_length': len(content),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'language': '中文' if '[中文介绍]' in content else 'English'
                    }
                    
                    if existing_page is not None:
                        # 更新现有页面
                        self.all_content[existing_page] = page_data
                        print(f"  🔄 已更新: {title}")
                    else:
                        # 添加新页面
                        self.all_content.append(page_data)
                        print(f"  ✅ 已保存: {title}")
                    
                    # 更新向量数据
                    if existing_page is not None:
                        # 更新现有向量
                        self.faiss_vectors[existing_page] = embedding
                        self.faiss_metadata[existing_page] = {
                            'title': title,
                            'url': url,
                            'content_length': len(content),
                            'timestamp': page_data['timestamp'],
                            'language': page_data['language']
                        }
                    else:
                        # 添加新向量
                        self.faiss_vectors.append(embedding)
                        self.faiss_metadata.append({
                            'title': title,
                            'url': url,
                            'content_length': len(content),
                            'timestamp': page_data['timestamp'],
                            'language': page_data['language']
                        })
                    
                    print(f"    内容长度: {len(content)} 字符")
                    print(f"    Embedding 维度: {embedding.shape}")
                    print(f"    语言: {page_data['language']}")
                else:
                    print(f"  ⚠ Embedding 生成失败，跳过: {title}")
            else:
                print(f"  ⚠ 内容过短 ({len(content) if content else 0} 字符)，跳过: {title}")
            
            # 提取页面中的链接
            new_links = self.extract_links_from_page(soup, url)
            for link in new_links:
                if link not in self.visited_urls:
                    self.url_queue.append((link, depth + 1))
            
            return new_links
            
        except Exception as e:
            print(f"  ✗ 爬取失败: {str(e)}")
            return set()
    
    def discover_initial_links(self):
        """发现初始链接集合"""
        print("正在发现初始链接...")
        
        initial_urls = [
            self.base_url,
            f"{self.base_url}/Getting_Started/",
            f"{self.base_url}/sitemap.xml",
        ]
        
        discovered_links = set()
        
        for url in initial_urls:
            try:
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    links = self.extract_links_from_page(soup, url)
                    discovered_links.update(links)
                    print(f"从 {url} 发现 {len(links)} 个链接")
            except Exception as e:
                print(f"无法访问 {url}: {str(e)}")
        
        # 添加中英文目录
        known_directories = [
            # 英文目录
            "/SensorandSensing/",
            "/Networking/", 
            "/EdgeComputing/",
            "/Cloud/",
            "/TechnologyTopics/",
            "/Contributions/",
            "/WeeklyWiki/",
            "/XIAO/",
            "/Grove/",
            "/SenseCAP/",
            "/reComputer/",
            "/WioTerminal/",
            "/Odyssey/",
            "/RaspberryPi/",
            "/NVIDIAJetson/",
            "/ESPDevices/",
            "/BeagleBone/",
            "/Arduino/",
            "/Microbit/",
            # 中文目录
            "/zh/",
            "/zh-cn/",
            "/zh/SensorandSensing/",
            "/zh/Networking/",
            "/zh/EdgeComputing/",
            "/zh/Cloud/",
            "/zh/TechnologyTopics/",
            "/zh/Contributions/",
            "/zh/WeeklyWiki/",
            "/zh/XIAO/",
            "/zh/Grove/",
            "/zh/SenseCAP/",
            "/zh/reComputer/",
            "/zh/WioTerminal/",
            "/zh/Odyssey/",
            "/zh/RaspberryPi/",
            "/zh/NVIDIAJetson/",
            "/zh/ESPDevices/",
            "/zh/BeagleBone/",
            "/zh/Arduino/",
            "/zh/Microbit/"
        ]
        
        for directory in known_directories:
            full_url = urljoin(self.base_url, directory)
            if self.is_valid_wiki_url(full_url):
                discovered_links.add(full_url)
        
        print(f"总共发现 {len(discovered_links)} 个初始链接")
        return discovered_links
    
    def build_faiss_index(self):
        """构建 FAISS 索引"""
        if not self.faiss_vectors:
            print("⚠️ 没有向量数据，跳过索引构建")
            return False
        
        print(f"\n🔧 构建 FAISS 索引...")
        print(f"   向量数量: {len(self.faiss_vectors)}")
        print(f"   向量维度: {self.dimension}")
        
        try:
            # 过滤掉None值
            valid_vectors = [v for v in self.faiss_vectors if v is not None]
            if not valid_vectors:
                print("⚠️ 没有有效的向量数据")
                return False
            
            # 将向量转换为 numpy 数组
            vectors_array = np.array(valid_vectors, dtype=np.float32)
            
            # 创建新索引
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
            self.faiss_index.add(vectors_array)
            
            print(f"✅ FAISS 索引构建完成")
            print(f"   索引大小: {self.faiss_index.ntotal}")
            
            return True
            
        except Exception as e:
            print(f"❌ FAISS 索引构建失败: {str(e)}")
            return False
    
    def save_embeddings_and_index(self):
        """保存 embedding 向量和 FAISS 索引"""
        print(f"\n💾 保存 Embedding 和索引...")
        
        # 保存页面数据
        output_file = f"{self.data_dir}/seeed_wiki_embeddings.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_content, f, ensure_ascii=False, indent=2)
        print(f"📄 页面数据已保存到: {output_file}")
        
        # 保存 FAISS 索引
        faiss.write_index(self.faiss_index, self.faiss_index_file)
        print(f"🔍 FAISS 索引已保存到: {self.faiss_index_file}")
        
        # 保存向量元数据
        with open(self.faiss_metadata_file, 'wb') as f:
            pickle.dump(self.faiss_metadata, f)
        print(f"📊 向量元数据已保存到: {self.faiss_metadata_file}")
        
        # 保存URL哈希值
        self.save_url_hashes()
        print(f"🔗 URL哈希值已保存到: {self.url_hash_file}")
        
        # 更新最后更新时间
        self.update_last_update_time()
        print(f"⏰ 最后更新时间已更新")
        
        # 保存为数据库格式
        db_data = {
            'metadata': {
                'total_pages': len(self.all_content),
                'total_vectors': len([v for v in self.faiss_vectors if v is not None]),
                'vector_dimension': self.dimension,
                'crawl_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'base_url': self.base_url,
                'max_depth': self.max_depth,
                'content_type': '中英文页面介绍摘要',
                'embedding_model': 'nomic-embed-text',
                'index_type': 'FAISS_IndexFlatIP',
                'languages': ['中文', 'English'],
                'last_update': datetime.now().isoformat()
            },
            'pages': self.all_content
        }
        
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(db_data, f, ensure_ascii=False, indent=2)
        print(f"🗄️ 数据库格式已保存到: {self.db_file}")
        
        # 统计信息
        total_chars = sum(page.get('content_length', 0) for page in self.all_content)
        avg_chars = total_chars / len(self.all_content) if self.all_content else 0
        
        # 语言统计
        chinese_pages = len([p for p in self.all_content if p.get('language') == '中文'])
        english_pages = len([p for p in self.all_content if p.get('language') == 'English'])
        
        print(f"\n📊 爬取统计:")
        print(f"   总页面数: {len(self.all_content)}")
        print(f"   中文页面: {chinese_pages}")
        print(f"   英文页面: {english_pages}")
        print(f"   总向量数: {len([v for v in self.faiss_vectors if v is not None])}")
        print(f"   总字符数: {total_chars:,}")
        print(f"   平均字符数: {avg_chars:.1f}")
        print(f"   向量维度: {self.dimension}")
        
        print(f"\n✅ 所有数据已保存完成！")
        print(f"💡 现在可以使用优化版问答系统了")
        print(f"🚀 启动命令: python optimized_qa.py")
    
    def run_incremental_update(self):
        """运行增量更新"""
        print("🔄 开始增量更新 Seeed Studio Wiki")
        print(f"基础 URL: {self.base_url}")
        print(f"最大深度: {self.max_depth}")
        print(f"Embedding 模型: {self.embedding_model}")
        
        # 检查是否需要更新
        if not self.should_update():
            print("⏰ 距离上次更新未满24小时，跳过更新")
            return
        
        # 发现初始链接
        initial_links = self.discover_initial_links()
        
        # 将初始链接添加到队列
        for link in initial_links:
            self.url_queue.append((link, 0))
        
        # 开始爬取
        processed_count = 0
        new_pages_count = 0
        updated_pages_count = 0
        
        while self.url_queue:
            url, depth = self.url_queue.popleft()
            
            if url in self.visited_urls:
                continue
            
            # 检查是否是新页面或需要更新
            existing_page = None
            for page in self.all_content:
                if page['url'] == url:
                    existing_page = page
                    break
            
            if existing_page:
                # 检查是否需要更新
                try:
                    response = self.session.head(url, timeout=10)
                    if response.status_code == 200:
                        # 获取页面内容检查是否有更新
                        full_response = self.session.get(url, timeout=15)
                        soup = BeautifulSoup(full_response.content, 'html.parser')
                        title, content = self.extract_page_content(soup, url)
                        
                        if self.is_page_updated(url, content):
                            # 页面有更新，重新爬取
                            self.scrape_page(url, depth)
                            updated_pages_count += 1
                        else:
                            print(f"⏭️ 页面无更新，跳过: {existing_page['title']}")
                            self.visited_urls.add(url)
                except Exception as e:
                    print(f"⚠️ 检查页面更新失败: {str(e)}")
                    self.visited_urls.add(url)
            else:
                # 新页面，爬取
                self.scrape_page(url, depth)
                new_pages_count += 1
            
            processed_count += 1
            
            # 显示进度
            if processed_count % 10 == 0:
                print(f"\n📊 进度: 已处理 {processed_count} 个页面，队列中还有 {len(self.url_queue)} 个")
                print(f"📁 新页面: {new_pages_count}，更新页面: {updated_pages_count}")
            
            # 添加延迟避免请求过快
            time.sleep(0.5)
        
        # 构建 FAISS 索引
        if self.build_faiss_index():
            # 保存所有数据
            self.save_embeddings_and_index()
        
        print(f"\n🎉 增量更新完成！")
        print(f"📊 统计信息:")
        print(f"  - 总共访问: {len(self.visited_urls)} 个页面")
        print(f"  - 新页面: {new_pages_count} 个")
        print(f"  - 更新页面: {updated_pages_count} 个")
        print(f"  - 总页面数: {len(self.all_content)} 个")
    
    def run_full_crawl(self):
        """运行完整爬取"""
        print("🚀 开始完整爬取 Seeed Studio Wiki (中英文页面介绍 + Embedding)")
        print(f"基础 URL: {self.base_url}")
        print(f"最大深度: {self.max_depth}")
        print(f"Embedding 模型: {self.embedding_model}")
        print(f"向量索引: FAISS 索引")
        
        # 清空现有数据
        self.all_content = []
        self.faiss_vectors = []
        self.faiss_metadata = []
        self.url_hashes = {}
        self.visited_urls = set()
        
        # 发现初始链接
        initial_links = self.discover_initial_links()
        
        # 将初始链接添加到队列
        for link in initial_links:
            self.url_queue.append((link, 0))
        
        # 开始爬取
        processed_count = 0
        while self.url_queue:
            url, depth = self.url_queue.popleft()
            
            if url in self.visited_urls:
                continue
            
            # 爬取页面
            self.scrape_page(url, depth)
            processed_count += 1
            
            # 显示进度
            if processed_count % 10 == 0:
                print(f"\n📊 进度: 已处理 {processed_count} 个页面，队列中还有 {len(self.url_queue)} 个")
                print(f"📁 已保存 {len(self.all_content)} 个页面")
                print(f"🔍 已生成 {len(self.faiss_vectors)} 个向量")
            
            # 添加延迟避免请求过快
            time.sleep(0.5)
        
        # 构建 FAISS 索引
        if self.build_faiss_index():
            # 保存所有数据
            self.save_embeddings_and_index()
        
        print(f"\n🎉 完整爬取完成！")
        print(f"📊 统计信息:")
        print(f"  - 总共访问: {len(self.visited_urls)} 个页面")
        print(f"  - 成功保存: {len(self.all_content)} 个页面")
        print(f"  - 生成向量: {len(self.faiss_vectors)} 个")
        print(f"  - 最大深度: {max([page['depth'] for page in self.all_content]) if self.all_content else 0}")
        
        return self.all_content
    
    def run_continuous_monitor(self):
        """持续运行监控模式 - 检查新页面并定时更新"""
        print("🔄 启动持续监控模式")
        print("📊 功能说明:")
        print("   - 实时检查新页面并更新到本地")
        print("   - 每天凌晨12点自动进行完整数据库更新")
        print("   - 按 Ctrl+C 停止监控")
        
        def daily_update_job():
            print(f"\n⏰ 执行每日定时更新任务 - {datetime.now()}")
            try:
                print("🔄 开始每日完整更新...")
                self.run_incremental_update()
                print("✅ 每日更新完成")
            except Exception as e:
                print(f"❌ 每日更新失败: {str(e)}")
        
        def continuous_check_job():
            """持续检查新页面的任务"""
            try:
                print(f"\n🔍 执行持续检查任务 - {datetime.now()}")
                self.run_quick_check()
            except Exception as e:
                print(f"❌ 持续检查失败: {str(e)}")
        
        # 设置定时任务
        schedule.every().day.at("00:00").do(daily_update_job)  # 每天凌晨12点
        schedule.every(30).minutes.do(continuous_check_job)    # 每30分钟检查一次
        
        print("⏰ 定时任务设置:")
        print("   - 每日凌晨 00:00: 完整数据库更新")
        print("   - 每 30 分钟: 快速检查新页面")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次定时任务
        except KeyboardInterrupt:
            print("\n⏹️ 持续监控已停止")
    
    def run_quick_check(self):
        """快速检查新页面（不进行深度爬取）"""
        print("🔍 快速检查新页面...")
        
        # 获取主页面的链接
        try:
            response = self.session.get(self.base_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                new_links = self.extract_links_from_page(soup, self.base_url)
                
                # 检查是否有新页面
                new_pages_found = 0
                for link in new_links:
                    if link not in self.url_hashes:
                        # 发现新页面，立即爬取
                        print(f"🆕 发现新页面: {link}")
                        try:
                            self.scrape_page(link, 0)
                            new_pages_found += 1
                        except Exception as e:
                            print(f"⚠️ 爬取新页面失败: {str(e)}")
                
                if new_pages_found > 0:
                    print(f"✅ 快速检查完成，发现并爬取了 {new_pages_found} 个新页面")
                    # 保存更新
                    self.save_embeddings_and_index()
                else:
                    print("✅ 快速检查完成，没有发现新页面")
            else:
                print(f"⚠️ 无法访问主页: {response.status_code}")
        except Exception as e:
            print(f"❌ 快速检查失败: {str(e)}")
    
    def schedule_daily_update(self):
        """设置每日定时更新（兼容旧版本）"""
        def daily_update_job():
            print(f"\n⏰ 执行定时更新任务 - {datetime.now()}")
            try:
                self.run_incremental_update()
            except Exception as e:
                print(f"❌ 定时更新失败: {str(e)}")
        
        # 每天凌晨2点执行更新
        schedule.every().day.at("02:00").do(daily_update_job)
        
        print("⏰ 已设置每日凌晨2点自动更新")
        print("🔄 定时任务已启动，按 Ctrl+C 停止")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
        except KeyboardInterrupt:
            print("\n⏹️ 定时任务已停止")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Seeed Wiki 爬虫')
    parser.add_argument('--mode', choices=['full', 'incremental', 'schedule', 'monitor'], 
                       default='incremental', help='运行模式')
    parser.add_argument('--force', action='store_true', help='强制完整爬取')
    parser.add_argument('--check-interval', type=int, default=30, 
                       help='监控模式下的检查间隔（分钟）')
    
    args = parser.parse_args()
    
    scraper = OptimizedWikiScraper()
    
    try:
        if args.mode == 'full' or args.force:
            scraper.run_full_crawl()
        elif args.mode == 'schedule':
            scraper.schedule_daily_update()
        elif args.mode == 'monitor':
            scraper.run_continuous_monitor()
        else:
            scraper.run_incremental_update()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断爬取")
        if scraper.all_content:
            scraper.save_embeddings_and_index()
    except Exception as e:
        print(f"\n❌ 爬取过程中发生错误: {str(e)}")
        if scraper.all_content:
            scraper.save_embeddings_and_index()

if __name__ == "__main__":
    main()
