# Seeed Studio Wiki 智能问答系统

一个基于 Seeed Studio Wiki 的智能问答系统，使用 Ollama + FAISS 实现快速的知识检索和问答功能。

## 🚀 核心特性

- **预生成 Embedding**: 爬取时直接生成向量，避免启动时等待
- **FAISS 向量索引**: 使用 FAISS 进行高效的相似度搜索
- **Ollama 集成**: 使用 `nomic-embed-text` 模型生成高质量向量
- **快速启动**: 无需重新生成向量，直接加载预保存的索引
- **多语言支持**: 支持中英文混合查询

## 📁 项目结构

```
chat/
├── scrape_with_embeddings.py    # 主要爬虫脚本
├── optimized_qa.py              # 优化问答系统
├── requirements.txt             # Python 依赖
├── faiss_index.bin             # FAISS 向量索引
├── faiss_metadata.pkl          # 向量元数据
├── seeed_wiki_embeddings_db.json # Wiki 页面数据
├── 优化系统使用说明.md          # 详细使用说明
└── README.md                   # 项目说明
```

## 🛠️ 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 使用方法

### 1. 爬取 Wiki 内容并生成向量

```bash
python scrape_with_embeddings.py
```

这将：
- 爬取 Seeed Studio Wiki 的英文页面
- 提取每个页面的介绍部分
- 使用 Ollama 生成 768 维向量
- 保存到 FAISS 索引和 JSON 文件

### 2. 启动问答系统

```bash
python optimized_qa.py
```

系统将：
- 加载预保存的 FAISS 索引
- 启动 Ollama 服务
- 提供交互式问答界面

## 💡 示例问题

- "介绍一下XIAO系列产品"
- "Grove传感器模块有什么特点？"
- "SenseCAP的功能是什么？"
- "Edge Computing是什么？"
- "reComputer有什么特色？"

## 🔧 系统要求

- Python 3.8+
- Ollama 服务运行中
- `nomic-embed-text` 模型已安装

## 📊 性能特点

- **启动时间**: < 5 秒（相比传统方式快 10-20 倍）
- **搜索速度**: < 1 秒（FAISS 索引）
- **向量维度**: 768 维（nomic-embed-text）
- **支持页面**: 英文 Wiki 页面介绍摘要

## 🎯 设计理念

- **爬取时向量化**: 避免每次启动时重新生成向量
- **轻量级模型**: 使用 `nomic-embed-text` 而非大型模型
- **本地存储**: FAISS 索引 + JSON 数据，无需外部数据库
- **快速检索**: 基于余弦相似度的向量搜索

## 📝 注意事项

- 确保 Ollama 服务正在运行
- 首次运行需要安装 `nomic-embed-text` 模型
- 爬虫支持中断恢复，按 Ctrl+C 可安全停止
- 系统会自动检测和安装必要的模型

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！
# Local_Chat_Bot
