# Seeed Wiki 爬虫使用说明

## 功能特性

### 🚀 主要功能
- **中英文内容爬取**: 支持爬取 Seeed Studio Wiki 的中文和英文页面
- **增量更新**: 只爬取新增或更新的页面，避免重复爬取
- **定时检测**: 每24小时自动检测新页面并更新
- **本地 Embedding**: 使用 Ollama nomic-embed-text 模型生成本地向量
- **FAISS 索引**: 使用 FAISS 进行高效的向量检索

### 📊 数据管理
- **智能缓存**: 保存页面哈希值，只更新有变化的页面
- **数据持久化**: 所有数据保存在 `./data_base/` 目录
- **多格式存储**: JSON、FAISS 索引、Pickle 元数据

## 安装依赖

```bash
pip install requests beautifulsoup4 numpy faiss-cpu ollama schedule
```

确保 Ollama 服务正在运行：
```bash
# 安装 nomic-embed-text 模型
ollama pull nomic-embed-text
```

## 使用方法

### 1. 测试功能
```bash
python test_scraper.py
```

### 2. 增量更新（推荐）
```bash
python scrape_with_embeddings.py --mode incremental
```

### 3. 完整爬取
```bash
python scrape_with_embeddings.py --mode full
```

### 4. 定时更新
```bash
python scrape_with_embeddings.py --mode schedule
```

### 5. 持续监控模式（推荐）
```bash
python scrape_with_embeddings.py --mode monitor
```

### 6. 强制完整爬取
```bash
python scrape_with_embeddings.py --mode full --force
```

## 运行模式说明

### 🔄 增量更新模式 (incremental)
- **默认模式**，推荐日常使用
- 检查现有页面是否有更新
- 只爬取新增或修改的页面
- 每24小时检查一次是否需要更新

### 🚀 完整爬取模式 (full)
- 清空现有数据，重新爬取所有页面
- 适用于首次运行或数据重置
- 会爬取所有中英文页面

### ⏰ 定时更新模式 (schedule)
- 设置每日凌晨2点自动更新
- 后台运行，持续监控
- 按 Ctrl+C 停止定时任务

### 🔄 持续监控模式 (monitor) - 推荐
- **实时监控**: 每30分钟检查新页面
- **自动更新**: 发现新页面立即爬取并保存
- **定时维护**: 每天凌晨12点进行完整数据库更新
- **持续运行**: 后台守护进程，24/7运行
- **智能检测**: 只爬取新增或更新的页面

## 数据文件说明

```
./data_base/
├── seeed_wiki_embeddings_db.json    # 主数据库文件
├── seeed_wiki_embeddings.json       # 页面数据备份
├── faiss_index.bin                  # FAISS 向量索引
├── faiss_metadata.pkl               # 向量元数据
├── url_hashes.json                  # URL 哈希值缓存
└── last_update.json                 # 最后更新时间
```

## 配置参数

### 爬取深度
```python
self.max_depth = 4  # 在脚本中修改
```

### 内容长度限制
```python
max_chars = 800  # 在 extract_page_content 方法中修改
```

### 更新间隔
```python
# 在 should_update 方法中修改
return time_diff.total_seconds() >= 24 * 3600  # 24小时
```

## 监控和日志

### 进度显示
- 每处理10个页面显示进度
- 显示新页面和更新页面数量
- 显示队列中待处理页面数量

### 统计信息
- 总页面数（中英文分别统计）
- 向量数量和维度
- 内容字符数统计
- 最后更新时间

## 错误处理

### 网络错误
- 自动重试机制
- 超时设置（15秒）
- 请求延迟（0.5秒）

### 数据损坏
- 自动备份现有数据
- 错误时保存已爬取内容
- 支持从备份恢复

## 性能优化

### 内存管理
- 流式处理大量页面
- 及时释放不需要的数据
- 使用生成器减少内存占用

### 速度优化
- FAISS 索引加速检索
- 增量更新减少重复工作
- 并行处理（可选）

## 常见问题

### Q: Ollama 服务连接失败
A: 确保 Ollama 服务正在运行：
```bash
ollama serve
```

### Q: 内存不足
A: 减少爬取深度或内容长度限制

### Q: 网络超时
A: 增加超时时间或检查网络连接

### Q: 数据文件损坏
A: 删除损坏文件，重新运行完整爬取

## 开发扩展

### 添加新的内容类型
1. 修改 `is_valid_wiki_url` 方法
2. 更新 `extract_page_content` 方法
3. 添加新的语言检测逻辑

### 自定义 Embedding 模型
1. 修改 `embedding_model` 变量
2. 确保 Ollama 中已安装对应模型
3. 测试向量维度兼容性

### 添加新的数据源
1. 继承 `OptimizedWikiScraper` 类
2. 重写相关方法
3. 保持数据格式兼容性

## 许可证

本项目基于 MIT 许可证开源。
