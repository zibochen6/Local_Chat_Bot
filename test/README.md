# 🎯 产品展示系统

基于MQTT协议的产品智能讲解系统，让来访客人通过手机NFC触碰产品卡片获取AI讲解。

## ✨ 功能特点

- 📱 **无需下载应用** - 手机端直接通过浏览器访问
- 🔌 **MQTT通信** - 轻量级消息传输协议
- 🤖 **AI讲解生成** - 集成本地大模型进行产品讲解
- 🎨 **现代化UI** - 响应式设计，支持移动端
- 🚀 **一键启动** - 简单启动脚本，同时运行所有服务

## 🏗️ 系统架构

```
手机端 (NFC触碰) → Web页面 → MQTT请求 → Jetson端处理 → AI讲解生成 → 返回结果
```

## 📁 项目结构

```
test/
├── config.py              # 配置文件
├── mqtt_server.py         # MQTT服务器主程序
├── web_server.py          # Web服务器
├── mqtt_client_test.py    # MQTT客户端测试
├── start_server.py        # 一键启动脚本
├── requirements.txt       # Python依赖
├── templates/             # HTML模板
│   ├── index.html        # 主页
│   └── product.html      # 产品详情页
└── README.md             # 项目说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd test
pip install -r requirements.txt
```

### 2. 启动MQTT代理服务器

**注意**: 需要先安装并启动MQTT代理服务器（如Mosquitto）

```bash
# Ubuntu/Debian
sudo apt-get install mosquitto mosquitto-clients

# 启动服务
sudo systemctl start mosquitto
sudo systemctl enable mosquitto

# 检查状态
sudo systemctl status mosquitto
```

### 3. 启动产品展示系统

```bash
# 一键启动所有服务
python start_server.py

# 或者分别启动
python mqtt_server.py    # 终端1
python web_server.py     # 终端2
```

### 4. 访问系统

- 🌐 **Web界面**: http://localhost:5000
- 🔌 **MQTT服务器**: localhost:1883

## 📱 使用流程

1. **NFC触碰**: 用手机触碰产品NFC卡片
2. **自动跳转**: 手机自动打开产品页面
3. **获取讲解**: 点击"获取讲解"按钮
4. **AI生成**: 系统调用本地大模型生成讲解
5. **查看结果**: 显示AI生成的产品详细讲解

## 🧪 测试功能

### MQTT客户端测试

```bash
python mqtt_client_test.py
```

测试内容：
- 发送产品讲解请求
- 接收AI讲解响应
- 验证通信流程

### 产品数据库

系统内置了3个示例产品：

| 产品ID | 产品名称 | 主要功能 |
|--------|----------|----------|
| 001 | 智能音箱 | 语音控制、高品质音效、智能家居集成 |
| 002 | 智能手表 | 健康监测、运动追踪、消息通知 |
| 003 | 智能摄像头 | 高清录制、夜视功能、移动侦测 |

## ⚙️ 配置说明

### 修改产品信息

编辑 `config.py` 文件中的 `PRODUCTS_DB` 字典：

```python
PRODUCTS_DB = {
    "004": {
        "name": "新产品名称",
        "description": "产品描述...",
        "features": ["功能1", "功能2", "功能3"]
    }
}
```

### 修改服务器配置

```python
# MQTT配置
MQTT_BROKER = "localhost"      # MQTT代理服务器地址
MQTT_PORT = 1883               # MQTT端口

# Web服务器配置
FLASK_HOST = "0.0.0.0"        # 监听地址
FLASK_PORT = 5000              # 监听端口
```

## 🔧 集成本地大模型

在 `mqtt_server.py` 的 `generate_ai_explanation` 方法中集成您的本地大模型：

```python
def generate_ai_explanation(self, product_info: Dict[str, Any]) -> str:
    """生成AI讲解（集成本地大模型）"""
    # 调用本地大模型API
    prompt = f"请详细介绍{product_info['name']}的特点和功能"
    
    # 这里调用您的本地大模型
    # response = local_llm.generate(prompt)
    
    return "AI生成的讲解内容..."
```

## 📊 监控和日志

系统提供详细的日志记录：

- MQTT连接状态
- 产品请求处理
- AI讲解生成过程
- 错误和异常信息

## 🚨 故障排除

### 常见问题

1. **MQTT连接失败**
   - 检查Mosquitto服务是否运行
   - 确认端口1883是否开放

2. **Web页面无法访问**
   - 检查Flask服务器是否启动
   - 确认防火墙设置

3. **依赖安装失败**
   - 使用虚拟环境
   - 升级pip版本

### 日志查看

```bash
# 查看系统日志
journalctl -u mosquitto

# 查看Python程序输出
python mqtt_server.py 2>&1 | tee mqtt.log
```

## 🔒 安全考虑

- 生产环境建议启用MQTT认证
- 配置防火墙规则
- 定期更新依赖包

## 📈 扩展功能

- 添加用户认证
- 集成更多AI模型
- 支持多语言
- 添加产品图片
- 实现语音播报

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用MIT许可证。

---

**注意**: 这是一个演示系统，生产环境使用前请进行充分测试和安全配置。
