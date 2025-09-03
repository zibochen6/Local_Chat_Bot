#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web服务器 - 提供手机端访问页面
"""

import json
import logging
import time
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, session, redirect
from config import FLASK_HOST, FLASK_PORT, PRODUCTS_DB, JETSON_IP
import ollama
import json
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'recomputer_secret_key'  # 用于session

# 存储产品请求状态
request_status = {}

@app.route('/')
def index():
    """主页"""
    # 获取用户语言偏好，默认为中文
    language = session.get('language', 'zh')
    return render_template('index.html', products=PRODUCTS_DB, language=language, jetson_ip=JETSON_IP)

@app.route('/switch_language/<language>')
def switch_language(language):
    """切换语言"""
    if language in ['zh', 'en']:
        session['language'] = language
        print(f"语言已切换到: {language}")
    return redirect(request.referrer or '/')

@app.route('/api/ai_explanation', methods=['POST'])
def get_ai_explanation():
    """获取AI讲解（调用本地大模型）"""
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        language = data.get('language', 'zh')
        
        if not product_id:
            return jsonify({'error': '缺少产品ID'}), 400
            
        product = PRODUCTS_DB.get(product_id)
        if not product:
            return jsonify({'error': '产品不存在'}), 404
            
        # 调用本地大模型生成讲解
        ai_explanation = generate_ai_explanation_with_llm(product, language)
        
        response = {
            'product_id': product_id,
            'ai_explanation': ai_explanation,
            'language': language,
            'timestamp': int(time.time())
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"获取AI讲解时出错: {e}")
        return jsonify({'error': '服务器内部错误'}), 500

def generate_ai_explanation_with_llm(product, language='zh'):
    """使用本地大模型生成产品讲解"""
    try:
        # 构建产品信息上下文
        if language == 'zh':
            context = f"""
产品名称: {product['name']}
产品描述: {product['description']}
主要特性: {', '.join(product['features'])}
技术规格:
- AI性能: {product['specs']['ai_performance']}
- GPU: {product['specs']['gpu']}
- CPU: {product['specs']['cpu']}
- 内存: {product['specs']['memory']}
- 存储: {product['specs']['storage']}
- 视频编码: {product['specs']['video_encoder']}
- 视频解码: {product['specs']['video_decoder']}
- 接口: {product['specs']['interfaces']}

请详细介绍这款产品的特点、应用场景和技术优势，用中文回答。
"""
            system_prompt = "你是Seeed Studio的技术专家，请用专业且易懂的中文介绍产品。"
        else:
            context = f"""
Product Name: {product['name_en']}
Product Description: {product['description_en']}
Key Features: {', '.join(product['features_en'])}
Technical Specifications:
- AI Performance: {product['specs']['ai_performance']}
- GPU: {product['specs']['gpu']}
- CPU: {product['specs']['cpu']}
- Memory: {product['specs']['memory']}
- Storage: {product['specs']['storage']}
- Video Encoder: {product['specs']['video_encoder']}
- Video Decoder: {product['specs']['video_decoder']}
- Interfaces: {product['specs']['interfaces']}

Please provide a detailed introduction to this product's features, applications, and technical advantages in English.
"""
            system_prompt = "You are a technical expert at Seeed Studio. Please introduce the product professionally and understandably in English."
        
        # 调用Ollama本地大模型
        response = ollama.chat(
            model='qwen2.5:3b',  # 使用qwen2.5:3b模型
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': context}
            ],
            options={
                'temperature': 0.7,
                'top_p': 0.9,
                'num_predict': 500,
            }
        )
        
        return response['message']['content'].strip()
        
    except Exception as e:
        logger.error(f"AI讲解生成失败: {e}")
        # 返回备用讲解
        if language == 'zh':
            return f"""
{product['name']}是矽递科技基于NVIDIA Jetson Orin NX平台开发的高性能边缘AI计算设备。

这款产品具有以下核心优势：

🎯 **强大的AI性能**: 提供70-100 TOPS的AI计算能力，能够运行复杂的深度学习模型和算法。

🚀 **先进的硬件架构**: 采用1024核NVIDIA Ampere架构GPU，配备32个Tensor Core，支持最新的AI框架如TensorFlow、PyTorch等。

💾 **高速内存系统**: 8-16GB LPDDR5内存，带宽高达102.4GB/s，确保AI任务的高效执行。

📹 **专业视频处理**: 支持4K视频编解码，12路1080p30编码和4路4K30解码能力，适合视频分析应用。

🔌 **丰富的接口配置**: 提供4个USB 3.2接口、HDMI 2.1、2个CSI摄像头接口、千兆以太网、M.2扩展槽等，满足各种应用需求。

reComputer J40x特别适用于机器人、无人机、智能监控、工业自动化、边缘AI推理等需要本地AI处理能力的场景，为用户提供专业级的边缘计算解决方案。
"""
        else:
            return f"""
{product['name_en']} is a high-performance edge AI computing device developed by Seeed Studio based on the NVIDIA Jetson Orin NX platform.

This product offers the following core advantages:

🎯 **Powerful AI Performance**: Provides 70-100 TOPS of AI computing power, capable of running complex deep learning models and algorithms.

🚀 **Advanced Hardware Architecture**: Features a 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores, supporting the latest AI frameworks such as TensorFlow and PyTorch.

💾 **High-Speed Memory System**: 8-16GB LPDDR5 memory with bandwidth up to 102.4GB/s, ensuring efficient execution of AI tasks.

📹 **Professional Video Processing**: Supports 4K video codec, 12-channel 1080p30 encoding and 4-channel 4K30 decoding capabilities, suitable for video analysis applications.

🔌 **Rich Interface Configuration**: Provides 4 USB 3.2 interfaces, HDMI 2.1, 2 CSI camera interfaces, Gigabit Ethernet, M.2 expansion slots, etc., meeting various application requirements.

reComputer J40x is particularly suitable for robotics, drones, intelligent monitoring, industrial automation, edge AI inference, and other scenarios that require local AI processing capabilities, providing users with professional-grade edge computing solutions.
"""

@app.route('/product/<product_id>')
def product_page(product_id):
    """产品页面"""
    product = PRODUCTS_DB.get(product_id, None)
    if not product:
        return "产品不存在", 404
    
    # 获取用户语言偏好，默认为中文
    language = session.get('language', 'zh')
    return render_template('product.html', product=product, product_id=product_id, language=language, jetson_ip=JETSON_IP)

@app.route('/api/request_explanation', methods=['POST'])
def request_explanation():
    """请求产品讲解API"""
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        
        if not product_id:
            return jsonify({'error': '缺少产品ID'}), 400
            
        # 生成请求ID
        request_id = str(int(time.time()))
        
        # 存储请求状态
        request_status[request_id] = {
            'product_id': product_id,
            'status': 'pending',
            'timestamp': time.time()
        }
        
        logger.info(f"收到产品讲解请求 - 产品ID: {product_id}, 请求ID: {request_id}")
        
        # 这里可以发送MQTT消息到服务器
        # 目前返回模拟响应
        response = {
            'request_id': request_id,
            'status': 'success',
            'message': '请求已接收，正在生成讲解...',
            'product_id': product_id
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"处理讲解请求时出错: {e}")
        return jsonify({'error': '服务器内部错误'}), 500

@app.route('/api/explanation/<request_id>')
def get_explanation(request_id):
    """获取产品讲解结果"""
    try:
        if request_id not in request_status:
            return jsonify({'error': '请求ID不存在'}), 404
            
        request_info = request_status[request_id]
        product_id = request_info['product_id']
        product = PRODUCTS_DB.get(product_id, None)
        
        if not product:
            return jsonify({'error': '产品不存在'}), 404
            
        # 模拟AI讲解生成
        ai_explanation = generate_ai_explanation(product)
        
        response = {
            'request_id': request_id,
            'product_id': product_id,
            'product_name': product['name'],
            'description': product['description'],
            'ai_explanation': ai_explanation,
            'timestamp': int(time.time())
        }
        
        # 更新请求状态
        request_status[request_id]['status'] = 'completed'
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"获取讲解结果时出错: {e}")
        return jsonify({'error': '服务器内部错误'}), 500

def generate_ai_explanation(product: Dict[str, Any]) -> str:
    """生成AI讲解（模拟）"""
    name = product['name']
    features = product['features']
    
    explanation = f"""
欢迎了解我们的{name}！

{product['description']}

让我为您详细介绍这款产品的特色功能：

"""
    
    for i, feature in enumerate(features, 1):
        explanation += f"{i}. {feature}\n"
        
    explanation += f"""
这款{name}采用了最新的技术，为用户提供了卓越的使用体验。
如果您有任何问题，我们的客服团队随时为您服务。

感谢您对我们产品的关注！
"""
    
    return explanation.strip()

@app.route('/api/products')
def get_products():
    """获取所有产品列表"""
    return jsonify(PRODUCTS_DB)

@app.route('/api/product/<product_id>')
def get_product(product_id):
    """获取单个产品信息"""
    product = PRODUCTS_DB.get(product_id, None)
    if not product:
        return jsonify({'error': '产品不存在'}), 404
    
    return jsonify(product)

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({'error': '页面不存在'}), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({'error': '服务器内部错误'}), 500

def main():
    """主函数"""
    logger.info(f"启动Web服务器 {FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)

if __name__ == "__main__":
    main()
