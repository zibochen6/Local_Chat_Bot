#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MQTT服务器 - 处理产品讲解请求
"""

import json
import logging
import time
from typing import Dict, Any

import paho.mqtt.client as mqtt
from config import (
    MQTT_BROKER, MQTT_PORT, MQTT_TOPIC_PRODUCT_REQUEST, 
    MQTT_TOPIC_PRODUCT_RESPONSE, PRODUCTS_DB, DEFAULT_PRODUCT
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductMQTTServer:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        
        # 设置客户端ID
        self.client_id = f"product_server_{int(time.time())}"
        self.client._client_id = self.client_id
        
        # 连接状态
        self.connected = False
        
    def on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self.connected = True
            logger.info(f"成功连接到MQTT代理服务器 {MQTT_BROKER}:{MQTT_PORT}")
            
            # 订阅产品请求主题
            client.subscribe(MQTT_TOPIC_PRODUCT_REQUEST)
            logger.info(f"已订阅主题: {MQTT_TOPIC_PRODUCT_REQUEST}")
            
        else:
            logger.error(f"连接失败，返回码: {rc}")
            
    def on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        self.connected = False
        logger.warning(f"与MQTT代理服务器断开连接，返回码: {rc}")
        
    def on_message(self, client, userdata, msg):
        """消息接收回调"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            logger.info(f"收到消息 - 主题: {topic}, 内容: {payload}")
            
            if topic == MQTT_TOPIC_PRODUCT_REQUEST:
                self.handle_product_request(payload)
                
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
            
    def handle_product_request(self, payload: str):
        """处理产品请求"""
        try:
            # 解析请求数据
            request_data = json.loads(payload)
            product_id = request_data.get('product_id')
            request_id = request_data.get('request_id', str(int(time.time())))
            
            logger.info(f"处理产品请求 - ID: {product_id}, 请求ID: {request_id}")
            
            # 获取产品信息
            product_info = self.get_product_info(product_id)
            
            # 生成AI讲解（这里可以集成本地大模型）
            ai_explanation = self.generate_ai_explanation(product_info)
            
            # 构建响应
            response = {
                'request_id': request_id,
                'product_id': product_id,
                'product_name': product_info['name'],
                'description': product_info['description'],
                'ai_explanation': ai_explanation,
                'timestamp': int(time.time())
            }
            
            # 发送响应
            self.publish_response(response)
            
        except json.JSONDecodeError:
            logger.error("无效的JSON格式")
        except Exception as e:
            logger.error(f"处理产品请求时出错: {e}")
            
    def get_product_info(self, product_id: str) -> Dict[str, Any]:
        """获取产品信息"""
        return PRODUCTS_DB.get(product_id, DEFAULT_PRODUCT)
        
    def generate_ai_explanation(self, product_info: Dict[str, Any]) -> str:
        """生成AI讲解（模拟本地大模型）"""
        name = product_info['name']
        features = product_info['features']
        
        # 这里可以集成真实的本地大模型
        # 目前使用模板生成讲解
        explanation = f"""
欢迎了解我们的{name}！

{product_info['description']}

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
        
    def publish_response(self, response: Dict[str, Any]):
        """发布响应消息"""
        try:
            payload = json.dumps(response, ensure_ascii=False)
            result = self.client.publish(MQTT_TOPIC_PRODUCT_RESPONSE, payload)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"成功发送产品讲解响应 - 产品ID: {response['product_id']}")
            else:
                logger.error(f"发送响应失败，返回码: {result.rc}")
                
        except Exception as e:
            logger.error(f"发送响应时出错: {e}")
            
    def connect(self):
        """连接到MQTT代理服务器"""
        try:
            logger.info(f"正在连接到MQTT代理服务器 {MQTT_BROKER}:{MQTT_PORT}")
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            
        except Exception as e:
            logger.error(f"连接MQTT代理服务器失败: {e}")
            return False
            
        return True
        
    def start(self):
        """启动MQTT服务器"""
        if not self.connect():
            return False
            
        try:
            # 启动网络循环
            self.client.loop_start()
            logger.info("MQTT服务器已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动MQTT服务器失败: {e}")
            return False
            
    def stop(self):
        """停止MQTT服务器"""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("MQTT服务器已停止")
            
        except Exception as e:
            logger.error(f"停止MQTT服务器时出错: {e}")
            
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.connected

def main():
    """主函数"""
    server = ProductMQTTServer()
    
    try:
        if server.start():
            logger.info("MQTT服务器运行中... 按Ctrl+C停止")
            
            # 保持运行
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("收到停止信号")
    finally:
        server.stop()

if __name__ == "__main__":
    main()
