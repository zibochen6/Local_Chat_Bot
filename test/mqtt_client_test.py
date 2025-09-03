#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MQTT客户端测试 - 模拟手机端发送产品讲解请求
"""

import json
import time
import logging
from typing import Dict, Any

import paho.mqtt.client as mqtt
from config import (
    MQTT_BROKER, MQTT_PORT, MQTT_TOPIC_PRODUCT_REQUEST, 
    MQTT_TOPIC_PRODUCT_RESPONSE
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductMQTTClient:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        
        # 设置客户端ID
        self.client_id = f"product_client_{int(time.time())}"
        self.client._client_id = self.client_id
        
        # 连接状态
        self.connected = False
        
        # 存储响应
        self.responses = {}
        
    def on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self.connected = True
            logger.info(f"成功连接到MQTT代理服务器 {MQTT_BROKER}:{MQTT_PORT}")
            
            # 订阅产品响应主题
            client.subscribe(MQTT_TOPIC_PRODUCT_RESPONSE)
            logger.info(f"已订阅主题: {MQTT_TOPIC_PRODUCT_RESPONSE}")
            
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
            
            if topic == MQTT_TOPIC_PRODUCT_RESPONSE:
                self.handle_product_response(payload)
                
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
            
    def handle_product_response(self, payload: str):
        """处理产品讲解响应"""
        try:
            response_data = json.loads(payload)
            request_id = response_data.get('request_id')
            
            # 存储响应
            self.responses[request_id] = response_data
            
            logger.info(f"收到产品讲解响应 - 请求ID: {request_id}")
            logger.info(f"产品名称: {response_data.get('product_name')}")
            logger.info(f"AI讲解: {response_data.get('ai_explanation', '')[:100]}...")
            
        except json.JSONDecodeError:
            logger.error("无效的JSON格式")
        except Exception as e:
            logger.error(f"处理产品响应时出错: {e}")
            
    def request_product_explanation(self, product_id: str) -> str:
        """请求产品讲解"""
        try:
            if not self.connected:
                logger.error("未连接到MQTT代理服务器")
                return None
                
            # 生成请求ID
            request_id = str(int(time.time()))
            
            # 构建请求
            request = {
                'product_id': product_id,
                'request_id': request_id,
                'timestamp': int(time.time())
            }
            
            # 发送请求
            payload = json.dumps(request, ensure_ascii=False)
            result = self.client.publish(MQTT_TOPIC_PRODUCT_REQUEST, payload)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"成功发送产品讲解请求 - 产品ID: {product_id}, 请求ID: {request_id}")
                return request_id
            else:
                logger.error(f"发送请求失败，返回码: {result.rc}")
                return None
                
        except Exception as e:
            logger.error(f"发送产品讲解请求时出错: {e}")
            return None
            
    def wait_for_response(self, request_id: str, timeout: int = 30) -> Dict[str, Any]:
        """等待响应"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id in self.responses:
                return self.responses[request_id]
            time.sleep(0.1)
            
        logger.warning(f"等待响应超时 - 请求ID: {request_id}")
        return None
        
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
        """启动MQTT客户端"""
        if not self.connect():
            return False
            
        try:
            # 启动网络循环
            self.client.loop_start()
            logger.info("MQTT客户端已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动MQTT客户端失败: {e}")
            return False
            
    def stop(self):
        """停止MQTT客户端"""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("MQTT客户端已停止")
            
        except Exception as e:
            logger.error(f"停止MQTT客户端时出错: {e}")
            
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.connected

def test_product_requests():
    """测试产品讲解请求"""
    client = ProductMQTTClient()
    
    try:
        if not client.start():
            logger.error("启动MQTT客户端失败")
            return
            
        # 等待连接建立
        time.sleep(2)
        
        if not client.is_connected():
            logger.error("MQTT客户端未连接")
            return
            
        # 测试产品ID列表
        test_products = ["001", "002", "003", "999"]  # 999是无效ID
        
        for product_id in test_products:
            logger.info(f"\n{'='*50}")
            logger.info(f"测试产品ID: {product_id}")
            
            # 发送请求
            request_id = client.request_product_explanation(product_id)
            
            if request_id:
                # 等待响应
                response = client.wait_for_response(request_id, timeout=10)
                
                if response:
                    logger.info(f"✅ 成功获取产品讲解")
                    logger.info(f"产品名称: {response.get('product_name')}")
                    logger.info(f"AI讲解: {response.get('ai_explanation', '')[:200]}...")
                else:
                    logger.warning("⚠️ 未收到响应")
            else:
                logger.error("❌ 发送请求失败")
                
            # 等待一下再测试下一个
            time.sleep(2)
            
    except KeyboardInterrupt:
        logger.info("收到停止信号")
    finally:
        client.stop()

def main():
    """主函数"""
    logger.info("启动MQTT客户端测试...")
    test_product_requests()

if __name__ == "__main__":
    main()
