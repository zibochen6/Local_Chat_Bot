# 配置文件
import os
from dotenv import load_dotenv

load_dotenv()

# MQTT配置
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_PRODUCT_REQUEST = "product/request"
MQTT_TOPIC_PRODUCT_RESPONSE = "product/response"

# Flask配置
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

JETSON_IP = "192.168.6.236"

# 产品数据库配置
PRODUCTS_DB = {
    "001": {
        "name": "reComputer J40x",
        "name_en": "reComputer J40x",
        "description": "基于NVIDIA Jetson Orin NX的高性能边缘AI计算设备，专为AI和边缘计算应用而设计。",
        "description_en": "High-performance edge AI computing device based on NVIDIA Jetson Orin NX, specifically designed for AI and edge computing applications.",
        "features": ["AI性能70-100 TOPS", "8-16GB LPDDR5内存", "1024核Ampere GPU", "2x NVDLA v2加速器", "4K视频编解码"],
        "features_en": ["AI Performance 70-100 TOPS", "8-16GB LPDDR5 Memory", "1024-core Ampere GPU", "2x NVDLA v2 Accelerators", "4K Video Codec"],
        "specs": {
            "ai_performance": "70-100 TOPS",
            "gpu": "1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores",
            "cpu": "8-core Arm Cortex-A78AE v8.2 64-bit CPU",
            "memory": "8-16GB 128-bit LPDDR5 102.4GB/s",
            "storage": "128GB NVMe SSD",
            "video_encoder": "12x 1080p30 (H.265)",
            "video_decoder": "4x 4K30 (H.265) | 9x 1080p60 (H.265)",
            "interfaces": "4x USB 3.2, HDMI 2.1, 2x CSI, Gigabit Ethernet, M.2 Key E/M, CAN, GPIO"
        }
    },
    "002": {
        "name": "智能手表",
        "description": "这是一款智能手表产品，具有以下特点：\n1. 健康监测功能\n2. 运动追踪\n3. 消息通知\n4. 防水设计\n5. 长续航电池",
        "features": ["健康监测", "运动追踪", "消息通知", "防水设计", "长续航"]
    },
    "003": {
        "name": "智能摄像头",
        "description": "这是一款智能摄像头产品，具有以下特点：\n1. 高清视频录制\n2. 夜视功能\n3. 移动侦测\n4. 云存储支持\n5. 双向通话",
        "features": ["高清录制", "夜视功能", "移动侦测", "云存储", "双向通话"]
    }
}

# 默认产品
DEFAULT_PRODUCT = {
    "name": "未知产品",
    "description": "抱歉，未找到该产品的信息。请检查产品ID是否正确。",
    "features": []
}
