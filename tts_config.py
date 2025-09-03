# TTS性能优化配置文件
import torch

class TTSConfig:
    """TTS推理性能优化配置"""
    
    # 设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # CUDA优化设置
    CUDA_OPTIMIZATIONS = {
        'benchmark': True,           # 启用cuDNN自动调优
        'deterministic': False,      # 关闭确定性模式提高速度
        'memory_fraction': 0.9,      # GPU内存使用比例
        'empty_cache': True,         # 是否清理GPU缓存
    }
    
    # 推理参数
    INFERENCE_PARAMS = {
        'speed': 1.5,               # 推理速度倍数
        'language': 'ZH',           # 语言设置
        'warmup_text': '你好',      # 预热文本
    }
    
    # 批处理设置
    BATCH_SETTINGS = {
        'batch_size': 1,            # 批处理大小
        'output_dir': 'outputs',    # 输出目录
    }
    
    # 性能监控
    PERFORMANCE_MONITORING = {
        'enable_timing': True,      # 启用时间统计
        'enable_memory_monitoring': True,  # 启用内存监控
        'log_performance': True,    # 记录性能日志
    }
    
    @classmethod
    def apply_cuda_optimizations(cls):
        """应用CUDA优化设置"""
        if cls.DEVICE == 'cuda':
            torch.backends.cudnn.benchmark = cls.CUDA_OPTIMIZATIONS['benchmark']
            torch.backends.cudnn.deterministic = cls.CUDA_OPTIMIZATIONS['deterministic']
            
            if cls.CUDA_OPTIMIZATIONS['empty_cache']:
                torch.cuda.empty_cache()
            
            if cls.CUDA_OPTIMIZATIONS['memory_fraction'] < 1.0:
                torch.cuda.set_per_process_memory_fraction(
                    cls.CUDA_OPTIMIZATIONS['memory_fraction']
                )
    
    @classmethod
    def get_device_info(cls):
        """获取设备信息"""
        if cls.DEVICE == 'cuda':
            return {
                'device': cls.DEVICE,
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'cuda_version': torch.version.cuda
            }
        else:
            return {
                'device': cls.DEVICE,
                'cpu_count': torch.get_num_threads()
            }
