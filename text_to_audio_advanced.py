from melo.api import TTS
import time
import torch
import gc
import os
import json
from pathlib import Path
from tts_config import TTSConfig
import numpy as np

class OptimizedTTS:
    """优化的TTS推理类"""
    
    def __init__(self, config=None):
        self.config = config or TTSConfig()
        self.model = None
        self.speaker_id = None
        self.cache_dir = Path('.tts_cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        # 应用优化设置
        self.config.apply_cuda_optimizations()
        
        # 性能统计
        self.stats = {
            'total_inferences': 0,
            'total_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def load_model(self):
        """加载模型（带缓存机制）"""
        if self.model is not None:
            return self.model
            
        print("正在加载TTS模型...")
        start_time = time.time()
        
        # 检查是否有缓存的模型
        cache_file = self.cache_dir / 'model_cache.pt'
        if cache_file.exists():
            try:
                print("尝试从缓存加载模型...")
                # 这里可以添加模型缓存逻辑
                pass
            except:
                print("缓存加载失败，重新加载模型...")
        
        self.model = TTS(language=self.config.INFERENCE_PARAMS['language'], 
                        device=self.config.DEVICE)
        
        load_time = time.time() - start_time
        print(f"模型加载耗时: {load_time:.2f}秒")
        
        # 获取说话人ID
        self.speaker_id = self.model.hps.data.spk2id['ZH']
        
        # 模型预热
        self._warmup_model()
        
        return self.model
    
    def _warmup_model(self):
        """模型预热"""
        print("正在进行模型预热...")
        warmup_text = self.config.INFERENCE_PARAMS['warmup_text']
        
        start_time = time.time()
        self.model.tts_to_file(warmup_text, self.speaker_id, 
                              str(self.cache_dir / 'warmup.wav'), 
                              speed=1.0)
        warmup_time = time.time() - start_time
        print(f"预热耗时: {warmup_time:.2f}秒")
        
        # 清理预热文件
        warmup_file = self.cache_dir / 'warmup.wav'
        if warmup_file.exists():
            warmup_file.unlink()
    
    def _get_cache_key(self, text, speed):
        """生成缓存键"""
        import hashlib
        cache_str = f"{text}_{speed}_{self.config.INFERENCE_PARAMS['language']}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _check_cache(self, text, speed):
        """检查缓存"""
        cache_key = self._get_cache_key(text, speed)
        cache_file = self.cache_dir / f"{cache_key}.wav"
        
        if cache_file.exists():
            self.stats['cache_hits'] += 1
            return str(cache_file)
        
        self.stats['cache_misses'] += 1
        return None
    
    def _save_to_cache(self, text, speed, output_path):
        """保存到缓存"""
        cache_key = self._get_cache_key(text, speed)
        cache_file = self.cache_dir / f"{cache_key}.wav"
        
        # 复制文件到缓存
        import shutil
        shutil.copy2(output_path, cache_file)
        
        # 保存元数据
        metadata = {
            'text': text,
            'speed': speed,
            'language': self.config.INFERENCE_PARAMS['language'],
            'timestamp': time.time()
        }
        
        meta_file = self.cache_dir / f"{cache_key}.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def tts_inference(self, text, output_path='output.wav', speed=None, use_cache=True):
        """优化的TTS推理"""
        if self.model is None:
            self.load_model()
        
        speed = speed or self.config.INFERENCE_PARAMS['speed']
        
        # 检查缓存
        if use_cache:
            cached_file = self._check_cache(text, speed)
            if cached_file:
                print(f"使用缓存结果: {cached_file}")
                # 复制缓存文件到目标位置
                import shutil
                shutil.copy2(cached_file, output_path)
                return output_path
        
        print(f"开始推理文本: {text[:30]}...")
        start_time = time.time()
        
        # 执行TTS推理
        self.model.tts_to_file(text, self.speaker_id, output_path, speed=speed)
        
        inference_time = time.time() - start_time
        print(f"推理耗时: {inference_time:.2f}秒")
        
        # 更新统计
        self.stats['total_inferences'] += 1
        self.stats['total_time'] += inference_time
        
        # 保存到缓存
        if use_cache:
            self._save_to_cache(text, speed, output_path)
        
        return output_path
    
    def batch_inference(self, texts, output_dir='outputs', speed=None):
        """批量推理优化"""
        if self.model is None:
            self.load_model()
        
        speed = speed or self.config.INFERENCE_PARAMS['speed']
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = []
        total_start_time = time.time()
        
        # 预处理：检查缓存
        cached_texts = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            output_path = output_dir / f'output_{i+1}.wav'
            cached_file = self._check_cache(text, speed)
            
            if cached_file:
                cached_texts.append((i, text, cached_file, output_path))
            else:
                uncached_texts.append((i, text, output_path))
        
        # 处理缓存的结果
        for i, text, cached_file, output_path in cached_texts:
            import shutil
            shutil.copy2(cached_file, output_path)
            results.append({
                'index': i,
                'text': text,
                'output_path': str(output_path),
                'inference_time': 0,
                'cached': True
            })
            print(f"文本 {i+1}: 使用缓存")
        
        # 处理未缓存的结果
        for i, text, output_path in uncached_texts:
            start_time = time.time()
            self.tts_inference(text, str(output_path), speed, use_cache=True)
            inference_time = time.time() - start_time
            
            results.append({
                'index': i,
                'text': text,
                'output_path': str(output_path),
                'inference_time': inference_time,
                'cached': False
            })
            print(f"文本 {i+1}: {inference_time:.2f}秒")
        
        total_time = time.time() - total_start_time
        print(f"批量处理完成，总耗时: {total_time:.2f}秒")
        
        return results
    
    def get_performance_stats(self):
        """获取性能统计"""
        if self.stats['total_inferences'] > 0:
            avg_time = self.stats['total_time'] / self.stats['total_inferences']
            cache_hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
        else:
            avg_time = 0
            cache_hit_rate = 0
        
        return {
            **self.stats,
            'average_inference_time': avg_time,
            'cache_hit_rate': cache_hit_rate,
            'device_info': self.config.get_device_info()
        }
    
    def clear_cache(self):
        """清理缓存"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            print("缓存已清理")
        
        # 重置统计
        self.stats = {
            'total_inferences': 0,
            'total_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

def main():
    """主函数"""
    # 创建优化TTS实例
    tts = OptimizedTTS()
    
    # 单次推理
    text = "你好我是小seeed,有什么问题可以问我。"
    output_file = tts.tts_inference(text, 'output_optimized.wav')
    
    # 批量推理示例
    texts = [
        "你好我是小seeed,有什么问题可以问我。",
        "今天天气真不错。",
        "人工智能技术发展很快。"
    ]
    
    batch_results = tts.batch_inference(texts)
    
    # 显示性能统计
    stats = tts.get_performance_stats()
    print("\n=== 性能统计 ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # 清理资源
    if tts.config.DEVICE == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
