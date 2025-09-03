from melo.api import TTS
import time
import torch
import gc

def optimize_tts_inference():
    # 性能优化设置
    torch.backends.cudnn.benchmark = True  # 启用 cuDNN 自动调优
    torch.backends.cudnn.deterministic = False  # 关闭确定性模式以提高速度
    
    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        # 设置 CUDA 内存分配策略
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)  # 使用90%的GPU内存
    
    # 模型初始化（只初始化一次）
    print("正在加载TTS模型...")
    start_time = time.time()
    model = TTS(language='ZH', device=device)
    load_time = time.time() - start_time
    print(f"模型加载耗时: {load_time:.2f}秒")
    
    # 获取说话人ID
    speaker_ids = model.hps.data.spk2id
    speaker_id = speaker_ids['ZH']
    
    # 模型预热（第一次推理通常较慢）
    print("正在进行模型预热...")
    warmup_text = "你好"
    start_time = time.time()
    model.tts_to_file(warmup_text, speaker_id, 'warmup.wav', speed=1.0)
    warmup_time = time.time() - start_time
    print(f"预热耗时: {warmup_time:.2f}秒")
    
    # 清理预热文件
    import os
    if os.path.exists('warmup.wav'):
        os.remove('warmup.wav')
    
    # 主要文本处理
    text = "你好我是小seeed,有什么问题可以问我。"
    
    # 优化推理参数
    speed = 1.5  # 提高速度到1.5倍
    output_path = 'output.wav'
    
    print("开始主要推理...")
    start_time = time.time()
    
    # 执行TTS推理
    model.tts_to_file(text, speaker_id, output_path, speed=speed)
    
    inference_time = time.time() - start_time
    print(f"推理耗时: {inference_time:.2f}秒")
    
    # 性能统计
    total_time = load_time + warmup_time + inference_time
    print(f"总耗时: {total_time:.2f}秒")
    print(f"音频已保存到: {output_path}")
    
    # 清理GPU内存
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    return output_path

def batch_tts_inference(texts, output_dir='outputs'):
    """批量处理多个文本以提高效率"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化模型（只初始化一次）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TTS(language='ZH', device=device)
    speaker_ids = model.hps.data.spk2id
    speaker_id = speaker_ids['ZH']
    
    # 预热
    model.tts_to_file("你好", speaker_id, 'temp.wav', speed=1.0)
    if os.path.exists('temp.wav'):
        os.remove('temp.wav')
    
    results = []
    for i, text in enumerate(texts):
        output_path = os.path.join(output_dir, f'output_{i+1}.wav')
        start_time = time.time()
        model.tts_to_file(text, speaker_id, output_path, speed=1.5)
        inference_time = time.time() - start_time
        results.append({
            'text': text,
            'output_path': output_path,
            'inference_time': inference_time
        })
        print(f"文本 {i+1}: {inference_time:.2f}秒")
    
    return results

if __name__ == "__main__":
    # 单次推理
    output_file = optimize_tts_inference()
    
    # 批量推理示例（可选）
    # texts = [
    #     "你好我是小seeed,有什么问题可以问我。",
    #     "今天天气真不错。",
    #     "人工智能技术发展很快。"
    # ]
    # batch_results = batch_tts_inference(texts)