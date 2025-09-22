
from kokoro import KPipeline, KModel
import pygame
import numpy as np
import torch
import time
import soundfile as sf
import os

# 检查GPU可用性
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name()}")

# 初始化pygame音频系统
pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=1024)
print("✓ pygame音频系统初始化完成")

# 设置模型路径
repo_id = 'hexgrad/Kokoro-82M-v1.1-zh'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 检查模型文件是否存在
model_path = 'ckpts/kokoro-v1.1/kokoro-v1_1-zh.pth'
config_path = 'ckpts/kokoro-v1.1/config.json'

if not os.path.exists(model_path):
    print(f"❌ 模型文件不存在: {model_path}")
    print("请先下载模型文件:")
    print("export HF_ENDPOINT=https://hf-mirror.com")
    print("huggingface-cli download --resume-download hexgrad/Kokoro-82M-v1.1-zh --local-dir ./ckpts/kokoro-v1.1")
    exit(1)

# 加载音色文件
voice_zf = "zf_001"
voice_af = 'af_heart'
voice_zf_path = f'ckpts/kokoro-v1.1/voices/{voice_zf}.pt'
voice_af_path = f'ckpts/kokoro-v1.1/voices/{voice_af}.pt'

if os.path.exists(voice_zf_path):
    voice_zf_tensor = torch.load(voice_zf_path, weights_only=True)
    print(f"✓ 加载音色: {voice_zf}")
else:
    print(f"❌ 音色文件不存在: {voice_zf_path}")
    voice_zf_tensor = None

if os.path.exists(voice_af_path):
    voice_af_tensor = torch.load(voice_af_path, weights_only=True)
    print(f"✓ 加载音色: {voice_af}")
else:
    print(f"❌ 音色文件不存在: {voice_af_path}")
    voice_af_tensor = None

# 加载模型
print("正在加载Kokoro模型...")
model = KModel(model=model_path, config=config_path, repo_id=repo_id).to(device).eval()
print("✓ Kokoro模型加载完成")

# 初始化英文管道
print("初始化英文管道...")
en_pipeline = KPipeline(lang_code='a', repo_id=repo_id, model=model)

# 定义英文回调函数，用于处理中英混杂文本中的英文部分
def en_callable(text):
    """
    英文回调函数，处理中英混杂文本中的英文部分
    返回英文文本的音素表示
    """
    print(f"    处理英文文本: '{text}'")
    
    # 特殊词汇的音素映射
    if text == 'Kokoro':
        return 'kˈOkəɹO'
    elif text == 'Sol':
        return 'sˈOl'
    elif text == 'reComputer':
        return 'riːkəmˈpjuːtər'
    elif text == 'Jetson':
        return 'ˈdʒɛtsən'
    elif text == 'Hello':
        return 'həˈloʊ'
    elif text == 'world':
        return 'wɜːrld'
    elif text == 'Welcome':
        return 'ˈwelkəm'
    elif text == 'to':
        return 'tuː'
    elif text == 'TTS':
        return 'tiːtiːˈes'
    elif text == 'AI':
        return 'eɪˈaɪ'
    elif text == 'technology':
        return 'tekˈnɑːlədʒi'
    elif text == 'is':
        return 'ɪz'
    elif text == 'advancing':
        return 'ədˈvænsɪŋ'
    elif text == 'rapidly':
        return 'ˈræpɪdli'
    elif text == 'It\'s':
        return 'ɪts'
    elif text == 'a':
        return 'ə'
    elif text == 'beautiful':
        return 'ˈbjuːtɪfəl'
    elif text == 'day':
        return 'deɪ'
    elif text == 'today':
        return 'təˈdeɪ'
    
    # 对于其他英文词汇，使用英文管道生成音素
    try:
        result = next(en_pipeline(text, voice=voice_af_tensor if voice_af_tensor is not None else voice_zf_tensor))
        return result.phonemes
    except Exception as e:
        print(f"    警告: 无法处理英文文本 '{text}': {e}")
        # 返回原始文本作为fallback
        return text

# 初始化中文管道，使用en_callable处理英文部分
print("初始化中文管道（支持中英混合）...")
pipeline = KPipeline(lang_code='z', repo_id=repo_id, model=model, en_callable=en_callable)




print("✓ Kokoro TTS管道初始化完成")

# 中英文混合测试文本
texts = [
    "这是一个中文语音合成测试，测试英文输出：reComputer、Jetson。",
    "Hello world! 你好世界！这是一个中英文混合的语音合成测试。",
    "欢迎使用Kokoro TTS引擎，它支持中英文混合语音合成。Welcome to Kokoro TTS!",
    "人工智能技术发展迅速，AI technology is advancing rapidly。",
    "今天天气很好，It's a beautiful day today。",
    "Kokoro是一个优秀的TTS引擎，Sol是另一个选择。",
    "Hello，欢迎使用我们的AI语音合成系统！"
]

# 选择要使用的音色
selected_voice = voice_zf_tensor if voice_zf_tensor is not None else voice_af_tensor
voice_name = voice_zf if voice_zf_tensor is not None else voice_af

if selected_voice is None:
    print("❌ 没有可用的音色文件，使用默认音色")
    selected_voice = 'af_heart'
    voice_name = 'af_heart'

print(f"使用音色: {voice_name}")
print(f"英文回调函数已配置，支持中英混合文本处理")

# 测试每个文本
for i, text in enumerate(texts):
    print(f"\n{'='*50}")
    print(f"测试文本 {i+1}: {text}")
    print(f"{'='*50}")
    
    print("正在生成语音（中英混合处理）...")
    start_time = time.time()
    generator = pipeline(text, voice=selected_voice)

    # 处理音频生成
    for j, (gs, ps, audio) in enumerate(generator):
        segment_start_time = time.time()
        print(f"  片段 {j}: 生成状态={gs}, 处理状态={ps}")
        
        # 转换音频格式为pygame可播放的格式
        if isinstance(audio, (np.ndarray, torch.Tensor)):
            # 如果是torch.Tensor，转换为numpy数组
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
                print(f"    从torch.Tensor转换为numpy数组")
            
            print(f"    音频形状: {audio.shape}, 数据类型: {audio.dtype}")
            print(f"    音频范围: [{np.min(audio):.4f}, {np.max(audio):.4f}]")
            
            # 确保音频数据是float32格式，范围在[-1, 1]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # 归一化音频
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val
                print(f"    音频已归一化，最大值为: {max_val:.4f}")
            
            # 转换为int16格式
            audio_int16 = (audio * 32767).astype(np.int16)
            print(f"    转换后音频范围: [{np.min(audio_int16)}, {np.max(audio_int16)}]")
            
            # 创建pygame Sound对象并播放
            try:
                sound = pygame.sndarray.make_sound(audio_int16)
                sound.play()
                print(f"    开始播放片段 {j}...")
                
                # 等待当前片段播放完成
                while pygame.mixer.get_busy():
                    pygame.time.wait(10)
                
                segment_end_time = time.time()
                segment_duration = segment_end_time - segment_start_time
                print(f"    ✓ 片段 {j} 播放完成 (用时: {segment_duration:.2f}秒)")
            except Exception as e:
                print(f"    ❌ 播放片段 {j} 时出错: {e}")
        else:
            print(f"    ❌ 片段 {j} 音频数据格式错误: {type(audio)}")
    
    text_time = time.time() - start_time
    print(f"✓ 文本 {i+1} 处理完成 (总用时: {text_time:.2f}秒)")
    
    # 在文本之间添加短暂停顿
    if i < len(texts) - 1:
        print("⏸️  文本间停顿...")
        pygame.time.wait(1000)  # 1秒停顿

print(f"\n🎉 所有中英文混合语音播放完成！")
print(f"📊 共处理了 {len(texts)} 个测试文本")
