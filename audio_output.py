from TTS.api import TTS
import numpy as np
from scipy.io.wavfile import write
import torch
import torch.serialization

# 添加安全全局变量以解决PyTorch 2.6兼容性问题
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig
    from TTS.config.shared_configs import BaseDatasetConfig
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig])
except ImportError:
    pass

# 选择支持中英文混合的多语言模型
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# 输入需要合成的文本 - 中英文混合
text = "你好，Jetson! Let's build a fast real-time speech system. 这是一个非常棒的语音合成系统！"

# 合成语音，输出为numpy数组
# 使用中文作为主要语言，但模型会自动处理中英文混合
audio = tts.tts(text=text, speaker="female", language="zh")

# 保存为 WAV 文件
rate = 24000  # XTTS 默认采样率
write("output.wav", rate, (np.array(audio) * 32767).astype(np.int16))

print("音频已保存为 output.wav")
print(f"文本长度: {len(text)} 字符")
print(f"音频长度: {len(audio)/rate:.2f} 秒")
