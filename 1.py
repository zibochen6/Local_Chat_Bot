import numpy as np, sounddevice as sd
from TTS.api import TTS

# 1) 加载 XTTS v2，多语言多说话人
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# 2) 中英混读文本
text = "Hello，欢迎使用 Seeed 语音助手。Let's get started！"

# 3) 可选：提供 3~10 秒 speaker_wav 以固定音色；若没有也能发声
speaker_wav = None  # 例如 "/home/seeed/myvoice.wav"

# 4) 推理（语言自动感知，显式给 'zh' 效果更稳）
audio = tts.tts(
    text=text,
    speaker_wav=speaker_wav,  # 或者传 None 用默认/随机音色
    language="zh"             # 混读场景指定中文更稳，英文会自动读出
)

# 5) 播放
sd.play(np.array(audio), samplerate=24000)
sd.wait()
