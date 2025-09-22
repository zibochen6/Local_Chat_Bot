# Kokoro TTS 中英文混合语音合成

基于 [Kokoro](https://github.com/hexgrad/kokoro) 的轻量级 TTS 引擎，支持中英文混合语音合成。

## 特性

- 🎯 **轻量级**: 仅 82M 参数，CPU 上可近乎实时运行
- 🌍 **多语言支持**: 支持中文、英文、日文等 8 种语言
- 🎵 **多音色**: 提供多种男女人物音色
- 🔄 **中英混合**: 完美支持中英文混合文本合成，使用 `en_callable` 回调函数处理英文部分
- ⚡ **高性能**: GPU 上可达 50 倍实时速度
- 🎤 **智能音素**: 自动识别中英文并应用相应的音素处理

## 安装依赖

```bash
pip install kokoro
pip install ordered-set
pip install cn2an
pip install pypinyin_dict
pip install pygame
pip install soundfile
pip install torch
```

## 下载模型

运行下载脚本：

```bash
./download_kokoro_models.sh
```

或手动下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download hexgrad/Kokoro-82M-v1.1-zh --local-dir ./ckpts/kokoro-v1.1
```

## 使用方法

```bash
python tts/test.py
```

## 中英文混合处理

代码实现了 `en_callable` 回调函数来处理中英文混合文本：

- **英文管道**: 使用 `lang_code='a'` 处理纯英文文本
- **中文管道**: 使用 `lang_code='z'` 和 `en_callable` 参数处理中英混合文本
- **音素映射**: 为常见英文词汇提供精确的音素表示
- **自动回退**: 对于未映射的英文词汇，自动使用英文管道生成音素

## 测试文本

代码包含以下中英文混合测试文本：

1. "这是一个中文语音合成测试，测试英文输出：reComputer、Jetson。"
2. "Hello world! 你好世界！这是一个中英文混合的语音合成测试。"
3. "欢迎使用Kokoro TTS引擎，它支持中英文混合语音合成。Welcome to Kokoro TTS!"
4. "人工智能技术发展迅速，AI technology is advancing rapidly。"
5. "今天天气很好，It's a beautiful day today。"
6. "Kokoro是一个优秀的TTS引擎，Sol是另一个选择。"
7. "Hello，欢迎使用我们的AI语音合成系统！"

## 音色选择

代码会自动检测可用的音色文件：
- `zf_001`: 中文女声
- `af_heart`: 英文女声

## 参考链接

- [Kokoro GitHub](https://github.com/hexgrad/kokoro)
- [CSDN 博客教程](https://blog.csdn.net/u010522887/article/details/146720024)
- [Hugging Face 模型](https://hf-mirror.com/hexgrad/Kokoro-82M-v1.1-zh)
