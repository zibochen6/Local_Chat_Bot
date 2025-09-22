import sys
import subprocess
import time
import os
import wave
import numpy as np
import signal
import tempfile
import threading
from collections import deque

# Signal handler for clean exit
def signal_handler(sig, frame):
    print('\n🛑 正在退出...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

import sherpa_ncnn


def create_recognizer():
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
    # for download links.
    print("🔄 正在加载语音识别模型...")
    recognizer = sherpa_ncnn.Recognizer(
        tokens="/home/seeed/Local_Chat_Bot/audio/sherpa-ncnn/models/tokens.txt",
        encoder_param="/home/seeed/Local_Chat_Bot/audio/sherpa-ncnn/models/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin="/home/seeed/Local_Chat_Bot/audio/sherpa-ncnn/models/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param="/home/seeed/Local_Chat_Bot/audio/sherpa-ncnn/models/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin="/home/seeed/Local_Chat_Bot/audio/sherpa-ncnn/models/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param="/home/seeed/Local_Chat_Bot/audio/sherpa-ncnn/models/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin="/home/seeed/Local_Chat_Bot/audio/sherpa-ncnn/models/joiner_jit_trace-pnnx.ncnn.bin",
        num_threads=4,
    )
    print("✅ 语音识别模型加载完成")
    return recognizer


class SmartVoiceBot:
    def __init__(self):
        self.recognizer = create_recognizer()
        self.wake_word = "你好"
        self.is_listening = False
        self.conversation_buffer = deque(maxlen=20)  # 保存最近20段音频
        self.silence_count = 0
        self.max_silence = 3  # 3次静默后结束对话 (约3秒)
        self.min_conversation_length = 1  # 至少1个音频段就可以考虑结束
        self.audio_buffer = []  # 音频缓冲区，用于累积音频
        self.has_speech = False  # 标记是否检测到语音
        self.last_speech_time = 0  # 最后一次检测到语音的时间
        self.has_meaningful_content = False  # 是否已经获得有意义的内容
        
    def record_chunk(self, duration=1):
        """录制音频片段"""
        temp_file = f"/tmp/voice_chunk_{int(time.time() * 1000)}.wav"
        
        try:
            cmd = [
                "arecord", 
                "-D", "sysdefault",
                "-f", "S16_LE", 
                "-r", "16000",
                "-c", "1",
                "-d", str(duration),
                temp_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 1)
            
            if result.returncode != 0:
                return None, 0
                
            # Load audio
            with wave.open(temp_file, 'rb') as f:
                frames = f.readframes(f.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            audio_level = np.abs(audio_data).max()
            return audio_data, audio_level
            
        except Exception as e:
            return None, 0
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def recognize_audio(self, audio_data):
        """识别音频数据 - 优化版本"""
        try:
            # 音频预处理
            if len(audio_data) < 16000:  # 少于1秒的音频，识别效果可能不好
                return ""
            
            # 检查音频电平
            audio_max = np.abs(audio_data).max()
            if audio_max < 0.01:  # 音频太小，可能是噪音
                return ""
            
            # 音频归一化 - 更保守的归一化
            if audio_max > 0:
                # 归一化到合理范围，避免过度放大
                if audio_max < 0.1:
                    audio_data = audio_data / audio_max * 0.1
                elif audio_max > 0.8:
                    audio_data = audio_data / audio_max * 0.8
                # 如果在合理范围内，保持原样
            
            # Create fresh recognizer
            model_dir = "/home/seeed/Local_Chat_Bot/audio/sherpa-ncnn/models"
            recognizer = sherpa_ncnn.Recognizer(
                tokens=f"{model_dir}/tokens.txt",
                encoder_param=f"{model_dir}/encoder_jit_trace-pnnx.ncnn.param",
                encoder_bin=f"{model_dir}/encoder_jit_trace-pnnx.ncnn.bin",
                decoder_param=f"{model_dir}/decoder_jit_trace-pnnx.ncnn.param",
                decoder_bin=f"{model_dir}/decoder_jit_trace-pnnx.ncnn.bin",
                joiner_param=f"{model_dir}/joiner_jit_trace-pnnx.ncnn.param",
                joiner_bin=f"{model_dir}/joiner_jit_trace-pnnx.ncnn.bin",
                num_threads=4,
            )
            
            # 直接处理完整音频，不分段
            recognizer.accept_waveform(16000, audio_data)
            
            # Add silence to finalize
            silence = np.zeros(int(0.8 * 16000), dtype=np.float32)
            recognizer.accept_waveform(16000, silence)
            recognizer.input_finished()
            
            result = recognizer.text.strip()
            
            # 后处理：去除明显的识别错误
            if len(result) < 1:  # 空结果
                return ""
                
            return result
            
        except Exception as e:
            print(f"识别错误: {e}")
            return ""
    
    def detect_wake_word(self, text):
        """检测唤醒词"""
        return self.wake_word in text
    
    def is_conversation_end(self):
        """智能判断对话是否结束"""
        current_time = time.time()
        
        # 如果已经有有意义的内容，使用更短的等待时间
        if self.has_meaningful_content:
            # 有内容后，2秒静默就结束
            if self.silence_count >= 2:
                print(f"🎯 已获得完整内容，{self.silence_count}秒静默后结束对话")
                return True
        else:
            # 没有内容时，等待稍长一些
            if self.silence_count >= self.max_silence:
                print(f"🔇 检测到连续{self.silence_count}次静默，准备结束对话...")
                return True
            
        # 检查结束词
        if len(self.conversation_buffer) > 0:
            recent_text = " ".join([item["text"] for item in list(self.conversation_buffer)[-2:]])
            end_phrases = ["谢谢", "再见", "好的", "知道了", "明白了", "完毕", "结束", "没了", "就这样"]
            
            for phrase in end_phrases:
                if phrase in recent_text:
                    print(f"🔚 检测到结束词 '{phrase}'")
                    return True
        
        # 如果超过15秒没有新的有效语音，强制结束
        if self.last_speech_time > 0 and (current_time - self.last_speech_time) > 15:
            print("⏰ 超时强制结束对话")
            return True
                
        return False
    
    def continuous_listen(self):
        """连续监听模式"""
        print("🔄 监听唤醒词中...")
        
        while True:
            audio_data, audio_level = self.record_chunk(duration=1)
            
            if audio_data is not None and audio_level > 0.02:
                text = self.recognize_audio(audio_data)
                
                if text and self.detect_wake_word(text):
                    print(f"🎯 检测到唤醒词: '{text}'")
                    print("👂 开始倾听您的问题...")
                    self.start_conversation()
                    print("🔄 重新监听唤醒词中...")
            
            time.sleep(0.1)
    
    def start_conversation(self):
        """开始对话模式 - 智能优化版本"""
        self.is_listening = True
        self.conversation_buffer.clear()
        self.silence_count = 0
        self.audio_buffer = []
        self.has_speech = False
        self.last_speech_time = 0
        self.has_meaningful_content = False
        
        full_conversation = []
        continuous_audio = []  # 连续音频片段
        speech_detected = False
        
        print("💬 对话开始 (智能快速结束)")
        print("-" * 40)
        
        while self.is_listening:
            audio_data, audio_level = self.record_chunk(duration=1)
            
            if audio_data is not None:
                if audio_level > 0.02:  # 有声音 (提高阈值，减少噪音干扰)
                    continuous_audio.append(audio_data)
                    speech_detected = True
                    self.silence_count = 0  # 重置静默计数
                    self.last_speech_time = time.time()  # 更新最后语音时间
                    print("🎵", end="", flush=True)  # 显示录音进度
                    
                else:
                    # 静默
                    if speech_detected and len(continuous_audio) > 1:  # 需要至少2段音频
                        # 有语音后的静默，处理累积的音频
                        combined_audio = np.concatenate(continuous_audio)
                        
                        # 显示音频信息用于调试
                        audio_level = np.abs(combined_audio).max()
                        print(f"\n📊 处理音频: {len(combined_audio)} 采样点 ({len(combined_audio)/16000:.1f}秒), 电平:{audio_level:.3f}")
                        
                        text = self.recognize_audio(combined_audio)
                        
                        if text:
                            print(f"🗣️  '{text}'")
                            full_conversation.append(text)
                            
                            # 检查是否是有意义的内容
                            if self.is_meaningful_content(text):
                                self.has_meaningful_content = True
                                print("✨ 检测到完整问题")
                            
                            self.conversation_buffer.append({
                                "text": text,
                                "level": audio_level,
                                "timestamp": time.time()
                            })
                        else:
                            print("🔇 此段音频未识别到文本")
                        
                        # 清空音频缓冲
                        continuous_audio = []
                        speech_detected = False
                    
                    self.silence_count += 1
                    print(".", end="", flush=True)  # 显示静默进度
                    
                # 检查是否应该结束对话
                if self.is_conversation_end():
                    # 处理剩余的音频
                    if len(continuous_audio) > 0:
                        combined_audio = np.concatenate(continuous_audio)
                        text = self.recognize_audio(combined_audio)
                        if text:
                            print(f"\n🗣️  '{text}'")
                            full_conversation.append(text)
                    
                    self.end_conversation(full_conversation)
                    break
            
            time.sleep(0.1)
    
    def is_meaningful_content(self, text):
        """判断是否是有意义的完整内容"""
        # 检查是否包含疑问词，可能是完整问题
        question_words = ["什么", "怎么", "为什么", "哪里", "哪个", "几", "多少", "谁", "吗", "呢", "?", "？"]
        for word in question_words:
            if word in text:
                return True
        
        # 检查是否是完整的短语或句子 (长度大于2个字符)
        if len(text.strip()) >= 3:
            return True
            
        return False
    
    def end_conversation(self, full_conversation):
        """结束对话"""
        self.is_listening = False
        
        print("\n" + "-" * 40)
        print("🔚 对话结束")
        
        if full_conversation:
            complete_question = " ".join(full_conversation)
            print(f"📝 完整问题: '{complete_question}'")
            print(f"💭 这里可以调用AI模型回答问题...")
            
            # 这里可以添加AI回答逻辑
            # response = call_ai_model(complete_question)
            # print(f"🤖 AI回答: {response}")
        else:
            print("😅 没有检测到完整问题")
        
        print("🔄 返回待机状态，等待唤醒词...\n")

def main():
    print("🤖 智能语音对话机器人")
    print("💡 唤醒词: '你好'")
    print("🎯 功能: 智能检测对话开始和结束")
    
    # Create recognizer to test
    print("📱 正在加载语音识别模型...")
    test_recognizer = create_recognizer()
    print("✅ 模型加载完成")
    
    # Initialize voice bot
    bot = SmartVoiceBot()
    
    print("\n🎙️ 开始监听... (按 Ctrl+C 退出)")
    print("💬 请说'你好'来唤醒机器人")
    print("=" * 50)
    
    try:
        bot.continuous_listen()
    except KeyboardInterrupt:
        print("\n🛑 用户退出")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 程序退出")
    except Exception as e:
        print(f"❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()