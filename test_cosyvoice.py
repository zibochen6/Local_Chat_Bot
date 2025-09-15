#!/usr/bin/env python3
import sys
sys.path.append('/home/seeed/Local_Chat_Bot/audio/CosyVoice/third_party/Matcha-TTS')
sys.path.append('/home/seeed/Local_Chat_Bot/audio/CosyVoice')

print("Testing imports...")
try:
    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
    print("✓ CosyVoice imports successful")
except Exception as e:
    print(f"✗ CosyVoice import failed: {e}")
    sys.exit(1)

try:
    from cosyvoice.utils.file_utils import load_wav
    print("✓ load_wav import successful")
except Exception as e:
    print(f"✗ load_wav import failed: {e}")
    sys.exit(1)

print("Testing model initialization...")
try:
    cosyvoice = CosyVoice2('/home/seeed/Local_Chat_Bot/audio/CosyVoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    print("✓ CosyVoice2 initialization successful")
except Exception as e:
    print(f"✗ CosyVoice2 initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("All tests passed!")
