import vosk, sys, sounddevice as sd, queue, json

model = vosk.Model("model-zh-en")  # 下载中英文混合模型
q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(bytes(indata))

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    rec = vosk.KaldiRecognizer(model, 16000)
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            print(json.loads(rec.Result())['text'])
