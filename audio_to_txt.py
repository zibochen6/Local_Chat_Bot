import wave

import numpy as np

import sys
sys.path.append('/home/seeed/miniforge3/envs/chat/lib/python3.10/site-packages/sherpa_ncnn/lib')

import sherpa_ncnn


def main():
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

    filename = (
        "/home/seeed/Local_Chat_Bot/audio/sherpa-ncnn/models/test_wavs/1.wav"
    )
    with wave.open(filename) as f:
        assert f.getframerate() == recognizer.sample_rate, (
            f.getframerate(),
            recognizer.sample_rate,
        )
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768

    recognizer.accept_waveform(recognizer.sample_rate, samples_float32)

    tail_paddings = np.zeros(
        int(recognizer.sample_rate * 0.5), dtype=np.float32
    )
    recognizer.accept_waveform(recognizer.sample_rate, tail_paddings)

    recognizer.input_finished()

    print(recognizer.text)


if __name__ == "__main__":
    main()