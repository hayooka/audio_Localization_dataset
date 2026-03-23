# -*- coding: utf-8 -*-
"""
Speech-to-Text using Vosk on ReSpeaker channel 1
"""

import pyaudio
import numpy as np
import json
import os
from vosk import Model, KaldiRecognizer

MODEL_PATH  = r"C:\Users\farah\Downloads\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"
CHUNK       = 1024
FORMAT      = pyaudio.paInt16
CHANNELS    = 6
RATE        = 16000
DEVICE_NAME = "reSpeaker XVF3800"

def find_reSpeaker():
    p = pyaudio.PyAudio()
    device_index = None
    print("Searching for ReSpeaker device...")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if DEVICE_NAME in info['name']:
            device_index = i
            print(f"✓ Found ReSpeaker at index {device_index}: {info['name']}")
            break
    p.terminate()
    return device_index

def speech_to_text():
    device_index = find_reSpeaker()
    if device_index is None:
        print(f"✗ ReSpeaker '{DEVICE_NAME}' not found")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"✗ Vosk model not found at: {MODEL_PATH}")
        return

    print(f"\n📦 Loading Vosk model...")
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, RATE)
    print("✓ Vosk model loaded")

    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, input_device_index=device_index, frames_per_buffer=CHUNK)
        print("\n🔴 Listening... Speak now! (Ctrl+C to stop)\n" + "-"*50)

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)
            audio_bytes = audio[:, 2].tobytes()  # channel 1

            if rec.AcceptWaveform(audio_bytes):
                text = json.loads(rec.Result()).get("text", "")
                if text:
                    print(f"🗣 {text}")

    except KeyboardInterrupt:
        print("\n\n🔇 Stopped")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        if 'stream' in locals():
            stream.stop_stream(); stream.close()
        p.terminate()

if __name__ == "__main__":
    print("="*50 + "\nReSpeaker Speech-to-Text (Vosk)\n" + "="*50)
    speech_to_text()
