# -*- coding: utf-8 -*-
"""
Sound Recognition using YAMNet on ReSpeaker channel 1
"""

import pyaudio
import numpy as np
import os
import time

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 6
RATE = 16000
DEVICE_NAME = "reSpeaker XVF3800"
DURATION = 0.96  # YAMNet expects ~1 second chunks
SAMPLES_NEEDED = int(DURATION * RATE)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

def load_yamnet():
    import tensorflow_hub as hub
    print("\n📦 Loading YAMNet model...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_names = []
    with open(model.class_map_path().numpy(), "r") as f:
        for line in f.readlines()[1:]:
            class_names.append(line.split(",")[2].strip())
    print(f"✓ Loaded {len(class_names)} sound classes")
    return model, class_names

def test_microphone_levels():
    device_index = find_reSpeaker()
    if device_index is None:
        return
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, input_device_index=device_index, frames_per_buffer=CHUNK)
    print("Reading audio levels for 3 seconds...")
    try:
        for _ in range(30):
            data = stream.read(CHUNK)
            audio = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)
            rms = np.sqrt(np.mean(audio[:, 1].astype(np.float32)**2))
            print(f"Level: {'█' * min(int(rms/500), 50)} ({rms:.1f})")
            time.sleep(0.1)
    finally:
        stream.stop_stream(); stream.close(); p.terminate()

def sound_recognition():
    device_index = find_reSpeaker()
    if device_index is None:
        print(f"✗ ReSpeaker '{DEVICE_NAME}' not found")
        return

    try:
        model, class_names = load_yamnet()
    except Exception as e:
        print(f"✗ Failed to load YAMNet: {e}")
        return

    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, input_device_index=device_index, frames_per_buffer=CHUNK)
        print(f"\n🔴 Listening... (Ctrl+C to stop)\n" + "-"*60)
        audio_buffer = []
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)
            channel_1 = audio[:, 1].astype(np.float32) / 32768.0
            audio_buffer.extend(channel_1)

            if len(audio_buffer) >= SAMPLES_NEEDED:
                import tensorflow as tf
                waveform = np.array(audio_buffer[:SAMPLES_NEEDED])
                audio_buffer = audio_buffer[SAMPLES_NEEDED:]
                scores, _, _ = model(waveform)
                mean_scores = tf.reduce_mean(scores, axis=0).numpy()
                top_indices = np.argsort(mean_scores)[-5:][::-1]

                print(f"\n[{time.strftime('%H:%M:%S')}] Top detected sounds:")
                has_sound = False
                for i in top_indices:
                    if mean_scores[i] > 0.1:
                        print(f"   • {class_names[i]:<30} {mean_scores[i]*100:>5.1f}%")
                        has_sound = True
                if not has_sound:
                    print("   🔇 No clear sounds detected (confidence < 10%)")
                print("-"*60)

    except KeyboardInterrupt:
        print("\n\n🔇 Stopped")
    finally:
        if stream:
            stream.stop_stream(); stream.close()
        p.terminate()

if __name__ == "__main__":
    print("="*60 + "\nReSpeaker Sound Recognition (YAMNet)\n" + "="*60)
    if input("\nTest mic levels first? (y/n): ").lower() == 'y':
        test_microphone_levels()
    input("\nPress Enter to start sound recognition...")
    sound_recognition()
