# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 15:54:10 2026

@author: farah
"""

import pyaudio
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import time

# ================= CONFIGURATION =================
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 6
RATE = 16000
DEVICE_NAME = "reSpeaker XVF3800"
DURATION = 0.96  # YAMNet expects ~1 second chunks (96ms * 10 = 960ms)
SAMPLES_NEEDED = int(DURATION * RATE)  # 15360 samples for 0.96 seconds at 16kHz

# Disable GPU (optional - use CPU only)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def find_reSpeaker():
    """Find the ReSpeaker device index"""
    p = pyaudio.PyAudio()
    device_index = None
   
    print("Searching for ReSpeaker device...")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if DEVICE_NAME in info['name']:
            device_index = i
            print(f"✅ Found ReSpeaker at index {device_index}")
            print(f"   Device name: {info['name']}")
            print(f"   Max input channels: {int(info['maxInputChannels'])}")
            print(f"   Default sample rate: {int(info['defaultSampleRate'])}")
            break
   
    p.terminate()
    return device_index

def load_yamnet():
    """Load YAMNet model and class names"""
    print("\n🔊 Loading YAMNet model from TensorFlow Hub...")
   
    # Load model
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("✅ YAMNet model loaded successfully!")
   
    # Load class names
    class_map_path = model.class_map_path().numpy()
    class_names = []
    with open(class_map_path, "r") as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            class_names.append(line.split(",")[2].strip())
   
    print(f"✅ Loaded {len(class_names)} sound classes")
    return model, class_names

def sound_recognition():
    """Main function for sound recognition using ReSpeaker channel 1"""
   
    # Find ReSpeaker device
    device_index = find_reSpeaker()
    if device_index is None:
        print(f"❌ ReSpeaker device '{DEVICE_NAME}' not found")
        return
   
    # Load YAMNet
    try:
        model, class_names = load_yamnet()
    except Exception as e:
        print(f"❌ Failed to load YAMNet: {e}")
        print("   Make sure tensorflow and tensorflow-hub are installed:")
        print("   pip install tensorflow tensorflow-hub")
        return
   
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = None
   
    try:
        # Open stream to ReSpeaker
        print(f"\n🔊 Opening ReSpeaker stream (Channel 1 for sound recognition)...")
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
       
        print(f"\n🔴 Listening for sounds... (Press Ctrl+C to stop)")
        print(f"   Processing {DURATION}s chunks every {DURATION}s")
        print("-" * 60)
       
        # Audio buffer
        audio_buffer = []
       
        while True:
            # Read audio chunk from ReSpeaker
            data = stream.read(CHUNK, exception_on_overflow=False)
           
            # Convert to numpy array and extract channel 1
            audio = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)
            channel_1 = audio[:, 1].astype(np.float32) / 32768.0  # Convert to float32 [-1, 1]
           
            # Add to buffer
            audio_buffer.extend(channel_1)
           
            # When we have enough samples, process
            if len(audio_buffer) >= SAMPLES_NEEDED:
                # Take exactly the needed number of samples
                waveform = np.array(audio_buffer[:SAMPLES_NEEDED])
               
                # Remove processed samples from buffer (keep the rest for next iteration)
                audio_buffer = audio_buffer[SAMPLES_NEEDED:]
               
                # Process with YAMNet
                scores, embeddings, spectrogram = model(waveform)
               
                # Get mean scores across time
                mean_scores = tf.reduce_mean(scores, axis=0).numpy()
               
                # Get top 5 sounds
                top_indices = np.argsort(mean_scores)[-5:][::-1]
               
                # Display results
                print(f"\n[{time.strftime('%H:%M:%S')}] Top detected sounds:")
                has_sound = False
                for i in top_indices:
                    confidence = mean_scores[i]
                    label = class_names[i]
                    if confidence > 0.1:  # Only show if confidence > 10%
                        print(f"   • {label:<30} {confidence*100:>5.1f}%")
                        has_sound = True
               
                if not has_sound:
                    print("   🔇 No clear sounds detected (confidence < 10%)")
               
                print("-" * 60)
               
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("\n✅ Stream closed")

def test_microphone_levels():
    """Quick test to check microphone input levels"""
    print("\n🎤 Testing microphone levels (speak or make noise)...")
   
    device_index = find_reSpeaker()
    if device_index is None:
        return
   
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK
    )
   
    try:
        print("Reading audio levels for 3 seconds...")
        for i in range(30):  # Read 30 chunks (~3 seconds at CHUNK=1024, RATE=16000)
            data = stream.read(CHUNK)
            audio = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)
            channel_1 = audio[:, 1]
           
            # Calculate RMS level
            rms = np.sqrt(np.mean(channel_1.astype(np.float32)**2))
           
            # Simple level indicator
            bars = int(rms / 500)  # Adjust scaling as needed
            level = "█" * min(bars, 50)
            print(f"Level: {level} ({rms:.1f})")
           
            time.sleep(0.1)
           
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    print("=" * 60)
    print("ReSpeaker Sound Recognition Test (YAMNet - Channel 1)")
    print("=" * 60)
   
    # Optional: Test microphone levels first
    test = input("\nTest microphone levels first? (y/n): ").lower()
    if test == 'y':
        test_microphone_levels()
   
    input("\nPress Enter to start sound recognition...")
    sound_recognition()