import sounddevice as sd
import numpy as np
import wave
import os
from gammatone.gtgram import gtgram
from datetime import datetime
import time

# ========================
# SETTINGS
# ========================
DEVICE_NAME = "reSpeaker"    # partial match — works for XVF3800 and similar
CHANNELS = 6
RATE = 16000
DURATION = 0.02          # 20ms (same as FRAME_MS)
FRAME_MS = 20
FRAME_SIZE = int(RATE * FRAME_MS / 1000)  # 320 samples
FRAME_TIME = FRAME_SIZE / RATE
NUM_FILTERS = 20

RECORD_MINUTES = 7       # total recording duration per position
TOTAL_FRAMES = int((RECORD_MINUTES * 60) / DURATION)

ANGLES = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165,
          180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
OUTPUT_BASE = "raw"

# ========================
# *** CHANGE THIS BEFORE EACH RECORDING SESSION ***
# ========================
POSITION = 0   # choose from ANGLES list above

# ========================
# AUTO-DETECT RESPEAKER
# ========================
device_index = None
for i, dev in enumerate(sd.query_devices()):
    if DEVICE_NAME.lower() in dev['name'].lower() and dev['max_input_channels'] >= CHANNELS:
        device_index = i
        print(f"Found '{dev['name']}' at index {device_index}")
        break

if device_index is None:
    raise RuntimeError(f"ReSpeaker not found. Run `python -m sounddevice` to list devices.")

if POSITION not in ANGLES:
    raise ValueError(f"Invalid position {POSITION}. Choose from: {ANGLES}")

POS_NAME = f"position_{POSITION:03d}deg"
OUT_DIR  = os.path.join(OUTPUT_BASE, POS_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

print("🎤 4-Mic Recording with Gammatone Filter Bank")
print("="*60)
print("📍 Microphone Mapping:")
print("   Mic0 (ch2) → RIGHT")
print("   Mic1 (ch3) → FRONT")
print("   Mic2 (ch4) → LEFT")
print("   Mic3 (ch5) → BACK")
print("="*60)
print(f"📍 Position  : {POSITION}°  ({POS_NAME})")
print(f"⏱  Duration  : {RECORD_MINUTES} minutes  ({TOTAL_FRAMES} frames)")
print(f"💾 Output    : {OUT_DIR}/")
print("="*60)

print("\n⏳ Starting in 3 seconds — place speaker at the correct position...")
time.sleep(3)

# ========================
# WAV FILE SETUP
# ========================
def open_wav(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = wave.open(path, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)   # 16-bit
    w.setframerate(RATE)
    return w

wav_right = open_wav(os.path.join(OUT_DIR, "mic_right.wav"))
wav_front = open_wav(os.path.join(OUT_DIR, "mic_front.wav"))
wav_left  = open_wav(os.path.join(OUT_DIR, "mic_left.wav"))
wav_back  = open_wav(os.path.join(OUT_DIR, "mic_back.wav"))

# ========================
# VAD LOG SETUP
# ========================
vad_log_path  = os.path.join(OUT_DIR, "vad_log.txt")
vad_log_lines = ["frame_index,vad_active,gammatone_energy"]

# ========================
# COUNTERS
# ========================
frame_idx     = 0
speech_frames = 0
start_time    = datetime.now()

print("\n🔴 Recording...\n")

# ========================
# MAIN LOOP
# ========================
try:
    while frame_idx < TOTAL_FRAMES:
        audio = sd.rec(int(DURATION * RATE), samplerate=RATE,
                       channels=CHANNELS, dtype='int16', device=device_index)
        sd.wait()

        right = audio[:, 2].astype(np.float32)
        front = audio[:, 3].astype(np.float32)
        left  = audio[:, 4].astype(np.float32)
        back  = audio[:, 5].astype(np.float32)

        right_int16 = right.astype(np.int16)
        front_int16 = front.astype(np.int16)
        left_int16  = left.astype(np.int16)
        back_int16  = back.astype(np.int16)


        if frame_idx % (30 * 1000 // FRAME_MS) == 0:
            elapsed    = (datetime.now() - start_time).seconds
            remaining  = max(0, RECORD_MINUTES * 60 - elapsed)
            speech_pct = speech_frames / frame_idx * 100
            print(f"  ⏱  {elapsed//60:02d}:{elapsed%60:02d} elapsed  |  "
                  f"{remaining//60:02d}:{remaining%60:02d} remaining  |  "
                  f"Speech: {speech_pct:.1f}%  |  "
                  f"Frame: {frame_idx}/{TOTAL_FRAMES}")

except KeyboardInterrupt:
    print(f"\n⚠️  Interrupted at frame {frame_idx}/{TOTAL_FRAMES}")

finally:
    wav_right.close()
    wav_front.close()
    wav_left.close()
    wav_back.close()

    with open(vad_log_path, "w") as f:
        f.write("\n".join(vad_log_lines))

    elapsed    = (datetime.now() - start_time).seconds
    saved_mins = frame_idx * FRAME_MS / 1000 / 60
    speech_pct = speech_frames / max(frame_idx, 1) * 100

    print(f"\n{'='*60}")
    print(f"✅ Recording complete!")
    print(f"   Saved     : {saved_mins:.1f} min  ({frame_idx} frames)")
    print(f"   Speech    : {speech_pct:.1f}% of recording")
    print(f"   VAD log   : {vad_log_path}")
    print(f"   Files saved:")
    for mic in ["mic_right", "mic_front", "mic_left", "mic_back"]:
        p = os.path.join(OUT_DIR, f"{mic}.wav")
        size_mb = os.path.getsize(p) / 1e6 if os.path.exists(p) else 0
        print(f"     {p}  ({size_mb:.1f} MB)")
    print(f"\n▶️  Next: change POSITION at the top and run again")
    print("="*60)
