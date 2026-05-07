import pyaudio
import numpy as np
import wave
import os
import glob
from datetime import datetime

# ========================
# SETTINGS
# ========================
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 6
RATE = 16000
DEVICE_NAME = "reSpeaker XVF3800"

# ✅ الزوايا الجديدة فقط
ANGLES = [15, 30, 60, 75, 105, 120, 150, 165, 195, 210, 240, 255, 285, 300, 330, 345]

RECORD_MINUTES_LIST = [5, 3]

# ========================
# USB DETECTION
# ========================
def find_usb():
    # Check all users under /media (works regardless of username)
    drives = glob.glob("/media/*/*/")
    if drives:
        return drives[0].rstrip("/")
    return os.path.join(os.path.expanduser("~"), "raw_backup")

USB_PATH = find_usb()

# ========================
# WAV FUNCTION
# ========================
def open_wav(path):
    w = wave.open(path, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(RATE)
    return w

# ========================
# CHECK WAV DURATION
# ========================
def is_valid_wav(path, expected_minutes):
    if not os.path.exists(path):
        return False
    try:
        with wave.open(path, 'rb') as w:
            frames = w.getnframes()
            rate = w.getframerate()
            duration = frames / float(rate)
            return duration >= expected_minutes * 60 * 0.9
    except:
        return False

# ========================
# CHECK IF DONE
# ========================
def already_done(folder, minutes):
    files = [
        "mic_right.wav",
        "mic_front.wav",
        "mic_left.wav",
        "mic_back.wav"
    ]
    return all(
        is_valid_wav(os.path.join(folder, f), minutes)
        for f in files
    )

# ========================
# FIND DEVICE
# ========================
p = pyaudio.PyAudio()
device_index = None

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if DEVICE_NAME in str(info['name']):
        device_index = i
        print(f"✅ Found '{DEVICE_NAME}' at index {device_index}")
        break

if device_index is None:
    print("❌ ReSpeaker not found")
    p.terminate()
    exit()

# ========================
# OPEN STREAM
# ========================
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=device_index,
    frames_per_buffer=CHUNK
)

# ========================
# RECORD FUNCTION
# ========================
def record(angle, mode, minutes):

    OUT_DIR = os.path.join(
        USB_PATH,
        "dataset",
        f"{angle}",
        mode,
        f"{minutes}min"
    )

    os.makedirs(OUT_DIR, exist_ok=True)

    # ⛔ Skip إذا مكتمل
    if already_done(OUT_DIR, minutes):
        print(f"⏭️ Skipping (done): {OUT_DIR}")
        return

    input(f"\n⚠️ {mode} | زاوية {angle}° | {minutes}min\nاضغطي Enter للبدء...")

    print(f"🔴 Recording {minutes} minutes...")

    total_chunks = int(minutes * 60 * RATE / CHUNK)

    wav_right = open_wav(os.path.join(OUT_DIR, "mic_right.wav"))
    wav_front = open_wav(os.path.join(OUT_DIR, "mic_front.wav"))
    wav_left  = open_wav(os.path.join(OUT_DIR, "mic_left.wav"))
    wav_back  = open_wav(os.path.join(OUT_DIR, "mic_back.wav"))

    start_time = datetime.now()

    try:
        for chunk_idx in range(total_chunks):

            data  = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)

            right = audio[:, 2]
            front = audio[:, 3]
            left  = audio[:, 4]
            back  = audio[:, 5]

            wav_right.writeframes(right.tobytes())
            wav_front.writeframes(front.tobytes())
            wav_left.writeframes(left.tobytes())
            wav_back.writeframes(back.tobytes())

            if chunk_idx % int(30 * RATE / CHUNK) == 0:
                elapsed = (datetime.now() - start_time).seconds
                remaining = max(0, minutes * 60 - elapsed)
                print(f"⏱ {elapsed//60:02d}:{elapsed%60:02d} | {remaining//60:02d}:{remaining%60:02d}")

    except KeyboardInterrupt:
        print("⛔ Stopped")

    finally:
        wav_right.close()
        wav_front.close()
        wav_left.close()
        wav_back.close()
        print(f"✅ Saved: {OUT_DIR}")

# ========================
# MAIN LOOP
# ========================
try:

    # 🔵 speakerM
    for angle in ANGLES:
        for minutes in RECORD_MINUTES_LIST:
            record(angle, "speakerM", minutes)

    # 🔴 respeakerM
    for angle in ANGLES:
        for minutes in RECORD_MINUTES_LIST:
            record(angle, "respeakerM", minutes)

except KeyboardInterrupt:
    print("\n⛔ Program stopped")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("🧹 Closed")