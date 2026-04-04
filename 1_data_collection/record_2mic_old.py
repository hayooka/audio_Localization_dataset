import pyaudio
import numpy as np
import csv
from gammatone.gtgram import gtgram
import webrtcvad

# ========================
# SETTINGS
# ========================
RATE = 48000
FRAME_MS = 20
CHUNK = int(RATE * FRAME_MS / 1000)   # 960 samples
FRAME_TIME = CHUNK / RATE
FORMAT = pyaudio.paInt16
NUM_FILTERS = 20

MIC1_INDEX = 1    # Left mic
MIC2_INDEX = 13   # Right mic

LABEL = int(input("Enter Label for this recording (0=LEFT, 1=RIGHT): "))

# ========================
# INIT
# ========================
p = pyaudio.PyAudio()
vad = webrtcvad.Vad(2)   # 0–3 (2 = balanced)

stream1 = p.open(format=FORMAT, channels=1, rate=RATE, input=True,
                 input_device_index=MIC1_INDEX, frames_per_buffer=CHUNK)

stream2 = p.open(format=FORMAT, channels=1, rate=RATE, input=True,
                 input_device_index=MIC2_INDEX, frames_per_buffer=CHUNK)

print("🎤 Recording... Ctrl+C to stop\n")

# ========================
# MAIN LOOP
# ========================
try:
    while True:
        data1 = stream1.read(CHUNK, exception_on_overflow=False)
        data2 = stream2.read(CHUNK, exception_on_overflow=False)

        if not (vad.is_speech(data1, RATE) or vad.is_speech(data2, RATE)):
            continue

        left = np.frombuffer(data1, dtype=np.int16).astype(np.float32)
        right = np.frombuffer(data2, dtype=np.int16).astype(np.float32)

        left_gt = gtgram(left, RATE, FRAME_TIME, FRAME_TIME, NUM_FILTERS, 50)
        right_gt = gtgram(right, RATE, FRAME_TIME, FRAME_TIME, NUM_FILTERS, 50)

        PL_f = np.mean(left_gt, axis=1)
        PR_f = np.mean(right_gt, axis=1)

        features = np.concatenate((PL_f, PR_f))
        row = list(features) + [LABEL]

        with open("dataset.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"Frame saved | Label={LABEL}")

except KeyboardInterrupt:
    print("\n⏹️ Recording stopped")

finally:
    stream1.close()
    stream2.close()
    p.terminate()
