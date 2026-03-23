import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import webrtcvad
from gammatone.gtgram import gtgram
import seaborn as sns

# ========================
# SETTINGS
# ========================
RATE = 48000
FRAME_MS = 20
CHUNK = int(RATE * FRAME_MS / 1000)   # 960 samples
FRAME_TIME = CHUNK / RATE
FORMAT = pyaudio.paInt16
NUM_FILTERS = 20

MIC1_INDEX = 2   # Left mic
MIC2_INDEX = 1   # Right mic

# ========================
# INIT
# ========================
p = pyaudio.PyAudio()
vad = webrtcvad.Vad(2)

stream1 = p.open(format=FORMAT, channels=1, rate=RATE, input=True,
                 input_device_index=MIC1_INDEX, frames_per_buffer=CHUNK)
stream2 = p.open(format=FORMAT, channels=1, rate=RATE, input=True,
                 input_device_index=MIC2_INDEX, frames_per_buffer=CHUNK)

print("🎤 Recording... Ctrl+C to stop\n")

left_signal_all, right_signal_all = [], []
left_total_power, right_total_power = [], []
left_filter_power, right_filter_power = [], []

# ========================
# MAIN LOOP
# ========================
try:
    while True:
        data1 = stream1.read(CHUNK, exception_on_overflow=False)
        data2 = stream2.read(CHUNK, exception_on_overflow=False)

        if not (vad.is_speech(data1, RATE) or vad.is_speech(data2, RATE)):
            continue

        left  = np.frombuffer(data1, dtype=np.int16).astype(np.float32)
        right = np.frombuffer(data2, dtype=np.int16).astype(np.float32)

        left_signal_all.append(left)
        right_signal_all.append(right)

        left_gt  = gtgram(left,  RATE, FRAME_TIME, FRAME_TIME, NUM_FILTERS, 50)
        right_gt = gtgram(right, RATE, FRAME_TIME, FRAME_TIME, NUM_FILTERS, 50)

        PL_f = np.mean(left_gt,  axis=1)
        PR_f = np.mean(right_gt, axis=1)

        left_filter_power.append(PL_f)
        right_filter_power.append(PR_f)

        PL_total = np.sum(PL_f)
        PR_total = np.sum(PR_f)

        left_total_power.append(PL_total)
        right_total_power.append(PR_total)

        source = "LEFT 🎤" if PL_total > PR_total else "RIGHT 🎤"
        print(f"Power L={PL_total:.1f} | R={PR_total:.1f} → {source}")

except KeyboardInterrupt:
    print("\n⏹️ Stopped")

stream1.close(); stream2.close(); p.terminate()

# ========================
# PLOTS
# ========================
left_signal_all  = np.concatenate(left_signal_all)
right_signal_all = np.concatenate(right_signal_all)
t = np.arange(len(left_signal_all)) / RATE

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1); plt.plot(t, left_signal_all)
plt.title("Left Microphone Signal"); plt.ylabel("Amplitude"); plt.grid()
plt.subplot(2, 1, 2); plt.plot(t, right_signal_all)
plt.title("Right Microphone Signal"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.grid()
plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 4))
plt.plot(left_total_power,  label="Left Power")
plt.plot(right_total_power, label="Right Power")
plt.xlabel("Frame Index"); plt.ylabel("Total Energy")
plt.title("Sound Localization (Frame-by-Frame)")
plt.legend(); plt.grid(); plt.show()

left_fp  = np.array(left_filter_power)
right_fp = np.array(right_filter_power)

plt.figure(figsize=(10, 5))
plt.plot(left_fp.mean(axis=0),  label="Left")
plt.plot(right_fp.mean(axis=0), label="Right")
plt.xlabel("Filter Index (Low → High Frequency)"); plt.ylabel("Average Energy")
plt.title("Gammatone Filter Bank Energy Distribution")
plt.legend(); plt.grid(); plt.show()

for arr, title in [(left_fp, "Left"), (right_fp, "Right")]:
    plt.figure(figsize=(12, 6))
    sns.heatmap(arr.T, cmap='viridis', cbar=True)
    plt.xlabel("Frame Index"); plt.ylabel("Filter Index (Low→High)")
    plt.title(f"{title} Mic Gammatone Filter Bank Energy (Time × Filter)")
    plt.show()
