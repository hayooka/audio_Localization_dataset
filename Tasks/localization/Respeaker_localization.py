import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import webrtcvad
from gammatone.gtgram import gtgram

# ========================
# SETTINGS
# ========================
DEVICE_INDEX = 25
CHANNELS = 6
RATE = 16000
DURATION = 0.02  # 20ms
FRAME_MS = 20
FRAME_SIZE = int(RATE * FRAME_MS / 1000)  # 320 samples
FRAME_TIME = FRAME_SIZE / RATE
NUM_FILTERS = 20

print("🎤 4-Mic Real-time Localization with Gammatone Filter Bank")
print("="*60)
print("📍 Microphone Mapping:")
print("   Mic0 (ch2) → RIGHT")
print("   Mic1 (ch3) → FRONT")
print("   Mic2 (ch4) → LEFT")
print("   Mic3 (ch5) → BACK")
print("="*60)

# ========================
# INIT
# ========================
vad = webrtcvad.Vad(2)

front_signal_all, right_signal_all = [], []
left_signal_all, back_signal_all = [], []
front_total_power, right_total_power = [], []
left_total_power, back_total_power = [], []
front_filter_power, right_filter_power = [], []
left_filter_power, back_filter_power = [], []

print("\n🎤 Recording... Ctrl+C to stop\n")

# ========================
# MAIN LOOP
# ========================
try:
    while True:
        audio = sd.rec(int(DURATION * RATE), samplerate=RATE,
                       channels=CHANNELS, dtype='int16', device=DEVICE_INDEX)
        sd.wait()

        right = audio[:, 2].astype(np.float32)
        front = audio[:, 3].astype(np.float32)
        left  = audio[:, 4].astype(np.float32)
        back  = audio[:, 5].astype(np.float32)

        front_int16 = front.astype(np.int32)
        if not vad.is_speech(front_int16.tobytes(), RATE):
            continue

        front_signal_all.append(front)
        right_signal_all.append(right)
        left_signal_all.append(left)
        back_signal_all.append(back)

        front_gt = gtgram(front, RATE, FRAME_TIME, FRAME_TIME, NUM_FILTERS, 50)
        right_gt = gtgram(right, RATE, FRAME_TIME, FRAME_TIME, NUM_FILTERS, 50)
        left_gt  = gtgram(left,  RATE, FRAME_TIME, FRAME_TIME, NUM_FILTERS, 50)
        back_gt  = gtgram(back,  RATE, FRAME_TIME, FRAME_TIME, NUM_FILTERS, 50)

        Pf_f = np.mean(front_gt, axis=1)
        Pr_f = np.mean(right_gt, axis=1)
        Pl_f = np.mean(left_gt,  axis=1)
        Pb_f = np.mean(back_gt,  axis=1)

        front_filter_power.append(Pf_f)
        right_filter_power.append(Pr_f)
        left_filter_power.append(Pl_f)
        back_filter_power.append(Pb_f)

        Pf_total = np.sum(Pf_f)
        Pr_total = np.sum(Pr_f)
        Pl_total = np.sum(Pl_f)
        Pb_total = np.sum(Pb_f)

        front_total_power.append(Pf_total)
        right_total_power.append(Pr_total)
        left_total_power.append(Pl_total)
        back_total_power.append(Pb_total)

        source = ["FRONT ↑", "RIGHT →", "LEFT ←", "BACK ↓"][
            np.argmax([Pf_total, Pr_total, Pl_total, Pb_total])]
        print(f"Power F={Pf_total:8.1f} | R={Pr_total:8.1f} | L={Pl_total:8.1f} | B={Pb_total:8.1f} → {source}")

except KeyboardInterrupt:
    print("\n⏹️ Stopped")

# ========================
# PLOTS
# ========================
print("\n📊 Generating plots...")

front_signal_all = np.concatenate(front_signal_all)
right_signal_all = np.concatenate(right_signal_all)
left_signal_all  = np.concatenate(left_signal_all)
back_signal_all  = np.concatenate(back_signal_all)
t = np.arange(len(front_signal_all)) / RATE

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
for ax, sig, title in zip(axes,
    [front_signal_all, right_signal_all, left_signal_all, back_signal_all],
    ["FRONT", "RIGHT", "LEFT", "BACK"]):
    ax.plot(t, sig); ax.set_title(f"{title} Microphone"); ax.set_ylabel("Amplitude"); ax.grid()
axes[-1].set_xlabel("Time (s)")
plt.tight_layout(); plt.show()

plt.figure(figsize=(12, 6))
plt.plot(front_total_power, label="FRONT", color='red')
plt.plot(right_total_power, label="RIGHT", color='green')
plt.plot(left_total_power,  label="LEFT",  color='blue')
plt.plot(back_total_power,  label="BACK",  color='orange')
plt.xlabel("Frame Index"); plt.ylabel("Total Energy")
plt.title("4-Mic Sound Localization (Frame-by-Frame)")
plt.legend(); plt.grid(); plt.show()

print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
print(f"Total frames: {len(front_total_power)}")
print(f"Avg Power — FRONT: {np.mean(front_total_power):.1f} | RIGHT: {np.mean(right_total_power):.1f} "
      f"| LEFT: {np.mean(left_total_power):.1f} | BACK: {np.mean(back_total_power):.1f}")
