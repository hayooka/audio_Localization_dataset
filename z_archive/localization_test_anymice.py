"""
Localization Test — works with ReSpeaker OR any regular mic.
Without ReSpeaker only a single virtual mic is available, so angle
prediction will be poor (no inter-mic cues), but the script still runs.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from silero_vad import load_silero_vad
from audioloc import _load, _infer
from audioloc._features import extract_chunk_all, CHUNK_SAMPLES, RATE

# ── Config ─────────────────────────────────────────────────────────────────────
ANGLES        = list(range(0, 360, 15))
SPEAK_SEC     = 5
FRAMES        = int(SPEAK_SEC * RATE / CHUNK_SAMPLES)
VAD_THRESHOLD = 0.2
RMS_THRESHOLD = 110
SAVE_CSV      = 'localization_test.csv'

# ── Device selection ───────────────────────────────────────────────────────────
def find_device():
    """Return (device_index, n_channels, is_respeaker)."""
    for i, d in enumerate(sd.query_devices()):
        if 'respeaker' in d['name'].lower() and d['max_input_channels'] >= 6:
            print(f"  Found ReSpeaker: [{i}] {d['name']}")
            return i, 6, True
    default = sd.default.device[0]
    d = sd.query_devices(default)
    ch = min(d['max_input_channels'], 2)
    print(f"  No ReSpeaker — using: [{default}] {d['name']}  ({ch} ch)")
    print("  WARNING: angle prediction requires 4 mics; results will be unreliable.")
    return default, ch, False

def build_ch(block, is_respeaker):
    """Map raw block columns to the 4 named mic channels."""
    if is_respeaker:
        # ReSpeaker: channels 2-5 are the 4 mics
        return {
            'mic_right': block[:, 2],
            'mic_front': block[:, 3],
            'mic_left':  block[:, 4],
            'mic_back':  block[:, 5],
        }
    # Fallback: replicate available channels across all 4 slots
    mono = block[:, 0]
    ch1  = block[:, 1] if block.shape[1] > 1 else mono
    return {
        'mic_right': mono,
        'mic_front': ch1,
        'mic_left':  mono,
        'mic_back':  ch1,
    }

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading models...")
_load()
vad_model = load_silero_vad()

print("\nAvailable input devices:")
for i, d in enumerate(sd.query_devices()):
    if d['max_input_channels'] > 0:
        print(f"  [{i}] {d['name']}  ({d['max_input_channels']} ch)")

device_index, n_ch, is_respeaker = find_device()
print()

# ── Instructions ───────────────────────────────────────────────────────────────
print("=" * 55)
print("  LOCALIZATION TEST — 24 ANGLES")
print("=" * 55)
print(f"  Device      : {'ReSpeaker (6 ch)' if is_respeaker else f'Regular mic ({n_ch} ch) — unreliable'}")
print(f"  Angles      : {len(ANGLES)}")
print(f"  Speak time  : {SPEAK_SEC}s per angle")
print(f"  RMS thresh  : {RMS_THRESHOLD}")
print("=" * 55)
input("\nPress Enter to start...\n")

# ── Test loop ──────────────────────────────────────────────────────────────────
results = []

for angle in ANGLES:
    print(f"\n{'='*55}")
    print(f"  Move to:  {angle}°")
    print(f"  Press Enter and START SPEAKING for {SPEAK_SEC} seconds")
    print(f"{'='*55}")
    input("  Ready? Press Enter...")

    print(f"  Recording {SPEAK_SEC}s at {angle}°...\n")
    print(f"  {'Frame':>6}  {'VAD':>6}  {'RMS':>6}  {'Speech':>7}  {'Pred':>6}  {'Conf':>7}  {'Latency':>10}")
    print(f"  {'-'*57}")

    for fi in range(FRAMES):
        t0 = time.perf_counter()

        block = sd.rec(CHUNK_SAMPLES, samplerate=RATE, channels=n_ch,
                       dtype='int16', device=device_index)
        sd.wait()
        block = block.astype('float32')
        if block.ndim == 1:
            block = block[:, None]

        ch = build_ch(block, is_respeaker)

        speech_prob = max(
            vad_model(torch.tensor(ch[m][:512] / 32768.0), RATE).item()
            for m in ch
        )
        rms = np.array([np.sqrt(np.mean(ch[m]**2)) for m in ch]).max()
        is_speech = speech_prob >= VAD_THRESHOLD and rms >= RMS_THRESHOLD

        if is_speech:
            _, feat = extract_chunk_all(ch)
            preds, _, confs = _infer(feat.reshape(1, -1))
            pred_angle = preds[0]
            conf       = confs[0] * 100
        else:
            pred_angle = None
            conf       = 0.0

        latency_ms = (time.perf_counter() - t0) * 1000

        print(f"  {fi+1:>6}  {speech_prob:>6.2f}  {rms:>6.0f}  "
              f"{'YES' if is_speech else 'no':>7}  "
              f"{str(pred_angle)+'°' if pred_angle is not None else '---':>6}  "
              f"{conf:>6.1f}%  {latency_ms:>8.1f}ms")

        results.append({
            'true_angle': angle, 'frame': fi,
            'vad_prob':   round(speech_prob, 3),
            'rms':        round(rms, 1),
            'is_speech':  is_speech,
            'pred_angle': pred_angle,
            'confidence': round(conf, 1),
            'latency_ms': round(latency_ms, 2),
        })

    print(f"\n  Done. Move to next angle.\n")

# ── Save CSV ───────────────────────────────────────────────────────────────────
with open(SAVE_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nResults saved to {SAVE_CSV}")

# ── Analysis ───────────────────────────────────────────────────────────────────
speech_frames = [r for r in results if r['is_speech'] and r['pred_angle'] is not None]
latencies     = [r['latency_ms'] for r in results if r['is_speech']]

if not speech_frames:
    print("No speech frames detected — check RMS_THRESHOLD and mic.")
    exit()

def angular_error(a, b):
    diff = abs(a - b)
    return min(diff, 360 - diff)

correct = sum(1 for r in speech_frames if r['pred_angle'] == r['true_angle'])
total   = len(speech_frames)
errors  = [angular_error(r['pred_angle'], r['true_angle']) for r in speech_frames]

print(f"\n{'='*55}")
print(f"  RESULTS SUMMARY")
print(f"{'='*55}")
print(f"  Total speech frames : {total}")
print(f"  Accuracy            : {correct}/{total} = {correct/total*100:.1f}%")
print(f"  Angular MAE         : {np.mean(errors):.1f}°")
print(f"  Angular RMSE        : {np.sqrt(np.mean(np.array(errors)**2)):.1f}°")
print(f"  Avg latency         : {np.mean(latencies):.1f}ms")
print(f"{'='*55}")

print(f"\n  {'Angle':>6}  {'Frames':>7}  {'Correct':>8}  {'Acc':>6}  {'MAE':>6}")
print(f"  {'-'*42}")
for angle in ANGLES:
    af = [r for r in speech_frames if r['true_angle'] == angle]
    if not af:
        print(f"  {angle:>5}°  {'0':>7}  {'---':>8}  {'---':>6}  {'---':>6}")
        continue
    c   = sum(1 for r in af if r['pred_angle'] == angle)
    mae = np.mean([angular_error(r['pred_angle'], angle) for r in af])
    print(f"  {angle:>5}°  {len(af):>7}  {c:>8}  {c/len(af)*100:>5.1f}%  {mae:>5.1f}°")

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Localization Test Results', fontsize=14)

cm = np.zeros((len(ANGLES), len(ANGLES)), dtype=int)
for r in speech_frames:
    if r['pred_angle'] in ANGLES:
        cm[ANGLES.index(r['true_angle']), ANGLES.index(r['pred_angle'])] += 1

ax = axes[0]
ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(ANGLES))); ax.set_yticks(range(len(ANGLES)))
ax.set_xticklabels(ANGLES, rotation=90, fontsize=6)
ax.set_yticklabels(ANGLES, fontsize=6)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
for i in range(len(ANGLES)):
    for j in range(len(ANGLES)):
        if cm[i, j] > 0:
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=5,
                    color='white' if cm[i, j] > cm.max()/2 else 'black')

ax = axes[1]
accs = []
for angle in ANGLES:
    af = [r for r in speech_frames if r['true_angle'] == angle]
    accs.append(sum(1 for r in af if r['pred_angle'] == angle) / len(af) * 100 if af else 0)
ax.bar(ANGLES, accs, width=12, color='steelblue', edgecolor='white')
ax.set_xlabel('True Angle (°)'); ax.set_ylabel('Accuracy (%)')
ax.set_title(f'Per-Angle Accuracy  (overall={correct/total*100:.1f}%)')
ax.set_xticks(ANGLES); ax.set_xticklabels(ANGLES, rotation=90, fontsize=7)
ax.set_ylim(0, 110); ax.axhline(correct/total*100, color='red', linestyle='--')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.hist(latencies, bins=30, color='coral', edgecolor='white')
ax.axvline(np.mean(latencies), color='red', linestyle='--', label=f'mean={np.mean(latencies):.1f}ms')
ax.set_xlabel('Latency (ms)'); ax.set_ylabel('Frames')
ax.set_title('Inference Latency')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('localization_test_results.png', dpi=150)
plt.show()
print("Plot saved: localization_test_results.png")
