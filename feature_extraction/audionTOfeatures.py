# -*- coding: utf-8 -*-
"""
Sound Localization Feature Extraction - 8 Positions
Loads recorded WAVs → extracts RMS, IPD, GCC-PHAT → saves to CSV for ML
"""

import numpy as np
import wave
import os
import pandas as pd
from scipy.signal import hilbert

# ========================
# SETTINGS
# ========================
RAW_DIR = r"C:\Users\farah\OneDrive\Desktop\dataset\raw"
OUTPUT_DIR = r"C:\Users\farah\OneDrive\Desktop\dataset"

POSITIONS = [0, 45, 90, 135, 225, 270, 315]
MIC_NAMES = ["mic_right", "mic_front", "mic_left", "mic_back"]
RATE = 16000
ANALYSIS_SEC = 360  # 7 minutes
CHUNK_SEC = 2

print(f"\n🔍 DEBUG MODE: Checking file structure\n")
print(f"Looking in: {RAW_DIR}\n")

# ========================
# CHECK IF FILES EXIST
# ========================
def check_files():
    print("Checking for position folders...\n")
    found_files = {}

    for angle in POSITIONS:
        pos_name = f"position_{angle:03d}deg"
        pos_dir = os.path.join(RAW_DIR, pos_name)

        print(f"\n📁 {pos_name}/")

        if not os.path.exists(pos_dir):
            print(f"   ✗ FOLDER NOT FOUND: {pos_dir}")
            continue

        found_files[angle] = {}
        files_in_dir = os.listdir(pos_dir)
        print(f"   Files found: {files_in_dir}")

        for mic in MIC_NAMES:
            possible_names = [
                f"{mic}.wav", f"{mic}.WAV", mic, mic.upper(),
                f"{mic.split('_')[1]}.wav",
            ]
            found = False
            for name in possible_names:
                full_path = os.path.join(pos_dir, name)
                if os.path.exists(full_path):
                    print(f"   ✓ {mic:<15} → {name}")
                    found_files[angle][mic] = full_path
                    found = True
                    break
            if not found:
                print(f"   ✗ {mic:<15} → NOT FOUND")

    return found_files

found_files = check_files()

# ========================
# HELPER FUNCTIONS
# ========================
def find_file(pos_dir, mic):
    for name in [f"{mic}.wav", f"{mic}.WAV", mic, mic.upper(),
                 f"{mic.split('_')[1]}.wav", f"{mic.split('_')[1]}.WAV"]:
        p = os.path.join(pos_dir, name)
        if os.path.exists(p):
            return p
    return None

def load_wav(path, max_sec=ANALYSIS_SEC):
    try:
        print(f"   Loading: {os.path.basename(path)}", end="")
        with wave.open(path, 'rb') as w:
            framerate = w.getframerate()
            n_frames = w.getnframes()
            print(f" ({n_frames} frames @ {framerate}Hz)")
            nread = min(n_frames, int(max_sec * framerate))
            if nread <= 0:
                return np.zeros(int(max_sec * RATE), dtype=np.float32), framerate
            raw = w.readframes(nread)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            print(f"      ✓ Loaded {len(audio)} samples")
            return audio, framerate
    except Exception as e:
        print(f"      ✗ ERROR: {e}")
        return np.zeros(int(max_sec * RATE), dtype=np.float32), RATE

# ========================
# LOAD DATA
# ========================
print(f"\n{'='*80}\nLOADING AUDIO FILES\n{'='*80}\n")

data = {}
file_count = 0

for angle in POSITIONS:
    pos_name = f"position_{angle:03d}deg"
    pos_dir = os.path.join(RAW_DIR, pos_name)
    if not os.path.exists(pos_dir):
        print(f"✗ {pos_name}: FOLDER NOT FOUND")
        continue
    print(f"\n📁 {pos_name}/")
    data[angle] = {}
    for mic in MIC_NAMES:
        path = find_file(pos_dir, mic)
        if path:
            audio, rate = load_wav(path)
            data[angle][mic] = audio
            if len(audio) > 0:
                file_count += 1
        else:
            print(f"   ✗ {mic:<15} NOT FOUND")
            data[angle][mic] = np.zeros(int(ANALYSIS_SEC * RATE), dtype=np.float32)

print(f"\n{'='*80}\nSummary: Loaded {file_count}/28 files successfully\n{'='*80}")

if file_count == 0:
    print("\n✗ ERROR: NO FILES LOADED!")
    exit()

# ========================
# FEATURE EXTRACTION
# ========================
def rms(audio):
    return float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0

def get_ipd(sig1, sig2):
    if len(sig1) < 2 or len(sig2) < 2:
        return 0.0
    try:
        analytic1 = hilbert(sig1)
        analytic2 = hilbert(sig2)
        ipd = np.arctan2(np.sin(np.angle(analytic1) - np.angle(analytic2)),
                         np.cos(np.angle(analytic1) - np.angle(analytic2)))
        return float(np.mean(np.degrees(ipd)))
    except Exception as e:
        print(f"⚠️  IPD error: {e}")
        return 0.0

def get_gcc_phat(sig1, sig2, sr=16000):
    if len(sig1) < 2 or len(sig2) < 2:
        return 0.0, 0.0
    try:
        fft_len = 2 * len(sig1)
        X1 = np.fft.rfft(sig1, n=fft_len)
        X2 = np.fft.rfft(sig2, n=fft_len)
        Pxx = X1 * np.conj(X2)
        mag = np.abs(Pxx)
        mag[mag < 1e-10] = 1e-10
        gcc = np.fft.irfft(Pxx / mag, n=fft_len)[:len(sig1)]
        peak_idx = np.argmax(gcc)
        if peak_idx > len(gcc) / 2:
            peak_idx = peak_idx - len(gcc)
        tdoa_ms = float((peak_idx / sr) * 1000)
        strength = float(np.max(gcc) / (np.std(gcc) + 1e-10))
        return tdoa_ms, strength
    except Exception as e:
        print(f"⚠️  GCC-PHAT error: {e}")
        return 0.0, 0.0

# ========================
# EXTRACT FEATURES PER CHUNK
# ========================
print(f"\n{'='*80}\nEXTRACTING FEATURES\n{'='*80}\n")

all_samples = []
n_chunks = ANALYSIS_SEC // CHUNK_SEC
chunk_samples = int(CHUNK_SEC * RATE)

for angle_idx, angle in enumerate(POSITIONS):
    print(f"Processing {angle_idx+1}/{len(POSITIONS)}: {angle}°", end="")
    if angle not in data:
        print(" ✗ SKIPPED")
        continue

    chunks_extracted = 0
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_samples
        end = start + chunk_samples
        chunks = {mic: data[angle][mic][start:end] for mic in MIC_NAMES}

        if any(len(c) == 0 for c in chunks.values()):
            continue

        try:
            rms_features = [rms(chunks[m]) for m in MIC_NAMES]
            ipd_features = [
                get_ipd(chunks["mic_right"], chunks["mic_front"]),
                get_ipd(chunks["mic_right"], chunks["mic_left"]),
                get_ipd(chunks["mic_front"], chunks["mic_back"]),
            ]
            gcc_tdoa, gcc_strength = [], []
            for i in range(4):
                for j in range(i+1, 4):
                    t, s = get_gcc_phat(chunks[MIC_NAMES[i]], chunks[MIC_NAMES[j]], sr=RATE)
                    gcc_tdoa.append(t)
                    gcc_strength.append(s)

            sample = {'position': angle, 'chunk': chunk_idx,
                      **{f'rms_mic{i}': rms_features[i] for i in range(4)},
                      **{f'ipd_pair{i}': ipd_features[i] for i in range(3)},
                      **{f'gcc_tdoa_{i}': gcc_tdoa[i] for i in range(6)},
                      **{f'gcc_strength_{i}': gcc_strength[i] for i in range(6)}}
            all_samples.append(sample)
            chunks_extracted += 1
        except Exception as e:
            print(f"\n⚠️  Chunk {chunk_idx}: {e}")
            continue

    print(f" ✓ ({chunks_extracted} chunks)")

print(f"\n✓ Total samples: {len(all_samples)}")
if len(all_samples) == 0:
    exit()

# ========================
# LABELS & SAVE
# ========================
position_to_label = {0: 0, 45: 1, 90: 2, 135: 3, 180: 4, 225: 5, 270: 6, 315: 7}
for s in all_samples:
    s['label'] = position_to_label[s['position']]

df = pd.DataFrame(all_samples)
os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_path = os.path.join(OUTPUT_DIR, "sound_localization_data_8pos.csv")
df.to_csv(csv_path, index=False)

print(f"\n✓ CSV saved: {csv_path}")
print(f"   Rows: {len(df)}  |  Columns: {len(df.columns)}")
