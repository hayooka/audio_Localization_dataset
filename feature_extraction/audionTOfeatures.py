# -*- coding: utf-8 -*-
"""
Feature Extraction — Sound Localization
Processes datasetE1 + datasetE2.
5min recordings -> train.csv
3min recordings -> test.csv
Features per 2s chunk: RMS (4 mics), IPD (3 pairs), GCC-PHAT TDOA + strength (6 pairs)
"""

import numpy as np
import wave
import os
import pandas as pd
from scipy.signal import hilbert

# ========================
# SETTINGS
# ========================
DATASETS = {
    'DATA': r'C:\Users\ahmma\Desktop\farah\(24 angles)dataset',
}
OUTPUT_DIR = r'C:\Users\ahmma\Desktop\farah\features'

ANGLES = list(range(0, 360, 15))
MICS      = ['mic_right', 'mic_front', 'mic_left', 'mic_back']
RATE      = 16000
CHUNK_SEC = 0.03  # 30ms = 480 samples at 16kHz

POSITION_TO_LABEL = {angle: i for i, angle in enumerate(ANGLES)}

# ========================
# FEATURE FUNCTIONS
# ========================
def load_wav(path):
    with wave.open(path, 'rb') as w:
        raw = w.readframes(w.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return audio

def rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))

def get_ipd(sig1, sig2):
    a1 = np.angle(hilbert(sig1))
    a2 = np.angle(hilbert(sig2))
    return float(np.mean(np.degrees(np.arctan2(np.sin(a1 - a2), np.cos(a1 - a2)))))

def get_gcc_phat(sig1, sig2):
    fft_len = 2 * len(sig1)
    X1  = np.fft.rfft(sig1, n=fft_len)
    X2  = np.fft.rfft(sig2, n=fft_len)
    Pxx = X1 * np.conj(X2)
    mag = np.abs(Pxx)
    mag[mag < 1e-10] = 1e-10
    gcc = np.fft.irfft(Pxx / mag, n=fft_len)[:len(sig1)]
    peak_idx = int(np.argmax(gcc))
    if peak_idx > len(gcc) // 2:
        peak_idx -= len(gcc)
    tdoa_ms  = float((peak_idx / RATE) * 1000)
    strength = float(np.max(gcc) / (np.std(gcc) + 1e-10))
    return tdoa_ms, strength

# ========================
# EXTRACT FROM ONE FILE SET
# ========================
def extract_chunks(base, angle, duration):
    chunk_samples = int(CHUNK_SEC * RATE)
    signals = {}
    for mic in MICS:
        path = os.path.join(base, str(angle), 'speakerM', duration, mic + '.wav')
        if not os.path.exists(path):
            return []
        signals[mic] = load_wav(path)

    n_chunks = min(len(s) for s in signals.values()) // chunk_samples
    samples  = []

    for ci in range(n_chunks):
        s = ci * chunk_samples
        e = s + chunk_samples
        ch = {m: signals[m][s:e] for m in MICS}

        rms_feats = [rms(ch[m]) for m in MICS]

        ipd_feats = [
            get_ipd(ch['mic_right'], ch['mic_front']),
            get_ipd(ch['mic_right'], ch['mic_left']),
            get_ipd(ch['mic_front'], ch['mic_back']),
        ]

        gcc_tdoa, gcc_str = [], []
        for i in range(4):
            for j in range(i + 1, 4):
                t, s_val = get_gcc_phat(ch[MICS[i]], ch[MICS[j]])
                gcc_tdoa.append(t)
                gcc_str.append(s_val)

        samples.append({
            'dataset': os.path.basename(base),
            'position': angle,
            'chunk':    ci,
            'label':    POSITION_TO_LABEL[angle],
            **{f'rms_{MICS[i]}':       rms_feats[i]  for i in range(4)},
            **{f'ipd_pair{i}':          ipd_feats[i]  for i in range(3)},
            **{f'gcc_tdoa_{i}':         gcc_tdoa[i]   for i in range(6)},
            **{f'gcc_strength_{i}':     gcc_str[i]    for i in range(6)},
        })

    return samples

# ========================
# MAIN LOOP
# ========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

for duration, split_name in [('5min', 'train'), ('3min', 'test')]:
    for ds_name, base in DATASETS.items():
        samples = []
        print(f'\n{"="*60}')
        print(f'  {split_name.upper()} — {ds_name}  ({duration} recordings)')
        print(f'{"="*60}')

        for angle in ANGLES:
            chunks = extract_chunks(base, angle, duration)
            if chunks:
                samples.extend(chunks)
                print(f'  {angle:>3}°  ->  {len(chunks)} chunks')
            else:
                print(f'  {angle:>3}°  ->  MISSING')

        df = pd.DataFrame(samples)
        out_path = os.path.join(OUTPUT_DIR, f'{split_name}_{ds_name}.csv')
        df.to_csv(out_path, index=False)
        print(f'\n  Saved: {out_path}')
        print(f'  Rows: {len(df)}  |  Columns: {len(df.columns)}  |  Classes: {df["label"].nunique()}')

print('\nDone.')
