# -*- coding: utf-8 -*-
"""
Feature Extraction — Sound Localization
Processes (24 angles)dataset.
5min recordings -> train.csv
3min recordings -> test.csv
Features per 30ms chunk: RMS (4 mics), IPD (3 pairs), GCC-PHAT TDOA + strength (6 pairs)
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
CHUNK_SEC = 0.03  # 50ms = 800 samples at 16kHz
N_MELS    = 40
N_FFT     = 1024  # next power of 2 >= 800

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
    a1 = np.angle(hilbert(sig1))  # type: ignore[arg-type]
    a2 = np.angle(hilbert(sig2))  # type: ignore[arg-type]
    return float(np.mean(np.degrees(np.arctan2(np.sin(a1 - a2), np.cos(a1 - a2)))))

def _build_mel_filterbank():
    def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)
    mel_pts = np.linspace(hz_to_mel(0), hz_to_mel(RATE / 2), N_MELS + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.floor((N_FFT + 1) * hz_pts / RATE).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1))
    for m in range(1, N_MELS + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / (c - l + 1e-10)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / (r - c + 1e-10)
    return fb

MEL_FB = _build_mel_filterbank()

def get_logmel_energy(chunk):
    """Log-mel energy vector of shape (N_MELS,) for one chunk."""
    power = np.abs(np.fft.rfft(chunk, n=N_FFT)) ** 2
    return np.log(MEL_FB @ power + 1e-10)

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

        rms_feats    = [rms(ch[m]) for m in MICS]
        logmel_feats = [get_logmel_energy(ch[m]) for m in MICS]  # 4 x (N_MELS,)

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
            **{f'rms_{MICS[i]}':            rms_feats[i]       for i in range(4)},
            **{f'ipd_pair{i}':              ipd_feats[i]       for i in range(3)},
            **{f'gcc_tdoa_{i}':             gcc_tdoa[i]        for i in range(6)},
            **{f'gcc_strength_{i}':         gcc_str[i]         for i in range(6)},
            **{f'logmel_{MICS[i]}_b{b}':    logmel_feats[i][b] for i in range(4) for b in range(N_MELS)},
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
        chunk_ms = int(CHUNK_SEC * 1000)
        out_path = os.path.join(OUTPUT_DIR, f'{split_name}_{ds_name}{chunk_ms}.csv')
        df.to_csv(out_path, index=False)
        print(f'\n  Saved: {out_path}')
        print(f'  Rows: {len(df)}  |  Columns: {len(df.columns)}  |  Classes: {df["label"].nunique()}')

print('\nDone.')
