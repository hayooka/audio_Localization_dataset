# -*- coding: utf-8 -*-
"""
Complete Method Comparison with REAL Separability Calculation
=============================================================
All features extracted from audio data.
Separability scores calculated via Fisher's Discriminant Ratio.
Methods: RMS, TDOA Raw, IPD, Spectrogram, GCC-PHAT, GCC Strength, Hybrid
"""

import numpy as np
import wave
import os
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ========================
# SETTINGS
# ========================
RAW_DIR    = r"C:\Users\farah\OneDrive\Desktop\dataset\raw"
POSITIONS  = [0, 45, 90, 135, 225, 270, 315]
MIC_NAMES  = ["mic_right", "mic_front", "mic_left", "mic_back"]
MIC_LABELS = ["Mic0 (Right)", "Mic1 (Front)", "Mic2 (Left)", "Mic3 (Back)"]
MIC_COLORS = ["#378ADD", "#1D9E75", "#9F77DD", "#E24B4A"]
RATE       = 16000
ANALYSIS_SEC = 360

# ========================
# PROGRESS BAR
# ========================
def print_progress(current, total, label="Progress", width=50):
    if total == 0:
        return
    percent = (current / total) * 100
    filled = int(width * current // total)
    bar = '█' * filled + '░' * (width - filled)
    print(f'\r{label}: |{bar}| {percent:>5.1f}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()

# ========================
# LOAD DATA
# ========================
def find_file(pos_dir, mic):
    for name in [f"{mic}.wav", mic]:
        p = os.path.join(pos_dir, name)
        if os.path.exists(p):
            return p
    return None

def load_wav(path, max_sec=ANALYSIS_SEC):
    try:
        with wave.open(path, 'rb') as w:
            rate = w.getframerate()
            nread = min(w.getnframes(), int(max_sec * rate))
            raw = w.readframes(nread)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            return audio, rate
    except:
        return np.zeros(int(max_sec * RATE)), RATE

def rms(audio):
    return float(np.sqrt(np.mean(audio**2)))

print(f"\n🤖 Loading from: {RAW_DIR}\n")

total_files = len(POSITIONS) * len(MIC_NAMES)
current_file = 0
data = {}

for angle in POSITIONS:
    pos_name = f"position_{angle:03d}deg"
    pos_dir  = os.path.join(RAW_DIR, pos_name)
    data[angle] = {}
    for mic in MIC_NAMES:
        current_file += 1
        print_progress(current_file, total_files, "📂 Loading audio files")
        path = find_file(pos_dir, mic)
        data[angle][mic] = load_wav(path)[0] if path else np.zeros(int(RATE * ANALYSIS_SEC))

print()

# ========================
# FEATURE EXTRACTION
# ========================
def get_spectrogram(audio, n_fft=2048, hop_length=512):
    window = np.hanning(n_fft)
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    spec = np.zeros((n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spec[:, i] = np.abs(np.fft.rfft(frame * window))
    log_spec = 10 * np.log10(spec**2 + 1e-10)
    return log_spec - np.max(log_spec)

def get_ipd(sig1, sig2):
    analytic1 = hilbert(sig1)
    analytic2 = hilbert(sig2)
    ipd = np.arctan2(np.sin(np.angle(analytic1) - np.angle(analytic2)),
                     np.cos(np.angle(analytic1) - np.angle(analytic2)))
    return np.mean(np.degrees(ipd))

def get_gcc_phat(sig1, sig2, sr=16000):
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
    tdoa_ms = (peak_idx / sr) * 1000
    strength = np.max(gcc) / (np.std(gcc) + 1e-10)
    return tdoa_ms, strength

def get_tdoa_raw(sig1, sig2, sr=16000):
    cc = np.correlate(sig1, sig2, mode='full')
    lags = np.arange(-(len(sig1)-1), len(sig2)) / sr * 1000
    return lags[np.argmax(cc)]

# Extract all features
print("\n" + "="*80 + "\nEXTRACTING FEATURES\n" + "="*80 + "\n")

rms_values, tdoa_values, ipd_values, spec_values = {}, {}, {}, {}
gcc_values, gcc_strengths = {}, {}

for angle in POSITIONS:
    rms_values[angle]  = np.array([rms(data[angle][m]) for m in MIC_NAMES])
    spec_values[angle] = np.array([np.mean(get_spectrogram(data[angle][m])) for m in MIC_NAMES])
    ipd_values[angle]  = np.array([get_ipd(data[angle][MIC_NAMES[i]], data[angle][MIC_NAMES[j]])
                                    for i, j in [(0,1),(0,2),(1,3)]])
    tdoas, strengths, raw_tdoas = [], [], []
    for i in range(4):
        for j in range(i+1, 4):
            t, s = get_gcc_phat(data[angle][MIC_NAMES[i]], data[angle][MIC_NAMES[j]], sr=RATE)
            tdoas.append(t); strengths.append(s)
            raw_tdoas.append(get_tdoa_raw(data[angle][MIC_NAMES[i]], data[angle][MIC_NAMES[j]], sr=RATE))
    gcc_values[angle]    = np.array(tdoas)
    gcc_strengths[angle] = np.array(strengths)
    tdoa_values[angle]   = np.array(raw_tdoas)
    print(f"{angle:3d}° done")

# ========================
# SEPARABILITY (Fisher's Discriminant Ratio)
# ========================
def calc_separability_score(feature_dict):
    values = np.array([feature_dict[pos] for pos in POSITIONS])
    overall_mean = np.mean(values)
    if len(values.shape) == 1:
        position_means = values
        within_var = np.var(values)
    else:
        position_means = np.array([np.mean(feature_dict[pos]) for pos in POSITIONS])
        within_var = np.mean([np.var(feature_dict[pos]) for pos in POSITIONS])
    between_var = np.sum((position_means - overall_mean) ** 2) / len(POSITIONS)
    fisher_ratio = between_var / (within_var + 1e-10)
    return min(10, fisher_ratio), fisher_ratio

rms_sep,          rms_fisher          = calc_separability_score(rms_values)
tdoa_sep,         tdoa_fisher         = calc_separability_score(tdoa_values)
ipd_sep,          ipd_fisher          = calc_separability_score(ipd_values)
spec_sep,         spec_fisher         = calc_separability_score(spec_values)
gcc_sep,          gcc_fisher          = calc_separability_score(gcc_values)
gcc_strength_sep, gcc_strength_fisher = calc_separability_score(gcc_strengths)
hybrid_sep = (spec_sep + ipd_sep) / 2

print(f"\nRMS:           Fisher={rms_fisher:>6.2f}  → score={rms_sep:>5.1f}/10")
print(f"TDOA Raw:      Fisher={tdoa_fisher:>6.2f}  → score={tdoa_sep:>5.1f}/10")
print(f"IPD:           Fisher={ipd_fisher:>6.2f}  → score={ipd_sep:>5.1f}/10")
print(f"Spec:          Fisher={spec_fisher:>6.2f}  → score={spec_sep:>5.1f}/10")
print(f"GCC-PHAT TDOA: Fisher={gcc_fisher:>6.2f}  → score={gcc_sep:>5.1f}/10")
print(f"GCC Strength:  Fisher={gcc_strength_fisher:>6.2f}  → score={gcc_strength_sep:>5.1f}/10")
print(f"Hybrid:        combined    → score={hybrid_sep:>5.1f}/10")

# ========================
# PLOTS
# ========================
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

spec_data     = np.array([spec_values[p]     for p in POSITIONS])
ipd_data      = np.array([ipd_values[p]      for p in POSITIONS])
gcc_data      = np.array([gcc_values[p]      for p in POSITIONS])
rms_data      = np.array([rms_values[p]      for p in POSITIONS])
tdoa_data     = np.array([tdoa_values[p]     for p in POSITIONS])
strength_data = np.array([gcc_strengths[p]   for p in POSITIONS])

for ax, data_arr, title, ylabel, colors in [
    (fig.add_subplot(gs[0, 0]), spec_data,     'METHOD 1: Spectrogram Energy', 'Energy',              MIC_COLORS),
    (fig.add_subplot(gs[1, 0]), rms_data,      'BASELINE: RMS Energy',         'RMS',                 MIC_COLORS),
]:
    ax.set_title(title, fontweight='bold', fontsize=11)
    for i, color in enumerate(colors):
        ax.plot(POSITIONS, data_arr[:, i], 'o-', label=MIC_LABELS[i], color=color, linewidth=2, markersize=8)
    ax.set_xlabel('Position (°)'); ax.set_ylabel(ylabel); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

for ax, data_arr, title, ylabel in [
    (fig.add_subplot(gs[0, 1]), ipd_data,      'METHOD 2: IPD Features',       'IPD (degrees)'),
    (fig.add_subplot(gs[0, 2]), gcc_data,      'METHOD 3: GCC-PHAT TDOA',      'TDOA (ms)'),
    (fig.add_subplot(gs[1, 1]), tdoa_data,     'BASELINE: Raw TDOA',           'TDOA (ms)'),
    (fig.add_subplot(gs[1, 2]), strength_data, 'METHOD 3: GCC-PHAT Strength',  'Correlation Strength'),
]:
    ax.set_title(title, fontweight='bold', fontsize=11)
    for i in range(data_arr.shape[1] if data_arr.ndim > 1 else 1):
        ax.plot(POSITIONS, data_arr[:, i] if data_arr.ndim > 1 else data_arr,
                'o-', label=f'Pair {i}', linewidth=2, markersize=8)
    ax.set_xlabel('Position (°)'); ax.set_ylabel(ylabel); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax7 = fig.add_subplot(gs[2, :])
ax7.set_title('Feature Separability Ranking (Higher = Better) — Calculated from Real Data',
              fontweight='bold', fontsize=12)
methods      = ['RMS\n(Weak)', 'TDOA\n(Raw)', 'IPD\n(Phase)', 'Spec\n(Energy)',
                'GCC-PHAT\n(Timing)', 'GCC Strength\n(Confidence)', 'Hybrid\n(Spec+IPD)']
separability = [rms_sep, tdoa_sep, ipd_sep, spec_sep, gcc_sep, gcc_strength_sep, hybrid_sep]
colors       = ['#E24B4A', '#378ADD', '#9F77DD', '#F9A825', '#1D9E75', '#FF6B9D', '#5B9BD5']
bars = ax7.barh(methods, separability, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
ax7.axvline(5, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Min Good')
ax7.axvline(8, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Excellent')
ax7.set_xlim([0, 10]); ax7.legend(loc='lower right', fontsize=10)
for bar, sep in zip(bars, separability):
    ax7.text(sep + 0.2, bar.get_y() + bar.get_height()/2, f'{sep:.1f}',
             va='center', fontweight='bold', fontsize=11)

plt.suptitle('Comprehensive Method Evaluation on All Speaker Positions\n(Scores Calculated from Real Audio Data)',
             fontsize=14, fontweight='bold', y=0.995)
plt.show()
