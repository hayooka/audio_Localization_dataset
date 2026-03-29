"""
Real-time Sound Localization Inference
Uses ReSpeaker 4-mic array -> extracts TDOA/IPD/GCC features -> predicts angle

Run: python realtime_inference.py
"""

import pyaudio
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import hilbert
import sys
import os

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = r'C:\Users\ahmma\Desktop\farah\audio_Localization_dataset\training\model_E1.pt'
RATE         = 16000
CHUNK_SEC    = 2
CHUNK        = RATE * CHUNK_SEC
CHANNELS     = 6        # ReSpeaker 4-mic UAC1.0: 6 channels (use ch 0-3)
MIC_CHANNELS = [0, 1, 2, 3]  # right, front, left, back
ANGLES       = [0, 45, 90, 135, 180, 225, 270, 315]

FEATURE_COLS = [
    'rms_mic_right', 'rms_mic_front', 'rms_mic_left', 'rms_mic_back',
    'ipd_pair0', 'ipd_pair1', 'ipd_pair2',
    'gcc_tdoa_0', 'gcc_tdoa_1', 'gcc_tdoa_2',
    'gcc_tdoa_3', 'gcc_tdoa_4', 'gcc_tdoa_5',
    'gcc_strength_0', 'gcc_strength_1', 'gcc_strength_2',
    'gcc_strength_3', 'gcc_strength_4', 'gcc_strength_5',
]

# ── Model definition (must match training) ────────────────────────────────────
class LocalizationCNN(nn.Module):
    def __init__(self, n_features=19, n_classes=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * n_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# ── Feature extraction (identical to training pipeline) ───────────────────────
MICS = ['mic_right', 'mic_front', 'mic_left', 'mic_back']

def rms(s):
    return float(np.sqrt(np.mean(s ** 2)))

def get_ipd(s1, s2):
    a1 = np.angle(hilbert(s1))
    a2 = np.angle(hilbert(s2))
    return float(np.mean(np.degrees(np.arctan2(np.sin(a1 - a2), np.cos(a1 - a2)))))

def get_gcc_phat(s1, s2):
    fft_len = 2 * len(s1)
    X1  = np.fft.rfft(s1, n=fft_len)
    X2  = np.fft.rfft(s2, n=fft_len)
    Pxx = X1 * np.conj(X2)
    mag = np.abs(Pxx); mag[mag < 1e-10] = 1e-10
    gcc = np.fft.irfft(Pxx / mag, n=fft_len)[:len(s1)]
    peak = int(np.argmax(gcc))
    if peak > len(gcc) // 2: peak -= len(gcc)
    tdoa_ms  = float((peak / RATE) * 1000)
    strength = float(np.max(gcc) / (np.std(gcc) + 1e-10))
    return tdoa_ms, strength

def extract_features(signals):
    ch = {MICS[i]: signals[i] for i in range(4)}
    feats = (
        [rms(ch[m]) for m in MICS] +
        [get_ipd(ch['mic_right'], ch['mic_front']),
         get_ipd(ch['mic_right'], ch['mic_left']),
         get_ipd(ch['mic_front'], ch['mic_back'])] +
        [v for i in range(4) for j in range(i+1, 4)
           for v in get_gcc_phat(ch[MICS[i]], ch[MICS[j]])]
    )
    return np.array(feats, dtype=np.float32)

# ── Find ReSpeaker device ──────────────────────────────────────────────────────
def find_respeaker(p):
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        name = d['name'].lower()
        if d['maxInputChannels'] >= 6 and ('respeaker' in name or 'uac' in name):
            return i, d
    return None, None

def list_devices(p):
    print('\nAvailable input devices:')
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d['maxInputChannels'] > 0:
            print(f'  [{i}] {d["maxInputChannels"]}ch  {int(d["defaultSampleRate"])}Hz  {d["name"]}')

# ── Load model ────────────────────────────────────────────────────────────────
print('Loading model...')
checkpoint   = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model        = LocalizationCNN()
model.load_state_dict(checkpoint['model_state'])
model.eval()
scaler_mean  = checkpoint['scaler_mean']
scaler_std   = checkpoint['scaler_std']
print('Model loaded.')

# ── Open audio stream ─────────────────────────────────────────────────────────
p = pyaudio.PyAudio()
dev_idx, dev_info = find_respeaker(p)

if dev_idx is None:
    print('\nReSpeaker not found. Plug in the device and retry.')
    list_devices(p)
    p.terminate()
    sys.exit(1)

print(f'Using device [{dev_idx}]: {dev_info["name"]}')

stream = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=dev_idx,
    frames_per_buffer=CHUNK,
)

print(f'\nListening... (Ctrl+C to stop)\n')

# ── Inference loop ────────────────────────────────────────────────────────────
try:
    while True:
        raw  = stream.read(CHUNK, exception_on_overflow=False)
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        # Deinterleave 6 channels, keep first 4
        multi = data.reshape(-1, CHANNELS)
        signals = [multi[:, ch] / 32768.0 for ch in MIC_CHANNELS]

        feats = extract_features(signals)
        feats = (feats - scaler_mean) / (scaler_std + 1e-10)

        x = torch.tensor(feats).unsqueeze(0).unsqueeze(0)  # (1, 1, 19)
        with torch.no_grad():
            logits = model(x)
            probs  = torch.softmax(logits, dim=1).squeeze().numpy()
            pred   = int(np.argmax(probs))

        angle      = ANGLES[pred]
        confidence = probs[pred] * 100
        bar        = '#' * int(confidence / 5)

        print(f'\rPredicted: {angle:>3}deg  ({confidence:>5.1f}%)  {bar:<20}', end='', flush=True)

except KeyboardInterrupt:
    print('\nStopped.')

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
