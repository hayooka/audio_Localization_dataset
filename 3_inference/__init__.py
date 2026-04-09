"""
audioloc — 4-microphone sound localization (ALL features)
----------------------------------------------------------
Install:
    pip install git+https://github.com/hayooka/audio_Localization_dataset.git

WAV files:
    from audioloc import predict
    angle, per_chunk = predict('mic_right.wav', 'mic_front.wav',
                               'mic_left.wav',  'mic_back.wav')

Real-time (ReSpeaker USB — PC or Raspberry Pi):
    from audioloc import predict_realtime
    predict_realtime()
"""

import os
import numpy as np
import torch
import torch.nn as nn
from urllib.request import urlretrieve

from ._features import _load_wav, extract_chunk_all, CHUNK_SAMPLES, RATE, MICS

# ── Constants ──────────────────────────────────────────────────────────────────
DEVICE_NAME = "reSpeaker"   # partial name match — same as data collection scripts
MODEL_URL   = 'https://github.com/hayooka/audio_Localization_dataset/releases/download/v1.01/audioLOC_ALL.pt'
MODEL_CACHE = os.path.join(os.path.expanduser('~'), '.audioloc', 'audioLOC_ALL.pt')
MODEL_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '4_training', 'audioLOC.pt')

# ── CNN architecture (must match training) ─────────────────────────────────────
class _LocalizationCNN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 64),               nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# ── Model state (loaded once on first call) ────────────────────────────────────
_state = {}

def _load():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # prefer local trained file, fall back to cached download
    if os.path.exists(MODEL_LOCAL):
        ckpt_path = MODEL_LOCAL
    else:
        ckpt_path = MODEL_CACHE
        if not os.path.exists(ckpt_path):
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            print('Downloading ALL-features model...')
            urlretrieve(MODEL_URL, ckpt_path)
            print('Done.')

    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_feat    = len(ckpt['feature_cols'])
    n_classes = ckpt['model_state']['fc.9.bias'].shape[0]
    model     = _LocalizationCNN(n_feat, n_classes).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    _state.update(
        model  = model,
        mean   = ckpt['scaler_mean'],
        std    = ckpt['scaler_std'],
        angles = list(range(0, 360, 360 // n_classes)),
        device = device,
    )

# ── Inference ──────────────────────────────────────────────────────────────────
def _infer(X):
    """X: all features."""
    X_norm = (X - _state['mean']) / (_state['std'] + 1e-10)
    X_t    = torch.tensor(X_norm).float().unsqueeze(1).to(_state['device'])
    with torch.no_grad():
        probs = torch.softmax(_state['model'](X_t), dim=1).cpu().numpy()
    preds = probs.argmax(1)
    confs = probs[range(len(preds)), preds]
    return [_state['angles'][p] for p in preds], preds, confs

# ── ReSpeaker auto-detection (same logic as data collection scripts) ────────────
def _find_respeaker(sd):
    for i, dev in enumerate(sd.query_devices()):
        if DEVICE_NAME.lower() in str(dev['name']).lower() and dev['max_input_channels'] >= 6:
            return i
    return None

# ── 1. WAV prediction ──────────────────────────────────────────────────────────
def predict(path_right, path_front, path_left, path_back,
            rms_threshold=100.0, max_seconds=5):
    """
    Predict direction from 4 pre-recorded WAV files (16 kHz mono).

    Returns
    -------
    majority_angle : int   — degrees (0, 15, 30, ..., 345)
    per_chunk      : list  — per-30ms-chunk predictions
    """
    if not _state:
        _load()

    signals = {
        'mic_right': _load_wav(path_right),
        'mic_front': _load_wav(path_front),
        'mic_left':  _load_wav(path_left),
        'mic_back':  _load_wav(path_back),
    }
    n_chunks = min(len(s) for s in signals.values()) // CHUNK_SAMPLES
    if max_seconds is not None:
        n_chunks = min(n_chunks, int(max_seconds * RATE / CHUNK_SAMPLES))

    rms_list, feat_list = [], []
    for ci in range(n_chunks):
        s  = ci * CHUNK_SAMPLES
        ch = {m: signals[m][s:s + CHUNK_SAMPLES] for m in MICS}
        rms, feat = extract_chunk_all(ch)
        rms_list.append(rms)
        feat_list.append(feat)

    if not feat_list:
        return 0, []

    rms_arr  = np.stack(rms_list)
    feat_arr = np.stack(feat_list)
    feat_arr = feat_arr[(rms_arr >= rms_threshold).all(axis=1)]

    if len(feat_arr) == 0:
        return 0, []

    per_chunk, preds, _ = _infer(feat_arr)
    angles   = _state['angles']
    majority = angles[int(np.argmax(np.bincount(preds, minlength=len(angles))))]
    return majority, per_chunk

# ── 2. Real-time prediction ────────────────────────────────────────────────────
def predict_realtime(rms_threshold=100.0, device_index=None):
    """
    Predict live from ReSpeaker USB mic array.
    Auto-detects device by name, or pass device_index explicitly.
    Press Ctrl+C to stop.
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError(
            "Real-time mode requires sounddevice.\n"
            "Install it with:  pip install sounddevice"
        )

    if device_index is None:
        device_index = _find_respeaker(sd)
        if device_index is None:
            raise RuntimeError(
                "ReSpeaker not found. Make sure it is plugged in.\n"
                "Run `python -m sounddevice` to list available devices."
            )
    print(f"Using device index {device_index}")

    if not _state:
        _load()

    print("Listening (ALL features)... Press Ctrl+C to stop.\n")
    print(f"{'Angle':>8}  {'Confidence':>10}")
    print("-" * 22)

    try:
        while True:
            block = sd.rec(CHUNK_SAMPLES, samplerate=RATE, channels=6,
                           dtype='int16', device=device_index)
            sd.wait()
            block = block.astype('float32')

            # ReSpeaker USB: mics are on channels 2-5
            ch = {
                'mic_right': block[:, 2],
                'mic_front': block[:, 3],
                'mic_left':  block[:, 4],
                'mic_back':  block[:, 5],
            }

            rms, feat = extract_chunk_all(ch)

            if (rms < rms_threshold).any():
                print(f"{'---':>8}  (silence)")
                continue

            per_chunk, _, confs = _infer(feat.reshape(1, -1))
            print(f"{per_chunk[0]:>7}°  {confs[0]*100:>9.1f}%")

    except KeyboardInterrupt:
        print("\nStopped.")
