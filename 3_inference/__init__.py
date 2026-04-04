"""
audioloc — 4-microphone sound localization
-------------------------------------------
Install:
    pip install git+https://github.com/hayooka/audio_Localization_dataset.git

WAV files:
    from audioloc import predict
    angle, chunks = predict('mic_right.wav', 'mic_front.wav',
                            'mic_left.wav',  'mic_back.wav')

Real-time (live mic):
    from audioloc import predict_realtime
    predict_realtime()
"""

import os
import numpy as np
import torch
import torch.nn as nn
from urllib.request import urlretrieve

from ._features import extract_features, extract_chunk, MICS, CHUNK_SAMPLES, RATE, ALL_FEATURE_COLS

# RMS indices are always positions 0-3 in ALL_FEATURE_COLS
_RMS_IDX = [ALL_FEATURE_COLS.index(c)
            for c in ['rms_mic_right', 'rms_mic_front', 'rms_mic_left', 'rms_mic_back']]

_MODELS = {
    'all': {
        'url':   'https://github.com/hayooka/audio_Localization_dataset/releases/download/v1.0/audioLOC.pt',
        'cache': os.path.join(os.path.expanduser('~'), '.audioloc', 'audioLOC.pt'),
    },
    'gcctdoa': {
        'url':   'https://github.com/hayooka/audio_Localization_dataset/releases/download/v1.0/audioLOC_GCCTDOA.pt',
        'cache': os.path.join(os.path.expanduser('~'), '.audioloc', 'audioLOC_GCCTDOA.pt'),
    },
}

# ── Model ─────────────────────────────────────────────────────────────────────
class _LocalizationCNN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# ── Model cache (one entry per model name) ────────────────────────────────────
_states = {}   # { 'all': {...}, 'gcctdoa': {...} }

def _load(model_name='all'):
    if model_name not in _MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(_MODELS)}")
    cfg    = _MODELS[model_name]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(cfg['cache']):
        os.makedirs(os.path.dirname(cfg['cache']), exist_ok=True)
        print(f'Downloading {model_name} weights...')
        urlretrieve(cfg['url'], cfg['cache'])
        print('Download complete.')
    ckpt = torch.load(cfg['cache'], map_location=device, weights_only=False)
    feature_cols = ckpt['feature_cols']
    n_classes    = ckpt['model_state']['fc.9.bias'].shape[0]
    model = _LocalizationCNN(n_features=len(feature_cols), n_classes=n_classes).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    col_idx = [ALL_FEATURE_COLS.index(c) for c in feature_cols]
    _states[model_name] = dict(
        model        = model,
        scaler_mean  = ckpt['scaler_mean'],
        scaler_std   = ckpt['scaler_std'],
        col_idx      = col_idx,
        angles       = list(range(0, 360, 360 // n_classes)),
        device       = device,
    )

def _infer(X, model_name):
    """Select model features, normalize, run inference. X is full 899-feature array."""
    st          = _states[model_name]
    model       = st['model']
    scaler_mean = st['scaler_mean']
    scaler_std  = st['scaler_std']
    col_idx     = st['col_idx']
    angles      = st['angles']
    device      = st['device']
    X_sel  = X[:, col_idx]
    X_norm = (X_sel - scaler_mean) / (scaler_std + 1e-10)
    X_t    = torch.tensor(X_norm).float().unsqueeze(1).to(device)
    with torch.no_grad():
        preds = model(X_t).argmax(1).cpu().numpy()
    return [angles[p] for p in preds], preds

# ── 1. WAV file prediction ─────────────────────────────────────────────────────
def predict(path_right, path_front, path_left, path_back,
            rms_threshold=100.0, max_seconds=5, model='gcctdoa'):
    """
    Predict sound direction from 4 pre-recorded WAV files (16 kHz, mono).

    Parameters
    ----------
    path_right / path_front / path_left / path_back : str
        Path to each mic's WAV file.
    rms_threshold : float
        Chunks below this RMS on any mic are skipped (silence removal).
    max_seconds : float or None
        Only process this many seconds from the start (default 5).
        Pass None to process the entire file.
    model : str
        Which model to use: 'all' (default, all features) or 'gcctdoa'.

    Returns
    -------
    angle : int
        Majority-vote direction in degrees (0, 15, 30, ..., 345).
    per_chunk : list[int]
        Per-chunk angle predictions.
    """
    if model not in _states:
        _load(model)

    X = extract_features(path_right, path_front, path_left, path_back,
                         max_seconds=max_seconds)

    mask = (X[:, _RMS_IDX] >= rms_threshold).all(axis=1)
    X = X[mask]
    if len(X) == 0:
        return 0, []

    per_chunk, preds = _infer(X, model)
    angles  = _states[model]['angles']
    majority = angles[int(np.argmax(np.bincount(preds, minlength=len(angles))))]
    return majority, per_chunk


# ── 2. Real-time prediction ────────────────────────────────────────────────────
def predict_realtime(device=None, rms_threshold=100.0):
    """
    Predict sound direction live from a 4-channel microphone (e.g. ReSpeaker).
    Prints the predicted angle every 30ms. Press Ctrl+C to stop.

    Parameters
    ----------
    device : int or str or None
        sounddevice input device. None = system default.
        Run `python -m sounddevice` to list available devices.
    rms_threshold : float
        Chunks below this RMS on any channel are shown as '---' (silence).
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError(
            "Real-time mode requires sounddevice.\n"
            "Install it with:  pip install sounddevice"
        )

    if not _state:
        _load()

    angles      = _state['angles']
    feature_cols = _state['feature_cols']
    rms_names   = ['rms_mic_right', 'rms_mic_front', 'rms_mic_left', 'rms_mic_back']
    rms_idx     = [feature_cols.index(c) for c in rms_names]

    print("Listening... Press Ctrl+C to stop.\n")
    print(f"{'Angle':>8}  {'Confidence':>10}")
    print("-" * 22)

    try:
        with sd.InputStream(device=device, channels=6,
                            samplerate=RATE, blocksize=CHUNK_SAMPLES,
                            dtype='int16') as stream:
            while True:
                block, _ = stream.read(CHUNK_SAMPLES)   # shape: (CHUNK_SAMPLES, 6)
                block = block.astype('float32')          # ch 0-1 unused (ReSpeaker raw)

                ch = {
                    'mic_right': block[:, 2],
                    'mic_front': block[:, 3],
                    'mic_left':  block[:, 4],
                    'mic_back':  block[:, 5],
                }

                feat = extract_chunk(ch).reshape(1, -1)

                # silence check
                rms_vals = feat[0, rms_idx]
                if (rms_vals < rms_threshold).any():
                    print(f"{'---':>8}  (silence)")
                    continue

                per_chunk, preds = _infer(feat)
                angle = per_chunk[0]
                # confidence: softmax probability of top class
                X_norm = (feat - _state['scaler_mean']) / (_state['scaler_std'] + 1e-10)
                X_t    = torch.tensor(X_norm).float().unsqueeze(1).to(_state['device'])
                with torch.no_grad():
                    probs = torch.softmax(_state['model'](X_t), dim=1).cpu().numpy()[0]
                conf = probs[preds[0]]
                print(f"{angle:>7}°  {conf*100:>9.1f}%")

    except KeyboardInterrupt:
        print("\nStopped.")
