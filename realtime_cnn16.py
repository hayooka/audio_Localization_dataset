"""
Real-time localization — CNN with 16ms chunks
---------------------------------------------
Usage:
    python realtime_cnn16.py
    python realtime_cnn16.py --model 4_training/audioLOC.pt
    python realtime_cnn16.py --device 0
"""
import os, argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
from audioloc._features import extract_chunk_all, CHUNK_SAMPLES, RATE

REPO_ROOT     = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(REPO_ROOT, '4_training', 'audioLOC.pt')
ANGLES        = list(range(0, 360, 15))

# ── Model ──────────────────────────────────────────────────────────────────────
class LocalizationCNN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32,  kernel_size=3, padding=1),
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

# ── Device selection ───────────────────────────────────────────────────────────
def find_device():
    for i, d in enumerate(sd.query_devices()):
        if 'respeaker' in d['name'].lower() and d['max_input_channels'] >= 6:
            return i
    return sd.default.device[0]

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  default=DEFAULT_MODEL)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--rms',    type=float, default=100.0)
    args = parser.parse_args()

    device_index = args.device if args.device is not None else find_device()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading CNN model: {args.model}")
    ckpt   = torch.load(args.model, map_location=dev, weights_only=False)
    n_feat = len(ckpt['feature_cols'])

    model = LocalizationCNN(n_feat, len(ANGLES)).to(dev)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    mean = ckpt['scaler_mean']
    std  = ckpt['scaler_std']

    d_info = sd.query_devices(device_index)
    print(f"Mic device  : [{device_index}] {d_info['name']}")
    print(f"Chunk       : {CHUNK_SAMPLES} samples = {CHUNK_SAMPLES/RATE*1000:.0f}ms")
    print(f"RMS thresh  : {args.rms}")
    print(f"Running on  : {dev.upper()}")
    print("\nPress Ctrl+C to stop.\n")
    print(f"  {'Frame':>6}  {'Angle':>6}  {'Conf':>6}")
    print(f"  {'-'*24}")

    frame = 0

    try:
        while True:
            block = sd.rec(CHUNK_SAMPLES, samplerate=RATE,
                           channels=6, dtype='int16', device=device_index)
            sd.wait()
            block = block.astype('float32')

            ch = {
                'mic_right': block[:, 2],
                'mic_front': block[:, 3],
                'mic_left':  block[:, 4],
                'mic_back':  block[:, 5],
            }

            rms_vals, feat = extract_chunk_all(ch)
            frame += 1

            if (rms_vals < args.rms).any():
                print(f"  {frame:>6}  {'---':>6}  (silence)")
                continue

            feat_norm = (feat - mean) / (std + 1e-10)
            X_t = torch.tensor(feat_norm).float().unsqueeze(0).unsqueeze(0).to(dev)
            with torch.no_grad():
                probs = torch.softmax(model(X_t), dim=1).cpu().numpy()[0]

            pred_idx = int(probs.argmax())
            conf     = probs[pred_idx] * 100
            print(f"  {frame:>6}  {ANGLES[pred_idx]:>5}°  {conf:>5.1f}%")

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == '__main__':
    main()

