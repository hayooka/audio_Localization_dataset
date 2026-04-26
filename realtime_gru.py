"""
Real-time localization — SequenceGRU (16ms chunks, seq_len=32)
--------------------------------------------------------------
Usage:
    python realtime_gru.py
    python realtime_gru.py --model models/sequence/audioLOC_sequence_16ms_seq_gru_audio1_audio2.pt
    python realtime_gru.py --device 0
"""
import os, argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
from audioloc._features import extract_chunk_all, CHUNK_SAMPLES, RATE

REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(REPO_ROOT, 'models', 'sequence',
                             'audioLOC_sequence_16ms_seq_gru.pt')
ANGLES = list(range(0, 360, 15))

# ── Model ──────────────────────────────────────────────────────────────────────
class SequenceGRU(nn.Module):
    def __init__(self, n_features, n_classes, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_features, embed_dim),
            nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
    def forward(self, x):
        out, _ = self.gru(self.embed(x))
        return self.classifier(out[:, -1, :])

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

    print(f"Loading GRU model: {args.model}")
    ckpt    = torch.load(args.model, map_location=dev, weights_only=False)
    cfg     = ckpt['config']
    n_feat  = len(ckpt['feature_cols'])
    seq_len = cfg['seq_len']

    model = SequenceGRU(
        n_features = n_feat,
        n_classes  = len(ANGLES),
        embed_dim  = cfg['embed_dim'],
        hidden_dim = cfg['hidden_dim'],
        num_layers = cfg['layers'],
        dropout    = cfg['dropout'],
    ).to(dev)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    mean = ckpt['scaler_mean']
    std  = ckpt['scaler_std']

    d_info = sd.query_devices(device_index)
    print(f"Mic device  : [{device_index}] {d_info['name']}")
    print(f"Chunk       : {CHUNK_SAMPLES} samples = {CHUNK_SAMPLES/RATE*1000:.0f}ms")
    print(f"Sequence    : {seq_len} frames = {seq_len*CHUNK_SAMPLES/RATE*1000:.0f}ms context")
    print(f"RMS thresh  : {args.rms}")
    print(f"Running on  : {dev.upper()}")
    print("\nPress Ctrl+C to stop.\n")
    print(f"  {'Frame':>6}  {'Angle':>6}  {'Conf':>6}  {'Status'}")
    print(f"  {'-'*36}")

    buffer = np.zeros((seq_len, n_feat), dtype=np.float32)
    filled = 0
    frame  = 0

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

            # always update buffer to keep temporal context
            buffer = np.roll(buffer, -1, axis=0)
            buffer[-1] = feat
            filled = min(filled + 1, seq_len)
            frame += 1

            if (rms_vals < args.rms).any():
                print(f"  {frame:>6}  {'---':>6}  {'':>6}  silence")
                continue

            if filled < seq_len:
                print(f"  {frame:>6}  {'---':>6}  {'':>6}  warming {filled}/{seq_len}")
                continue

            seq_norm = (buffer - mean) / (std + 1e-10)
            X_t = torch.tensor(seq_norm).float().unsqueeze(0).to(dev)
            with torch.no_grad():
                probs = torch.softmax(model(X_t), dim=1).cpu().numpy()[0]

            pred_idx = int(probs.argmax())
            conf     = probs[pred_idx] * 100
            print(f"  {frame:>6}  {ANGLES[pred_idx]:>5}°  {conf:>5.1f}%")

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == '__main__':
    main()
