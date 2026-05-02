"""
SoundSense backend server
--------------------------
Serves soundsense_v4.html and pushes live JSON over WebSocket.

Architecture:
  [capture thread] — one PyAudio stream from ReSpeaker
       │
       ├──► gru_queue    ──► [GRU thread]     angle / confidence  (every 16ms)
       └──► yamnet_queue ──► [YAMNet thread]  sound label / scores (every ~1s)

Both inference threads run truly in parallel from the same audio stream.

Usage:
    python soundsense_server.py
    python soundsense_server.py --model audioLOC_GRU.pt --device 0 --port 8000
"""

import os, sys, argparse, threading, queue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # CPU only (RPi / no GPU needed)

import numpy as np
import pyaudio
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from _features import extract_chunk_all, CHUNK_SAMPLES, RATE

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE          = os.path.dirname(os.path.abspath(__file__))
HTML_PATH     = os.path.join(HERE, 'soundsense_v4.html')

GRU_MODEL_URL   = 'https://github.com/hayooka/audio_Localization_dataset/releases/download/v2.0/audioLOC_GRU.pt'
GRU_MODEL_CACHE = os.path.join(os.path.expanduser('~'), '.soundsense', 'audioLOC_GRU.pt')

def ensure_model(path: str, url: str) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f'Downloading GRU model from GitHub release...')
    from urllib.request import urlretrieve
    urlretrieve(url, path)
    print(f'Saved to {path}')
    return path

DEFAULT_MODEL = GRU_MODEL_CACHE

# ── Constants ──────────────────────────────────────────────────────────────────
N_CHANNELS      = 6
YAMNET_DURATION = 0.96
YAMNET_SAMPLES  = int(YAMNET_DURATION * RATE)   # 15 360 samples (~1 s)
# ReSpeaker XVF3800 channel layout (6-ch USB stream):
#   ch 0 — processed/mixed output  (not used)
#   ch 1 — mono reference mic      → YAMNet (sound recognition)
#   ch 2 — mic_right  ┐
#   ch 3 — mic_front  │ → GRU (DOA localization)
#   ch 4 — mic_left   │
#   ch 5 — mic_back   ┘
YAMNET_AUDIO_CH = 1
ANGLES          = list(range(0, 360, 15))

ALERT_KEYWORDS = {
    'siren', 'alarm', 'smoke detector', 'fire alarm', 'buzzer',
    'emergency vehicle', 'ambulance', 'fire engine', 'police car',
    'reversing beeps', 'civil defense',
}

def _is_alert(label: str) -> bool:
    low = label.lower()
    return any(kw in low for kw in ALERT_KEYWORDS)

# ── GRU model ──────────────────────────────────────────────────────────────────
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

# ── Shared output state ────────────────────────────────────────────────────────
_lock  = threading.Lock()
_state = {
    'angle':        0,
    'confidence':   0.0,
    'sound_label':  'Listening…',
    'sound_scores': [],
    'is_alert':     False,
    'alert_msg':    '',
    'alert_hint':   '',
    'transcript':   '',
}

def _update(**kwargs):
    with _lock:
        _state.update(kwargs)

def _snapshot():
    with _lock:
        return dict(_state)

# ── Device detection ───────────────────────────────────────────────────────────
def find_respeaker(p: pyaudio.PyAudio):
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if 'respeaker' in info['name'].lower() and int(info['maxInputChannels']) >= 6:
            return i
    return None

# ── Shared audio queues ────────────────────────────────────────────────────────
# Each queue receives raw 6-ch int16 numpy blocks from the capture thread.
# maxsize keeps memory bounded if an inference thread falls behind.
_gru_q    = queue.Queue(maxsize=50)
_yamnet_q = queue.Queue(maxsize=200)

# ── 1. Capture thread — ONE stream, feeds both queues ─────────────────────────
def capture_thread(device_index: int, stop_event: threading.Event):
    p      = pyaudio.PyAudio()
    stream = p.open(
        format             = pyaudio.paInt16,
        channels           = N_CHANNELS,
        rate               = RATE,
        input              = True,
        input_device_index = device_index,
        frames_per_buffer  = CHUNK_SAMPLES,   # 256 samples = 16ms
    )
    print('[Capture] Stream open.')
    try:
        while not stop_event.is_set():
            raw = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
            blk = np.frombuffer(raw, dtype=np.int16).reshape(-1, N_CHANNELS)
            # non-blocking puts: drop oldest if a consumer is too slow
            try: _gru_q.put_nowait(blk)
            except queue.Full: _gru_q.get_nowait(); _gru_q.put_nowait(blk)
            try: _yamnet_q.put_nowait(blk)
            except queue.Full: pass   # YAMNet can afford to skip a chunk
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print('[Capture] Stopped.')

# ── 2. GRU inference thread ────────────────────────────────────────────────────
def gru_thread(model_path: str, rms_thresh: float, stop_event: threading.Event):
    dev  = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = ensure_model(model_path, GRU_MODEL_URL)
    print(f'[GRU] Loading: {model_path}')
    ckpt    = torch.load(model_path, map_location=dev, weights_only=False)
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

    mean    = ckpt['scaler_mean']
    std     = ckpt['scaler_std']
    buf     = np.zeros((seq_len, n_feat), dtype=np.float32)
    filled  = 0
    print(f'[GRU] Ready — seq={seq_len} frames ({seq_len * CHUNK_SAMPLES / RATE * 1000:.0f}ms context)')

    while not stop_event.is_set():
        try:
            blk = _gru_q.get(timeout=0.5)
        except queue.Empty:
            continue

        blk_f = blk.astype('float32')
        ch = {
            'mic_right': blk_f[:, 2],
            'mic_front': blk_f[:, 3],
            'mic_left':  blk_f[:, 4],
            'mic_back':  blk_f[:, 5],
        }
        rms_vals, feat = extract_chunk_all(ch)

        buf    = np.roll(buf, -1, axis=0)
        buf[-1] = feat
        filled  = min(filled + 1, seq_len)

        if (rms_vals < rms_thresh).any() or filled < seq_len:
            continue

        seq_norm = (buf - mean) / (std + 1e-10)
        X_t = torch.tensor(seq_norm).float().unsqueeze(0).to(dev)
        with torch.no_grad():
            probs = torch.softmax(model(X_t), dim=1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        _update(angle=ANGLES[pred_idx], confidence=round(float(probs[pred_idx] * 100), 1))

    print('[GRU] Stopped.')

# ── 3. YAMNet inference thread ─────────────────────────────────────────────────
def yamnet_thread(stop_event: threading.Event):
    try:
        import tensorflow_hub as hub
    except ImportError:
        print('[YAMNet] tensorflow-hub not installed — skipping.')
        return

    print('[YAMNet] Loading from TF Hub…')
    yamnet         = hub.load('https://tfhub.dev/google/yamnet/1')
    class_map_path = yamnet.class_map_path().numpy()
    class_names    = []
    with open(class_map_path) as f:
        for line in f.readlines()[1:]:
            class_names.append(line.split(',')[2].strip())
    print(f'[YAMNet] Ready — {len(class_names)} classes')

    audio_buf = []

    while not stop_event.is_set():
        try:
            blk = _yamnet_q.get(timeout=0.5)
        except queue.Empty:
            continue

        ch1 = blk[:, YAMNET_AUDIO_CH].astype(np.float32) / 32768.0
        audio_buf.extend(ch1)

        if len(audio_buf) < YAMNET_SAMPLES:
            continue

        waveform  = np.array(audio_buf[:YAMNET_SAMPLES], dtype=np.float32)
        audio_buf = audio_buf[YAMNET_SAMPLES:]

        scores, _, _ = yamnet(waveform)
        mean_scores  = scores.numpy().mean(axis=0)
        top_idx      = np.argsort(mean_scores)[-5:][::-1]

        scores_out = [
            {'label': class_names[i], 'score': round(float(mean_scores[i]), 3)}
            for i in top_idx if mean_scores[i] > 0.05
        ]
        if not scores_out:
            continue

        top_label = scores_out[0]['label']
        alert     = _is_alert(top_label)
        _update(
            sound_label  = top_label,
            sound_scores = scores_out,
            is_alert     = alert,
            alert_msg    = f'Warning — {top_label} detected' if alert else '',
            alert_hint   = 'Check your surroundings immediately'  if alert else '',
        )

    print('[YAMNet] Stopped.')

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI()

@app.get('/', response_class=HTMLResponse)
async def index():
    with open(HTML_PATH, encoding='utf-8') as f:
        return f.read()

@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    import asyncio, json
    await ws.accept()
    try:
        while True:
            await asyncio.sleep(0.1)
            await ws.send_text(json.dumps(_snapshot()))
    except WebSocketDisconnect:
        pass

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  default=DEFAULT_MODEL)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--rms',    type=float, default=100.0)
    parser.add_argument('--host',   default='0.0.0.0')
    parser.add_argument('--port',   type=int, default=8000)
    args = parser.parse_args()

    if args.device is None:
        p = pyaudio.PyAudio()
        args.device = find_respeaker(p)
        p.terminate()
        if args.device is None:
            sys.exit('ReSpeaker not found. Plug it in or pass --device <index>.')
    print(f'Audio device index: {args.device}')

    stop = threading.Event()

    for target, targs, name in [
        (capture_thread, (args.device, stop),              'Capture'),
        (gru_thread,     (args.model, args.rms, stop),     'GRU'),
        (yamnet_thread,  (stop,),                          'YAMNet'),
    ]:
        threading.Thread(target=target, args=targs, name=name, daemon=True).start()

    print(f'\nOpen  http://<rpi-ip>:{args.port}  in any browser on the same Wi-Fi.\n')
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
    finally:
        stop.set()

if __name__ == '__main__':
    main()
