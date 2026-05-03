"""
SoundSense backend server  –  v2 (dual-mic YAMNet + STT auto-clear)
--------------------------------------------------------------------
Architecture:
  [capture thread] — one PyAudio stream from ReSpeaker
       │
       ├──► gru_queue    ──► [GRU thread]     angle / confidence  (every 16ms)
       ├──► yamnet_queue ──► [YAMNet thread]  dual-mic agreement filter (~1 s)
       └──► stt_queue    ──► [STT thread]     transcript (1.2s chunks, 0.3s overlap)

KEY CHANGES vs v1
─────────────────
1.  Dual-mic YAMNet agreement filter
    • Reads ch 1 (reference mic) AND ch 2 (mic_right) in parallel.
    • Runs YAMNet inference on each buffer independently.
    • Accepts a label only when BOTH mics agree AND both confidence
      scores exceed YAMNET_AGREE_THRESH (default 0.25).
    • Relaxed matching: labels whose class-map indices are within
      YAMNET_INDEX_TOLERANCE of each other are treated as the same
      (reduces false negatives for similar sounds, e.g. "Speech" vs
      "Conversation").
    • On disagreement → outputs "Silence" so the UI stays clean.
    • Weighted fusion: final scores = mean(mic1_scores, mic2_scores)
      so the confidence bars reflect both mics.

2.  STT auto-clear timestamp
    • Backend now includes `transcript_ts` (Unix timestamp) in every
      WebSocket push so the frontend can detect silence gaps.
    • Frontend clears the transcript after 7 s of no new speech.

3.  All other threads (capture, GRU, STT) are unchanged.

Usage:
    python soundsense_server.py
    python soundsense_server.py --model audioLOC_GRU.pt --device 0 --port 8000
"""

import os, sys, argparse, threading, queue, time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES']  = '-1'   # CPU only (RPi / no GPU needed)

import numpy as np
import pyaudio
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from _features import extract_chunk_all, CHUNK_SAMPLES, RATE

# ── Paths ───────────────────────────────────────────────────────────────────────
HERE          = os.path.dirname(os.path.abspath(__file__))
HTML_PATH     = os.path.join(HERE, 'soundsense_v5.html')   # updated HTML filename

GRU_MODEL_URL   = 'https://github.com/hayooka/audio_Localization_dataset/releases/download/v2.0/audioLOC_GRU.pt'
GRU_MODEL_CACHE = os.path.join(os.path.expanduser('~'), '.soundsense', 'audioLOC_GRU.pt')

def ensure_model(path: str, url: str) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f'[GRU] Downloading model from GitHub release …')
    from urllib.request import urlretrieve
    urlretrieve(url, path)
    print(f'[GRU] Saved to {path}')
    return path

DEFAULT_MODEL = GRU_MODEL_CACHE

# ── Constants ───────────────────────────────────────────────────────────────────
N_CHANNELS      = 6
YAMNET_DURATION = 0.96
YAMNET_SAMPLES  = int(YAMNET_DURATION * RATE)   # 15 360 samples (~1 s)

# ReSpeaker XVF3800 channel layout (6-ch USB stream):
#   ch 0 — processed/mixed output  (not used)
#   ch 1 — mono reference mic      → YAMNet mic A  + STT
#   ch 2 — mic_right               → YAMNet mic B  (second independent channel)
#   ch 3 — mic_front  ┐
#   ch 4 — mic_left   │ → GRU (DOA localization)
#   ch 5 — mic_back   ┘
YAMNET_CH_A = 1   # reference mic
YAMNET_CH_B = 2   # directional mic (mic_right, physically separated)

# ── Dual-mic agreement parameters ──────────────────────────────────────────────
YAMNET_AGREE_THRESH  = 0.25   # both mics must exceed this confidence
YAMNET_INDEX_TOLERANCE = 3    # class-map indices within ±3 are treated as equal
                               # (catches "Speech" / "Conversation" / "Male speech"
                               #  which sit adjacent in YAMNet's 521-class map)
SILENCE_LABEL = 'Silence'

ANGLES = list(range(0, 360, 15))

ALERT_KEYWORDS = {
    'siren', 'alarm', 'smoke detector', 'fire alarm', 'buzzer',
    'emergency vehicle', 'ambulance', 'fire engine', 'police car',
    'reversing beeps', 'civil defense',
}

def _is_alert(label: str) -> bool:
    low = label.lower()
    return any(kw in low for kw in ALERT_KEYWORDS)

# ── GRU model ───────────────────────────────────────────────────────────────────
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

# ── Shared output state ─────────────────────────────────────────────────────────
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
    'transcript_ts': 0.0,   # NEW: Unix timestamp of last STT update
}

def _update(**kwargs):
    with _lock:
        _state.update(kwargs)

def _snapshot():
    with _lock:
        return dict(_state)

# ── Device detection ────────────────────────────────────────────────────────────
def find_respeaker(p: pyaudio.PyAudio):
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if 'respeaker' in info['name'].lower() and int(info['maxInputChannels']) >= 6:
            return i
    return None

# ── Shared audio queues ─────────────────────────────────────────────────────────
_gru_q    = queue.Queue(maxsize=50)
_yamnet_q = queue.Queue(maxsize=200)
_stt_q    = queue.Queue(maxsize=200)

# ── 1. Capture thread ───────────────────────────────────────────────────────────
def capture_thread(device_index: int, stop_event: threading.Event):
    p      = pyaudio.PyAudio()
    stream = p.open(
        format             = pyaudio.paInt16,
        channels           = N_CHANNELS,
        rate               = RATE,
        input              = True,
        input_device_index = device_index,
        frames_per_buffer  = CHUNK_SAMPLES,
    )
    print('[Capture] Stream open.')
    try:
        while not stop_event.is_set():
            raw = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
            blk = np.frombuffer(raw, dtype=np.int16).reshape(-1, N_CHANNELS)
            try: _gru_q.put_nowait(blk)
            except queue.Full: _gru_q.get_nowait(); _gru_q.put_nowait(blk)
            try: _yamnet_q.put_nowait(blk)
            except queue.Full: pass
            try: _stt_q.put_nowait(blk)
            except queue.Full: pass
    finally:
        stream.stop_stream(); stream.close(); p.terminate()
        print('[Capture] Stopped.')

# ── 2. GRU inference thread  (unchanged) ────────────────────────────────────────
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

    mean   = ckpt['scaler_mean']
    std    = ckpt['scaler_std']
    buf    = np.zeros((seq_len, n_feat), dtype=np.float32)
    filled = 0
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

# ── 3. YAMNet dual-mic inference thread  (NEW) ─────────────────────────────────
def yamnet_thread(stop_event: threading.Event):
    """
    Runs YAMNet on two independent microphone channels in parallel.

    Agreement rule
    ──────────────
    • Both mics must produce the SAME top-1 label (or labels whose
      class indices are within YAMNET_INDEX_TOLERANCE of each other).
    • Both top-1 confidence scores must exceed YAMNET_AGREE_THRESH.
    • If either condition fails → output SILENCE_LABEL so the UI stays
      clean and no incorrect label is displayed.

    Fusion
    ──────
    When both mics agree, the final per-class scores sent to the UI are
    the element-wise mean of both mics' score vectors.  This gives more
    stable confidence bars than using only one mic.
    """
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
    n_classes = len(class_names)
    print(f'[YAMNet] Ready — {n_classes} classes | '
          f'agree_thresh={YAMNET_AGREE_THRESH} | '
          f'index_tolerance={YAMNET_INDEX_TOLERANCE}')

    buf_a = []   # reference mic  (ch 1)
    buf_b = []   # directional mic (ch 2)

    while not stop_event.is_set():
        try:
            blk = _yamnet_q.get(timeout=0.5)
        except queue.Empty:
            continue

        # Accumulate both channels simultaneously
        buf_a.extend(blk[:, YAMNET_CH_A].astype(np.float32) / 32768.0)
        buf_b.extend(blk[:, YAMNET_CH_B].astype(np.float32) / 32768.0)

        # Wait until we have a full inference window for both
        if len(buf_a) < YAMNET_SAMPLES or len(buf_b) < YAMNET_SAMPLES:
            continue

        # Consume one window from each buffer (non-overlapping for stability)
        wav_a = np.array(buf_a[:YAMNET_SAMPLES], dtype=np.float32)
        wav_b = np.array(buf_b[:YAMNET_SAMPLES], dtype=np.float32)
        buf_a = buf_a[YAMNET_SAMPLES:]
        buf_b = buf_b[YAMNET_SAMPLES:]

        # ── Independent inference ──────────────────────────────────────────────
        scores_a, _, _ = yamnet(wav_a)
        scores_b, _, _ = yamnet(wav_b)

        mean_a = scores_a.numpy().mean(axis=0)   # shape (n_classes,)
        mean_b = scores_b.numpy().mean(axis=0)

        top_idx_a = int(np.argmax(mean_a))
        top_idx_b = int(np.argmax(mean_b))
        top_conf_a = float(mean_a[top_idx_a])
        top_conf_b = float(mean_b[top_idx_b])

        # ── Agreement check ────────────────────────────────────────────────────
        # Relaxed: indices within tolerance are treated as matching
        indices_agree = abs(top_idx_a - top_idx_b) <= YAMNET_INDEX_TOLERANCE
        conf_ok       = (top_conf_a >= YAMNET_AGREE_THRESH and
                         top_conf_b >= YAMNET_AGREE_THRESH)

        if not (indices_agree and conf_ok):
            # Disagreement or low confidence — output silence
            _update(
                sound_label  = SILENCE_LABEL,
                sound_scores = [],
                is_alert     = False,
                alert_msg    = '',
                alert_hint   = '',
            )
            continue

        # ── Fuse scores (mean of both mics) ────────────────────────────────────
        fused = (mean_a + mean_b) / 2.0

        # Pick the agreed-upon label (use mic A's top index as reference,
        # then re-rank by fused scores among top-5)
        top_indices = np.argsort(fused)[-5:][::-1]

        scores_out = [
            {'label': class_names[i], 'score': round(float(fused[i]), 3)}
            for i in top_indices
            if fused[i] > 0.05
        ]
        if not scores_out:
            continue

        top_label = scores_out[0]['label']
        alert     = _is_alert(top_label)

        print(f'[YAMNet] AGREED  {top_label!r}  '
              f'A={top_conf_a:.2f}  B={top_conf_b:.2f}  '
              f'idx_A={top_idx_a}  idx_B={top_idx_b}')

        _update(
            sound_label  = top_label,
            sound_scores = scores_out,
            is_alert     = alert,
            alert_msg    = f'Warning — {top_label} detected' if alert else '',
            alert_hint   = 'Check your surroundings immediately'  if alert else '',
        )

    print('[YAMNet] Stopped.')

# ── 4. STT thread  (google-cloud-speech streaming) ─────────────────────────────
STT_LANG = 'ar-KW'   # change to 'en-US' for English

def _audio_generator(stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            blk = _stt_q.get(timeout=0.5)
        except queue.Empty:
            continue
        pcm = blk[:, YAMNET_CH_A].astype(np.int16).tobytes()
        yield pcm

def stt_thread(stop_event: threading.Event):
    try:
        from google.cloud import speech as gc_speech
    except ImportError:
        print('[STT] google-cloud-speech not installed — skipping.')
        return

    print(f'[STT] Ready — language={STT_LANG}  (google-cloud-speech streaming)')

    config = gc_speech.RecognitionConfig(
        encoding=gc_speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=STT_LANG,
        enable_automatic_punctuation=True,
    )
    streaming_config = gc_speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    client     = gc_speech.SpeechClient()
    transcript = ''

    def _make_requests():
        for pcm in _audio_generator(stop_event):
            yield gc_speech.StreamingRecognizeRequest(audio_content=pcm)

    try:
        responses = client.streaming_recognize(streaming_config, _make_requests())
        for response in responses:
            if stop_event.is_set():
                break
            for result in response.results:
                text = result.alternatives[0].transcript.strip()
                if not text:
                    continue
                if result.is_final:
                    transcript = (transcript + ' ' + text).strip()
                    if len(transcript) > 200:
                        transcript = transcript[-200:]
                    _update(transcript=transcript, transcript_ts=time.time())
                else:
                    _update(transcript=(transcript + ' ' + text).strip(),
                            transcript_ts=time.time())
    except Exception as e:
        print(f'[STT] Error: {e}')

    print('[STT] Stopped.')

# ── FastAPI ─────────────────────────────────────────────────────────────────────
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

# ── Entry point ─────────────────────────────────────────────────────────────────
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
        (stt_thread,     (stop,),                          'STT'),
    ]:
        threading.Thread(target=target, args=targs, name=name, daemon=True).start()

    print(f'\nOpen  http://<rpi-ip>:{args.port}  in any browser on the same Wi-Fi.\n')
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
    finally:
        stop.set()

if __name__ == '__main__':
    main()
