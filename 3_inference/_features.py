import numpy as np
import wave
from scipy.signal import hilbert

RATE          = 16000
CHUNK_SEC     = 0.05
CHUNK_SAMPLES = int(CHUNK_SEC * RATE)   # 480 samples per 30ms chunk
GCC_VEC_SIZE  = 100
N_PAIRS       = 6
N_MELS        = 40
N_FFT         = 1024
MICS          = ['mic_right', 'mic_front', 'mic_left', 'mic_back']

# ── Mel filterbank (built once) ────────────────────────────────────────────────
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


def _get_ipd(sig1, sig2):
    """Broadband IPD via Hilbert transform → scalar (degrees)."""
    a1 = np.angle(np.array(hilbert(sig1), dtype=np.complex128))
    a2 = np.angle(np.array(hilbert(sig2), dtype=np.complex128))
    return float(np.mean(np.degrees(np.arctan2(np.sin(a1 - a2), np.cos(a1 - a2)))))


def _get_ipd_mel(sig1, sig2):
    """Energy-weighted IPD per mel band → (N_MELS,)."""
    X1 = np.fft.rfft(sig1, n=N_FFT)
    X2 = np.fft.rfft(sig2, n=N_FFT)
    phase_diff = np.angle(X1 * np.conj(X2))
    power      = np.abs(X1) * np.abs(X2) + 1e-10
    weighted   = MEL_FB @ (phase_diff * power)
    weights    = MEL_FB @ power
    return (weighted / (weights + 1e-10)).astype(np.float32)


def _get_logmel(chunk):
    """Log-mel energy vector → (N_MELS,)."""
    power = np.abs(np.fft.rfft(chunk, n=N_FFT)) ** 2
    return np.log(MEL_FB @ power + 1e-10).astype(np.float32)


def _load_wav(path):
    with wave.open(path, 'rb') as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32)


def _rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))


def _get_gcc_phat(sig1, sig2):
    """Returns (tdoa_ms, strength)."""
    fft_len = 2 * len(sig1)
    X1  = np.fft.rfft(sig1, n=fft_len)
    X2  = np.fft.rfft(sig2, n=fft_len)
    Pxx = X1 * np.conj(X2)
    mag = np.abs(Pxx); mag[mag < 1e-10] = 1e-10
    gcc = np.fft.irfft(Pxx / mag, n=fft_len)[:len(sig1)]
    peak_idx = int(np.argmax(gcc))
    if peak_idx > len(gcc) // 2:
        peak_idx -= len(gcc)
    tdoa_ms  = float((peak_idx / RATE) * 1000)
    strength = float(np.max(gcc) / (np.std(gcc) + 1e-10))
    return tdoa_ms, strength


def _get_gcc_vector(sig1, sig2):
    """Returns normalised GCC-PHAT cross-correlation vector of length GCC_VEC_SIZE."""
    fft_len = 2 * len(sig1)
    X1  = np.fft.rfft(sig1, n=fft_len)
    X2  = np.fft.rfft(sig2, n=fft_len)
    Pxx = X1 * np.conj(X2)
    mag = np.abs(Pxx); mag[mag < 1e-10] = 1e-10
    gcc = np.fft.irfft(Pxx / mag, n=fft_len)
    gcc = np.roll(gcc, len(gcc) // 2)
    centre = len(gcc) // 2
    half   = GCC_VEC_SIZE // 2
    gcc    = gcc[centre - half : centre + half]
    mx = np.max(np.abs(gcc))
    if mx > 0:
        gcc = gcc / mx
    return gcc.astype(np.float32)


def extract_chunk_tdoa(ch):
    """
    Extract only what the GCC-TDOA model needs from one 30ms chunk.

    Parameters
    ----------
    ch : dict  { 'mic_right': np.float32[480], 'mic_front': ..., ... }

    Returns
    -------
    rms      : np.float32[4]    — one value per mic, used for silence detection
    features : np.float32[606]  — 6 TDOA values + 6×100 GCC vectors
    """
    rms = np.array([_rms(ch[m]) for m in MICS], dtype=np.float32)

    gcc_tdoa, gcc_vecs = [], []
    for i in range(4):
        for j in range(i + 1, 4):
            tdoa, _ = _get_gcc_phat(ch[MICS[i]], ch[MICS[j]])
            vec     = _get_gcc_vector(ch[MICS[i]], ch[MICS[j]])
            gcc_tdoa.append(tdoa)
            gcc_vecs.append(vec)

    features = np.concatenate([
        np.array(gcc_tdoa, dtype=np.float32),
        np.concatenate(gcc_vecs),
    ])
    return rms, features


def extract_chunk_all(ch):
    """
    Extract all 895 features matching train_ALL_features.py FEATURE_COLS order.

    Feature order:
      ipd_pair{0,1,2}                         →   3
      ipd_mel_{0,1,2}_b{0..39}                → 120
      gcc_tdoa_{0..5}                          →   6
      gcc_strength_{0..5}                      →   6
      gcc_vec_{0..5}_t{0..99}                  → 600
      logmel_{mic}_b{0..39}  (4 mics)          → 160
                                          Total: 895

    Parameters
    ----------
    ch : dict  { 'mic_right': np.float32[480], 'mic_front': ..., ... }

    Returns
    -------
    rms      : np.float32[4]    — one value per mic, used for silence detection
    features : np.float32[895]
    """
    rms = np.array([_rms(ch[m]) for m in MICS], dtype=np.float32)

    # IPD scalar — 3 pairs: (right,front), (right,left), (front,back)
    ipd_pairs = [
        _get_ipd(ch['mic_right'], ch['mic_front']),
        _get_ipd(ch['mic_right'], ch['mic_left']),
        _get_ipd(ch['mic_front'], ch['mic_back']),
    ]

    # IPD per mel-band — same 3 pairs × 40 bands
    ipd_mel = [
        _get_ipd_mel(ch['mic_right'], ch['mic_front']),
        _get_ipd_mel(ch['mic_right'], ch['mic_left']),
        _get_ipd_mel(ch['mic_front'], ch['mic_back']),
    ]

    # GCC-PHAT — all 6 pairs (i < j from MICS order)
    gcc_tdoa, gcc_str, gcc_vecs = [], [], []
    for i in range(4):
        for j in range(i + 1, 4):
            tdoa, strength = _get_gcc_phat(ch[MICS[i]], ch[MICS[j]])
            vec            = _get_gcc_vector(ch[MICS[i]], ch[MICS[j]])
            gcc_tdoa.append(tdoa)
            gcc_str.append(strength)
            gcc_vecs.append(vec)

    # Log-mel energy — 4 mics × 40 bands
    logmel = [_get_logmel(ch[m]) for m in MICS]

    features = np.concatenate([
        np.array(ipd_pairs, dtype=np.float32),        #   3
        np.concatenate(ipd_mel),                       # 120
        np.array(gcc_tdoa,  dtype=np.float32),         #   6
        np.array(gcc_str,   dtype=np.float32),         #   6
        np.concatenate(gcc_vecs),                      # 600
        np.concatenate(logmel),                        # 160
    ])
    return rms, features
