import numpy as np
import wave
from scipy.signal import hilbert

RATE          = 16000
CHUNK_SEC     = 0.03
N_MELS        = 40
N_FFT         = 1024
GCC_VEC_SIZE  = 100
MICS          = ['mic_right', 'mic_front', 'mic_left', 'mic_back']
CHUNK_SAMPLES = int(CHUNK_SEC * RATE)

def _build_mel_filterbank():
    def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)
    mel_pts = np.linspace(hz_to_mel(0), hz_to_mel(RATE / 2), N_MELS + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.floor((N_FFT + 1) * hz_pts / RATE).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1))
    for m in range(1, N_MELS + 1):
        l, c, r = bins[m-1], bins[m], bins[m+1]
        fb[m-1, l:c] = (np.arange(l, c) - l) / (c - l + 1e-10)
        fb[m-1, c:r] = (r - np.arange(c, r)) / (r - c + 1e-10)
    return fb

MEL_FB = _build_mel_filterbank()

def _load_wav(path):
    with wave.open(path, 'rb') as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32)

def _rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))

def _get_ipd(sig1, sig2):
    a1 = np.angle(hilbert(sig1))
    a2 = np.angle(hilbert(sig2))
    return float(np.mean(np.degrees(np.arctan2(np.sin(a1-a2), np.cos(a1-a2)))))

def _get_ipd_mel(sig1, sig2):
    X1 = np.fft.rfft(sig1, n=N_FFT)
    X2 = np.fft.rfft(sig2, n=N_FFT)
    phase_diff = np.angle(X1 * np.conj(X2))
    power      = np.abs(X1) * np.abs(X2) + 1e-10
    weighted   = MEL_FB @ (phase_diff * power)
    weights    = MEL_FB @ power
    return (weighted / (weights + 1e-10)).astype(np.float32)

def _get_logmel(chunk):
    power = np.abs(np.fft.rfft(chunk, n=N_FFT)) ** 2
    return np.log(MEL_FB @ power + 1e-10).astype(np.float32)

def _get_gcc_phat(sig1, sig2):
    fft_len = 2 * len(sig1)
    X1  = np.fft.rfft(sig1, n=fft_len)
    X2  = np.fft.rfft(sig2, n=fft_len)
    Pxx = X1 * np.conj(X2)
    mag = np.abs(Pxx); mag[mag < 1e-10] = 1e-10
    gcc = np.fft.irfft(Pxx / mag, n=fft_len)[:len(sig1)]
    peak_idx = int(np.argmax(gcc))
    if peak_idx > len(gcc) // 2:
        peak_idx -= len(gcc)
    return float((peak_idx / RATE) * 1000), float(np.max(gcc) / (np.std(gcc) + 1e-10))

def _get_gcc_vector(sig1, sig2):
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


def extract_chunk(ch):
    """
    Extract features from one 30ms chunk.

    Parameters
    ----------
    ch : dict  { 'mic_right': np.float32 array, 'mic_front': ..., ... }

    Returns
    -------
    np.float32 array of shape (899,)
    """
    rms_feats     = [_rms(ch[m]) for m in MICS]
    logmel_feats  = [_get_logmel(ch[m]) for m in MICS]
    ipd_feats     = [
        _get_ipd(ch['mic_right'], ch['mic_front']),
        _get_ipd(ch['mic_right'], ch['mic_left']),
        _get_ipd(ch['mic_front'], ch['mic_back']),
    ]
    ipd_mel_feats = [
        _get_ipd_mel(ch['mic_right'], ch['mic_front']),
        _get_ipd_mel(ch['mic_right'], ch['mic_left']),
        _get_ipd_mel(ch['mic_front'], ch['mic_back']),
    ]
    gcc_tdoa, gcc_str, gcc_vecs = [], [], []
    for i in range(4):
        for j in range(i+1, 4):
            t, s = _get_gcc_phat(ch[MICS[i]], ch[MICS[j]])
            v    = _get_gcc_vector(ch[MICS[i]], ch[MICS[j]])
            gcc_tdoa.append(t); gcc_str.append(s); gcc_vecs.append(v)

    return np.concatenate([
        rms_feats, ipd_feats, np.concatenate(ipd_mel_feats),
        gcc_tdoa, gcc_str, np.concatenate(gcc_vecs),
        np.concatenate(logmel_feats),
    ]).astype(np.float32)


def extract_features(path_right, path_front, path_left, path_back):
    """
    Load 4 WAV files and extract features for every 30ms chunk.
    Returns np.array of shape (n_chunks, 899).
    """
    signals = {
        'mic_right': _load_wav(path_right),
        'mic_front': _load_wav(path_front),
        'mic_left':  _load_wav(path_left),
        'mic_back':  _load_wav(path_back),
    }
    n_chunks = min(len(s) for s in signals.values()) // CHUNK_SAMPLES
    features = []
    for ci in range(n_chunks):
        s = ci * CHUNK_SAMPLES
        e = s + CHUNK_SAMPLES
        ch = {m: signals[m][s:e] for m in MICS}
        features.append(extract_chunk(ch))
    return np.stack(features)
