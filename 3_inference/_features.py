import numpy as np
import wave

RATE          = 16000
CHUNK_SEC     = 0.03
CHUNK_SAMPLES = int(CHUNK_SEC * RATE)   # 480 samples per 30ms chunk
GCC_VEC_SIZE  = 100
N_PAIRS       = 6
MICS          = ['mic_right', 'mic_front', 'mic_left', 'mic_back']


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
