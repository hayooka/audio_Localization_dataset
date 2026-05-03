# -*- coding: utf-8 -*-
"""
Optimized Feature Extraction — Sound Localization
Processes (24 angles) dataset.
5min recordings -> train
3min recordings -> test
Features per chunk:
  RMS            (4 mics)
  IPD scalar     (3 pairs)
  IPD mel        (3 pairs × 40 bands = 120)
  GCC-PHAT TDOA + strength (6 pairs each = 12)
  GCC vector     (6 pairs × 100 points = 600)
  Log-mel        (4 mics × 40 bands = 160)
  Total: 4 + 3 + 120 + 12 + 600 + 160 = 899 features
"""

import numpy as np
import wave
import os
import json
from scipy.signal import hilbert

# ========================
# SETTINGS
# ========================

# Repository paths
REPO_ROOT = r'C:\Users\ahmma\Desktop\farah\audio_Localization_dataset'

# Input dataset path (where your raw audio files are)
DATASETS = {
    'DATA': r'C:\Users\ahmma\Desktop\farah\(24 angles)dataset',
}

# Output paths inside repository
OUTPUT_DIR = os.path.join(REPO_ROOT, 'features')
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Feature extraction settings
ANGLES = list(range(0, 360, 15))
MICS = ['mic_right', 'mic_front', 'mic_left', 'mic_back']
RATE = 16000
CHUNK_SEC = 0.016  # 16ms = 256 samples at 16kHz (perfect power of 2)
N_MELS = 40
N_FFT = 256  # Perfect match for 256 samples

POSITION_TO_LABEL = {angle: i for i, angle in enumerate(ANGLES)}
GCC_VECTOR_SIZE = 100

# ========================
# FEATURE FUNCTIONS
# ========================

def load_wav(path):
    """Load WAV file and return as float32 array"""
    with wave.open(path, 'rb') as w:
        raw = w.readframes(w.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return audio

def rms(audio):
    """Root mean square energy"""
    return float(np.sqrt(np.mean(audio ** 2)))

def get_ipd(sig1, sig2):
    """Interaural Phase Difference (scalar)"""
    a1 = np.angle(hilbert(sig1))
    a2 = np.angle(hilbert(sig2))
    return float(np.mean(np.degrees(np.arctan2(np.sin(a1 - a2), np.cos(a1 - a2)))))

def _build_mel_filterbank():
    """Build mel filterbank matrix"""
    def hz_to_mel(f): 
        return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m): 
        return 700 * (10 ** (m / 2595) - 1)
    
    mel_pts = np.linspace(hz_to_mel(0), hz_to_mel(RATE / 2), N_MELS + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((N_FFT + 1) * hz_pts / RATE).astype(int)
    
    fb = np.zeros((N_MELS, N_FFT // 2 + 1))
    for m in range(1, N_MELS + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / (c - l + 1e-10)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / (r - c + 1e-10)
    return fb

MEL_FB = _build_mel_filterbank()

def get_ipd_mel(sig1, sig2):
    """IPD per mel band — shape (N_MELS,)"""
    X1 = np.fft.rfft(sig1, n=N_FFT)
    X2 = np.fft.rfft(sig2, n=N_FFT)
    phase_diff = np.angle(X1 * np.conj(X2))
    power = np.abs(X1) * np.abs(X2) + 1e-10
    weighted = MEL_FB @ (phase_diff * power)
    weights = MEL_FB @ power
    return (weighted / (weights + 1e-10)).astype(np.float32)

def get_logmel_energy(chunk):
    """Log-mel energy vector of shape (N_MELS,)"""
    power = np.abs(np.fft.rfft(chunk, n=N_FFT)) ** 2
    return np.log(MEL_FB @ power + 1e-10)

def get_gcc_phat(sig1, sig2):
    """GCC-PHAT TDOA (ms) and strength"""
    fft_len = 2 * len(sig1)
    X1 = np.fft.rfft(sig1, n=fft_len)
    X2 = np.fft.rfft(sig2, n=fft_len)
    Pxx = X1 * np.conj(X2)
    mag = np.abs(Pxx)
    mag[mag < 1e-10] = 1e-10
    gcc = np.fft.irfft(Pxx / mag, n=fft_len)[:len(sig1)]
    peak_idx = int(np.argmax(gcc))
    if peak_idx > len(gcc) // 2:
        peak_idx -= len(gcc)
    tdoa_ms = float((peak_idx / RATE) * 1000)
    strength = float(np.max(gcc) / (np.std(gcc) + 1e-10))
    return tdoa_ms, strength

def get_gcc_vector(sig1, sig2):
    """Full GCC-PHAT vector centred at zero-lag"""
    fft_len = 2 * len(sig1)
    X1 = np.fft.rfft(sig1, n=fft_len)
    X2 = np.fft.rfft(sig2, n=fft_len)
    Pxx = X1 * np.conj(X2)
    mag = np.abs(Pxx)
    mag[mag < 1e-10] = 1e-10
    gcc = np.fft.irfft(Pxx / mag, n=fft_len)
    gcc = np.roll(gcc, len(gcc) // 2)
    centre = len(gcc) // 2
    half = GCC_VECTOR_SIZE // 2
    gcc = gcc[centre - half:centre + half]
    mx = np.max(np.abs(gcc))
    if mx > 0:
        gcc = gcc / mx
    return gcc.astype(np.float32)

# ========================
# EXTRACT FROM ONE FILE SET
# ========================

def extract_chunks(base, angle, duration):
    """Extract all chunks from a single angle recording"""
    chunk_samples = int(CHUNK_SEC * RATE)
    signals = {}
    
    # Load all mic signals
    for mic in MICS:
        path = os.path.join(base, str(angle), 'speakerM', duration, mic + '.wav')
        if not os.path.exists(path):
            return []
        signals[mic] = load_wav(path)
    
    n_chunks = min(len(s) for s in signals.values()) // chunk_samples
    samples = []
    
    for ci in range(n_chunks):
        s = ci * chunk_samples
        e = s + chunk_samples
        ch = {m: signals[m][s:e] for m in MICS}
        
        # RMS features
        rms_feats = [rms(ch[m]) for m in MICS]
        
        # Log-mel features
        logmel_feats = [get_logmel_energy(ch[m]) for m in MICS]
        
        # IPD scalar features
        ipd_feats = [
            get_ipd(ch['mic_right'], ch['mic_front']),
            get_ipd(ch['mic_right'], ch['mic_left']),
            get_ipd(ch['mic_front'], ch['mic_back']),
        ]
        
        # IPD mel features
        ipd_mel_feats = [
            get_ipd_mel(ch['mic_right'], ch['mic_front']),
            get_ipd_mel(ch['mic_right'], ch['mic_left']),
            get_ipd_mel(ch['mic_front'], ch['mic_back']),
        ]
        
        # GCC features for all mic pairs (6 pairs)
        mic_pairs = [
            ('mic_right', 'mic_front'),
            ('mic_right', 'mic_left'),
            ('mic_right', 'mic_back'),
            ('mic_front', 'mic_left'),
            ('mic_front', 'mic_back'),
            ('mic_left', 'mic_back')
        ]
        
        gcc_tdoa = []
        gcc_str = []
        gcc_vecs = []
        
        for mic1, mic2 in mic_pairs:
            t, s_val = get_gcc_phat(ch[mic1], ch[mic2])
            vec = get_gcc_vector(ch[mic1], ch[mic2])
            gcc_tdoa.append(t)
            gcc_str.append(s_val)
            gcc_vecs.append(vec)
        
        # Build feature array (899 features)
        feature_list = []
        
        # RMS (4)
        feature_list.extend(rms_feats)
        
        # IPD scalar (3)
        feature_list.extend(ipd_feats)
        
        # IPD mel (3 × 40 = 120)
        for i in range(3):
            feature_list.extend(ipd_mel_feats[i])
        
        # GCC TDOA (6)
        feature_list.extend(gcc_tdoa)
        
        # GCC strength (6)
        feature_list.extend(gcc_str)
        
        # GCC vectors (6 × 100 = 600)
        for i in range(6):
            feature_list.extend(gcc_vecs[i])
        
        # Log-mel (4 × 40 = 160)
        for i in range(4):
            feature_list.extend(logmel_feats[i])
        
        samples.append({
            'dataset': os.path.basename(base),
            'position': angle,
            'chunk': ci,
            'label': POSITION_TO_LABEL[angle],
            'features': np.array(feature_list, dtype=np.float32)
        })
    
    return samples

# ========================
# SAVE FUNCTIONS
# ========================

def save_as_npz(samples, output_path):
    """Save features as compressed numpy array with separate JSON metadata"""
    if not samples:
        print(f"  Warning: No samples to save for {output_path}")
        return False
    
    # Extract arrays
    features = np.stack([s['features'] for s in samples])
    labels = np.array([s['label'] for s in samples])
    positions = np.array([s['position'] for s in samples])
    chunks = np.array([s['chunk'] for s in samples])
    
    # Save arrays as NPZ
    np.savez_compressed(
        output_path,
        features=features,
        labels=labels,
        positions=positions,
        chunks=chunks
    )
    
    # Save metadata as separate JSON file
    metadata = {
        'feature_names': None,
        'n_features': int(features.shape[1]),
        'n_samples': int(len(samples)),
        'n_classes': int(labels.max() + 1),
        'chunk_sec': float(CHUNK_SEC),
        'chunk_samples': int(CHUNK_SEC * RATE),
        'n_mels': int(N_MELS),
        'n_fft': int(N_FFT),
        'sample_rate': int(RATE),
        'gcc_vector_size': int(GCC_VECTOR_SIZE),
        'mics': MICS,
        'angles': ANGLES
    }
    
    metadata_path = output_path.replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    file_size_mb = features.nbytes / 1024 / 1024
    print(f"    Saved: {os.path.basename(output_path)} ({file_size_mb:.2f} MB)")
    print(f"    Metadata: {os.path.basename(metadata_path)}")
    print(f"    Features shape: {features.shape}")
    
    return True

# ========================
# LOAD FUNCTIONS (FOR TRAINING)
# ========================

def load_features_for_training(npz_path):
    """Load pre-extracted features for training"""
    # Load arrays
    data = np.load(npz_path)
    
    # Load metadata
    metadata_path = npz_path.replace('.npz', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return {
        'features': data['features'],
        'labels': data['labels'],
        'positions': data['positions'],
        'chunks': data['chunks'],
        'metadata': metadata
    }

def load_features_memory_mapped(npz_path):
    """Memory-efficient loading for large datasets"""
    # Load metadata only
    metadata_path = npz_path.replace('.npz', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Memory-map the arrays (doesn't load into RAM)
    data = np.load(npz_path, mmap_mode='r')
    
    return {
        'features': data['features'],
        'labels': data['labels'],
        'positions': data['positions'],
        'chunks': data['chunks'],
        'metadata': metadata
    }

# ========================
# MAIN LOOP
# ========================

def main():
    """Main extraction function"""
    print("="*70)
    print(" FEATURE EXTRACTION - SOUND LOCALIZATION")
    print("="*70)
    print(f"Repository root: {REPO_ROOT}")
    print(f"Settings:")
    print(f"  Chunk size: {CHUNK_SEC*1000:.0f}ms ({int(CHUNK_SEC*RATE)} samples)")
    print(f"  FFT size: {N_FFT}")
    print(f"  Mel bands: {N_MELS}")
    print(f"  Features per chunk: {4 + 3 + 120 + 12 + 600 + 160} = 899")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("="*70)
    
    for duration, split_name in [('5min', 'train'), ('3min', 'test')]:
        for ds_name, base in DATASETS.items():
            all_samples = []
            print(f'\n{"-"*60}')
            print(f'{split_name.upper()} — {ds_name} ({duration} recordings)')
            print(f'{"-"*60}')
            
            for angle in ANGLES:
                chunks = extract_chunks(base, angle, duration)
                if chunks:
                    all_samples.extend(chunks)
                    print(f'  {angle:>3}°  ->  {len(chunks):4} chunks')
                else:
                    print(f'  {angle:>3}°  ->  MISSING')
            
            if all_samples:
                chunk_ms = int(CHUNK_SEC * 1000)
                out_path = os.path.join(OUTPUT_DIR, f'{split_name}_{ds_name}_{chunk_ms}ms.npz')
                save_as_npz(all_samples, out_path)
                
                print(f'\n  Total chunks: {len(all_samples)}')
                print(f'  Total features: {len(all_samples[0]["features"])}')
                print(f'  Classes: {len(set(s["label"] for s in all_samples))}')
            else:
                print(f'\n  WARNING: No chunks extracted for {split_name}_{ds_name}')
    
    print("\n" + "="*70)
    print(" EXTRACTION COMPLETE!")
    print("="*70)
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print(f"Models will be saved to: {MODELS_DIR}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    
    # Example of how to load for training
    print("\nTo load data for training, use:")
    print("  data = load_features_for_training(os.path.join(REPO_ROOT, 'features', 'train_DATA_16ms.npz'))")
    print("  X_train = data['features']")
    print("  y_train = data['labels']")

# ========================
# EXAMPLE TRAINING SCRIPT
# ========================

def example_training_script():
    """Example of how to use the extracted features for training"""
    # Load training data
    train_path = os.path.join(OUTPUT_DIR, 'train_DATA_16ms.npz')
    test_path = os.path.join(OUTPUT_DIR, 'test_DATA_16ms.npz')
    
    if not os.path.exists(train_path):
        print(f"Training file not found: {train_path}")
        print("Please run extraction first.")
        return
    
    train_data = load_features_for_training(train_path)
    test_data = load_features_for_training(test_path)
    
    X_train = train_data['features']
    y_train = train_data['labels']
    X_test = test_data['features']
    y_test = test_data['labels']
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    print(f"Metadata: {train_data['metadata']}")
    
    # Example with sklearn
    # from sklearn.ensemble import RandomForestClassifier
    # model = RandomForestClassifier(n_estimators=100)
    # model.fit(X_train, y_train)
    # accuracy = model.score(X_test, y_test)
    # print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
    
    # Uncomment to run example after extraction
    # if os.path.exists(os.path.join(OUTPUT_DIR, 'train_DATA_16ms.npz')):
    #     print("\n" + "="*70)
    #     print(" RUNNING EXAMPLE")
    #     print("="*70)
    #     example_training_script()