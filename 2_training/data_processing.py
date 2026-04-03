"""
Data Processing — Sound Localization
Techniques to improve S2 generalization (train→test distribution shift).

Usage: import functions into any training script as needed.
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 1) DATA AUGMENTATION  (operate on raw feature rows)
# ══════════════════════════════════════════════════════════════════════════════

def augment_add_noise(X, rms_energy, noise_factor=0.01):
    """Add Gaussian noise scaled to each frame's RMS energy.
    rms_energy: (N,1) array of per-frame energy, or None for unit scale."""
    X_aug = X.copy()
    scale = rms_energy if rms_energy is not None else np.ones((len(X), 1), dtype=np.float32)
    noise = np.random.randn(*X.shape).astype(np.float32)
    X_aug += noise_factor * scale * noise
    return X_aug

def augment_gain(X, logmel_cols_idx, gain_range=(0.8, 1.2)):
    """Random gain scaling on log-mel features — simulates different speaker volumes."""
    X_aug = X.copy()
    gains = np.random.uniform(*gain_range, size=(len(X), 1)).astype(np.float32)
    X_aug[:, logmel_cols_idx] += np.log(gains)  # log scale: multiply = add in log domain
    return X_aug

def augment_time_shift(X, logmel_cols_idx, max_shift=2):
    """Circular-shift log-mel bands to simulate small temporal jitter."""
    X_aug = X.copy()
    for i in range(len(X_aug)):
        shift = np.random.randint(-max_shift, max_shift + 1)
        X_aug[i, logmel_cols_idx] = np.roll(X_aug[i, logmel_cols_idx], shift)
    return X_aug

def augment_all(X, logmel_cols_idx, X_rms=None):
    """Apply all augmentations and return original + augmented stacked.
    X_rms: optional (N, 4) float32 array of RMS values for noise scaling."""
    rms_energy = X_rms.mean(axis=1, keepdims=True) if X_rms is not None else None
    X_noise = augment_add_noise(X, rms_energy)
    X_gain  = augment_gain(X, logmel_cols_idx)
    X_shift = augment_time_shift(X, logmel_cols_idx)
    return np.concatenate([X, X_noise, X_gain, X_shift], axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# 2) MIX TRAIN + PORTION OF TEST (semi-supervised style)
# ══════════════════════════════════════════════════════════════════════════════

def mix_train_test(X_train, y_train, X_test, y_test, test_ratio=0.2, seed=42):
    """
    Add a portion of the test set into training.
    Only use when you have labelled test data (supervised setting).
    test_ratio: fraction of test set to include in training.
    """
    rng  = np.random.default_rng(seed)
    n    = int(len(X_test) * test_ratio)
    idx  = rng.choice(len(X_test), size=n, replace=False)
    mask = np.ones(len(X_test), dtype=bool)
    mask[idx] = False

    X_tr_mix = np.concatenate([X_train, X_test[idx]], axis=0)
    y_tr_mix = np.concatenate([y_train, y_test[idx]], axis=0)
    X_te_rem = X_test[mask]
    y_te_rem = y_test[mask]

    print(f'  Mixed train: {len(X_tr_mix)}  Remaining test: {len(X_te_rem)}')
    return X_tr_mix, y_tr_mix, X_te_rem, y_te_rem


# ══════════════════════════════════════════════════════════════════════════════
# 3) EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stops training when validation accuracy stops improving.
    Usage:
        es = EarlyStopping(patience=10)
        for epoch in ...:
            ...
            if es(val_acc, model): break
        model = es.best_model
    """
    def __init__(self, patience=10):
        self.patience   = patience
        self.best_acc   = -1.0
        self.counter    = 0
        self.best_model = None

    def __call__(self, val_acc, model):
        import copy
        if val_acc > self.best_acc:
            self.best_acc   = val_acc
            self.counter    = 0
            self.best_model = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience


# ══════════════════════════════════════════════════════════════════════════════
# 4) OVERLAP WINDOWING  (for feature extraction — use in audionTOfeatures.py)
# ══════════════════════════════════════════════════════════════════════════════

def get_overlap_chunks(signal_len, chunk_samples, overlap=0.5):
    """
    Returns (start, end) pairs for overlapping windows.
    overlap=0.5 means 50% overlap → 2x more chunks.
    overlap=0.0 means no overlap  → same as current behavior.
    """
    step = int(chunk_samples * (1 - overlap))
    starts = range(0, signal_len - chunk_samples + 1, step)
    return [(s, s + chunk_samples) for s in starts]
