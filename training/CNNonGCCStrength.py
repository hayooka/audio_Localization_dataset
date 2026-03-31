"""
Localization CNN
S1: Train on 80% of train_E1 -> Test on 20% of train_E1  (internal validation)
S2: Train on 80% of train_E1 -> Test on test_E1           (held-out test)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from data_processing import augment_all, EarlyStopping

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_DIR = r'C:\Users\ahmma\Desktop\farah\features'
CHUNK_TAG    = '30'   # e.g. '30', '50', '80' — matches train_DATA{tag}.csv
EPOCHS       = 50
BATCH_SIZE   = 64
LR           = 1e-3
ANGLES       = list(range(0, 360, 15))   # 24 classes
N_CLASSES    = len(ANGLES)               # 24

DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
FEATURE_TAG  = 'GCC_STRENGTH'
RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

_MICS   = ['mic_right', 'mic_front', 'mic_left', 'mic_back']
_N_MELS = 40

_GCC_VEC_SIZE = 100
_N_PAIRS      = 6

FEATURE_COLS = [
    'gcc_strength_0', 'gcc_strength_1', 'gcc_strength_2',
    'gcc_strength_3', 'gcc_strength_4', 'gcc_strength_5',
    *[f'gcc_vec_{i}_t{t}' for i in range(_N_PAIRS) for t in range(_GCC_VEC_SIZE)],
]

RMS_COLS      = ['rms_mic_right', 'rms_mic_front', 'rms_mic_left', 'rms_mic_back']
RMS_THRESHOLD = 100.0  # drop frame if ANY mic RMS is below this

# indices used by augmentation functions
_LOGMEL_IDX = [i for i, c in enumerate(FEATURE_COLS) if 'logmel' in c]

EARLY_STOP_PATIENCE = 10  # stop if val acc doesn't improve for this many epochs

# ── Model ─────────────────────────────────────────────────────────────────────
class LocalizationCNN(nn.Module):
    def __init__(self, n_features, n_classes=N_CLASSES):
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
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# ── Helpers ───────────────────────────────────────────────────────────────────
def prepare(X_tr_raw, y_tr_raw, X_te_raw, y_te_raw, X_rms=None):
    # # augment train only: original + noise + gain + time-shift copies
    # X_tr_aug = augment_all(X_tr_raw.astype(np.float32), _LOGMEL_IDX, X_rms)
    # y_tr_aug = np.tile(y_tr_raw, 4).astype(np.int64)
    # print(f'  Augmented train: {len(X_tr_raw)} → {len(X_tr_aug)} samples')
    X_tr_aug = X_tr_raw.astype(np.float32)
    y_tr_aug = y_tr_raw.astype(np.int64)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_aug)
    X_te = scaler.transform(X_te_raw.astype(np.float32))
    return X_tr, y_tr_aug, X_te, y_te_raw.astype(np.int64), scaler

def eval_batched(model, X_te_t, y_te, batch_size=512):
    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_te_t), batch_size):
            xb = X_te_t[i:i+batch_size].to(DEVICE)
            all_preds.append(model(xb).argmax(1).cpu())
    preds = torch.cat(all_preds).numpy()
    return (preds == y_te).mean(), preds

def train_and_eval(X_tr, y_tr, X_te, y_te, label):
    X_tr_t = torch.tensor(X_tr).float().unsqueeze(1)
    y_tr_t = torch.tensor(y_tr)
    X_te_t = torch.tensor(X_te).float().unsqueeze(1)  # stays on CPU

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=BATCH_SIZE, shuffle=True)

    model     = LocalizationCNN(n_features=len(FEATURE_COLS)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_acc_hist, test_acc_hist, loss_hist = [], [], []
    es = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    for epoch in range(EPOCHS):
        model.train()
        correct, total, total_loss = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (out.argmax(1) == yb).sum().item()
            total      += len(yb)
        scheduler.step()

        te_acc, _ = eval_batched(model, X_te_t, y_te)
        train_acc_hist.append(correct / total)
        test_acc_hist.append(te_acc)
        loss_hist.append(total_loss / total)

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1:>3}/{EPOCHS}  loss={loss_hist[-1]:.4f}  '
                  f'train={train_acc_hist[-1]*100:.1f}%  test={te_acc*100:.1f}%')

        if es(te_acc, model):
            print(f'  Early stop at epoch {epoch+1}  (best val={es.best_acc*100:.1f}%)')
            break

    if es.best_model is not None:
        model.load_state_dict(es.best_model)
    acc_frac, preds = eval_batched(model, X_te_t, y_te)
    acc = acc_frac * 100
    print(f'\n  {label}: {acc:.2f}%')
    print(classification_report(y_te, preds,
          target_names=[f'{a}deg' for a in ANGLES], zero_division=0))

    return acc, preds, y_te, train_acc_hist, test_acc_hist, loss_hist, model

# ── Load data ─────────────────────────────────────────────────────────────────
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

train_df = pd.read_csv(f'{FEATURES_DIR}/train_DATA{CHUNK_TAG}.csv')
test_df  = pd.read_csv(f'{FEATURES_DIR}/test_DATA{CHUNK_TAG}.csv')

# silence removal: drop frame if ANY mic RMS is below threshold
for name, df in [('train', train_df), ('test', test_df)]:
    before = len(df)
    mask = (df[RMS_COLS] >= RMS_THRESHOLD).all(axis=1)
    df.drop(index=df.index[~mask], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f'  {name}: dropped {before - len(df)} silent frames, kept {len(df)}')

X_all    = train_df[FEATURE_COLS].values
y_all    = train_df['label'].to_numpy(dtype=np.int64)
X_te     = test_df[FEATURE_COLS].values
y_te     = test_df['label'].to_numpy(dtype=np.int64)
X_rms_all = train_df[RMS_COLS].to_numpy(dtype=np.float32)  # (N, 4) — for augmentation only

# 80/20 split of training data (stratified so all angles represented)
X_tr80, X_val20, y_tr80, y_val20, X_rms80, _ = train_test_split(
    X_all, y_all, X_rms_all, test_size=0.2, random_state=42, stratify=y_all)

print(f'\nTrain (80%): {len(X_tr80)}  Val (20%): {len(X_val20)}  Test: {len(X_te)}')

# ── Run experiments ───────────────────────────────────────────────────────────
experiments = [
    ('S1: 80% train -> 20% train', X_tr80, y_tr80, X_val20, y_val20, X_rms80),
    ('S2: 80% train -> test',      X_tr80, y_tr80, X_te,    y_te,    X_rms80),
]

results = {}
for name, Xtr, ytr, Xte, yte, Xrms in experiments:
    print(f'\n{"="*55}\n  {name}\n  Train: {len(Xtr)}  Test: {len(Xte)}\n{"="*55}')
    X_tr_s, y_tr_s, X_te_s, y_te_s, scaler = prepare(Xtr, ytr, Xte, yte, Xrms)
    acc, preds, y_te_arr, tr_hist, te_hist, loss_h, model = train_and_eval(
        X_tr_s, y_tr_s, X_te_s, y_te_s, name)
    results[name] = dict(acc=acc, preds=preds, y_te=y_te_arr,
                         tr_hist=tr_hist, te_hist=te_hist,
                         loss_hist=loss_h, model=model)

    # Save S2 model (trained on 80% of train, evaluated on real test set)
    if name == 'S2: 80% train -> test':
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_E1.pt')
        torch.save({
            'model_state':  model.state_dict(),
            'scaler_mean':  scaler.mean_,
            'scaler_std':   scaler.scale_,
            'feature_cols': FEATURE_COLS,
        }, save_path)
        print(f'  Model saved: {save_path}')

# ── Summary table ─────────────────────────────────────────────────────────────
summary_lines = [
    f'\n{"="*40}',
    '  RESULTS SUMMARY',
    f'{"="*40}',
    *[f'  {name:<35}  {r["acc"]:>6.2f}%' for name, r in results.items()],
    f'{"="*40}',
]
print('\n'.join(summary_lines))
summary_path = os.path.join(RESULTS_DIR, f'summary_{FEATURE_TAG}.txt')
with open(summary_path, 'w') as f:
    f.write('\n'.join(summary_lines) + '\n')
print(f'Summary saved: {summary_path}')

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 14))

for col, (name, r) in enumerate(results.items()):
    # Accuracy curves
    ax = axes[0, col]
    ax.plot(r['tr_hist'], label='Train')
    ax.plot(r['te_hist'], label='Test')
    ax.set_title(name, fontsize=9)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Loss curve
    ax = axes[1, col]
    ax.plot(r['loss_hist'], color='coral')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss', fontsize=9); ax.grid(True, alpha=0.3)

    # Confusion matrix
    ax = axes[2, col]
    cm = confusion_matrix(r['y_te'], r['preds'])
    ax.imshow(cm, cmap='Blues', vmin=0)
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels([f'{a}' for a in ANGLES], rotation=90, fontsize=6)
    ax.set_yticklabels([f'{a}' for a in ANGLES], fontsize=6)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{r["acc"]:.1f}%', fontsize=10)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=4,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

plt.suptitle('Localization CNN — E1 Dataset (24 angles)', fontsize=13)
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, f'results_{FEATURE_TAG}.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f'\nPlot saved: {plot_path}')
