"""
Localization CNN — All 4 Experiments
S1: Train E1  -> Test E1
S2: Train E1  -> Test E2
S3: Train E2  -> Test E1
S4: Train E1+E2 -> Test E1+E2
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_DIR = r'C:\Users\ahmma\Desktop\farah\features'
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 1e-3
ANGLES       = [0, 45, 90, 135, 180, 225, 270, 315]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FEATURE_COLS = [
    'rms_mic_right', 'rms_mic_front', 'rms_mic_left', 'rms_mic_back',
    'ipd_pair0', 'ipd_pair1', 'ipd_pair2',
    'gcc_tdoa_0', 'gcc_tdoa_1', 'gcc_tdoa_2',
    'gcc_tdoa_3', 'gcc_tdoa_4', 'gcc_tdoa_5',
    'gcc_strength_0', 'gcc_strength_1', 'gcc_strength_2',
    'gcc_strength_3', 'gcc_strength_4', 'gcc_strength_5',
]

# ── Model ─────────────────────────────────────────────────────────────────────
class LocalizationCNN(nn.Module):
    def __init__(self, n_features=19, n_classes=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * n_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_csv(names):
    """Load and concatenate one or more CSV files."""
    dfs = [pd.read_csv(f'{FEATURES_DIR}/{n}.csv') for n in names]
    return pd.concat(dfs, ignore_index=True)

def prepare(df_train, df_test):
    X_tr = df_train[FEATURE_COLS].values.astype(np.float32)
    y_tr = df_train['label'].values.astype(np.int64)
    X_te = df_test[FEATURE_COLS].values.astype(np.float32)
    y_te = df_test['label'].values.astype(np.int64)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, y_tr, X_te, y_te

def train_and_eval(X_tr, y_tr, X_te, y_te, label):
    # Keep on CPU for DataLoader, move batches to GPU in loop
    X_tr_t = torch.tensor(X_tr).unsqueeze(1)
    y_tr_t = torch.tensor(y_tr)
    X_te_t = torch.tensor(X_te).unsqueeze(1).to(DEVICE)
    y_te_t = torch.tensor(y_te).to(DEVICE)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=BATCH_SIZE, shuffle=True)

    model     = LocalizationCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_acc_hist, test_acc_hist = [], []

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == yb).sum().item()
            total   += len(yb)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            te_acc = (model(X_te_t).argmax(1) == y_te_t).float().mean().item()
        train_acc_hist.append(correct / total)
        test_acc_hist.append(te_acc)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).argmax(1).cpu().numpy()

    acc = (preds == y_te).mean() * 100
    print(f'\n  {label}: {acc:.2f}%')
    print(classification_report(y_te, preds,
          target_names=[f'{a}deg' for a in ANGLES], zero_division=0))

    return acc, preds, y_te, train_acc_hist, test_acc_hist, model

# ── Load all splits ───────────────────────────────────────────────────────────
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

tr_E1 = load_csv(['train_E1'])
tr_E2 = load_csv(['train_E2'])
te_E1 = load_csv(['test_E1'])
te_E2 = load_csv(['test_E2'])
tr_both = load_csv(['train_E1', 'train_E2'])
te_both = load_csv(['test_E1',  'test_E2'])

# ── Run experiments ───────────────────────────────────────────────────────────
experiments = [
    ('S1: E1 -> E1', tr_E1,   te_E1),
    ('S2: E1 -> E2', tr_E1,   te_E2),
    ('S3: E2 -> E1', tr_E2,   te_E1),
    ('S4: E1+E2 -> E1+E2', tr_both, te_both),
]

results = {}
for name, tr, te in experiments:
    print(f'\n{"="*55}\n  {name}\n  Train: {len(tr)}  Test: {len(te)}\n{"="*55}')
    X_tr, y_tr, X_te, y_te = prepare(tr, te)
    acc, preds, y_te_arr, tr_hist, te_hist, model = train_and_eval(
        X_tr, y_tr, X_te, y_te, name)
    results[name] = dict(acc=acc, preds=preds, y_te=y_te_arr,
                         tr_hist=tr_hist, te_hist=te_hist, model=model)

# ── Summary table ─────────────────────────────────────────────────────────────
print(f'\n{"="*40}')
print('  RESULTS SUMMARY')
print(f'{"="*40}')
for name, r in results.items():
    print(f'  {name:<25}  {r["acc"]:>6.2f}%')
print(f'{"="*40}')

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 8))

for col, (name, r) in enumerate(results.items()):
    # Accuracy curves
    ax = axes[0, col]
    ax.plot(r['tr_hist'], label='Train')
    ax.plot(r['te_hist'], label='Test')
    ax.set_title(name, fontsize=9)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Confusion matrix
    ax = axes[1, col]
    cm = confusion_matrix(r['y_te'], r['preds'])
    im = ax.imshow(cm, cmap='Blues', vmin=0)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels([f'{a}' for a in ANGLES], rotation=45, fontsize=7)
    ax.set_yticklabels([f'{a}' for a in ANGLES], fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{r["acc"]:.1f}%', fontsize=10)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, cm[i,j], ha='center', va='center', fontsize=6,
                    color='white' if cm[i,j] > cm.max()/2 else 'black')

plt.suptitle('Localization CNN — All Experiments', fontsize=13)
plt.tight_layout()
plt.savefig('results_all.png', dpi=150, bbox_inches='tight')
print('\nPlot saved: results_all.png')
