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

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_DIR = r'C:\Users\ahmma\Desktop\farah\features'
EPOCHS       = 50
BATCH_SIZE   = 64
LR           = 1e-3
ANGLES       = list(range(0, 360, 15))   # 24 classes
N_CLASSES    = len(ANGLES)               # 24

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
    def __init__(self, n_features=19, n_classes=N_CLASSES):
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
def prepare(X_tr_raw, y_tr_raw, X_te_raw, y_te_raw):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw.astype(np.float32))
    X_te = scaler.transform(X_te_raw.astype(np.float32))
    return X_tr, y_tr_raw.astype(np.int64), X_te, y_te_raw.astype(np.int64), scaler

def train_and_eval(X_tr, y_tr, X_te, y_te, label):
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

    train_acc_hist, test_acc_hist, loss_hist = [], [], []

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

        model.eval()
        with torch.no_grad():
            te_acc = (model(X_te_t).argmax(1) == y_te_t).float().mean().item()
        train_acc_hist.append(correct / total)
        test_acc_hist.append(te_acc)
        loss_hist.append(total_loss / total)

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1:>3}/{EPOCHS}  loss={loss_hist[-1]:.4f}  '
                  f'train={train_acc_hist[-1]*100:.1f}%  test={te_acc*100:.1f}%')

    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).argmax(1).cpu().numpy()

    acc = (preds == y_te).mean() * 100
    print(f'\n  {label}: {acc:.2f}%')
    print(classification_report(y_te, preds,
          target_names=[f'{a}deg' for a in ANGLES], zero_division=0))

    return acc, preds, y_te, train_acc_hist, test_acc_hist, loss_hist, model

# ── Load data ─────────────────────────────────────────────────────────────────
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

train_df = pd.read_csv(f'{FEATURES_DIR}/train_DATA.csv')
test_df  = pd.read_csv(f'{FEATURES_DIR}/test_DATA.csv')

X_all = train_df[FEATURE_COLS].values
y_all = train_df['label'].to_numpy(dtype=np.int64)
X_te  = test_df[FEATURE_COLS].values
y_te  = test_df['label'].to_numpy(dtype=np.int64)

# 80/20 split of training data (stratified so all angles represented)
X_tr80, X_val20, y_tr80, y_val20 = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

print(f'\nTrain (80%): {len(X_tr80)}  Val (20%): {len(X_val20)}  Test: {len(X_te)}')

# ── Run experiments ───────────────────────────────────────────────────────────
experiments = [
    ('S1: 80% train -> 20% train', X_tr80, y_tr80, X_val20, y_val20),
    ('S2: 80% train -> test',      X_tr80, y_tr80, X_te,    y_te),
]

results = {}
for name, Xtr, ytr, Xte, yte in experiments:
    print(f'\n{"="*55}\n  {name}\n  Train: {len(Xtr)}  Test: {len(Xte)}\n{"="*55}')
    X_tr_s, y_tr_s, X_te_s, y_te_s, scaler = prepare(Xtr, ytr, Xte, yte)
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
print(f'\n{"="*40}')
print('  RESULTS SUMMARY')
print(f'{"="*40}')
for name, r in results.items():
    print(f'  {name:<35}  {r["acc"]:>6.2f}%')
print(f'{"="*40}')

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
    im = ax.imshow(cm, cmap='Blues', vmin=0)
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
plt.savefig('results_all.png', dpi=150, bbox_inches='tight')
print('\nPlot saved: results_all.png')
