"""
Localization CNN — ALL features (895-D)
Train  : full 5-min recordings  (train_DATA{tag}.csv)
Val    : first 1 min per angle  from 3-min recordings (test_DATA{tag}.csv)
Test   : last  2 min per angle  from 3-min recordings
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
from data_processing import EarlyStopping

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(REPO_ROOT, 'features')
RESULTS_DIR  = os.path.join(REPO_ROOT, 'Results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CHUNK_TAG   = '16'           # matches train_DATA16.csv / test_DATA16.csv
CHUNK_MS    = int(CHUNK_TAG) # 16 ms
VAL_SEC     = 60             # first 60s of 3-min recording = validation

EPOCHS      = 50
BATCH_SIZE  = 64
LR          = 1e-3
ANGLES      = list(range(0, 360, 15))
N_CLASSES   = len(ANGLES)    # 24
EARLY_STOP_PATIENCE = 10

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
FEATURE_TAG = 'ALL16'

_MICS         = ['mic_right', 'mic_front', 'mic_left', 'mic_back']
_N_MELS       = 40
_GCC_VEC_SIZE = 100
_N_PAIRS      = 6

FEATURE_COLS = [
    'ipd_pair0', 'ipd_pair1', 'ipd_pair2',
    *[f'ipd_mel_{i}_b{b}'   for i in range(3)      for b in range(_N_MELS)],
    *[f'gcc_tdoa_{i}'       for i in range(6)],
    *[f'gcc_strength_{i}'   for i in range(6)],
    *[f'gcc_vec_{i}_t{t}'   for i in range(_N_PAIRS) for t in range(_GCC_VEC_SIZE)],
    *[f'logmel_{m}_b{b}'    for m in _MICS          for b in range(_N_MELS)],
]

RMS_COLS      = [f'rms_{m}' for m in _MICS]
RMS_THRESHOLD = 100.0

# ── Model ─────────────────────────────────────────────────────────────────────
class LocalizationCNN(nn.Module):
    def __init__(self, n_features, n_classes=N_CLASSES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32,  kernel_size=3, padding=1),
            nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 64),               nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# ── Helpers ───────────────────────────────────────────────────────────────────
def prepare(X_tr, y_tr, X_val, y_val, X_te, y_te):
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr.astype(np.float32))
    X_val = scaler.transform(X_val.astype(np.float32))
    X_te  = scaler.transform(X_te.astype(np.float32))
    return (X_tr,  y_tr.astype(np.int64),
            X_val, y_val.astype(np.int64),
            X_te,  y_te.astype(np.int64),
            scaler)

def angular_metrics(y_true, y_pred):
    td = np.array([ANGLES[i] for i in y_true], dtype=np.float32)
    pd_ = np.array([ANGLES[i] for i in y_pred], dtype=np.float32)
    diff = np.minimum(np.abs(td - pd_), 360 - np.abs(td - pd_))
    return float(np.mean(diff)), float(np.sqrt(np.mean(diff ** 2)))

def eval_batched(model, X_t, y, batch_size=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            preds.append(model(X_t[i:i+batch_size].to(DEVICE)).argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    return (preds == y).mean(), preds

def train_and_eval(X_tr, y_tr, X_val, y_val, X_te, y_te):
    X_tr_t  = torch.tensor(X_tr).float().unsqueeze(1)
    y_tr_t  = torch.tensor(y_tr)
    X_val_t = torch.tensor(X_val).float().unsqueeze(1)
    X_te_t  = torch.tensor(X_te).float().unsqueeze(1)

    loader    = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)
    model     = LocalizationCNN(n_features=len(FEATURE_COLS)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    es        = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    tr_hist, val_hist, te_hist, loss_hist = [], [], [], []

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

        val_acc, _ = eval_batched(model, X_val_t, y_val)
        te_acc,  _ = eval_batched(model, X_te_t,  y_te)
        tr_hist.append(correct / total)
        val_hist.append(val_acc)
        te_hist.append(te_acc)
        loss_hist.append(total_loss / total)

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1:>3}/{EPOCHS}  loss={loss_hist[-1]:.4f}  '
                  f'train={tr_hist[-1]*100:.1f}%  val={val_acc*100:.1f}%  test={te_acc*100:.1f}%')

        if es(val_acc, model):
            print(f'  Early stop at epoch {epoch+1}  (best val={es.best_acc*100:.1f}%)')
            break

    if es.best_model is not None:
        model.load_state_dict(es.best_model)

    _, preds = eval_batched(model, X_te_t, y_te)
    acc  = (preds == y_te).mean() * 100
    mae, rmse = angular_metrics(y_te, preds)
    print(f'\n  Test accuracy: {acc:.2f}%  MAE={mae:.1f}°  RMSE={rmse:.1f}°')
    print(classification_report(y_te, preds,
          target_names=[f'{a}°' for a in ANGLES], zero_division=0))

    return acc, preds, y_te, tr_hist, val_hist, te_hist, loss_hist, model, mae, rmse

# ── Load data ─────────────────────────────────────────────────────────────────
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

train_df = pd.read_csv(os.path.join(FEATURES_DIR, f'train_DATA{CHUNK_TAG}.csv'))
test_df  = pd.read_csv(os.path.join(FEATURES_DIR, f'test_DATA{CHUNK_TAG}.csv'))

# Silence removal
for name, df in [('train', train_df), ('test', test_df)]:
    before = len(df)
    mask = (df[RMS_COLS] >= RMS_THRESHOLD).all(axis=1)
    df.drop(index=df.index[~mask], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f'  {name}: dropped {before - len(df)} silent frames, kept {len(df)}')

# ── Split ─────────────────────────────────────────────────────────────────────
# train_DATA{tag}.csv  = 5-min recordings → all training
# test_DATA{tag}.csv   = 3-min recordings → first VAL_SEC per angle = val,
#                                           rest = test

val_chunks_per_angle = int(VAL_SEC * 1000 / CHUNK_MS)   # e.g. 60*1000/16 = 3750

val_parts, test_parts = [], []
for lbl in sorted(test_df['label'].unique()):
    angle_df = test_df[test_df['label'] == lbl].sort_values(by='chunk')  # type: ignore[call-overload]
    val_parts.append(angle_df.iloc[:val_chunks_per_angle])
    test_parts.append(angle_df.iloc[val_chunks_per_angle:])

val_df  = pd.concat(val_parts,  ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

X_train = train_df[FEATURE_COLS].values
y_train = train_df['label'].to_numpy(dtype=np.int64)
X_val   = val_df[FEATURE_COLS].values
y_val   = val_df['label'].to_numpy(dtype=np.int64)
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df['label'].to_numpy(dtype=np.int64)

print(f'\n  Train  (5-min full):       {len(X_train):>7} frames')
print(f'  Val    (first 1min/angle): {len(X_val):>7} frames  ({val_chunks_per_angle} per angle)')
print(f'  Test   (last  2min/angle): {len(X_test):>7} frames')

# ── Train ─────────────────────────────────────────────────────────────────────
X_tr_s, y_tr_s, X_val_s, y_val_s, X_te_s, y_te_s, scaler = prepare(
    X_train, y_train, X_val, y_val, X_test, y_test)

acc, preds, y_te_arr, tr_hist, val_hist, te_hist, loss_hist, model, mae, rmse = train_and_eval(
    X_tr_s, y_tr_s, X_val_s, y_val_s, X_te_s, y_te_s)

# ── Save model ────────────────────────────────────────────────────────────────
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audioLOC.pt')
torch.save({
    'model_state':  model.state_dict(),
    'scaler_mean':  scaler.mean_,
    'scaler_std':   scaler.scale_,
    'feature_cols': FEATURE_COLS,
}, save_path)
print(f'\nModel saved: {save_path}')

# ── Summary ───────────────────────────────────────────────────────────────────
summary = [
    f'\n{"="*55}', '  RESULTS', f'{"="*55}',
    f'  Chunk size  : {CHUNK_MS}ms',
    f'  Train frames: {len(X_train)}',
    f'  Val frames  : {len(X_val)}',
    f'  Test frames : {len(X_test)}',
    f'  Accuracy    : {acc:.2f}%',
    f'  MAE         : {mae:.1f}°',
    f'  RMSE        : {rmse:.1f}°',
    f'{"="*55}',
]
print('\n'.join(summary))
summary_path = os.path.join(RESULTS_DIR, f'summary_{FEATURE_TAG}.txt')
with open(summary_path, 'w') as f:
    f.write('\n'.join(summary) + '\n')

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Localization CNN — {CHUNK_MS}ms chunks  |  Acc={acc:.1f}%  MAE={mae:.1f}°', fontsize=12)

ax = axes[0, 0]
ax.plot(tr_hist,  label='Train')
ax.plot(val_hist, label='Val')
ax.plot(te_hist,  label='Test')
ax.set_title('Accuracy'); ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(loss_hist, color='coral')
ax.set_title('Training Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
cm = confusion_matrix(y_te_arr, preds)
ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels(ANGLES, rotation=90, fontsize=6)
ax.set_yticklabels(ANGLES, fontsize=6)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix ({acc:.1f}%)')
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        if cm[i, j] > 0:
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=4,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

ax = axes[1, 1]
class_acc = [(preds[y_te_arr == i] == i).mean() * 100
             if (y_te_arr == i).any() else 0 for i in range(N_CLASSES)]
ax.bar(ANGLES, class_acc, width=12, color='steelblue', edgecolor='white')
ax.axhline(acc, color='red', linestyle='--', label=f'overall={acc:.1f}%')
ax.set_xlabel('Angle (°)'); ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 110)
ax.set_title('Per-Angle Accuracy')
ax.set_xticks(ANGLES); ax.set_xticklabels(ANGLES, rotation=90, fontsize=7)
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, f'results_{FEATURE_TAG}.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f'Plot saved: {plot_path}')
