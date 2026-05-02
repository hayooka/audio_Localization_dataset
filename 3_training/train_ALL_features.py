"""
Localization CNN — ALL features (735-D, Log-Mel removed)
Train  : full 5-min recordings  (train_DATA{tag}.csv)
Val    : first 1 min per angle  from 3-min recordings (test_DATA{tag}.csv)
Test   : last  2 min per angle  from 3-min recordings
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
import json
from datetime import datetime

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience   = patience
        self.best_acc   = -1.0
        self.counter    = 0
        self.best_model = None
    def __call__(self, val_acc, model):
        if val_acc > self.best_acc:
            self.best_acc   = val_acc
            self.counter    = 0
            self.best_model = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(REPO_ROOT, '0_Dataset', 'features')
PLOTS_DIR    = os.path.join(REPO_ROOT, '4_results', 'plots', 'CNN')
MODELS_DIR   = os.path.join(REPO_ROOT, '4_results', 'models', 'CNN')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CHUNK_TAG   = '16'           # matches train_DATA16.csv / test_DATA16.csv
CHUNK_MS    = 16             # 16 ms
VAL_SEC     = 60             # first 60s of 3-min recording = validation

EPOCHS      = 50
BATCH_SIZE  = 64
LR          = 1e-3
ANGLES      = list(range(0, 360, 15))
N_CLASSES   = len(ANGLES)    # 24
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
FEATURE_TAG = 'ALL16'

_MICS         = ['mic_right', 'mic_front', 'mic_left', 'mic_back']
_N_MELS       = 40
_GCC_VEC_SIZE = 100
_N_PAIRS      = 6

# ── Tunable settings — edit here ──────────────────────────────────────────────
RMS_THRESHOLD       = 50.0   # lower = more frames kept
REMOVE_LOGMEL       = True   # True = drop Log-Mel features before training
EARLY_STOP_PATIENCE = 10     # stop if no val improvement for this many epochs
SEQ_LEN             = 2      # context window in frames (2 × 16ms = 32ms)
STRIDE              = 1      # 1 = 50% overlap for seq_len=2
# ──────────────────────────────────────────────────────────────────────────────

_LOGMEL_COLS = [f'logmel_{m}_b{b}' for m in _MICS for b in range(_N_MELS)]

FEATURE_COLS = [
    'ipd_pair0', 'ipd_pair1', 'ipd_pair2',
    *[f'ipd_mel_{i}_b{b}'   for i in range(3)       for b in range(_N_MELS)],
    *[f'gcc_tdoa_{i}'       for i in range(6)],
    *[f'gcc_strength_{i}'   for i in range(6)],
    *[f'gcc_vec_{i}_t{t}'   for i in range(_N_PAIRS) for t in range(_GCC_VEC_SIZE)],
    *(() if REMOVE_LOGMEL else _LOGMEL_COLS),
]

RMS_COLS = [f'rms_{m}' for m in _MICS]

# ── Sequence Dataset ──────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, X, y, chunks, seq_len, stride=1):
        self.X      = torch.tensor(X, dtype=torch.float32)
        self.y      = torch.tensor(y, dtype=torch.long)
        self.chunks = chunks
        self.seq_len = seq_len
        self.starts = self._make_starts(stride)

    def _make_starts(self, stride):
        starts = []
        for i in range(0, len(self.y) - self.seq_len + 1, stride):
            window = np.arange(i, i + self.seq_len)
            if not np.all(self.y[window].numpy() == self.y[i].item()):
                continue
            if not np.all(np.diff(self.chunks[window]) == 1):
                continue
            starts.append(i)
        return starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        x = self.X[s:s + self.seq_len].reshape(-1)  # flatten: seq_len × n_features
        return x, self.y[s + self.seq_len - 1]

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
        return self.fc(self.conv(x.unsqueeze(1)))

# ── Helpers ───────────────────────────────────────────────────────────────────
def angular_metrics(y_true, y_pred):
    td  = np.array([ANGLES[i] for i in y_true], dtype=np.float32)
    pd_ = np.array([ANGLES[i] for i in y_pred], dtype=np.float32)
    diff = np.minimum(np.abs(td - pd_), 360 - np.abs(td - pd_))
    return float(np.mean(diff)), float(np.sqrt(np.mean(diff ** 2)))

def eval_loader(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            all_preds.append(model(xb.to(DEVICE)).argmax(1).cpu())
            all_labels.append(yb)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()

def train_and_eval(train_ds, val_ds, test_ds):
    n_features   = SEQ_LEN * len(FEATURE_COLS)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=512,        shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512,        shuffle=False, pin_memory=True)

    model     = LocalizationCNN(n_features=n_features).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    es        = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    tr_hist, val_hist, te_hist, loss_hist = [], [], [], []

    for epoch in range(EPOCHS):
        model.train()
        correct, total, total_loss = 0, 0, 0.0
        for xb, yb in train_loader:
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

        val_preds, val_labels = eval_loader(model, val_loader)
        te_preds,  te_labels  = eval_loader(model, test_loader)
        val_acc = (val_preds == val_labels).mean()
        te_acc  = (te_preds  == te_labels).mean()
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

    preds, y_te = eval_loader(model, test_loader)
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
val_chunks_per_angle = int(VAL_SEC * 1000 / CHUNK_MS)

val_parts, test_parts = [], []
for lbl in sorted(test_df['label'].unique()):
    angle_df = test_df[test_df['label'] == lbl].sort_values('chunk')
    val_parts.append(angle_df.iloc[:val_chunks_per_angle])
    test_parts.append(angle_df.iloc[val_chunks_per_angle:])

val_df  = pd.concat(val_parts,  ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

X_train = train_df[FEATURE_COLS].values.astype(np.float32)
y_train = train_df['label'].to_numpy(dtype=np.int64)
c_train = train_df['chunk'].to_numpy(dtype=np.int64)
X_val   = val_df[FEATURE_COLS].values.astype(np.float32)
y_val   = val_df['label'].to_numpy(dtype=np.int64)
c_val   = val_df['chunk'].to_numpy(dtype=np.int64)
X_test  = test_df[FEATURE_COLS].values.astype(np.float32)
y_test  = test_df['label'].to_numpy(dtype=np.int64)
c_test  = test_df['chunk'].to_numpy(dtype=np.int64)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ── Build sequence datasets ───────────────────────────────────────────────────
train_ds = SequenceDataset(X_train, y_train, c_train, SEQ_LEN, STRIDE)
val_ds   = SequenceDataset(X_val,   y_val,   c_val,   SEQ_LEN, STRIDE)
test_ds  = SequenceDataset(X_test,  y_test,  c_test,  SEQ_LEN, STRIDE)

print(f'\n  Train sequences: {len(train_ds)}  (seq_len={SEQ_LEN}, stride={STRIDE})')
print(f'  Val   sequences: {len(val_ds)}')
print(f'  Test  sequences: {len(test_ds)}')

# ── Train ─────────────────────────────────────────────────────────────────────
acc, preds, y_te_arr, tr_hist, val_hist, te_hist, loss_hist, model, mae, rmse = train_and_eval(
    train_ds, val_ds, test_ds)

# ── Save model ────────────────────────────────────────────────────────────────
save_path = os.path.join(MODELS_DIR, 'audioLOC_CNN.pt')
torch.save({
    'model_state':  model.state_dict(),
    'scaler_mean':  scaler.mean_,
    'scaler_std':   scaler.scale_,
    'feature_cols': FEATURE_COLS,
    'seq_len':      SEQ_LEN,
}, save_path)
print(f'\nModel saved: {save_path}')

# ── Save history JSON ─────────────────────────────────────────────────────────
run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
history = [
    {'epoch': i + 1, 'loss': loss_hist[i], 'train_acc': tr_hist[i], 'val_acc': val_hist[i]}
    for i in range(len(loss_hist))
]
result_data = {
    'run_id': run_id,
    'best_val_acc': float(max(val_hist)) * 100,
    'test_acc': float(acc),
    'mae': float(mae),
    'rmse': float(rmse),
    'history': history,
    'config': {
        'chunk_ms': CHUNK_MS,
        'seq_len': SEQ_LEN,
        'stride': STRIDE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'patience': EARLY_STOP_PATIENCE,
        'rms_threshold': RMS_THRESHOLD,
        'remove_logmel': REMOVE_LOGMEL,
    },
}
json_path = os.path.join(PLOTS_DIR, f'result_{run_id}.json')
with open(json_path, 'w') as f:
    json.dump(result_data, f, indent=2)
print(f'History saved: {json_path}')

# ── Summary ───────────────────────────────────────────────────────────────────
summary = [
    f'\n{"="*55}', '  RESULTS', f'{"="*55}',
    f'  Chunk size  : {CHUNK_MS}ms',
    f'  Seq len     : {SEQ_LEN} frames ({SEQ_LEN*CHUNK_MS}ms context)',
    f'  Stride      : {STRIDE}',
    f'  Train seqs  : {len(train_ds)}',
    f'  Val seqs    : {len(val_ds)}',
    f'  Test seqs   : {len(test_ds)}',
    f'  Accuracy    : {acc:.2f}%',
    f'  MAE         : {mae:.1f}°',
    f'  RMSE        : {rmse:.1f}°',
    f'{"="*55}',
]
print('\n'.join(summary))
summary_path = os.path.join(PLOTS_DIR, f'summary_{FEATURE_TAG}.txt')
with open(summary_path, 'w') as f:
    f.write('\n'.join(summary) + '\n')

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Localization CNN — {CHUNK_MS}ms  seq={SEQ_LEN}  |  Acc={acc:.1f}%  MAE={mae:.1f}°', fontsize=12)

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
plot_path = os.path.join(PLOTS_DIR, f'results_{FEATURE_TAG}.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f'Plot saved: {plot_path}')
