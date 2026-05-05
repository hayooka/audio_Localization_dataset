"""
Re-evaluate saved CNN and GRU models on the test set.
Saves per-class and overall metrics (accuracy, precision, recall, F1, loss) to JSON.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, classification_report
import json
import os

REPO         = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(REPO, '0_Dataset', 'features')
CNN_PT       = os.path.join(REPO, '4_Results', 'models', 'CNN', 'audioLOC_CNN.pt')
GRU_PT       = os.path.join(REPO, '4_Results', 'models', 'GRU', 'audioLOC_sequence_20260429_081236.pt')
OUT_DIR      = os.path.join(REPO, '4_Results', 'plots')
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

ANGLES    = list(range(0, 360, 15))
N_CLASSES = len(ANGLES)
RMS_COLS  = ['rms_mic_right', 'rms_mic_front', 'rms_mic_left', 'rms_mic_back']

# ── Shared dataset ─────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, X, y, chunks, seq_len, stride=1):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.long)
        self.chunks  = chunks
        self.seq_len = seq_len
        self.starts  = self._make_starts(stride)

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
        x = self.X[s:s + self.seq_len].reshape(-1)
        return x, self.y[s + self.seq_len - 1]

# ── CNN architecture ───────────────────────────────────────────────────────────
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

# ── GRU architecture ──────────────────────────────────────────────────────────
class SequenceGRU(nn.Module):
    def __init__(self, n_features, embed_dim=256, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, N_CLASSES),
        )
    def forward(self, x):
        # x: (B, seq_len * n_features) → (B, seq_len, n_features)
        B = x.size(0)
        n_feat = x.size(1) // self.seq_len if hasattr(self, 'seq_len') else x.size(1)
        # reshape handled externally — x is already (B, seq_len, n_features)
        e = self.embed(x)
        _, h = self.gru(e)
        return self.classifier(h[-1])

# ── Helpers ───────────────────────────────────────────────────────────────────
def angular_metrics(y_true, y_pred):
    td  = np.array([ANGLES[i] for i in y_true], dtype=np.float32)
    pd_ = np.array([ANGLES[i] for i in y_pred], dtype=np.float32)
    diff = np.minimum(np.abs(td - pd_), 360 - np.abs(td - pd_))
    return float(np.mean(diff)), float(np.sqrt(np.mean(diff ** 2)))

def compute_loss(model, loader, criterion):
    model.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            total_loss += criterion(out, yb).item() * len(yb)
            total += len(yb)
    return total_loss / total

def eval_loader(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(DEVICE)).argmax(1).cpu())
            labels.append(yb)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()

def compute_metrics(y_true, y_pred, model, loader, criterion):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    acc  = float((y_true == y_pred).mean() * 100)
    mae, rmse = angular_metrics(y_true, y_pred)
    loss = compute_loss(model, loader, criterion)
    return {
        'test_acc':   acc,
        'macro_precision': float(mp),
        'macro_recall':    float(mr),
        'macro_f1':        float(mf),
        'test_loss':  loss,
        'mae':        mae,
        'rmse':       rmse,
        'per_class': [
            {
                'angle':     ANGLES[i],
                'precision': float(p[i]),
                'recall':    float(r[i]),
                'f1':        float(f[i]),
            }
            for i in range(N_CLASSES)
        ],
    }

# ── Load test data (shared) ───────────────────────────────────────────────────
print('Loading data...')
test_df = pd.read_csv(os.path.join(FEATURES_DIR, 'test_DATA16.csv'))
mask = (test_df[RMS_COLS] >= 50.0).all(axis=1)
test_df = test_df[mask].reset_index(drop=True)

VAL_SEC   = 60
CHUNK_MS  = 16
val_chunks_per_angle = int(VAL_SEC * 1000 / CHUNK_MS)

val_parts, test_parts = [], []
for lbl in sorted(test_df['label'].unique()):
    angle_df = test_df[test_df['label'] == lbl].sort_values('chunk')
    val_parts.append(angle_df.iloc[:val_chunks_per_angle])
    test_parts.append(angle_df.iloc[val_chunks_per_angle:])
test_df = pd.concat(test_parts, ignore_index=True)

# ── Evaluate CNN ──────────────────────────────────────────────────────────────
print('\nEvaluating CNN...')
cnn_ckpt     = torch.load(CNN_PT, map_location=DEVICE, weights_only=False)
feature_cols = cnn_ckpt['feature_cols']
seq_len_cnn  = cnn_ckpt['seq_len']

scaler = StandardScaler()
scaler.mean_  = cnn_ckpt['scaler_mean']
scaler.scale_ = cnn_ckpt['scaler_std']

X_test = scaler.transform(test_df[feature_cols].values.astype(np.float32))
y_test = test_df['label'].to_numpy(dtype=np.int64)
c_test = test_df['chunk'].to_numpy(dtype=np.int64)

test_ds  = SequenceDataset(X_test, y_test, c_test, seq_len_cnn, stride=seq_len_cnn)
test_ldr = DataLoader(test_ds, batch_size=512, shuffle=False)

n_features = seq_len_cnn * len(feature_cols)
cnn_model  = LocalizationCNN(n_features=n_features).to(DEVICE)
cnn_model.load_state_dict(cnn_ckpt['model_state'])
cnn_model.eval()

criterion = nn.CrossEntropyLoss()
y_pred, y_true = eval_loader(cnn_model, test_ldr)
cnn_metrics = compute_metrics(y_true, y_pred, cnn_model, test_ldr, criterion)
print(f"  CNN  — Acc={cnn_metrics['test_acc']:.2f}%  F1={cnn_metrics['macro_f1']:.3f}  MAE={cnn_metrics['mae']:.1f}°")

out_cnn = os.path.join(OUT_DIR, 'CNN', 'eval_metrics_CNN.json')
with open(out_cnn, 'w') as f:
    json.dump(cnn_metrics, f, indent=2)
print(f'  Saved: {out_cnn}')

# ── Evaluate GRU ──────────────────────────────────────────────────────────────
print('\nEvaluating GRU...')
gru_ckpt     = torch.load(GRU_PT, map_location=DEVICE, weights_only=False)
cfg          = gru_ckpt['config']
feature_cols = gru_ckpt['feature_cols']
seq_len_gru  = cfg['seq_len']
eval_stride  = cfg['eval_stride']

scaler2 = StandardScaler()
scaler2.mean_  = gru_ckpt['scaler_mean']
scaler2.scale_ = gru_ckpt['scaler_std']

X_test2 = scaler2.transform(test_df[feature_cols].values.astype(np.float32))

# GRU dataset: each item is (seq_len, n_features) → model expects (B, seq_len, n_features)
class GRUDataset(Dataset):
    def __init__(self, X, y, chunks, seq_len, stride):
        self.X, self.y, self.chunks = (torch.tensor(X, dtype=torch.float32),
                                       torch.tensor(y, dtype=torch.long),
                                       chunks)
        self.seq_len = seq_len
        self.starts  = self._make_starts(stride)

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

    def __len__(self): return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        return self.X[s:s + self.seq_len], self.y[s + self.seq_len - 1]

test_ds2  = GRUDataset(X_test2, y_test, c_test, seq_len_gru, eval_stride)
test_ldr2 = DataLoader(test_ds2, batch_size=512, shuffle=False)

n_feat_gru = len(feature_cols)
gru_model  = SequenceGRU(
    n_features=n_feat_gru,
    embed_dim=cfg['embed_dim'],
    hidden_dim=cfg['hidden_dim'],
    num_layers=cfg['layers'],
    dropout=cfg['dropout'],
).to(DEVICE)
gru_model.load_state_dict(gru_ckpt['model_state'])
gru_model.eval()

# GRU forward needs (B, seq_len, n_features) — override eval_loader
def eval_gru(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            e  = model.embed(xb)
            _, h = model.gru(e)
            out = model.classifier(h[-1])
            preds.append(out.argmax(1).cpu())
            labels.append(yb)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()

def compute_loss_gru(model, loader, criterion):
    model.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            e = model.embed(xb)
            _, h = model.gru(e)
            out = model.classifier(h[-1])
            total_loss += criterion(out, yb).item() * len(yb)
            total += len(yb)
    return total_loss / total

criterion_gru = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
y_pred2, y_true2 = eval_gru(gru_model, test_ldr2)

p, r, f, _ = precision_recall_fscore_support(y_true2, y_pred2, labels=list(range(N_CLASSES)), zero_division=0)
mp, mr, mf, _ = precision_recall_fscore_support(y_true2, y_pred2, average='macro', zero_division=0)
mae2, rmse2 = angular_metrics(y_true2, y_pred2)
loss2 = compute_loss_gru(gru_model, test_ldr2, criterion_gru)

gru_metrics = {
    'test_acc':        float((y_true2 == y_pred2).mean() * 100),
    'macro_precision': float(mp),
    'macro_recall':    float(mr),
    'macro_f1':        float(mf),
    'test_loss':       loss2,
    'mae':             mae2,
    'rmse':            rmse2,
    'per_class': [
        {'angle': ANGLES[i], 'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i])}
        for i in range(N_CLASSES)
    ],
}
print(f"  GRU  — Acc={gru_metrics['test_acc']:.2f}%  F1={gru_metrics['macro_f1']:.3f}  MAE={gru_metrics['mae']:.1f}°")

out_gru = os.path.join(OUT_DIR, 'GRU', 'eval_metrics_GRU.json')
with open(out_gru, 'w') as f:
    json.dump(gru_metrics, f, indent=2)
print(f'  Saved: {out_gru}')

print('\nDone.')
