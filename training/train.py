"""
Localization CNN — Train on E1, Test on E1
Input: 19 extracted features per 2s chunk
Output: 8 classes (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for running as script
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_DIR = r'C:\Users\ahmma\Desktop\farah\features'
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 1e-3
DEVICE       = 'cpu'  # RTX 5050 (sm_120) not yet supported by PyTorch 2.5
ANGLES       = [0, 45, 90, 135, 180, 225, 270, 315]

FEATURE_COLS = [
    'rms_mic_right', 'rms_mic_front', 'rms_mic_left', 'rms_mic_back',
    'ipd_pair0', 'ipd_pair1', 'ipd_pair2',
    'gcc_tdoa_0', 'gcc_tdoa_1', 'gcc_tdoa_2',
    'gcc_tdoa_3', 'gcc_tdoa_4', 'gcc_tdoa_5',
    'gcc_strength_0', 'gcc_strength_1', 'gcc_strength_2',
    'gcc_strength_3', 'gcc_strength_4', 'gcc_strength_5',
]

print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ── Load data ─────────────────────────────────────────────────────────────────
train_df = pd.read_csv(f'{FEATURES_DIR}/train_E1.csv')
test_df  = pd.read_csv(f'{FEATURES_DIR}/test_E1.csv')

X_train = train_df[FEATURE_COLS].values.astype(np.float32)
y_train = train_df['label'].values.astype(np.int64)
X_test  = test_df[FEATURE_COLS].values.astype(np.float32)
y_test  = test_df['label'].values.astype(np.int64)

# Normalize using train statistics only
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f'Train: {X_train.shape}  |  Test: {X_test.shape}')

# Reshape to (batch, 1, 19) for 1D CNN
X_train_t = torch.tensor(X_train).unsqueeze(1).to(DEVICE)
y_train_t = torch.tensor(y_train).to(DEVICE)
X_test_t  = torch.tensor(X_test).unsqueeze(1).to(DEVICE)
y_test_t  = torch.tensor(y_test).to(DEVICE)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE, shuffle=True
)

# ── Model ─────────────────────────────────────────────────────────────────────
class LocalizationCNN(nn.Module):
    def __init__(self, n_features=19, n_classes=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model     = LocalizationCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

total_params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {total_params:,}')

# ── Training ──────────────────────────────────────────────────────────────────
train_acc_hist, val_acc_hist, loss_hist = [], [], []

for epoch in range(EPOCHS):
    model.train()
    correct, total, total_loss = 0, 0, 0.0

    for xb, yb in train_loader:
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
        val_out = model(X_test_t)
        val_acc = (val_out.argmax(1) == y_test_t).float().mean().item()

    train_acc = correct / total
    avg_loss  = total_loss / total
    train_acc_hist.append(train_acc)
    val_acc_hist.append(val_acc)
    loss_hist.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1:>3}/{EPOCHS}  loss={avg_loss:.4f}  '
              f'train_acc={train_acc*100:.1f}%  val_acc={val_acc*100:.1f}%')

# ── Evaluation ────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(1).cpu().numpy()

print(f'\n{"="*50}')
print(f'  Test Accuracy (E1 -> E1): {(preds == y_test).mean()*100:.2f}%')
print(f'{"="*50}')
print(classification_report(y_test, preds,
      target_names=[f'{a}°' for a in ANGLES]))

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(train_acc_hist, label='Train')
axes[0].plot(val_acc_hist,   label='Test (E1)')
axes[0].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy')
axes[0].legend(); axes[0].grid(True, alpha=0.4)

axes[1].plot(loss_hist, color='coral')
axes[1].set(title='Training Loss', xlabel='Epoch', ylabel='Loss')
axes[1].grid(True, alpha=0.4)

cm = confusion_matrix(y_test, preds)
im = axes[2].imshow(cm, cmap='Blues')
axes[2].set(title='Confusion Matrix',
            xticks=range(8), yticks=range(8),
            xticklabels=[f'{a}°' for a in ANGLES],
            yticklabels=[f'{a}°' for a in ANGLES])
axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('True')
plt.colorbar(im, ax=axes[2])
for i in range(8):
    for j in range(8):
        axes[2].text(j, i, cm[i,j], ha='center', va='center',
                     color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=8)

plt.suptitle('Localization CNN — Train E1 / Test E1', fontsize=13)
plt.tight_layout()
plt.savefig('results_E1_E1.png', dpi=150, bbox_inches='tight')
plt.show()
print('Plot saved: results_E1_E1.png')

# ── Save model ────────────────────────────────────────────────────────────────
torch.save({'model_state': model.state_dict(),
            'scaler_mean': scaler.mean_,
            'scaler_std':  scaler.scale_,
            'feature_cols': FEATURE_COLS}, 'model_E1.pt')
print('Model saved: model_E1.pt')
