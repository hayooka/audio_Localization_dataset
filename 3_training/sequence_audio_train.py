# -*- coding: utf-8 -*-
"""
Sequence model for 16ms ALL-feature audio localization.

Option 1: use the extracted frame-level feature vectors directly as a sequence:
  CSV features -> consecutive frame windows -> GRU -> angle class

Default split:
  train_DATA16.csv: Audio1. A short time block per angle is held out for validation.
  test_DATA16.csv: Audio2. The full file is used for final testing only.
"""

import argparse
import copy
import json
import os
from datetime import datetime
from typing import cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(REPO_ROOT, "0_Dataset", "features")
RESULTS_DIR = os.path.join(REPO_ROOT, "4_results", "plots", "GRU")
MODELS_DIR  = os.path.join(REPO_ROOT, "4_results", "models", "GRU")

CHUNK_TAG = "16"
ANGLES = list(range(0, 360, 15))
N_CLASSES = len(ANGLES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

MICS = ["mic_right", "mic_front", "mic_left", "mic_back"]
N_MELS = 40
GCC_VEC_SIZE = 100
N_PAIRS = 6

# ── Tunable settings — edit here ──────────────────────────────────────────────
RMS_THRESHOLD = 50.0    # lower = more frames kept (was 100.0)
REMOVE_LOGMEL = False  # True = drop Log-Mel features before training
STRIDE        = 16      # training stride: 16=50% overlap, 8=75% overlap
PATIENCE      = 8       # stop if no val improvement for this many epochs
# ──────────────────────────────────────────────────────────────────────────────

RMS_COLS = ["rms_mic_right", "rms_mic_front", "rms_mic_left", "rms_mic_back"]

_LOGMEL_COLS = [f"logmel_{mic}_b{b}" for mic in MICS for b in range(N_MELS)]

FEATURE_COLS = [
    "ipd_pair0", "ipd_pair1", "ipd_pair2",
    *[f"ipd_mel_{i}_b{b}" for i in range(3) for b in range(N_MELS)],
    *[f"gcc_tdoa_{i}" for i in range(6)],
    *[f"gcc_strength_{i}" for i in range(6)],
    *[f"gcc_vec_{i}_t{t}" for i in range(N_PAIRS) for t in range(GCC_VEC_SIZE)],
    *(() if REMOVE_LOGMEL else _LOGMEL_COLS),
]
READ_COLS = ["dataset", "position", "chunk", "label", *RMS_COLS, *FEATURE_COLS]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, X, y, starts, seq_len):
        self.X = X
        self.y = y
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        end = start + self.seq_len
        return torch.from_numpy(self.X[start:end]), torch.tensor(self.y[start], dtype=torch.long)


class SequenceGRU(nn.Module):
    def __init__(self, n_features, embed_dim=256, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, N_CLASSES),
        )

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])


def angular_metrics(y_true, y_pred):
    true_deg = np.array([ANGLES[i] for i in y_true], dtype=np.float32)
    pred_deg = np.array([ANGLES[i] for i in y_pred], dtype=np.float32)
    diff = np.abs(true_deg - pred_deg)
    diff = np.minimum(diff, 360 - diff)
    return float(np.mean(diff)), float(np.sqrt(np.mean(diff ** 2)))


def load_csvs(args):
    train_path = os.path.join(FEATURES_DIR, f"train_DATA{CHUNK_TAG}.csv")
    test_path = os.path.join(FEATURES_DIR, f"test_DATA{CHUNK_TAG}.csv")

    print(f"Loading train CSV: {train_path}")
    train_df = pd.read_csv(train_path, usecols=READ_COLS)
    print(f"Loading test CSV:  {test_path}")
    test_df = pd.read_csv(test_path, usecols=READ_COLS)

    for name, df in [("train", train_df), ("test", test_df)]:
        before = len(df)
        mask = (df[RMS_COLS] >= args.rms_threshold).all(axis=1)
        df.drop(index=df.index[~mask], inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"  {name}: dropped {before - len(df)} silent frames, kept {len(df)}")

    return train_df, test_df


def split_test_df(test_df):
    chunks_per_second = int(1000 / int(CHUNK_TAG))
    val_chunks_per_angle = 60 * chunks_per_second
    val_parts = []
    test_parts = []

    for label in sorted(test_df["label"].unique()):
        angle_df = test_df[test_df["label"] == label].sort_values("chunk")
        if len(angle_df) <= val_chunks_per_angle:
            raise ValueError(
                f"Not enough 3-minute chunks for label {label}: "
                f"{len(angle_df)} available, need more than {val_chunks_per_angle}"
            )
        val_parts.append(angle_df.iloc[:val_chunks_per_angle])
        test_parts.append(angle_df.iloc[val_chunks_per_angle:])

    return pd.concat(val_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def split_audio1_blocks(train_df, args):
    train_parts = []
    val_parts = []
    chunks_per_second = int(1000 / int(CHUNK_TAG))
    val_chunks = int(round(args.audio1_val_seconds * chunks_per_second))

    for label in sorted(train_df["label"].unique()):
        angle_df = train_df[train_df["label"] == label].sort_values("chunk")
        if len(angle_df) <= val_chunks:
            raise ValueError(
                f"Not enough Audio1 chunks for label {label}: "
                f"{len(angle_df)} available, need more than {val_chunks}"
            )

        if args.audio1_val_block == "start":
            val_df = angle_df.iloc[:val_chunks]
            tr_df = angle_df.iloc[val_chunks:]
        elif args.audio1_val_block == "end":
            val_df = angle_df.iloc[-val_chunks:]
            tr_df = angle_df.iloc[:-val_chunks]
        else:
            start = max((len(angle_df) - val_chunks) // 2, 0)
            end = start + val_chunks
            val_df = angle_df.iloc[start:end]
            tr_df = pd.concat([angle_df.iloc[:start], angle_df.iloc[end:]])

        train_parts.append(tr_df)
        val_parts.append(val_df)

    return pd.concat(train_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True)


def limit_per_class(df, max_per_class):
    if not max_per_class:
        return df
    parts = []
    for label in sorted(df["label"].unique()):
        parts.append(df[df["label"] == label].head(max_per_class))
    return pd.concat(parts, ignore_index=True)


def frame_arrays(train_df, val_df, test_df, args):
    train_df = limit_per_class(train_df, args.max_train_per_class)
    val_df = limit_per_class(val_df, args.max_val_per_class)
    test_df = limit_per_class(test_df, args.max_test_per_class)

    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    train_chunks = train_df["chunk"].to_numpy(dtype=np.int64)
    train_positions = train_df["position"].to_numpy(dtype=np.int64)
    X_val = val_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val = val_df["label"].to_numpy(dtype=np.int64)
    val_chunks = val_df["chunk"].to_numpy(dtype=np.int64)
    val_positions = val_df["position"].to_numpy(dtype=np.int64)
    X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int64)
    test_chunks = test_df["chunk"].to_numpy(dtype=np.int64)
    test_positions = test_df["position"].to_numpy(dtype=np.int64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    train_meta = (y_train, train_chunks, train_positions)
    val_meta = (y_val, val_chunks, val_positions)
    test_meta = (y_test, test_chunks, test_positions)
    return X_train, y_train, train_meta, X_val, y_val, val_meta, X_test, y_test, test_meta, scaler


def make_sequence_starts(labels, chunks, positions, seq_len, stride, strict_consecutive=True):
    starts = []
    labels = np.asarray(labels)
    chunks = np.asarray(chunks)
    positions = np.asarray(positions)

    for position in sorted(np.unique(positions)):
        idx = np.flatnonzero(positions == position)
        if len(idx) < seq_len:
            continue

        # Work inside one original angle/position only. With strict mode, also
        # require chunk numbers to be consecutive so silence-filtered gaps are
        # not stitched into fake continuous sequences.
        for offset in range(0, len(idx) - seq_len + 1, stride):
            window_idx = idx[offset:offset + seq_len]
            start = int(window_idx[0])
            if not np.all(np.diff(window_idx) == 1):
                continue
            if not np.all(labels[window_idx] == labels[start]):
                continue
            if strict_consecutive and not np.all(np.diff(chunks[window_idx]) == 1):
                continue
            starts.append(start)
    return np.asarray(starts, dtype=np.int64)


def make_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


def eval_model(model, loader):
    preds = []
    labels = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            preds.append(model(xb).argmax(1).cpu())
            labels.append(yb)
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    return float((y_pred == y_true).mean()), y_true, y_pred


def train(args):
    train_df, test_df = load_csvs(args)
    if args.split_mode == "audio1_val_audio2_test":
        train_df, val_df = split_audio1_blocks(train_df, args)
        test_split_df = test_df
        print("Split mode: Audio1 train/validation block split; full Audio2 as final test")
        print(f"  Audio1 validation block: {args.audio1_val_seconds:g}s per angle from {args.audio1_val_block}")
    else:
        val_df, test_split_df = split_test_df(test_df)
        print("Split mode: first minute of Audio2 for validation; remaining Audio2 for test")

    X_train, y_train, train_meta, X_val, y_val, val_meta, X_test, y_test, test_meta, scaler = frame_arrays(
        train_df, val_df, test_split_df, args
    )

    train_starts = make_sequence_starts(*train_meta, args.seq_len, args.stride, args.strict_consecutive)
    val_starts = make_sequence_starts(*val_meta, args.seq_len, args.eval_stride, args.strict_consecutive)
    test_starts = make_sequence_starts(*test_meta, args.seq_len, args.eval_stride, args.strict_consecutive)

    y_train_ds = y_train.copy()
    y_test_ds = y_test.copy()
    rng = np.random.default_rng(SEED)
    if args.shuffle_train_labels:
        shuffled = y_train_ds[train_starts].copy()
        rng.shuffle(shuffled)
        y_train_ds[train_starts] = shuffled
        print("SANITY CHECK: training sequence labels were shuffled. Accuracy should collapse to chance.")
    if args.shuffle_test_labels:
        shuffled = y_test_ds[test_starts].copy()
        rng.shuffle(shuffled)
        y_test_ds[test_starts] = shuffled
        print("SANITY CHECK: test sequence labels were shuffled. Reported test accuracy should collapse to chance.")

    print("Frame sizes:")
    print(f"  Train frames: {len(X_train)}")
    print(f"  Val frames:   {len(X_val)}")
    print(f"  Test frames:  {len(X_test)}")
    print("Sequence sizes:")
    print(f"  Train sequences: {len(train_starts)}")
    print(f"  Val sequences:   {len(val_starts)}")
    print(f"  Test sequences:  {len(test_starts)}")
    print(f"  Strict consecutive chunks: {args.strict_consecutive}")

    train_ds = SequenceDataset(X_train, y_train_ds, train_starts, args.seq_len)
    val_ds = SequenceDataset(X_val, y_val, val_starts, args.seq_len)
    test_ds = SequenceDataset(X_test, y_test_ds, test_starts, args.seq_len)

    train_loader = make_loader(train_ds, args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True)

    model = SequenceGRU(
        n_features=X_train.shape[1],
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = -1.0
    best_state = None
    best_epoch = 0
    patience = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        scheduler.step()
        train_acc = correct / total
        val_acc, _, _ = eval_model(model, val_loader)
        history.append({
            "epoch": epoch,
            "loss": total_loss / total,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"Epoch {epoch:>3}/{args.epochs} "
                f"loss={history[-1]['loss']:.4f} "
                f"train={train_acc*100:.1f}% "
                f"val={val_acc*100:.1f}% "
                f"lr={history[-1]['lr']:.2e}"
            )

        if patience >= args.patience:
            print(f"Early stop at epoch {epoch} (best val={best_val*100:.2f}% at epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    test_acc, y_true, y_pred = eval_model(model, test_loader)
    mae, rmse = angular_metrics(y_true, y_pred)
    print(f"FINAL: test={test_acc*100:.2f}% MAE={mae:.1f} RMSE={rmse:.1f}")
    save_outputs(args, model, scaler, history, best_val, best_epoch, test_acc, mae, rmse, y_true, y_pred)


def save_outputs(args, model, scaler, history, best_val, best_epoch, test_acc, mae, rmse, y_true, y_pred):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    prefix = args.run_id

    config = vars(args).copy()
    summary_path = os.path.join(RESULTS_DIR, f"summary_{prefix}.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Sequence GRU ALL-features model\n")
        handle.write(f"Best epoch: {best_epoch}\n")
        handle.write(f"Best validation accuracy: {best_val*100:.2f}%\n")
        handle.write(f"Test accuracy: {test_acc*100:.2f}%\n")
        handle.write(f"MAE: {mae:.1f} deg\n")
        handle.write(f"RMSE: {rmse:.1f} deg\n\n")
        handle.write(json.dumps(config, indent=2))
        handle.write("\n\n")
        report = cast(str, classification_report(
            y_true,
            y_pred,
            target_names=[f"{a}deg" for a in ANGLES],
            zero_division=0,
        ))
        handle.write(report)

    model_path = os.path.join(MODELS_DIR, f"audioLOC_sequence_{prefix}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_std": scaler.scale_,
        "feature_cols": FEATURE_COLS,
        "config": config,
        "metrics": {
            "best_val_acc": best_val * 100,
            "test_acc": test_acc * 100,
            "mae": mae,
            "rmse": rmse,
        },
    }, model_path)

    result_path = os.path.join(RESULTS_DIR, f"result_{prefix}.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_acc": best_val * 100,
            "test_acc": test_acc * 100,
            "mae": mae,
            "rmse": rmse,
            "history": history,
            "config": config,
        }, handle, indent=2)

    plot_path = os.path.join(RESULTS_DIR, f"plot_{prefix}.png")
    plot_outputs(history, test_acc, mae, rmse, y_true, y_pred, plot_path)

    print(f"Saved summary: {summary_path}")
    print(f"Saved model:   {model_path}")
    print(f"Saved plot:    {plot_path}")


def plot_outputs(history, test_acc, mae, rmse, y_true, y_pred, plot_path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].plot(epochs, [h["train_acc"] for h in history], label="Train")
    axes[0, 0].plot(epochs, [h["val_acc"] for h in history], label="Validation")
    axes[0, 0].set_title("Sequence Accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, [h["loss"] for h in history], color="coral")
    axes[0, 1].set_title("Training Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)

    cm = confusion_matrix(y_true, y_pred)
    im = axes[1, 0].imshow(cm, cmap="Blues")
    axes[1, 0].set_xticks(range(N_CLASSES))
    axes[1, 0].set_yticks(range(N_CLASSES))
    axes[1, 0].set_xticklabels([str(a) for a in ANGLES], rotation=90, fontsize=8)
    axes[1, 0].set_yticklabels([str(a) for a in ANGLES], fontsize=8)
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("True")
    axes[1, 0].set_title(f"Confusion Matrix ({test_acc*100:.1f}%)")
    plt.colorbar(im, ax=axes[1, 0])

    class_acc = []
    for label in range(N_CLASSES):
        mask = y_true == label
        class_acc.append(float((y_pred[mask] == label).mean() * 100) if mask.sum() else 0.0)
    axes[1, 1].bar(range(N_CLASSES), class_acc)
    axes[1, 1].set_title("Per-Class Accuracy")
    axes[1, 1].set_xlabel("Angle")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_xticks(range(0, N_CLASSES, 4))
    axes[1, 1].set_xticklabels([str(ANGLES[i]) for i in range(0, N_CLASSES, 4)], rotation=45)
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Sequence GRU | Test {test_acc*100:.2f}% | MAE {mae:.1f} deg | RMSE {rmse:.1f} deg")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GRU over sequences of 16ms ALL features.")
    parser.add_argument("--seq-len", type=int, default=32, help="Frames per sequence. 32 frames = 512ms.")
    parser.add_argument("--stride", type=int, default=STRIDE, help="Training sequence stride in frames.")
    parser.add_argument("--eval-stride", type=int, default=32, help="Validation/test stride in frames.")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rms-threshold", type=float, default=RMS_THRESHOLD)
    parser.add_argument(
        "--strict-consecutive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require every sequence to use consecutive chunk ids after silence filtering.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["audio1_val_audio2_test", "audio2_val_test"],
        default="audio2_val_test",
        help="Main evaluation uses Audio1 block validation and full Audio2 test.",
    )
    parser.add_argument("--audio1-val-seconds", type=float, default=30.0)
    parser.add_argument("--audio1-val-block", choices=["start", "middle", "end"], default="end")
    parser.add_argument("--max-train-per-class", type=int, default=0, help="Use for quick smoke tests only.")
    parser.add_argument("--max-val-per-class", type=int, default=0, help="Use for quick smoke tests only.")
    parser.add_argument("--max-test-per-class", type=int, default=0, help="Use for quick smoke tests only.")
    parser.add_argument("--shuffle-train-labels", action="store_true", help="Sanity check: train on random labels.")
    parser.add_argument("--shuffle-test-labels", action="store_true", help="Sanity check: evaluate against random labels.")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(json.dumps(vars(args), indent=2))
    train(args)


if __name__ == "__main__":
    main()
