import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

ANGLES = list(range(0, 360, 15))
AFFECTED = {270, 285, 300, 315, 330, 345}
OUT = 'realtime_test/realtime_test/gru-c-respeaker/individual_plots'
os.makedirs(OUT, exist_ok=True)

def round_to_15(angle):
    return round(angle / 15) * 15 % 360

def angular_error(a, b):
    diff = abs(a - b)
    return min(diff, 360 - diff)

BASE = 'realtime_test/realtime_test/gru-c-respeaker'
cnn_df = pd.read_csv(f'{BASE}/test_all_vs_rs_cnn.csv')
gru_df = pd.read_csv(f'{BASE}/test_all_vs_rs_gru.csv')

def load_fix_rs(path):
    df = pd.read_csv(path)
    mask = df['true_angle'].isin(AFFECTED)
    df.loc[mask, 'rs_raw']  = df.loc[mask, 'rs_raw'] + 256
    df.loc[mask, 'rs_pred'] = df.loc[mask, 'rs_raw'].apply(round_to_15)
    df['pred_angle'] = df['rs_pred']
    return df

rs1 = load_fix_rs('realtime_test/realtime_test/ressultcnn16ms/test_cnn_vs_rs_rs.csv')
rs2 = load_fix_rs('realtime_test/realtime_test/resultgru/test_gru_vs_rs_rs.csv')
rs_df = pd.concat([rs1, rs2], ignore_index=True)

cnn_s = cnn_df[cnn_df['is_speech']==True].dropna(subset=['pred_angle']).copy()
gru_s = gru_df[gru_df['is_speech']==True].dropna(subset=['pred_angle']).copy()
rs_s  = rs_df[rs_df['is_speech']==True].copy()

for df in [cnn_s, gru_s, rs_s]:
    df['error'] = df.apply(lambda r: angular_error(int(r['pred_angle']), int(r['true_angle'])), axis=1)

def save(fname):
    path = f'{OUT}/{fname}'
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {fname}')

def confusion_matrix_plot(speech, tag, fname):
    cm = np.zeros((len(ANGLES), len(ANGLES)), dtype=int)
    for _, r in speech.iterrows():
        if int(r['pred_angle']) in ANGLES:
            cm[ANGLES.index(int(r['true_angle'])), ANGLES.index(int(r['pred_angle']))] += 1
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(ANGLES))); ax.set_yticks(range(len(ANGLES)))
    ax.set_xticklabels(ANGLES, rotation=90, fontsize=7)
    ax.set_yticklabels(ANGLES, fontsize=7)
    ax.set_xlabel('Predicted Angle'); ax.set_ylabel('True Angle')
    ax.set_title(f'{tag} -- Confusion Matrix')
    for i in range(len(ANGLES)):
        for j in range(len(ANGLES)):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=5,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
    save(fname)

def per_angle_exact(speech, tag, fname):
    total = len(speech)
    correct = (speech['error'] == 0).sum()
    accs = []
    for a in ANGLES:
        af = speech[speech['true_angle'] == a]
        accs.append((af['error'] == 0).sum() / max(len(af), 1) * 100)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(ANGLES)), accs, color='steelblue')
    ax.axhline(correct / total * 100, color='red', linestyle='--',
               label=f'Overall {correct/total*100:.1f}%')
    ax.set_xticks(range(len(ANGLES))); ax.set_xticklabels(ANGLES, rotation=45, fontsize=8)
    ax.set_xlabel('Angle (deg)'); ax.set_ylabel('Exact Accuracy (%)')
    ax.set_title(f'{tag} -- Per-Angle Exact Accuracy')
    ax.set_ylim(0, 110); ax.legend(); ax.grid(True, alpha=0.3)
    for i, v in enumerate(accs):
        if v > 0:
            ax.text(i, v + 1, f'{v:.0f}', ha='center', fontsize=6)
    save(fname)

def per_angle_15(speech, tag, fname):
    total = len(speech)
    w15 = (speech['error'] <= 15).sum()
    accs = []
    for a in ANGLES:
        af = speech[speech['true_angle'] == a]
        accs.append((af['error'] <= 15).sum() / max(len(af), 1) * 100)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(ANGLES)), accs, color='mediumseagreen')
    ax.axhline(w15 / total * 100, color='red', linestyle='--',
               label=f'Overall {w15/total*100:.1f}%')
    ax.set_xticks(range(len(ANGLES))); ax.set_xticklabels(ANGLES, rotation=45, fontsize=8)
    ax.set_xlabel('Angle (deg)'); ax.set_ylabel('+-15 deg Accuracy (%)')
    ax.set_title(f'{tag} -- Per-Angle +-15 deg Accuracy')
    ax.set_ylim(0, 110); ax.legend(); ax.grid(True, alpha=0.3)
    for i, v in enumerate(accs):
        if v > 0:
            ax.text(i, v + 1, f'{v:.0f}', ha='center', fontsize=6)
    save(fname)

def error_dist(speech, tag, fname):
    mae = speech['error'].mean()
    rmse = np.sqrt((speech['error'] ** 2).mean())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(speech['error'], bins=range(0, 196, 15), color='coral', edgecolor='white')
    ax.axvline(mae, color='navy', linestyle='--', label=f'MAE={mae:.1f} deg')
    ax.set_xlabel('Angular Error (deg)'); ax.set_ylabel('Frames')
    ax.set_title(f'{tag} -- Angular Error Distribution   RMSE={rmse:.1f} deg')
    ax.legend(); ax.grid(True, alpha=0.3)
    save(fname)

def latency_hist(df, tag, fname):
    lat = df[df['is_speech'] == True]['latency_ms']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lat, bins=30, color='orchid', edgecolor='white')
    ax.axvline(lat.mean(), color='navy', linestyle='--', label=f'Mean={lat.mean():.1f}ms')
    ax.set_xlabel('Latency (ms)'); ax.set_ylabel('Frames')
    ax.set_title(f'{tag} -- Inference Latency')
    ax.legend(); ax.grid(True, alpha=0.3)
    save(fname)

def comparison_per_angle(s_dict, metric, ylabel, title, fname, colors):
    fig, ax = plt.subplots(figsize=(12, 5))
    for (tag, speech), color in zip(s_dict.items(), colors):
        vals = []
        for a in ANGLES:
            af = speech[speech['true_angle'] == a]
            if metric == 'exact':
                vals.append((af['error'] == 0).sum() / max(len(af), 1) * 100)
            else:
                vals.append((af['error'] <= 15).sum() / max(len(af), 1) * 100)
        ax.plot(range(len(ANGLES)), vals, marker='o', markersize=4, label=tag, color=color)
    ax.set_xticks(range(len(ANGLES))); ax.set_xticklabels(ANGLES, rotation=45, fontsize=8)
    ax.set_xlabel('Angle (deg)'); ax.set_ylabel(ylabel)
    ax.set_ylim(0, 110); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    save(fname)

def summary_bars(fname):
    models = ['CNN', 'GRU', 'ReSpeaker']
    speeches = [cnn_s, gru_s, rs_s]
    exact = [(s['error'] == 0).mean() * 100 for s in speeches]
    w15   = [(s['error'] <= 15).mean() * 100 for s in speeches]
    mae   = [s['error'].mean() for s in speeches]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Real-Time Localization -- CNN vs GRU vs ReSpeaker', fontsize=13)
    x = np.arange(3); w = 0.35
    ax = axes[0]
    b1 = ax.bar(x - w/2, exact, w, label='Exact Acc', color='steelblue')
    b2 = ax.bar(x + w/2, w15,   w, label='+-15 Acc',  color='mediumseagreen')
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0, 110)
    ax.set_ylabel('Accuracy (%)'); ax.set_title('Accuracy Comparison')
    ax.legend(); ax.grid(True, alpha=0.3)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', fontsize=9)
    ax = axes[1]
    bars = ax.bar(x, mae, color=['steelblue', 'mediumseagreen', 'coral'])
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel('MAE (deg)'); ax.set_title('Mean Angular Error')
    ax.grid(True, alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f} deg', ha='center', fontsize=9)
    save(fname)

# ── Generate all ──────────────────────────────────────────────────────────────
confusion_matrix_plot(cnn_s, 'CNN (seq=2)',   'cnn_confusion_matrix.png')
confusion_matrix_plot(gru_s, 'GRU (seq=32)',  'gru_confusion_matrix.png')
confusion_matrix_plot(rs_s,  'ReSpeaker DOA', 'rs_confusion_matrix.png')

per_angle_exact(cnn_s, 'CNN (seq=2)',   'cnn_per_angle_exact.png')
per_angle_exact(gru_s, 'GRU (seq=32)',  'gru_per_angle_exact.png')
per_angle_exact(rs_s,  'ReSpeaker DOA', 'rs_per_angle_exact.png')

per_angle_15(cnn_s, 'CNN (seq=2)',   'cnn_per_angle_15.png')
per_angle_15(gru_s, 'GRU (seq=32)',  'gru_per_angle_15.png')
per_angle_15(rs_s,  'ReSpeaker DOA', 'rs_per_angle_15.png')

error_dist(cnn_s, 'CNN (seq=2)',   'cnn_error_dist.png')
error_dist(gru_s, 'GRU (seq=32)',  'gru_error_dist.png')
error_dist(rs_s,  'ReSpeaker DOA', 'rs_error_dist.png')

latency_hist(cnn_df, 'CNN (seq=2)',   'cnn_latency.png')
latency_hist(gru_df, 'GRU (seq=32)',  'gru_latency.png')
latency_hist(rs_df,  'ReSpeaker DOA', 'rs_latency.png')

comparison_per_angle(
    {'CNN': cnn_s, 'GRU': gru_s, 'ReSpeaker': rs_s}, 'exact',
    'Exact Accuracy (%)', 'Per-Angle Exact Accuracy -- CNN vs GRU vs ReSpeaker',
    'compare_per_angle_exact_all3.png', ['steelblue', 'mediumseagreen', 'coral'])

comparison_per_angle(
    {'CNN': cnn_s, 'GRU': gru_s, 'ReSpeaker': rs_s}, '15',
    '+-15 deg Accuracy (%)', 'Per-Angle +-15 Accuracy -- CNN vs GRU vs ReSpeaker',
    'compare_per_angle_15_all3.png', ['steelblue', 'mediumseagreen', 'coral'])

comparison_per_angle(
    {'CNN': cnn_s, 'GRU': gru_s}, 'exact',
    'Exact Accuracy (%)', 'Per-Angle Exact Accuracy -- CNN vs GRU',
    'cnn_vs_gru_per_angle_exact.png', ['steelblue', 'mediumseagreen'])

comparison_per_angle(
    {'CNN': cnn_s, 'GRU': gru_s}, '15',
    '+-15 deg Accuracy (%)', 'Per-Angle +-15 Accuracy -- CNN vs GRU',
    'cnn_vs_gru_per_angle_15.png', ['steelblue', 'mediumseagreen'])

summary_bars('summary_all_three.png')

print('\nAll plots saved to:', OUT)
