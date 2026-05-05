"""
Generate separate real-time result plots from raw CSV logs.
Plots: 1) per-angle bar chart, 2) polar plot, 3) summary comparison bar
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE = r"C:\Users\ahmma\Downloads\New folder\realtime_test\realtime_test"
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '4_Results', 'plots')
os.makedirs(OUT, exist_ok=True)

ANGLES = list(range(0, 360, 15))
TOL    = 15  # +/-15 tolerance

# -- Load and aggregate per-angle ----------------------------------------------
def load_csv(path):
    df = pd.read_csv(path)
    if 'pred_angle' not in df.columns and 'rs_pred' in df.columns:
        df = df.rename(columns={'rs_pred': 'pred_angle'})
    df = df[df['is_speech'] == True].dropna(subset=['pred_angle'])
    df['pred_angle'] = df['pred_angle'].astype(int)
    return df

cnn_df = load_csv(os.path.join(BASE, 'gru-c-respeaker', 'test_all_vs_rs_cnn.csv'))
gru_df = load_csv(os.path.join(BASE, 'gru-c-respeaker', 'test_all_vs_rs_gru.csv'))
rs_df  = load_csv(os.path.join(BASE, 'ressultcnn16ms',  'test_cnn_vs_rs_rs.csv'))

# RS has a +255° offset for angles 255°-345° — correct those predictions
mask = rs_df['true_angle'].isin(range(255, 360, 15))
rs_df.loc[mask, 'pred_angle'] = (rs_df.loc[mask, 'pred_angle'] + 255) % 360

def per_angle_stats(df):
    rows = []
    for ang in ANGLES:
        sub = df[df['true_angle'] == ang]
        n   = len(sub)
        if n == 0:
            rows.append({'angle': ang, 'exact': 0.0, 'within15': 0.0, 'n': 0})
            continue
        exact = (sub['pred_angle'] == ang).mean() * 100
        diff  = sub['pred_angle'].apply(lambda p: min(abs(p - ang), 360 - abs(p - ang)))
        w15   = (diff <= TOL).mean() * 100
        rows.append({'angle': ang, 'exact': exact, 'within15': w15, 'n': n})
    return pd.DataFrame(rows)

cnn = per_angle_stats(cnn_df)
gru = per_angle_stats(gru_df)
rs  = per_angle_stats(rs_df)

def overall(raw_df):
    exact = (raw_df['pred_angle'] == raw_df['true_angle']).mean() * 100
    diff  = raw_df.apply(lambda r: min(abs(r['pred_angle'] - r['true_angle']),
                                       360 - abs(r['pred_angle'] - r['true_angle'])), axis=1)
    w15   = (diff <= TOL).mean() * 100
    mae   = diff.mean()
    return exact, w15, mae

cnn_e, cnn_w, cnn_mae = overall(cnn_df)
gru_e, gru_w, gru_mae = overall(gru_df)
rs_e,  rs_w,  rs_mae  = overall(rs_df)

print(f"CNN  exact={cnn_e:.1f}%  +-15={cnn_w:.1f}%  MAE={cnn_mae:.1f}")
print(f"GRU  exact={gru_e:.1f}%  +-15={gru_w:.1f}%  MAE={gru_mae:.1f}")
print(f"RS   exact={rs_e:.1f}%   +-15={rs_w:.1f}%   MAE={rs_mae:.1f}")

x      = np.arange(len(ANGLES))
labels = [f'{a}' for a in ANGLES]

# -- Plot 1: Per-angle bar chart -----------------------------------------------
for name, stats, color_ex, color_w15 in [
    ('CNN',       cnn, '#2196F3', '#90CAF9'),
    ('GRU',       gru, '#4CAF50', '#A5D6A7'),
    ('ReSpeaker', rs,  '#FF9800', '#FFCC80'),
]:
    fig, ax = plt.subplots(figsize=(14, 5))
    w = 0.35
    ax.bar(x - w/2, stats['exact'],    width=w, label='Exact', color=color_ex,  edgecolor='white')
    ax.bar(x + w/2, stats['within15'], width=w, label='+-15°', color=color_w15, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a}°' for a in ANGLES], rotation=90, fontsize=7)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 110)
    ax.set_title(f'{name} — Per-Angle Real-Time Accuracy (Exact & ±15°)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(OUT, f'realtime_per_angle_{name.replace(" ", "_")}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# -- Plot 2: Polar plot --------------------------------------------------------
for name, raw_df, color in [
    ('CNN',       cnn_df, 'steelblue'),
    ('GRU',       gru_df, 'green'),
    ('ReSpeaker', rs_df,  'orange'),
]:
    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection='polar')

    true_rad = np.deg2rad(raw_df['true_angle'].values.astype(float))
    pred_rad = np.deg2rad(raw_df['pred_angle'].values.astype(float))

    ax.scatter(true_rad, pred_rad, alpha=0.15, s=30, color=color)

    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(theta, theta, 'r--', linewidth=1, label='Perfect')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(45)
    ax.set_title(f'{name} — Predicted vs True Angle', va='bottom', pad=20)
    ax.legend(loc='lower right')
    plt.tight_layout()
    path = os.path.join(OUT, f'realtime_polar_{name.replace(" ", "_")}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

# -- Plot 3: Summary comparison bar --------------------------------------------
models  = ['CNN', 'GRU', 'ReSpeaker']
exact_v = [cnn_e,   gru_e,   rs_e]
w15_v   = [cnn_w,   gru_w,   rs_w]
mae_v   = [cnn_mae, gru_mae, rs_mae]
colors  = ['#2196F3', '#4CAF50', '#FF9800']

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle('Real-Time Localization — Overall Comparison', fontsize=12)

for ax, vals, title, ylabel in [
    (axes[0], exact_v, 'Exact Accuracy',    'Accuracy (%)'),
    (axes[1], w15_v,   '+-15 Accuracy',     'Accuracy (%)'),
    (axes[2], mae_v,   'Mean Angular Error', 'MAE (°)'),
]:
    bars = ax.bar(models, vals, color=colors, edgecolor='white', width=0.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    if 'Error' not in title:
        ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
path = os.path.join(OUT, 'realtime_summary_comparison.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {path}')

print('\nAll plots done.')
