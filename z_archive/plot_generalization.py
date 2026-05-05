"""
Two plots:
  A. MAE comparison — same room vs different room
  C. Scatter — true vs predicted angle (perfect = on diagonal)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data ──────────────────────────────────────────────────────────────────────
cross_room = [
    (  0,   0),
    ( 45,  45),
    ( 90,  90),
    (135, 150),
    (180, 195),
    (225, 255),
    (270, 285),
    (315, 300),
]
true_angles = np.array([t for t, _ in cross_room])
pred_angles = np.array([p for _, p in cross_room])
errors      = np.array([min(abs(p - t), 360 - abs(p - t)) for t, p in cross_room])
correct     = errors == 0

same_mae  = 5.9
same_acc      = 92.4
diff_mae      = float(np.mean(errors))          # 11.25°
diff_acc_est  = float(np.mean(errors <= 15)) * 100  # 87.5% — within one 15° step

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Model Generalization — 1D CNN  |  895-D ALL Features  |  24 Angle Classes',
             fontsize=12, fontweight='bold', y=1.01)

# ── Plot A: MAE bar chart ─────────────────────────────────────────────────────
ax = axes[0]

maes        = [same_mae,  diff_mae]
colors      = ['#1565C0', '#E65100']
labels      = ['Same Room\n(held-out test)', 'Different Room\n(unseen)']
inner_labels = [f'Acc: {same_acc}%', f'Est. Acc: {diff_acc_est:.0f}%']

bars = ax.bar(labels, maes, color=colors, width=0.45, alpha=0.88, edgecolor='white')

for bar, mae, inner in zip(bars, maes, inner_labels):
    # MAE value on top
    ax.text(bar.get_x() + bar.get_width() / 2, mae + 0.2,
            f'{mae}°', ha='center', va='bottom', fontsize=14, fontweight='bold')
    # metric inside bar
    ax.text(bar.get_x() + bar.get_width() / 2, mae / 2,
            inner, ha='center', va='center',
            fontsize=10, color='white', fontweight='bold')

ax.set_ylabel('Mean Angular Error (°)', fontsize=11)
ax.set_title('Mean Angular Error by Environment', fontsize=11)
ax.set_ylim(0, 18)
ax.axhline(15, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='15° threshold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# ── Plot C: Scatter true vs predicted ────────────────────────────────────────
ax = axes[1]

# perfect diagonal
ax.plot([0, 345], [0, 345], color='gray', linestyle='--', linewidth=1.2,
        alpha=0.6, label='Perfect prediction', zorder=1)

# scatter points
ax.scatter(true_angles[correct],  pred_angles[correct],
           color='#43A047', s=120, zorder=3, label='Exact (0° error)')
ax.scatter(true_angles[~correct], pred_angles[~correct],
           color='#E53935', s=120, zorder=3, label=f'Off (MAE={diff_mae}°)')

# error lines connecting true to predicted
for t, p in zip(true_angles[~correct], pred_angles[~correct]):
    ax.plot([t, t], [t, p], color='#E53935', linewidth=1.2, alpha=0.6, zorder=2)
    ax.text(t + 4, (t + p) / 2, f'{abs(p-t)}°', fontsize=8,
            color='#E53935', va='center')

ax.set_xlim(-10, 360)
ax.set_ylim(-10, 360)
ax.set_xlabel('True Angle (°)', fontsize=11)
ax.set_ylabel('Predicted Angle (°)', fontsize=11)
ax.set_title('Different Room — True vs Predicted\nPoints on diagonal = perfect', fontsize=11)
ax.set_xticks(range(0, 361, 45))
ax.set_yticks(range(0, 361, 45))
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
ax.set_aspect('equal')
ax.spines[['top', 'right']].set_visible(False)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'generalization_results.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
