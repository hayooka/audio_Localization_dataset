import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

categories = ["Speech", "Alarms", "Tools", "Footsteps", "Animals", "Overall"]
accuracy   = [100, 95, 90, 88, 92, 92]

fig, ax = plt.subplots(figsize=(10, 4.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

bars = ax.bar(categories, accuracy, color='#1565C0', width=0.5, zorder=3)

for bar, value in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8,
            f"{value}%", ha='center', va='bottom', fontsize=10, color='#1a1a1a')

ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_xlabel("Sound Category", fontsize=11)
ax.set_title("Sound Recognition — Real-Time Accuracy per Category", fontsize=12)
ax.set_ylim(0, 110)
ax.tick_params(axis='both', labelsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sound_recognition_accuracy.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor='white')
print(f"Saved: {out}")
plt.show()