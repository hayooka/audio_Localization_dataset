import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

REPO = os.path.dirname(os.path.abspath(__file__))
CNN_JSON = os.path.join(REPO, '4_Results', 'plots', 'CNN', 'result_20260501_034954.json')
GRU_JSON = os.path.join(REPO, '4_Results', 'plots', 'GRU', 'result_20260429_081236.json')
OUT_DIR  = os.path.join(REPO, '4_Results', 'plots')

def plot_curves(json_path, model_name, out_path):
    with open(json_path) as f:
        data = json.load(f)

    history   = data['history']
    epochs    = [h['epoch'] for h in history]
    train_acc = [h['train_acc'] * 100 for h in history]
    val_acc   = [h['val_acc'] * 100   for h in history]
    loss      = [h['loss']            for h in history]

    best_val  = data['best_val_acc']
    test_acc  = data['test_acc']
    mae       = data['mae']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{model_name} — 16ms  |  Best Val={best_val:.1f}%  Test={test_acc:.1f}%  MAE={mae:.1f}°',
                 fontsize=12)

    ax1.plot(epochs, train_acc, label='Train', color='steelblue')
    ax1.plot(epochs, val_acc,   label='Validation', color='darkorange')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 105)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, loss, color='coral')
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

plot_curves(CNN_JSON, 'CNN (ALL features)', os.path.join(OUT_DIR, 'training_curve_CNN.png'))
plot_curves(GRU_JSON, 'GRU (ALL features)', os.path.join(OUT_DIR, 'training_curve_GRU.png'))
