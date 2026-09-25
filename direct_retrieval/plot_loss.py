"""
Plots the training/validation loss evolution saved by direct_main_commented.py
(the *_loss_history.npz file next to your checkpoint).

Usage:
    python plot_loss.py /path/to/results/model_loss_history.npz
    python plot_loss.py /path/to/results/model_loss_history.npz --log
    python plot_loss.py /path/to/results/model_loss_history.npz --save loss_plot.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='plot loss evolution')
parser.add_argument('history_path', type=str, help='path to *_loss_history.npz')
parser.add_argument('--log', action='store_true', help='use a log scale on the y-axis')
parser.add_argument('--save', type=str, default=None, help='if given, save the plot to this path instead of showing it')
args = parser.parse_args()

hist = np.load(args.history_path)
train_loss = hist['train_loss']
val_loss = hist['val_loss']
epochs = np.arange(len(train_loss))

print(f'{len(epochs)} epochs of loss history loaded.')
print(f'Final training loss:   {train_loss[-1]:.4f}')
print(f'Final validation loss: {val_loss[-1]:.4f}')
best_epoch = int(np.argmin(val_loss))
print(f'Best validation loss:  {val_loss[best_epoch]:.4f} (epoch {best_epoch})')

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label='training loss')
plt.plot(epochs, val_loss, label='validation loss')
plt.axvline(best_epoch, color='gray', linestyle='--', alpha=0.5,
            label=f'best val epoch ({best_epoch})')
plt.xlabel('epoch')
plt.ylabel('loss')
if args.log:
    plt.yscale('log')
plt.legend()
plt.grid(alpha=0.3)
plt.title('Training / validation loss evolution')
plt.tight_layout()

if args.save:
    plt.savefig(args.save, dpi=150)
    print(f'Plot saved to: {args.save}')
else:
    plt.show()
