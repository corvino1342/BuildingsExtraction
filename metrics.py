import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

model_architecture = 'unet_Massachusetts'

df = pd.read_csv(f'runs/metrics_{model_architecture}.csv')

# Metric Plots
metrics = ['epoch_loss', 'dice', 'iou', 'pixel_acc', 'precision', 'recall']
epochs = df['epoch']


for i, metric in enumerate(metrics, 1):
    # --- Fancy style setup ---
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, 6))

    # --- Plot training & validation metrics ---
    plt.plot(epochs, df[f'train_{metric}'], color='#4c9aff', linewidth=2.5, alpha=0.9, label='Training Loss')
    plt.plot(epochs, df[f'val_{metric}'], color='#ffa726', linewidth=2.5, alpha=0.9, label='Validation Loss')

    # --- Mark and annotate the last values ---
    last_epoch = epochs.iloc[-1]
    last_train = df[f'train_{metric}'].iloc[-1]
    last_val = df[f'val_{metric}'].iloc[-1]

    # Mark the last points
    plt.scatter(last_epoch, last_train, color='#4c9aff', s=80, zorder=5)
    plt.scatter(last_epoch, last_val, color='#ffa726', s=80, zorder=5)

    # Annotate the last values
    plt.text(last_epoch + 1, last_train,
             f'{last_train:.2f}', color='#4c9aff', fontsize=11, weight='bold')
    plt.text(last_epoch + 1, last_val,
             f'{last_val:.2f}', color='#ffa726', fontsize=11, weight='bold')

    # --- Titles and labels ---
    plt.title(f'Training and Validation {metric}', fontsize=16, weight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel(f'{metric}', fontsize=13)

    # --- Legend and style tweaks ---
    plt.legend(fontsize=12, frameon=True, facecolor='white', shadow=True, loc='best')
    plt.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.tight_layout()

    # --- Optional: Save or show ---
    #plt.savefig('fancy_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

plt.tight_layout()
total_seconds = df['time'].sum()
hours, remainder = divmod(total_seconds, 3600)
minutes = remainder // 60
print(f"Time spent: {int(hours)}:{int(minutes):02d} hours")

for metric in df.keys():
    print(f'{metric}: {df[metric].iloc[-1]:.3f}')
