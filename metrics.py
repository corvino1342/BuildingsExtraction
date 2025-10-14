import matplotlib.pyplot as plt
import pandas as pd

model_architecture = 'unet_Massachusetts'

df = pd.read_csv(f'runs/metrics_{model_architecture}.csv')

# Metric Plots
metrics = ['epoch_loss', 'dice', 'iou', 'pixel_acc', 'precision', 'recall']
epochs = df['epoch']


for i, metric in enumerate(metrics, 1):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, df[f'train_{metric}'], label=f'train_{metric}', linewidth=2)
    plt.plot(epochs, df[f'val_{metric}'], label=f'val_{metric}', linewidth=2)
    plt.xlabel('Epoch')
    plt.title(f'{metric} over epochs')
    plt.legend()
    plt.show()

plt.tight_layout()
total_seconds = df['time'].sum()
hours, remainder = divmod(total_seconds, 3600)
minutes = remainder // 60
print(f"Time spent: {int(hours)}:{int(minutes):02d} hours")

for metric in df.keys():
    print(f'{metric}: {df[metric].iloc[-1]:.3f}')
