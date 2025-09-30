import matplotlib.pyplot as plt
import pandas as pd

model_architecture = 'unet_4'

df = pd.read_csv(f'runs/train_metrics_{model_architecture}.csv')


# Metric Plots
metrics = ['loss', 'dice', 'iou', 'pixel_acc', 'precision', 'recall']
epochs = df['epoch']

plt.figure(figsize=(15, 10))

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    plt.plot(epochs, df[metric], label=metric, linewidth=2)
    plt.xlabel('Epoch')
    plt.title(f'{metric} over Epochs')
    plt.legend()

plt.tight_layout()
plt.show()
total_seconds = df['time'].sum()
hours, remainder = divmod(total_seconds, 3600)
minutes = remainder // 60
print(f"Time spent: {int(hours)}:{int(minutes):02d} hours")

for metric in df.keys():
    print(f'{metric}: {df[metric].iloc[-1]:.3f}')
