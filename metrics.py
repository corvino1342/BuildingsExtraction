import matplotlib.pyplot as plt
import pandas as pd

model_architecture = 'unet_2'

df = pd.read_csv(f'runs/train_metrics_{model_architecture}.csv')


# Plot metrics
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