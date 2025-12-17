import matplotlib.pyplot as plt
import pandas as pd

metrics = ['epoch_loss', 'iou', 'precision', 'recall', 'f1']

def ShortModelName(model_name):
    """
    Create a compact, human-readable model identifier for plots.
    Example:
    unet_AID_WBCE_lr0p0001_n28000_dim256x256_bs32
    -> UNet | AID | WBCE | 256 | bs32
    """
    parts = model_name.split('_')

    arch = parts[0].upper() if parts else "MODEL"

    dt = "AID" if "AID" in parts else "MBD" if "MBD" in parts else "DATASET"

    loss = "WBCE" if "WBCE" in parts else "BCE" if "BCE" in parts else "LOSS"

    dim = next((p.replace("dim", "") for p in parts if p.startswith("dim")), "dim?")
    dim = dim.split('x')[0]  # keep only one spatial dimension

    bs = next((p for p in parts if p.startswith("bs")), "bs?")

    return f"{arch} | {dt} | {loss} | {dim} | {bs}"

def F1Score(model_names):
    for model_name in model_names:
        df = pd.read_csv(f'/Users/corvino/PycharmProjects/BuildingsExtraction/runs/{model_name}/metrics.csv')

        df[f'train_f1'] = 2 * (df['train_precision'] * df['train_recall']) / (df['train_precision'] + df['train_recall'])
        df[f'val_f1'] = 2 * (df['val_precision'] * df['val_recall']) / (df['val_precision'] + df['val_recall'])

        df.to_csv(f'/Users/corvino/PycharmProjects/BuildingsExtraction/runs/{model_name}/metrics.csv')

def Plots(model_names):

    colors = [
        (0.12, 0.47, 0.71),  # blue
        (1.00, 0.50, 0.05),  # orange
        (0.17, 0.63, 0.17),  # green
        (0.84, 0.15, 0.16),  # red
        (0.58, 0.40, 0.74),  # purple
        (0.55, 0.34, 0.29),  # brown
        (0.00, 0.62, 0.60),  # cyan
    ]

    for i, metric in enumerate(metrics, 1):
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(14, 7))
        for model_name in model_names:

            df = pd.read_csv(f'/Users/corvino/PycharmProjects/BuildingsExtraction/runs/{model_name}/metrics.csv')
            # Metric Plots
            epochs = df['epoch']

            short_name = ShortModelName(model_name)

            idx = model_names.index(model_name)
            # --- Plot training & validation metrics ---
            plt.plot(
                epochs,
                df[f'train_{metric}'],
                color=colors[idx],
                linewidth=1.5,
                alpha=0.9,
                linestyle='-',
                label=f'Train | {short_name}'
            )
            plt.plot(
                epochs,
                df[f'val_{metric}'],
                color=colors[idx],
                linewidth=2,
                alpha=0.9,
                linestyle=':',
                label=f'Val   | {short_name}'
            )

            # --- Mark and annotate the last values ---
            last_epoch = epochs.iloc[-1]
            last_train = df[f'train_{metric}'].iloc[-1]
            last_val = df[f'val_{metric}'].iloc[-1]

            # Mark the last points
            plt.scatter(last_epoch, last_train, color=colors[idx], s=80, zorder=5)
            plt.scatter(last_epoch, last_val, color=colors[idx], s=80, zorder=5)

            # Annotate the last values
            plt.text(
                last_epoch + 1,
                last_train,
                f'{last_train:.2f}',
                color=colors[idx],
                fontsize=11,
                weight='bold'
            )
            plt.text(
                last_epoch + 1,
                last_val,
                f'{last_val:.2f}',
                color=colors[idx],
                fontsize=11,
                weight='bold'
            )

            # --- Titles and labels ---
            plt.title(f'Training and Validation {metric}', fontsize=16, weight='bold', pad=15)
            plt.xlabel('Epoch', fontsize=13)
            plt.ylabel(f'{metric}', fontsize=13)

            # --- Legend and style tweaks ---
            plt.legend(fontsize=9, frameon=True, facecolor='white', shadow=True, bbox_to_anchor=(1, 1))
            plt.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
            plt.tight_layout()

        # --- Optional: Save or show ---
        plt.savefig(f'/Users/corvino/PycharmProjects/BuildingsExtraction/predictions/{metric}.tif', dpi=300, bbox_inches='tight')
        #plt.show()

def ValuesReached(model_names):
    for model_name in model_names:
        print(f'\n--------- {ShortModelName(model_name)} ---------\n')
        df = pd.read_csv(f'/Users/corvino/PycharmProjects/BuildingsExtraction/runs/{model_name}/metrics.csv')

        total_seconds = df['time'].sum()
        hours, remainder = divmod(total_seconds, 3600)
        minutes = remainder // 60
        print(f"Time spent: {int(hours)}:{int(minutes):02d} hours for {df['epoch'].iloc[-1]} epochs\n")

        print('VALUES REACHED')
        print(f"\t --- TRAIN --- \t --- VALID ---")
        print(f"Loss --- {df['train_epoch_loss'].iloc[-1]:.3f} --- \t --- {df['val_epoch_loss'].iloc[-1]:.3f} ---")
        print(f"IoU  --- {df['train_iou'].iloc[-1]:.3f} --- \t --- {df['val_iou'].iloc[-1]:.3f} ---")
        print(f"Prec --- {df['train_precision'].iloc[-1]:.3f} --- \t --- {df['val_precision'].iloc[-1]:.3f} ---")
        print(f"Rec  --- {df['train_recall'].iloc[-1]:.3f} --- \t --- {df['val_recall'].iloc[-1]:.3f} ---")
        print(f"F1   --- {df['train_f1'].iloc[-1]:.3f} --- \t --- {df['val_f1'].iloc[-1]:.3f} ---\n")


model_names = ['unet_IAD_BCE_lr0p0001_n28000_dim256x256_bs32',
               'unet_IAD_WBCE_lr0p0001_n44800_dim256x256_bs32',
               'unet_MBD_BCE_lr0p0001_n3945_dim256x256_bs32',
               'unet_MBD_WBCE_lr0p0001_n3945_dim256x256_bs32']

F1Score(model_names)
Plots(model_names)
ValuesReached(model_names)