import matplotlib.pyplot as plt
import pandas as pd
import hashlib
import colorsys

metrics = ['epoch_loss', 'iou', 'precision', 'recall', 'f1']

def model_color_from_name(model_name, saturation=0.65, value=0.85):
    """
    Generate a deterministic RGB color from a model name.
    The model name is mapped to a hue value, ensuring that even
    small changes in the name produce clearly different colors.
    """
    h = hashlib.md5(model_name.encode()).hexdigest()
    # Map hash to hue in [0, 1)
    hue = int(h[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (r, g, b)

def short_model_name(model_name):
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


def Metrics(model_names):
    for i, metric in enumerate(metrics, 1):
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(10, 6))
        for model_name in model_names:
            base_color = model_color_from_name(model_name)
            train_color = base_color
            val_color = (
                min(base_color[0] + 0.15, 1.0),
                min(base_color[1] + 0.15, 1.0),
                min(base_color[2] + 0.15, 1.0),
            )
            df = pd.read_csv(f'/Users/corvino/PycharmProjects/BuildingsExtraction/runs/{model_name}/metrics.csv')
            # Metric Plots
            epochs = df['epoch']

            df[f'train_f1'] = 2 * (df['train_precision'] * df['train_recall']) / (df['train_precision'] + df['train_recall'])
            df[f'val_f1'] = 2 * (df['val_precision'] * df['val_recall']) / (df['val_precision'] + df['val_recall'])

            short_name = short_model_name(model_name)

            # --- Plot training & validation metrics ---
            plt.plot(
                epochs,
                df[f'train_{metric}'],
                color=train_color,
                linewidth=2.5,
                alpha=0.9,
                linestyle='-',
                label=f'Train | {short_name}'
            )
            plt.plot(
                epochs,
                df[f'val_{metric}'],
                color=val_color,
                linewidth=2.5,
                alpha=0.9,
                linestyle='--',
                label=f'Val   | {short_name}'
            )

            # --- Mark and annotate the last values ---
            last_epoch = epochs.iloc[-1]
            last_train = df[f'train_{metric}'].iloc[-1]
            last_val = df[f'val_{metric}'].iloc[-1]

            # Mark the last points
            plt.scatter(last_epoch, last_train, color=train_color, s=80, zorder=5)
            plt.scatter(last_epoch, last_val, color=val_color, s=80, zorder=5)

            # Annotate the last values
            plt.text(
                last_epoch + 1,
                last_train,
                f'{last_train:.2f}',
                color=train_color,
                fontsize=11,
                weight='bold'
            )
            plt.text(
                last_epoch + 1,
                last_val,
                f'{last_val:.2f}',
                color=val_color,
                fontsize=11,
                weight='bold'
            )

            # --- Titles and labels ---
            plt.title(f'Training and Validation {metric}', fontsize=16, weight='bold', pad=15)
            plt.xlabel('Epoch', fontsize=13)
            plt.ylabel(f'{metric}', fontsize=13)

            # --- Legend and style tweaks ---
            plt.legend(fontsize=12, frameon=True, facecolor='white', shadow=True, loc='best')
            plt.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
            plt.tight_layout()

            # --- Optional: Save or show ---
            #plt.savefig(f'predictions/{metric}_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

        plt.tight_layout()
        total_seconds = df['time'].sum()
        hours, remainder = divmod(total_seconds, 3600)
        minutes = remainder // 60
        print(f"Time spent: {int(hours)}:{int(minutes):02d} hours")

        for metric in df.keys():
            print(f'{metric}: {df[metric].iloc[-1]:.3f}')


model_name = ['unet_AID_BCE_lr0p0001_n28000_dim256x256_bs32',
              'unet_AID_WBCE_lr0p0001_n28000_dim256x256_bs32']

Metrics(model_name)
