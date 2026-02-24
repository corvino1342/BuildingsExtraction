import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

#python metrics.py --runs_path /home/antoniocorvino/Projects/BuildingsExtraction/runs --models unetLL_IAD_BCEplusDL_n56000_dim256x256_bs32 unetLL_IAD_WBCEplusDL_n56000_dim256x256_bs32  unetLL_WHUtiles_WBCEplusDL_n24000_dim256x256_bs32  --compute_f1 --plots  --summary


# --------------------------------------------------
# Configuration
# --------------------------------------------------
METRICS = ['epoch_loss', 'iou', 'precision', 'recall', 'f1']

COLORS = [
    (0.12, 0.47, 0.71),  # blue
    (1.00, 0.50, 0.05),  # orange
    (0.17, 0.63, 0.17),  # green
    (0.84, 0.15, 0.16),  # red
    (0.58, 0.40, 0.74),  # purple
    (0.55, 0.34, 0.29),  # brown
    (0.00, 0.62, 0.60),  # cyan
]


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def ShortModelName(model_name):
    parts = model_name.split('_')

    arch = parts[0].upper() if parts else "MODEL"
    dataset = parts[1] if len(parts) > 1 else "DATA"
    loss = parts[2] if len(parts) > 2 else "LOSS"

    lr_part = next((p for p in parts if p.startswith("lr")), None)
    if lr_part:
        lr_raw = lr_part[2:]
        lr = f"{float(lr_raw.replace('p', '.')):.0e}" if 'p' in lr_raw else lr_raw
    else:
        lr = "lr?"

    dim = next((p.replace("dim", "") for p in parts if p.startswith("dim")), "dim?")
    dim = dim.split('x')[0]

    bs = next((p for p in parts if p.startswith("bs")), "bs?")

    return f"{arch} | {dataset} | {loss} | {lr} | {dim} | {bs}"


# --------------------------------------------------
# Core logic
# --------------------------------------------------
def compute_f1(runs_path, model_names):
    for model in model_names:
        csv_path = f"{runs_path}/{model}/metrics.csv"
        df = pd.read_csv(csv_path)

        df['train_f1'] = (
            2 * df['train_precision'] * df['train_recall']
            / (df['train_precision'] + df['train_recall'])
        )
        df['val_f1'] = (
            2 * df['val_precision'] * df['val_recall']
            / (df['val_precision'] + df['val_recall'])
        )

        df.to_csv(csv_path, index=False)


def plot_metrics(runs_path, model_names):
    out_dir = f"{runs_path}/comparison"
    os.makedirs(out_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-darkgrid')

    for metric in METRICS:
        plt.figure(figsize=(14, 7))

        for idx, model in enumerate(model_names):
            df = pd.read_csv(f"{runs_path}/{model}/metrics.csv")
            epochs = df['epoch']
            short_name = ShortModelName(model)

            plt.plot(
                epochs,
                df[f"train_{metric}"],
                color=COLORS[idx % len(COLORS)],
                linewidth=3,
                linestyle='-',
                label=f"Train | {short_name}"
            )

            plt.plot(
                epochs,
                df[f"val_{metric}"],
                color=COLORS[idx % len(COLORS)],
                linewidth=3,
                linestyle=':',
                label=f"Val   | {short_name}"
            )

        plt.title(f"Training and Validation {metric}", fontsize=16, weight='bold')
        plt.xlabel("Epoch", fontsize=13)
        plt.ylabel(metric, fontsize=13)
        plt.legend(fontsize=9, frameon=True, bbox_to_anchor=(1, 1))
        plt.tight_layout()

        plt.savefig(f"{out_dir}/{metric}.png", dpi=300, bbox_inches='tight')
        plt.close()


def print_summary(runs_path, model_names):
    for model in model_names:
        df = pd.read_csv(f"{runs_path}/{model}/metrics.csv")

        total_seconds = df['time'].sum()
        hours, rem = divmod(total_seconds, 3600)
        minutes = rem // 60

        print(f"\n--------- {ShortModelName(model)} ---------")
        print(f"Time spent: {int(hours)}:{int(minutes):02d} hours ({df['epoch'].iloc[-1]} epochs)\n")

        print("FINAL VALUES")
        print(f"{'':8s} TRAIN      VALID")
        print(f"Loss    {df['train_epoch_loss'].iloc[-1]:.3f}     {df['val_epoch_loss'].iloc[-1]:.3f}")
        print(f"IoU     {df['train_iou'].iloc[-1]:.3f}     {df['val_iou'].iloc[-1]:.3f}")
        print(f"Prec    {df['train_precision'].iloc[-1]:.3f}     {df['val_precision'].iloc[-1]:.3f}")
        print(f"Recall  {df['train_recall'].iloc[-1]:.3f}     {df['val_recall'].iloc[-1]:.3f}")
        print(f"F1      {df['train_f1'].iloc[-1]:.3f}     {df['val_f1'].iloc[-1]:.3f}")


# --------------------------------------------------
# CLI
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Training metrics analysis")

    parser.add_argument("--runs_path", type=str, required=True)
    parser.add_argument("--models", nargs="+", required=True)

    parser.add_argument("--compute_f1", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--summary", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.compute_f1:
        compute_f1(args.runs_path, args.models)

    if args.plots:
        plot_metrics(args.runs_path, args.models)

    if args.summary:
        print_summary(args.runs_path, args.models)


if __name__ == "__main__":
    main()