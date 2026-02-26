import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import csv
import torch


#############################################################################
#############################################################################
#############################################################################


# --------------------------------------------------
# Metrics logger
# --------------------------------------------------
class MetricsLogger:
    def __init__(self, output_dir, filename="metrics.csv"):
        os.makedirs(output_dir, exist_ok=True)
        self.csv_path = os.path.join(output_dir, filename)

        self.header = [
            "epoch",
            "train_loss", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_iou", "val_precision", "val_recall",
            "epoch_time_sec"
        ]

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self.header)

    def log(self, epoch, tr, va, epoch_time):
        row = [
            epoch,
            tr["loss"], tr["iou"], tr["precision"], tr["recall"],
            va["loss"], va["iou"], va["precision"], va["recall"],
            round(epoch_time, 2)
        ]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)


# --------------------------------------------------
# Metrics (correct per-image averaging)
# --------------------------------------------------
def iou_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def precision_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(preds) > threshold).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    pp = preds.sum(dim=(1, 2, 3))
    return ((tp + eps) / (pp + eps)).mean()


def recall_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(preds) > threshold).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    ap = targets.sum(dim=(1, 2, 3))
    return ((tp + eps) / (ap + eps)).mean()

#############################################################################
#############################################################################
#############################################################################


