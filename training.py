import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataset import MyDataset
import numpy as np

import time
import csv

# Which model I have to choose?

# UNet1

# Input-> [Down1]->                                                              [Up1]-> Output
#                  [Down2]->                                              [Up2]->
#                           [Down3]->                              [Up3]->
#                                    [Down4]->              [Up4]->
#                                             [Bottleneck]->


# UNet2

# Input-> [Down1]->                                              [Up1]-> Output
#                  [Down2]->                              [Up2]->
#                           [Down3]->              [Up3]->
#                                    [Bottleneck]->

from unet import UNet1

model_loaded = UNet1

def iou_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean()

def precision_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    true_positive = (preds * targets).sum()
    predicted_positive = preds.sum()
    return (true_positive + eps) / (predicted_positive + eps)

def recall_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    true_positive = (preds * targets).sum()
    actual_positive = targets.sum()
    return (true_positive + eps) / (actual_positive + eps)

starting_time = time.time()

model_name = 'unet_AID'

# Initial setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
 # device = 'cpu'
print(device)

dataset_portion = 0.05

num_epochs = 30
batch_size = 128
learning_rate = 1e-4
georef = False

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset_path = '/mnt/nas151/sar/Footprint/datasets/'
dataset_kind = 'AerialImageDataset'
# Dataset and DataLoader
train_dataset_full = MyDataset(image_dir=dataset_path + dataset_kind + '/tiles/train/images',
                          mask_dir=dataset_path + dataset_kind + '/tiles/train/gt',
                          transform=transform)

# In order to twy with less images, this part will mix and select a portion (dataset_portion) of the entire training dataset.
num_samples = len(train_dataset_full)
subset_size = int(dataset_portion * num_samples)

indices = np.random.permutation(num_samples)[:subset_size]

train_dataset = Subset(train_dataset_full, indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


val_dataset = MyDataset(image_dir=dataset_path + dataset_kind + '/tiles/val/images',
                          mask_dir=dataset_path + dataset_kind + '/tiles/val/gt',
                          transform=transform)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

tot_batches = int(len(train_dataset)/batch_size)

print(f'Training dataset dimension: {len(train_dataset)}')
print(f'Validation dataset dimension: {len(val_dataset)}')

print(f"Batch size: {batch_size}")
print(f"Number of batches: {tot_batches}")

# model initialization, loss and optimizer
model = model_loaded(n_channels=3, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Saving metrics
csv_metrics = f"/home/antoniocorvino/Projects/BuildingsExtraction/runs/metrics_{model_name}.csv"

with open(csv_metrics, mode='w', newline='') as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(["epoch",
                         "train_epoch_loss", "train_dice", "train_iou", "train_pixel_acc", "train_precision", "train_recall",
                         "val_epoch_loss", "val_dice", "val_iou", "val_pixel_acc", "val_precision", "val_recall",
                         "time"])


if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    print(f"\nGPU ID: {gpu_id}")
    print(f"GPU Total Memory: {(torch.cuda.get_device_properties(gpu_id).total_memory)/1024**3:.2f} GB")

print("\nTraining is started...\n")

# Training loop
for epoch in range(num_epochs):
    print(f"EPOCH ---- {epoch}/{num_epochs}")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    start_time = time.time()
    model.train()

    epoch_train_loss = 0.0
    batch_number = 0
    iou_total = 0.0
    prec_total = 0.0
    recall_total = 0.0

    for images, masks in train_loader:
        batch_number += 1

        images = images.to(device)
        masks = masks.to(device).float()  # This must be "float" for the loss function

        outputs = model(images)

        loss = criterion(outputs, masks)

        iou_total += iou_score(outputs, masks)
        prec_total += precision_score(outputs, masks)
        recall_total += recall_score(outputs, masks)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed_time = time.time() - start_time

        if batch_number % int(0.1 * tot_batches) == 0:
            print(f"\rProgress: {(100 * batch_number/tot_batches):.0f}% -- time: {int(elapsed_time//60):.02d}:{int(elapsed_time%60):.02d}", end="")

        epoch_train_loss += loss.item()


    train_iou = (iou_total / len(train_loader)).item()
    train_prec = (prec_total / len(train_loader)).item()
    train_recall = (recall_total / len(train_loader)).item()

    print(f"\nTime spent for the epoch: {(elapsed_time/60):.2f} minutes")


    # --- VALIDATION STEP ---
    model.eval()
    epoch_val_loss = 0.0
    val_iou = 0.0
    val_prec = 0.0
    val_recall = 0.0

    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images = val_images.to(device)
            val_masks = val_masks.to(device).float()

            val_outputs = model(val_images)
            loss_val = criterion(val_outputs, val_masks)
            epoch_val_loss += loss_val.item()

            val_iou += iou_score(val_outputs, val_masks)
            val_prec += precision_score(val_outputs, val_masks)
            val_recall += recall_score(val_outputs, val_masks)

    # Average validation metrics
    epoch_val_loss /= len(val_loader)
    val_iou = (val_iou / len(val_loader)).item()
    val_prec = (val_prec / len(val_loader)).item()
    val_recall = (val_recall / len(val_loader)).item()

    print(f'TRAINING - Loss: {epoch_train_loss:.4f}\tIntersection Over Union: {train_iou:.4f}\tPrecision: {train_prec:.4f}\tRecall: {train_recall:.4f}')
    print(f"VALIDATION - Loss: {epoch_val_loss:.4f}\tIntersection Over Union: {val_iou:.4f}\tPrecision: {val_prec:.4f}\tRecall: {val_recall:.4f}\n\n")

    # Save metrics to CSV
    with open(csv_metrics, mode='a', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([epoch + 1,
                             epoch_train_loss, train_iou, train_prec, train_recall,
                             epoch_val_loss, val_iou, val_prec, val_recall,
                             round(elapsed_time, 2)])
# Save the model
torch.save(model.state_dict(), f"/home/antoniocorvino/Projects/BuildingsExtraction/runs/{model_name}.pth")
print(f'Total time: {((time.time() - starting_time)/60):.2f} minutes')
