# %%
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataset import MyDataset
import numpy as np

import time
import csv
from unet import UNet, UNetL, UNetLL


# %%
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


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
                probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


# %%
starting_time = time.time()

architecture_name = 'unetLL'
model_dataset = 'WHUtiles'
dataset_name = 'WHUBuildingDataset'
tile_dimension = 256
batch_size = 32
weightedBCE = True
diceLoss = True
earlystop = False

# Initial setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# device = 'cpu'
print(f"Device: {device}\n")
memory_fraction = 1.
print(f"Used a fraction of {memory_fraction} GPU's memory")
print(f"Weighted BCE:\t{weightedBCE}")

training_dataset_portion = 1
validation_dataset_portion = 0.5
num_epochs = 30
learning_rate = 0.001

best_val_loss = float("inf")
patience = 5
early_stop_counter = 0

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset_path = '/mnt/nas151/sar/Footprint/datasets/'

WHU_goal = 'tiles'

WHU_train_image_dir = f"{dataset_name}/{WHU_goal}/train/images"
WHU_train_mask_dir = f"{dataset_name}/{WHU_goal}/train/gt"

WHU_val_image_dir = f"{dataset_name}/{WHU_goal}/val/images"
WHU_val_mask_dir = f"{dataset_name}/{WHU_goal}/val/gt"


# Dataset and DataLoader
train_dataset_full = MyDataset(image_dir=WHU_train_image_dir,
                               mask_dir=WHU_train_mask_dir,
                               transform=transform)

# In order to train with fewer images, this part will mix and select a portion (dataset_portion) of the entire training dataset.
num_samples = len(train_dataset_full)
subset_size = int(training_dataset_portion * num_samples)
indices = np.random.permutation(num_samples)[:subset_size]
train_dataset = Subset(train_dataset_full, indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset_full = MyDataset(image_dir=WHU_val_image_dir,
                             mask_dir=WHU_val_mask_dir,
                             transform=transform)

# In order to train with fewer images, this part will mix and select a portion (dataset_portion) of the entire training dataset.
num_samples = len(val_dataset_full)
subset_size = int(validation_dataset_portion * num_samples)
indices = np.random.permutation(num_samples)[:subset_size]
val_dataset = Subset(val_dataset_full, indices)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

tot_batches = int(len(train_dataset) / batch_size)

print(f"Tiles Dimension: {tile_dimension}x{tile_dimension}")

print(f'Training dataset dimension: {len(train_dataset)}')
print(f'Validation dataset dimension: {len(val_dataset)}')

print(f"{tot_batches} batches of {batch_size} images")

if weightedBCE:

    loss_name = 'WBCE'
    # First attempt. I need to change for the frequency of buildings and background pixels
    start = time.time()
    print('Computing the weights...')
    tile_length = len(train_loader.dataset[0][1][0]) * len(train_loader.dataset[0][1][0][0])
    pos_freq = 0
    neg_freq = 0
    for tile in tqdm(train_loader.dataset):
        pos_freq += torch.sum(tile[1][0])

    neg_freq = tile_length * len(train_loader.dataset) - pos_freq
    weight = torch.tensor(neg_freq / pos_freq)
    print(f'Positive weight:\t{weight:.2f}')
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
else:
    loss_name = 'BCE'
    bce_criterion = nn.BCEWithLogitsLoss()

if diceLoss:
    loss_name += 'plusDL'
    dice_criterion = DiceLoss()
else:
    dice_criterion = None

# model initialization, loss and optimizer
if architecture_name == 'unet':
    model = UNet(n_channels=3, n_classes=1).to(device)
elif architecture_name == 'unetL':
    model = UNetL(n_channels=3, n_classes=1).to(device)
elif architecture_name == 'unetLL':
    model = UNetLL(n_channels=3, n_classes=1).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=20)

model_name = f'{architecture_name}_{model_dataset}_{loss_name}_n{len(train_dataset)}_dim{tile_dimension}x{tile_dimension}_bs{batch_size}'

os.makedirs(f'/home/antoniocorvino/Projects/BuildingsExtraction/runs/{model_name}', exist_ok=True)

# Saving metrics
csv_metrics = f"/home/antoniocorvino/Projects/BuildingsExtraction/runs/{model_name}/metrics.csv"

with open(csv_metrics, mode='w', newline='') as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(["epoch",
                         "train_epoch_loss", "train_iou", "train_precision", "train_recall",
                         "val_epoch_loss", "val_iou", "val_precision", "val_recall",
                         "time"])

if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    print(f"\nGPU ID: {gpu_id}")
    print(f"GPU Total Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024 ** 3:.2f} GB")
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=gpu_id)

# %%
best_loss = 1e4

# Training loop
for epoch in range(num_epochs):
    print(f"EPOCH ---- {epoch + 1}/{num_epochs}")
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Current Learning Rate: {current_lr:.6f}')
    print("\nTraining is started...\n")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(gpu_id) / 1024 ** 3:.2f} GB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved(gpu_id) / 1024 ** 3:.2f} GB")
    start_time = time.time()
    model.train()

    epoch_train_loss = 0.0
    batch_number = 0
    iou_total = 0.0
    prec_total = 0.0
    recall_total = 0.0

    for images, masks in train_loader:

        batch_number += 1

        # print(f"\nBatch #{batch_number}")
        images = images.to(device)
        masks = masks.to(device).float()  # This must be "float" for the loss function

        # print(f"Memory allocated by the batch: {torch.cuda.memory_allocated(gpu_id)/1024**2:.2f} MB")
        # print(f"Memory reserved by the batch: {torch.cuda.memory_reserved(gpu_id)/1024**2:.2f} MB")
        # batch_memory = images.element_size() * images.nelement() + masks.element_size() * masks.nelement()
        # print(f"Memory occupied by the batch (images and masks only): {batch_memory / 1024 ** 2:.2f} MB")

        outputs = model(images)

        if dice_criterion:
            loss = bce_criterion(outputs, masks)
        else:
            loss = bce_criterion(outputs, masks)
        with torch.no_grad():
            iou_batch = iou_score(outputs, masks)
            prec_batch = precision_score(outputs, masks)
            recall_batch = recall_score(outputs, masks)

        iou_total += iou_batch
        prec_total += prec_batch
        recall_total += recall_batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed_time = time.time() - start_time

        if batch_number % int(0.1 * tot_batches) == 0:
            print(
                f"\rProgress: {(100 * batch_number / tot_batches):.0f}% -- time: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}",
                end="")

        epoch_train_loss += loss.item()
    torch.cuda.empty_cache()

    epoch_train_loss /= len(train_loader)

    train_iou = (iou_total / len(train_loader)).item()
    train_prec = (prec_total / len(train_loader)).item()
    train_recall = (recall_total / len(train_loader)).item()

    print("\n")
    print("\nValidation is started...")

    # --- VALIDATION STEP ---
    starting_val = time.time()
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
            loss_val = bce_criterion(val_outputs, val_masks)
            epoch_val_loss += loss_val.item()

            val_iou += iou_score(val_outputs, val_masks)
            val_prec += precision_score(val_outputs, val_masks)
            val_recall += recall_score(val_outputs, val_masks)

    # Average validation metrics
    epoch_val_loss /= len(val_loader)
    val_iou = (val_iou / len(val_loader)).item()
    val_prec = (val_prec / len(val_loader)).item()
    val_recall = (val_recall / len(val_loader)).item()

    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        print(best_loss, epoch)
        torch.save(model.state_dict(),
                   f"/home/antoniocorvino/Projects/BuildingsExtraction/runs/{model_name}/best_model.pth")

    print(f"\nTime for Validation: {int(time.time() - starting_val):02d}s\n")

    print(
        f'\nTRAINING\tLoss: {epoch_train_loss:.4f}\tIoU: {train_iou:.4f}\tPrecision: {train_prec:.4f}\tRecall: {train_recall:.4f}')
    print(
        f"VALIDATION\tLoss: {epoch_val_loss:.4f}\tIoU: {val_iou:.4f}\tPrecision: {val_prec:.4f}\tRecall: {val_recall:.4f}\n\n")

    # Save metrics to CSV
    with open(csv_metrics, mode='a', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([epoch + 1,
                             epoch_train_loss, train_iou, train_prec, train_recall,
                             epoch_val_loss, val_iou, val_prec, val_recall,
                             round(elapsed_time, 2)])
    scheduler.step()
    # Checkpoint
    if (epoch + 1) % 10 == 0:
        print(f"Checkpoint {epoch + 1}")
        torch.save(model.state_dict(),
                   f"/home/antoniocorvino/Projects/BuildingsExtraction/runs/{model_name}/checkpoint_{epoch + 1}.pth")

print(f'Total time: {((time.time() - starting_time) / 60):.2f} minutes')
# %%
