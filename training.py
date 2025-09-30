import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MyDataset

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

from unet import UNet2

model_loaded = UNet2


def dice_score(preds, targets, threshold=0.5, eps=1e-6):
    """
    Computes the Dice coefficient.
    preds: predicted logits (before sigmoid)
    targets: ground truth masks
    """
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean()

def iou_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean()

def pixel_accuracy(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    correct = (preds == targets).float().sum()
    total = torch.numel(preds)
    return correct / total

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

model_name = 'unet_5'

# Initial setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
 # device = 'cpu'
print(device)

num_epochs = 50
batch_size = 4
learning_rate = 1e-3
georef = False

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Dataset and DataLoader
if georef:
    train_dataset = MyDataset(image_dir="dataset/training/georef",
                              mask_dir="dataset/training/masks",
                              transform=transform)
else:
    train_dataset = MyDataset(image_dir="dataset/training/images",
                              mask_dir="dataset/training/masks",
                              transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = MyDataset(image_dir="dataset/validation/images",
                        mask_dir="dataset/validation/masks",
                        transform=transform)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f'Training dataset dimension: {len(train_dataset)}')
print(f'Validation dataset dimension: {len(val_dataset)}')

# model initialization, loss and optimizer
model = model_loaded(n_channels=3, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Saving metrics
csv_metrics = f"runs/train_metrics_{model_name}.csv"

with open(csv_metrics, mode='w', newline='') as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(["epoch",
                         "epoch_train_loss", "train_dice", "train_iou", "train_pixel_acc", "train_precision", "train_recall",
                         "epoch_val_loss", "val_dice", "val_iou", "val_pixel_acc", "val_precision", "val_recall",
                         "time"])


# Training loop
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()

    epoch_train_loss = 0.0
    batch_number = 0
    train_dice = 0.0
    iou_total = 0.0
    pixel_acc_total = 0.0
    prec_total = 0.0
    recall_total = 0.0

    for images, masks in train_loader:
        batch_number += 1

        images = images.to(device)
        masks = masks.to(device).float()  # This must be "float" for the loss function

        outputs = model(images)

        # LOSS SCORE
        loss = criterion(outputs, masks)

        # DICE SCORE
        train_dice += dice_score(outputs, masks)
        iou_total += iou_score(outputs, masks)
        pixel_acc_total += pixel_accuracy(outputs, masks)
        prec_total += precision_score(outputs, masks)
        recall_total += recall_score(outputs, masks)

        print(f'Batch Number: {batch_number}/{len(train_loader)}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    elapsed_time = time.time() - start_time

    train_dice = (train_dice / len(train_loader)).item()
    train_iou = (iou_total / len(train_loader)).item()
    train_pixel_acc = (pixel_acc_total / len(train_loader)).item()
    train_prec = (prec_total / len(train_loader)).item()
    train_recall = (recall_total / len(train_loader)).item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_train_loss:.4f}, Time: {elapsed_time:.2f} seconds")

    # --- VALIDATION STEP ---
    model.eval()
    epoch_val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0
    val_pixel_acc = 0.0
    val_prec = 0.0
    val_recall = 0.0

    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images = val_images.to(device)
            val_masks = val_masks.to(device).float()

            val_outputs = model(val_images)
            loss_val = criterion(val_outputs, val_masks)
            epoch_val_loss += loss_val.item()

            val_dice += dice_score(val_outputs, val_masks)
            val_iou += iou_score(val_outputs, val_masks)
            val_pixel_acc += pixel_accuracy(val_outputs, val_masks)
            val_prec += precision_score(val_outputs, val_masks)
            val_recall += recall_score(val_outputs, val_masks)

    # Average validation metrics
    epoch_val_loss /= len(val_loader)
    val_dice = (val_dice / len(val_loader)).item()
    val_iou = (val_iou / len(val_loader)).item()
    val_pixel_acc = (val_pixel_acc / len(val_loader)).item()
    val_prec = (val_prec / len(val_loader)).item()
    val_recall = (val_recall / len(val_loader)).item()

    print(f"Validation - Loss: {epoch_val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

    # Save metrics to CSV
    with open(csv_metrics, mode='a', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([epoch + 1,
                             epoch_train_loss, train_dice, train_iou, train_pixel_acc, train_prec, train_recall,
                             epoch_val_loss, val_dice, val_iou, val_pixel_acc, val_prec, val_recall,
                             round(elapsed_time, 2)])
# Save the model
torch.save(model.state_dict(), f"runs/{model_name}.pth")
print(f'Total time: {((time.time() - starting_time)/60):.2f} minutes')