import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


from unet import UNet2
model_loaded = UNet2

from dataset import MyDataset

import time
import csv

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

model_name = 'unet_2'

# Initial setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

 # device = 'cpu'
print(device)

num_epochs = 100
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

print(f'Dataset dimension: {len(train_dataset)}')

# model initialization, loss and optimizer
model = model_loaded(n_channels=3, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

 # Prepare directory for metrics
csv_path = f"runs/train_metrics_{model_name}.csv"

with open(csv_path, mode='w', newline='') as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(["epoch",
                         "loss",
                         "dice",
                         "iou",
                         "pixel_acc",
                         "precision",
                         "recall",
                         "time"])

loss = []

# Training loop
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()

    epoch_loss = 0.0
    batch_number = 0
    epoch_dice = 0.0
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
        dice = dice_score(outputs, masks).item()
        iou_total += iou_score(outputs, masks).item()
        pixel_acc_total += pixel_accuracy(outputs, masks).item()
        prec_total += precision_score(outputs, masks).item()
        recall_total += recall_score(outputs, masks).item()

        epoch_dice += dice

        print(f'Batch Number: {batch_number}/{len(train_loader)}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()


    elapsed_time = time.time() - start_time

    epoch_dice /= len(train_loader)
    avg_iou = iou_total / len(train_loader)
    avg_pixel_acc = pixel_acc_total / len(train_loader)
    avg_prec = prec_total / len(train_loader)
    avg_recall = recall_total / len(train_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time: {elapsed_time:.2f} seconds")

    # Save metrics to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([epoch + 1,
                             epoch_loss,
                             epoch_dice,
                             avg_iou,
                             avg_pixel_acc,
                             avg_prec,
                             avg_recall,
                             round(elapsed_time, 2)])
# Save the model
torch.save(model.state_dict(), f"runs/{model_name}.pth")
print(f'Total time: {((time.time() - starting_time)/60):.2f} minutes')