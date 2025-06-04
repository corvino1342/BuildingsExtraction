import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from unet import UNet
from dataset import MyDataset


# Initial setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

num_epochs = 10
batch_size = 4
learning_rate = 1e-4
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

# model initialization, loss and optimizer
model = UNet(n_channels=3, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

writer = SummaryWriter(log_dir='runs')

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    batch_number = 0
    for images, masks in train_loader:
        batch_number += 1

        images = images.to(device)
        masks = masks.to(device).float()  # This must be "float" for the loss function

        outputs = model(images)
        loss = criterion(outputs, masks)

        print(f'Batch Number: {batch_number}/{len(train_loader)}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if epoch % 2 == 0:
            grid = vutils.make_grid(torch.sigmoid(outputs) > 0.5)
            writer.add_image('Predicted Masks', grid, epoch)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), f"runs/unet_epoch_{epoch}.pth")