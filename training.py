import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from unet import UNet
from dataset import MyDataset

# Initial setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20
batch_size = 2
learning_rate = 1e-4
georef = False

# Optional transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Dataset and DataLoader
if georef:
    train_dataset = MyDataset(image_dir="dataset/georef", mask_dir="dataset/masks", transform=transform)
else:
    train_dataset = MyDataset(image_dir="dataset/images", mask_dir="dataset/masks", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# model initialization, loss and optimizer
model = UNet(n_channels=3, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device).float()  # This must be "float" for the loss function

        outputs = model(images)
        loss = criterion(outputs, masks)

        print("Image batch shape:", images.shape)
        print("Mask batch shape:", masks.shape)
        print("Output shape:", outputs.shape)
        print("Loss:", loss.item())
        break  # Optional: just to test 1 batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "trained_models/unet_1.pth")