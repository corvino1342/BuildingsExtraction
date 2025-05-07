import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from your_dataset import YourDataset  # you’ll need a custom dataset class
from unet import UNet

# Define hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=1).to(device)  # adjust channels and classes
criterion = nn.CrossEntropyLoss()  # or CrossEntropyLoss if more than 2 classes
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_dataset = YourDataset(transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "unet_model.pth")