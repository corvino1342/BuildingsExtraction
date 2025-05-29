import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from unet import UNet

# === Initial Configuration ===

tile_number = 20

model_path = "trained_models/unet_1.pth"

image_path = f"dataset/test/images/tile{tile_number}.png"  # Sostituisci con il tuo path
mask_path = f"dataset/test/masks/tile{tile_number}.png"


# === Load the model ===
model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# === Prepare the image ===
transform = transforms.Compose([
    transforms.ToTensor(),
])

img = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")
input_tensor = transform(img).unsqueeze(0)  # Aggiunge dimensione batch → [1, 3, H, W]

# === Inference ===
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.sigmoid(output)
    binary_mask = (prediction > 0.5).float()

# === Visualization ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Real Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(binary_mask.squeeze().numpy(), cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()