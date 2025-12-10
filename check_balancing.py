from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

dataset_name = 'AerialImageDataset'
dataset_type = 'train'

gt_path = f'/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/{dataset_name}/tiles_128/{dataset_type}/gt'
full_maps = sorted(os.path.splitext(f)[0] for f in os.listdir(gt_path) if
                   f.lower().endswith(('.tif', '.tiff', '.png', '.jpg')))

bfraction = []

for image_name in full_maps:
    # Mask loading
    img = Image.open(f'{gt_path}/{image_name}.tif').convert("L")
    arr = np.array(img)

    # If a pixel is greater than 128, then is a building. They are only 0 either 255
    binary = (arr >= 128).astype(np.uint8)

    # Pixel counting
    num_pixels = binary.size
    num_building = binary.sum()
    num_background = num_pixels - num_building

    # Building fraction
    bfraction.append(num_building / num_pixels)

mean_b = np.mean(bfraction)
std_b = np.std(bfraction)

plt.figure(figsize=(10, 6))
plt.hist(bfraction, bins=50, edgecolor="black")
plt.xlabel("Building fraction")
plt.ylabel("Frequency")
plt.title("Building Fraction Distribution")
plt.axvline(mean_b, linestyle="--", linewidth=1.5, label=f"Mean = {mean_b:.3f}")
plt.axvline(mean_b - std_b, linestyle=":", linewidth=1, label=f"Mean - STD = {mean_b - std_b:.3f}")
plt.axvline(mean_b + std_b, linestyle=":", linewidth=1, label=f"Mean + STD = {mean_b + std_b:.3f}")
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig("building_fraction_distribution.png")
