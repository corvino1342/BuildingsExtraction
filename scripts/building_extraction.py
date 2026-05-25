import math
import os
from torchvision import transforms
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import torch
from src.models.factory import build_model

def infer_model(runs_path, model_path, arch, img_tensor, infer, map_name, device="cuda"):
    """Run inference on the input image and save probability maps."""
    model = build_model(arch).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)

    os.makedirs(f"{runs_path}/experiments/{infer}", exist_ok=True)
    probs.path = f"{runs_path}/experiments/{infer}/{map_name}.npy"
    np.save(probs.path, probs.squeeze().cpu().numpy())

    print("Saved probability maps.")

    return probs

def generate_grid_in_pixels(bbox, image_width, image_height, tile_size_meters, output_file):
    """
    Generates a grid of tiles in pixel coordinates, covering the input bbox.
    Returns a list of dictionaries, where each dictionary contains:
    - center_pixel: (row, col) of the tile center in pixel coordinates.
    - bbox_pixel: [(y1, x1), (y1, x2), (y2, x2), (y2, x1)] of the tile in pixel coordinates.
    - bbox_real: Real-world bounding box [(min_lat, min_lon), ...] of the tile.

    Args:
        bbox: Real-world bounding box of the screenshot: [min_lon, min_lat, max_lon, max_lat]
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.
        screenshot_height: Height of the screenshot in pixels.
        tile_size_meters: Size of each tile in meters.
        output_file: Optional path to save the results to a JSON file.

    Returns:
        A list of tile dictionaries with pixel and real-world coordinates.
    """
    # Extract real-world bbox coordinates
    min_lat, min_lon, max_lat, max_lon = bbox

    # Calculate meters per pixel
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * math.cos(math.radians((min_lat + max_lat) / 2))

    scale_lat = (max_lat - min_lat) * meters_per_degree_lat / image_height
    scale_lon = (max_lon - min_lon) * meters_per_degree_lon / image_width

    # Calculate tile size in pixels
    tile_size_pixels_lat = tile_size_meters / scale_lat
    tile_size_pixels_lon = tile_size_meters / scale_lon

    # Calculate the center of the screenshot in pixels
    center_pixel_col = image_width // 2
    center_pixel_row = image_height // 2

    # Calculate the number of tiles needed to cover the bbox
    n_tiles = int(math.ceil(max(max_lat - min_lat, max_lon - min_lon) * 111320 / tile_size_meters))

    # Generate the grid in pixel coordinates
    tiles = []
    for i in range(-n_tiles, n_tiles + 1):
        for j in range(-n_tiles, n_tiles + 1):
            # Calculate the pixel coordinates of the tile's center
            center_pixel_col_tile = center_pixel_col + i * tile_size_pixels_lon
            center_pixel_row_tile = center_pixel_row + j * tile_size_pixels_lat

            # Calculate the pixel coordinates of the tile's corners
            x_min_pixel = center_pixel_col_tile - tile_size_pixels_lon / 2
            y_min_pixel = center_pixel_row_tile - tile_size_pixels_lat / 2
            x_max_pixel = center_pixel_col_tile + tile_size_pixels_lon / 2
            y_max_pixel = center_pixel_row_tile + tile_size_pixels_lat / 2

            # Convert pixel coordinates to real-world lat/lon
            min_lon_tile = min_lon + (x_min_pixel / image_width) * (max_lon - min_lon)
            max_lon_tile = min_lon + (x_max_pixel / image_width) * (max_lon - min_lon)
            min_lat_tile = min_lat + (y_min_pixel / image_height) * (max_lat - min_lat)
            max_lat_tile = min_lat + (y_max_pixel / image_height) * (max_lat - min_lat)

            # Create the real-world bounding box for the tile
            tile_bbox_real = [
                (min_lat_tile, min_lon_tile),  # Bottom-left
                (min_lat_tile, max_lon_tile),  # Bottom-right
                (max_lat_tile, max_lon_tile),  # Top-right
                (max_lat_tile, min_lon_tile),  # Top-left
            ]

            # Append the tile data
            tiles.append({
                "center_pixel": (int(center_pixel_row_tile), int(center_pixel_col_tile)),
                "bbox_pixel": [
                    (int(y_min_pixel), int(x_min_pixel)),  # Bottom-left
                    (int(y_min_pixel), int(x_max_pixel)),  # Bottom-right
                    (int(y_max_pixel), int(x_max_pixel)),  # Top-right
                    (int(y_max_pixel), int(x_min_pixel)),  # Top-left
                ],
                "center_real": ((min_lat_tile + max_lat_tile) / 2, (min_lon_tile + max_lon_tile) / 2),
                "bbox_real": tile_bbox_real,
            })

    # Save to file (if output_file is provided)
    if output_file:
        import json
        with open(output_file, "w") as f:
            json.dump(tiles, f, indent=4)

    return tiles

def compute_tile_stats(mask, tiles):
    """
    Computes the sum of building pixels for each tile in the mask.

    Args:
        mask: Binary mask (same size as screenshot).
        tiles: List of tiles from `generate_grid_in_pixels`.

    Returns:
        List of tiles with added `building_sum` field.
    """
    for tile in tiles:
        y_min, x_min = tile["bbox_pixel"][0]
        y_max, x_max = tile["bbox_pixel"][2]

        # Extract the tile from the mask
        tile_mask = mask[y_min:y_max, x_min:x_max]

        # Sum the building pixels (1s)
        tile["building_sum"] = int(np.sum(tile_mask))

    return tiles

def show_grid_overlay(base_img, tiles, threshold, alpha=1):
    """
    Overlays a grid on a base image, highlighting tiles with building sums > threshold.

    Args:
        base_img: Base image (NumPy array, RGB format).
        tiles: List of tiles from `generate_grid_in_pixels`.
        threshold: Building sum threshold to highlight tiles.
        alpha: Transparency of the overlay (0.0 to 1.0).
    """


    print(base_img.shape, len(tiles), "tiles to overlay.")

    # Create a blank mask (transparent background)
    mask = np.zeros_like(base_img, dtype=np.uint8)
    mask[:] = [0, 0, 0, 0]  # transparent background

    # Highlight tiles above threshold
    for tile in tiles:
        if tile["building_sum"] > threshold:
            # Draw the tile in the mask
            y_min, x_min = tile["bbox_pixel"][0]
            y_max, x_max = tile["bbox_pixel"][2]

            # Set the tile region to black with some transparency
            mask[y_min:y_max, x_min:x_max] = [0, 0, 0, 200]  # black with some transparency

    # Overlay the mask on the base image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(base_img, alpha=alpha)
    ax.imshow(mask, alpha=alpha)

    # Draw grid lines for clarity
    # for tile in tiles:
    #     y_min, x_min = tile["bbox_pixel"][0]
    #     y_max, x_max = tile["bbox_pixel"][2]
    #     ax.plot([x_min, x_max], [y_min, y_min], color="red", linewidth=0.1)  # Bottom
    #     ax.plot([x_min, x_max], [y_max, y_max], color="red", linewidth=0.1)  # Top
    #     ax.plot([x_min, x_min], [y_min, y_max], color="red", linewidth=0.1)  # Left
    #     ax.plot([x_max, x_max], [y_min, y_max], color="red", linewidth=0.1)  # Right

    # ax.set_title(f"Grid Overlay (Threshold={threshold})")
    # ax.axis("off")
    plt.savefig(f"grid_overlay_threshold_{threshold}.png", bbox_inches='tight')
    plt.show()

def save_tile_centers_to_txt(tiles, threshold, building_height, output_file="tile_centers"):
    """
    Saves the centers of tiles with building_sum > threshold to a text file.
    Format: lat, lon, {building_height} meters

    Args:
        tiles: List of tiles with 'center' and 'building_sum'.
        threshold: Building sum threshold to filter tiles.
        building_height: Height of buildings in the mask.
        output_file: Path to save the text file.
    """

    with open(f"{output_file}.txt", "w") as f:
        for tile in tiles:
            if tile["building_sum"] > threshold:
                lat, lon = tile["center_real"]
                f.write(f"{lon}, -{lat}, {building_height} meters\n")

    print(f"✅ Saved centers of {sum(1 for t in tiles if t['building_sum'] > threshold)} tiles to {output_file}.txt")

############################################################################
# MAIN EXECUTION
############################################################################

runs_path = "/home/unet/Projects/BuildingsExtraction/"

map_name = "stazione_caltanissetta"

model = "unetLL_bce_dim256_n3425_bs16"

image_path = f"/home/unet/Projects/BuildingsExtraction/{map_name}.png"

model_path = f"{runs_path}/runs/MassachusettsBuildingDataset/{model}/best_model.pth"

arch = "unetLL"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img = Image.open(image_path).convert("RGB")
img_np = np.array(img)

transform = transforms.ToTensor()
img_tensor = transform(img).unsqueeze(0).to(device)

print(f"Running {arch}...")
p = infer_model(runs_path, model_path, arch, img_tensor, infer="example", map_name=map_name)



# [min_lon, min_lat, max_lon, max_lat]
image = {"stazione_caltanissetta": [14.046683, 37.531238, 14.062496, 37.540893],
        "uscita_tunnel": [14.053819, 37.486517, 14.060519, 37.491406]}

name_image = "uscita_tunnel"
bbox = image[name_image]

# Define your real-world bbox and dimensions
tile_size_meters = 30

threshold = 100  # Minimum number of building pixels to consider a tile as "building-rich"
image_height, image_width = Image.open(f"/home/unet/Projects/BuildingsExtraction/{name_image}.png").size
building_height = 100  

# Generate the grid in pixel coordinates
tiles = generate_grid_in_pixels(
    bbox=bbox,
    image_width=image_width,
    image_height=image_height,
    tile_size_meters=tile_size_meters,
    output_file="tiles.json"
)

# Load your building mask
mask = np.load(f"/home/unet/Projects/BuildingsExtraction/experiments/example/{name_image}.npy")

# Compute building sums for each tile
tiles = compute_tile_stats(mask, tiles)

print(len(tiles), "tiles generated and stats computed.")

# Print stats for the first 3 tiles
for i, tile in enumerate(tiles[:3]):
    print(f"\nTile {i}:")
    print(f"  Center Pixel: {tile['center_pixel']}")
    print(f"  Real-world Bbox: {tile['bbox_real']}")
    print(f"  Building Pixel Sum: {tile['building_sum']}")

show_grid_overlay(base_img=np.array(Image.open(f"/home/unet/Projects/BuildingsExtraction/{name_image}.png")), tiles=tiles, threshold=threshold, alpha=1)

save_tile_centers_to_txt(tiles=tiles, threshold=threshold, building_height=building_height, output_file=name_image)

