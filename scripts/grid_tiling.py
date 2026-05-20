
import numpy as np
# from shapely.geometry import Polygon
from PIL import Image
import matplotlib.pyplot as plt

import json
from typing import List, Dict, Union
import pyproj
from pyproj import Transformer
import math

# from sentinelhub import SHConfig, BBox, CRS, DataCollection, MimeType, SentinelHubRequest
from typing import Tuple, Optional

import ee


# caltanissetta - 37.536086 14.056348
# uscita tunnel - 37.490418 14.056694

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

def scale_meters_pixels(bbox, image):
    """
    Converts a distance in meters to pixels at a given latitude and longitude.
    This is useful for determining how many pixels correspond to a certain distance on the ground.

    Args:
        bbox: A tuple (min_lat, min_lon, max_lat, max_lon) representing the bounding box.
        image: A PIL Image object representing the image.
    Returns:
        A tuple (meters_per_pixel_x, meters_per_pixel_y) representing the scale in meters per pixel in the x and y directions.
    """
    min_lat = bbox[0]
    max_lat = bbox[2]
    min_lon = bbox[1]
    max_lon = bbox[3]

    image_width, image_height = image.size
    # Calculate the center latitude for scale calculation
    lat = (min_lat + max_lat) / 2

    # Calculate meters per pixel
    meters_per_degree_lat = 111320  # ~111 km per degree at the equator
    meters_per_degree_lon = 111320 * math.cos(math.radians((min_lat + max_lat) / 2))  # Adjust for latitude

    scale_lat = (max_lat - min_lat) * meters_per_degree_lat / image_height
    scale_lon = (max_lon - min_lon) * meters_per_degree_lon / image_width
    return scale_lat, scale_lon


def read_antennas_from_json(json_file_path):
    """
    Reads a JSON file containing antenna characteristics and returns a list of dictionaries,
    each representing an antenna with its properties (e.g., latDD, lonDD).

    Args:
        json_file_path: Path to the JSON file.

    Returns:
        A list of dictionaries, where each dictionary represents an antenna and contains:
        - latDD: Latitude in decimal degrees.
        - lonDD: Longitude in decimal degrees.
        - Any other fields present in the JSON.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    try:
        with open(json_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {json_file_path} does not exist.")
    except json.JSONDecodeError:
        raise json.JSONDecodeError("The JSON file is malformed.", doc=json_file_path, pos=0)
    
    return data

def generate_grid_in_pixels(bbox, screenshot_width, screenshot_height, tile_size_meters, output_file):
    """
    Generates a grid of tiles in pixel coordinates, covering the input bbox.
    Returns a list of dictionaries, where each dictionary contains:
    - center_pixel: (row, col) of the tile center in pixel coordinates.
    - bbox_pixel: [(y1, x1), (y1, x2), (y2, x2), (y2, x1)] of the tile in pixel coordinates.
    - bbox_real: Real-world bounding box [(min_lat, min_lon), ...] of the tile.

    Args:
        bbox: Real-world bounding box of the screenshot: [min_lon, min_lat, max_lon, max_lat]
        screenshot_width: Width of the screenshot in pixels.
        screenshot_height: Height of the screenshot in pixels.
        tile_size_meters: Size of each tile in meters.
        output_file: Optional path to save the results to a JSON file.

    Returns:
        A list of tile dictionaries with pixel and real-world coordinates.
    """
    # Extract real-world bbox coordinates
    min_lat, min_lon = bbox[0], bbox[1]
    max_lat, max_lon = bbox[2], bbox[3]

    # Calculate meters per pixel
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * math.cos(math.radians((min_lat + max_lat) / 2))

    scale_lat = (max_lat - min_lat) * meters_per_degree_lat / screenshot_height
    scale_lon = (max_lon - min_lon) * meters_per_degree_lon / screenshot_width

    # Calculate tile size in pixels
    tile_size_pixels_lat = tile_size_meters / scale_lat
    tile_size_pixels_lon = tile_size_meters / scale_lon

    # Calculate the center of the screenshot in pixels
    center_pixel_col = screenshot_width // 2
    center_pixel_row = screenshot_height // 2

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
            min_lon_tile = min_lon + (x_min_pixel / screenshot_width) * (max_lon - min_lon)
            max_lon_tile = min_lon + (x_max_pixel / screenshot_width) * (max_lon - min_lon)
            min_lat_tile = min_lat + (y_min_pixel / screenshot_height) * (max_lat - min_lat)
            max_lat_tile = min_lat + (y_max_pixel / screenshot_height) * (max_lat - min_lat)

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

def generate_grid_from_antenna(antenna_lon, antenna_lat, tile_size_meters, grid_radius_km, output_file=None):
    """
    Generates a grid of tiles around an antenna location.
    Returns a list of dictionaries, where each dictionary contains:
    - center: (lat, lon) of the tile center.
    - bbox: [(min_lat, min_lon), (min_lat, max_lon), (max_lat, max_lon), (max_lat, min_lon)] of the tile.

    Args:
        antenna_lon: Longitude of the antenna (in degrees).
        antenna_lat: Latitude of the antenna (in degrees).
        tile_size_meters: Side length of each square tile (in meters).
        grid_radius_km: Radius of the grid around the antenna (in kilometers).
        output_file: Optional path to save the results to a JSON file.

    Returns:
        A list of dictionaries, each representing a tile with its center and bounding box.
    """
    # Step 1: Find the UTM zone for the antenna's location
    utm_crs = pyproj.CRS.from_epsg(32600 + int((antenna_lon + 180) // 6) + 1)

    # Step 2: Create transformers for WGS84 <-> UTM
    wgs84 = pyproj.CRS("EPSG:4326")  # WGS84
    transformer_wgs84_to_utm = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    transformer_utm_to_wgs84 = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    # Step 3: Convert antenna coordinates to UTM (meters)
    antenna_x, antenna_y = transformer_wgs84_to_utm.transform(antenna_lon, antenna_lat)

    # Step 4: Calculate the number of tiles in each direction
    n_tiles = int(math.ceil(grid_radius_km * 1000 / tile_size_meters))

    # Step 5: Generate the grid
    tiles = []
    for i in range(-n_tiles, n_tiles + 1):
        for j in range(-n_tiles, n_tiles + 1):
            # Calculate the UTM coordinates of the tile's center and corners
            center_x = antenna_x + i * tile_size_meters + tile_size_meters / 2
            center_y = antenna_y + j * tile_size_meters + tile_size_meters / 2
            x_min = center_x - tile_size_meters / 2
            y_min = center_y - tile_size_meters / 2
            x_max = center_x + tile_size_meters / 2
            y_max = center_y + tile_size_meters / 2

            # Convert center and corners to latitude/longitude
            center_lon, center_lat = transformer_utm_to_wgs84.transform(center_x, center_y)
            min_lon, min_lat = transformer_utm_to_wgs84.transform(x_min, y_min)
            max_lon, _ = transformer_utm_to_wgs84.transform(x_max, y_min)
            _, max_lat = transformer_utm_to_wgs84.transform(x_max, y_max)

            # Create the bounding box (4 corners)
            bbox = [
                (min_lat, min_lon),  # Bottom-left
                (min_lat, max_lon),  # Bottom-right
                (max_lat, max_lon),  # Top-right
                (max_lat, min_lon),  # Top-left
            ]

            # Append the tile data
            tiles.append({
                "center": (center_lat, center_lon),
                "bbox": bbox,
            })

    # Step 6: Save to file (if output_file is provided)
    if output_file:
        import json
        with open(output_file, "w") as f:
            json.dump(tiles, f, indent=4)

    return tiles

def generate_grid_from_bbox(bbox, tile_size_meters, output_file):
    """
    Generates a grid of tiles covering the input bounding box.
    Returns a list of dictionaries, where each dictionary contains:
    - center: (lat, lon) of the tile center.
    - bbox: [(min_lat, min_lon), (min_lat, max_lon), (max_lat, max_lon), (max_lat, min_lon)] of the tile.

    Args:
        bbox: List of 4 coordinates in format: [min_lat, min_lon, max_lat, max_lon]
        tile_size_meters: Side length of each square tile (in meters).
        output_file: Optional path to save the results to a JSON file.

    Returns:
        A list of dictionaries, each representing a tile with its center and bounding box.
    """
    # Extract min/max lat/lon from the input bbox
    min_lat = bbox[0]
    max_lat = bbox[2]
    min_lon = bbox[1]
    max_lon = bbox[3]

    # Calculate the center of the input bbox
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Find the UTM zone for the center point
    utm_crs = pyproj.CRS.from_epsg(32600 + int((center_lon + 180) // 6) + 1)
    wgs84 = pyproj.CRS("EPSG:4326")
    transformer_wgs84_to_utm = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    transformer_utm_to_wgs84 = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    # Convert center to UTM (meters)
    antenna_x, antenna_y = transformer_wgs84_to_utm.transform(center_lon, center_lat)

    # Calculate the number of tiles in each direction
    n_tiles = int(math.ceil(max(max_lat - min_lat, max_lon - min_lon) * 111320 / tile_size_meters))  # ~111320 meters per degree

    print(f"Generating grid with {2*n_tiles+1} x {2*n_tiles+1} = {(2*n_tiles+1)**2} tiles...")

    # Generate the grid
    tiles = []
    for i in range(-n_tiles, n_tiles + 1):
        for j in range(-n_tiles, n_tiles + 1):
            # Calculate the UTM coordinates of the tile's center and corners
            center_x = antenna_x + i * tile_size_meters + tile_size_meters / 2
            center_y = antenna_y + j * tile_size_meters + tile_size_meters / 2
            x_min = center_x - tile_size_meters / 2
            y_min = center_y - tile_size_meters / 2
            x_max = center_x + tile_size_meters / 2
            y_max = center_y + tile_size_meters / 2

            # Convert center and corners to latitude/longitude
            center_lon_tile, center_lat_tile = transformer_utm_to_wgs84.transform(center_x, center_y)
            min_lon_tile, min_lat_tile = transformer_utm_to_wgs84.transform(x_min, y_min)
            max_lon_tile, _ = transformer_utm_to_wgs84.transform(x_max, y_min)
            _, max_lat_tile = transformer_utm_to_wgs84.transform(x_max, y_max)

            # Create the bounding box (4 corners)
            tile_bbox = [
                (min_lat_tile, min_lon_tile),  # Bottom-left
                (min_lat_tile, max_lon_tile),  # Bottom-right
                (max_lat_tile, max_lon_tile),  # Top-right
                (max_lat_tile, min_lon_tile),  # Top-left
            ]

            # Append the tile data
            tiles.append({
                "center": (center_lat_tile, center_lon_tile),
                "bbox": tile_bbox,
            })

    # Save to file (if output_file is provided)
    if output_file:
        with open(output_file, "w") as f:
            json.dump(tiles, f, indent=4)

    return tiles

def center_to_bbox(center, tile_size_meters):
    """
    Converts a tile center (lat, lon) to its bounding box (4 corners).
    Requires the tile size in meters and the center's latitude (for accurate conversion).

    Args:
        center: (lat, lon) of the tile center.
        tile_size_meters: Side length of the tile (in meters).

    Returns:
        Bounding box as a list of 4 corners: [(min_lat, min_lon), (min_lat, max_lon), (max_lat, max_lon), (max_lat, min_lon)].
    """
    center_lat, center_lon = center

    # Convert center to UTM
    utm_crs = pyproj.CRS.from_epsg(32600 + int((center_lon + 180) // 6) + 1)
    wgs84 = pyproj.CRS("EPSG:4326")
    transformer_wgs84_to_utm = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    transformer_utm_to_wgs84 = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    center_x, center_y = transformer_wgs84_to_utm.transform(center_lon, center_lat)

    # Calculate corners in UTM
    half_size = tile_size_meters / 2
    x_min, y_min = center_x - half_size, center_y - half_size
    x_max, y_max = center_x + half_size, center_y + half_size

    # Convert corners to lat/lon
    min_lon, min_lat = transformer_utm_to_wgs84.transform(x_min, y_min)
    max_lon, _ = transformer_utm_to_wgs84.transform(x_max, y_min)
    _, max_lat = transformer_utm_to_wgs84.transform(x_max, y_max)

    bbox = [
        (min_lat, min_lon),  # Bottom-left
        (min_lat, max_lon),  # Bottom-right
        (max_lat, max_lon),  # Top-right
        (max_lat, min_lon),  # Top-left
    ]
    return bbox

def bbox_to_center(bbox):
    """
    Converts a bounding box (4 corners) to its center (lat, lon).

    Args:
        bbox: List of 4 corners: [(min_lat, min_lon), (min_lat, max_lon), (max_lat, max_lon), (max_lat, min_lon)].

    Returns:
        Center of the bounding box as (lat, lon).
    """
    # Extract min/max lat and lon from the bounding box
    lats = [corner[0] for corner in bbox]
    lons = [corner[1] for corner in bbox]
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2
    return (center_lat, center_lon)

def download_sentinel2_rgb(
    bbox: Tuple[float, float, float, float],
    output_file: str = "sentinel2_rgb.png",
    width: int = 1024,
    height: int = 1024,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    maxcc: float = 0.2,
) -> np.ndarray:
    """
    Downloads a true-color RGB satellite image from Sentinel-2 for a given bounding box.
    Uses the SentinelHubRequest API to fetch and process the image.

    Args:
        bbox: Tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84 (EPSG:4326).
        output_file: Path to save the RGB image (default: "sentinel2_rgb.png").
        width: Width of the output image in pixels (default: 1024).
        height: Height of the output image in pixels (default: 1024).
        client_id: Sentinel Hub client ID. If None, reads from ~/.sentinelhub/config.json.
        client_secret: Sentinel Hub client secret. If None, reads from ~/.sentinelhub/config.json.
        maxcc: Maximum cloud coverage (0-1, default: 0.2).

    Returns:
        The RGB image as a NumPy array (shape: height x width x 3).
    """
    # Configure Sentinel Hub (global config)
    config = SHConfig()
    config.sh_client_id = 'sh-f6c245a1-4a4e-4500-b2e6-ea5f054a8fac'       # Your credentials
    config.sh_client_secret = 'YjX7lMUqB28jFzFbdHXIiaQemTmRHeE7'      # Your credentials
    config.save()  # Save to ~/.sentinelhub/config.json



    # Configure Sentinel Hub (use provided credentials or fall back to saved config)
    request_config = SHConfig()
    if client_id and client_secret:
        request_config.sh_client_id = client_id
        request_config.sh_client_secret = client_secret
    elif not (config.sh_client_id and config.sh_client_secret):
        raise ValueError(
            "Sentinel Hub credentials not provided and not found in ~/.sentinelhub/config.json. "
            "Sign up at https://www.sentinelhub.com/ and create a configuration."
        )
    else:
        request_config = config  # Use the global config

    # Create the bounding box object
    bbox_obj = BBox(bbox, crs=CRS.WGS84)

    # Define the evalscript for RGB (B04=Red, B03=Green, B02=Blue)
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"],
                units: "DN"
            }],
            output: {
                bands: 3,
                sampleType: "UINT8"
            }
        };
    }
    function evaluatePixel(sample) {
        return [sample.B04 * 2.5, sample.B03 * 2.5, sample.B02 * 2.5];
    }
    """

    # Create a request (REMOVED the `layer` argument, which is invalid)
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox_obj,
        size=(width, height),
        config=request_config
    )

    # Get the image
    try:
        img = request.get_data()[0]  # Returns a NumPy array (height x width x 4)
        img = img[:, :, :3]  # Remove alpha channel if present
    except Exception as e:
        raise Exception(f"Failed to download image: {e}")

    # Save the image
    Image.fromarray(img).save(output_file)
    print(f"✅ RGB image saved to {output_file}")
    return img


# --- AUTHENTICATE ONCE (outside the function) ---
# Run this ONCE in a Python shell or at the start of your script:
# ee.Authenticate()
def download_google_earth_rgb(bbox, output_file="caltanissetta.png"):

    # Initialize Earth Engine
    ee.Initialize(project='radio-coverage')

    # Define your region of interest
    region = ee.Geometry.Rectangle(bbox)

    # Get a high-resolution image (e.g., WorldView-3)
    image = ee.ImageCollection("COPERNICUS/S2_SR").filterBounds(region).first()
    if not image:
        raise ValueError("No images found for the given bounding box.")

    # Get the RGB visualization URL
    url = image.getThumbURL({
        "bands": ["Red", "Green", "Blue"],
        "region": region,
        "scale": 1,  # Adjust for your needs
    })

    # Download the image
    import requests
    from PIL import Image
    from io import BytesIO
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save(output_file)
    print(f"✅ Image saved to {output_file}")
    return img


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


# [min_lon, min_lat, max_lon, max_lat]
image = {"stazione_caltanissetta": [14.046683, 37.531238, 14.062496, 37.540893],
        "uscita_tunnel": [14.053819, 37.486517, 14.060519, 37.491406]}

name_image = "stazione_caltanissetta"
bbox = image[name_image]

# Define your real-world bbox and dimensions
tile_size_meters = 30

threshold = 100  # Minimum number of building pixels to consider a tile as "building-rich"
image_height, image_width = Image.open(f"/home/unet/Projects/BuildingsExtraction/{name_image}.png").size
building_height = 100  

# Generate the grid in pixel coordinates
tiles = generate_grid_in_pixels(
    bbox=bbox,
    screenshot_width=image_width,
    screenshot_height=image_height,
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


show_grid_overlay( base_img=np.array(Image.open(f"/home/unet/Projects/BuildingsExtraction/{name_image}.png")), tiles=tiles, threshold=threshold, alpha=1)

save_tile_centers_to_txt(tiles=tiles, threshold=threshold, building_height=building_height, output_file=name_image)
