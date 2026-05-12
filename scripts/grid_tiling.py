
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

def generate_grid(antenna_lon, antenna_lat, tile_size_meters, grid_radius_km, output_file=None):
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
ee.Authenticate()
def download_google_earth_rgb(bbox, output_file="google_earth_rgb.png"):

    # Initialize Earth Engine
    ee.Initialize(project='radiocoverage')

    # Define your region of interest
    region = ee.Geometry.Rectangle(bbox)

    # Get a high-resolution image (e.g., WorldView-3)
    image = ee.ImageCollection("projects/Maxar/WorldView3").filterBounds(region).first()
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

# Example usage:
bbox = [12.49, 41.89, 12.51, 41.90]  # Rome, Italy (as [min_lon, min_lat, max_lon, max_lat])
download_google_earth_rgb(bbox=bbox)
