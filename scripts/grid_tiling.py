
import numpy as np
from shapely.geometry import Polygon
from PIL import Image
import matplotlib.pyplot as plt

import json
from typing import List, Dict, Union
import pyproj
from pyproj import Transformer
import math

def read_antennas_from_json(
    json_file_path: str,
    lat_key: str = "latDD",
    lon_key: str = "lonDD",
) -> List[Dict[str, Union[float, str, int, bool]]]:
    """
    Reads a JSON file containing antenna characteristics and returns a list of dictionaries,
    each representing an antenna with its properties (e.g., latDD, lonDD).

    Args:
        json_file_path: Path to the JSON file.
        lat_key: Key for latitude in the JSON (default: "latDD").
        lon_key: Key for longitude in the JSON (default: "lonDD").

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




def download_aerial_image(polygon_vertices, output_file="aerial_image.png", resolution=10):
    """
    Downloads an aerial image for a polygon defined by its vertices (lat/lon).

    Args:
        polygon_vertices: List of (lon, lat) tuples defining the polygon vertices.
        output_file: Path to save the output image (default: "aerial_image.png").
        resolution: Desired resolution in meters (default: 10).
    """
    # Configure Sentinel Hub API
    config = SHConfig()
    # Replace with your credentials (get them from https://apps.sentinel-hub.com/dashboard/)
    config.sh_client_id = 'YOUR_CLIENT_ID'
    config.sh_client_secret = 'YOUR_CLIENT_SECRET'
    config.save()

    # Create a Shapely polygon from the vertices
    polygon = Polygon(polygon_vertices)

    # Get the bounding box of the polygon
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    bbox = BBox([min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)

    # Calculate the size of the output image in pixels
    # Approximate meters per degree at the polygon's latitude
    lat = (min_lat + max_lat) / 2
    meters_per_degree = 111320  # Approximate meters per degree of latitude
    meters_per_degree_lon = meters_per_degree * np.cos(np.radians(lat))
    width_px = int((max_lon - min_lon) * meters_per_degree_lon / resolution)
    height_px = int((max_lat - min_lat) * meters_per_degree / resolution)

    # Request the image from Sentinel-2 (true color: B04=Red, B03=Green, B02=Blue)
    try:
        img = bbox_to_img(
            bbox=bbox,
            width=width_px,
            height=height_px,
            config=config,
            data_collection=DataCollection.SENTINEL2_L2A,
            bands=["B04", "B03", "B02"],  # RGB
            maxcc=0.2,  # Maximum cloud coverage (0-1)
        )

        # Normalize and save the image
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        img_pil.save(output_file)
        print(f"Aerial image saved to {output_file}")

        # Optional: Display the image
        plt.imshow(img_pil)
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error downloading image: {e}")

# Example usage:

json_path = "/home/antoniocorvino/Projects/BuildingsExtraction/session_data.json"

antenna = read_antennas_from_json(json_path)

antenna_lon = antenna['networks']['aaaa']['antennas'][0]['lonDD']
antenna_lat = antenna['networks']['aaaa']['antennas'][0]['latDD']

print(generate_grid(antenna_lon, antenna_lat, tile_size_meters=90, grid_radius_km=0.1))

# Download the image
# download_aerial_image(polygon_vertices, output_file="rome_aerial.png")