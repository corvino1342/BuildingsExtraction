import os
import osmnx as ox
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer


# --- CONFIGURATION ---
output_dir = "dataset"
image_size = (1024, 1024)  # in pixels


# Create folders
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/masks", exist_ok=True)


def latlon_to_mercator_bbox(south, north, west, east):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    xmin, ymin = transformer.transform(west, south)
    xmax, ymax = transformer.transform(east, north)

    return xmin, ymin, xmax, ymax


# --- 1. DOWNLOAD MAP IMAGE (OPTIONAL) ---
def download_map_tile(bbox, image_size, tile_path):

    gdf = ox.features_from_bbox(bbox=(bbox[3], bbox[1], bbox[2], bbox[0]), tags={"building": True})
    fig, ax = ox.plot_footprints(gdf=gdf, save=False, show=False, close=True)

    fig.set_size_inches(image_size[0] / 100, image_size[1] / 100)  # DPI adjustment
    fig.savefig(tile_path, dpi=100, bbox_inches='tight', pad_inches=0)


def download_map(bbox):
    bbox = latlon_to_mercator_bbox(*bbox) # in Web Mercator meters

    print(bbox)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot empty ax with extent set to bbox
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])

    # Add basemap tiles from Esri World Imagery (satellite imagery)
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

    plt.axis('off')
    plt.savefig(f'dataset/images/tile{i}.png', bbox_inches='tight', pad_inches=0)
    plt.close()


# Download and save map tiles

for i in range(10):
    map_shift = i*0.01
    bbox = (40.855+map_shift, 40.859+map_shift, 14.385+map_shift, 14.389+map_shift)  # (south, north, west, east)
    tile_name = f"tile{i}"

    image_path = f"{output_dir}/masks/{tile_name}.png"
    download_map_tile(bbox, image_size, image_path)
    download_map(bbox)


print("Dataset image and mask saved")