import os
import osmnx as ox
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer
from PIL import Image
import numpy as np
from decimal import Decimal, getcontext      # I need to use it to ensure the correct operations with decimals
from osmnx._errors import InsufficientResponseError
import geopandas as gpd

getcontext().prec = 9
# CONFIGURATION
dataset_dim = 10

dataset_type = 'test' # training or test

# INITIAL COORDINATES
base_lat, base_lon = Decimal("40.857"), Decimal("14.387")
half_dimension = Decimal("0.002")

# Create folders
os.makedirs(f"dataset/{dataset_type}/images", exist_ok=True)
os.makedirs(f"dataset/{dataset_type}/masks", exist_ok=True)
os.makedirs(f"dataset/{dataset_type}/geojson", exist_ok=True)

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


# Conversion for the satellite map
def latlon_to_mercator_bbox(south, north, west, east):

    xmin, ymin = transformer.transform(west, south)
    xmax, ymax = transformer.transform(east, north)

    return xmin, ymin, xmax, ymax

# DOWNLOAD MAP IMAGE
def download_map_tile(latmin, latmax, lonmin, lonmax, extent):
    xmin, xmax, ymin, ymax = extent

    try:
        # The right order of coordinates is: west, south, east, north
        bbox = (float(lonmin), float(latmin), float(lonmax), float(latmax))
        gdf = ox.features_from_bbox(bbox, tags={"building": True})

        # Reproject to match extent (Web Mercator)
        gdf = gdf.to_crs("EPSG:3857")
        fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100, facecolor='black')  # Black background
        gdf.plot(ax=ax, color='white', edgecolor='none')  # White buildings

        ax.set_axis_off()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(f'dataset/{dataset_type}/masks/tile{i}.png', dpi=100, bbox_inches=None, pad_inches=0, facecolor='black')
        plt.close(fig)

        # Save the GeoDataFrame as a GeoJSON
        gdf.to_file(f"dataset/{dataset_type}/geojson/tile{i}.geojson", driver='GeoJSON')
        fig.set_size_inches(5.12, 5.12)
        skipping_flag = False

    except InsufficientResponseError:
        print("Empty tile!")
        skipping_flag = True
    return skipping_flag

# DOWNLOAD MAP SATELLITE
def download_map(extent, skipping_flag):

    if not skipping_flag:
        zoom = 18
        xmin, xmax, ymin, ymax = extent

        # Compute bounding box for 512x512 pixels at given zoom

        # Convert center point to mercator
        img, _ = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, source=ctx.providers.Esri.WorldImagery)
        tile_size = 512

        # Resize to 512×512 if necessary
        if img.shape[0] != tile_size or img.shape[1] != tile_size:
            img = np.array(Image.fromarray(img).resize((tile_size, tile_size), resample=Image.BILINEAR))

        # Save image
        plt.imsave(f'dataset/{dataset_type}/images/tile{i}.png', img)# Download and save map tiles

# TILE LOOP
for i in range(dataset_dim):

    print(f'\n##################\nTILE NUMBER: {i}/{dataset_dim}\n##################\n')
    map_shift = Decimal(i*0.01)

    lat = base_lat + map_shift
    lon = base_lon + map_shift

    latsouth = lat - half_dimension
    latnorth = lat + half_dimension
    lonwest = lon - half_dimension
    loneast = lon + half_dimension

    print(f'\t\tCOORDINATES\n\t\t\t{latnorth}\t\t\t\n{lonwest}\t\t\t\t\t{loneast}\n\t\t\t{latsouth}\n')
    xmin, ymin, xmax, ymax = latlon_to_mercator_bbox(south=latsouth, north=latnorth, west=lonwest, east=loneast)

    extent_mercator = (xmin, xmax, ymin, ymax)

    skip = download_map_tile(latmin=latsouth, latmax=latnorth, lonmin=lonwest, lonmax=loneast, extent=extent_mercator)
    download_map(extent=extent_mercator, skipping_flag=skip)


print("Dataset image and mask saved")