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

dataset_type = 'training' # training or test

# INITIAL COORDINATES TRAINING
base_lat, base_lon = Decimal("40.964"), Decimal("14.223")

# INITIAL COORDINATES TRAINING
# base_lat, base_lon = Decimal("40.546"), Decimal("15.483")


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
def download_map_tile(latmin, latmax, lonmin, lonmax, extent, filename):
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
        fig.savefig(f'dataset/{dataset_type}/masks/tile-{filename}.png', dpi=100, bbox_inches=None, pad_inches=0, facecolor='black')
        plt.close(fig)

        # Save the GeoDataFrame as a GeoJSON
        gdf.to_file(f"dataset/{dataset_type}/geojson/tile-{filename}.geojson", driver='GeoJSON')
        fig.set_size_inches(5.12, 5.12)
        skipping_flag = False

    except InsufficientResponseError:
        print("Empty tile!")
        skipping_flag = True
    return skipping_flag

# DOWNLOAD MAP SATELLITE
def download_map(extent, skipping_flag, filename):

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
        plt.imsave(f'dataset/{dataset_type}/images/tile-{filename}.png', img)# Download and save map tiles

# TILE LOOP
"""
dataset_dim = 200

for i in range(dataset_dim):

    print(f'\n##################\nTILE NUMBER: {i}/{dataset_dim}\n##################\n')
    map_shift = Decimal(i*0.01)
    half_dimension = Decimal("0.001")

    # (+ NORTH shift) (- SOUTH shift)
    lat = base_lat + map_shift

    # (+ EAST shift) (- WEST shift)
    lon = base_lon - map_shift

    latsouth = lat - half_dimension
    latnorth = lat + half_dimension
    lonwest = lon - half_dimension
    loneast = lon + half_dimension

    print(f'\t\tCOORDINATES\n\t\t\t{latnorth}\t\t\t\n{lonwest}\t\t\t\t\t{loneast}\n\t\t\t{latsouth}\n')
    xmin, ymin, xmax, ymax = latlon_to_mercator_bbox(south=latsouth, north=latnorth, west=lonwest, east=loneast)

    extent_mercator = (xmin, xmax, ymin, ymax)

    skip = download_map_tile(latmin=latsouth, latmax=latnorth, lonmin=lonwest, lonmax=loneast, extent=extent_mercator, filename=f'{i}')
    download_map(extent=extent_mercator, skipping_flag=skip, filename=f'{i}')
"""


# TILE GRID
rows = 24
cols = 25

half_dimension = Decimal("0.002")

lat_start = base_lat - (rows // 2) * half_dimension * 2
lon_start = base_lon - (rows // 2) * half_dimension * 2
index = 0
for i in range(rows):
    for j in range(cols):
        index += 1
        print(f'\n##################\nTILE NUMBER: {index}/{rows*cols}\n##################\n')

        latmin = lat_start + i * half_dimension * 2
        latmax = latmin + half_dimension * 2
        lonmin = lon_start + j * half_dimension * 2
        lonmax = lonmin + half_dimension * 2

        print(f'\t\tCOORDINATES\n\t\t\t{latmax}\t\t\t\n{lonmin}\t\t\t\t\t{lonmax}\n\t\t\t{latmin}\n')
        xmin, ymin, xmax, ymax = latlon_to_mercator_bbox(south=latmin,
                                                         north=latmax,
                                                         west=lonmin,
                                                         east=lonmax)

        extent_mercator = (xmin, xmax, ymin, ymax)

        skip = download_map_tile(latmin=latmin,
                                 latmax=latmax,
                                 lonmin=lonmin,
                                 lonmax=lonmax,
                                 extent=extent_mercator,
                                 filename=f'{i}_{j}')

        download_map(extent=extent_mercator,
                     skipping_flag=skip,
                     filename=f'{i}_{j}')

print("Dataset image and mask saved")