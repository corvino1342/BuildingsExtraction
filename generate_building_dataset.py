import os
import osmnx as ox
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer
from PIL import Image
import numpy as np
from decimal import Decimal, getcontext      # I need to use it to ensure the correct operations with decimals


getcontext().prec = 9
# CONFIGURATION
output_dir = "dataset"
dataset_dim = 30

# INITIAL COORDINATES
base_lat, base_lon = Decimal("40.857"), Decimal("14.387")
half_dimension = Decimal("0.002")

# Create folders
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/masks", exist_ok=True)
os.makedirs(f"{output_dir}/geojson", exist_ok=True)

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


# Conversion for the satellite map
def latlon_to_mercator_bbox(south, north, west, east):

    xmin, ymin = transformer.transform(west, south)
    xmax, ymax = transformer.transform(east, north)

    return xmin, ymin, xmax, ymax

# DOWNLOAD MAP IMAGE
def download_map_tile(latmin, latmax, lonmin, lonmax):


    # The right order of coordinates is: west, south, east, north
    bbox = (float(lonmin), float(latmin), float(lonmax), float(latmax))

    gdf = ox.features_from_bbox(bbox, tags={"building": True})

    #fig, ax = ox.plot_footprints(gdf=gdf, save=False, show=False, close=True)
    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100, facecolor='black')  # Black background
    gdf.plot(ax=ax, color='white', edgecolor='none')  # White buildings

    ax.set_axis_off()
    ax.set_xlim(gdf.total_bounds[[0, 2]])
    ax.set_ylim(gdf.total_bounds[[1, 3]])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(f'dataset/masks/tile{i}.png', dpi=100, bbox_inches=None, pad_inches=0, facecolor='black')
    plt.close(fig)

    # Save the GeoDataFrame as a GeoJSON
    gdf.to_file(f"dataset/geojson/tile{i}.geojson", driver='GeoJSON')
    fig.set_size_inches(5.12, 5.12)

# DOWNLOAD MAP SATELLITE
def download_map(latmin, latmax, lonmin, lonmax):

    zoom = 18
    xmin, ymin, xmax, ymax = latlon_to_mercator_bbox(south=latmin,
                                                     north=latmax,
                                                     west=lonmin,
                                                     east=lonmax)

    print(xmin, ymin, xmax, ymax)
    print(latmin, latmax, lonmin, lonmax)

    # Compute bounding box for 512x512 pixels at given zoom
    tile_size = 512

    # Convert center point to mercator
    img, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, source=ctx.providers.Esri.WorldImagery)

    # Resize to 512×512 if necessary
    if img.shape[0] != tile_size or img.shape[1] != tile_size:
        img = np.array(Image.fromarray(img).resize((tile_size, tile_size), resample=Image.BILINEAR))

    # Save image
    plt.imsave(f'dataset/images/tile{i}.png', img)# Download and save map tiles

for i in range(dataset_dim):

    print(f'##################\nTILE NUMBER: {i}/{dataset_dim}\n##################\n')

    map_shift = Decimal(i*0.01)

    lat = base_lat + map_shift
    lon = base_lon + map_shift

    latsouth = lat - half_dimension
    latnorth = lat + half_dimension
    lonwest = lon - half_dimension
    loneast = lon + half_dimension

    print(f'\t\tCOORDINATES\n\t\t\t{latnorth}\t\t\t\n{lonwest}\t\t\t\t\t{loneast}\n\t\t\t{latsouth}\n\n')

    download_map_tile(latmin=latsouth, latmax=latnorth, lonmin=lonwest, lonmax=loneast)
    download_map(latmin=latsouth, latmax=latnorth, lonmin=lonwest, lonmax=loneast)


print("Dataset image and mask saved")