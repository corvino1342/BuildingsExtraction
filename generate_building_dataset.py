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
import albumentations as A

getcontext().prec = 9
# CONFIGURATION

HorizontalFlip = A.Compose([A.HorizontalFlip(p=1)])
VerticalFlip = A.Compose([A.VerticalFlip(p=1)])

dataset_type = 'training' # training or test

# INITIAL COORDINATES
if dataset_type == 'training':
    base_lat, base_lon = Decimal("40.719"), Decimal("14.483")
elif dataset_type == 'test':
    base_lat, base_lon = Decimal("40.798"), Decimal("14.770")


# Create folders
os.makedirs(f"dataset/{dataset_type}/images", exist_ok=True)
os.makedirs(f"dataset/{dataset_type}/masks", exist_ok=True)
os.makedirs(f"dataset/{dataset_type}/geojson", exist_ok=True)

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


# Conversion for the satellite map
def latlon_to_mercator_bbox(south, north, west, east):

    xmin, ymin = transformer.transform(west, south)
    xmax, ymax = transformer.transform(east, north)

    return xmin, xmax, ymin, ymax

# DOWNLOAD MAP GDF
def map_gdf(latmin, latmax, lonmin, lonmax, gdf=None, skip=False):

    try:
        # The right order of coordinates is: west, south, east, north
        bbox = (float(lonmin), float(latmin), float(lonmax), float(latmax))
        gdf = ox.features_from_bbox(bbox, tags={"building": True})

        # Reproject to match extent (Web Mercator)
        gdf = gdf.to_crs("EPSG:3857")

    except InsufficientResponseError:
        skip = True
        print("Empty tile!")

    return gdf, skip


def png_saves(gdf, extent, filename):

    ########### MASK ###########
    xmin, xmax, ymin, ymax = extent

    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100, facecolor='black')  # Black background
    gdf.plot(ax=ax, color='white', edgecolor='none')  # White buildings

    ax.set_axis_off()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(f'dataset/{dataset_type}/masks/tile-{filename}.png',
                dpi=100, bbox_inches=None, pad_inches=0, facecolor='black')
    plt.close(fig)

    # The flipping must need that images and masks to be arrays
    mask_img = np.array(Image.open(f'dataset/{dataset_type}/masks/tile-{filename}.png').convert("L"))

    # Save the GeoDataFrame as a GeoJSON
    gdf.to_file(f"dataset/{dataset_type}/geojson/tile-{filename}.geojson", driver='GeoJSON')
    fig.set_size_inches(5.12, 5.12)

    ########### SATELLITE ###########
    zoom = 18
    # Convert center point to mercator
    img, _ = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, source=ctx.providers.Esri.WorldImagery)
    tile_size = 512

    # Resize to 512×512 if necessary
    if img.shape[0] != tile_size or img.shape[1] != tile_size:
        img = np.array(Image.fromarray(img).resize((tile_size, tile_size), resample=Image.BILINEAR))

    # Save image
    plt.imsave(f'dataset/{dataset_type}/images/tile-{filename}.png', img) # Download and save map tiles

    # Apply augmentation

    ###### HORIZONTAL FLIPPING ######
    augmented = HorizontalFlip(image=img, mask=mask_img)
    image_aug = augmented["image"]
    mask_aug = augmented["mask"]

    Image.fromarray(image_aug).save(f"dataset/{dataset_type}/images/tile-{filename}_h.png")
    Image.fromarray(mask_aug).save(f"dataset/{dataset_type}/masks/tile-{filename}_h.png")

    ###### VERTICAL FLIPPING ######
    augmented = VerticalFlip(image=img, mask=mask_img)
    image_aug = augmented["image"]
    mask_aug = augmented["mask"]

    Image.fromarray(image_aug).save(f"dataset/{dataset_type}/images/tile-{filename}_v.png")
    Image.fromarray(mask_aug).save(f"dataset/{dataset_type}/masks/tile-{filename}_v.png")


# TILE GRID
rows = 6
cols = 6

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

        extent_mercator = latlon_to_mercator_bbox(south=latmin,
                                                  north=latmax,
                                                  west=lonmin,
                                                  east=lonmax)

        gdf, skip = map_gdf(latmin=latmin,
                       latmax=latmax,
                       lonmin=lonmin,
                       lonmax=lonmax)

        if not skip:

            png_saves(gdf=gdf,
                      extent=extent_mercator,
                      filename=f'{i}_{j}')


print("Dataset image and mask saved")