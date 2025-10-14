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
from rasterio import features
from affine import Affine

getcontext().prec = 9

# ----------------------------
# CONFIGURATION
# ----------------------------

HorizontalFlip = A.Compose([A.HorizontalFlip(p=1)])
VerticalFlip = A.Compose([A.VerticalFlip(p=1)])

dataset_type = 'train' # training, validation or test

# ----------------------------
# INITIAL COORDINATES
# ----------------------------

base_lat, base_lon = Decimal("40.719"), Decimal("14.483")

if dataset_type == 'test':
    base_lat, base_lon = Decimal("40.646"), Decimal("15.017")
elif dataset_type == 'validation':
    base_lat, base_lon = Decimal("40.358"), Decimal("15.823")

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

# ----------------------------
# DOWNLOAD MAP GDF
# ----------------------------

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

def rasterize_mask(gdf, xmin, ymin, xmax, ymax, out_shape=(512, 512)):
    transform = Affine((xmax - xmin) / out_shape[1], 0, xmin,
                       0, (ymin - ymax) / out_shape[0], ymax)
    mask = features.rasterize(
        ((geom, 1) for geom in gdf.geometry),
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    return mask

def png_saves(gdf, extent, filename):

    # ----------------------------
    ########### MASK and SATELLITE ###########
    # ----------------------------

    zoom = 19
    tile_size = 512

    xmin, xmax, ymin, ymax = extent

    # Convert center point to mercator and get image extent
    img, img_extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, source=ctx.providers.Esri.WorldImagery)

    # Use img_extent to rasterize mask aligned with image
    mask_img = rasterize_mask(gdf, img_extent[0], img_extent[2], img_extent[1], img_extent[3], out_shape=(tile_size,tile_size))

    # Resize to 512×512 if necessary
    if img.shape[0] != tile_size or img.shape[1] != tile_size:
        img = np.array(Image.fromarray(img).resize((tile_size, tile_size), resample=Image.BILINEAR))

    # Save image and mask
    Image.fromarray(mask_img * 255).save(f'dataset/{dataset_type}/masks/tile-{filename}.png')
    plt.imsave(f'dataset/{dataset_type}/images/tile-{filename}.png', img)

    # Save the GeoDataFrame as a GeoJSON
    gdf.to_file(f"dataset/{dataset_type}/geojson/tile-{filename}.geojson", driver='GeoJSON')

    # Data augmentation with horizontal and vertical flipping
    if dataset_type != 'test':
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

# ----------------------------
# TILE GRID
# ----------------------------

rows = 5
cols = 5

tile_dimension = Decimal("0.001")

lat_start = base_lat - (rows // 2) * tile_dimension
lon_start = base_lon - (rows // 2) * tile_dimension
index = 0
for i in range(rows):
    for j in range(cols):
        index += 1
        print(f'\n##################\nTILE NUMBER: {index}/{rows*cols}\n##################\n')

        latmin = lat_start + i * tile_dimension
        latmax = latmin + tile_dimension
        lonmin = lon_start + j * tile_dimension
        lonmax = lonmin + tile_dimension

        print(f'COORDINATES\n{latmax}, {lonmin}\t\t\t\t{latmax}, {lonmax}\n\n\n\n'
              f'{latmin}, {lonmin}\t\t\t\t{latmin}, {lonmax}\n')

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

print("Maps and Masks saved!")