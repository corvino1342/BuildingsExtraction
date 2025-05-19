import os
import osmnx as ox
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer


# CONFIGURATION
output_dir = "dataset"

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
    print(f'Coordinates:\n{latmin, latmax, lonmin, lonmax}')
    gdf = ox.features_from_bbox((latmax, latmin, lonmax, lonmin), tags={"building": True})

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

def download_map(latmin, latmax, lonmin, lonmax, output, zoom=18):

    xmin, ymin, xmax, ymax = latlon_to_mercator_bbox(south=latmin,
                                                     north=latmax,
                                                     west=lonmin,
                                                     east=lonmax)

    print(xmin, ymin, xmax, ymax)
    print(latmin, latmax, lonmin, lonmax)

    # Compute bounding box for 512x512 pixels at given zoom
    tile_size = 512
    res = 156543.03392804097 / (2 ** zoom)
    half_size = (tile_size // 2) * res

    # Convert center point to mercator
    img, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, source=ctx.providers.Esri.WorldImagery)

    # Save image
    plt.imsave(output, img)# Download and save map tiles

for i in range(10):
    map_shift = i*0.01

    tile_dimension = 0.002

    lat = 40.857 + map_shift
    lon = 14.387 + map_shift

    bbox = (40.855+map_shift, 40.859+map_shift, 14.385+map_shift, 14.389+map_shift)  # (south, north, west, east)

    latmin = lat - tile_dimension
    latmax = lat + tile_dimension
    lonmin = lon - tile_dimension
    lonmax = lon + tile_dimension


    print(f'Bounding Box:\n{bbox}\n')

    download_map_tile(latmin=latmin, latmax=latmax, lonmin=lonmin, lonmax=lonmax)
    #download_map(latmin=latmin, latmax=latmax, lonmin=lonmin, lonmax=lonmax, output=f'dataset/images/tile{i}.png', zoom=18)


print("Dataset image and mask saved")