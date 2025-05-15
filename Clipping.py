from shapely.geometry import box
import geopandas as gpd
import json
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt

# Define bounding box in EPSG:4326 (lat/lon)
minx, miny, maxx, maxy = 14.385, 40.855, 14.389, 40.859
bbox = box(minx, miny, maxx, maxy)

fileName = 'map1'

# Open raster
with rasterio.open(f"georef/{fileName}_georef.tif") as src:
    print("Raster CRS:", src.crs)
    print(bbox)
    print("Raster bounds:", src.bounds)

    # Convert to GeoJSON geometry
    bbox_geom = [json.loads(gpd.GeoSeries([bbox]).to_json())['features'][0]['geometry']]

    print(bbox_geom)
    # Clip the raster
    out_image, out_transform = mask(src, bbox_geom, crop=True)
    out_meta = src.meta.copy()
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    with rasterio.open(f"clipped/{fileName}_clipped.tif", "w", **out_meta) as dest:
        dest.write(out_image)
plt.imshow(out_image[0], cmap='gray')
plt.show()
print("Raster clipped and saved.")