import os
import geopandas as gpd
from shapely.geometry import Polygon
from PIL import Image, ImageDraw

def rasterize_buildings(buildings_gdf, bounds, out_size=(512, 512)): # I need to rasterize buildings to create an image from osm files
    minx, miny, maxx, maxy = bounds
    w, h = out_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for geom in buildings_gdf.geometry:
        if geom.is_empty or not geom.is_valid or not isinstance(geom, Polygon):
            continue
        try:
            coords = [(int((x - minx) / (maxx - minx) * w),
                       int((1 - (y - miny) / (maxy - miny)) * h)) for x, y in geom.exterior.coords]
            draw.polygon(coords, outline=1, fill=1)
        except Exception as e:
            print(f"Skipping geometry due to error: {e}")

    return mask

def process_geojson_folder(input_folder, output_folder, out_size=(512, 512)):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".geojson"):
            filepath = os.path.join(input_folder, filename)
            print(f"Processing {filepath}...")

            try:
                gdf = gpd.read_file(filepath)
                buildings = gdf[gdf["building"].notnull()]
                if buildings.empty:
                    print(f"No buildings found in {filename}")
                    continue

                bounds = gdf.total_bounds
                mask = rasterize_buildings(buildings, bounds, out_size=out_size)

                out_name = os.path.splitext(filename)[0] + "_mask.png"
                out_path = os.path.join(output_folder, out_name)
                mask.save(out_path)
                print(f"Saved mask to {out_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_folder = "osm_geojson"  # Folder containing GeoJSON files
    output_folder = "masks"       # Where to save mask images
    process_geojson_folder(input_folder, output_folder)