import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt

place = 'avigliano3'
gdf = gpd.read_file(f"/osm/{place}Footprints.osm", layer='multipolygons')
buildings = gdf[gdf["building"].notnull()].copy()

print(buildings.other_tags)


buildings.plot(figsize=(8, 8))
plt.title(f"Buildings in {place}")
plt.savefig(f"/osm/{place}Footprints.png")