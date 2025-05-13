import osmnx as ox

map_label = 'map1'

bbox = (40.855, 40.859, 14.385, 14.389)

print("Bounding box:", bbox)

tags = {"building": True}

# Use of the Bounding Box to filter just the buildings
# north, south, east, west
buildings = ox.features_from_bbox(bbox=(bbox[3], bbox[1], bbox[2], bbox[0]), tags=tags)

print(buildings.head())
buildings.to_file(f"osm_geojson/{map_label}.geojson", driver="GeoJSON")