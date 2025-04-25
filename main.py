# This is the main script of the U-Net algorithm

# A U-Net architecture is a particular class of CNN involved
# in the segmentation of biomedical images.

# First we have an encoding path, with 2 or more convolutions 3x3
# followed each by a ReLU activation function and a pooling operation 2x2

# The next step is to decode the information with the up-sampling of the map
# with a 2x2 convolution, and other 2 3x3 convolutions, each followed by a ReLU

# The main problem is to use the images I can get from www.openstreetmap.org

from libraries import *

# Load multiple tags from your .osm file
tags = {"building": True, "highway": True}
gdf = ox.features_from_xml("map.osm", tags=tags)

print(gdf.keys())

fig, ax = plt.subplots(figsize=(6, 6))

for feat in gdf.keys():
    color = '#%06x' % random.randint(0, 0xFFFFFF)
    feature = gdf[gdf[f'{feat}'].notnull()]
    feature.plot(ax=ax, color=color, linewidth=1, label=f"{feat}")

plt.legend()
plt.axis("off")
plt.savefig("map_colored.png", dpi=300, bbox_inches="tight", pad_inches=0)
plt.show()
