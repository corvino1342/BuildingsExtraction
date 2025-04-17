# This is the main script of the U-Net algorithm

# A U-Net architecture is a particular class of CNN involved
# in the segmentation of biomedical images.

# First we have an encoding path, with 2 or more convolutions 3x3
# followed each by a ReLU activation function and a pooling operation 2x2

# The next step is to decode the information with the up-sampling of the map
# with a 2x2 convolution, and other 2 3x3 convolutions, each followed by a ReLU

