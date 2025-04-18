from libraries import *


# Class to define the double convolution for the UNet.
# After each 2D convolution we have a batch normalization and a ReLU activation function
# The size for the convolution is 3x3 with 1 padding to get the same dimension
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        def forward(self, x):
            return self.double_conv(x)


# Downsampling using a pooling of 2x2.
# It reduces the spatial dimensions using max pooling, then applies the DoubleConv defined above
# This is the step of reducing the resolution of the image progressively with this encoding step
# while increasing feature depth
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# Upsampling using a pooling of 2x2
# This step increases the spatial dimensions using either bilinear interpolation or transposed convolution
# After that, there is a feature map from the downsampling path.
# This is the decoding step, divided into upsample -> combine -> refine
class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # These are the 2 upsampling methods, based on the flag bilinear. DoubleConv is used to refine features
        # This does a smooth interpolation
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels//2)
        # This uses a learnable upsampling layer
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    # x1 is the lower-resolution feature map from the decoder
    # x2 is the matching feature map from the encoder (skip connection)
    def forward(self, x1, x2):

        # This is useful to align the dimensions of x1 and x2
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                        diffY//2, diffY - diffY//2])
        ''' 
        For padding issues:
        https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        '''

        # This is used to concatenate along the channel dimension x1 and x2. The .conv is used to fuse them
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# The output of the Convolutional Network
# The final layer that converts the deep feature map into the desired number of output channels
# Basically, it is a classifier
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)