from .unet import UNet, UNetL, UNetLL
from .deeplab import DeepLabV3

def build_model(name, in_channels=3, num_classes=1):

    name = name.lower()

    if name == "unet":
        return UNet(in_channels, num_classes)

    if name == "unetl":
        return UNetL(in_channels, num_classes)

    if name == "unetll":
        return UNetLL(in_channels, num_classes)

    if name == "deeplabv3":
        return DeepLabV3(in_channels, num_classes)

    raise ValueError(f"Unknown model: {name}")