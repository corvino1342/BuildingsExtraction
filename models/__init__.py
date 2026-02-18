from .unet import UNet, UNetL, UNetLL
from .deeplab import build_deeplab

def build_model(name):
    if name.startswith("unet"):
        return UNet()
    elif name.sstartswith("deeplab"):
        return build_deeplab()

    else:
        raise ValueError(f"Unknown model: {name}")