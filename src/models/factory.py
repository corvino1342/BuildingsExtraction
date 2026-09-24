from src.models.unet import UNet, UNetL, UNetLL


def build_model(arch, in_channels=3, n_classes=1):
    if arch == "unet":
        return UNet(n_channels=in_channels, n_classes=n_classes)
    elif arch == "unetL":
        return UNetL(n_channels=in_channels, n_classes=n_classes)
    elif arch == "unetLL":
        return UNetLL(n_channels=in_channels, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
