import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()

        weights = "DEFAULT" if pretrained else None
        self.model = deeplabv3_resnet50(weights=weights)

        # Replace classifier head
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.model(x)
        return out["out"]