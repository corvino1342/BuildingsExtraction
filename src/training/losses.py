import torch
import torch.nn as nn
from tqdm import tqdm


# --------------------------------------------------
# Dice loss
# --------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
                probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        return 1 - dice.mean()

def build_loss(loss_name, train_dataset, device):

    dice = None

    if "wbce" in loss_name:
        pos, total = 0, 0
        for _, mask in tqdm(train_dataset, desc="Computing Weights"):
            pos += mask.sum().item()
            total += mask.numel()
        neg = total - pos
        weight = torch.tensor(neg / (pos + 1e-6)).to(device)
        print(f"Weight: {weight.item():.2f}")
        bce = nn.BCEWithLogitsLoss(pos_weight=weight)
    else:
        bce = nn.BCEWithLogitsLoss()

    if "dice" in loss_name:
        dice = DiceLoss()

    return bce, dice