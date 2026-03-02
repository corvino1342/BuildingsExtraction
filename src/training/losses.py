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

# --------------------------------------------------
# Tversky loss
# --------------------------------------------------
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1).float()

        TP = (probs * targets).sum(dim=1)
        FP = (probs * (1 - targets)).sum(dim=1)
        FN = ((1 - probs) * targets).sum(dim=1)

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return 1 - tversky.mean()

# --------------------------------------------------
# Focal Tversky loss
# --------------------------------------------------
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1).float()

        TP = (probs * targets).sum(dim=1)
        FP = (probs * (1 - targets)).sum(dim=1)
        FN = ((1 - probs) * targets).sum(dim=1)

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        loss = (1 - tversky) ** self.gamma
        return loss.mean()

# --------------------------------------------------
# Combined Loss
# --------------------------------------------------
class CombinedLoss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, logits, targets):
        total_loss = 0
        for loss in self.losses:
            total_loss += loss(logits, targets)
        return total_loss

def build_loss(loss_name, train_dataset, device):

    loss_name = loss_name.lower()

    # ---------------------------------------
    # Weighted BCE
    # ---------------------------------------
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

    # ---------------------------------------
    # Loss selection
    # ---------------------------------------

    if loss_name == "bce":
        return bce

    elif loss_name == "wbce":
        return bce

    elif loss_name == "dice":
        return DiceLoss()

    elif loss_name == "wbce_dice":
        return CombinedLoss([bce, DiceLoss()])

    elif loss_name == "tversky":
        return TverskyLoss(alpha=0.3, beta=0.7)

    elif loss_name == "focal_tversky":
        return FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=2.0)

    else:
        raise ValueError(f"Unknown loss: {loss_name}")