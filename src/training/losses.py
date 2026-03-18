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
# Focal loss
# --------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = -alpha_t * (1 - p_t).pow(self.gamma) * torch.log(p_t + self.eps)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
    
# --------------------------------------------------
# Dice BCE loss
# --------------------------------------------------
class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=1.0, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())

        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1).float()
        intersection = (probs * targets_flat).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1 - dice.mean()

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss
    
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
    
    elif loss_name == "focal":
        return FocalLoss(alpha=0.25, gamma=2.0)

    else:
        raise ValueError(f"Unknown loss: {loss_name}")