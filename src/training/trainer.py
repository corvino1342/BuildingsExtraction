import torch
from src.evaluation.metrics import iou_score, precision_score, recall_score


class Trainer:

    def __init__(self, model, optimizer, scaler, bce, dice, device, use_amp=True):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.bce = bce
        self.dice = dice
        self.device = device
        self.use_amp = use_amp


    def train_one_epoch(self, loader):

        self.model.train()

        loss_sum = iou_sum = p_sum = r_sum = 0.0

        for imgs, masks in loader:

            imgs = imgs.to(self.device)
            masks = masks.to(self.device).float()

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):

                out = self.model(imgs)
                loss = self.bce(out, masks)

                if self.dice:
                    loss += self.dice(out, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                loss_sum += loss.item()
                iou_sum += iou_score(out, masks).item()
                p_sum += precision_score(out, masks).item()
                r_sum += recall_score(out, masks).item()

        n = len(loader)

        return {
            "loss": loss_sum / n,
            "iou": iou_sum / n,
            "precision": p_sum / n,
            "recall": r_sum / n,
        }

    # --------------------------------------------------
    # Training / validation
    # --------------------------------------------------


    @torch.no_grad()
    def validate(self, loader):

        self.model.eval()

        loss_sum = iou_sum = p_sum = r_sum = 0.0

        for imgs, masks in loader:

            imgs = imgs.to(self.device)
            masks = masks.to(self.device).float()

            out = self.model(imgs)
            loss = self.bce(out, masks)

            if self.dice:
                loss += self.dice(out, masks)

            loss_sum += loss.item()
            iou_sum += iou_score(out, masks).item()
            p_sum += precision_score(out, masks).item()
            r_sum += recall_score(out, masks).item()

        n = len(loader)

        return {
            "loss": loss_sum / n,
            "iou": iou_sum / n,
            "precision": p_sum / n,
            "recall": r_sum / n,
        }

