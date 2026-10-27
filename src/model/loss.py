# src/model/loss.py
import torch
import torch.nn.functional as F


def masked_bce(logits, target, valid):
    """Per-pixel binary cross-entropy, averaged over real pixels only."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * valid).sum() / valid.sum().clamp(min=1)


def masked_dice(logits, target, valid, eps=1.0):
    """Soft Dice loss (1 - overlap score) over the whole batch, real pixels only."""
    p = torch.sigmoid(logits) * valid
    t = target * valid
    inter = (p * t).sum()
    return 1 - (2 * inter + eps) / (p.sum() + t.sum() + eps)


def loss_fn(logits, target, valid):
    """BCE gives stable gradients; Dice stops the 'predict no buildings' shortcut."""
    return 0.5 * masked_bce(logits, target, valid) + 0.5 * masked_dice(logits, target, valid)


class PixelMeter:
    """Accumulates TP/FP/FN over a whole dataset, then reports IoU, precision, recall, F1."""

    def __init__(self):
        self.tp = self.fp = self.fn = 0.0

    @torch.no_grad()
    def update(self, logits, target, valid, thr=0.5):
        v = valid > 0.5
        pred = (torch.sigmoid(logits) > thr) & v
        true = (target > 0.5) & v
        self.tp += (pred & true).sum().item()
        self.fp += (pred & ~true).sum().item()
        self.fn += (true & ~pred).sum().item()

    def compute(self):
        eps = 1e-9
        tp, fp, fn = self.tp, self.fp, self.fn
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        return dict(iou=tp / (tp + fp + fn + eps), precision=prec, recall=rec,
                    f1=2 * prec * rec / (prec + rec + eps))