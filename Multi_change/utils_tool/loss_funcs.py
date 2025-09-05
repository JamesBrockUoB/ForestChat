import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        """
        Focal Loss for segmentation (works for binary C=2 and multi-class C>2).
        Args:
            alpha: balancing factor (float or list of class weights for each class).
            gamma: focusing parameter.
            reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) raw scores from model (C >= 2).
            targets: (B, H, W) ground truth class indices in [0..C-1].
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        targets_one_hot = (
            F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        )

        ce_loss = -targets_one_hot * torch.log(probs + 1e-8)  # cross-entropy per class
        pt = (probs * targets_one_hot).sum(dim=1)  # probability of the true class

        focal_weight = (1 - pt) ** self.gamma
        focal = focal_weight.unsqueeze(1) * ce_loss

        if self.alpha is not None:
            alpha = torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype)
            alpha_t = alpha[targets]  # per-pixel weight
            focal = focal * alpha_t.unsqueeze(1)

        focal = focal.sum(dim=1)  # sum over classes

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal
