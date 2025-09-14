import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=0.5, reduction="mean", eps=1e-8):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        inputs_soft = F.softmax(inputs, dim=1).clamp(min=self.eps, max=1.0 - self.eps)

        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Gather p_t directly instead of using BCE → avoids sqrt(0) issue
        p_t = (
            (inputs_soft * targets_one_hot)
            .sum(dim=1)
            .clamp(min=self.eps, max=1.0 - self.eps)
        )

        focal_factor = (1.0 - p_t).pow(self.gamma)  # safe now, never 0^fractional
        loss = -self.alpha * focal_factor * torch.log(p_t)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]

        inputs_soft = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)
        inputs_flat = inputs_soft.reshape(inputs_soft.shape[0], num_classes, -1)
        targets_flat = targets_one_hot.reshape(
            targets_one_hot.shape[0], num_classes, -1
        )

        intersection = (inputs_flat * targets_flat).sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (
            inputs_flat.sum(dim=2) + targets_flat.sum(dim=2) + self.smooth
        )
        return 1 - dice.mean()


def compute_cd_loss(seg_pred, seg_label):
    focal = FocalLoss(alpha=1.0, gamma=0.5)
    dice = DiceLoss()
    return focal(seg_pred, seg_label) + dice(seg_pred, seg_label)


# ----------------------
# EDWA Dynamic Task Weighting
# ----------------------
def compute_multitask_loss(l_cd_curr, l_cc_curr, l_cd_prev, l_cc_prev, T=2.0):
    eps = 1e-8
    # Relative rates
    w_cd = l_cd_curr / (l_cd_prev + eps)
    w_cc = l_cc_curr / (l_cc_prev + eps)

    # Clamp to avoid overflow in exp
    w_cd_clamped = torch.clamp(w_cd / T, -20.0, 20.0)
    w_cc_clamped = torch.clamp(w_cc / T, -20.0, 20.0)

    exp_cd = torch.exp(w_cd_clamped)
    exp_cc = torch.exp(w_cc_clamped)

    lambda_cd = 2 * exp_cd / (exp_cd + exp_cc + eps)
    lambda_cc = 2 * exp_cc / (exp_cd + exp_cc + eps)

    if l_cd_curr.item() > l_cc_curr.item():
        alpha_s = lambda_cc * (l_cd_curr.item() / (l_cc_curr.item() + eps))
        final_loss = alpha_s * l_cc_curr + lambda_cd * l_cd_curr
    else:
        alpha_s = lambda_cd * (l_cc_curr.item() / (l_cd_curr.item() + eps))
        final_loss = alpha_s * l_cd_curr + lambda_cc * l_cc_curr

    return final_loss
