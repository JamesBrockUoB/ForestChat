import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    """
    Initialize weights for neural network modules using best practices.

    This function recursively initializes weights in a module:
    - Conv2D layers: Kaiming normal initialization (fan_in, relu)
    - BatchNorm and GroupNorm: Weights set to 1, biases to 0
    - Linear layers: Kaiming normal initialization (fan_in, relu)
    - Sequential containers: Each component initialized individually
    - Pooling, ModuleList, Loss functions: Skipped (no initialization needed)

    Args:
        module: PyTorch module whose weights will be initialized
    """
    # Process all named children in the module
    for name, child_module in module.named_children():
        # Skip modules that don't need initialization
        if isinstance(
            child_module,
            (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.ModuleList, nn.BCELoss),
        ):
            continue

        # Initialize convolutional layers
        elif isinstance(child_module, nn.Conv2d):
            nn.init.kaiming_normal_(
                child_module.weight, mode="fan_in", nonlinearity="relu"
            )
            if child_module.bias is not None:
                nn.init.zeros_(child_module.bias)

        # Initialize normalization layers
        elif isinstance(child_module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(child_module.weight)
            if child_module.bias is not None:
                nn.init.zeros_(child_module.bias)

        # Initialize linear layers
        elif isinstance(child_module, nn.Linear):
            nn.init.kaiming_normal_(
                child_module.weight, mode="fan_in", nonlinearity="relu"
            )
            if child_module.bias is not None:
                nn.init.zeros_(child_module.bias)

        # Handle Sequential containers
        elif isinstance(child_module, nn.Sequential):
            for seq_name, seq_module in child_module.named_children():
                if isinstance(seq_module, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        seq_module.weight, mode="fan_in", nonlinearity="relu"
                    )
                    if seq_module.bias is not None:
                        nn.init.zeros_(seq_module.bias)

                elif isinstance(seq_module, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(seq_module.weight)
                    if seq_module.bias is not None:
                        nn.init.zeros_(seq_module.bias)

                elif isinstance(seq_module, nn.Linear):
                    nn.init.kaiming_normal_(
                        seq_module.weight, mode="fan_in", nonlinearity="relu"
                    )
                    if seq_module.bias is not None:
                        nn.init.zeros_(seq_module.bias)
                else:
                    # Recursively initialize other modules in sequential container
                    weight_init(seq_module)

        # Recursively handle other module types
        elif len(list(child_module.children())) > 0:
            weight_init(child_module)


def adjust_learning_rate(
    args,
    optimizer,
    epoch=None,
    batch_idx=None,
    total_batches=None,
    lr_factor=1.0,
    shrink_factor=None,
    verbose=True,
):
    """
    Adjust learning rate based on scheduler type, epoch, iteration, or explicit shrinking.

    This function supports multiple learning rate adjustment strategies:
    1. Step decay: Reduces LR at fixed intervals
    2. Polynomial decay: Smoothly reduces LR according to a polynomial function
    3. Manual shrinking: Explicitly shrinks LR by a specified factor
    4. Warm-up phase: Gradually increases LR at the beginning of training

    Args:
        args: Command line arguments containing lr_mode, lr, step_loss, max_epochs
        optimizer: Optimizer instance whose learning rate will be adjusted
        epoch: Current epoch (required for step and poly modes)
        iter: Current iteration (required for poly mode and warm-up)
        max_batches: Total batches per epoch (required for poly mode)
        lr_factor: Additional scaling factor for the learning rate (default: 1.0)
        shrink_factor: If provided, explicitly shrink LR by this factor (0-1)
        verbose: Whether to print the learning rate change (default: True)

    Returns:
        float: Current learning rate after adjustment
    """

    base_lr = args.lr

    # --- Manual shrink override ---
    if shrink_factor is not None:
        if not 0 < shrink_factor < 1:
            raise ValueError(f"shrink_factor must be in (0,1), got {shrink_factor}")

        for pg in optimizer.param_groups:
            pg["lr"] *= shrink_factor

        if verbose:
            print(f"[LR Shrink] New LR: {optimizer.param_groups[0]['lr']:.6f}")
        return optimizer.param_groups[0]["lr"]

    # --- Scheduler-based update ---
    if args.lr_mode == "step":
        # Step decay every N epochs
        lr = base_lr * (0.1 ** (epoch // args.step_loss))

    elif args.lr_mode == "poly":
        # Polynomial decay across total epochs
        if total_batches is not None and batch_idx is not None:
            progress = (epoch + batch_idx / total_batches) / args.max_epochs
        else:
            progress = epoch / args.max_epochs
        lr = base_lr * (1 - progress) ** 0.9

    else:
        raise ValueError(f"Unknown lr_mode '{args.lr_mode}'")

    # --- Warmup (optional) ---
    if epoch == 0 and batch_idx is not None and batch_idx < 200:
        warmup = 0.9 * (batch_idx + 1) / 200 + 0.1
        lr = base_lr * warmup

    # --- Apply scaling factor ---
    lr *= lr_factor

    # --- Update optimizer ---
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    return lr


def BCEDiceLoss(inputs, targets):
    """
    Combined BCE and Dice loss for binary segmentation.

    Args:
        inputs: Model predictions after sigmoid
        targets: Ground truth binary masks

    Returns:
        Combined loss value
    """
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(
            weight=weight, ignore_index=ignore_index, reduction="mean"
        )

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
    label_change: changed part
    """

    def __init__(self, reduction="mean"):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0.0, reduction=reduction)

    def forward(self, x1, x2, label_change):
        b, c, h, w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b * h * w, c])
        x2 = torch.reshape(x2, [b * h * w, c])

        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target, [b * h * w])

        loss = self.loss_f(x1, x2, target)
        return loss
