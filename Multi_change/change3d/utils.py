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


def BCEDiceLoss(inputs, targets, eps=1e-5, weight=None):
    """
    Combined BCE and Dice loss for binary segmentation.

    Args:
        inputs: Model predictions after sigmoid
        targets: Ground truth binary masks

    Returns:
        Combined loss value
    """
    bce = F.binary_cross_entropy(inputs, targets, weight=weight)

    inter = (inputs * targets).sum(dim=(2, 3))
    union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    dice_loss = 1 - dice.mean()

    return bce + dice_loss
