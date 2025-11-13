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


def adjust_lr(
    args,
    optimizer,
    epoch=None,
    iter=None,
    max_batches=None,
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
    if shrink_factor is not None:
        # Manual shrinking mode (from the second implementation)
        if not 0 < shrink_factor < 1:
            raise ValueError(
                f"Shrink factor must be between 0 and 1, got {shrink_factor}"
            )

        if verbose:
            print("\nDECAYING learning rate.")

        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * shrink_factor

        if verbose:
            print(f"The new learning rate is {optimizer.param_groups[0]['lr']:.6f}\n")

        return optimizer.param_groups[0]["lr"]

    # Scheduler-based learning rate adjustment
    if args.lr_mode == "step":
        if epoch is None:
            raise ValueError("Epoch must be provided for step lr_mode")
        lr = args.lr * (0.1 ** (epoch // args.step_loss))

    elif args.lr_mode == "poly":
        if any(param is None for param in [epoch, iter, max_batches]):
            raise ValueError(
                "Epoch, iter, and max_batches must be provided for poly lr_mode"
            )

        cur_iter = iter
        max_iter = max_batches * args.num_epochs
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError(f"Unknown lr mode {args.lr_mode}")

    # Apply warm-up phase if we're in the first epoch
    if epoch == 0 and iter is not None and iter < 200:
        lr = args.lr * 0.9 * (iter + 1) / 200 + 0.1 * args.lr

    # Apply additional lr factor
    lr *= lr_factor

    # Update learning rate for all parameter groups
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr
