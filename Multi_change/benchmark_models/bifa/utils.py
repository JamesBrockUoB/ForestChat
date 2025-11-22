from typing import Iterable, Set

import torch
import torch.nn.functional as F
from torch import Tensor, einsum
from torch.optim import lr_scheduler


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))
    if seg.ndim == 4:
        seg = seg.squeeze(dim=1)
    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def get_scheduler(optimizer, args):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.scheduler_lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.num_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.schedular_lr_policy == "step":
        step_size = args.num_epochs // args.scheduler_n_step
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=args.scheduler_gamma
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", args.lr_policy
        )
    return scheduler


def cross_entropy(input, target, weight=None, reduction="mean", ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(
            input, size=target.shape[1:], mode="bilinear", align_corners=True
        )

    return F.cross_entropy(
        input=input,
        target=target,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
    )


def dice_loss(predicts, target, weight=None):
    idc = [0, 1]
    probs = torch.softmax(predicts, dim=1)
    # target = target.unsqueeze(1)
    target = class2one_hot(target, 2)
    assert simplex(probs) and simplex(target)

    pc = probs[:, idc, ...].type(torch.float32)
    tc = target[:, idc, ...].type(torch.float32)
    intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
    union: Tensor = einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc)

    divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (
        union + 1e-10
    )

    loss = divided.mean()
    return loss


def ce_dice(input, target, weight=None):
    ce_loss = cross_entropy(input, target)
    dice_loss_ = dice_loss(input, target)
    loss = 0.5 * ce_loss + 0.5 * dice_loss_
    return loss
