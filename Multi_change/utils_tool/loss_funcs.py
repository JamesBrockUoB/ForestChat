from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize


class EDWA:
    """
    Enhanced Dynamic Weight Averaging (EDWA) for multitask learning
    """

    def __init__(self, num_epochs, T=2.0, device="cpu"):
        self.T = T
        self.num_epochs = num_epochs
        self.device = device

        self.loss_history_cd = []
        self.loss_history_cc = []

        self.lambda_weights = np.ones((2, num_epochs), dtype=np.float32)

        self.eps = 1e-8
        self.max_exp_arg = 20.0
        self.min_ratio = 1e-6
        self.max_ratio = 1e6

    def update_epoch_losses(self, L_cd_avg, L_cc_avg):
        self.loss_history_cd.append(float(L_cd_avg))
        self.loss_history_cc.append(float(L_cc_avg))

        idx = len(self.loss_history_cd) - 1
        if idx < 2:
            self.lambda_weights[:, idx] = 1.0
        else:
            w_cd = self.loss_history_cd[idx - 1] / (
                self.loss_history_cd[idx - 2] + self.eps
            )
            w_cc = self.loss_history_cc[idx - 1] / (
                self.loss_history_cc[idx - 2] + self.eps
            )

            a_cd = np.clip(w_cd / self.T, -self.max_exp_arg, self.max_exp_arg)
            a_cc = np.clip(w_cc / self.T, -self.max_exp_arg, self.max_exp_arg)

            exp_cd, exp_cc = np.exp(a_cd), np.exp(a_cc)
            denom = exp_cd + exp_cc + self.eps

            self.lambda_weights[0, idx] = 2.0 * exp_cd / denom
            self.lambda_weights[1, idx] = 2.0 * exp_cc / denom

    def combine(self, L_cd, L_cc, epoch):
        lambda_cd_np = self.lambda_weights[0, epoch]
        lambda_cc_np = self.lambda_weights[1, epoch]

        lambda_cd = torch.tensor(lambda_cd_np, dtype=L_cd.dtype, device=self.device)
        lambda_cc = torch.tensor(lambda_cc_np, dtype=L_cc.dtype, device=self.device)

        if L_cd.item() > L_cc.item():
            ratio = float(L_cd.item() / (L_cc.item() + self.eps))
            ratio = np.clip(ratio, self.min_ratio, self.max_ratio)
            alpha_s = lambda_cc * ratio
            total_loss = alpha_s * L_cc + lambda_cd * L_cd
        else:
            ratio = float(L_cc.item() / (L_cd.item() + self.eps))
            ratio = np.clip(ratio, self.min_ratio, self.max_ratio)
            alpha_s = lambda_cd * ratio
            total_loss = alpha_s * L_cd + lambda_cc * L_cc

        return total_loss


def calc_uncertainty_weighting_loss(det_loss, cap_loss, log_vars):
    precision_det = torch.exp(-log_vars[0])
    precision_cap = torch.exp(-log_vars[1])
    loss = precision_det * det_loss + log_vars[0] * 0.5
    loss += precision_cap * cap_loss + log_vars[1] * 0.5
    return loss


def graddrop(grads):
    P = 0.5 * (1.0 + grads.sum(1) / (grads.abs().sum(1) + 1e-8))
    U = torch.rand_like(grads[:, 0])
    M = P.gt(U).view(-1, 1) * grads.gt(0) + P.lt(U).view(-1, 1) * grads.lt(0)
    g = (grads * M.float()).mean(1)
    return g


def pcgrad(grads, rng, num_tasks):
    grad_vec = grads.t()

    shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
    for i in range(num_tasks):
        task_indices = np.arange(num_tasks)
        task_indices[i] = task_indices[-1]
        shuffled_task_indices[i] = task_indices[:-1]
        rng.shuffle(shuffled_task_indices[i])
    shuffled_task_indices = shuffled_task_indices.T

    normalized_grad_vec = grad_vec / (
        grad_vec.norm(dim=1, keepdim=True) + 1e-8
    )  # num_tasks x dim
    modified_grad_vec = deepcopy(grad_vec)
    for task_indices in shuffled_task_indices:
        normalized_shuffled_grad = normalized_grad_vec[task_indices]  # num_tasks x dim
        dot = (modified_grad_vec * normalized_shuffled_grad).sum(
            dim=1, keepdim=True
        )  # num_tasks x dim
        modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
    g = modified_grad_vec.mean(dim=0)
    return g


def cagrad(grads, num_tasks, alpha=0.5, rescale=1):
    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

    x_start = np.ones(num_tasks) / num_tasks
    bnds = tuple((0, 1) for x in x_start)
    cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (
            x.reshape(1, num_tasks).dot(A).dot(b.reshape(num_tasks, 1))
            + c
            * np.sqrt(
                x.reshape(1, num_tasks).dot(A).dot(x.reshape(num_tasks, 1)) + 1e-8
            )
        ).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha**2)
    else:
        return g / (1 + alpha)


def grad2vec(target, grads: torch.Tensor, task: int, grad_dims: list[int] = None):
    """
    Flatten and store gradients of all parameters in a model or list of modules into a vector.

    Args:
        target (nn.Module or list[nn.Module]): Model or list of modules.
        grads (torch.Tensor): Tensor to store flattened gradients, shape [total_params, num_tasks].
        task (int): Index of the task to store gradients for.
        grad_dims (list[int], optional): Number of parameters per parameter. Computed if None.
    """
    # Convert single module to list
    if isinstance(target, nn.Module):
        modules = [target]
    elif isinstance(target, (list, tuple)):
        modules = target
    else:
        raise ValueError("target must be a nn.Module or a list/tuple of modules.")

    # Flatten all parameters
    all_params = []
    for m in modules:
        all_params.extend(list(m.parameters()))

    # Compute grad_dims if not provided
    if grad_dims is None:
        grad_dims = [p.numel() for p in all_params]

    if grads.size(0) != sum(grad_dims):
        raise ValueError(
            f"Size mismatch: grads has {grads.size(0)} elements, "
            f"expected {sum(grad_dims)} based on model parameters."
        )

    # Reset task column
    grads[:, task].fill_(0.0)

    # Store gradients
    cnt = 0
    for param, dim in zip(all_params, grad_dims):
        if param.grad is not None:
            grad_cur = param.grad.data.detach().clone().view(-1)
            beg = sum(grad_dims[:cnt])
            en = beg + dim
            grads[beg:en, task].copy_(grad_cur)
        cnt += 1


def overwrite_grad(
    target, newgrad: torch.Tensor, num_tasks: int = 1, grad_dims: list[int] = None
):
    """
    Overwrite gradients of all parameters in a model or list of modules from a flattened gradient vector.

    Args:
        target (nn.Module or list[nn.Module]): Model or list of submodules to update.
        newgrad (torch.Tensor): Flattened gradient vector (sum of task gradients).
        num_tasks (int): Number of tasks used to scale the gradient.
        grad_dims (list[int], optional): Number of parameters per parameter. If None, computed automatically.
    """
    # Convert single module to list
    if isinstance(target, nn.Module):
        modules = [target]
    elif isinstance(target, (list, tuple)):
        modules = target
    else:
        raise ValueError("target must be a nn.Module or a list/tuple of modules.")

    # Flatten all parameters
    all_params = []
    for m in modules:
        all_params.extend(list(m.parameters()))

    # Compute grad_dims if not provided
    if grad_dims is None:
        grad_dims = [p.numel() for p in all_params]

    if newgrad.numel() != sum(grad_dims):
        raise ValueError(
            f"Size mismatch: newgrad has {newgrad.numel()} elements, "
            f"expected {sum(grad_dims)} based on model parameters."
        )

    # Scale gradient
    newgrad = newgrad * num_tasks

    # Overwrite each parameter's gradient
    cnt = 0
    for param, dim in zip(all_params, grad_dims):
        beg = sum(grad_dims[:cnt])
        en = beg + dim
        param.grad = newgrad[beg:en].contiguous().view(param.size()).clone()
        cnt += 1
