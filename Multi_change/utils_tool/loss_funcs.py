import numpy as np
import torch


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
