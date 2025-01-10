import torch

import numpy as np

from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.cosine_epochs = total_epochs - warmup_epochs
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [self.base_lr * warmup_factor for _ in self.optimizer.param_groups]
        else:
            # Cosine annealing phase
            cosine_epoch = self.last_epoch - self.warmup_epochs
            cosine_factor = 0.5 * (1 + np.cos(torch.pi * cosine_epoch / self.cosine_epochs))
            return [self.base_lr * cosine_factor for _ in self.optimizer.param_groups]


def logscalemagnitude(z):
    #return z
    # log transform magnitude of z
    z_mag_log = torch.log1p(torch.abs(z) + 1e-6)
    # keep phase of z
    z_phase = torch.angle(z)
    # reconstruct z from log magnitude and phase
    z = torch.polar(z_mag_log, z_phase)
    return z
