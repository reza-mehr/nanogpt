#%% Imports:

import math
import torch
from torch import nn

from nanogpt.config import TrainConfig

'''
This file contains the code for optimizer settings including learning rate scheduling and selective weight decay.

'''

#%% Selective weight decay:

def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    '''
    Function to apply weight decay to all or 2D+ tensors only, based on provided configuration file.

    Parameters
    ----------
    model: nn.Module
        Model being trained.
    cfg: TrainConfig
        Specified configurations.

    Returns
    -------
    optim: torch.optim.AdamW
        Configured optimizer.

    '''
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.decay_groups:
        groups = [
            {'params': [p for p in params if p.dim() >= 2], 'weight_decay': cfg.weight_decay},
            {'params': [p for p in params if p.dim() < 2], 'weight_decay': 0.0},        # do not decay
        ]
    else:
        groups = [{'params': params, 'weight_decay': cfg.weight_decay}]

    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), fused=cfg.adamw_fused)


#%% Learning rate scheduling:

def _warmup_cosine(step: int, cfg: TrainConfig) -> float:
    '''
    Function to perform linear warmup and then cosine decay to the requested floor.

    Parameters
    ----------
    step: int
        Current step.
    cfg: TrainConfig
        Specified configurations.

    Returns
    -------
    lr: float
        Learning rate at current step.

    '''
    # Initialization:
    max_lr = cfg.learning_rate
    min_lr = cfg.min_learning_rate
    warmup_iters = cfg.lr_warmup_iters
    max_iters = cfg.max_iters

    # Increase learning rate toward 'max_lr' for warm-up iterations:
    if step <= warmup_iters: return max_lr * step / warmup_iters

    # Cosine decay from warm-up iterations onwards:
    progress = (step - warmup_iters) / (max_iters - warmup_iters)   # 0 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))              # 1 -> 0

    return min_lr + coeff * (max_lr - min_lr)


def get_lr(step: int, cfg: TrainConfig) -> float:
    '''
    Function to perform linear warmup and then cosine decay to the requested floor.

    Parameters
    ----------
    step: int
        Current step.
    cfg: TrainConfig
        Specified configurations.

    Returns
    -------
    lr: float
        Learning rate at current step.

    Raises
    ------
    If an unknown scheduling rate is requested.

    '''
    if cfg.lr_schedule == 'constant': return cfg.learning_rate
    if cfg.lr_schedule == 'cosine': return _warmup_cosine(step, cfg)
    raise ValueError(f'Unknown lr_schedule: {cfg.lr_schedule}')


def set_learning_rate(optimizer, step, cfg):
    '''
    Function to update the learning rate according to the requested schedule.

    Parameters
    ----------
    optimizer: torch.optim
        Optimizer object to set the learning rate for.
    step: int
        Current step.
    cfg: TrainConfig
        Specified configurations.

    Returns
    -------
    lr: float
        Learning rate.

    '''
    lr = get_lr(step, cfg)
    for group in optimizer.param_groups:
        group['lr'] = lr

    return lr
