#%% Imports:

import math

from nanogpt.config import TrainConfig

'''
This file contains the code for learning rate scheduling.
'''

#%% Main code:

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

    '''
    lr = get_lr(step, cfg)
    for group in optimizer.param_groups:
        group['lr'] = lr
