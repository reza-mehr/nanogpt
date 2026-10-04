#%% Imports:

import math

from nanogpt.schedule import get_lr
from nanogpt.config import ModelConfig, TrainConfig

'''
This file contains tests for learning rate scheduler.

'''

#%% Set up the configs:

cfg = TrainConfig(lr_schedule='cosine', learning_rate=1.0, lr_warmup_iters=10, max_iters=110)

#%% Main code:

def test_warmup_is_linear():
    assert get_lr(step=1, cfg=cfg) == 0.1
    assert get_lr(step=5, cfg=cfg) == 0.5
    assert get_lr(step=10, cfg=cfg) == 1.0          # reaches the peak at the end of warmup

def test_cosine_midpoint_and_end():
    assert math.isclose(get_lr(60, cfg), 0.55)      # halfway: average of max and min
    assert math.isclose(get_lr(110, cfg), 0.1)      # ends exactly at min_lr

def test_never_increases_after_warmup():
    lrs = [get_lr(s, cfg) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
