#%% Imports:

import math

from nanogpt.config import ModelConfig, TrainConfig
from nanogpt.model import GPT
from nanogpt.optimizer import build_optimizer, get_lr

'''
This file contains tests for optimizer including learning rate scheduling and selective weight decay.

'''

#%% Set up the configs:

# Number of words in the synthetic dataset:
VOCAB = 16

tiny_cfg = TrainConfig(
    model=ModelConfig(name='gpt', vocab_size=VOCAB, block_size=8, n_embed=16, head_num=2, head_size=8, n_blocks=2),
    lr_schedule='cosine', learning_rate=1.0, lr_warmup_iters=10, max_iters=110, decay_groups=True,
    )

#%% Optimizer with selective weight decay:

def test_param_groups():
    tiny_model = GPT(tiny_cfg.model)
    opt = build_optimizer(tiny_model, tiny_cfg)

    decay, no_decay = opt.param_groups
    assert all(p.dim() >= 2 for p in decay['params'])
    assert all(p.dim() < 2 for p in no_decay['params'])
    n_grouped = len(decay['params']) + len(no_decay['params'])
    assert n_grouped == len(list(tiny_model.parameters()))

#%% Learning rate scheduling tests:

def test_warmup_is_linear():
    assert get_lr(step=1, cfg=tiny_cfg) == 0.1
    assert get_lr(step=5, cfg=tiny_cfg) == 0.5
    assert get_lr(step=10, cfg=tiny_cfg) == 1.0          # reaches the peak at the end of warmup

def test_cosine_midpoint_and_end():
    assert math.isclose(get_lr(60, tiny_cfg), 0.55)      # halfway: average of max and min
    assert math.isclose(get_lr(110, tiny_cfg), 0.1)      # ends exactly at min_lr

def test_never_increases_after_warmup():
    lrs = [get_lr(s, tiny_cfg) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
