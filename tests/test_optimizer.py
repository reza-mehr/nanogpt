'''
This file contains tests for optimizer including learning rate scheduling, selective weight decay, and gradient accumulation.

'''

#%% Imports:

import math
import torch
import numpy as np
import pytest
from dataclasses import replace

from nanogpt.config import ModelConfig, TrainConfig
from nanogpt.data import TokenSplit, get_batch_rand
from nanogpt.distributed import DDPInfo
from nanogpt.model import GPT
from nanogpt.optimizer import build_optimizer, get_lr
import nanogpt.trainer
from nanogpt.trainer import train_step

#%% Training configurations:

VOCAB, T, B = 16, 8, 2

tiny_cfg = TrainConfig(
    model=ModelConfig(name='gpt', vocab_size=VOCAB, block_size=T, n_embed=16, head_num=2, head_size=8, n_blocks=2),
    batch_size=B, lr_schedule='cosine', learning_rate=1.0, lr_warmup_iters=10, max_iters=110, decay_groups=True,
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

#%% Gradient accumulation test:

@pytest.fixture(scope='module')
def train_data(tmp_path_factory):
    '''Fixture to generate random training data.'''
    rng = np.random.default_rng(0)
    n_tokens = 100
    path = tmp_path_factory.mktemp('grad_accum') / 'train.bin'
    rng.integers(0, VOCAB, n_tokens, dtype=np.uint16).astype('<u2').tofile(path)
    return TokenSplit(str(path), n_tokens)


def test_accumulated_gradient_matches(train_data, monkeypatch):
    '''Function to test that the accumulated gradient of two sub-steps matches the gradient of one large batch.'''

    # Configs for one batch of 2*B sequences, and for two accumulated sub-steps of B sequences:
    device = 'cpu'
    model_cfg = replace(tiny_cfg.model, dropout=0.)                     # dropout masks would differ between the runs
    cfg_full = replace(tiny_cfg, model=model_cfg, batch_size=2*B, lr_schedule='constant', learning_rate=3e-4,
                       decay_groups=False, device=device)
    cfg_accum = replace(cfg_full, batch_size=B, batch_tokens=2*B*T)     # specify total batch tokens
    assert cfg_accum.get_grad_accum_iters() == 2

    # One fixed batch shared by both runs, so they see the same sequences:
    torch.manual_seed(0)
    x, y = get_batch_rand(train_data, cfg_full)                         # (2*B, T)

    def run(cfg, batches):
        '''Function to run the production training step on the given batches and return the gradients.'''
        batches = iter(batches)
        monkeypatch.setattr(nanogpt.trainer, 'get_batch', lambda *args, **kwargs: next(batches))
        torch.manual_seed(0)                            # same initial weights in both runs
        model = GPT(cfg.model)
        optim = build_optimizer(model, cfg)
        ddp = DDPInfo(device=device)                    # single process, no environment reads
        train_step(cfg, model, train_data, optim, 1, ddp)
        assert next(batches, None) is None              # train_step consumed every batch
        return {name: p.grad for name, p in model.named_parameters()}

    grads_full = run(cfg_full, [(x, y)])
    grads_accum = run(cfg_accum, [(x[:B], y[:B]), (x[B:], y[B:])])

    torch.testing.assert_close(grads_accum, grads_full)
