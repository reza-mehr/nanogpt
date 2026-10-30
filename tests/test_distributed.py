'''
This file contains tests related to the implementation of distributed data parallelism (DDP).

'''

#%% Imports:

import pytest
import torch.multiprocessing as mp
from dataclasses import replace
import numpy as np

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from nanogpt.config import TrainConfig, ModelConfig
from nanogpt.data import TokenSplit
from nanogpt.distributed import DDPInfo
from nanogpt.model import GPT
from nanogpt.optimizer import build_optimizer
from nanogpt.trainer import train_step

#%% Training configurations:

VOCAB, T, B = 16, 8, 2

tiny_cfg = TrainConfig(
    model=ModelConfig(
        name='gpt', vocab_size=VOCAB, block_size=T, n_embed=16, head_num=2, head_size=8, n_blocks=2,
        dropout=0.                  # dropout masks would differ between the runs
        ),
        batch_mode='sequential',    # sequential deterministic batch generation
        device='cpu',               # test on 'cpu'
        max_iters=1, lr_schedule='constant', learning_rate=3.e-4,
    )

#%% Fixtures:

@pytest.fixture(scope='module')
def train_data(tmp_path_factory):
    '''Fixture to generate random training data.'''
    rng = np.random.default_rng(0)
    n_tokens = 100
    path = tmp_path_factory.mktemp('grad_accum') / 'train.bin'
    rng.integers(0, VOCAB, n_tokens, dtype=np.uint16).astype('<u2').tofile(path)
    return TokenSplit(str(path), n_tokens)

#%% Utility functions:

def _get_serial_grads(cfg: TrainConfig, train_data: TokenSplit) -> dict:
    '''Function to train the model using a single process.'''

    # Initialization:
    torch.manual_seed(0)                            # same initial weights in both runs
    model = GPT(cfg.model)
    optim = build_optimizer(model, cfg)
    ddp = DDPInfo(device=cfg.device)

    # Training loop:
    for step in range(1, cfg.max_iters+1):
        train_step(cfg, model, train_data, optim, step, ddp)

    return {name: p.grad for name, p in model.named_parameters()}


def _get_ddp_grads(cfg: TrainConfig, train_data: TokenSplit, world_size: int, tmp_path_factory) -> dict:
    '''Function to perform DDP training for a single step.'''

    # Spawn worker processes under current test process:
    tmp_dir = tmp_path_factory.mktemp('ddp_run')
    mp.spawn(_worker, args=(world_size, tmp_dir, cfg, train_data), nprocs=world_size)

    # Load the saved gradients from rank 0 process:
    grads_dict = torch.load(tmp_dir / 'ddp.pt')

    return grads_dict


def _worker(rank, world_size, tmp_dir, cfg, train_data):
    '''Function to perform one training step per process.'''

    init_process_group('gloo', init_method=f'file://{tmp_dir}/rendezvous', rank=rank, world_size=world_size)
    try:
        # Initializaiton:
        torch.manual_seed(rank)                            # same initial weights in both runs
        model = GPT(cfg.model)
        model_ddp = DDP(model, device_ids=None)
        optim = build_optimizer(model, cfg)
        ddp = DDPInfo(rank=rank, local_rank=rank, world_size=world_size, device=cfg.device)

        # Train for one step and save the gradients:
        train_step(cfg, model_ddp, train_data, optim, 1, ddp)
        if ddp.is_main:
            grads_dict = {name: p.grad for name, p in model.named_parameters()}
            torch.save(grads_dict, tmp_dir / 'ddp.pt')

    finally:
        destroy_process_group()


#%% Main code:

@pytest.mark.parametrize('world_size', [1, 2])
def test_single_step_grads_identical(tmp_path_factory, train_data, world_size):
    '''Function to test that a single DDP step of 'batch_size' is the same as a single step of single process with 'batch_size * world_size'.'''

    # Initialization:
    cfg_serial = replace(tiny_cfg, batch_size=world_size*B)
    cfg_ddp = replace(cfg_serial, batch_size=B)

    # Train both models:
    grads_serial = _get_serial_grads(cfg_serial, train_data)
    grads_ddp = _get_ddp_grads(cfg_ddp, train_data, world_size, tmp_path_factory)

    torch.testing.assert_close(grads_ddp, grads_serial)


def test_accum_grads_identical(tmp_path_factory, train_data):
    '''Function to test that the same number of gradient accumulation substeps produce identical weights.'''

    # Initialization:
    world_size = 2
    batch_tokens = 2 * B * T * world_size
    cfg_serial = replace(tiny_cfg, batch_tokens=batch_tokens, batch_size=B)
    cfg_ddp = replace(cfg_serial, batch_tokens=batch_tokens, batch_size=B)

    assert cfg_serial.get_grad_accum_iters(world_size=1) == 2 * world_size
    assert cfg_ddp.get_grad_accum_iters(world_size=world_size) == 2

    # Train both models:
    grads_serial = _get_serial_grads(cfg_serial, train_data)
    grads_ddp = _get_ddp_grads(cfg_ddp, train_data, world_size, tmp_path_factory)

    torch.testing.assert_close(grads_ddp, grads_serial)


