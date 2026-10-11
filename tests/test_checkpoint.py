
#%% Imports:

import numpy as np
from dataclasses import replace

import pytest
import torch

from nanogpt.config import ModelConfig, TrainConfig
from nanogpt.data import TokenSplit
from nanogpt.trainer import train

'''
This file contains the code to test checkpoint resumption.

'''

#%% Data and config setup for this test:

# Number of words in the synthetic dataset:
VOCAB = 16

@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, VOCAB, 2000).astype('<u2')
    tokens[:1800].tofile(tmp_path / 'train.bin')
    tokens[1800:].tofile(tmp_path / 'val.bin')
    return TokenSplit(tmp_path / 'train.bin'), TokenSplit(tmp_path / 'val.bin')


def tiny_cfg(tmp_path, run_name: str) -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(name='bigram', vocab_size=VOCAB, block_size=8, n_embed=16,
                          head_num=2, head_size=8, n_blocks=2, dropout=0.1),     # dropout exercises RNG restore
        batch_size=4, max_iters=20, eval_interval=5, eval_iters=2,
        ckpt_interval=5, learning_rate=1e-3, lr_warmup_iters=5, seed=0, device='cpu',
        runs_root=str(tmp_path), run_name=run_name,
    )

#%% Main code:

@pytest.mark.parametrize('stop_after', [5, 7])
def test_resume_matches_continuous_run(tmp_path, data, stop_after):

    # Generate synthetic data:
    train_data, val_data = data

    # Run A: 20 steps without interruption:
    model_a, best_loss_a = train(tiny_cfg(tmp_path, 'a'), train_data, val_data)

    # Run B: same config, "crashes" after stop_after, then resumes to the end:
    cfg_b = tiny_cfg(tmp_path, 'b')
    train(cfg_b, train_data, val_data, stop_after=stop_after)
    assert (cfg_b.out_dir / 'latest.pt').exists()
    model_b, best_loss_b = train(replace(cfg_b, resume=True), train_data, val_data)

    # Assertions:
    assert best_loss_a==best_loss_b
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for name in sd_a:
        assert torch.equal(sd_a[name], sd_b[name]), f'weights differ in {name}'
