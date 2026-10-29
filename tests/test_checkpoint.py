
'''
This file contains the code to test checkpoint resumption and the inference-only checkpoint.

'''

#%% Imports:

import subprocess
import numpy as np
from dataclasses import replace

import pytest
import torch

from nanogpt import checkpoint
from nanogpt.checkpoint import git_info, load_model_tokenizer, save_checkpoint
from nanogpt.checkpoint import check_inference_checkpoint, save_inference_checkpoint
from nanogpt.config import ModelConfig, TrainConfig
from nanogpt.data import TokenSplit
from nanogpt.model import build_model
from nanogpt.optimizer import build_optimizer
from nanogpt.tokenizers import CharTokenizer
from nanogpt.trainer import train

#%% Data and config setup for this test:

VOCAB = 16                      # number of tokens in the synthetic dataset
TEXT = 'abcdefghijklmnop'       # VOCAB distinct characters, so the tokenizer matches the synthetic data

@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, VOCAB, 2000).astype('<u2')
    tokens[:1800].tofile(tmp_path / 'train.bin')
    tokens[1800:].tofile(tmp_path / 'val.bin')
    return TokenSplit(tmp_path / 'train.bin', 1800), TokenSplit(tmp_path / 'val.bin', 200)


def tiny_cfg(tmp_path, run_name: str) -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(name='gpt', vocab_size=VOCAB, block_size=8, n_embed=16,
                          head_num=2, head_size=8, n_blocks=2, dropout=0.1),     # dropout exercises RNG restore
        batch_size=4, max_iters=20, eval_interval=5, eval_iters=2,
        ckpt_interval=5, learning_rate=1e-3, lr_warmup_iters=5, seed=0, device='cpu',
        out_dir=str(tmp_path), run_name=run_name,
    )

#%% Main code:

@pytest.mark.parametrize('stop_after', [5, 7])
def test_resume_matches_continuous_run(tmp_path, data, stop_after):
    '''Function to test that an interrupted run resumes as expected.'''

    # Generate synthetic data:
    train_data, val_data = data
    tok_meta = CharTokenizer(TEXT).to_meta()    # real metadata: the end of train() rebuilds the tokenizer from model.pt

    # Run A: 20 steps without interruption:
    model_a, best_loss_a = train(tiny_cfg(tmp_path, 'a'), train_data, val_data, tok_meta)

    # Run B: same config, "crashes" after stop_after, then resumes to the end:
    cfg_b = tiny_cfg(tmp_path, 'b')
    train(cfg_b, train_data, val_data, tok_meta, stop_after=stop_after)
    assert (cfg_b.run_dir / 'latest.pt').exists()
    model_b, best_loss_b = train(replace(cfg_b, resume=True), train_data, val_data, tok_meta)

    # Assertions:
    assert best_loss_a==best_loss_b
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for name in sd_a:
        assert torch.equal(sd_a[name], sd_b[name]), f'weights differ in {name}'

#%% Inference-only checkpoint:

def _training_checkpoint(tmp_path):
    '''Function to save a training checkpoint of a fresh tiny GPT model and return its path.'''
    cfg = tiny_cfg(tmp_path, 'export')
    cfg = replace(cfg, model=replace(cfg.model, name='gpt'))
    torch.manual_seed(0)
    model = build_model(cfg.model)
    optim = build_optimizer(model, cfg)
    path = tmp_path / 'best.pt'
    save_checkpoint(path, model, optim, 1, CharTokenizer(TEXT).to_meta(), 1.0, cfg, git_info(), [])
    return path


def test_inference_checkpoint_contents(tmp_path):
    '''Function to test that the export keeps only what inference needs and stores bf16 weights.'''

    src, dst = _training_checkpoint(tmp_path), tmp_path / 'model.pt'
    save_inference_checkpoint(src, dst)
    ckpt = torch.load(dst, map_location='cpu', weights_only=False)

    assert set(ckpt) == {'model', 'cfg', 'tokenizer', 'step', 'best_val', 'git'}    # no optimizer or RNG states
    assert all(v.dtype == torch.bfloat16 for v in ckpt['model'].values() if v.is_floating_point())


def test_inference_checkpoint_round_trip(tmp_path):
    '''Function to test that the export loads as an fp32 model whose logits match the training checkpoint's.'''

    src, dst = _training_checkpoint(tmp_path), tmp_path / 'model.pt'
    save_inference_checkpoint(src, dst)
    assert check_inference_checkpoint(src, dst) <= 1e-2

    model, tok, _ = load_model_tokenizer(dst, 'cpu')
    assert all(p.dtype == torch.float32 for p in model.parameters())                # bf16 storage, fp32 compute
    assert tok.decode(tok.encode('abc')) == 'abc'                                   # tokenizer rebuilt


def test_inference_checkpoint_detects_wrong_weights(tmp_path):
    '''Function to test the negative control: perturbed weights fail the check.'''

    src, dst = _training_checkpoint(tmp_path), tmp_path / 'model.pt'
    save_inference_checkpoint(src, dst)

    ckpt = torch.load(dst, map_location='cpu', weights_only=False)
    torch.manual_seed(1)
    ckpt['model'] = {k: v + 0.1 * torch.randn_like(v) if v.is_floating_point() else v for k, v in ckpt['model'].items()}
    torch.save(ckpt, dst)

    with pytest.raises(RuntimeError, match='logits differ'):
        check_inference_checkpoint(src, dst)

#%% Code version:

def _fake_git(tag: str | None):
    '''Function to return a stand-in for subprocess.check_output that answers git queries without a repository.'''
    def check_output(cmd, **kwargs):
        if cmd[1] == 'describe':
            if tag is None:
                raise subprocess.CalledProcessError(128, cmd)       # what git does when no tag is reachable
            return f'{tag}\n'
        return {'rev-parse': 'abc1234\n', 'diff': '', 'ls-files': ''}[cmd[1]]
    return check_output


@pytest.mark.parametrize('tag', ['v0.1-3-gabc1234', None], ids=['tagged', 'untagged'])
def test_git_info_records_tag_and_commit(monkeypatch, tag):
    '''Function to test that git_info() records the readable tag, and that a missing tag does not lose the commit.'''
    monkeypatch.setattr(checkpoint.subprocess, 'check_output', _fake_git(tag))
    info = git_info()
    assert info['tag'] == tag
    assert info['commit'] == 'abc1234' and info['dirty'] is False
