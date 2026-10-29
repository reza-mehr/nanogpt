
#%% Imports:

import pytest
from dataclasses import replace
from pathlib import Path

from nanogpt.config import PRESETS, ModelConfig, TrainConfig

'''
This file contains the code to test the config classes.

'''

#%% Test defaults with replace:

def test_looped_attention_after_replace():
    '''Function to test that replace() into the looped path accepts a default (MHA) config, keeping 'kv_head_num' unresolved.'''
    batched = ModelConfig()
    looped = replace(batched, batched_attention=False, scaled_dot_prod=False)     # must not raise


def test_head_num_override_keeps_mha():
    '''Function to test that overriding 'head_num' on a default config stays MHA by setting the 'n_kv_head' property, instead of silently becoming GQA.'''

    cfg = ModelConfig()
    assert cfg.kv_head_num is None              # argument unchanged
    assert cfg.n_kv_head == cfg.head_num        # property set to MHA

    cfg = replace(cfg, head_num=12)
    assert cfg.kv_head_num is None              # argument unchanged
    assert cfg.n_kv_head == 12                  # property still set to MHA


@pytest.mark.parametrize('overrides, feature', [
    (dict(scaled_dot_prod=True), 'Scaled dot-product attention'),
    (dict(kv_head_num=3), 'GQA'),
    (dict(pos_embed='rope'), 'Rotary positional embedding'),
], ids=['sdpa', 'gqa', 'rope'])
def test_looped_attention_unsupported(overrides, feature):
    '''Function to test that features unsupported on the looped path raise an error naming the feature.'''

    looped = dict(batched_attention=False, scaled_dot_prod=False)
    with pytest.raises(ValueError, match=feature):
        ModelConfig(**(looped | overrides))


def test_looped_attention_reports_all_conflicts():
    '''Function to test that all unsupported features on the looped path are reported together.'''

    with pytest.raises(ValueError) as err:
        ModelConfig(batched_attention=False, scaled_dot_prod=True, pos_embed='rope')
    message = str(err.value)
    assert 'Scaled dot-product attention' in message and 'Rotary positional embedding' in message


#%% Grouped-query attention (GQA) tests:

def test_kv_heads_num_default():
    cfg = ModelConfig(head_num=8)
    assert cfg.n_kv_head == cfg.head_num

@pytest.mark.parametrize('kv_head_num', [1, 4, 8])    # MQA, GQA, MHA
def test_kv_heads_num_valid(kv_head_num):
    cfg = ModelConfig(head_num=8, kv_head_num=kv_head_num)
    assert cfg.n_kv_head == kv_head_num

@pytest.mark.parametrize('kv_head_num', [0, -1, 3, 16])
def test_kv_heads_num_invalid(kv_head_num):
    with pytest.raises(ValueError):
        ModelConfig(head_num=8, kv_head_num=kv_head_num)

#%% Rotary positional embedding (RoPE) tests:

def test_rope_head_size_even():
    '''Function to ensure that only even head sizes are allowed when RoPE positional embedding is requested.'''

    ModelConfig(head_size=64, pos_embed='learned')      # no errors
    ModelConfig(head_size=63, pos_embed='learned')      # no errors
    ModelConfig(head_size=64, pos_embed='rope')         # no errors
    with pytest.raises(ValueError):
        ModelConfig(head_size=63, pos_embed='rope')

def test_rope_base():
    '''Function to ensure that RoPE base range is enforced when RoPE positional embedding is requested.'''

    ModelConfig(pos_embed='learned', rope_base=-11.)    # no errors
    ModelConfig(pos_embed='learned', rope_base=11.)     # no errors
    ModelConfig(pos_embed='rope', rope_base=11.)        # no errors
    with pytest.raises(ValueError):
        ModelConfig(pos_embed='rope', rope_base=0.)
    with pytest.raises(ValueError):
        ModelConfig(pos_embed='rope', rope_base=-11.)

#%% Training run folder names:

def test_default_run_dir_names_preset_and_device():
    '''Function to test that a training run without a name gets a folder named after its preset and device.'''
    cfg = replace(PRESETS['small_cfg'], device='cpu', out_dir='runs')
    assert cfg.run_dir == Path('runs/train_small_cfg_cpu')
    assert replace(cfg, device='mps').run_dir.name == 'train_small_cfg_mps'    # resolved lazily, never stale
    assert replace(cfg, run_name='mine').run_dir == Path('runs/mine')
    assert TrainConfig(device='cpu').run_dir.name == 'train_custom_cpu'        # built outside the presets


@pytest.mark.parametrize('run_name', ['benchmark_x', 'profile_y'])
def test_run_name_rejects_non_training_prefixes(run_name):
    '''Function to test that training runs cannot take the folder prefixes that mark benchmarks and profiles.'''
    with pytest.raises(ValueError, match='must not start with'):
        TrainConfig(run_name=run_name)

#%% Test vocab size padding:

def test_reject_incorrect_pad_vocab_to_multiple():
    '''Function to ensure that the inpermissible values of 'pad_vocab_to_multiple' raise a ValueError.'''
    with pytest.raises(ValueError,  match='pad_vocab_to_multiple'):
        ModelConfig(vocab_size=61, pad_vocab_to_multiple=0)
    with pytest.raises(ValueError,  match='pad_vocab_to_multiple'):
        ModelConfig(vocab_size=61, pad_vocab_to_multiple=3)
    with pytest.raises(ValueError,  match='pad_vocab_to_multiple'):
        ModelConfig(vocab_size=61, pad_vocab_to_multiple=-3)

def test_no_vocab_size_pad_default():
    '''Function to ensure that the 'vocab_size_padded' is the same as 'vocab_size' when 'pad_vocab_to_multiple=1'.'''
    cfg = ModelConfig(vocab_size=61)
    assert cfg.vocab_size_padded == cfg.vocab_size
    cfg = ModelConfig(vocab_size=61, pad_vocab_to_multiple=1)
    assert cfg.vocab_size_padded == cfg.vocab_size

def test_no_vocab_size_pad_if_already_multiple():
    '''Function to test that the 'vocab_size' is not raised if it is already a multiple of 'pad_vocab_to_multiple'.'''
    cfg = ModelConfig(vocab_size=64, pad_vocab_to_multiple=8)
    assert cfg.vocab_size_padded == cfg.vocab_size
    cfg = ModelConfig(vocab_size=64, pad_vocab_to_multiple=64)
    assert cfg.vocab_size_padded == cfg.vocab_size

def test_vocab_size_pad():
    '''Function to test that the 'vocab_size' is raised to a multiple of 'pad_vocab_to_multiple'.'''
    cfg = ModelConfig(vocab_size=61, pad_vocab_to_multiple=8)
    assert cfg.vocab_size_padded == 64
    cfg = ModelConfig(vocab_size=61, pad_vocab_to_multiple=128)
    assert cfg.vocab_size_padded == 128

#%% Test batch tokens:

def test_batch_tokens():
    '''Function to test various assignment scenarios for 'batch_tokens'.'''

    # Default 'batch_tokens' value:
    cfg = TrainConfig(model=ModelConfig(block_size=8), batch_size=2)
    assert cfg.batch_tokens is None
    assert cfg.get_grad_accum_iters() == 1

    # Impermissible 'batch_tokens' values:
    with pytest.raises(ValueError, match='batch_tokens'):
        TrainConfig(model=ModelConfig(block_size=8), batch_size=2, batch_tokens=-16)

    # 'batch_tokens' not a multiple of B*T:
    with pytest.raises(ValueError, match='batch_tokens'):
        TrainConfig(model=ModelConfig(block_size=8), batch_size=2, batch_tokens=17)

    # Given 'batch_tokens' value:
    cfg = TrainConfig(model=ModelConfig(block_size=8), batch_size=2, batch_tokens=16)
    assert cfg.get_grad_accum_iters() == 1

    # With gradient accumulation:
    cfg = TrainConfig(model=ModelConfig(block_size=8), batch_size=2, batch_tokens=32)
    assert cfg.get_grad_accum_iters() == 2
