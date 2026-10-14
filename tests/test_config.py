
#%% Imports:

import pytest
from dataclasses import replace

from nanogpt.config import ModelConfig

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
