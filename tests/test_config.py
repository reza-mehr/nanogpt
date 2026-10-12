
#%% Imports:

import pytest
from dataclasses import replace

from nanogpt.config import ModelConfig

'''
This file contains the code to test the config classes.

'''

#%% Main code:

def test_kv_heads_num_default():
    cfg = ModelConfig(head_num=8)
    assert cfg.n_kv_head == cfg.head_num

@pytest.mark.parametrize('n_kv_head', [1, 4, 8])
def test_kv_heads_num_valid(n_kv_head):
    cfg = ModelConfig(head_num=8, n_kv_head=n_kv_head)
    assert cfg.n_kv_head == n_kv_head

@pytest.mark.parametrize('n_kv_head', [0, -1, 3, 16])
def test_kv_heads_num_invalid(n_kv_head):
    with pytest.raises(ValueError):
        ModelConfig(head_num=8, n_kv_head=n_kv_head)
