
'''
This file contains code to test the generation utilities and functions.

'''

#%% Imports:

import pytest
from dataclasses import replace
import torch

from nanogpt.config import small_cfg
from nanogpt.model import build_model
from nanogpt.tokenizers import CharTokenizer
from nanogpt.generation import truncate_at_end, generate_text

#%% Global variables:

SAMPLE = 'Once upon a time, there was a little dog.\nHe liked to run!'

#%% Tests:

def test_truncate_at_end():
    assert truncate_at_end([1, 2, 9, 3, 9], end_id=9, start=0) == [1, 2]
    assert truncate_at_end([9, 2, 3, 9, 4], end_id=9, start=1) == [9, 2, 3]   # skips the prompt
    assert truncate_at_end([1, 2, 3], end_id=9, start=0) == [1, 2, 3]         # no end token
    assert truncate_at_end([1, 0, 2], end_id=0, start=0) == [1]               # ID 0 is valid
    assert truncate_at_end([1, 9, 2], end_id=None, start=0) == [1, 9, 2]      # no stop configured


@pytest.mark.parametrize('device',
            ['cpu',
             pytest.param('mps', marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason='no MPS')),
             pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='no GPU')),
             ])
def test_random_seed_restoration(device):
    '''Function to test that the training and generation random seeds are unaffected by each other and managed independently.'''

    # Initialization:
    tok = CharTokenizer(SAMPLE)
    cfg = replace(small_cfg, model=replace(small_cfg.model, vocab_size=tok.vocab_size), device=device)
    model = build_model(cfg.model).to(device)
    prompt, new_tokens = '', 10
    seed = cfg.seed + 1

    def _get_rng_state(device):
        '''Function to get the RNG state for the current device.'''
        if device == 'cpu': return torch.get_rng_state()
        elif device == 'mps': return torch.mps.get_rng_state()
        else: return torch.cuda.get_rng_state()

    # Test that the training seed stays independent of the generation seed:
    torch.manual_seed(cfg.seed)
    before, before_cpu = _get_rng_state(device), _get_rng_state('cpu')
    text1 = generate_text(model, tok, prompt, new_tokens, temperature=1.0, top_k=None, use_cache=False, seed=seed)
    assert torch.equal(before, _get_rng_state(device))
    assert torch.equal(before_cpu, _get_rng_state('cpu'))   # fork_rng always forks CPU generator regardless of 'device'

    # Test that for a fixed seed, generation stays independent of the training seed:
    torch.manual_seed(cfg.seed+2)
    text2 = generate_text(model, tok, prompt, new_tokens, temperature=1.0, top_k=None, use_cache=False, seed=seed)
    assert text1 == text2
