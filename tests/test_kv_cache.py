#%% Imports:

import torch
import pytest

from nanogpt.config import ModelConfig
from nanogpt.registry import build_model
import nanogpt.model        # noqa: F401  (registers the models)
from nanogpt.kv_cache import KVCache

#%% Main code:

@pytest.mark.parametrize('sdpa', [True, False])
def test_cached_logits_match_full_forward(sdpa):
    '''Function to test consistency of cahced key-values with the ones generated live.'''

    # Initialization:
    torch.manual_seed(0)
    n_blocks, batch_size, head_num, max_len, head_size, device = 2, 2, 2, 32, 8, 'cpu'
    cfg = ModelConfig(name='gpt', vocab_size=16, block_size=max_len, n_embed=16,
                      head_num=head_num, head_size=head_size, n_blocks=n_blocks, dropout=0.0, scaled_dot_prod=sdpa)
    model = build_model(cfg).to(device).eval()
    seq = torch.randint(0, 16, (batch_size, 20))

    with torch.no_grad():
        full, _ = model(seq)                                   # (B, 20, V): one pass

        cache = KVCache(n_blocks, batch_size, head_num, max_len, head_size, device)
        prefill, _ = model(seq[:, :8], cache=cache)            # first 8 tokens
        steps = [prefill]
        for t in range(8, 20):                                 # then one token at a time
            out, _ = model(seq[:, t:t + 1], cache=cache)
            steps.append(out)
        cached = torch.cat(steps, dim=1)

    assert torch.allclose(full, cached, atol=1e-5)
