#%% Imports:

import torch
import pytest

from nanogpt.config import ModelConfig
from nanogpt.model import build_model
from nanogpt.kv_cache import KVCache
from nanogpt.utils import autocast_ctx

#%% Main code:

@pytest.mark.parametrize('sdpa', [True, False])
@pytest.mark.parametrize('dtype', ['fp32', 'mixed_bf16', 'bf16'])
def test_cached_logits_match_full_forward(sdpa, dtype):
    '''Function to test consistency of cahced key-values with the ones generated live.'''

    # Initialization:
    torch.manual_seed(0)
    n_blocks, batch_size, head_num, max_len, head_size, device = 2, 2, 2, 32, 8, 'cpu'
    cfg = ModelConfig(name='gpt', vocab_size=16, block_size=max_len, n_embed=16,
                      head_num=head_num, head_size=head_size, n_blocks=n_blocks, dropout=0.0, scaled_dot_prod=sdpa)
    model = build_model(cfg).to(device).eval()
    seq = torch.randint(0, 16, (batch_size, 20))

    with torch.no_grad():
        with autocast_ctx(device, dtype):
            full, _ = model(seq)                        # (B, 20, V): one pass

        cache = KVCache(n_blocks, batch_size, head_num, max_len, head_size, device, dtype)
        with autocast_ctx(device, dtype):
            prefill, _ = model(seq[:, :8], cache=cache) # first 8 tokens
        steps = [prefill]
        for t in range(8, 20):                          # then one token at a time
            with autocast_ctx(device, dtype):
                out, _ = model(seq[:, t:t + 1], cache=cache)
            steps.append(out)
        cached = torch.cat(steps, dim=1)

    full, cached = full.float(), cached.float()
    if dtype == 'fp32':
        assert torch.allclose(full, cached, atol=1e-5)
    else:           # use a smaller tolerance for 'bf16' to factor in rounding errors
        with torch.no_grad():
            ref32, _ = model(seq)                       # fp32 ground truth
        bf16_err = (full - ref32).abs().max()
        cache_err = (full - cached).abs().max()
        assert cache_err <= 3 * bf16_err + 1e-3, f'cache {cache_err:.4f} vs bf16 noise {bf16_err:.4f}'
