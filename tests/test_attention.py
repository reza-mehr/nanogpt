
#%% Imports:

import torch
from dataclasses import replace

from nanogpt.config import ModelConfig
from nanogpt.model import MultiHeadLooped, MultiHead

#%% Main code:

def test_batched_attention_matches_naive():
    '''Function to test that the looped and batched versions of the multi-head attention are identical.'''

    # Initialization:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=16, block_size=8, n_embed=16, head_num=4, head_size=5, n_blocks=1, dropout=0.0)
    naive = MultiHeadLooped(cfg).eval()
    cfg = replace(cfg, batched_attention=True)
    batched = MultiHead(cfg).eval()

    # Copy weights from looped to batched version:
    with torch.no_grad():
        q = torch.cat([h.query.weight for h in naive.multi_head], dim=0)
        k = torch.cat([h.key.weight for h in naive.multi_head], dim=0)
        v = torch.cat([h.value.weight for h in naive.multi_head], dim=0)
        batched.c_attn.weight.copy_(torch.cat([q, k, v], dim=0))
        batched.c_proj.load_state_dict(naive.proj.state_dict())

    # Ensure the implementations return the same output for identical input:
    x = torch.randn(2, 8, 16)
    assert torch.allclose(naive(x), batched(x), atol=1e-6)
