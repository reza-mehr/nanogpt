
#%% Imports:

import statistics
from dataclasses import dataclass
from tqdm import tqdm

import torch
import tyro

from nanogpt.config import ModelConfig
from nanogpt.model import GPT
from nanogpt.benchmarking import make_gen_fns, bench_decode

'''
This file contains the code to benchmark token generation throughput with and without key-value caching.
Note that the current benchmark measure the following combined times:
- Time to first token (TTFT): prefill time which grows with context.
- Time per output token (TPOT): decode loop time which should be almost perfectly flat.
I use a large number for new tokens (more loop iterations) to amortize the TTFT effect.

'''

#%% Configuration setup:

@dataclass
class BenchConfig:
    device: str = 'mps'
    block_size: int = 2048
    batch_size: int = 1
    context_inits: tuple[int, ...] = (128, 256, 512, 1024)
    new_tokens: int = 256               # large value to amortize the prefill time
    trials: int = 3

    def __post_init__(self):
            max_context_needed = max(self.context_inits) + self.new_tokens
            assert max_context_needed <= self.block_size, f'context overflow ({max_context_needed} > {self.block_size})'

#%% Main code:

@torch.inference_mode()
def main(b: BenchConfig):
    '''Function to benchmark token generation throughput with and without key-value caching.'''

    # Initialization:
    torch.manual_seed(0)
    cfg = ModelConfig(name='gpt', vocab_size=16, block_size=b.block_size, n_embed=384, head_num=6,
                      head_size=64, n_blocks=6, dropout=0.0)
    model = GPT(cfg).to(b.device).eval()
    gen_fns = make_gen_fns(model, b.batch_size, b.device)
    result = {fn_name: [] for fn_name in gen_fns}

    # Loop over initial context lengths:
    for context in tqdm(b.context_inits):
        # Generate the initial context:
        idx = torch.randint(0, cfg.vocab_size, (b.batch_size, context), device=b.device)

        # Loop over naive and cached generation functions:
        for fn_name, gen_fn in gen_fns.items():
            tok_per_secs = [bench_decode(b.device, gen_fn, idx, b.new_tokens) for _ in range(b.trials)]
            result[fn_name].append(statistics.median(tok_per_secs))

    # Print the results:
    name_w, col_w = 20, 10
    header = f"\n{'context':<{name_w}}" + ''.join(f'{c:>{col_w}}' for c in b.context_inits)
    print(header)
    print('-' * len(header))

    for name, values in result.items():
        label = f'{name} (tokens/sec)'
        print(f'{label:<{name_w}}' + ''.join(f'{v:>{col_w},.0f}' for v in values))
    if {'naive', 'cached'} <= result.keys():
        speedup = [c / n for c, n in zip(result['cached'], result['naive'])]
        print(f"{'speedup':<{name_w}}" + ''.join(f'{s:>{col_w - 1}.1f}x' for s in speedup))
    print()

#%% Main call:

if __name__=='__main__':
     main(tyro.cli(BenchConfig))
