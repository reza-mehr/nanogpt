
#%% Imports:

import statistics
from dataclasses import dataclass
from tqdm import tqdm

import torch
import tyro

from nanogpt.config import ModelConfig
from nanogpt.model import GPT
from nanogpt.benchmarking import make_gen_fns, bench_decode, Col, print_table
from nanogpt.utils import prepare_for_inference

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
    precisions: tuple[str, ...] = ('fp32', 'mixed_bf16', 'bf16')
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

    # Construct generate functions for dtypes:
    gen_fns = dict()
    for precision in b.precisions:
        m, ctx = prepare_for_inference(model, precision)
        for fn_name, gen_fn in make_gen_fns(m, b.batch_size, b.device, ctx).items():
             gen_fns[(precision, fn_name)] = gen_fn
    results = {key: [] for key in gen_fns}

    total = len(b.context_inits) * len(gen_fns) * b.trials
    with tqdm(total=total, desc='benchmark') as pbar:
        # Loop over initial context lengths:
        for context in b.context_inits:
            # Generate the initial context:
            idx = torch.randint(0, cfg.vocab_size, (b.batch_size, context), device=b.device)

            # Loop over generation functions:
            for (precision, fn_name), gen_fn in gen_fns.items():
                pbar.set_postfix(context=context, precision=precision, method=fn_name)
                runs = []
                for _ in range(b.trials):
                    runs.append(bench_decode(b.device, gen_fn, idx, b.new_tokens))
                    pbar.update(1)
                results[(precision, fn_name)].append(statistics.median(runs))

    # Print the table to terminal:
    rows = [{'label': f'{name} {precision} (tok/s)', **{f'c{c}': v for c, v in zip(b.context_inits, vals)}}
            for (precision, name), vals in results.items()]
    cols = [Col('label', 'initial context')] + [Col(f'c{c}', str(c), ',.0f') for c in b.context_inits]
    print_table(rows, cols)


#%% Main call:

if __name__=='__main__':
     main(tyro.cli(BenchConfig))
