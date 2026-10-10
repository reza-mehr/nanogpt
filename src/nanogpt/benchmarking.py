
#%% Imports:

import torch
import time
from pathlib import Path

from dataclasses import dataclass

from nanogpt.config import ModelConfig
from nanogpt.registry import build_model
import nanogpt.model  # noqa: F401
from nanogpt.kv_cache import KVCache
from nanogpt.utils import autocast_ctx, get_active_dtype

'''
This file contains the utility functions used in benchmarking.

'''

#%% Global variables:

@dataclass
class Col:
    key: str          # dict key in each row
    title: str        # header text
    fmt: str = ''     # e.g. ',.0f'; text columns leave it empty


#%% Utility functions:

def sync_device(device: str) -> None:
    '''Function to ensure that the tasks are measured and not the queue time.'''
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()

#%% Train utility functions:

def make_step(model_cfg: ModelConfig, device: str, batch_size: int, T: int, precision: str = 'fp32'):
    '''Function to build a model, optimizer, and fixed batch and return a function that conducts one training step.'''

    # Initialization:
    torch.manual_seed(0)
    model = build_model(model_cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    x = torch.randint(0, model_cfg.vocab_size, (batch_size, T), device=device)

    def step(measure_after_forward: bool = False):
        '''One optimization step; optionally return memory right after forward (MPS).'''
        with autocast_ctx(device, precision):
            _, loss = model(x, x)
        opt.zero_grad(set_to_none=True)
        mem = torch.mps.current_allocated_memory() / 2**20 if measure_after_forward else None
        loss.backward()
        opt.step()
        return mem

    return step

#%% Generate utility functions:

def make_gen_fns(model, batch_size, device, context):
    '''
    Function to construct the key-value cache outside the generate function.

    Parameters
    ----------
    model: GPT instance
        Model to generate the tokens.
    batch_size: int
        Batch size to use in the cache.
    context: AbstractContextManager
        To wrap the model.generate_cached() method for correct precision.

    Returns
    -------
    gen_fns: dict
        Keys are 'naive, cached' and values are model member methods.

    '''
    cfg = model.cfg
    dtype = get_active_dtype(model)
    cache = KVCache(cfg.n_blocks, batch_size, cfg.head_num, cfg.block_size,
                    cfg.head_size, device, dtype)           # allocated once, untimed

    def naive(idx, new_tokens):
        with context:
            return model.generate(idx, new_tokens)

    def cached(idx, new_tokens):
        with context:
            return model.generate_cached(idx, new_tokens, cache=cache)

    return {'naive': naive, 'cached': cached}


def bench_decode(device, fn, idx, new_tokens: int = 32) -> float:
    '''
    Function to benchmark token generation with and without key-value cache.

    Parameters
    ----------
    device: string
        Device to benchmark on.
    fn: callable
        Generation method.
    idx (B, T): tensor
        Initial context to start generation.
    new_tokens: int, optional
        Number of new tokens to generate. The default is 32.

    Returns
    -------
    throughput: float
        Number of tokens generated per second.

    '''
    # Initialization:
    fn(idx, new_tokens=4)                          # warm up

    # Benchmark the generation:
    sync_device(device)
    t0 = time.perf_counter()
    fn(idx, new_tokens=new_tokens)
    sync_device(device)

    return new_tokens / (time.perf_counter() - t0)     # tokens/sec at this context

#%% Reporting utility functions:

def _cell_text(c: Col, v) -> str:
    '''Format a cell value: numbers with the column's format, text as-is.'''
    return format(v, c.fmt) if c.fmt else str(v)


def print_table(rows: list[dict], cols: list[Col]) -> None:
    '''Function to print aligned console table, text left-aligned, numbers right-aligned.'''

    def cell(c, w, s):
        return f'{s:>{w}}' if c.fmt else f'{s:<{w}}'

    # Column widths: wide enough for the title and every value, plus a 2-space gap:
    widths = [max(len(c.title), *(len(_cell_text(c, r[c.key])) for r in rows)) + 2 for c in cols]

    header = ''.join(cell(c, w, c.title) for c, w in zip(cols, widths))
    print('\n' + header)
    print('-' * len(header))
    for r in rows:
        print(''.join(cell(c, w, _cell_text(c, r[c.key])) for c, w in zip(cols, widths)))
    print()


def write_markdown(rows: list[dict], cols: list[Col], path, header_note: str = '') -> None:
    '''Function to save the table as Markdown, for the README.'''
    # Column widths: wide enough for the title and every value (min 3 for the --- separator):
    widths = [max(3, len(c.title), *(len(_cell_text(c, r[c.key])) for r in rows)) for c in cols]

    def cell(c, w, s):
        return f'{s:>{w}}' if c.fmt else f'{s:<{w}}'

    def line(cells):
        return '| ' + ' | '.join(cells) + ' |'

    lines = [header_note, ''] if header_note else []
    lines.append(line(cell(c, w, c.title) for c, w in zip(cols, widths)))
    lines.append(line(('-' * (w - 1) + ':') if c.fmt else '-' * w for c, w in zip(cols, widths)))
    for r in rows:
        lines.append(line(cell(c, w, _cell_text(c, r[c.key])) for c, w in zip(cols, widths)))
    Path(path).write_text('\n'.join(lines) + '\n')
