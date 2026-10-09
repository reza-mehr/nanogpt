
#%% Imports:

import torch
import time

from nanogpt.config import ModelConfig
from nanogpt.registry import build_model
import nanogpt.model  # noqa: F401
from nanogpt.kv_cache import KVCache

'''
This file contains the utility functions used in benchmarking.

'''

#%% Utility functions:

def sync_device(device: str) -> None:
    '''Function to ensure that the tasks are measured and not the queue time.'''
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()

#%% Train utility functions:

def make_step(model_cfg: ModelConfig, device: str, batch_size: int, T: int):
    '''Function to build a model, optimizer, and fixed batch and return a function that conducts one training step.'''

    # Initialization:
    torch.manual_seed(0)
    model = build_model(model_cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    x = torch.randint(0, model_cfg.vocab_size, (batch_size, T), device=device)

    def step(measure_after_forward: bool = False):
        '''One optimization step; optionally return memory right after forward (MPS).'''
        _, loss = model(x, x)
        opt.zero_grad(set_to_none=True)
        mem = torch.mps.current_allocated_memory() / 2**20 if measure_after_forward else None
        loss.backward()
        opt.step()
        return mem

    return step

#%% Generate utility functions:

def make_gen_fns(model, batch_size, device):
    '''
    Function to construct the key-value cache outside the generate function.

    Parameters
    ----------
    model: GPT instance
        Model to generate the tokens.
    batch_size: int
        Batch size to use in the cache.
    device: string
        Device to benchmark.

    Returns
    -------
    gen_fns: dict
        Keys are 'naive, cached' and values are model member methods.

    '''
    cfg = model.cfg
    cache = KVCache(cfg.n_blocks, batch_size, cfg.head_num, cfg.block_size,
                    cfg.head_size, device)                      # allocated once, untimed

    def naive(idx, new_tokens):
        return model.generate(idx, new_tokens)

    def cached(idx, new_tokens):
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
