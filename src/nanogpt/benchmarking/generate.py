'''
This file contains the generation benchmark: its configuration, metrics, and the measurement of decode throughput,
time to first token, and memory for the inference journey and the key-value head sweep.

'''

#%% Imports:

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from tqdm import tqdm

from nanogpt.benchmarking.common import BenchBase, GENERATE_BRANCHES, GENERATE_JOURNEY, base_model_cfg, build_inference_model
from nanogpt.benchmarking.common import free_device_memory, journey_variants, make_kv_cache, median_time
from nanogpt.reporting import report_pivot

#%% Configuration:

@dataclass
class GenerateBenchConfig(BenchBase):
    suite: Literal['journey', 'gqa'] = 'journey'            # inference journey, or key-value head sweep
    prompt_lens: tuple[int, ...] = (128, 256, 512, 1024)    # prompt (prefill) lengths, the table columns
    new_tokens: int = 64                                    # generated tokens per timed call (at least 2)
    precision: Literal['fp32', 'mixed_bf16', 'bf16'] = 'bf16'   # inference mode, 'gqa' suite only
    trials: int = 3                                         # timed trials; the median is reported

    def __post_init__(self):
        '''Function to validate input arguments.'''
        if self.new_tokens < 2:
            raise ValueError(f"'new_tokens' {self.new_tokens} must be at least 2 to separate decode from prefill.")

    @property
    def kind(self) -> str:
        return 'benchmark_generate_gqa' if self.suite == 'gqa' else 'benchmark_generate'

    @property
    def batch(self) -> int:
        return self.batch_size or 1                         # latency by default; larger values measure throughput

#%% Metrics:

# Entries: (row key, table title, number format)
GENERATE_METRICS = {
    'journey': [('tps', 'Decode throughput (tokens/sec)', ',.0f'),
                ('ttft', 'Time to first token (ms, lower is better)', ',.1f'),
                ('mem', 'Memory after generation (MiB)', ',.0f')],
    'gqa':     [('tps', 'Decode throughput (tokens/sec)', ',.0f'),
                ('ttft', 'Time to first token (ms, lower is better)', ',.1f'),
                ('mem', 'Memory after generation (MiB)', ',.0f'),
                ('cache_mib', 'KV cache in use (MiB)', ',.1f')],
}

#%% Generation benchmark:

def device_memory_mib(device: str) -> float:
    '''Function to read device memory in MiB: peak on CUDA, currently allocated on MPS, NaN on CPU.'''
    if device == 'cuda':
        return torch.cuda.max_memory_allocated() / 2**20
    if device == 'mps':
        return torch.mps.current_allocated_memory() / 2**20
    return math.nan


def bench_generate_cell(model: nn.Module, ctx, use_cache: bool, b: GenerateBenchConfig, T: int) -> dict:
    '''
    Function to measure decode throughput, time to first token, and memory for one prompt length.

    Decode throughput is 'batch / TPOT', with TPOT = (t_N - t_1) / (N - 1): the time per output token without the prefill, from two timed generate calls with 1 and N new tokens.

    Parameters
    ----------
    model: nn.Module
        Model in eval mode, already prepared for its inference precision.
    ctx: context manager
        Precision context from 'prepare_for_inference'.
    use_cache: bool
        If True, generate with a preallocated key-value cache.
    b: GenerateBenchConfig
        Benchmark settings.
    T: int
        Prompt length.

    Returns
    -------
    metrics: dict
        'tps' (decode tokens/sec), 'ttft' (ms), and 'mem' (MiB).

    '''
    # Initialization:
    device, B, N = b.device, b.batch, b.new_tokens
    idx = torch.randint(0, b.vocab_size, (B, T), device=device)
    cache = make_kv_cache(model.cfg, B) if use_cache else None     # allocated once, untimed

    def gen(n: int):
        with ctx:
            model.generate(idx, n, use_cache=use_cache, cache=cache)

    # Warm up with the identical calls, so every decode shape is compiled before timing:
    for _ in range(b.warmup):
        gen(1)
        gen(N)

    # Time the prefill-only call and the full call; their difference isolates decoding:
    t1 = median_time(lambda: gen(1), device, b.trials)
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    tN = median_time(lambda: gen(N), device, b.trials)
    tpot = (tN - t1) / (N - 1)

    return dict(tps=B / tpot, ttft=1e3 * t1, mem=device_memory_mib(device))


def run_generate_journey(b: GenerateBenchConfig) -> list[dict]:
    '''Function to benchmark every inference journey variant at every prompt length and return flat rows.'''
    base = base_model_cfg(b.preset, b.vocab_size, block_size=max(b.prompt_lens) + b.new_tokens)
    rows = []
    for variant, ref, model_over, run_over in tqdm(journey_variants(GENERATE_JOURNEY, GENERATE_BRANCHES),
                                                   desc='generate journey'):
        free_device_memory(b.device)
        model, ctx = build_inference_model(base, model_over, run_over['precision'], b.device)
        for T in b.prompt_lens:
            rows.append(dict(variant=variant, ref=ref, T=T,
                             **bench_generate_cell(model, ctx, run_over['use_cache'], b, T)))
        del model
    return rows


def run_generate_gqa(b: GenerateBenchConfig) -> list[dict]:
    '''Function to sweep the number of key-value heads on the fully optimized model and return flat rows.'''

    # Initialization:
    base = base_model_cfg(b.preset, b.vocab_size, block_size=max(b.prompt_lens) + b.new_tokens)
    best = dict(batched_attention=True, ln_torch=True, scaled_dot_prod=True)
    n_kvs = [d for d in range(base.head_num, 0, -1) if base.head_num % d == 0]     # MHA first, MQA last
    bytes_per = 4 if b.precision == 'fp32' else 2

    def label(n):
        return f'n_kv={n} ({"MHA" if n == base.head_num else "MQA" if n == 1 else "GQA"})'

    # Loop over the number of key-value heads, all compared with MHA:
    rows = []
    for n_kv in tqdm(n_kvs, desc='gqa sweep'):
        free_device_memory(b.device)
        model, ctx = build_inference_model(base, best | dict(kv_head_num=n_kv), b.precision, b.device)
        for T in b.prompt_lens:
            used = T + b.new_tokens - 1                     # cache positions filled by the end of generation
            cache_mib = 2 * base.n_blocks * b.batch * n_kv * used * base.head_size * bytes_per / 2**20
            rows.append(dict(variant=label(n_kv), ref=label(n_kvs[0]), T=T, cache_mib=cache_mib,
                             **bench_generate_cell(model, ctx, True, b, T)))
        del model
    return rows


def run_generate(b: GenerateBenchConfig) -> list[dict]:
    '''Function to run the selected generation suite and return flat rows.'''
    return run_generate_gqa(b) if b.suite == 'gqa' else run_generate_journey(b)


def report_generate(rows: list[dict], b: GenerateBenchConfig, config: dict, env: dict) -> None:
    '''Function to log the generation tables and save the Markdown report.'''
    is_journey = b.suite == 'journey'
    report_pivot(rows, b.run_dir / 'report.md', row_key='variant', col_key='T', metrics=GENERATE_METRICS[b.suite],
                 config=config, env=env, ref_key='ref',
                 total_label='total vs base' if is_journey else '',
                 total_row=GENERATE_JOURNEY[-1][0] if is_journey else None)
