'''
This file contains the benchmarking and profiling code: configurations, optimization journeys, measurement routines,
benchmark-specific reports, and the memory model fit. Shared tables, records, and plot styling live in reporting.py.

Every benchmark writes to 'results/<kind>_<preset>_<device>/':
- report.md: the tables, followed by the environment
- results.csv: flat raw rows, enough to rebuild the report with '--from-csv' (the profiler also writes ops.csv)
- env.json: code version, software, hardware, and run settings

'''

#%% Imports:

import gc
import math
import re
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule
import numpy as np
from tqdm import tqdm

from nanogpt.config import PRESETS, ModelConfig, TrainConfig
from nanogpt.kv_cache import KVCache
from nanogpt.model import build_model
from nanogpt.utils import autocast_ctx, enable_regions, prepare_for_inference, region, sync_device

from nanogpt.reporting import Col, pivot_sections, report_pivot, table_section, write_report
from nanogpt.reporting import load_json, save_json, load_rows_csv, save_rows_csv
from nanogpt.reporting import SERIES_COLORS, new_axes, plot_markers, save_figure

#%% Global variables:

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)

#%% Benchmark configurations:

@dataclass
class BenchBase:
    preset: Literal['main_cfg', 'small_cfg', 'stories_cfg'] = 'main_cfg'
    # model preset; 'small_cfg' for quick debug runs
    device: str = device
    vocab_size: int = 65                    # synthetic tokens, no dataset needed
    batch_size: int | None = None           # None -> the benchmark's default batch size
    warmup: int = 3                         # untimed repeats before measuring (compiles kernels)
    out_dir: Path = Path('results')         # parent folder of each benchmark's output folder
    from_csv: bool = False                  # if True, skip measuring and rebuild the report from the CSV files

    @property
    def kind(self) -> str:
        '''Function to return the benchmark name used in the output folder; set by each subclass.'''
        raise NotImplementedError

    @property
    def batch(self) -> int:
        '''Function to return the batch size in use: explicit 'batch_size', otherwise the benchmark default.'''
        raise NotImplementedError

    @property
    def run_dir(self) -> Path:
        '''Function to return the output folder for the report, CSV, and environment files.'''
        return self.out_dir / f'{self.kind}_{self.preset}_{self.device}'


@dataclass
class TrainBenchConfig(BenchBase):
    block_sizes: tuple[int, ...] = (128, 256, 512, 1024)     # sequence lengths, the table columns
    warmup: int = 5
    iters: int = 20                                         # training steps per timed trial
    trials: int = 3                                         # timed trials; the median is reported

    @property
    def kind(self) -> str:
        return 'benchmark_train'

    @property
    def batch(self) -> int:
        return self.batch_size or PRESETS[self.preset].batch_size

    def settings(self) -> dict:
        '''Function to return the run settings recorded next to the environment.'''
        return {'preset': self.preset, 'batch size': self.batch,
                'warmup / iters / trials': f'{self.warmup} / {self.iters} / {self.trials}'}


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

    def settings(self) -> dict:
        '''Function to return the run settings recorded next to the environment.'''
        return {'suite': self.suite, 'preset': self.preset, 'batch size': self.batch, 'new tokens': self.new_tokens,
                'precision': self.precision if self.suite == 'gqa' else 'per variant',
                'warmup / trials': f'{self.warmup} / {self.trials}'}


@dataclass
class ProfileConfig(BenchBase):
    mode: Literal['train', 'prefill', 'decode'] = 'train'   # training step, prompt prefill, or one decode step
    variants: tuple[str, ...] = ()                          # journey variants to profile; empty -> first and last
    block_sizes: tuple[int, ...] = (128, 1024)              # sequence (or prompt) lengths
    wait: int = 1                                           # profiler schedule: skipped steps
    warmup: int = 2                                         # profiler schedule: recorded but discarded steps
    active: int = 3                                         # profiler schedule: recorded steps, averaged
    top_ops: int = 10                                       # rows in each top-ops table
    export_trace: bool = True                               # if True, save a Chrome trace per case (open in Perfetto)

    @property
    def kind(self) -> str:
        return f'profile_{self.mode}'

    @property
    def batch(self) -> int:
        return self.batch_size or (PRESETS[self.preset].batch_size if self.mode == 'train' else 1)

    @property
    def journey(self) -> list[tuple[str, str, dict, dict]]:
        '''Function to return the variants of the journey that matches the profiling mode.'''
        if self.mode == 'train':
            return journey_variants(TRAIN_JOURNEY)
        return journey_variants(GENERATE_JOURNEY, GENERATE_BRANCHES)

    @property
    def selected(self) -> list[str]:
        '''Function to return the variant names to profile: explicit 'variants', otherwise the first and last steps.'''
        steps = TRAIN_JOURNEY if self.mode == 'train' else GENERATE_JOURNEY
        return list(self.variants) or [steps[0][0], steps[-1][0]]

    def settings(self) -> dict:
        '''Function to return the run settings recorded next to the environment.'''
        return {'mode': self.mode, 'preset': self.preset, 'batch size': self.batch,
                'variants': ', '.join(self.selected),
                'schedule (wait / warmup / active)': f'{self.wait} / {self.warmup} / {self.active}',
                'note': 'regions synchronize the device at their boundaries, so times are attributable but slightly inflated'}

#%% Optimization journeys:

# Cumulative training journey (each variant adds exactly one change to the previous one):
# Entries: (variant name, ModelConfig overrides, TrainConfig overrides)
TRAIN_JOURNEY = [
    ('baseline',      dict(batched_attention=False, ln_torch=False, scaled_dot_prod=False),
                      dict(adamw_fused=False, precision='fp32')),
    ('+batched attn', dict(batched_attention=True), {}),
    ('+torch LN',     dict(ln_torch=True), {}),
    ('+fused AdamW',  {}, dict(adamw_fused=True)),
    ('+SDPA',         dict(scaled_dot_prod=True), {}),
    ('+mixed bf16',   {}, dict(precision='mixed_bf16')),
]

# Cumulative inference journey:
# Entries: (variant name, ModelConfig overrides, inference overrides)
GENERATE_JOURNEY = [
    ('baseline',      dict(batched_attention=False, ln_torch=False, scaled_dot_prod=False),
                      dict(use_cache=False, precision='fp32')),
    ('+batched attn', dict(batched_attention=True), {}),
    ('+torch LN',     dict(ln_torch=True), {}),
    ('+SDPA',         dict(scaled_dot_prod=True), {}),
    ('+KV cache',     {}, dict(use_cache=True)),
    ('+bf16 (cast)',  {}, dict(precision='bf16')),
]

# Non-cumulative alternatives, compared with their parent variant:
# Entries: (variant name, parent variant, ModelConfig overrides, inference overrides)
GENERATE_BRANCHES = [
    ('+autocast bf16 (alt)', '+KV cache', {}, dict(precision='mixed_bf16')),
]


def iter_variants(steps: list[tuple]):
    '''
    Function to yield cumulative variants: each step's overrides are merged on top of all previous ones.

    Parameters
    ----------
    steps: list[tuple]
        Entries '(name, delta_1, ..., delta_k)', each delta a dict of overrides for one config section.

    Yields
    ------
    variant: tuple
        '(name, overrides_1, ..., overrides_k)' with the cumulative overrides up to this step.

    '''
    acc = None
    for name, *deltas in steps:
        acc = [a | d for a, d in zip(acc, deltas)] if acc else [dict(d) for d in deltas]
        yield (name, *acc)


def journey_variants(steps: list[tuple], branches: list[tuple] = ()) -> list[tuple[str, str, dict, dict]]:
    '''
    Function to list every variant of a journey with its reference variant for percent changes.

    Parameters
    ----------
    steps: list[tuple]
        Cumulative steps '(name, model delta, run delta)'; each step's reference is the previous step.
    branches: list[tuple], optional
        Alternatives '(name, parent, model delta, run delta)' built on, and compared with, their parent.
        The default is no branches.

    Returns
    -------
    variants: list[tuple[str, str, dict, dict]]
        '(name, reference name, model overrides, run overrides)', steps first and branches last.

    '''
    out, acc, prev = [], {}, ''
    for name, model_over, run_over in iter_variants(steps):
        out.append((name, prev, model_over, run_over))
        acc[name], prev = (model_over, run_over), name
    for name, parent, model_delta, run_delta in branches:
        model_over, run_over = acc[parent]
        out.append((name, parent, model_over | model_delta, run_over | run_delta))
    return out


def find_variant(variants: list[tuple[str, str, dict, dict]], name: str) -> tuple[dict, dict]:
    '''Function to return the (model overrides, run overrides) of a named variant, listing the options if unknown.'''
    found = {v[0]: (v[2], v[3]) for v in variants}
    if name not in found:
        raise ValueError(f"Unknown variant '{name}'. Available: {list(found)}")
    return found[name]

#%% Metrics and report columns:

# Entries: (row key, table title, number format)
TRAIN_METRICS = [('tps', 'Training throughput (tokens/sec)', ',.0f'),
                 ('mem', 'Peak memory (MB)', ',.0f')]

GENERATE_METRICS = {
    'journey': [('tps', 'Decode throughput (tokens/sec)', ',.0f'),
                ('ttft', 'Time to first token (ms, lower is better)', ',.1f'),
                ('mem', 'Memory after generation (MB)', ',.0f')],
    'gqa':     [('tps', 'Decode throughput (tokens/sec)', ',.0f'),
                ('ttft', 'Time to first token (ms, lower is better)', ',.1f'),
                ('mem', 'Memory after generation (MB)', ',.0f'),
                ('cache_mb', 'KV cache in use (MB)', ',.1f')],
}

PROFILE_SUMMARY_METRICS = [('ms', 'Step time under the profiler (ms, lower is better)', ',.2f'),
                           ('n_ops', 'aten op calls per step (lower is better)', ',.0f')]

PROFILE_BREAKDOWN_METRICS = [('pct', 'Share of step time (%)', '.1f'),
                             ('ms', 'Time per step (ms)', ',.2f')]

# Labelled code regions (see model.py and 'make_step'), in report order:
PROFILE_REGIONS = ('embed', 'layernorm', 'attn.qkv', 'attn.rope', 'attn.cache', 'attn.expand_kv', 'attn.core',
                   'attn.proj', 'mlp', 'lm_head', 'loss', 'backward', 'optimizer')

OPS_COLS = [Col('op', 'op'), Col('self_ms', 'self CPU ms/step', ',.3f'), Col('calls', 'calls/step', ',.1f')]

FIT_COLS = [Col('variant', 'variant'), Col('c', 'c (MB)', ',.0f'), Col('a', 'a (MB/token)', ',.3f'),
            Col('b', 'b (MB/token²)', '.2e'), Col('quad_pct', 'T² share at max T (%)', '.0f'),
            Col('r2', 'R²', '.4f'), Col('n', 'points', 'd')]

#%% Model and config builders:

def base_train_cfg(preset: str, device: str, vocab_size: int, batch_size: int) -> TrainConfig:
    '''Function to build a benchmark training config from a preset: synthetic vocabulary, no dropout, learned positions.'''
    preset_cfg = PRESETS[preset]
    model = replace(preset_cfg.model, vocab_size=vocab_size, dropout=0.0, pos_embed='learned')
    return replace(preset_cfg, model=model, device=device, batch_size=batch_size)


def base_model_cfg(preset: str, vocab_size: int, block_size: int) -> ModelConfig:
    '''Function to build a benchmark model config from a preset: synthetic vocabulary, no dropout, learned positions.'''
    return replace(PRESETS[preset].model, vocab_size=vocab_size, dropout=0.0, pos_embed='learned', block_size=block_size)


def build_inference_model(base: ModelConfig, model_over: dict, precision: str, device: str):
    '''
    Function to build a seeded model in eval mode and prepare it for an inference precision.

    Parameters
    ----------
    base: ModelConfig
        Base model config.
    model_over: dict
        Overrides applied to 'base'.
    precision: str
        Inference mode for 'prepare_for_inference': 'fp32', 'mixed_bf16', or 'bf16'.
    device: str
        Device to place the model on.

    Returns
    -------
    model: nn.Module
        Model ready for inference.
    ctx: context manager
        Precision context to wrap every forward pass in.

    '''
    torch.manual_seed(0)
    model = build_model(replace(base, **model_over)).to(device).eval()
    return prepare_for_inference(model, precision)


def make_kv_cache(cfg: ModelConfig, batch_size: int) -> KVCache:
    '''Function to allocate a key-value cache.'''
    return KVCache(cfg.n_blocks, batch_size, cfg.n_kv_head, cfg.block_size, cfg.head_size)

#%% Measurement utilities:

def free_device_memory(device: str) -> None:
    '''Function to release unreferenced objects and cached allocator blocks before the next measurement.'''
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    elif device == 'cuda':
        torch.cuda.empty_cache()


def median_time(fn: Callable[[], object], device: str, trials: int) -> float:
    '''Function to return the median wall time in seconds of 'fn' over 'trials' synchronized runs.'''
    times = []
    for _ in range(trials):
        sync_device(device)
        t0 = time.perf_counter()
        fn()
        sync_device(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def device_memory_mb(device: str) -> float:
    '''Function to read device memory in MB: peak on CUDA, currently allocated on MPS, NaN on CPU.'''
    if device == 'cuda':
        return torch.cuda.max_memory_allocated() / 2**20
    if device == 'mps':
        return torch.mps.current_allocated_memory() / 2**20
    return math.nan

#%% Training benchmark:

def make_step(cfg: TrainConfig):
    '''Function to build a model, optimizer, and fixed batch and return a function that conducts one training step.'''

    # Initialization:
    torch.manual_seed(0)
    model_cfg = cfg.model
    vocab_size, T = model_cfg.vocab_size, model_cfg.block_size
    device, batch_size, precision = cfg.device, cfg.batch_size, cfg.precision

    model = build_model(model_cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=cfg.adamw_fused)
    x = torch.randint(0, vocab_size, (batch_size, T), device=device)

    def step(measure_after_forward: bool = False):
        '''One optimization step; optionally return memory right after forward (MPS).'''
        with autocast_ctx(device, precision):
            _, loss = model(x, x)
        opt.zero_grad(set_to_none=True)
        mem = torch.mps.current_allocated_memory() / 2**20 if measure_after_forward else None
        with region('backward'):
            loss.backward()
        with region('optimizer'):
            opt.step()
        return mem

    return step


def measure_peak_mb(step_fn, device: str) -> float:
    '''Function to measure peak memory for one training step. CUDA: true peak. MPS: memory after forward.'''
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        step_fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2**20
    if device == 'mps':
        return step_fn(measure_after_forward=True)
    return math.nan


def bench_train_cell(cfg: TrainConfig, b: TrainBenchConfig) -> tuple[float, float]:
    '''Function to measure training throughput (tokens/sec) and peak memory (MB) for one config.'''

    # Initialization:
    batch_size, T, device = cfg.batch_size, cfg.model.block_size, cfg.device
    free_device_memory(device)
    step = make_step(cfg)

    # Warm-up steps to ensure transient and one-time processes do not affect the benchmarking:
    for _ in range(b.warmup):
        step()

    # Measure the memory and the time per trial:
    memory_mb = measure_peak_mb(step, device)

    def run_iters():
        for _ in range(b.iters):
            step()

    seconds = median_time(run_iters, device, b.trials)

    return b.iters * batch_size * T / seconds, memory_mb


def run_train_journey(b: TrainBenchConfig) -> list[dict]:
    '''Function to benchmark every training journey variant at every block size and return flat rows.'''
    base, rows = base_train_cfg(b.preset, b.device, b.vocab_size, b.batch), []
    for variant, ref, model_over, train_over in tqdm(journey_variants(TRAIN_JOURNEY), desc='train journey'):
        for T in b.block_sizes:
            cfg = replace(base, model=replace(base.model, **model_over, block_size=T), **train_over)
            tps, mem = bench_train_cell(cfg, b)
            rows.append(dict(variant=variant, ref=ref, T=T, tps=tps, mem=mem))
    return rows

#%% Generation benchmark:

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
        'tps' (decode tokens/sec), 'ttft' (ms), and 'mem' (MB).

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

    return dict(tps=B / tpot, ttft=1e3 * t1, mem=device_memory_mb(device))


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
            cache_mb = 2 * base.n_blocks * b.batch * n_kv * used * base.head_size * bytes_per / 2**20
            rows.append(dict(variant=label(n_kv), ref=label(n_kvs[0]), T=T, cache_mb=cache_mb,
                             **bench_generate_cell(model, ctx, True, b, T)))
        del model
    return rows


def run_generate(b: GenerateBenchConfig) -> list[dict]:
    '''Function to run the selected generation suite and return flat rows.'''
    return run_generate_gqa(b) if b.suite == 'gqa' else run_generate_journey(b)


def report_generate(rows: list[dict], b: GenerateBenchConfig, meta: dict) -> None:
    '''Function to print the generation tables and save the Markdown report.'''
    is_journey = b.suite == 'journey'
    report_pivot(rows, b.run_dir / 'report.md', row_key='variant', col_key='T', metrics=GENERATE_METRICS[b.suite],
                 meta=meta, ref_key='ref',
                 total_label='total vs base' if is_journey else '',
                 total_row=GENERATE_JOURNEY[-1][0] if is_journey else None)

#%% Profiling:

def make_profile_step(p: ProfileConfig, variant: str, T: int) -> Callable[[], None]:
    '''
    Function to build the model for one variant and return a function that runs one step of the profiled mode.

    Parameters
    ----------
    p: ProfileConfig
        Profiling settings; 'p.mode' selects a training step, a prompt prefill, or one decode step.
    variant: str
        Variant name from the journey matching the mode.
    T: int
        Sequence length (training) or prompt length (prefill and decode).

    Returns
    -------
    step: callable
        Runs one step; repeated calls do identical work.

    '''
    model_over, run_over = find_variant(p.journey, variant)

    # Training step:
    if p.mode == 'train':
        base = base_train_cfg(p.preset, p.device, p.vocab_size, p.batch)
        step = make_step(replace(base, model=replace(base.model, **model_over, block_size=T), **run_over))
        return lambda: step()

    # Inference model and inputs:
    base = base_model_cfg(p.preset, p.vocab_size, block_size=T + 1)
    model, ctx = build_inference_model(base, model_over, run_over['precision'], p.device)
    cache = make_kv_cache(model.cfg, p.batch) if run_over['use_cache'] else None
    idx = torch.randint(0, p.vocab_size, (p.batch, T + 1), device=p.device)
    prompt, token = idx[:, :T], idx[:, T:]                  # (B, T), (B, 1)

    def step():
        with torch.inference_mode(), ctx:
            if p.mode == 'prefill':
                if cache is not None: cache.reset()
                model(prompt, cache=cache)
            elif cache is not None:                         # decode one token against a cache filled to T
                cache.reset()
                cache.advance(T)
                model(token, cache=cache)
            else:                                           # decode without a cache recomputes the full context
                model(idx)

    return step


def profile_case(step_fn: Callable[[], None], p: ProfileConfig, trace_path: Path) -> tuple[dict, float, list[dict]]:
    '''
    Function to profile one case with labelled regions and summarize it per step.

    Parameters
    ----------
    step_fn: callable
        Runs one step of the profiled mode.
    p: ProfileConfig
        Profiling settings, including the wait/warmup/active schedule.
    trace_path: Path
        Chrome trace file, written if 'p.export_trace'.

    Returns
    -------
    times: dict
        CPU time per step in ms for every recorded event name, including 'step' and the labelled regions.
    n_ops: float
        aten op calls per step.
    top_ops: list[dict]
        The 'p.top_ops' aten ops with the largest self CPU time: 'op', 'self_ms', and 'calls' per step.

    '''
    # Initialization:
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if p.device == 'cuda' else [])
    sched = schedule(wait=p.wait, warmup=p.warmup, active=p.active, repeat=1)

    # Profile with the model's regions enabled (they synchronize the device, so times include device work):
    enable_regions(p.device)
    try:
        with profile(activities=activities, schedule=sched) as prof:
            for _ in range(p.wait + p.warmup + p.active):
                with record_function('step'):
                    step_fn()
                    sync_device(p.device)
                prof.step()                                 # inform the profiler that a step finished
    finally:
        enable_regions(None)

    if p.export_trace:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))

    # Per-step summaries over the active steps:
    events = prof.key_averages()
    times = {e.key: e.cpu_time_total / p.active / 1e3 for e in events}
    aten = [e for e in events if e.key.startswith('aten::')]
    n_ops = sum(e.count for e in aten) / p.active
    top = sorted(aten, key=lambda e: e.self_cpu_time_total, reverse=True)[:p.top_ops]
    top_ops = [dict(op=e.key, self_ms=e.self_cpu_time_total / p.active / 1e3, calls=e.count / p.active) for e in top]

    return times, n_ops, top_ops


def run_profile(p: ProfileConfig) -> dict[str, list[dict]]:
    '''
    Function to profile the selected variants at every length and return the result tables.

    Returns
    -------
    tables: dict[str, list[dict]]
        'results': one 'step' row per case (time and op count), plus one row per labelled region and 'other'
        (time per step and share of the step). 'ops': the top aten ops per case.

    '''
    rows, ops = [], []
    variants = p.selected
    for T in p.block_sizes:
        for n, variant in enumerate(variants):
            free_device_memory(p.device)
            case = f'{variant} T={T}'
            step_fn = make_profile_step(p, variant, T)
            times, n_ops, top_ops = profile_case(step_fn, p, p.run_dir / 'traces' / f'{_slug(case)}.json')
            del step_fn

            # Whole step, compared with the previous variant:
            step_ms = times.get('step', math.nan)
            ref = variants[n - 1] if n > 0 else ''
            rows.append(dict(variant=variant, ref=ref, T=T, case=case, region='step', ms=step_ms, pct=100.0, n_ops=n_ops))

            # Labelled regions and the unlabelled remainder (residual additions, zero_grad, ...):
            region_ms = {r: times.get(r, 0.0) for r in PROFILE_REGIONS}
            region_ms['other'] = max(step_ms - sum(region_ms.values()), 0.0)
            for r, ms in region_ms.items():
                rows.append(dict(variant=variant, ref='', T=T, case=case, region=r, ms=ms,
                                 pct=100 * ms / step_ms, n_ops=math.nan))
            ops += [dict(case=case, **o) for o in top_ops]

    # Drop regions that never ran in this mode (e.g. 'backward' when decoding):
    ran = {r['region'] for r in rows if r['region'] != 'step' and r['ms'] > 0}
    rows = [r for r in rows if r['region'] == 'step' or r['region'] in ran]

    return {'results': rows, 'ops': ops}


def report_profile(tables: dict[str, list[dict]], p: ProfileConfig, meta: dict) -> None:
    '''Function to print the profiling tables (summary, region breakdown, top ops) and save the Markdown report.'''
    rows, ops = tables['results'], tables.get('ops', [])
    steps = [r for r in rows if r['region'] == 'step']
    parts = [r for r in rows if r['region'] != 'step']
    multi = len({r['variant'] for r in steps}) > 1

    sections = pivot_sections(steps, 'variant', 'T', PROFILE_SUMMARY_METRICS, ref_key='ref',
                              total_label='total vs base' if multi else '')
    sections += pivot_sections(parts, 'region', 'case', PROFILE_BREAKDOWN_METRICS, ref_key='ref', total_label='')
    for case in dict.fromkeys(o['case'] for o in ops):
        sections.append(table_section([o for o in ops if o['case'] == case], OPS_COLS,
                                      f'Top ops by self CPU time ({case})'))

    write_report(p.run_dir / 'report.md', sections, meta)


def _slug(text: str) -> str:
    '''Function to turn a case label into a safe file name, e.g. '+SDPA T=128' -> 'SDPA_T_128'.'''
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')

#%% Memory model fitting:

def fit_quadratic(x: list[float], y: list[float]) -> tuple[float, float, float, float]:
    '''
    Function to fit y = c + a x + b x^2 by least squares.

    Parameters
    ----------
    x, y: list[float]
        Data points; at least three distinct x values.

    Returns
    -------
    c, a, b: float
        Constant, linear, and quadratic coefficients.
    r2: float
        Coefficient of determination of the fit.

    '''
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    b, a, c = np.polyfit(x, y, deg=2)                       # highest power first
    residual = y - (c + a * x + b * x**2)
    total = y - y.mean()
    r2 = 1 - (residual @ residual) / (total @ total) if total @ total > 0 else math.nan
    return float(c), float(a), float(b), float(r2)


def fit_memory(rows: list[dict], group_key: str = 'variant', x_key: str = 'T', y_key: str = 'mem') -> list[dict]:
    '''
    Function to fit the memory model mem(T) = c + a T + b T^2 per group, e.g. per journey variant.

    The constant 'c' is the length-independent memory (weights, gradients, optimizer state), the linear term 'a T'
    covers activations that grow with length, and the quadratic term 'b T^2' covers the T x T attention scores.

    Parameters
    ----------
    rows: list[dict]
        Flat benchmark rows, e.g. loaded with 'load_rows_csv'.
    group_key, x_key, y_key: str, optional
        Row keys for the group, the length, and the memory. The defaults are 'variant', 'T', and 'mem'.

    Returns
    -------
    fits: list[dict]
        One row per group with enough finite points: coefficients, R², the quadratic share of the fitted memory
        at the largest length, and the number of points. Groups with fewer than three points are skipped.

    '''
    fits = []
    for group in dict.fromkeys(r[group_key] for r in rows):    # unique, in first-seen order
        points = [(r[x_key], r[y_key]) for r in rows
                  if r[group_key] == group and not math.isnan(float(r[y_key]))]
        if len({x for x, _ in points}) < 3:
            print(f"Skipping '{group}': fewer than three lengths with measured memory.")
            continue
        x, y = zip(*sorted(points))
        c, a, b, r2 = fit_quadratic(x, y)
        x_max = x[-1]
        quad_pct = 100 * b * x_max**2 / (c + a * x_max + b * x_max**2)
        fits.append(dict(variant=group, c=c, a=a, b=b, quad_pct=quad_pct, r2=r2, n=len(x)))
    return fits


def plot_memory_fit(rows: list[dict], fits: list[dict], path: Path, group_key: str = 'variant',
                    x_key: str = 'T', y_key: str = 'mem', title: str = '') -> None:
    '''
    Function to plot measured memory (markers) and the fitted quadratic model (lines) per group and save the figure.

    Parameters
    ----------
    rows: list[dict]
        Flat benchmark rows with the measured points.
    fits: list[dict]
        Fitted coefficients from 'fit_memory'; groups without a fit are not plotted.
    path: Path
        Output image file, e.g. 'memory_fit.png'.
    group_key, x_key, y_key: str, optional
        Row keys for the group, the length, and the memory. The defaults are 'variant', 'T', and 'mem'.
    title: str, optional
        Figure title. The default is empty.

    '''
    # Initialization:
    fig, ax = new_axes(title or 'Memory vs length: markers measured, lines fitted c + aT + bT²',
                       f'{x_key} (tokens)', 'memory (MB)', n_series=len(fits))

    # One color per group, in fixed order: measured points and the fitted curve:
    for color, fit in zip(SERIES_COLORS, fits):
        points = sorted((r[x_key], r[y_key]) for r in rows if r[group_key] == fit['variant'])
        x, y = np.array(points, dtype=float).T
        x_dense = np.linspace(0, x.max(), 200)
        ax.plot(x_dense, fit['c'] + fit['a'] * x_dense + fit['b'] * x_dense**2, color=color, linewidth=2,
                label=f"{fit['variant']}  (R² {fit['r2']:.3f})")
        plot_markers(ax, x, y, color)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    save_figure(fig, ax, path)


def report_memory_fit(run_dir: Path, group_key: str = 'variant', x_key: str = 'T', y_key: str = 'mem') -> None:
    '''
    Function to fit the memory model to a benchmark's results and save the fit table and plot in the same folder.

    Reads 'results.csv' (and 'env.json' if present) from 'run_dir' and writes 'memory_fit.csv', 'memory_fit.md',
    and 'memory_fit.png' next to them.

    Parameters
    ----------
    run_dir: Path
        Benchmark output folder, e.g. 'results/benchmark_train_main_cfg_mps'.
    group_key, x_key, y_key: str, optional
        Row keys for the group, the length, and the memory. The defaults are 'variant', 'T', and 'mem'.

    '''
    # Load the results and fit every group:
    rows = load_rows_csv(run_dir / 'results.csv')
    fits = fit_memory(rows, group_key, x_key, y_key)
    if not fits:
        print(f'No group in {run_dir} has measured memory at three or more lengths; nothing to fit.')
        return

    # Save the coefficients, the report, and the plot:
    meta = load_json(run_dir / 'env.json') or None
    save_rows_csv(fits, run_dir / 'memory_fit.csv')
    title = 'Memory model fit: mem(T) = c + a T + b T²'
    write_report(run_dir / 'memory_fit.md', [table_section(fits, FIT_COLS, title), '![memory fit](memory_fit.png)'], meta)
    plot_memory_fit(rows, fits, run_dir / 'memory_fit.png', group_key, x_key, y_key)
    print(f'Saved memory_fit.csv, memory_fit.md, and memory_fit.png in {run_dir}')

#%% Running and reloading:

def run_or_load(run_dir: Path, run_fn: Callable[[], dict[str, list[dict]]], meta_fn: Callable[[], dict],
                from_csv: bool) -> tuple[dict[str, list[dict]], dict]:
    '''
    Function to run a benchmark and save its tables and metadata, or load both from a previous run.

    Parameters
    ----------
    run_dir: Path
        Benchmark output folder holding '<table>.csv' files and 'env.json'.
    run_fn: callable
        Runs the benchmark and returns tables keyed by CSV file stem, e.g. {'results': rows}.
    meta_fn: callable
        Returns the environment and settings metadata; called before 'run_fn' so it captures the start state.
    from_csv: bool
        If True, load instead of running.

    Returns
    -------
    tables: dict[str, list[dict]]
        Flat rows per table.
    meta: dict
        Environment and settings metadata.

    '''
    meta_path = run_dir / 'env.json'
    if from_csv:
        tables = {path.stem: load_rows_csv(path) for path in sorted(run_dir.glob('*.csv'))}
        return tables, load_json(meta_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_fn()
    tables = run_fn()
    for name, rows in tables.items():
        save_rows_csv(rows, run_dir / f'{name}.csv')
    save_json(meta, meta_path)
    return tables, meta
