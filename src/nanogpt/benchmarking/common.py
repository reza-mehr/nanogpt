'''
This file contains the benchmarking code shared by the benchmark and profiling modules: the base configuration,
the optimization journeys, the model and config builders, the measurement utilities, the training step, and the
run-or-reload routine.

'''

#%% Imports:

import gc
import logging
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal
from contextlib import contextmanager
from collections.abc import Generator

import torch

from nanogpt.config import DEVICE, PRESETS, ModelConfig, TrainConfig, config_record
from nanogpt.kv_cache import KVCache
from nanogpt.model import build_model
from nanogpt.utils import autocast_ctx, matmul_precision_ctx, prepare_for_inference, region, sync_device
from nanogpt.distributed import DDPInfo, wrap_model

from nanogpt.reporting import collect_env, load_json, save_json, load_rows_csv, save_rows_csv
from nanogpt.reporting import prepare_run_dir, setup_logger

#%% Base benchmark configuration:

@dataclass
class BenchBase:
    preset: Literal['main_cfg', 'small_cfg', 'stories_cfg'] = 'main_cfg'
    # model preset; 'small_cfg' for quick debug runs
    device: str = DEVICE
    vocab_size: int = 65                    # synthetic tokens, no dataset needed
    batch_size: int | None = None           # None -> the benchmark's default batch size
    warmup: int = 3                         # untimed repeats before measuring (compiles kernels)
    out_dir: Path = Path('results')         # parent folder of each benchmark's output folder
    from_csv: bool = False                  # if True, skip measuring and rebuild the report from the CSV files
    overwrite: bool = False                 # if True, clear an existing output folder without asking
    log_level: int = logging.INFO           # logging level, 20 for INFO, 10 for DEBUG

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


#%% Utility functions:

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

#%% Training step:

@contextmanager
def make_step(cfg: TrainConfig) -> Generator[Callable[..., float | None]]:
    '''
    Function to build a model, optimizer, and fixed batch and yield a function that conducts one training step.
    Use as 'with make_step(cfg) as step:', the matmul precision holds for the whole block.
    '''

    # Initialization:
    torch.manual_seed(0)
    model_cfg = cfg.model
    vocab_size, T = model_cfg.vocab_size, model_cfg.block_size
    device, batch_size, precision = cfg.device, cfg.batch_size, cfg.precision
    assert cfg.get_grad_accum_iters() == 1, "benchmarking unavailable with gradient accumulation"

    model = build_model(model_cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=cfg.adamw_fused)
    x = torch.randint(0, vocab_size, (batch_size, T), device=device)

    # Reset Dynamo cache of compiled graphs to ensure that each benchmarking cell compiles from scratch:
    torch._dynamo.reset()
    model_wrapped = wrap_model(cfg, model, DDPInfo())       # compile the model if requested

    def step(measure_after_forward: bool = False):
        '''One optimization step; optionally return memory right after forward (MPS).'''
        with autocast_ctx(device, precision):
            _, loss = model_wrapped(x, x)
        opt.zero_grad(set_to_none=True)
        mem = torch.mps.current_allocated_memory() / 2**20 if measure_after_forward else None
        with region('backward'):
            loss.backward()
        with region('optimizer'):
            opt.step()
        return mem

    with matmul_precision_ctx(cfg):         # matmul precision is set once for the life-cycle of 'step' function
        yield step

#%% Running and reloading:

def run_or_load(cfg: BenchBase, run_fn: Callable[[], dict[str, list[dict]]]) -> tuple[dict[str, list[dict]], dict, dict]:
    '''
    Function to run a benchmark or profile and save its records, or load them back with 'cfg.from_csv'.

    A fresh run prepares 'cfg.run_dir' (asking before clearing an existing one), starts run.log, and saves config.json and env.json before measuring, so they capture the start state; the tables follow as CSV files.

    Parameters
    ----------
    cfg: BenchBase
        Benchmark or profile configuration; sets the folder, the overwrite and reload flags, and the log level.
    run_fn: callable
        Runs the benchmark and returns tables keyed by CSV file stem, e.g. {'results': rows}.

    Returns
    -------
    tables: dict[str, list[dict]]
        Flat rows per table.
    config: dict
        Full configuration, as saved in config.json.
    env: dict
        Environment, as saved in env.json.

    Raises
    ------
    FileNotFoundError
        If reloading from a folder that does not exist.

    '''
    run_dir = cfg.run_dir

    # Reload the records of an earlier run:
    if cfg.from_csv:
        if not run_dir.is_dir():
            raise FileNotFoundError(f'Nothing to reload: {run_dir} does not exist.')
        setup_logger(run_dir, cfg.log_level)
        tables = {path.stem: load_rows_csv(path) for path in sorted(run_dir.glob('*.csv'))}
        return tables, load_json(run_dir / 'config.json'), load_json(run_dir / 'env.json')

    # Fresh run: folder, log, configuration, and environment first, then the measurements:
    prepare_run_dir(run_dir, cfg.overwrite)
    setup_logger(run_dir, cfg.log_level)
    config, env = config_record(cfg), collect_env(cfg.device)
    save_json(config, run_dir / 'config.json')
    save_json(env, run_dir / 'env.json')

    tables = run_fn()
    for name, rows in tables.items():
        save_rows_csv(rows, run_dir / f'{name}.csv')
    return tables, config, env
