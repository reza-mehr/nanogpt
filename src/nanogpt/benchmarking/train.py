'''
This file contains the training benchmark: its configuration, metrics, and the measurement of training throughput
and peak memory for every variant of the training journey.

'''

#%% Imports:

import math
from dataclasses import dataclass, replace

import torch
from tqdm import tqdm

from nanogpt.config import PRESETS, TrainConfig
from nanogpt.benchmarking.common import BenchBase, TRAIN_JOURNEY, base_train_cfg, free_device_memory, journey_variants
from nanogpt.benchmarking.common import make_step, median_time

#%% Configuration:

@dataclass
class TrainBenchConfig(BenchBase):
    block_sizes: tuple[int, ...] = (128, 256, 512, 1024)    # sequence lengths, the table columns
    warmup: int = 5
    iters: int = 20                                         # training steps per timed trial
    trials: int = 3                                         # timed trials; the median is reported

    @property
    def kind(self) -> str:
        return 'benchmark_train'

    @property
    def batch(self) -> int:
        return self.batch_size or PRESETS[self.preset].batch_size

#%% Metrics:

# Entries: (row key, table title, number format)
TRAIN_METRICS = [('tps', 'Training throughput (tokens/sec)', ',.0f'),
                 ('mem', 'Peak memory (MiB)', ',.0f')]

#%% Training benchmark:

def measure_peak_mib(step_fn, device: str) -> float:
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
    '''Function to measure training throughput (tokens/sec) and peak memory (MiB) for one config.'''

    # Initialization:
    batch_size, T, device = cfg.batch_size, cfg.model.block_size, cfg.device
    free_device_memory(device)

    with make_step(cfg) as step:
        # Warm-up steps to ensure transient and one-time processes do not affect the benchmarking:
        for _ in range(b.warmup):
            step()

        # Measure the memory and the time per trial:
        memory_mib = measure_peak_mib(step, device)

        # Benchmark for requested iterations and trials:
        def run_iters():
            for _ in range(b.iters):
                step()

        seconds = median_time(run_iters, device, b.trials)

    return b.iters * batch_size * T / seconds, memory_mib


def run_train_journey(b: TrainBenchConfig) -> list[dict]:
    '''Function to benchmark every training journey variant at every block size and return flat rows.'''
    base, rows = base_train_cfg(b.preset, b.device, b.vocab_size, b.batch), []
    for variant, ref, model_over, train_over in tqdm(journey_variants(TRAIN_JOURNEY), desc='train journey'):
        for T in b.block_sizes:
            cfg = replace(base, model=replace(base.model, **model_over, block_size=T), **train_over)
            tps, mem = bench_train_cell(cfg, b)
            rows.append(dict(variant=variant, ref=ref, T=T, tps=tps, mem=mem))
    return rows
