
#%% Imports:

import statistics
import time
from dataclasses import dataclass

import torch
import tyro

from nanogpt.config import ModelConfig
from nanogpt.registry import build_model
import nanogpt.model  # noqa: F401
from nanogpt.benchmarking import make_step, sync_device

#%% Benchmarking configurations:

@dataclass
class BenchConfig:
    device: str = 'mps'
    batch_size: int = 16
    seq_lens: tuple[int, ...] = (128, 256, 512, 1024)
    scaled_dot_prod: bool = True
    warmup: int = 5
    iters: int = 20
    trials: int = 3


#%% Functions:

def measure_peak_mb(step_fn, device: str) -> float:
    '''Function to measure peak memory for one training step. CUDA: true peak. MPS: memory after forward.'''
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        step_fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2**20
    if device == 'mps':
        return step_fn(measure_after_forward=True)
    return float('nan')


def bench_one(model_cfg: ModelConfig, b: BenchConfig, T: int) -> tuple[float, float]:
    '''Function to run a single benchmarking experiment.'''

    # Construct the model, optimizer, and the corresponding step function:
    step = make_step(model_cfg, b.device, b.batch_size, T)

    # Warm-up steps to ensure transient and one-time processes do not affect the benchmarking:
    for _ in range(b.warmup):
        step()
    if b.device == 'cuda': torch.cuda.reset_peak_memory_stats()

    # Measure the memory used:
    memory_mb = measure_peak_mb(step, b.device)

    # Main benchmarking loop:
    times = []
    for _ in range(b.trials):       # loop over trials
        sync_device(b.device)
        t0 = time.perf_counter()
        for _ in range(b.iters):    # loop over iterations
            step()
        sync_device(b.device)
        times.append(time.perf_counter() - t0)

    # Compute the tokens processed per second:
    tok_per_sec = b.iters * b.batch_size * T / statistics.median(times)

    return tok_per_sec, memory_mb


def main(b: BenchConfig) -> None:
    '''Main function to run and report benchmarking results.'''

    header = f"\n{'implementation':<14}{'T':>6}{'tokens/sec':>14}{'memory (MB)':>14}"
    print(header)
    print('-' * len(header))

    # Loop over context sizes:
    for T in b.seq_lens:
        for sdpa_flag in [False, True]:      # manual or scaled dot-product attention
            cfg = ModelConfig(name='gpt', vocab_size=65, block_size=T, n_embed=384,
                              head_num=6, head_size=64, n_blocks=6, dropout=0.0, scaled_dot_prod=sdpa_flag)
            tps, mem = bench_one(cfg, b, T)

            impl = 'sdpa' if sdpa_flag else 'manual'
            print(f"{impl:<14}{T:>6}{tps:>14,.0f}{mem:>14,.0f}")
        print()


#%% Main call:

if __name__ == '__main__':
    main(tyro.cli(BenchConfig))
