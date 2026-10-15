'''
This file contains the code to fit the memory model mem(T) = c + a T + b T^2 to a saved benchmark and plot it.
The fitting and plotting code lives in src/nanogpt/benchmarking.py.

- c: length-independent memory (weights, gradients, optimizer state)
- a T: activations that grow linearly with the length
- b T^2: the T x T attention scores (expected to vanish with a fused attention kernel)

Fit the training benchmark (use 6-8 lengths for a well-determined fit):
    caffeinate -is uv run scripts/benchmark_train.py --preset main_cfg --block-sizes 64 128 256 384 512 768 1024
    uv run scripts/fit_memory.py --run-dir results/benchmark_train_main_cfg_mps

Outputs go to the same folder as memory_fit.csv, memory_fit.md, and memory_fit.png; the log is appended to its run.log.

'''

#%% Imports:

from dataclasses import dataclass
from pathlib import Path

import tyro

from nanogpt.benchmarking import report_memory_fit
from nanogpt.reporting import setup_logger

#%% Configuration:

@dataclass
class FitConfig:
    run_dir: Path                   # benchmark output folder containing results.csv
    group_key: str = 'variant'      # one fitted curve per value of this column
    x_key: str = 'T'                # length column
    y_key: str = 'mem'              # memory column

#%% Main call:

if __name__ == '__main__':
    f = tyro.cli(FitConfig, description=__doc__)
    setup_logger(f.run_dir)
    report_memory_fit(f.run_dir, f.group_key, f.x_key, f.y_key)
