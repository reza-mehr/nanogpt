'''
This file contains the code to plot training runs: loss curves for one run, or a comparison of several runs.
The plotting and table code lives in src/nanogpt/reporting.py.

- Training run folder (contains results.csv): saves loss.png (train and val loss vs tokens seen) in that folder.
- Parent folder (its subfolders are runs): saves comparison.png, comparison.md, and comparison.csv in that folder,
  one curve per run, or per group of runs with '--group-by' (mean curve, min-max band, mean ± std table).
  Benchmark and profile folders (names starting with 'benchmark' or 'profile') are skipped.

Single run:
    uv run scripts/plot_runs.py --path runs/train_main_cfg_mps

Every run under a folder:
    uv run scripts/plot_runs.py --path runs/gqa

Ablation with seeds, grouped by a dotted config key:
    uv run scripts/plot_runs.py --path runs/gqa --group-by model.n_kv_head

'''

#%% Imports:

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from nanogpt.reporting import compare_runs, find_runs, log, plot_run, setup_logger

#%% Configuration:

@dataclass
class PlotConfig:
    path: Path                                  # a run folder, or a folder whose subfolders are runs
    group_by: str | None = None                 # dotted config key to group runs by, e.g. 'model.n_kv_head'
    split: Literal['train', 'val'] = 'val'      # loss to compare across runs
    per_run: bool = False                       # if True, also save loss.png in every compared run folder

#%% Main function:

def main(p: PlotConfig) -> None:
    '''Function to plot a single run, or compare the runs under a folder.'''
    setup_logger(None)                          # stdout only: a comparison spans runs, so no single run.log
    runs = find_runs(p.path)
    if runs == [p.path]:
        log.info(f'Saved {plot_run(p.path)}')
        return

    compare_runs(p.path, p.group_by, p.split)
    if p.per_run:
        for run_dir in runs:
            plot_run(run_dir)

#%% Main call:

if __name__ == '__main__':
    main(tyro.cli(PlotConfig, description=__doc__))
