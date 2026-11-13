'''
This file contains the code to plot training runs: loss and HellaSwag accuracy curves for one run, or a comparison of several runs.
The plotting and table code lives in src/nanogpt/reporting.py.

- Training run folder (contains results.csv): saves loss.png (train and val loss vs tokens seen) and hellaswag.png in that folder.
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
    plot_types: tuple[Literal['loss', 'hellaswag'], ...] = ('loss', 'hellaswag')
    # plot types to generate for each run folder
    group_by: str | None = None                 # dotted config key to group runs by, e.g. 'model.n_kv_head'
    split: Literal['train', 'val'] = 'val'      # loss to compare across runs
    per_run: bool = False                       # if True, also save requested plots for every compared run folder

#%% Main function:

def _mk_run_plots(run_dir: Path, plot_types) -> None:
    '''Function to plot all requested plot types for a run folder.'''
    for plot_type in plot_types:
        plot_path = plot_run(run_dir, plot_type)
        if plot_path: log.info(f'Saved {plot_path}')
        else: log.warning(f"No data to plot for '{plot_type}'.")


def main(p: PlotConfig) -> None:
    '''Function to plot a single run, or compare the runs under a folder.'''

    # Initialization:
    setup_logger(None)                          # stdout only: a comparison spans runs, so no single run.log
    runs = find_runs(p.path)

    # Save requested plot types for a single run folder:
    if runs == [p.path]:
        _mk_run_plots(p.path, p.plot_types)
        return

    # Compare the requested loss values for multiple run folders:
    compare_runs(p.path, p.group_by, p.split)

    # Save requested plot types for each individual run folder:
    if p.per_run:
        for run_dir in runs:
            _mk_run_plots(run_dir, p.plot_types)

#%% Main call:

if __name__ == '__main__':
    main(tyro.cli(PlotConfig, description=__doc__))
