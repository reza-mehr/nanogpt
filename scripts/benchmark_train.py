'''
This file contains the code to benchmark training throughput and peak memory as each optimization is added cumulatively (batched attention, torch LayerNorm, fused AdamW, SDPA, mixed bf16).
The journey, measurement, and reporting code lives in src/nanogpt/benchmarking.py.

Quick debug run of the whole pipeline:
    uv run scripts/benchmark_train.py --preset small_cfg --block-sizes 32 64 --iters 5 --trials 1

Full run (plugged in, other apps closed):
    caffeinate -is uv run scripts/benchmark_train.py --preset main_cfg

Rebuild the tables from saved results without benchmarking:
    uv run scripts/benchmark_train.py --preset main_cfg --from-csv

Outputs go to results/benchmark_train_<preset>_<device>/ as config.json, env.json, results.csv, report.md, and run.log.
An existing output folder is cleared only after confirmation (or with --overwrite).

'''

#%% Imports:

import tyro

from nanogpt.benchmarking import TRAIN_METRICS, TrainBenchConfig, run_or_load, run_train_journey
from nanogpt.reporting import report_pivot

#%% Main function:

def main(b: TrainBenchConfig) -> None:
    '''Function to run (or reload) the training benchmark journey and report the results.'''

    # Benchmark (or reload) the journey:
    tables, config, env = run_or_load(b, run_fn=lambda: {'results': run_train_journey(b)})

    # Log the tables and save the report:
    report_pivot(tables['results'], b.run_dir / 'report.md', row_key='variant', col_key='T',
                 metrics=TRAIN_METRICS, config=config, env=env, ref_key='ref')

#%% Main call:

if __name__ == '__main__':
    main(tyro.cli(TrainBenchConfig, description=__doc__))
