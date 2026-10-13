'''
This file contains the code to profile where the time goes in a training step, a prompt prefill, or one decode step.
The profiling, summarizing, and reporting code lives in src/nanogpt/benchmarking.py.

The report has three parts:
- Summary: step time and aten op calls per step per variant and length; op counts are the clearest evidence of per-op overhead (fewer, larger ops after batched attention and torch LayerNorm).
- Breakdown: time per step and share of the step for every labelled region in model.py ('attn.core', 'mlp', 'lm_head', ...) plus 'backward' and 'optimizer' in training; unlabelled time shows up as 'other'.
- Top ops: the aten ops with the most self CPU time per case.

Notes:
- Regions synchronize the device on entry and exit, so each region's time includes its device work. This makes times attributable on MPS and CUDA but adds some overhead: use shares and counts here, speeds from the benchmarks.
- Top-op self times are CPU-side; on MPS and CUDA they reflect dispatch cost, not kernel time (see the traces).

Quick debug run of the whole pipeline:
    uv run scripts/profile_model.py --preset small_cfg --block-sizes 32 --active 1

Training step, baseline versus fully optimized (the default variants are the first and last of the journey):
    caffeinate -is uv run scripts/profile_model.py --preset main_cfg --mode train

Prefill and decode, with chosen variants from the inference journey:
    caffeinate -is uv run scripts/profile_model.py --preset main_cfg --mode prefill
    caffeinate -is uv run scripts/profile_model.py --preset main_cfg --mode decode --variants '+SDPA' '+KV cache' '+bf16 (cast)'

Rebuild the tables from saved results without profiling:
    uv run scripts/profile_model.py --preset main_cfg --mode train --from-csv

Outputs go to results/profile_<mode>_<preset>_<device>/ as report.md, results.csv, ops.csv, env.json,
and traces/<case>.json (open at https://ui.perfetto.dev).

'''

#%% Imports:

import tyro

from nanogpt.benchmarking import ProfileConfig, collect_env, report_profile, run_or_load, run_profile

#%% Main function:

def main(p: ProfileConfig) -> None:
    '''Function to profile (or reload) the selected variants and report the results.'''

    # Profile (or reload) the cases:
    tables, meta = run_or_load(p.run_dir,
                               run_fn=lambda: run_profile(p),
                               meta_fn=lambda: collect_env(p.device) | p.settings(),
                               from_csv=p.from_csv)

    # Print the tables and save the report:
    report_profile(tables, p, meta)

#%% Main call:

if __name__ == '__main__':
    main(tyro.cli(ProfileConfig, description=__doc__))
