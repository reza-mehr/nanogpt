'''
This file contains the code to benchmark token generation: decode throughput, time to first token (TTFT), and memory.
The journey, measurement, and reporting code lives in src/nanogpt/benchmarking.py.

Suites:
- journey: cumulative inference optimizations (batched attention, torch LayerNorm, SDPA, KV cache, bf16 cast), plus an autocast bf16 branch compared with the KV cache row.
- gqa: key-value head sweep (MHA -> GQA -> MQA) on the fully optimized model, compared with MHA.

Decode throughput is 'batch / TPOT', with the time per output token TPOT = (t_N - t_1) / (N - 1) measured from two
generate calls with 1 and N new tokens, so the prefill (TTFT) does not leak into the decode numbers.

Quick debug run of the whole pipeline:
    uv run scripts/benchmark_generate.py --preset small_cfg --prompt-lens 16 32 --new-tokens 8 --trials 1
    uv run scripts/benchmark_generate.py --preset small_cfg --suite gqa --prompt-lens 16 32 --new-tokens 8 --trials 1

Full runs (plugged in, other apps closed):
    caffeinate -is uv run scripts/benchmark_generate.py --preset main_cfg
    caffeinate -is uv run scripts/benchmark_generate.py --preset main_cfg --suite gqa
    caffeinate -is uv run scripts/benchmark_generate.py --preset main_cfg --batch-size 16      # throughput instead of latency

Rebuild the tables from saved results without benchmarking:
    uv run scripts/benchmark_generate.py --preset main_cfg --from-csv

Outputs go to results/benchmark_generate[_gqa]_<preset>_<device>/ as report.md, results.csv, and env.json.

'''

#%% Imports:

import tyro

from nanogpt.benchmarking import GenerateBenchConfig, collect_env, report_generate, run_generate, run_or_load

#%% Main function:

def main(b: GenerateBenchConfig) -> None:
    '''Function to run (or reload) the selected generation benchmark suite and report the results.'''

    # Benchmark (or reload) the suite:
    tables, meta = run_or_load(b.run_dir,
                               run_fn=lambda: {'results': run_generate(b)},
                               meta_fn=lambda: collect_env(b.device) | b.settings(),
                               from_csv=b.from_csv)

    # Print the tables and save the report:
    report_generate(tables['results'], b, meta)

#%% Main call:

if __name__ == '__main__':
    main(tyro.cli(GenerateBenchConfig, description=__doc__))
