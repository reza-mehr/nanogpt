'''
This file contains the profiler: its configuration, report columns, and the profiling of a training step, a prompt
prefill, or one decode step with labelled regions, summarized per step.

'''

#%% Imports:

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from nanogpt.config import PRESETS
from nanogpt.utils import enable_regions, sync_device
from nanogpt.benchmarking.common import BenchBase, GENERATE_BRANCHES, GENERATE_JOURNEY, TRAIN_JOURNEY, journey_variants
from nanogpt.benchmarking.common import base_model_cfg, base_train_cfg, build_inference_model, free_device_memory
from nanogpt.benchmarking.common import make_kv_cache, make_step
from nanogpt.reporting import Col, pivot_sections, table_section, write_report

#%% Configuration:

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

#%% Metrics and report columns:

PROFILE_SUMMARY_METRICS = [('ms', 'Step time under the profiler (ms, lower is better)', ',.2f'),
                           ('n_ops', 'aten op calls per step (lower is better)', ',.0f')]

PROFILE_BREAKDOWN_METRICS = [('pct', 'Share of step time (%)', '.1f'),
                             ('ms', 'Time per step (ms)', ',.2f')]

# Labelled code regions (see model.py and 'common.make_step'), in report order:
PROFILE_REGIONS = ('embed', 'layernorm', 'attn.qkv', 'attn.rope', 'attn.cache', 'attn.expand_kv', 'attn.core',
                   'attn.proj', 'mlp', 'lm_head', 'loss', 'backward', 'optimizer')

OPS_COLS = [Col('op', 'op'), Col('self_ms', 'self CPU ms/step', ',.3f'), Col('calls', 'calls/step', ',.1f')]

# On CUDA the ops are ranked by GPU time; CPU time stays alongside (a large CPU share means launch-bound):
OPS_COLS_CUDA = [Col('op', 'op'), Col('self_device_ms', 'self GPU ms/step', ',.3f'),
                 Col('self_ms', 'self CPU ms/step', ',.3f'), Col('calls', 'calls/step', ',.1f')]

#%% Profiling:

def find_variant(variants: list[tuple[str, str, dict, dict]], name: str) -> tuple[dict, dict]:
    '''Function to return the (model overrides, run overrides) of a named variant, listing the options if unknown.'''
    found = {v[0]: (v[2], v[3]) for v in variants}
    if name not in found:
        raise ValueError(f"Unknown variant '{name}'. Available: {list(found)}")
    return found[name]


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
        The 'p.top_ops' aten ops with the largest self time: 'op', 'self_ms' (CPU), and 'calls' per step. On CUDA
        they are ranked by GPU time and also carry 'self_device_ms'; elsewhere they are ranked by CPU time.

    Raises
    ------
    RuntimeError
        If the profiler reports the same CPU event name more than once.

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

    # Per-step summaries over the active steps, from CPU-side events only:
    events = [e for e in prof.key_averages() if e.device_type == DeviceType.CPU]
    times = _cpu_ms_per_step(events, p.active)
    aten = [e for e in events if e.key.startswith('aten::')]
    n_ops = sum(e.count for e in aten) / p.active

    # Top ops: GPU time is the real cost on CUDA (kernels run asynchronously); CPU time is the cost elsewhere:
    on_cuda = p.device == 'cuda'
    rank_by = (lambda e: e.self_device_time_total) if on_cuda else (lambda e: e.self_cpu_time_total)
    top_ops = []
    for e in sorted(aten, key=rank_by, reverse=True)[:p.top_ops]:
        row = dict(op=e.key, self_ms=e.self_cpu_time_total / p.active / 1e3, calls=e.count / p.active)
        if on_cuda:
            row['self_device_ms'] = e.self_device_time_total / p.active / 1e3
        top_ops.append(row)

    return times, n_ops, top_ops


def _cpu_ms_per_step(events: list, active: int) -> dict[str, float]:
    '''
    Function to map each CPU event name to its CPU time per step in ms, never overwriting a duplicate silently.

    With CUDA activity, the profiler records every 'record_function' label twice: a CPU event and a GPU-side
    annotation that holds no CPU time. Callers pass CPU events only, so a name that still appears more than once
    means the profiler output is not understood, and the times cannot be trusted.

    Parameters
    ----------
    events: list
        CPU-side entries of 'prof.key_averages()'.
    active: int
        Number of recorded steps the totals cover.

    Returns
    -------
    times: dict[str, float]
        CPU time per step in ms, keyed by event name.

    Raises
    ------
    RuntimeError
        If an event name appears more than once.

    '''
    times = {}
    for e in events:
        if e.key in times:
            raise RuntimeError(f"Profiler reported the CPU event '{e.key}' more than once; refusing to pick one.")
        times[e.key] = e.cpu_time_total / active / 1e3
    return times


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
            if not step_ms > 0:                             # also catches NaN
                raise RuntimeError(f"Profiler recorded no CPU time for 'step' in '{case}' (got {step_ms}); "
                                   f'check the schedule and that CPU events are being read.')
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


def report_profile(tables: dict[str, list[dict]], p: ProfileConfig, config: dict, env: dict) -> None:
    '''Function to log the profiling tables (summary, region breakdown, top ops) and save the Markdown report.'''
    rows, ops = tables['results'], tables.get('ops', [])
    steps = [r for r in rows if r['region'] == 'step']
    parts = [r for r in rows if r['region'] != 'step']
    multi = len({r['variant'] for r in steps}) > 1

    sections = ['Regions synchronize the device at their boundaries, so times are attributable but slightly '
                'inflated: use the shares and op counts here, and the speeds from the benchmarks.']
    sections += pivot_sections(steps, 'variant', 'T', PROFILE_SUMMARY_METRICS, ref_key='ref',
                               total_label='total vs base' if multi else '')
    sections += pivot_sections(parts, 'region', 'case', PROFILE_BREAKDOWN_METRICS, ref_key='ref', total_label='')
    on_cuda = p.device == 'cuda'
    ops_cols, ranked_by = (OPS_COLS_CUDA, 'GPU') if on_cuda else (OPS_COLS, 'CPU')
    for case in dict.fromkeys(o['case'] for o in ops):
        sections.append(table_section([o for o in ops if o['case'] == case], ops_cols,
                                      f'Top ops by self {ranked_by} time ({case})'))

    write_report(p.run_dir / 'report.md', sections, config, env)


def _slug(text: str) -> str:
    '''Function to turn a case label into a safe file name, e.g. '+SDPA T=128' -> 'SDPA_T_128'.'''
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')
