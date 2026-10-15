'''
This file contains the smoke tests: end-to-end runs of training and every script on tiny CPU settings.
They check that each entry point runs and writes its expected files, not the numbers themselves.

All outputs go to pytest's temporary folders ('tmp_path'), outside the repository.

Fast loop while coding (skip these):
    uv run pytest -m "not smoke"

Before committing (everything):
    uv run pytest

'''

#%% Imports:

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nanogpt.config import PRESETS
from nanogpt.data import TokenSplit
from nanogpt.reporting import load_rows_csv, save_rows_csv
from nanogpt.tokenizers import CharTokenizer
from nanogpt.trainer import train

#%% Global variables:

pytestmark = pytest.mark.smoke                  # every test in this file is a smoke test
ROOT = Path(__file__).resolve().parents[1]      # repository root, where scripts/ lives
TEXT = 'abcdefghijklmnopqrstuvwxyz \n'          # character vocabulary for the training test
RUN_RECORDS = {'config.json', 'env.json', 'results.csv', 'report.md', 'run.log'}   # written by every run on every path

#%% Utility functions:

def _run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    '''Function to run a script with the test's Python, fail with its error output if it crashes, and return the result.'''
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / name), *args],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f'{name} {" ".join(args)} failed:\n{result.stderr[-3000:]}'
    return result


def _files(folder: Path) -> set[str]:
    '''Function to return the names of the files in a folder.'''
    return {p.name for p in folder.iterdir() if p.is_file()}


def _assert_files(folder: Path, expected: set[str]) -> None:
    '''Function to assert that 'folder' contains every expected file, naming any that are missing.'''
    missing = expected - _files(folder)
    assert not missing, f'missing in {folder}: {sorted(missing)}'


def _only_subfolder(folder: Path) -> Path:
    '''Function to return the single subfolder a script created inside 'folder'.'''
    subfolders = [p for p in folder.iterdir() if p.is_dir()]
    assert len(subfolders) == 1, f'expected one output folder in {folder}, found {subfolders}'
    return subfolders[0]


def _token_splits(tmp_path: Path, vocab_size: int, n_tokens: int = 4000) -> tuple[TokenSplit, TokenSplit]:
    '''Function to write random token files in the dataset format and return train and validation splits.'''
    rng = np.random.default_rng(0)
    splits = []
    for name in ('train', 'val'):
        path = tmp_path / f'{name}.bin'
        rng.integers(0, vocab_size, n_tokens, dtype=np.uint16).astype('<u2').tofile(path)
        splits.append(TokenSplit(str(path)))
    return splits[0], splits[1]

#%% Training:

@pytest.fixture(scope='module')
def trained_run(tmp_path_factory):
    '''Function to train one tiny run, shared by the tests in this module, and return its config and data.'''

    # Initialization:
    tmp_path = tmp_path_factory.mktemp('train')
    tok = CharTokenizer(TEXT)
    train_data, val_data = _token_splits(tmp_path, tok.vocab_size)
    small = PRESETS['small_cfg']
    cfg = replace(small, model=replace(small.model, vocab_size=tok.vocab_size, dropout=0.0),
                  device='cpu', max_iters=4, eval_interval=2, ckpt_interval=2, eval_iters=2, lr_warmup_iters=1,
                  out_dir=str(tmp_path / 'runs'), run_name='smoke', sample_prompts=('', 'ab'), sample_tokens=8)

    # Fresh run:
    train(cfg, train_data, val_data, tok.to_meta())

    return cfg, tok, train_data, val_data


def test_train_resume_and_records(trained_run):
    '''Function to test that training writes its run records and that resuming continues without duplicate metrics.'''

    # Records of the fresh run:
    cfg, tok, train_data, val_data = trained_run
    run_dir = cfg.run_dir
    _assert_files(run_dir, RUN_RECORDS | {'latest.pt', 'best.pt', 'model.pt'})
    assert [r['step'] for r in load_rows_csv(run_dir / 'results.csv')] == [2, 4]
    report = (run_dir / 'report.md').read_text()
    assert report.index('### Summary') < report.index('### Samples') < report.index('### Configuration')
    assert report.count('#### Prompt') == 2

    # Resume to a later step:
    train(replace(cfg, max_iters=6, resume=True), train_data, val_data, tok.to_meta())
    assert [r['step'] for r in load_rows_csv(run_dir / 'results.csv')] == [2, 4, 6]

    # Plot the single run, then compare the runs under the parent folder:
    _run_script('plot_runs.py', '--path', str(run_dir))
    _assert_files(run_dir, {'loss.png'})
    _run_script('plot_runs.py', '--path', str(run_dir.parent))
    _assert_files(run_dir.parent, {'comparison.png', 'comparison.md', 'comparison.csv'})

#%% Generation:

@pytest.mark.parametrize('cache_flag', ['--use-cache', '--no-use-cache'])
def test_generate(trained_run, cache_flag):
    '''Function to test that generation loads the inference checkpoint and prints text, with and without the cache.'''
    cfg = trained_run[0]
    result = _run_script('generate.py', '--out-dir', str(cfg.run_dir.parent), '--run-name', cfg.run_dir.name, '--ckpt', 'model.pt', '--prompt', 'ab', '--new-tokens', '4', '--device', 'cpu', cache_flag)
    assert result.stdout.strip(), 'generate.py printed nothing'

#%% Benchmarks:

def test_benchmark_train(tmp_path):
    '''Function to test that the training benchmark runs end to end and that its report rebuilds from the CSV.'''
    args = ['--preset', 'small_cfg', '--device', 'cpu', '--block-sizes', '8', '16',
            '--iters', '1', '--trials', '1', '--warmup', '1', '--out-dir', str(tmp_path)]
    _run_script('benchmark_train.py', *args)
    run_dir = _only_subfolder(tmp_path)
    _assert_files(run_dir, RUN_RECORDS)
    assert len(load_rows_csv(run_dir / 'results.csv')) == 6 * 2                    # variants x block sizes

    _run_script('benchmark_train.py', *args, '--from-csv')


@pytest.mark.parametrize('suite', ['journey', 'gqa'])
def test_benchmark_generate(tmp_path, suite):
    '''Function to test that each generation benchmark suite runs end to end and that its report rebuilds from the CSV.'''
    args = ['--preset', 'small_cfg', '--device', 'cpu', '--suite', suite, '--prompt-lens', '8',
            '--new-tokens', '4', '--trials', '1', '--warmup', '1', '--out-dir', str(tmp_path)]
    _run_script('benchmark_generate.py', *args)
    run_dir = _only_subfolder(tmp_path)
    _assert_files(run_dir, RUN_RECORDS)

    _run_script('benchmark_generate.py', *args, '--from-csv')


@pytest.mark.parametrize('mode', ['train', 'prefill', 'decode'])
def test_profile_model(tmp_path, mode):
    '''Function to test that profiling runs end to end in every mode and writes its report and op tables.'''

    _run_script('profile_model.py', '--preset', 'small_cfg', '--device', 'cpu', '--mode', mode,
                '--block-sizes', '8', '--wait', '0', '--warmup', '1', '--active', '1', '--no-export-trace',
                '--out-dir', str(tmp_path))
    run_dir = _only_subfolder(tmp_path)
    _assert_files(run_dir, RUN_RECORDS | {'ops.csv'})

    steps = [r for r in load_rows_csv(run_dir / 'results.csv') if r['region'] == 'step']
    assert steps and all(r['ms'] > 0 for r in steps), 'profiler recorded no step time'

#%% Analysis:

def test_fit_memory(tmp_path):
    '''Function to test the memory fit on synthetic results (CPU benchmarks have no memory to fit).'''
    rows = [dict(variant=v, ref='', T=T, tps=1.0, mem=100 + a * T + 1e-3 * T**2)
            for v, a in (('baseline', 2.0), ('+torch LN', 1.5)) for T in (64, 128, 256, 512)]
    save_rows_csv(rows, tmp_path / 'results.csv')
    _run_script('fit_memory.py', '--run-dir', str(tmp_path))
    _assert_files(tmp_path, {'memory_fit.csv', 'memory_fit.md', 'memory_fit.png'})
