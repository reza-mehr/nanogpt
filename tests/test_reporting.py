
'''
This file contains the code to test the shared run folder utilities: preparing a folder and finding training runs.

'''

#%% Imports:

import pytest

from nanogpt import reporting
from nanogpt.reporting import find_runs, prepare_run_dir

#%% Run folder preparation:

def test_prepare_creates_missing_folder(tmp_path):
    '''Function to test that a missing run folder is created without asking.'''
    run_dir = tmp_path / 'runs' / 'new'
    prepare_run_dir(run_dir)
    assert run_dir.is_dir()


@pytest.mark.parametrize('overwrite, answer', [(True, False), (False, True)], ids=['overwrite', 'confirmed'])
def test_prepare_clears_when_allowed(tmp_path, monkeypatch, overwrite, answer):
    '''Function to test that an existing run folder is cleared by '--overwrite' or by a confirmed prompt.'''
    monkeypatch.setattr(reporting, '_confirm', lambda question: answer)
    (tmp_path / 'old.txt').write_text('old')
    prepare_run_dir(tmp_path, overwrite=overwrite)
    assert tmp_path.is_dir() and not any(tmp_path.iterdir())


def test_prepare_refuses_without_confirmation(tmp_path, monkeypatch):
    '''Function to test that an existing run folder is kept, and an error raised, when clearing is not confirmed.'''
    monkeypatch.setattr(reporting, '_confirm', lambda question: False)
    (tmp_path / 'old.txt').write_text('old')
    with pytest.raises(FileExistsError):
        prepare_run_dir(tmp_path)
    assert (tmp_path / 'old.txt').read_text() == 'old'


def test_confirm_without_terminal_means_no(monkeypatch):
    '''Function to test that, with no terminal to ask on (scripts, CI), the answer is no instead of a hang.'''
    monkeypatch.setattr(reporting.sys.stdin, 'isatty', lambda: False, raising=False)
    assert reporting._confirm('Delete? ') is False

#%% Finding training runs:

def test_find_runs_skips_benchmarks_and_profiles(tmp_path):
    '''Function to test that only training runs are found, although every run folder holds results.csv.'''
    for name in ('train_main_cfg_cpu', 'shakespeare', 'benchmark_train_main_cfg_cpu', 'profile_decode_main_cfg_cpu'):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'results.csv').write_text('step\n1\n')
    (tmp_path / 'no_results').mkdir()

    assert [p.name for p in find_runs(tmp_path)] == ['shakespeare', 'train_main_cfg_cpu']
    assert find_runs(tmp_path / 'shakespeare') == [tmp_path / 'shakespeare']        # a run folder itself
    assert find_runs(tmp_path / 'benchmark_train_main_cfg_cpu') == []               # not a training run
    assert find_runs(tmp_path / 'missing') == []
