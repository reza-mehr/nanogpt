'''
This file contains the shared reporting utilities for training runs, benchmarks, and profiling:
- run folders: creation, and confirmation before clearing an existing one
- logging to terminal and run.log
- environment capture (code version, software, hardware)
- CSV and JSON records
- terminal and Markdown tables, including pivot tables with percent changes
- plot styling, and loss curves for single runs and run comparisons

'''

#%% Imports:

import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from nanogpt.config import NON_TRAINING_PREFIXES

#%% Global variables:

LOGGER_NAME = 'nanogpt'
log = logging.getLogger(LOGGER_NAME)                # shared by every module; handlers are set by 'setup_logger'

# Column width for printing a table (fits '(+999%)' and '(-100%)'):
_PCT_WIDTH = 7

# Validated categorical palette (light surface), assigned in fixed order, never cycled:
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
_SURFACE, _TEXT, _TEXT_MUTED, _GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'

#%% Run folders:

def prepare_run_dir(run_dir: Path, overwrite: bool = False) -> None:
    '''
    Function to create an empty run folder, clearing an existing non-empty one only after confirmation.

    Parameters
    ----------
    run_dir: Path
        Run folder to prepare.
    overwrite: bool, optional
        If True, clear an existing folder without asking. The default is False.

    Raises
    ------
    FileExistsError
        If the folder holds files and clearing it is not confirmed. Without a terminal to ask on (scripts, CI),
        the answer is no, so nothing is ever deleted silently.

    '''
    if run_dir.is_dir() and any(run_dir.iterdir()):
        question = f"'{run_dir}' already holds files. Delete them and start over? [y/N] "
        if not (overwrite or _confirm(question)):
            raise FileExistsError(f'{run_dir} already holds files: confirm, pass --overwrite, choose another '
                                  f'folder name, or resume a training run with --resume.')
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def _confirm(question: str) -> bool:
    '''Function to ask a yes/no question on the terminal; anything but yes, or no terminal at all, means no.'''
    if not sys.stdin.isatty():
        return False
    try:
        return input(question).strip().lower() in ('y', 'yes')
    except EOFError:
        return False

#%% Logging:

def setup_logger(run_dir: Path | None, level: int = logging.INFO) -> logging.Logger:
    '''
    Function to set up logging to stdout and, for a run, to 'run_dir/run.log'.
    Safe to call more than once per process; each call replaces the previous handlers.

    Parameters
    ----------
    run_dir: Path or None
        Existing run folder to append run.log to; None logs to stdout only (e.g. cross-run comparisons).
    level: int, optional
        Logging level, e.g. logging.INFO (20) or logging.DEBUG (10). The default is logging.INFO.

    Returns
    -------
    logger: logging.Logger
        Configured logger.

    '''
    # Initialization:
    log.setLevel(level)
    log.propagate = False                           # don't also print via the root logger
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Clear handlers from a previous run:
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)

    # Attach the new handlers:
    handlers = [logging.StreamHandler(sys.stdout)]
    if run_dir is not None:
        handlers.append(logging.FileHandler(run_dir / 'run.log', mode='a'))
    for handler in handlers:
        handler.setFormatter(fmt)
        log.addHandler(handler)

    return log


#%% Environment capture:

def collect_env(device: str, world_size: int = 1) -> dict:
    '''
    Function to collect the code version, software, and hardware details needed to interpret a run or benchmark.

    Parameters
    ----------
    device: str
        Device the code runs on.
    world_size: int, optional
        Number of distributed data parallel (DDP) processes.

    Returns
    -------
    env: dict
        Human-readable fields, e.g. git version, torch version, chip, RAM, and power source.

    '''
    ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30
    is_mac = sys.platform == 'darwin'
    env = {
        'date': datetime.now().isoformat(timespec='seconds'),
        'git': _run_cmd(['git', 'describe', '--tags', '--always', '--dirty']),   # e.g. v0.1-3-gabc1234-dirty
        'python': platform.python_version(),
        'torch': torch.__version__,
        'os': platform.platform(),
        'cpu': _run_cmd(['sysctl', '-n', 'machdep.cpu.brand_string']) if is_mac else platform.processor(),
        'ram (GB)': f'{ram_gb:.0f}',
        'torch cpu threads': torch.get_num_threads(),
        'world size': world_size,
        'device': device,
    }
    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        env['gpu'] = f'{props.name}, {props.total_memory / 2**30:.0f} GB'
    if is_mac:
        env['power'] = (_run_cmd(['pmset', '-g', 'batt']).splitlines() or ['n/a'])[0]     # AC vs battery

    return env


def _run_cmd(cmd: list[str]) -> str:
    '''Function to run a command and return its stripped output, or 'n/a' if it fails.'''
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'n/a'


def _meta_lines(meta: dict, bullet: str = '') -> list[str]:
    '''Function to format metadata as 'key: value' lines, optionally as Markdown bullets.'''
    return [f'{bullet}**{k}:** {v}' if bullet else f'{k}: {v}' for k, v in meta.items()]


def _flat_scalars(d: dict, prefix: str = '') -> dict:
    '''
    Function to flatten a nested config record into dotted keys, keeping scalars and lists of scalars only.
    Larger derived values (e.g. a profile's journey) stay in config.json but are left out of the report.

    '''
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out |= _flat_scalars(v, f'{key}.')
        elif not isinstance(v, (list, tuple)) or all(not isinstance(x, (list, tuple, dict)) for x in v):
            out[key] = v
    return out

#%% CSV and JSON records:

def save_rows_csv(rows: list[dict], path: Path) -> None:
    '''Function to save flat rows with identical keys to a CSV file, creating parent folders if missing.'''
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_row_csv(row: dict, path: Path) -> None:
    '''Function to append one row to a CSV file, writing the header first if the file is new; flushed immediately.'''
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new: writer.writeheader()
        writer.writerow(row)


def load_rows_csv(path: Path) -> list[dict]:
    '''Function to load rows saved by 'save_rows_csv' or 'append_row_csv', parsing ints and floats ('nan' included), else text.'''

    def parse(s):
        for cast in (int, float):
            try:
                return cast(s)
            except ValueError:
                pass
        return s

    with open(path, newline='') as f:
        return [{k: parse(v) for k, v in r.items()} for r in csv.DictReader(f)]


def truncate_rows_csv(path: Path, key: str, max_value: float) -> None:
    '''Function to drop the rows whose 'key' exceeds 'max_value', e.g. metrics logged after the step a run resumes from.'''
    if not path.exists(): return
    rows = [r for r in load_rows_csv(path) if r[key] <= max_value]
    if rows:
        save_rows_csv(rows, path)
    else:
        path.unlink()


def save_json(data: dict, path: Path) -> None:
    '''Function to save a dict as indented JSON, converting non-JSON values (e.g. paths) to strings.'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def load_json(path: Path) -> dict:
    '''Function to load a JSON file, or return an empty dict if it does not exist.'''
    return json.loads(path.read_text()) if path.exists() else {}

#%% Tables:

@dataclass
class Col:
    key: str                                        # dict key in each row
    title: str                                      # header text
    fmt: str = ''                                   # number format, e.g. ',.0f'; empty for text
    align: Literal['<', '>'] | None = None          # None -> right-aligned if 'fmt' is set, else left
    title_align: Literal['<', '>'] | None = None    # None -> same as the column alignment

    @property
    def justify(self) -> str:
        '''Function to return the cell alignment character: explicit 'align', otherwise by column type.'''
        return self.align or ('>' if self.fmt else '<')

    @property
    def title_justify(self) -> str:
        '''Function to return the title alignment character: explicit 'title_align', otherwise the cell alignment.'''
        return self.title_align or self.justify


def report_pivot(rows: list[dict], path: Path, row_key: str, col_key: str, metrics: list[tuple[str, str, str]],
                 total_label: str = 'total vs base', config: dict | None = None, env: dict | None = None,
                 ref_key: str | None = None, total_row: str | None = None) -> None:
    '''
    Function to log one pivot table per metric and save them all, with the configuration and the environment, to a
    single Markdown file.

    Parameters
    ----------
    rows: list[dict]
        Flat rows, live or loaded with 'load_rows_csv'.
    path: Path
        Output Markdown file; parent folders are created if missing.
    row_key, col_key: str
        Keys for the table rows and columns, see 'pivot_rows'.
    metrics: list[tuple[str, str, str]]
        '(value_key, title, fmt)' per table, e.g. ('tps', 'Throughput (tokens/sec)', ',.0f').
    total_label: str, optional
        Label of the total row; empty to omit it. The default is 'total vs base'.
    config, env: dict, optional
        Run configuration (config.json) and environment (env.json). The defaults are None (omitted).
    ref_key, total_row: str, optional
        Reference rows for percent changes and the total row, see 'pivot_rows'. The defaults are None.

    '''
    write_report(path, pivot_sections(rows, row_key, col_key, metrics, total_label, ref_key, total_row), config, env)


def pivot_sections(rows: list[dict], row_key: str, col_key: str, metrics: list[tuple[str, str, str]],
                   total_label: str = 'total vs base', ref_key: str | None = None,
                   total_row: str | None = None) -> list[str]:
    '''Function to log one pivot table per metric and return their Markdown sections.'''
    sections = []
    for value_key, title, fmt in metrics:
        table, cols = pivot_rows(rows, row_key, col_key, value_key, fmt, total_label, ref_key, total_row)
        sections.append(table_section(table, cols, title))
    return sections


def table_section(rows: list[dict], cols: list[Col], title: str) -> str:
    '''Function to log a titled table and return it as a Markdown section.'''
    print_table(rows, cols, title)
    return f'### {title}\n\n{markdown_table(rows, cols)}'


def write_report(path: Path, sections: list[str], config: dict | None = None, env: dict | None = None) -> None:
    '''
    Function to save a Markdown report: the given sections, then the configuration and the environment.
    The environment is also logged.

    Parameters
    ----------
    path: Path
        Output Markdown file; parent folders are created if missing.
    sections: list[str]
        Markdown sections to put first, e.g. a summary, tables, and samples.
    config: dict, optional
        Run configuration as saved in config.json; nested keys are flattened to dotted names. The default is None.
    env: dict, optional
        Environment as saved in env.json. The default is None.

    '''
    sections = list(sections)
    if config:
        sections.append('### Configuration\n\n' + '\n'.join(_meta_lines(_flat_scalars(config), bullet='- ')))
    if env:
        log.info('\nEnvironment\n' + '-'*35 + '\n' + '\n'.join(_meta_lines(env)) + '\n')
        sections.append('### Environment\n\n' + '\n'.join(_meta_lines(env, bullet='- ')))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n\n'.join(sections) + '\n')


def print_table(rows: list[dict], cols: list[Col], title: str = '', gap: int = 6) -> None:
    '''Function to log an aligned table, optionally preceded by a title, as one multi-line message.'''
    if not rows: return

    # Initialization:
    widths = [max(len(c.title), *(len(_cell_text(c, r[c.key])) for r in rows)) for c in cols]
    def line(cells):
        return (' ' * gap).join(cells)

    # Log the table (the leading newline keeps the header aligned below the timestamp):
    header = line(f'{c.title:{c.title_justify}{w}}' for c, w in zip(cols, widths))
    lines = ([title] if title else []) + [header, '-' * len(header)]
    lines += [line(_pad(c, w, _cell_text(c, r[c.key])) for c, w in zip(cols, widths)) for r in rows]
    log.info('\n' + '\n'.join(lines) + '\n')


def markdown_table(rows: list[dict], cols: list[Col]) -> str:
    '''Function to render rows as a Markdown table string with aligned, padded cells.'''
    widths = [max(3, len(c.title), *(len(_cell_text(c, r[c.key])) for r in rows)) for c in cols]

    def line(cells):
        return '| ' + ' | '.join(cells) + ' |'

    lines = [line(f'{c.title:{c.title_justify}{w}}' for c, w in zip(cols, widths)),
             line(('-' * (w - 1) + ':') if c.justify == '>' else '-' * w for c, w in zip(cols, widths))]
    lines += [line(_pad(c, w, _cell_text(c, r[c.key])) for c, w in zip(cols, widths)) for r in rows]
    return '\n'.join(lines)


def pivot_rows(rows: list[dict], row_key: str, col_key: str, value_key: str, fmt: str = ',.0f',
               total_label: str = 'total vs base', ref_key: str | None = None,
               total_row: str | None = None) -> tuple[list[dict], list[Col]]:
    '''
    Function to pivot flat rows into one row per 'row_key' value and one column per 'col_key' value.
    Each cell holds a '(value, reference value)' tuple, rendered as the value followed by its percent change. An optional final row gives the change of 'total_row' versus the first row.

    Parameters
    ----------
    rows: list[dict]
        Flat rows, e.g. one per (variant, T) benchmark cell.
    row_key: str
        Key whose values become table rows, in first-seen order, e.g. 'variant'.
    col_key: str
        Key whose values become table columns: sorted if numeric, otherwise in first-seen order.
    value_key: str
        Metric to tabulate, e.g. 'tps' or 'mem'.
    fmt: str, optional
        Number format for the values. The default is ',.0f'.
    total_label: str, optional
        Label of the total row; empty to omit it. The default is 'total vs base'.
    ref_key: str, optional
        Row field naming each row's reference row for the percent change (empty for none).
        The default is None, which compares each row with the previous one.
    total_row: str, optional
        Row compared with the first row in the total row. The default is None, meaning the last row.

    Returns
    -------
    table: list[dict]
        Pivoted rows.
    cols: list[Col]
        Column specs for 'print_table' and 'markdown_table'.

    '''
    # Initialization:
    row_ids = list(dict.fromkeys(r[row_key] for r in rows))        # unique, in first-seen order
    col_ids = list(dict.fromkeys(r[col_key] for r in rows))
    numeric = all(isinstance(j, (int, float)) for j in col_ids)
    if numeric: col_ids = sorted(col_ids)
    value = {(r[row_key], r[col_key]): r[value_key] for r in rows}
    keys = {j: f'{col_key}={j}' for j in col_ids}                    # unique column keys
    titles = {j: keys[j] if numeric else str(j) for j in col_ids}
    refs = ({r[row_key]: r[ref_key] for r in rows} if ref_key
            else {i: row_ids[n - 1] if n else None for n, i in enumerate(row_ids)})
    def get(i, j):
        return value.get((i, j), math.nan)

    # One row per 'row_key' value, each cell paired with its reference row's value:
    table = []
    for i in row_ids:
        ref = refs.get(i) or None                                   # '' from a CSV means no reference
        table.append({row_key: i, **{keys[j]: (get(i, j), get(ref, j) if ref else math.nan) for j in col_ids}})

    # Change of the total row relative to the first:
    if total_label and len(row_ids) > 1:
        last = total_row or row_ids[-1]
        table.append({row_key: total_label, **{keys[j]: f'{_fmt_pct(get(last, j), get(row_ids[0], j)):<{_PCT_WIDTH}}'
                                               for j in col_ids}})

    cols = [Col(row_key, row_key)] + [Col(keys[j], titles[j], fmt, title_align='<') for j in col_ids]

    return table, cols


def _pad(c: Col, w: int, s: str) -> str:
    '''Function to pad a cell string to width 'w' using the column alignment.'''
    return f'{s:{c.justify}{w}}'


def _fmt_pct(x, ref) -> str:
    '''Function to format the percent change of 'x' relative to 'ref' as '(+6%)'; empty if either is missing.'''
    if x is None or ref is None or math.isnan(x) or math.isnan(ref) or ref == 0:
        return ''
    p = round(100 * (x / ref - 1))
    return '(0%)' if p == 0 else f'({p:+d}%)'


def _cell_text(c: Col, v) -> str:
    '''
    Function to format a table cell.

    Parameters
    ----------
    c: Col
        Column spec; 'c.fmt' formats numbers.
    v: number, str, None, or tuple
        Cell value. A '(value, ref)' tuple renders the value followed by its percent change versus 'ref'
        in a fixed-width field, so values and percentages both align, e.g. '18,470 (+6%)  '.
        None or NaN renders as '—'. Strings render as-is.

    Returns
    -------
    text: str
        Formatted cell text.

    '''
    is_pair = isinstance(v, tuple)
    ref = math.nan
    if is_pair:
        v, ref = v
    missing = v is None or (isinstance(v, float) and math.isnan(v))
    text = '—' if missing else str(v) if (isinstance(v, str) or not c.fmt) else format(v, c.fmt)
    return f'{text} {_fmt_pct(v, ref) if not missing else "":<{_PCT_WIDTH}}' if is_pair else text

#%% Plot styling:

def new_axes(title: str, xlabel: str, ylabel: str, n_series: int = 1):
    '''
    Function to create a styled figure and axes: recessive grid and spines, text in text colors.

    Parameters
    ----------
    title, xlabel, ylabel: str
        Figure title and axis labels.
    n_series: int, optional
        Number of colored series to be drawn; more than the palette size raises an error. The default is 1.

    Returns
    -------
    fig, ax: matplotlib Figure and Axes
        Styled figure and axes.

    '''
    import matplotlib
    matplotlib.use('Agg')                                   # file output only, no window
    import matplotlib.pyplot as plt

    if n_series > len(SERIES_COLORS):
        raise ValueError(f'{n_series} series exceed the {len(SERIES_COLORS)} categorical colors; plot fewer groups.')

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.set_xlabel(xlabel, color=_TEXT_MUTED)
    ax.set_ylabel(ylabel, color=_TEXT_MUTED)
    ax.grid(True, color=_GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_TEXT_MUTED)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(_GRID)
    ax.set_title(title, color=_TEXT, loc='left')
    return fig, ax


def save_figure(fig, ax, path: Path) -> None:
    '''Function to add a frameless legend (if any series are labelled), save the figure, and close it.'''
    import matplotlib.pyplot as plt

    if ax.get_legend_handles_labels()[0]:
        legend = ax.legend(frameon=False, loc='best')
        for text in legend.get_texts():
            text.set_color(_TEXT)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_markers(ax, x, y, color: str, label: str | None = None) -> None:
    '''Function to draw measured points as 8px markers with a surface ring that separates overlapping points.'''
    ax.plot(x, y, linestyle='none', marker='o', markersize=8, color=color,
            markeredgecolor=_SURFACE, markeredgewidth=2, label=label)

#%% Training run reports:

# Entries: (row key, table title, number format)
RUN_COLS = [Col('group', 'run / group'), Col('runs', 'runs', 'd'), Col('tokens', 'tokens', ',.0f'),
            Col('final_val', 'final val', '.4f'), Col('final_val_std', '± std', '.4f'),
            Col('best_val', 'best val', '.4f'), Col('best_val_std', '± std', '.4f')]


def is_training_run(path: Path) -> bool:
    '''Function to tell a training run folder: it holds results.csv and its name has no benchmark or profile prefix.'''
    return (path / 'results.csv').is_file() and not path.name.startswith(NON_TRAINING_PREFIXES)


def find_runs(root: Path) -> list[Path]:
    '''Function to return training run folders: 'root' itself if it is one, otherwise its subfolders that are.'''
    if is_training_run(root):
        return [root]
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and is_training_run(p))


def get_dotted(d: dict, key: str):
    '''
    Function to read a nested value with a dotted key, e.g. 'model.n_kv_head'.

    Parameters
    ----------
    d: dict
        To read the value from.
    key: str
        Dotted key to read. Dotted means that the key string can have dots in it.

    Returns
    -------
    value: object
        Value corresponding to the 'key'.

    Raises
    ------
    KeyError
        If 'key' does not exist in 'd'.

    '''
    for k in key.split('.'):
        if not isinstance(d, dict) or k not in d:
            raise KeyError(f"'{key}' not found (missing '{k}')")
        d = d[k]
    return d


def plot_run(run_dir: Path) -> Path:
    '''
    Function to plot the train and validation loss of one run versus tokens seen and save it in the run folder.

    Parameters
    ----------
    run_dir: Path
        Training run folder containing results.csv.

    Returns
    -------
    path: Path
        Saved figure, 'run_dir/loss.png'.

    '''
    rows = load_rows_csv(run_dir / 'results.csv')
    fig, ax = new_axes(f'Loss: {run_dir.name}', 'tokens seen', 'cross-entropy loss', n_series=2)
    for color, split in zip(SERIES_COLORS, ('train', 'val')):
        x = [r['tokens'] for r in rows]
        y = [r[split] for r in rows]
        ax.plot(x, y, color=color, linewidth=2, label=split)
    _format_token_axis(ax)
    path = run_dir / 'loss.png'
    save_figure(fig, ax, path)
    return path


def compare_runs(root: Path, group_by: str | None = None, split: Literal['train', 'val'] = 'val') -> None:
    '''
    Function to compare several runs: loss curves (mean and min-max band per group) and a final/best loss table.

    Without 'group_by', every run is its own group. With 'group_by', runs whose config shares the value of that
    dotted key (e.g. 'model.n_kv_head') form one group, typically the seeds of one ablation setting.
    Writes comparison.png, comparison.md, and comparison.csv into 'root'.

    Parameters
    ----------
    root: Path
        Folder whose subfolders are training runs (each with results.csv, and config.json for grouping).
    group_by: str, optional
        Dotted config key to group runs by. The default is None.
    split: str, optional
        Loss to plot, 'val' or 'train'. The default is 'val'.

    '''
    # Group the runs (groups ordered by config value when numeric, otherwise by name):
    grouped: dict[object, list[Path]] = {}
    for run_dir in find_runs(root):
        config_path = run_dir / 'config.json'
        if group_by and not config_path.exists():
            raise FileNotFoundError(f"Cannot group by '{group_by}' since 'config.json' is missing: {config_path}")
        try:
            value = get_dotted(load_json(config_path), group_by) if group_by else run_dir.name
        except KeyError as err:
            raise KeyError(f"'{group_by}' not in {config_path} ({err})") from err
        grouped.setdefault(value, []).append(run_dir)
    if not grouped:
        log.info(f'No training runs with results.csv found under {root}.')
        return
    numeric = all(isinstance(v, (int, float)) for v in grouped)
    order = sorted(grouped, key=lambda v: v if numeric else str(v))
    groups = {(f'{group_by}={v}' if group_by else v): grouped[v] for v in order}

    # Plot one mean curve per group, with a min-max band across its runs:
    fig, ax = new_axes(f'{split.capitalize()} loss by {group_by or "run"}', 'tokens seen', f'{split} loss',
                       n_series=len(groups))
    table = []
    for color, (label, run_dirs) in zip(SERIES_COLORS, groups.items()):
        curves = [load_rows_csv(d / 'results.csv') for d in run_dirs]
        x = np.array([r['tokens'] for r in curves[0]], dtype=float)                # common grid: first run's evals
        ys = np.array([np.interp(x, [r['tokens'] for r in c], [r[split] for r in c]) for c in curves])
        ax.plot(x, ys.mean(axis=0), color=color, linewidth=2, label=f'{label} (n={len(run_dirs)})')
        if len(run_dirs) > 1:
            ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), color=color, alpha=0.15, linewidth=0)

        finals = np.array([c[-1]['val'] for c in curves])
        bests = np.array([min(r['val'] for r in c) for c in curves])
        table.append(dict(group=label, runs=len(run_dirs), tokens=curves[0][-1]['tokens'],
                          final_val=finals.mean(), final_val_std=finals.std(ddof=1) if len(finals) > 1 else math.nan,
                          best_val=bests.mean(), best_val_std=bests.std(ddof=1) if len(bests) > 1 else math.nan))
    _format_token_axis(ax)
    save_figure(fig, ax, root / 'comparison.png')

    # Save the table and the report:
    save_rows_csv(table, root / 'comparison.csv')
    title = f'Validation loss by {group_by or "run"} (mean over runs; band in the plot is min-max)'
    write_report(root / 'comparison.md', [table_section(table, RUN_COLS, title), '![comparison](comparison.png)'])
    log.info(f'Saved comparison.png, comparison.md, and comparison.csv in {root}')


def _format_token_axis(ax) -> None:
    '''Function to label a tokens axis in engineering units, e.g. 20M instead of 20000000.'''
    from matplotlib.ticker import EngFormatter
    ax.xaxis.set_major_formatter(EngFormatter(sep=''))
    ax.set_xlim(left=0)
