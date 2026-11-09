'''
This file contains the memory model fit mem(T) = c + a T + b T^2: least-squares fitting per group, the fit table,
and the plot of measured points against the fitted curves.

'''

#%% Imports:

import math
from pathlib import Path

import numpy as np

from nanogpt.reporting import Col, load_json, load_rows_csv, log, save_rows_csv, table_section, write_report
from nanogpt.reporting import SERIES_COLORS, new_axes, plot_markers, save_figure

#%% Report columns:

FIT_COLS = [Col('variant', 'variant'), Col('c', 'c (MiB)', ',.0f'), Col('a', 'a (MiB/token)', ',.3f'),
            Col('b', 'b (MiB/token²)', '.2e'), Col('quad_pct', 'T² share at max T (%)', '.0f'),
            Col('r2', 'R²', '.4f'), Col('n', 'points', 'd')]

#%% Memory model fitting:

def fit_quadratic(x: list[float], y: list[float]) -> tuple[float, float, float, float]:
    '''
    Function to fit y = c + a x + b x^2 by least squares.

    Parameters
    ----------
    x, y: list[float]
        Data points; at least three distinct x values.

    Returns
    -------
    c, a, b: float
        Constant, linear, and quadratic coefficients.
    r2: float
        Coefficient of determination of the fit.

    '''
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    b, a, c = np.polyfit(x, y, deg=2)                       # highest power first
    residual = y - (c + a * x + b * x**2)
    total = y - y.mean()
    r2 = 1 - (residual @ residual) / (total @ total) if total @ total > 0 else math.nan
    return float(c), float(a), float(b), float(r2)


def fit_memory(rows: list[dict], group_key: str = 'variant', x_key: str = 'T', y_key: str = 'mem') -> list[dict]:
    '''
    Function to fit the memory model mem(T) = c + a T + b T^2 per group, e.g. per journey variant.

    The constant 'c' is the length-independent memory (weights, gradients, optimizer state), the linear term 'a T'
    covers activations that grow with length, and the quadratic term 'b T^2' covers the T x T attention scores.

    Parameters
    ----------
    rows: list[dict]
        Flat benchmark rows, e.g. loaded with 'load_rows_csv'.
    group_key, x_key, y_key: str, optional
        Row keys for the group, the length, and the memory. The defaults are 'variant', 'T', and 'mem'.

    Returns
    -------
    fits: list[dict]
        One row per group with enough finite points: coefficients, R², the quadratic share of the fitted memory
        at the largest length, and the number of points. Groups with fewer than three points are skipped.

    '''
    fits = []
    for group in dict.fromkeys(r[group_key] for r in rows):    # unique, in first-seen order
        points = [(r[x_key], r[y_key]) for r in rows
                  if r[group_key] == group and not math.isnan(float(r[y_key]))]
        if len({x for x, _ in points}) < 3:
            log.warning(f"Skipping '{group}': fewer than three lengths with measured memory.")
            continue
        x, y = zip(*sorted(points))
        c, a, b, r2 = fit_quadratic(x, y)
        x_max = x[-1]
        quad_pct = 100 * b * x_max**2 / (c + a * x_max + b * x_max**2)
        fits.append(dict(variant=group, c=c, a=a, b=b, quad_pct=quad_pct, r2=r2, n=len(x)))
    return fits


def plot_memory_fit(rows: list[dict], fits: list[dict], path: Path, group_key: str = 'variant',
                    x_key: str = 'T', y_key: str = 'mem', title: str = '') -> None:
    '''
    Function to plot measured memory (markers) and the fitted quadratic model (lines) per group and save the figure.

    Parameters
    ----------
    rows: list[dict]
        Flat benchmark rows with the measured points.
    fits: list[dict]
        Fitted coefficients from 'fit_memory'; groups without a fit are not plotted.
    path: Path
        Output image file, e.g. 'memory_fit.png'.
    group_key, x_key, y_key: str, optional
        Row keys for the group, the length, and the memory. The defaults are 'variant', 'T', and 'mem'.
    title: str, optional
        Figure title. The default is empty.

    '''
    # Initialization:
    fig, ax = new_axes(title or 'Memory vs length: markers measured, lines fitted c + aT + bT²',
                       f'{x_key} (tokens)', 'memory (MiB)', n_series=len(fits))

    # One color per group, in fixed order: measured points and the fitted curve:
    for color, fit in zip(SERIES_COLORS, fits):
        points = sorted((r[x_key], r[y_key]) for r in rows if r[group_key] == fit['variant'])
        x, y = np.array(points, dtype=float).T
        x_dense = np.linspace(0, x.max(), 200)
        ax.plot(x_dense, fit['c'] + fit['a'] * x_dense + fit['b'] * x_dense**2, color=color, linewidth=2,
                label=f"{fit['variant']}  (R² {fit['r2']:.3f})")
        plot_markers(ax, x, y, color)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    save_figure(fig, ax, path)


def report_memory_fit(run_dir: Path, group_key: str = 'variant', x_key: str = 'T', y_key: str = 'mem') -> None:
    '''
    Function to fit the memory model to a benchmark's results and save the fit table and plot in the same folder.

    Reads 'results.csv' (and 'config.json' and 'env.json' if present) from 'run_dir' and writes 'memory_fit.csv',
    'memory_fit.md', and 'memory_fit.png' next to them.

    Parameters
    ----------
    run_dir: Path
        Benchmark output folder, e.g. 'results/benchmark_train_main_cfg_mps'.
    group_key, x_key, y_key: str, optional
        Row keys for the group, the length, and the memory. The defaults are 'variant', 'T', and 'mem'.

    '''
    # Load the results and fit every group:
    rows = load_rows_csv(run_dir / 'results.csv')
    fits = fit_memory(rows, group_key, x_key, y_key)
    if not fits:
        log.warning(f'No group in {run_dir} has measured memory at three or more lengths; nothing to fit.')
        return

    # Save the coefficients, the report, and the plot:
    save_rows_csv(fits, run_dir / 'memory_fit.csv')
    title = 'Memory model fit: mem(T) = c + a T + b T²'
    write_report(run_dir / 'memory_fit.md', [table_section(fits, FIT_COLS, title), '![memory fit](memory_fit.png)'],
                 load_json(run_dir / 'config.json'), load_json(run_dir / 'env.json'))
    plot_memory_fit(rows, fits, run_dir / 'memory_fit.png', group_key, x_key, y_key)
    log.info(f'Saved memory_fit.csv, memory_fit.md, and memory_fit.png in {run_dir}')
