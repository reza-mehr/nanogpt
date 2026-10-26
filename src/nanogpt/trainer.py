
'''
This file contains the trainer code for the model.

Each run folder ('cfg.run_dir', by default 'runs/train_<preset>_<device>') holds:
- config.json, env.json: the full run configuration and the environment at the start (code version, software, hardware)
- results.csv: one row per evaluation, appended and flushed as training goes, trimmed on resume
- run.log: everything logged during the run, appended across resumes
- tb/: TensorBoard event files for live monitoring
- latest.pt, best.pt: training checkpoints (weights, optimizer and RNG states)
- report.md: written when the run completes; summary and fixed-seed samples first, then configuration and environment
- model.pt: written when the run completes; inference-only export of best.pt (bf16 weights, configs, tokenizer)

A fresh run into a folder that already holds files asks before clearing it (see 'prepare_run_dir').

'''

#%% Imports:

import os
import shutil
import time
from dataclasses import dataclass
from logging import Logger

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from nanogpt.config import TrainConfig, config_record
from nanogpt.checkpoint import git_info, load_checkpoint, save_checkpoint
from nanogpt.checkpoint import check_inference_checkpoint, save_inference_checkpoint
from nanogpt.data import TokenSplit, get_batch
from nanogpt.generation import generate_text
from nanogpt.model import build_model
from nanogpt.optimizer import build_optimizer, set_learning_rate
from nanogpt.reporting import Col, append_row_csv, collect_env, load_json, load_rows_csv, prepare_run_dir, save_json
from nanogpt.reporting import setup_logger, table_section, truncate_rows_csv, write_report
from nanogpt.tokenizers import build_tokenizer
from nanogpt.utils import autocast_ctx, get_param_count

#%% Global variables:

# Columns of the summary table at the top of report.md:
SUMMARY_COLS = [Col('metric', 'metric'), Col('value', 'value')]

#%% Run state:

@dataclass
class RunState:
    '''Class to hold the logging tools and bookkeeping shared by the training loop functions.'''
    logger: Logger
    writer: SummaryWriter
    git: dict                       # commit, tag, dirty flag, diff, and untracked files to save to checkpoint
    tokenizer_meta: dict            # tokenizer metadata to save to checkpoint and to rebuild the tokenizer
    best_val: float                 # best smoothed validation loss so far
    mark_time: float                # wall time when the current training interval started (after the last eval)
    mark_step: int                  # last step before the current training interval

#%% Train functions:

def train(cfg: TrainConfig, train_data: TokenSplit, val_data: TokenSplit, tokenizer_meta: dict, stop_after: int | None = None):
    '''
    Function to run the loop to train a language model.

    Parameters
    ----------
    cfg: TrainConfig
        Configurations to use.
    train_data: TokenSplit
        Train data.
    val_data: TokenSplit
        Validation data.
    tokenizer_meta: dict
        Containing tokenizer metadata to include in the checkpoint.
    stop_after: int, optional
        Step to stop at for testing purposes. The default is None.

    Returns
    -------
    model: nn.Module
        Trained model.
    best_val: float
        Best smoothed loss value over validation data.

    '''
    # Fix the random seed for reproducibility:
    torch.manual_seed(cfg.seed)

    # Prepare the run folder; a fresh run must not mix its records with an earlier run of the same name:
    resuming = cfg.resume and (cfg.run_dir / 'latest.pt').exists()
    if not resuming:
        prepare_run_dir(cfg.run_dir, cfg.overwrite)

    # Logger:
    logger = setup_logger(cfg.run_dir, cfg.log_level)

    # Initialization:
    with torch.device(cfg.device):
        model = build_model(cfg.model)
    optim = build_optimizer(model, cfg)
    latest = cfg.run_dir / 'latest.pt'          # path to the latest checkpoints
    results_path = cfg.run_dir / 'results.csv'
    start_step = 1
    best_val = float('inf')

    logger.info(f"Built '{cfg.model.name}' model with {get_param_count(model)/1e6:.2f}M parameters.")
    if cfg.model.vocab_size_padded > cfg.model.vocab_size:
        logger.info(f"Raised 'vocab_size' to a multiple of {cfg.model.pad_vocab_to_multiple}: {cfg.model.vocab_size} -> {cfg.model.vocab_size_padded}")
    if cfg.grad_accum_iters > 1:
        B, T, ga_st = cfg.batch_size, cfg.model.block_size, cfg.grad_accum_iters
        logger.info(f"Performing {ga_st} gradient accumulation sub-steps with (B, T)=({B}, {T}) to realize {B*T*ga_st} tokens per batch.")

    # Load the latest checkpoint if requested, and drop metrics logged after it:
    if resuming:
        start_step, best_val = load_checkpoint(latest, model, optim, cfg.device)
        truncate_rows_csv(results_path, key='step', max_value=start_step)
        logger.info(f"Resumed from step {start_step}, best val loss {best_val:.4f}")
        start_step += 1
    else:
        # Record the configuration and the environment at the start of a fresh run:
        save_json(config_record(cfg), cfg.run_dir / 'config.json')
        save_json(collect_env(cfg.device), cfg.run_dir / 'env.json')

    # Compile the model for efficiency if requested:
    model_compiled = torch.compile(model) if cfg.compile_model else model

    # TensorBoard writer, created after resuming so it hides only the events logged after the checkpoint:
    writer = SummaryWriter(log_dir=cfg.run_dir / 'tb', purge_step=start_step)
    logger.info(f'TensorBoard: uv run tensorboard --logdir {cfg.run_dir / "tb"}\n')

    state = RunState(logger=logger, writer=writer, git=git_info(), tokenizer_meta=tokenizer_meta, best_val=best_val,
                     mark_time=time.perf_counter(), mark_step=start_step - 1)

    # Training loop:
    try:
        for step in range(start_step, cfg.max_iters+1):
            # Run a training step:
            lr, grad_norm = train_step(cfg, model_compiled, optim, step, train_data, writer)

            # Evaluate, record metrics, and save checkpoints:
            if step % cfg.eval_interval == 0 or step % cfg.ckpt_interval == 0 or step == cfg.max_iters:
                evaluate_and_save(cfg, model, optim, step, train_data, val_data, lr, grad_norm, state)

            # Emulate a crash for testing checkpoints:
            if stop_after is not None and step == stop_after: return model, state.best_val

    finally:
        writer.close()                    # flush event files, even on crash or early return

    # Records of the completed run:
    write_run_report(cfg, model, state)
    write_inference_model(cfg, state)

    return model, state.best_val


def train_step(cfg: TrainConfig, model: nn.Module, optim: Optimizer, step: int, train_data: TokenSplit, writer: None | SummaryWriter = None):
    '''
    Function to run a single optimization step and log its training loss, learning rate, and gradient norm.

    Parameters
    ----------
    cfg: TrainConfig
        Configurations to use.
    model: nn.Module
        Model to train.
    optim: Optimizer
        Optimizer.
    step: int
        Current optimization step.
    train_data: TokenSplit
        Train data.
    writer: SummaryWriter, optional
        Tensorboard summary writer utility. The default is None.

    Returns
    -------
    lr: float
        Learning rate used in this step.
    grad_norm: float
        Gradient norm before clipping.

    '''
    # Learning rate according to the schedule:
    lr = set_learning_rate(optim, step, cfg)

    # Initialization:
    optim.zero_grad(set_to_none=True)
    loss_total = 0.                                 # becomes a tensor on the loss's device after the first sub-step

    # Loop over grad accumulation sub_steps:
    for _ in range(cfg.grad_accum_iters):
        # Forward pass:
        xb, yb = get_batch(train_data, cfg)
        with autocast_ctx(cfg.device, cfg.precision):
            _, loss = model(xb, yb)

        # Loss computation:
        loss = loss / cfg.grad_accum_iters          # divide by number of sub-steps for correct reduction to mean
        loss_total += loss.detach()
        loss.backward()

    # Optimization step:
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)   # clip large gradients to limit noisy batches' impact
    optim.step()

    # Log training loss, learning rate, and gradient norm:
    if writer is not None:
        writer.add_scalar('train/loss', loss_total.item(), step)
        writer.add_scalar('train/lr', lr, step)
        writer.add_scalar('train/grad_norm', grad_norm.item(), step)

    return lr, grad_norm.item()


#%% Evaluation utilities:

def evaluate_and_save(cfg: TrainConfig, model: nn.Module, optim: Optimizer, step: int, train_data: TokenSplit, val_data: TokenSplit, lr: float, grad_norm: float, state: RunState):
    '''
    Function to estimate the smoothed losses, record them (log, TensorBoard, results.csv), and save checkpoints.

    Throughput covers only the training steps since the previous evaluation, so evaluation and checkpointing time
    does not dilute it.

    Parameters
    ----------
    cfg: TrainConfig
        Configurations to use.
    model: nn.Module
        Model being trained.
    optim: nn.optim
        Optimizer.
    step: int
        Current optimization step.
    train_data, val_data: TokenSplit
        Train and validation data.
    lr: float
        Learning rate of the current step.
    grad_norm: float
        Gradient norm of the current step, before clipping.
    state: RunState
        Logging tools and bookkeeping; 'best_val' and the interval marks are updated in place.

    '''
    # Initialization:
    latest, best = cfg.run_dir / 'latest.pt', cfg.run_dir / 'best.pt'   # checkpoint paths
    tokens_per_step = cfg.batch_size * cfg.model.block_size * cfg.grad_accum_iters

    # Throughput over the training interval since the last evaluation:
    train_s = time.perf_counter() - state.mark_time
    tok_per_sec = (step - state.mark_step) * tokens_per_step / train_s

    # Smoothed losses:
    losses = estimate_loss(cfg, model, train_data, val_data)
    train_loss, val_loss = losses['train'].item(), losses['val'].item()
    state.writer.add_scalar('eval/train_loss', train_loss, step)
    state.writer.add_scalar('eval/val_loss', val_loss, step)
    state.writer.add_scalar('eval/tok_per_sec', tok_per_sec, step)
    state.logger.info(f'step {step:5d} / {cfg.max_iters} | train {train_loss:.4f} | val {val_loss:.4f} | lr {lr:.2e} | norm {grad_norm:.4f} | {tok_per_sec:,.0f} tok/s')

    # Append one row to the results file (flushed immediately, so it survives a crash):
    append_row_csv(dict(step=step, tokens=step * tokens_per_step, train=train_loss, val=val_loss, lr=lr,
                        grad_norm=grad_norm, tok_per_sec=tok_per_sec, train_s=train_s), cfg.run_dir / 'results.csv')

    # Save the latest and potentially best checkpoints:
    if step % cfg.ckpt_interval == 0 or step == cfg.max_iters:
        best_checkpoint = (val_loss < state.best_val)
        if best_checkpoint: state.best_val = val_loss
        save_checkpoint(latest, model, optim, step, state.tokenizer_meta, state.best_val, cfg, state.git)
        if best_checkpoint:                         # latest checkpoint is the best checkpoint
            tmp = best.with_suffix('.tmp')
            shutil.copyfile(latest, tmp)
            os.replace(tmp, best)                   # atomic: never leaves a half-written checkpoint

    # Start the next training interval after evaluation and checkpointing:
    state.mark_time, state.mark_step = time.perf_counter(), step


@torch.no_grad()
def estimate_loss(cfg: TrainConfig, model: nn.Module, train_data: TokenSplit, val_data: TokenSplit):
    '''
    Function to estimate the smoothed loss over 'eval_iters'.

    Parameters
    ----------
    cfg: TrainConfig
        Configurations to use.
    model: nn.Module
        Model to compute the loss for.
    train_data: TokenSplit
        Train data.
    val_data: TokenSplit
        Validation data.

    Returns
    -------
    losses_smoothed: dict
        Loss value for train and validation data smoothed over 'eval_iters'.

    '''
    # Initialization:
    losses_smoothed = {}
    model.eval()

    # Loop over train and validation data:
    for split, data in zip(['train', 'val'], [train_data, val_data]):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):         # loop over 'eval_iters'
            xb, yb = get_batch(data, cfg)
            with autocast_ctx(cfg.device, cfg.precision):
                _, loss = model(xb, yb)
            losses[k] = loss
        losses_smoothed[split] = losses.mean()
    model.train()

    return losses_smoothed


#%% Run records:

def write_run_report(cfg: TrainConfig, model: nn.Module, state: RunState):
    '''
    Function to save report.md for a completed run: the summary and the fixed-seed samples first, then the
    configuration and the environment. The summary is computed from results.csv, so resumed runs are summarized in full.

    Parameters
    ----------
    cfg: TrainConfig
        Configurations used.
    model: nn.Module
        Trained model, for the parameter count and the samples.
    state: RunState
        Logging tools and the tokenizer metadata.

    '''
    summary = table_section(summary_rows(cfg, model), SUMMARY_COLS, f'Summary: {cfg.run_dir.name}')
    samples = samples_section(cfg, model, state)
    write_report(cfg.run_dir / 'report.md', [summary, samples], config_record(cfg), load_json(cfg.run_dir / 'env.json'))


def summary_rows(cfg: TrainConfig, model: nn.Module) -> list[dict]:
    '''Function to compute the headline numbers of a run from its results.csv, as preformatted table rows.'''
    rows = load_rows_csv(cfg.run_dir / 'results.csv')
    best, last = min(rows, key=lambda r: r['val']), rows[-1]
    train_s = sum(r['train_s'] for r in rows)
    mean_tok_per_sec = sum(r['tok_per_sec'] * r['train_s'] for r in rows) / train_s     # tokens trained / seconds
    summary = {
        'steps': f"{last['step']:,}",
        'tokens': f"{last['tokens']:,}",
        'final train loss': f"{last['train']:.4f}",
        'final val loss': f"{last['val']:.4f}",
        'best val loss': f"{best['val']:.4f} (step {best['step']:,})",
        'parameters': f'{get_param_count(model):,}',
        'training time (s)': f'{train_s:,.0f}',
        'mean throughput (tok/s)': f'{mean_tok_per_sec:,.0f}',
    }
    return [dict(metric=k, value=v) for k, v in summary.items()]


def samples_section(cfg: TrainConfig, model: nn.Module, state: RunState) -> str:
    '''
    Function to generate one fixed-seed sample per prompt from the final model, comparable across runs.

    Parameters
    ----------
    cfg: TrainConfig
        Configurations used; 'sample_prompts', 'sample_tokens', and 'sample_seed' control the samples.
    model: nn.Module
        Trained model.
    state: RunState
        Logging tools and the tokenizer metadata.

    Returns
    -------
    section: str
        Markdown section with the samples, or a note that sampling failed; sampling must not fail a finished run.

    '''
    header = (f'### Samples\n\nFinal model (step {cfg.max_iters}), seed {cfg.sample_seed}, '
              f'{cfg.sample_tokens} new tokens.')
    try:
        tok = build_tokenizer(state.tokenizer_meta)
        parts = [header]
        for prompt in cfg.sample_prompts:
            text = generate_text(model, tok, prompt, cfg.sample_tokens, temperature=1.0, top_k=None, use_cache=False, seed=cfg.sample_seed)
            parts.append(f'#### Prompt: {prompt!r}\n\n```text\n{text}\n```')
        return '\n\n'.join(parts)
    except Exception as err:
        state.logger.warning(f'Sampling failed: {err!r}')
        return f'{header}\n\nSampling failed: `{err!r}`'


def write_inference_model(cfg: TrainConfig, state: RunState):
    '''
    Function to export best.pt as model.pt, an inference-only checkpoint, and check that it reproduces best.pt.

    A failed check raises, so a bad export stops the run on the training machine rather than surfacing later.

    Parameters
    ----------
    cfg: TrainConfig
        Configurations used.
    state: RunState
        Logging tools.

    '''
    src, dst = cfg.run_dir / 'best.pt', cfg.run_dir / 'model.pt'
    save_inference_checkpoint(src, dst)
    rel_err = check_inference_checkpoint(src, dst)
    state.logger.info(f'Saved {dst} ({dst.stat().st_size / 2**20:,.0f} MiB, '
                      f'logits relative error vs best.pt {rel_err:.1e})')
