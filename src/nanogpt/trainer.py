
'''
This file conntains the trainer code for the model.

Each run folder ('cfg.out_dir') holds:
- config.json, env.json: the run configuration and the environment at the start (code version, software, hardware)
- metrics.csv: one row per evaluation, appended and flushed as training goes, trimmed on resume
- tb/: TensorBoard event files for live monitoring
- latest.pt, best.pt: checkpoints
- summary.json, samples.md: written when the run completes (final/best losses and throughput; fixed-seed samples)
- the logger's output file, if 'setup_logger' writes one

'''

#%% Imports:

import shutil
import time
from dataclasses import asdict, dataclass
from logging import Logger

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from nanogpt.config import TrainConfig
from nanogpt.checkpoint import git_info, load_checkpoint, save_checkpoint
from nanogpt.data import TokenSplit, get_batch
from nanogpt.generation import generate_text
from nanogpt.model import build_model
from nanogpt.optimizer import build_optimizer, set_learning_rate
from nanogpt.reporting import append_row_csv, collect_env, load_rows_csv, save_json, setup_logger, truncate_rows_csv
from nanogpt.tokenizers import build_tokenizer
from nanogpt.utils import autocast_ctx

#%% Run state:

@dataclass
class RunState:
    '''Class to hold the logging tools and bookkeeping shared by the training loop functions.'''
    logger: Logger
    writer: SummaryWriter
    git: dict                       # commit, dirty flag, and diff to save to checkpoint
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
    train_data: torch.Tensor
        Train data.
    val_data: torch.Tensor
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

    # Create the run folder:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    # Initialization:
    with torch.device(cfg.device):
        model = build_model(cfg.model)
    optim = build_optimizer(model, cfg)
    latest = cfg.run_dir / 'latest.pt'          # path to the latest checkpoints
    metrics_path = cfg.run_dir / 'metrics.csv'
    start_step = 1
    best_val = float('inf')

    # Logging tools:
    logger = setup_logger(cfg)
    writer = SummaryWriter(log_dir=cfg.run_dir / 'tb', purge_step=start_step)
    logger.info(f'TensorBoard: uv run tensorboard --logdir {cfg.run_dir / "tb"}\n')

    # Load the latest checkpoint if requested, and drop metrics logged after it:
    if cfg.resume and latest.exists():
        start_step, best_val = load_checkpoint(latest, model, optim, cfg.device)
        start_step += 1
        truncate_rows_csv(metrics_path, key='step', max_value=start_step - 1)
        logger.info(f"Resumed from step {start_step - 1}, best val loss {best_val:.4f}")
    else:
        # Record the configuration and the environment at the start of a fresh run:
        save_json(asdict(cfg), cfg.run_dir / 'config.json')
        save_json(collect_env(cfg.device), cfg.run_dir / 'env.json')

    state = RunState(logger=logger, writer=writer, git=git_info(), tokenizer_meta=tokenizer_meta, best_val=best_val,
                     mark_time=time.perf_counter(), mark_step=start_step - 1)

    # Training loop:
    try:
        for step in range(start_step, cfg.max_iters+1):
            # Run a training step:
            lr, grad_norm = train_step(cfg, model, optim, step, train_data, writer)

            # Evaluate, record metrics, and save checkpoints:
            if step % cfg.eval_interval == 0 or step % cfg.ckpt_interval == 0 or step == cfg.max_iters:
                evaluate_and_save(cfg, model, optim, step, train_data, val_data, lr, grad_norm, state)

            # Emulate a crash for testing checkpoints:
            if stop_after is not None and step == stop_after: return model, state.best_val

    finally:
        writer.close()                    # flush event files, even on crash or early return

    # Records of the completed run:
    write_summary(cfg, model, state)
    write_samples(cfg, model, state)

    return model, state.best_val


def train_step(cfg: TrainConfig, model: nn.Module, optim: Optimizer, step: int, train_data: TokenSplit, writer: SummaryWriter):
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
    writer: SummaryWriter
        Tensorboard summary writer utility.

    Returns
    -------
    lr: float
        Learning rate used in this step.
    grad_norm: float
        Gradient norm before clipping.

    '''
    # Forward pass:
    xb, yb = get_batch(train_data, cfg)
    with autocast_ctx(cfg.device, cfg.precision):
        _, loss = model(xb, yb)

    # Optimization step:
    lr = set_learning_rate(optim, step, cfg)                            # set the learning rate according to the schedule
    optim.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)   # clip large gradients to limit noisy batches' impact
    optim.step()

    # Log training loss, learning rate, and gradient norm:
    writer.add_scalar('train/loss', loss.item(), step)
    writer.add_scalar('train/lr', lr, step)
    writer.add_scalar('train/grad_norm', grad_norm.item(), step)

    return lr, grad_norm.item()


#%% Evaluation utilities:

def evaluate_and_save(cfg: TrainConfig, model: nn.Module, optim: Optimizer, step: int, train_data: TokenSplit, val_data: TokenSplit, lr: float, grad_norm: float, state: RunState):
    '''
    Function to estimate the smoothed losses, record them (log, TensorBoard, metrics.csv), and save checkpoints.

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
    tokens_per_step = cfg.batch_size * cfg.model.block_size

    # Throughput over the training interval since the last evaluation:
    train_s = time.perf_counter() - state.mark_time
    tok_per_sec = (step - state.mark_step) * tokens_per_step / train_s

    # Smoothed losses:
    losses = estimate_loss(cfg, model, train_data, val_data)
    train_loss, val_loss = losses['train'].item(), losses['val'].item()
    state.writer.add_scalar('eval/train_loss', train_loss, step)
    state.writer.add_scalar('eval/val_loss', val_loss, step)
    state.writer.add_scalar('eval/tok_per_sec', tok_per_sec, step)
    state.logger.info(f'step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {lr:.2e} | {tok_per_sec:,.0f} tok/s')

    # Append one row to the metrics file (flushed immediately, so it survives a crash):
    append_row_csv(dict(step=step, tokens=step * tokens_per_step, train=train_loss, val=val_loss, lr=lr,
                        grad_norm=grad_norm, tok_per_sec=tok_per_sec, train_s=train_s), cfg.run_dir / 'metrics.csv')

    # Save checkpoint:
    if step % cfg.ckpt_interval == 0 or step == cfg.max_iters:
        save_checkpoint(latest, model, optim, step, state.tokenizer_meta, val_loss, cfg, state.git)      # latest checkpoint
        if val_loss < state.best_val:
            state.best_val = val_loss
            shutil.copyfile(latest, best)                               # latest checkpoint is the best checkpoint

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

def write_summary(cfg: TrainConfig, model: nn.Module, state: RunState):
    '''
    Function to save summary.json for a completed run from its metrics.csv, so resumed runs are summarized in full.

    Parameters
    ----------
    cfg: TrainConfig
        Configurations used.
    model: nn.Module
        Trained model, for the parameter count.
    state: RunState
        Logging tools.

    '''
    rows = load_rows_csv(cfg.run_dir / 'metrics.csv')
    best, last = min(rows, key=lambda r: r['val']), rows[-1]
    train_s = sum(r['train_s'] for r in rows)
    summary = dict(
        run=cfg.run_name,
        steps=last['step'],
        tokens=last['tokens'],
        final_train=last['train'],
        final_val=last['val'],
        best_val=best['val'],
        best_step=best['step'],
        n_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        train_seconds=train_s,
        mean_tok_per_sec=sum(r['tok_per_sec'] * r['train_s'] for r in rows) / train_s,     # tokens trained / seconds
        seed=cfg.seed,
        git=collect_env(cfg.device)['git'],
    )
    save_json(summary, cfg.run_dir / 'summary.json')
    state.logger.info(f"Summary: best val {summary['best_val']:.4f} at step {summary['best_step']}, "
                      f"{summary['mean_tok_per_sec']:,.0f} tok/s over {summary['train_seconds']:,.0f} s")


def write_samples(cfg: TrainConfig, model: nn.Module, state: RunState):
    '''
    Function to save samples.md: one fixed-seed generation per prompt from the final model, comparable across runs.

    Parameters
    ----------
    cfg: TrainConfig
        Configurations used; 'sample_prompts', 'sample_tokens', and 'sample_seed' control the samples.
    model: nn.Module
        Trained model.
    state: RunState
        Logging tools and the tokenizer metadata.

    '''
    try:
        tok = build_tokenizer(state.tokenizer_meta)
        sections = [f'Final model (step {cfg.max_iters}), seed {cfg.sample_seed}, {cfg.sample_tokens} new tokens.']
        for prompt in cfg.sample_prompts:
            text = generate_text(model, tok, prompt, cfg.sample_tokens, temperature=1.0, top_k=None, use_cache=False, seed=cfg.sample_seed)
            sections.append(f'### Prompt: {prompt!r}\n\n```text\n{text}\n```')
        (cfg.run_dir / 'samples.md').write_text('\n\n'.join(sections) + '\n')
    except Exception as err:                                            # sampling must not fail a finished run
        state.logger.warning(f'Sampling failed: {err!r}')
