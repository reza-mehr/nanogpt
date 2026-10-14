
'''
This file contains the code to save and restore checkpoints.

'''

#%% Imports:

import os
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer

from nanogpt.config import TrainConfig, ModelConfig
from nanogpt.model import build_model
from nanogpt.tokenizers import Tokenizer, build_tokenizer

#%% Checkpoint utilities:

def git_info() -> dict:
    '''Function to get the status of the code at checkpointing time: commit, dirty flag, diff, and untracked files.'''
    def _git(*args: str) -> str:
        return subprocess.check_output(['git', *args], text=True, stderr=subprocess.DEVNULL)

    try:
        commit = _git('rev-parse', 'HEAD').strip()
        diff = _git('diff', 'HEAD')                     # staged and unstaged changes to tracked files
        untracked = _git('ls-files', '--others', '--exclude-standard').split()
        return {'commit': commit, 'dirty': bool(diff or untracked), 'diff': diff, 'untracked': untracked}
    except Exception:                                   # not a git repository, or git not installed
        return {'commit': None, 'dirty': None, 'diff': None, 'untracked': None}


def save_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer, step: int,
                    tokenizer_meta: dict, best_val: float, cfg: TrainConfig, git: dict) -> None:
    '''
    Function to save a checkpoint containing the model weights, optimizer and RNG states, tokenizer metadata, code version, configs, and best loss value.

    Parameters
    ----------
    path: Path
        Where to save the checkpoint.
    model: nn.Module
        Model to save.
    optimizer: Optimizer
        Optimizer whose state to save.
    step: int
        Number of completed optimization steps.
    tokenizer_meta: dict
        Output of tok.to_meta(), used to rebuild the tokenizer for generation.
    best_val: float
        Best validation loss so far.
    cfg: TrainConfig
        Full training configuration, including the model configuration.
    git: dict
        Commit, dirty flag, diff, and untracked, captured once at the start of the run.

    '''
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'tokenizer': tokenizer_meta,
        'best_val': best_val,
        'cfg': asdict(cfg),
        'rng': {
            'python': random.getstate(),
            'torch': torch.get_rng_state(),
            'mps': torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        'git': git,
    }
    tmp = path.with_suffix('.tmp')
    torch.save(ckpt, tmp)
    os.replace(tmp, path)          # atomic: never leaves a half-written checkpoint


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer, device: str) -> tuple[int, float]:
    '''
    Function to load a checkpoint to fully restore training state.

    Parameters
    ----------
    path: Path
        To load the checkpoint from.
    model: nn.Module
        Model to load the checkpoint into.
    optimizer: torch.optim
        Optimizer to load the checkpoint into.
    device: string
        Device to load the restored model and optimizer into. Could be 'cpu, cuda, mps'.

    Returns
    -------
    step: int
        The step at which the checkpoint was saved.
    best_val: float
        Best loss value up to the point at which the checkpoint was saved.

    '''
    # Load the checkpoint:
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Restore model and optimizer states:
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])

    # Restore random number generator (RNG) states:
    rng = ckpt['rng']
    random.setstate(rng['python'])
    torch.set_rng_state(rng['torch'].cpu())
    if rng['mps'] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(rng['mps'].cpu())
    if rng['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])

    return ckpt["step"], ckpt["best_val"]


#%% Inference-only checkpoint utilities:

def load_model_tokenizer(path: Path, device: str) -> tuple[nn.Module, Tokenizer, dict]:
    '''
    Function to rebuild a model from a checkpoint for inference.

    Parameters
    ----------
    path: Path
        Checkpoint to load.
    device: string
        Device to load the model onto: 'cpu', 'cuda', or 'mps'.

    Returns
    -------
    model: nn.Module
        Model in eval mode, built from the checkpoint's saved config.
    tok: Tokenizer
        Tokenizer to decode generation.
    cfg: dict
        The full saved config, e.g. for block_size or reporting.

    '''
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**ckpt['cfg']['model']))
    model.load_state_dict(ckpt['model'])
    tok = build_tokenizer(ckpt['tokenizer'])

    return model.to(device).eval(), tok, ckpt['cfg']


def save_inference_checkpoint(src: Path, dst: Path, dtype: torch.dtype = torch.bfloat16) -> None:
    '''
    Function to save an inference-only checkpoint: everything 'load_model_tokenizer' needs, without training state.
    - Keeps the weights (floating-point tensors cast to 'dtype'), the configs, and the tokenizer metadata, plus the step, best validation loss, and code version for provenance.
    - Drops the optimizer and RNG states, which hold about two thirds of a training checkpoint. Loading casts the weights back to the model's fp32 parameters.

    Parameters
    ----------
    src: Path
        Training checkpoint to export, e.g. best.pt.
    dst: Path
        Where to save the inference checkpoint, e.g. model.pt.
    dtype: torch.dtype, optional
        Storage dtype of the floating-point weights. The default is torch.bfloat16 (half the size of fp32).

    '''
    ckpt = torch.load(src, map_location='cpu', weights_only=False)
    weights = {k: v.to(dtype) if v.is_floating_point() else v for k, v in ckpt['model'].items()}
    slim = {key: ckpt[key] for key in ('cfg', 'tokenizer', 'step', 'best_val', 'git')}
    slim['model'] = weights

    tmp = dst.with_suffix('.tmp')
    torch.save(slim, tmp)
    os.replace(tmp, dst)           # atomic: never leaves a half-written checkpoint


@torch.no_grad()
def check_inference_checkpoint(src: Path, dst: Path, tol: float = 1e-2, seed: int = 0) -> float:
    '''
    Function to check that an inference checkpoint loads and reproduces the training checkpoint's logits.
    Both checkpoints go through 'load_model_tokenizer' on the CPU and run on the same random tokens. Logits are
    compared by relative error, since rounding weights to bf16 changes them slightly but never by more than ~1e-2.

    Parameters
    ----------
    src: Path
        Training checkpoint, the reference.
    dst: Path
        Inference checkpoint to check.
    tol: float, optional
        Largest accepted relative error ||a - b|| / ||a||. The default is 1e-2.
    seed: int, optional
        Seed for the random input tokens. The default is 0.

    Returns
    -------
    rel_err: float
        Relative error of the inference checkpoint's logits.

    Raises
    ------
    RuntimeError
        If the relative error exceeds 'tol'.

    '''
    # Initialization:
    ref_model, _, cfg = load_model_tokenizer(src, 'cpu')
    model, _, _ = load_model_tokenizer(dst, 'cpu')
    gen = torch.Generator().manual_seed(seed)
    T = min(cfg['model']['block_size'], 32)
    idx = torch.randint(cfg['model']['vocab_size'], (2, T), generator=gen)

    # Compare logits:
    ref_logits, _ = ref_model(idx)
    logits, _ = model(idx)
    rel_err = ((logits - ref_logits).norm() / ref_logits.norm()).item()
    if not rel_err <= tol:                  # also catches NaN
        raise RuntimeError(f'{dst} logits differ from {src} by {rel_err:.2e} (relative), above {tol:.0e}')

    return rel_err
