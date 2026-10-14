
#%% Imports:

import os
import torch
from torch import nn
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import Optimizer

from nanogpt.config import TrainConfig, ModelConfig
from nanogpt.model import build_model
from nanogpt.tokenizers import Tokenizer, build_tokenizer

'''
This file contains the code to save and restore checkpoints.

'''

#%% Utilities:

def git_info() -> dict:
    '''Function to git the status of the code at checkpointing time.'''
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


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
        Commit, dirty flag, and diff, captured once at the start of the run.

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
