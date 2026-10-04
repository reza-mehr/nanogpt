
#%% Imports:
import os
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

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


def save_checkpoint(path: Path, model, optimizer, step: int, best_val: float, cfg) -> None:
    '''
    Function to save a checkpoint containing the model weights, optimizer and RNG state, code, configs, and best loss value.

    Parameters
    ----------
    path: Path
        To save the checkpoint.
    model: nn.Module
        Model to save.
    optimizer: torch.optim
        Optimizer.
    step: int
        Optimization step.
    best_val: float
        Best loss value.
    cfg: dataclass
        Model configurations.

    '''
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "cfg": asdict(cfg),
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "git": git_info(),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)          # atomic: never leaves a half-written checkpoint


def load_checkpoint(path: Path, model, optimizer, device) -> tuple[int, float]:
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
