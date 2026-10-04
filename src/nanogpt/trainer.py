
#%% Imports:

import os
from os import path
import shutil
import torch
from dataclasses import replace

from nanogpt.data import get_batch
import nanogpt.model                                        # needed to register the models
from nanogpt.registry import build_model
from nanogpt.checkpoint import save_checkpoint, load_checkpoint

'''
This file conntains the trainer code for the model.

'''

#%% Loss function:

@torch.no_grad()
def estimate_loss(cfg, model, train_data, val_data):
    '''
    Function to estimate the smoothed loss over 'eval_iters'.

    Parameters
    ----------
    cfg: TrainConfig
        Configurations to use.
    model: nn.Module
        Model to compute the loss for.
    train_data: torch.Tensor
        Train data.
    val_data: torch.Tensor
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
            xb, yb = get_batch(cfg, data)
            _, loss = model(xb, yb)
            losses[k] = loss
        losses_smoothed[split] = losses.mean()
    model.train()

    return losses_smoothed


#%% Train function:

def train(cfg, train_data, val_data, stop_after=None):
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
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Initialization:
    model = build_model(cfg).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    latest, best = cfg.out_dir / 'latest.pt', cfg.out_dir / 'best.pt'   # checkpoints
    start_step = 1
    best_val = float('inf')

    # Load the latest checkpoint if requested:
    if cfg.resume and latest.exists():
        start_step, best_val = load_checkpoint(latest, model, optim, cfg.device)
        start_step += 1
        print(f"Resumed from step {start_step - 1}, best val loss {best_val:.4f}")

    # Training loop:
    for step in range(start_step, cfg.max_iters+1):
        # Forward pass:
        xb, yb = get_batch(cfg, train_data)
        _, loss = model(xb, yb)

        # Optimization step:
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        # Report smoothed loss:
        if step % cfg.eval_interval == 0 or step % cfg.ckpt_interval == 0 or step == cfg.max_iters:
            losses = estimate_loss(cfg, model, train_data, val_data)
            print(f"{step:04d} - training loss: {losses['train']:.4f}, validation loss: {losses['val']:.4f}")

        # Save checkpoint:
        if step % cfg.ckpt_interval == 0 or step == cfg.max_iters - 1:
            val_loss = losses['val']
            save_checkpoint(latest, model, optim, step, val_loss, cfg)      # latest checkpoint
            if val_loss < best_val:
                best_val = val_loss
                shutil.copyfile(latest, best)                               # latest checkpoint is the best checkpoint

        # Emulate a crash for testing checkpoints:
        if stop_after is not None and step == stop_after: return model

    return model, best_val
