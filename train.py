
#%% Imports:

import os
from os import path
import urllib.request
import shutil
import torch
from dataclasses import replace

from config import parse_config
from data import load_splits, get_batch
import model                                        # needed to register the models
from registry import build_model
from checkpoint import save_checkpoint, load_checkpoint

'''
This file conntains the code to train nano-GPT model on tiny Shakespeare dataset.

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

def train(cfg, train_data, val_data):

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

    return model

#%% Inference:

# context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
# print(decode(m.generate(context, 100)[0].tolist()), "\n\n")

#%% Cosine similarity:

# W = m.token_embedding_table.weight.detach()
# W = W / W.norm(dim=1, keepdim=True)  # normalize rows
# sim = W @ W.T  # cosine similarity, (65, 65)
# for ch in ['a', 'A', 'e', ' ']:
#     i = stoi[ch]
#     nearest = sim[i].topk(5).indices[1:]  # skip itself
#     print("- chars similar to", repr(ch), "->", [itos[j.item()] for j in nearest])

#%% Main function:

if __name__=='__main__':
    # Config setup:
    cfg = parse_config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)      # create the run folder
    torch.manual_seed(cfg.seed)

    # Download the tiny Shakespeare dataset:
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    db_path = "data/input.txt"
    if not path.exists(db_path): urllib.request.urlretrieve(url, db_path)

    # Load and split the dataset:
    train_data, val_data, tokenizer = load_splits(db_path, cfg.train_frac)
    cfg.model = replace(cfg.model, vocab_size=tokenizer.vocab_size)

    # Train the model:
    model = train(cfg, train_data, val_data)
