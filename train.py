
#%% Imports:

import os
from os import path
import urllib.request
import shutil
import torch
from dataclasses import replace

from config import parseConfig
import model                                        # needed to register the models
from registry import buildModel
from checkpoint import saveCheckpoint, loadCheckpoint

'''
This file conntains the code to train nano-GPT model on tiny Shakespeare dataset.

'''

#%% Config setup:

cfg = parseConfig()
cfg.out_dir.mkdir(parents=True, exist_ok=True)      # create the run folder

torch.manual_seed(cfg.seed)

#%% Data:

# Initialization:
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
db_path = "data/input.txt"

# Download data:
if not path.exists(db_path): urllib.request.urlretrieve(url, db_path)

# Load and inspect the data:
with open(db_path, "r", encoding="utf-8") as f:
    text = f.read()

# Extract unique characters:
chars = sorted(set(text))
vocab_size = len(chars)
cfg.model = replace(cfg.model, vocab_size=vocab_size)

#%% Tokenizer:

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda string: [stoi[ch] for ch in string]
decode = lambda codes: "".join(itos[i] for i in codes)

data = torch.tensor(encode(text), dtype=torch.long)

#%% Split:

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#%% Data loader:

def getBatch(split):
    '''
    Function to generate batches of data.

    Parameters
    ----------
    split: string
        Split type from 'train, val'.

    Returns
    -------
    x (batch_size, block_size): tensor
        Input token indices.
    y (batch_size, block_size): tensor
        Corresponding target toekn indices.

    '''
    # Initialization:
    device = cfg.device
    batch_size, block_size = cfg.batch_size, cfg.model.block_size
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))       # random indices
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


#%% Loss:

@torch.no_grad()
def estimateLoss():
    '''Function to estimate the smoothed loss over 'eval_iters'.'''
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = getBatch(split)
            _, loss = m(xb, yb)
            losses[k] = loss
        out[split] = losses.mean()
    m.train()
    return out


#%% Training:

# Initialization:
m = buildModel(cfg.model).to(cfg.device)
optim = torch.optim.AdamW(m.parameters(), lr=cfg.learning_rate)
latest, best = cfg.out_dir / "latest.pt", cfg.out_dir / "best.pt"       # checkpoints
start_step = 1
best_val = float('inf')

# Load the latest checkpoint if requested:
if cfg.resume and latest.exists():
    start_step, best_val = loadCheckpoint(latest, m, optim, cfg.device)
    start_step += 1
    print(f"Resumed from step {start_step - 1}, best val loss {best_val:.4f}")

# Training loop:
for step in range(start_step, cfg.max_iters+1):
    # Sample a batch of data:
    xb, yb = getBatch("train")

    # Forward pass:
    logits, loss = m(xb, yb)

    # Optimization step:
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    # Report smoothed loss:
    if step % cfg.eval_interval == 0 or step % cfg.ckpt_interval == 0 or step == cfg.max_iters:
        losses = estimateLoss()
        print(f"{step:04d} - training loss: {losses['train']:.4f}, validation loss: {losses['val']:.4f}")

    # Save checkpoint:
    if step % cfg.ckpt_interval == 0 or step == cfg.max_iters - 1:
        val_loss = losses['val']
        saveCheckpoint(latest, m, optim, step, val_loss, cfg)       # latest checkpoint
        if val_loss < best_val:
            best_val = val_loss
            shutil.copyfile(latest, best)                           # latest checkpoint is the best checkpoint

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
