# Imports:
import os
from os import path
import urllib.request
import numpy as np
import torch

from model import BigramLanguageModel, GPT, device

#%% Hyper-parameters:

# Large experiment:
batch_size = 64
block_size = 256
n_embed = 384
head_num = 6
head_size = n_embed // head_num
n_blocks = 6
dropout = 0.2

max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
eval_iters = 200


# Small experiment:
batch_size = 32
block_size = 8
n_embed = 32
head_num = 4
head_size = n_embed // head_num
n_blocks = 4
dropout = 0.1

max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
eval_iters = 200


torch.manual_seed(1337)

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

def get_batch(split):
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
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))       # random indices
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


#%% Loss:

@torch.no_grad()
def estimate_loss():
    '''Function to estimate the smoothed loss over 'eval_iters'.'''
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = m(xb, yb)
            losses[k] = loss
        out[split] = losses.mean()
    m.train()
    return out


#%% Training:

# Initialization:
m = BigramLanguageModel(vocab_size)
# m = GPT(vocab_size, block_size, n_embed, head_size, head_num, n_blocks, dropout)
m = m.to(device)
optim = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Training loop:
for step in range(max_iters):
    # Report smoothed loss:
    if step % eval_interval == 0:
        loss = estimate_loss()
        print(
            f"{step:04d} - training loss: {loss['train']:.4f}, validation loss: {loss['val']:.4f}"
        )

    # Sample a batch of data:
    xb, yb = get_batch("train")

    # Forward pass:
    logits, loss = m(xb, yb)

    # Optimization step:
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

#%% Inference:

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, 100)[0].tolist()), "\n\n")

#%% Cosine similarity:

W = m.token_embedding_table.weight.detach()
W = W / W.norm(dim=1, keepdim=True)  # normalize rows
sim = W @ W.T  # cosine similarity, (65, 65)
for ch in ['a', 'A', 'e', ' ']:
    i = stoi[ch]
    nearest = sim[i].topk(5).indices[1:]  # skip itself
    print("- chars similar to", repr(ch), "->", [itos[j.item()] for j in nearest])
