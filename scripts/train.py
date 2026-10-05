
#%% Imports:

import os
from os import path
import urllib.request
import torch
from dataclasses import replace

from nanogpt.config import parse_config
from nanogpt.data import load_splits
from nanogpt.trainer import train

'''
This file conntains the code to train nano-GPT model on tiny Shakespeare dataset.

'''

#%% Main function:

# Config setup:
cfg = parse_config()

# Download the tiny Shakespeare dataset:
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
db_path = "data/input.txt"
if not path.exists(db_path): urllib.request.urlretrieve(url, db_path)

# Load and split the dataset:
train_data, val_data, tokenizer = load_splits(db_path, cfg.data.train_frac)
cfg.model = replace(cfg.model, vocab_size=tokenizer.vocab_size)

# Train the model:
train(cfg, train_data, val_data)

#%% Inference:

# context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
# print(decode(model.generate(context, 100)[0].tolist()), "\n\n")

#%% Cosine similarity:

# W = model.token_embedding_table.weight.detach()
# W = W / W.norm(dim=1, keepdim=True)  # normalize rows
# sim = W @ W.T  # cosine similarity, (65, 65)
# for ch in ['a', 'A', 'e', ' ']:
#     i = stoi[ch]
#     nearest = sim[i].topk(5).indices[1:]  # skip itself
#     print("- chars similar to", repr(ch), "->", [itos[j.item()] for j in nearest])
