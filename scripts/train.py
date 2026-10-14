
#%% Imports:

from dataclasses import replace

from nanogpt.config import parse_config
from nanogpt.data import load_splits
from nanogpt.trainer import train

'''
This file contains the code to train nano-GPT model.

'''

#%% Main function:

# Config setup:
cfg = parse_config()

# Load the split dataset:
train_data, val_data, tokenizer_meta = load_splits(cfg.data.dir)
cfg.model = replace(cfg.model, vocab_size=tokenizer_meta['vocab_size'])

# Train the model:
train(cfg, train_data, val_data, tokenizer_meta)
