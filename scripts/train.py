
#%% Imports:

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

# Load and split the dataset:
train_data, val_data, tokenizer_meta = load_splits(cfg.data.dir)
cfg.model = replace(cfg.model, vocab_size=tokenizer_meta['vocab_size'])

# Train the model:
train(cfg, train_data, val_data, tokenizer_meta)

#%% Cosine similarity:

# W = model.token_embedding_table.weight.detach()
# W = W / W.norm(dim=1, keepdim=True)  # normalize rows
# sim = W @ W.T  # cosine similarity, (65, 65)
# for ch in ['a', 'A', 'e', ' ']:
#     i = stoi[ch]
#     nearest = sim[i].topk(5).indices[1:]  # skip itself
#     print("- chars similar to", repr(ch), "->", [itos[j.item()] for j in nearest])
