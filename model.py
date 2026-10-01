# Imports:
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

# %% Global variables:

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# %% Model primitives:


class Head(nn.Module):
    def __init__(self, block_size, n_embed, head_size, dropout):
        super().__init__()

        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, n_embed = x.shape
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)  # (B, T, head_size)

        # Compute causal normalized weights:
        wei = (
            q @ k.transpose(-2, -1) * n_embed ** (-0.5)
        )  # (B, T, head_size) @ # (B, head_size, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, -1)  # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHead(nn.Module):
    def __init__(self, block_size, n_embed, head_size, head_num, dropout):
        super().__init__()
        self.multi_head = nn.ModuleList(
            [Head(block_size, n_embed, head_size, dropout) for _ in range(head_num)]
        )
        self.proj = nn.Linear(head_size * head_num, head_size * head_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.concat([head(x) for head in self.multi_head], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),  # projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, block_size, n_embed, head_size, head_num, dropout):
        super().__init__()
        self.sa_head = MultiHead(block_size, n_embed, head_size, head_num, dropout)
        self.ffwd = FeedForward(head_size * head_num, dropout)
        self.ln1 = LayerNorm(head_size * head_num)
        self.ln2 = LayerNorm(head_size * head_num)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):  # x: (B, T, C)
        xmean = x.mean(-1, keepdim=True)  # (B, T, 1)
        xvar = x.var(-1, keepdim=True, unbiased=False)  # (B, T, 1)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        return self.gamma * xhat + self.beta  # broadcasts over (B, T)


# %% Transformer model:


class Transformer(nn.Module):
    def __init__(
        self, vocab_size, block_size, n_embed, head_size, head_num, n_blocks, dropout
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = nn.Sequential(
            *[
                Block(block_size, n_embed, head_size, head_num, dropout)
                for _ in range(n_blocks)
            ]
        )
        self.ln = LayerNorm(head_size * head_num)
        self.lm_head = nn.Linear(head_size * head_num, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embed)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, n_embed)
        x = tok_emb + pos_emb  # (B, T, n_embed)
        x = self.sa_head(x)  # (B, T, n_embed) -> (B, T, head_size*head_num)
        x = self.ln(x)
        logits = self.lm_head(x)  # (B, T, head_size*head_num) -> (B, T, vocab_size)
        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T)

        block_size = self.block_size

        # Loop over new tokens that will be generated:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # limit to last 'block_size' tokens
            logits, _ = self(idx_cond)  # (B, T, C)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


# %% Bigram model:


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)
        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T)

        # Loop over new tokens that will be generated:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)  # (B, T, C)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx
