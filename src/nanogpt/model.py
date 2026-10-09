# Imports:
import torch
from torch import nn
from torch.nn import functional as F

from nanogpt.config import ModelConfig
from nanogpt.registry import register_model
from nanogpt.kv_cache import KVCache

'''
Notation:
- B denotes batch size
- T denotes the sequence length and is less than or equal to 'block_size'
- C denotes the channel dimension, could be total number of tokens, embedding dimension, etc.
- end denotes the total number of cached positions

'''

#%% Model primitives:

class Head(nn.Module):
    '''Class to implement a single head of self-attention.'''
    def __init__(self, block_size, n_embed, head_size, dropout):
        '''
        Parameters
        ----------
        block_size: int
            Context length or token block size.
        n_embed: int
            Token embedding dimension.
        head_size: int
            Self-attention head dimension.
        dropout: float
            Dropout ratio.
            In modern LLMs it is often not used. It is needed here since tiny Shakespeare is a small dataset.

        Attributes
        ----------
        query (head_size, n_embed): nn.Linear
            Query weight matrix.
        key (head_size, n_embed): nn.Linear
            Key weight matrix.
        value (head_size, n_embed): nn.Linear
            Value weight matrix.
        tril (block_size, block_size): buffer
            Constant lower triangular matrix of ones for causal masking.
        dropout: nn.Dropout
            Dropout layer for training time regularization.

        '''
        super().__init__()

        self.query = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        '''
        Function to compute self-attention.

        Parameters
        ----------
        x (B, T, n_embed): tensor
            Token embeddings.

        Returns
        -------
        out (B, T, head_size): tensor
            Self-attention values.

        '''
        B, T, n_embed = x.shape
        q = self.query(x)                   # x @ W_q^T : (B, T, n_embed) @ (n_embed, head_size) -> (B, T, head_size)
        k = self.key(x)                     # x @ W_k^T : (B, T, n_embed) @ (n_embed, head_size) -> (B, T, head_size)

        head_size = k.size(-1)
        wei = q @ k.transpose(-2, -1)       # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei * head_size ** (-0.5)     # scaling for unit variance

        # Remove non-causal communications between tokens and normalize the weights:
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))    # (B, T, T)
        wei = F.softmax(wei, dim=-1)        # (B, T, T)

        wei = self.dropout(wei)

        v = self.value(x)                   # x @ W_v^T : (B, T, n_embed) @ (n_embed, head_size) -> (B, T, head_size)
        out = wei @ v                       # token communications: (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

        return out


class MultiHeadLooped(nn.Module):
    '''Class to implement the multi-head self-attention as a for loop over single heads.'''
    def __init__(self, cfg: ModelConfig):
        '''
        Parameters
        ----------
        cfg: ModelConfig
            Model configs; see config.py for details.

        Attributes
        ----------
        multi_head: nn.ModuleList
            list of self-attention heads from Head class.
        proj (n_embed, head_size * head_num): nn.Linear
            Linear projection layer.
        dropout: nn.Dropout
            Dropout layer for training time regularization.

        '''
        super().__init__()
        self.multi_head = nn.ModuleList(
            [ Head(cfg.block_size, cfg.n_embed, cfg.head_size, cfg.dropout) for _ in range(cfg.head_num) ]
        )
        self.proj = nn.Linear(cfg.head_size * cfg.head_num, cfg.n_embed)
        self.dropout = nn.Dropout(cfg.dropout)


    def forward(self, x):
        '''
        Function to compute the forward pass.

        Parameters
        ----------
        x (B, T, n_embed): tensor
            Token embeddings.

        Returns
        -------
        out (B, T, n_embed): tensor
            Multi-head self-attention values.

        '''
        x = torch.concat([head(x) for head in self.multi_head], dim=-1)     # (B, T, n_embed) -> (B, T, head_size * head_num)
        x = self.proj(x)                                                    # (B, T, head_size * head_num) -> (B, T, n_embed)
        x = self.dropout(x)                                                 # (B, T, n_embed)
        return x


class FeedForward(nn.Module):
    '''Class to implement the feed forward module for an attention block.'''
    def __init__(self, dim, dropout):
        '''
        Parameters
        ----------
        dim: int
            Linear layer input dimension.
        dropout: float
            Dropout ratio.

        Attributes
        ----------
        net: nn.Sequential
            Feed forward module.

        '''
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),    # projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        '''
        Function to compute the forward pass.

        Parameters
        ----------
        x (B, T, C): tensor
            Input.

        Returns
        -------
        out (B, T, C): tensor
            Output.

        '''
        return self.net(x)


class Block(nn.Module):
    '''Class to implement a single block (attention + feed forward).'''
    def __init__(self, cfg: ModelConfig, block_ind: int):
        '''
        Parameters
        ----------
        cfg: ModelConfig
            Model configs; see config.py for details.
        block_ind: int
            Block index required in key-value cache.

        Attributes
        ----------
        batched_attention: bool
            If True, use the batched implementation of self-attention for efficiency.

        multi_head: MultiHead or MultiHeadLooped instance
            Multi-head self-attention module.
        ffwd: FeedForward instance
            Feed forward module.
        ln1: nn.Linear or LayerNorm instance
            Layer norm applied to the input of the self-attention multi-head.
        ln2: nn.Linear or LayerNorm instance
            Layer norm applied to the input of the feed forward module.

        '''
        super().__init__()
        batched_attn = cfg.batched_attention
        self.batched_attention = batched_attn

        if batched_attn:
            self.multi_head = MultiHead(cfg, block_ind)
        else:
            self.multi_head = MultiHeadLooped(cfg)
        self.ffwd = FeedForward(cfg.n_embed, cfg.dropout)
        self.ln1 = nn.LayerNorm(cfg.n_embed) if cfg.ln_torch else LayerNorm(cfg.n_embed)
        self.ln2 = nn.LayerNorm(cfg.n_embed) if cfg.ln_torch else LayerNorm(cfg.n_embed)


    def forward(self, x, cache: KVCache | None = None):
        '''
        Function to compute the forward pass.

        Parameters
        ----------
        x (B, T, n_embed): tensor
            Token embeddings.
        cache: KVCache, optional
            Key-value cache. The default is None.

        Returns
        -------
        out (B, T, n_embed): tensor
            Output.

        '''
        # Attributes:
        batched_attn = self.batched_attention

        if batched_attn:
            x = x + self.multi_head(self.ln1(x), cache) # (B, T, n_embed) -> (B, T, n_embed)
        else:
            x = x + self.multi_head(self.ln1(x))        # (B, T, n_embed) -> (B, T, n_embed)

        x = x + self.ffwd(self.ln2(x))                  # (B, T, n_embed)
        return x


class LayerNorm(nn.Module):
    '''Class to implement layer norm.'''
    def __init__(self, dim, eps=1e-5):
        '''
        Parameters
        ----------
        dim: int
            Input dimension.
        eps: float, optional
            Small value to prevent division with zero. The default is 1e-5.

        Attributes
        ----------
        eps
        weight: nn.Parameter
            Layer norm scaling.
        bias: nn.Parameter
            Layer norm offset.

        '''
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))


    def forward(self, x):
        '''
        Function to apply layer normalization along feature dimension 'C'.

        Parameters
        ----------
        x (B, T, C): tensor
            Input.

        Returns
        -------
        out (B, T, C): tensor
            Output.

        '''
        xmean = x.mean(-1, keepdim=True)                # (B, T, 1)
        xvar = x.var(-1, keepdim=True, unbiased=False)  # (B, T, 1)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        return self.weight * xhat + self.bias            # broadcasts over (B, T)


#%% Batched multi-head attention:

class MultiHead(nn.Module):
    '''Class to implement batched multi-head self-attention using vectorization for efficiency.'''
    def __init__(self, cfg: ModelConfig, block_ind: int | None = None):
        '''
        Parameters
        ----------
        cfg: ModelConfig
            Model configs; see config.py for details.
        block_ind: int, optional
            Block index that this multi-head of self-attention belongs to, required in key-value cache. The default is None.

        Attributes
        ----------
        cfg
        block_ind

        c_attn (3*head_num*head_size, n_embed): nn.Linear
            Stacked weights for query, key, value matrices of all self-attention heads.
        c_proj (n_embed, head_num*head_size): nn.Linear
            Linear projection layer.
        mask (block_size, block_size): buffer
            Constant lower triangular matrix of ones for causal masking.
        dropout_attn: nn.Dropout
            Dropout layer for masked weights.
        dropout_resid: nn.Dropout
            Dropout layer to be applied to the output of the projection layer.

        '''
        super().__init__()
        self.cfg = cfg
        self.block_ind = block_ind

        self.c_attn = nn.Linear(in_features=cfg.n_embed, out_features=3 * cfg.head_num * cfg.head_size, bias=False)
        self.c_proj = nn.Linear(in_features=cfg.head_num * cfg.head_size, out_features=cfg.n_embed)
        self.register_buffer('mask', torch.tril(torch.ones((cfg.block_size, cfg.block_size))))
        self.dropout_attn = nn.Dropout(cfg.dropout)
        self.dropout_resid = nn.Dropout(cfg.dropout)


    def forward(self, x, cache: KVCache | None = None):
        '''
        Function to compute the forward pass.

        Parameters
        ----------
        x (B, T, n_embed): tensor
            Token embeddings.
        cache: KVCache, optional
            Key-value cache. The default is None.

        Returns
        -------
        out (B, T, n_embed): tensor
            Multi-head self-attention values.

        '''
        # Attributes:
        head_num = self.cfg.head_num
        head_size = self.cfg.head_size
        scaled_dot_prod = self.cfg.scaled_dot_prod
        dropout = self.cfg.dropout
        block_ind = self.block_ind

        # Batched self-attention calculations:
        B, T, n_embed = x.shape
        qkv = self.c_attn(x)                                    # x @ W^T : (B, T, n_embed) @ (n_embed, 3 * head_num * head_size) -> (B, T, 3 * head_num * head_size)
        q, k, v = qkv.split(head_num*head_size, dim=2)          # (B, T, 3 * head_num * head_size) -> 3 x (B, T, head_num * head_size)
        q = q.view(B, T, head_num, head_size).transpose(1, 2)   # (B, head_num, T, head_size)
        k = k.view(B, T, head_num, head_size).transpose(1, 2)   # (B, head_num, T, head_size)
        v = v.view(B, T, head_num, head_size).transpose(1, 2)   # (B, head_num, T, head_size)

        # Cache the keys and values if requested:
        start, is_causal = 0, True
        if cache is not None:
            start = cache.len
            assert block_ind is not None, 'block index must be provided for key-value caching'
            assert cache.len == 0 or T == 1, 'only full prefill or single-token decode supported'
            is_causal = cache.len == 0                          # prefill is causal, decode sees everything
            k, v = cache.update(block_ind, k, v)                # (B, head_num, end, head_size)
        end = start + T                                         # 'end==T' when cache is None

        # Use torch's scaled dot product implementation for efficiency:
        if scaled_dot_prod:
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=is_causal,                            # replaces 'tril' mask for causality
                dropout_p=dropout if self.training else 0.0,    # replaces 'dropout_attn'
            )                                                   # (B, head_num, T, head_size)
        else:
            # Causal weight calculations:
            wei = q @ k.transpose(-2, -1) * head_size ** (-0.5)                     # (B, head_num, T, head_size) @ (B, head_num, head_size, end) -> (B, head_num, T, end)
            wei = wei.masked_fill(self.mask[start:end, :end] == 0, float("-inf"))   # (B, head_num, T, end)
            wei = F.softmax(wei, dim=-1)                                            # (B, head_num, T, end)
            wei = self.dropout_attn(wei)

            # Token communications:
            out = wei @ v   # (B, head_num, T, end) @ (B, head_num, end, head_size) -> (B, head_num, T, head_size)

        # Project back to embedding space:
        out = out.transpose(1, 2).contiguous().view(B, T, head_num*head_size)       # (B, T, head_size * head_num)
        out = self.c_proj(out)                                  # (B, T, head_size * head_num) -> (B, T, n_embed)
        out = self.dropout_resid(out)                           # (B, T, n_embed)

        return out


#%% Transformer model:

@register_model('gpt')
class GPT(nn.Module):
    '''Class to implement the generative pre-trained transformer (GPT). '''
    def __init__(self, cfg: ModelConfig):
        '''
        Parameters
        ----------
        cfg: ModelConfig
            Model configs; see config.py for details.

        Attributes
        ----------
        cfg

        token_embedding_table (vocab_size, n_embed): nn.Embedding
            Embedding table mapping from each token index to embedding.
        position_embedding_table (block_size, n_embed): nn.Embedding
            Embedding table mapping from token positions in the block to embedding space.

        blocks: nn.ModuleList
            List of transformer blocks.
        ln: nn.Linear or LayerNorm instance
            Layer norm applied before the final linear layer.
        lm_head (vocab_size, n_embed): nn.Linear
            Linear layer to generate logits over tokens.

        '''
        super().__init__()
        self.cfg = cfg

        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.blocks = nn.ModuleList( [ Block(cfg, bind) for bind in range(cfg.n_blocks) ] )
        self.ln = nn.LayerNorm(cfg.n_embed) if cfg.ln_torch else LayerNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size)


    def forward(self, idx, targets=None, cache: KVCache | None = None):
        '''
        Function to compute model output and cross entropy loss.

        Parameters
        ----------
        idx (B, T): tensor
            Token indices.
        targets (B, T): tensor, optional
            Target values corresponding to the input. The default is None in which case, loss is not computed.
        cache: KVCache, optional
            Key-value cache. The default is None.

        Returns
        -------
        logits (B, T, vocab_size): tensor
            Logits corresponding to the token space.
        loss: tensor
            Loss value. None if 'targets' is None.

        '''
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)       # (B, T, n_embed)

        start = cache.len if cache is not None else 0
        pos = torch.arange(start, start + T, device=idx.device)    # absolute positions

        pos_emb = self.position_embedding_table(pos)    # (T, n_embed)
        x = tok_emb + pos_emb                           # (B, T, n_embed)

        for block in self.blocks: x = block(x, cache)   # (B, T, n_embed)
        if cache is not None: cache.advance(T)          # advance cache size once after all blocks are cached

        x = self.ln(x)                                  # (B, T, n_embed)
        logits = self.lm_head(x)                        # (B, T, n_embed) -> (B, T, vocab_size)
        if targets is None: return logits, None

        # Compute the loss:
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        logits = logits.view(B, T, C)

        return logits, loss


    @torch.no_grad()
    def generate(self, idx, new_tokens):
        '''
        Function to generate a batch of token indices given an input batch of token indices.
        Training mode is disabled and restored and gradients are not computed in this method.

        Parameters
        ----------
        idx (B, T): tensor
            Token indices.
        new_tokens: int
            Number of token indices to generate.

        Returns
        -------
        idx (B, T + new_tokens): tensor
            Token indices.

        '''
        # Disable training mode for inference:
        train_mode = self.training
        self.eval()

        # Attributes:
        block_size = self.cfg.block_size

        # Loop over new tokens to be generated:
        for _ in range(new_tokens):
            idx_cond = idx[:, -block_size:]                     # limit to context size (last 'block_size' tokens)
            logits, _ = self(idx_cond)                          # forward pass: (B, T, vocab_size)
            probs = F.softmax(logits[:, -1, :], dim=-1)         # probability distribution over tokens (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # new tokens: (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)

        if train_mode: self.train()                             # revert to original training mode

        return idx


    @torch.inference_mode()
    def generate_cached(self, idx, new_tokens: int):
        '''
        Function to generate a batch of token indices given an input batch of token indices.
        Training mode is disabled and restored and gradients are not computed in this method.

        Parameters
        ----------
        idx (B, T): tensor
            Token indices.
        new_tokens: int
            Number of token indices to generate.

        Returns
        -------
        idx (B, T + new_tokens): tensor
            Token indices.

        '''
        # Disable training mode for inference:
        train_mode = self.training
        self.eval()

        # Attributes:
        cfg = self.cfg

        # Error handling:
        B, T0 = idx.shape
        assert T0 + new_tokens <= cfg.block_size, 'requested number of new tokens exceeds the context size'

        # Construct the cache:
        cache = KVCache(cfg.n_blocks, B, cfg.head_num, cfg.block_size, cfg.head_size, idx.device, idx.dtype)

        # Prefill (cache) the key and values for the input context:
        logits, _ = self(idx, cache=cache)

        # Loop over new tokens to be generated:
        for i in range(new_tokens):
            probs = F.softmax(logits[:, -1, :], dim=-1)     # probability distribution over tokens (B, vocab_size)
            nxt = torch.multinomial(probs, num_samples=1)   # new tokens: (B, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if i < new_tokens - 1:
                logits, _ = self(nxt, cache=cache)          # decode: one token

        if train_mode: self.train()                         # revert to original training mode

        return idx

#%% Bigram model:

@register_model('bigram')
class BigramLanguageModel(nn.Module):
    '''This class implements the Bigram language model.'''
    def __init__(self, cfg: ModelConfig):
        '''
        Parameters
        ----------
        cfg: ModelConfig
            Model configs; see config.py for details.

        Attributes
        ----------
        token_embedding_table (vocab_size, n_embed): nn.Embedding
            Embedding table mapping from each token index to embedding.

        '''
        super().__init__()
        n_embed = cfg.vocab_size            # to produce distribution over tokens in Bigram model
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, n_embed)


    def forward(self, idx, targets=None):
        '''
        Function to compute model output and cross entropy loss.

        Parameters
        ----------
        idx (B, T): tensor
            Token indices.
        targets (B, T): tensor, optional
            Target values corresponding to the input. The default is None in which case, loss is not computed.

        Returns
        -------
        logits (B, T, vocab_size): tensor
            Logits corresponding to the input.
        loss: tensor
            Loss value. None if 'targets' is None.

        '''
        # Forward pass:
        logits = self.token_embedding_table(idx)    # (B, T) -> (B, T, vocab_size)
        if targets is None: return logits, None

        # Loss computation:
        B, T, vocab_size = logits.shape
        logits = logits.view(B * T, vocab_size)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        logits = logits.view(B, T, vocab_size)

        return logits, loss


    @torch.no_grad()
    def generate(self, idx, new_tokens):
        '''
        Function to generate a batch of token indices given an input batch of token indices.
        Training mode is disabled and restored and gradients are not computed in this method.


        Parameters
        ----------
        idx (B, T): tensor
            Token indices.
        new_tokens: int
            Number of token indices to generate.

        Returns
        -------
        idx (B, T + new_tokens): tensor
            Token indices.

        '''
        # Disable training mode for inference:
        train_mode = self.training
        self.eval()

        # Loop over new tokens to be generated:
        for _ in range(new_tokens):
            logits, _ = self(idx)                               # (B, T) -> (B, T, vocab_size)
            probs = F.softmax(logits[:, -1, :], dim=-1)         # (B, T, vocab_size) -> (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, vocab_size) -> (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)

        if train_mode: self.train()                             # revert to original training mode

        return idx
