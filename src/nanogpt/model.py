
'''
Notation:
- B denotes batch size
- T denotes the sequence length and is less than or equal to 'block_size'
- C denotes the channel dimension, could be total number of tokens, embedding dimension, etc.
- end denotes the total number of cached positions

Profiler regions (see 'region' in utils.py, no-ops unless enabled):
- 'embed': token and positional embeddings
- 'layernorm': every layer norm (pre-attention, pre-feed-forward, final)
- 'attn.qkv': query, key, value projections and reshapes
- 'attn.rope': rotary positional embedding
- 'attn.cache': key-value cache update
- 'attn.expand_kv': key-value expansion for grouped-query attention (manual path)
- 'attn.core': attention scores, mask, softmax, and weighted sum of values (or SDPA)
- 'attn.proj': head concatenation and output projection
- 'mlp': feed forward module
- 'lm_head': final linear layer to logits
- 'loss': cross entropy
- 'sample': next-token sampling during generation
Residual additions are left unlabelled and show up as unattributed time.

'''

# Imports:
import torch
from torch import nn
from torch.nn import functional as F

from nanogpt.config import ModelConfig
from nanogpt.registry import MODELS
from nanogpt.utils import sample_logits
from nanogpt.kv_cache import KVCache
from nanogpt.utils import build_rope_cos_sin, apply_rope, expand_kv, get_active_dtype, region

#%% Common interface to build models:

def build_model(cfg: ModelConfig):
    '''Function to build a model given a 'TrainConfig'.'''
    return MODELS.get(cfg.name)(cfg)

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
        with region('attn.qkv'):
            q = self.query(x)               # x @ W_q^T : (B, T, n_embed) @ (n_embed, head_size) -> (B, T, head_size)
            k = self.key(x)                 # x @ W_k^T : (B, T, n_embed) @ (n_embed, head_size) -> (B, T, head_size)
            v = self.value(x)               # x @ W_v^T : (B, T, n_embed) @ (n_embed, head_size) -> (B, T, head_size)

        with region('attn.core'):
            head_size = k.size(-1)
            wei = q @ k.transpose(-2, -1)   # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
            wei = wei * head_size ** (-0.5) # scaling for unit variance

            # Remove non-causal communications between tokens and normalize the weights:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))    # (B, T, T)
            wei = F.softmax(wei, dim=-1)    # (B, T, T)

            wei = self.dropout(wei)
            out = wei @ v                   # token communications: (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

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
        heads = [head(x) for head in self.multi_head]   # head_num x (B, T, head_size)
        with region('attn.proj'):
            x = torch.concat(heads, dim=-1) # (B, T, n_embed) -> (B, T, head_size * head_num)
            x = self.proj(x)                # (B, T, head_size * head_num) -> (B, T, n_embed)
            x = self.dropout(x)             # (B, T, n_embed)
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
        with region('mlp'):
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


    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None = None, cache: KVCache | None = None):
        '''
        Function to compute the forward pass.

        Parameters
        ----------
        x (B, T, n_embed): tensor
            Token embeddings.
        rope: tuple of tensors, optional
            Containing the cos and sin tables for RoPE; see build_rope_cos_sin() for details. The default is None when RoPE is not requested.
        cache: KVCache, optional
            Key-value cache. The default is None.

        Returns
        -------
        out (B, T, n_embed): tensor
            Output.

        '''
        # Attributes:
        batched_attn = self.batched_attention

        with region('layernorm'):
            h = self.ln1(x)                             # (B, T, n_embed)
        if batched_attn:
            x = x + self.multi_head(h, rope=rope, cache=cache)  # (B, T, n_embed) -> (B, T, n_embed)
        else:
            x = x + self.multi_head(h)                  # (B, T, n_embed) -> (B, T, n_embed)

        with region('layernorm'):
            h = self.ln2(x)                             # (B, T, n_embed)
        x = x + self.ffwd(h)                            # (B, T, n_embed)
        return x


class LayerNorm(nn.Module):
    '''Class to implement layer norm.'''
    def __init__(self, dim: int, eps: float = 1e-5):
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


    def forward(self, x: torch.Tensor):
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

        # Grouped-query attention dimensions (Q, K, V, respectively):
        out_features = cfg.head_num * cfg.head_size + 2 * cfg.n_kv_head * cfg.head_size
        self.c_attn = nn.Linear(in_features=cfg.n_embed, out_features=out_features, bias=False)

        self.c_proj = nn.Linear(in_features=cfg.head_num * cfg.head_size, out_features=cfg.n_embed)
        self.register_buffer('mask', torch.tril(torch.ones((cfg.block_size, cfg.block_size))))
        self.dropout_attn = nn.Dropout(cfg.dropout)
        self.dropout_resid = nn.Dropout(cfg.dropout)


    def forward(self, x, rope: tuple[torch.Tensor, torch.Tensor] | None = None, cache: KVCache | None = None):
        '''
        Function to compute the forward pass.

        Parameters
        ----------
        x (B, T, n_embed): tensor
            Token embeddings.
        rope: tuple of tensors
            Containing the cos and sin tables for RoPE; see build_rope_cos_sin() for details. The default is None when RoPE is not requested.
        cache: KVCache, optional
            Key-value cache. The default is None.

        Returns
        -------
        out (B, T, n_embed): tensor
            Multi-head self-attention values.

        '''
        # Attributes:
        head_num = self.cfg.head_num
        n_kv_head = self.cfg.n_kv_head
        head_size = self.cfg.head_size
        scaled_dot_prod = self.cfg.scaled_dot_prod
        dropout = self.cfg.dropout
        block_ind = self.block_ind

        # Dimensions:
        B, T, n_embed = x.shape
        q_dim = head_num * head_size                                # query dimension
        kv_dim = n_kv_head * head_size                              # key and value dimensions

        # Batched self-attention calculations:
        with region('attn.qkv'):
            qkv = self.c_attn(x)    # x @ W^T : (B, T, n_embed) @ (n_embed, q_dim + kv_dim) -> (B, T, q_dim + kv_dim)
            q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=2)     # (B, T, q_dim + kv_dim) -> (B, T, q_dim), (B, T, kv_dim), (B, T, kv_dim)
            q = q.view(B, T, head_num, head_size).transpose(1, 2)   # (B, head_num, T, head_size)
            k = k.view(B, T, n_kv_head, head_size).transpose(1, 2)  # (B, n_kv_head, T, head_size)
            v = v.view(B, T, n_kv_head, head_size).transpose(1, 2)  # (B, n_kv_head, T, head_size)

        # Rotary positional embedding (RoPE):
        if rope is not None:
            with region('attn.rope'):
                cos, sin = rope
                q = apply_rope(q, cos, sin)                         # (B, head_num, T, head_size)
                k = apply_rope(k, cos, sin)                         # (B, n_kv_head, T, head_size)

        # Cache the keys and values if requested:
        start, is_causal = 0, True
        if cache is not None:
            with region('attn.cache'):
                start = cache.len
                assert block_ind is not None, 'block index must be provided for key-value caching'
                assert cache.len == 0 or T == 1, 'only full prefill or single-token decode supported'
                is_causal = cache.len == 0                          # prefill is causal, decode sees everything
                k, v = cache.update(block_ind, k, v)                # (B, head_num, end, head_size)
        end = start + T                                             # 'end==T' when cache is None

        # Use torch's scaled dot product implementation for efficiency:
        if scaled_dot_prod:
            with region('attn.core'):
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    is_causal=is_causal,                            # replaces 'tril' mask for causality
                    dropout_p=dropout if self.training else 0.0,    # replaces 'dropout_attn'
                    enable_gqa=True,                                # enable grouped-query attention
                )                                                   # (B, head_num, T, head_size)
        else:
            # Expand keys and values to match query dimensions for grouped-query attention:
            with region('attn.expand_kv'):
                k, v = expand_kv(k, v, head_num, n_kv_head)         # (B, n_kv_head, T, head_size) -> (B, head_num, T, head_size)

            with region('attn.core'):
                # Causal weight calculations:
                wei = q @ k.transpose(-2, -1) * head_size ** (-0.5)                     # (B, head_num, T, head_size) @ (B, head_num, head_size, end) -> (B, head_num, T, end)
                wei = wei.masked_fill(self.mask[start:end, :end] == 0, float("-inf"))   # (B, head_num, T, end)
                wei = F.softmax(wei, dim=-1)                                            # (B, head_num, T, end)
                wei = self.dropout_attn(wei)

                # Token communications:
                out = wei @ v   # (B, head_num, T, end) @ (B, head_num, end, head_size) -> (B, head_num, T, head_size)

        # Project back to embedding space:
        with region('attn.proj'):
            out = out.transpose(1, 2).contiguous().view(B, T, head_num*head_size)       # (B, T, head_size * head_num)
            out = self.c_proj(out)                                  # (B, T, head_size * head_num) -> (B, T, n_embed)
            out = self.dropout_resid(out)                           # (B, T, n_embed)

        return out


#%% Transformer model:

@MODELS.register('gpt')
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

        # Positional embedding:
        if cfg.pos_embed=='learned':            # construct positional embedding table
            self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embed)
            rope_cos = rope_sin = None
        elif cfg.pos_embed=='rope':             # construct RoPE cosine and sine tables
            self.position_embedding_table = None
            rope_cos, rope_sin = build_rope_cos_sin(cfg.head_size, cfg.rope_base, cfg.block_size)
        self.register_buffer('rope_cos', rope_cos, persistent=False)
        self.register_buffer('rope_sin', rope_sin, persistent=False)

        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.blocks = nn.ModuleList( [ Block(cfg, bind) for bind in range(cfg.n_blocks) ] )
        self.ln = nn.LayerNorm(cfg.n_embed) if cfg.ln_torch else LayerNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size)


    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, cache: KVCache | None = None):
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
        # Initialization:
        B, T = idx.shape

        with region('embed'):
            # Token embedding:
            x = self.token_embedding_table(idx)                             # (B, T, n_embed)

            # Positional embedding:
            start = cache.len if cache is not None else 0
            rope = None
            if self.cfg.pos_embed=='learned':
                pos = torch.arange(start, start + T, device=idx.device)     # absolute positions
                pos_emb = self.position_embedding_table(pos)                # (T, n_embed)
                x = x + pos_emb                                             # (B, T, n_embed)
            elif self.cfg.pos_embed=='rope':
                rope = self.rope_cos[start: start + T], self.rope_sin[start: start + T]

        for block in self.blocks: x = block(x, rope=rope, cache=cache)      # (B, T, n_embed)
        if cache is not None: cache.advance(T)          # advance cache size once after all blocks are cached

        with region('layernorm'):
            x = self.ln(x)                              # (B, T, n_embed)
        with region('lm_head'):
            logits = self.lm_head(x)                    # (B, T, n_embed) -> (B, T, vocab_size)
        if targets is None: return logits, None

        # Compute the loss:
        with region('loss'):
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            logits = logits.view(B, T, C)

        return logits, loss


    @torch.inference_mode()
    def generate(self, idx: torch.Tensor, new_tokens: int, temperature: float = 1.0, top_k: int | None = None, use_cache: bool = True, cache: KVCache | None = None):
        '''
        Function to generate a batch of token indices given an input batch of token indices.

        Parameters
        ----------
        idx (B, T): tensor
            Token indices.
        new_tokens: int
            Number of token indices to generate.
        temperature: float, optional
            Divides the logits before softmax. Values < 1 sharpen the distribution, values > 1 flatten it, and 0 gives greedy (argmax) decoding. The default is 1.0, in which case the distribution is unaltered.
        top_k: int, optional
            Keep only the top_k most likely tokens; top_k=1 gives greedy decoding. The default is None, in which case all tokens are kept.
        use_cache: bool, optional
            If True, use key-value caching. The default is True.
        cache: KVCache, optional
            Key-value cache. The default is None.

        Returns
        -------
        idx_new (B, T + new_tokens): tensor
            Token indices.

        '''
        # Disable training mode for inference:
        train_mode = self.training
        self.eval()

        # Generate the new tokens:
        if use_cache: idx_new = self._generate_cached(idx, new_tokens, temperature, top_k, cache)
        else: idx_new = self._generate(idx, new_tokens, temperature, top_k)

        if train_mode: self.train()                             # revert to original training mode

        return idx_new


    def _generate(self, idx: torch.Tensor, new_tokens: int, temperature: float, top_k: int | None):
        '''
        Function to generate a batch of token indices given an input batch of token indices.
        Training mode is disabled and restored and gradients are not computed in this method.

        Parameters
        ----------
        idx (B, T): tensor
            Token indices.
        new_tokens: int
            Number of token indices to generate.
        temperature: float, optional
            Divides the logits before softmax. Values < 1 sharpen the distribution, values > 1 flatten it, and 0 gives greedy (argmax) decoding. The default is 1.0, in which case the distribution is unaltered.
        top_k: int, optional
            Keep only the top_k most likely tokens; top_k=1 gives greedy decoding. The default is None, in which case all tokens are kept.

        Returns
        -------
        idx (B, T + new_tokens): tensor
            Token indices.

        '''
        # Attributes:
        block_size = self.cfg.block_size

        # Loop over new tokens to be generated:
        for _ in range(new_tokens):
            idx_cond = idx[:, -block_size:]                     # limit to context size (last 'block_size' tokens)
            logits, _ = self(idx_cond)                          # forward pass: (B, T, vocab_size)
            with region('sample'):
                idx_next = sample_logits(logits[:, -1, :],
                                         temperature, top_k)    # sample new tokens: (B, vocab_size) -> (B, 1)
                idx = torch.cat((idx, idx_next), dim=1)         # (B, T+1)

        return idx


    def _generate_cached(self, idx: torch.Tensor, new_tokens: int, temperature: float, top_k: int | None, cache: KVCache | None = None):
        '''
        Function to generate a batch of token indices given an input batch of token indices.
        Training mode is disabled and restored and gradients are not computed in this method.

        Parameters
        ----------
        idx (B, T): tensor
            Token indices.
        new_tokens: int
            Number of token indices to generate.
        temperature: float, optional
                Divides the logits before softmax. Values < 1 sharpen the distribution, values > 1 flatten it, and 0 gives greedy (argmax) decoding. The default is 1.0, in which case the distribution is unaltered.
            top_k: int, optional
                Keep only the top_k most likely tokens; top_k=1 gives greedy decoding. The default is None, in which case all tokens are kept.
        cache: KVCache, optional
            Key-value cache. The default is None in which case, a new cache will be constructed.
            Ensure that the cache parameters match the model and input batch size.

        Returns
        -------
        idx (B, T + new_tokens): tensor
            Token indices.

        '''
        # Attributes:
        cfg = self.cfg

        # Error handling:
        B, T0 = idx.shape
        assert T0 + new_tokens <= cfg.block_size, 'requested number of new tokens exceeds the context size'

        # Construct the cache:
        if cache is None:
            dtype = get_active_dtype(self)
            cache = KVCache(cfg.n_blocks, B, cfg.n_kv_head, cfg.block_size, cfg.head_size, idx.device, dtype)
        else:
            cache.reset()                                       # a new generation always starts from an empty cache

        # Prefill (cache) the key and values for the input context:
        logits, _ = self(idx, cache=cache)

        # Loop over new tokens to be generated:
        for i in range(new_tokens):
            with region('sample'):
                idx_next = sample_logits(logits[:, -1, :],
                                         temperature, top_k)    # sample new tokens: (B, vocab_size) -> (B, 1)
                idx = torch.cat([idx, idx_next], dim=1)
            if i < new_tokens - 1:
                logits, _ = self(idx_next, cache=cache)         # decode: one token

        return idx


#%% Bigram model:

@MODELS.register('bigram')
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


    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
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


    @torch.inference_mode()
    def generate(self, idx: torch.Tensor, new_tokens: int, temperature: float = 1.0, top_k: int | None = None, use_cache: bool = True, cache: KVCache | None = None):
        '''
        Function to generate a batch of token indices given an input batch of token indices.

        Parameters
        ----------
        idx (B, T): tensor
            Token indices.
        new_tokens: int
            Number of token indices to generate.
        temperature: float, optional
            Divides the logits before softmax. Values < 1 sharpen the distribution, values > 1 flatten it, and 0 gives greedy (argmax) decoding. The default is 1.0, in which case the distribution is unaltered.
        top_k: int, optional
            Keep only the top_k most likely tokens; top_k=1 gives greedy decoding. The default is None, in which case all tokens are kept.

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
            idx_next = sample_logits(logits[:, -1, :],
                                     temperature, top_k)        # sample new tokens: (B, vocab_size) -> (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)

        if train_mode: self.train()                             # revert to original training mode

        return idx
