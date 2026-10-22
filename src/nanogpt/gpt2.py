'''
This file contains the code for the Hugging Face (HF) implementation of the GPT2 model.
The structure and parameter names match the HF version to enable direct loading of the pre-trained parameters.

'''
#%% Imports:

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

#%% Configs:

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_blocks: int = 12
    head_num: int = 12
    n_embed: int = 768

# GPT2 model sizes:
model_configs = {'gpt2': dict(n_blocks=12, head_num=12, n_embed=768),                  # 124M parameters
                 'gpt2-medium': dict(n_blocks=24, head_num=16, n_embed=1024),          # 350M parameters
                 'gpt2-large': dict(n_blocks=36, head_num=20, n_embed=1280),           # 774M parameters
                 'gpt2-xl': dict(n_blocks=48, head_num=25, n_embed=1600),              # 1558M parameters
                 }

#%% Model component classes:

class CausalSelfAttention(nn.Module):
    '''Class to implement the causal self-attention.'''
    def __init__(self, cfg: GPTConfig) -> None:
        '''
        Parameters
        ----------
        cfg: GPTConfig
            Model configurations.

        Attributes
        ----------
        cfg

        '''
        super().__init__()
        self.cfg = cfg

        self.c_attn = nn.Linear(cfg.n_embed, 3 * cfg.n_embed)   # query, key, and value for all heads in batched format
        self.c_proj = nn.Linear(cfg.n_embed, cfg.n_embed)       # output projection
        self.register_buffer('bias',
                             torch.tril( torch.ones((cfg.block_size, cfg.block_size)) ),
                             persistent=False)

    def forward(self, x):
        '''
        Function to compute the forward pass.

        Parameters
        ----------
        x (B, T, C): tensor
            Input tensor.

        Returns
        -------
        x (B, T, C): tensor
            Output tensor.

        '''
        # Initialization:
        nh = self.cfg.head_num
        B, T, C = x.shape
        hs = C // nh

        # Compute the batched attention and then split and rearrange q, k, v:
        qkv = self.c_attn(x)                                                # (B, T, 3 * C)
        q, k, v = qkv.split(C, dim=-1)                                      # 3 * (B, T, C)
        q = q.view(B, T, nh, hs).transpose(2,1)                             # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(2,1)                             # (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(2,1)                             # (B, nh, T, hs)

        # Compute the attention scores:
        attn = ( q @ k.transpose(-1, -2) ) * hs ** (-0.5)                   # (B, nh, T, T)
        attn = attn.masked_fill(self.bias[:T, :T]==0, float('-inf'))        # apply causal mask
        attn = F.softmax(attn, dim=-1)                                      # normalize the rows : (B, nh, T, T)
        y = attn @ v                                                        # compute the weighted sums: (B, nh, T, hs)

        # Reassemble the heads and project:
        y = y.transpose(2, 1).contiguous().view(B, T, C)                    # (B, T, C)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    '''Class to implement the feed-forward projection component.'''
    def __init__(self, cfg: GPTConfig) -> None:
        '''
        Parameters
        ----------
        cfg: GPTConfig
            Model configurations.

        '''
        super().__init__()

        self.c_fc = nn.Linear(cfg.n_embed, 4 * cfg.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * cfg.n_embed, cfg.n_embed)

    def forward(self, x):
        '''
        Function to compute the forward pass.

        Parameters
        ----------
        x (B, T, C): tensor
            Input tensor.

        Returns
        -------
        x (B, T, C): tensor
            Output tensor.

        '''
        x = self.c_fc(x)        # (B, T, C) -> (B, T, 4 * C)
        x = self.gelu(x)        # (B, T, 4 * C)
        x = self.c_proj(x)      # (B, T, 4 * C) -> (B, T, C)
        return x


class Block(nn.Module):
    '''Class to implement a single transformer block.'''
    def __init__(self, cfg: GPTConfig) -> None:
        '''
        Parameters
        ----------
        cfg: GPTConfig
            Model configurations.

        '''
        super().__init__()

        self.ln_1 = nn.LayerNorm(cfg.n_embed)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embed)
        self.mlp = MLP(cfg)

    def forward(self, x):
        '''
        Function to compute the forward pass.

        Parameters
        ----------
        x (B, T, C): tensor
            Input tensor.

        Returns
        -------
        x (B, T, C): tensor
            Output tensor.

        '''
        x = x + self.attn(self.ln_1(x))         # (B, T, C)
        x = x + self.mlp(self.ln_2(x))          # (B, T, C)
        return x


#%% GPT2 class:

class GPT2(nn.Module):
    '''Class to implement the Hugging Face version of the GPT2 model.'''
    def __init__(self, cfg: GPTConfig) -> None:
        '''
        Parameters
        ----------
        cfg: GPTConfig
            Model configurations.

        Attributes
        ----------
        cfg

        '''
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embed),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embed),
            h = nn.ModuleList([ Block(cfg) for _ in range(cfg.n_blocks) ]),
            ln_f = nn.LayerNorm(cfg.n_embed)
        ))
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)


    def forward(self, idx, targets=None):
        '''
        Function to generate logits and optionally compute the cross-entropy loss.

        Parameters
        ----------
        idx (B, T): tensor
            Input token indices.
        targets (B, T): tensor, optional
            Corresponding target token indices. The default is None in which case, loss is not computed.

        Returns
        -------
        logits (B, T, vocab_size): tensor
            Logits corresponding to the token space.
        loss: tensor
            Loss value. None if 'targets' is None.

        Raises
        ------
        AssertionError
            If the sequence length is larger than the context size.

        '''
        # Attributes:
        cfg = self.cfg
        transformer = self.transformer

        # Initialization:
        B, T = idx.shape
        loss = None

        assert T <= cfg.block_size, f"input context longer than block-size: {T} > {cfg.block_size}"

        # Compute the logits for next token predictions:
        pos = torch.arange(T, device=idx.device, dtype=torch.long)  # (T)
        x = transformer.wte(idx) + transformer.wpe(pos)             # (B, T, n_embed)
        for block in transformer.h:
            x = block(x)                                            # (B, T, n_embed)
        x = transformer.ln_f(x)                                     # (B, T, n_embed)
        logits = self.lm_head(x)                                    # (B, T, vocab_size)

        if targets is None: return logits, loss

        # Compute the cross-entropy loss:
        logits = logits.view(B * T, -1)
        targets = targets.view(-1)
        loss = F.cross_entropy(logits, targets)
        logits = logits.view(B, T, -1)

        return logits, loss


    @torch.inference_mode()
    def generate(self, idx, new_tokens, topk=50):
        '''
        Function to generate next tokens given a encoded prompt.

        Parameters
        ----------
        idx (B, T): tensor
            Encoded prompt tokens.
        new_tokens: int
            Number of new tokens to generate.
        topk: int
            Keep only the top_k most likely tokens. The default is '50'.

        Returns
        -------
        idx (B, T + new_tokens): tensor
            Input tokens with new samples concatenated.

        Raises
        ------
        AssertionError
            If the input sequence length plus new requested token number is larger than the context size.

        '''
        # Initialization:
        train_mode = self.training
        self.eval()
        B, T = idx.shape
        block_size = self.cfg.block_size

        assert T + new_tokens - 1 <= block_size, f"requested new tokens exceed block size"

        # Loop over new tokens:
        for _ in range(new_tokens):
            logits, _ = self(idx)                                       # (B, T, vocab_size)
            probs = F.softmax(logits[:, -1, :], dim=-1)                 # (B, vocab_size)
            topk_probs, topk_inds = torch.topk(probs, k=topk, dim=-1)   # (B, topk)
            idx_new = torch.multinomial(topk_probs, 1)                  # (B, 1)
            idx_new = topk_inds.gather(dim=-1, index=idx_new)           # map back to all vocabulary ids
            idx = torch.cat((idx, idx_new), dim=-1)

        if train_mode: self.train()

        return idx


    @classmethod
    def from_pretrained(cls, model_type):
        '''
        Function to load a pre-trained model weights from HF.

        Parameters
        ----------
        model_type: string
            GPT2 model type to load from 'gpt2, gpt2-medium, gpt2-large, gpt2-xl'.

        Returns
        -------
        model: nn.Module
            A class instance with weights copied from the pre-trained HF model.

        Raises
        ------
        ValueError
            If the requested model does not exist.

        '''
        # Error handling:
        if model_type not in model_configs: raise ValueError(f"Unknown model-type '{model_type}' requested!")

        # Load the pre-trained model:
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = {key: val for key, val in model_hf.state_dict().items()
                 if not ( key.endswith('.attn.masked_bias') or key.endswith('.attn.bias') )}

        # Construct the implemented instance:
        cfg = GPTConfig( **model_configs[model_type] )
        model = cls(cfg)
        sd = {key: val for key, val in model.state_dict().items()}

        assert set(sd_hf)==set(sd), f"mismatched state dictionary keys ({set(sd_hf) ^ set(sd)})"

        # Copy over the pre-trained weights:
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for key in sd_hf:
            if any(key.endswith(w) for w in transposed):        # transpose Conv1D weights
                assert sd_hf[key].shape[::-1]==sd[key].shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key].t())
            else:
                assert sd_hf[key].shape==sd[key].shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key])

        return model
