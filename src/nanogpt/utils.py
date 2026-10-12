
#%% Imports:

from contextlib import nullcontext
import copy

import torch
from torch.nn import functional as F
from torch import nn

#%% Global variables:

_DTYPE_NAMES = {torch.float32: 'fp32', torch.bfloat16: 'bf16'}

#%% Utility functions:

def sample_logits(logits, temperature=1.0, top_k=None):
    '''
    Function to sample one new token per sequence from the logits.

    Parameters
    ----------
    logits (B, vocab_size): tensor
        Logits of the last position for each sequence in the batch.
    temperature: float, optional
        Divides the logits before softmax. Values < 1 sharpen the distribution, values > 1 flatten it, and 0 gives greedy (argmax) decoding.
        The default is 1.0, in which case the distribution is unaltered.
    top_k: int, optional
        Keep only the top_k most likely tokens; top_k=1 gives greedy decoding.
        The default is None, in which case all tokens are kept.

    Returns
    -------
    idx_next (B, 1): tensor
        Newly sampled token per sequence in the batch.

    Raises
    ------
    ValueError
        If temperature < 0 or top_k < 1.

    '''
    # Validate inputs:
    if temperature < 0:
        raise ValueError(f"'temperature' must be >= 0, got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"'top_k' must be >= 1 or None, got {top_k}")

    # Greedy decoding:
    if temperature == 0 or top_k == 1:
        return torch.argmax(logits, dim=-1, keepdim=True)               # (B, 1)

    # Scale the logits by temperature (not in place, so the caller's tensor is untouched):
    logits = logits / temperature

    # Assign '-inf' to all logits outside the top_k:
    if top_k is not None and top_k < logits.size(-1):
        kth_largest = torch.topk(logits, top_k, dim=-1).values[:, -1:]  # (B, 1)
        logits = torch.where(logits >= kth_largest, logits, float('-inf'))

    # Probability distribution over tokens:
    probs = F.softmax(logits, dim=-1)                                   # (B, vocab_size)

    # Sample from the distribution:
    idx_next = torch.multinomial(probs, num_samples=1)                  # (B, 1)

    return idx_next


def expand_kv(k, v, head_num, n_kv_head):
    '''
    Function to repeat the keys and values for grouped-query attention (GQA) to make dimensions consistent with the queries.

    Parameters
    ----------
    k (B, n_kv_head, T, head_size): tensor
        Keys to expand.
    v (B, n_kv_head, T, head_size): tensor
        Values to expand.
    head_num: int
        Number of attention heads in a transformer block.
    n_kv_head: int
        Number of key-value heads in a transformer block.

    Returns
    -------
    k (B, head_num, T, head_size): tensor
        Expanded keys.
    v (B, head_num, T, head_size): tensor
        Expanded values.

    '''
    n_rep = head_num // n_kv_head
    if n_rep == 1: return k, v
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)
    return k, v


def autocast_ctx(device: str, dtype: str):
    '''Function to autocase forward passes for mixed-precision context; no-ops for fp32.'''
    if dtype == 'fp32':
        return nullcontext()
    return torch.autocast(device_type=device, dtype=torch.bfloat16)


def prepare_for_inference(model, mode: str):
    '''
    Function to return (model, context) for an inference precision mode.

    Parameters
    ----------
    model: nn.Module
        Model to run inference.
    mode: str
        Mode from 'fp32, mixed_bf16, bf16'.

    Returns
    -------
    model: nn.Module
        With parameters cast into requested precision based on 'mode'.
    context: AbstractContextManager
        Appropriate context manager for the precision casting.

    Raises
    ------
    ValueError
        If an unknown mode is requested.

    '''
    if mode == 'fp32':
        return model, nullcontext()
    if mode == 'mixed_bf16':
        return model, torch.autocast(device_type=next(model.parameters()).device.type,
                                     dtype=torch.bfloat16)
    if mode == 'bf16':
        return copy.deepcopy(model).to(torch.bfloat16), nullcontext()

    raise ValueError(f'Unknown precision mode: {mode}')


def get_active_dtype(model: nn.Module) -> str:
    '''
    Function to extract the compute dtype in effect for a model.

    Parameters
    ----------
    model: nn.Module
        Model whose forward pass will run in the current context.

    Returns
    -------
    dtype: string
        'bf16' inside a bf16 autocast block or for bf16 weights, else 'fp32'.

    Raises
    ------
    ValueError
        If the autocast or weight data-type is unsupported.

    '''
    param = next(model.parameters())

    # Autocast takes precedence over the weight dtype:
    if torch.is_autocast_enabled(param.device.type):
        torch_dtype = torch.get_autocast_dtype(param.device.type)
    else:
        torch_dtype = param.dtype

    # Error handling:
    if torch_dtype not in _DTYPE_NAMES: raise ValueError(f'Unsupported dtype: {torch_dtype}')

    return _DTYPE_NAMES[torch_dtype]
