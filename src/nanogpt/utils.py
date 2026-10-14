
#%% Imports:

from contextlib import contextmanager, nullcontext
import copy

import torch
from torch.nn import functional as F
from torch import nn
from torch.profiler import record_function

#%% Global variables:

# Device synchronized at region boundaries (None disables the labels):
_REGIONS_DEVICE: str | None = None

#%% Model utility functions:

def sample_logits(logits: torch.Tensor, temperature=1.0, top_k=None):
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


def expand_kv(k: torch.Tensor, v: torch.Tensor, head_num: int, n_kv_head: int):
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


#%% RoPE utility functions:

def build_rope_cos_sin(head_size: int, rope_base: float, length: int, start: int = 0, device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Function to build cos/sin tables for tokens in rotary positional embedding (RoPE) mode.
    - Each head's 'head_size' dimensions are treated as head_size/2 pairs of 2-D vectors.
    - 2-D vectors are rotated by angle pos x θ_i, where θ_i = base^(-2i/head_size).
    - Operations must be done in 'fp32' since 'bf16' does not have sufficient precision.

    Parameters
    ----------
    head_size: int
        Attention head size.
    rope_base: float
        RoPE base. Determines the rotation angle resolution.
    length: int
        Number of positions.
    start: int, optional
        First position, e.g. the key-value cache length when decoding. The default is 0.
    device: torch.device or str, optional
        Device to build the tables on. The default is None (the current default device).

    Returns
    -------
    cos (length, head_size): tensor
        Cosine of rotation angles to be applied to queries and keys.
    sin (length, head_size): tensor
        Sine of rotation angles to be applied to queries and keys.

    '''
    # Compute frequencies θ_i = base^(−2i/head_size):
    freq = torch.arange(0, head_size, 2, device=device, dtype=torch.float32) / head_size
    inv_freq = 1 / rope_base ** freq                    # (head_size/2)

    # Compute pos x θ_i outer products (entry [m, i] is the angle for pair i at position m):
    pos = torch.arange(start, start + length, device=device, dtype=torch.float32)   # token positions
    angles = pos[:, None] * inv_freq[None, :]           # (length, head_size/2)
    angles = torch.cat([angles, angles], dim=-1)        # (length, head_size)

    # Compute sine and cosine tables for rotation:
    cos = angles.cos()                                  # (length, head_size)
    sin = angles.sin()                                  # (length, head_size)

    return cos, sin


def rotate_half(x: torch.Tensor):
    '''
    Function to split the attention queries and keys into 2D vectors x1 and x2 and rotate them by 90 degrees as [-x2 | x1].

    Parameters
    ----------
    x (B, head_num, T, head_size): tensor
        Attention values from query or key.

    Returns
    -------
    x_rotated (B, head_num, T, head_size): tensor
        Rotated tensor [-x2 | x1].

    '''
    dim = x.size(-1) // 2       # head_size / 2
    x1 = x[..., :dim]
    x2 = x[..., dim:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    '''
    Function to rotate queries or keys by their position angles (RoPE).
    The function takes the angle cosines and sines from 'build_rope_cos_sin' and rotated half of 'x' from 'rotate_half' and computes the elementwise rotation: [x1 * cos - x2 * sin | x1 * sin + x2 * cos] where x = [x1 | x2].

    Parameters
    ----------
    x (B, n_heads, T, head_size): tensor
        Attention queries or keys from query or key.
    cos (T, head_size): tensor
        Cosine of rotation angles to be applied to attention queries or keys.
    sin (T, head_size): tensor
        Sine of rotation angles to be applied to attention queries or keys.

    Returns
    -------
    x_rope (B, n_heads, T, head_size): tensor
        'x' rotated by its position angles.

    '''
    x32 = x.float()             # cast to 'fp32' to ensure sufficient precision for RoPE computations
    x_rope = x32 * cos + rotate_half(x32) * sin
    return x_rope.to(x.dtype)


#%% Data type precision utility functions:

def autocast_ctx(device: str, dtype: str):
    '''Function to auto-cast forward passes for mixed-precision context; no-ops for fp32.'''
    if dtype == 'fp32':
        return nullcontext()
    return torch.autocast(device_type=device, dtype=torch.bfloat16)


def prepare_for_inference(model: nn.Module, mode: str):
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


#%% General utility functions:

def get_param_count(model: nn.Module):
    '''Function to return the total number of trainable parameters in the 'model'.'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#%% Profiler region utilities:

def sync_device(device: str) -> None:
    '''Function to ensure that the tasks are measured and not the queue time.'''
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def enable_regions(device: str | None) -> None:
    '''Function to enable the model's profiler labels for 'device', or disable them with None.'''
    global _REGIONS_DEVICE
    _REGIONS_DEVICE = device


@contextmanager
def _synced_region(name: str, device: str):
    '''Function to label a code region for the profiler, synchronizing the device on entry and exit so the region's time includes its device work.'''
    sync_device(device)                 # drain earlier queued work so it is not attributed to this region
    with record_function(name):
        yield
        sync_device(device)             # wait for this region's own device work


def region(name: str):
    '''Function to return a profiler label for a code region, or a no-op context when labels are disabled.'''
    return nullcontext() if _REGIONS_DEVICE is None else _synced_region(name, _REGIONS_DEVICE)

