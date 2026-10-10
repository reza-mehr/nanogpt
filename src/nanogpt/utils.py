
#%% Imports:

from contextlib import nullcontext
import copy

import torch
from torch import nn

#%% Global variables:

_DTYPE_NAMES = {torch.float32: 'fp32', torch.bfloat16: 'bf16'}

#%% Main code:

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
