
#%% Imports:
from contextlib import nullcontext
import torch

#%% Global variables:

_DTYPE_NAMES = {torch.float32: 'fp32', torch.bfloat16: 'bf16'}

#%% Main code:

def autocast_ctx(device: str, dtype: str):
    '''Function to autocase forward passes for mixed-precision context; no-ops for fp32.'''
    if dtype == 'fp32':
        return nullcontext()
    return torch.autocast(device_type=device, dtype=torch.bfloat16)


def get_active_dtype(device: torch.device) -> str:
    '''
    Function to extract dtype in effect.

    Parameters
    ----------
    device: torch.device
        Device to extract the dtype for.

    Returns
    -------
    dtype: string
        Extracted data type: 'bf16' inside a bf16 autocast block, else 'fp32'.

    Raises
    ------
    ValueError
        If the autocast data-type is unsupported for the device.

    '''
    if not torch.is_autocast_enabled(device.type): return 'fp32'

    # Extract the data type:
    torch_dtype = torch.get_autocast_dtype(device.type)

    # Error handling:
    if torch_dtype not in _DTYPE_NAMES: raise ValueError(f'Unsupported autocast dtype: {torch_dtype}')

    return _DTYPE_NAMES[torch_dtype]
