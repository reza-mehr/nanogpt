
'''
This file contains the code to generate batches from a tokenized text dataset.

'''

#%% Imports:

from pathlib import Path
import torch
import numpy as np
import json

from nanogpt.config import TrainConfig
from nanogpt.distributed import DDPInfo

#%% Data loader:

class TokenSplit:
    '''Class to implement split interface to a tokenized split on disk (flat array of token ids), read via memmap.'''

    def __init__(self, path: str, n_tokens: int, dtype: str = '<u2', in_memory: bool | None = None):
        '''
        Parameters
        ----------
        path: string
            Path to the binary split file.
        n_tokens: int
            Number of tokens in the split.
        dtype: str, optional
            Data type for the binary split. The default is '<u2' indicating 2 bytes per token id in little endian convention.
        in_memory: bool, optional
            Keep small datasets in RAM if True. The default is None in which case, files under 512 MiB are kept in RAM.

        Attributes
        ----------
        path
        dtype
        n_tokens: int
            Number of tokens in the split file.

        '''
        self.path = Path(path)
        self.dtype = np.dtype(dtype)

        n_tokens_file = self.path.stat().st_size // self.dtype.itemsize
        if not n_tokens == n_tokens_file: raise ValueError(f"expected {n_tokens} tokens, got {n_tokens_file}: {path}")
        self.n_tokens = n_tokens

        # Keep small files in RAM, memmap large ones:
        if in_memory is None:
            in_memory = self.path.stat().st_size < 512 * 2**20     # under 512 MiB
        self._array = np.fromfile(self.path, dtype=self.dtype) if in_memory else None


    def tokens(self):
        '''Function to open a fresh read-only memmap of the split or return the in-memory array for small datasets.'''
        return self._array if self._array is not None else np.memmap(self.path, dtype=self.dtype, mode='r')


def load_splits(data_dir) -> tuple[TokenSplit, TokenSplit, dict]:
    '''
    Function to open the train/val splits and metadata written by a prepare script.

    Parameters
    ----------
    data_dir: string
        Path to the folder containing the binary split file.

    Returns
    -------
    train_data: TokenSplit
        Training data.
    val_data: TokenSplit
        Validation data.
    meta: dict
        Split metadata.

    '''
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / 'meta.json').read_text())
    dtype = meta.get('dtype', '<u2')
    train_data = TokenSplit(data_dir / 'train.bin', meta['n_tokens']['train'], dtype)
    val_data = TokenSplit(data_dir / 'val.bin', meta['n_tokens']['val'], dtype)
    return train_data, val_data, meta


def get_batch(split: TokenSplit, cfg: TrainConfig, ddp: DDPInfo | None = None, step: int | None = None, substep: int | None = None):
    '''
    Function to generate cfg.batch_size batches.

    Parameters
    ----------
    split: TokenSplit
        Data to generate a batch from.
    cfg: TrainConfig
        Training configurations to use.
    ddp: DDPInfo, optional
        Distributed data parallelism instance. The default is None.
    step: int, optional
        Current step. The default is None.
    substep: int, optional
        Current substep within the step. The default is None.

    Returns
    -------
    x (B, T): tensor
        Input batch.
    y (B, T): tensor
        Target batch. 'y' is 'x' shifted by one token.

    '''
    if cfg.batch_mode == 'random':
        return get_batch_rand(split, cfg)
    elif cfg.batch_mode == 'sequential':
        if None in [ddp, step, substep]: raise ValueError("'ddp, step, substep' arguments are required!")
        return get_batch_seq(split, cfg, ddp, step, substep)
    else:
        raise ValueError(f"Unknown 'batch_mode={cfg.batch_mode!r}' requested!")


def get_batch_seq(split: TokenSplit, cfg: TrainConfig, ddp: DDPInfo, step: int, substep: int):
    '''
    Function to sample cfg.batch_size sequential windows.

    Parameters
    ----------
    split: TokenSplit
        Data to generate a batch from.
    cfg: TrainConfig
        Training configurations to use.
    ddp: DDPInfo
        Distributed data parallelism instance.
    step: int
        Current step (indexed from zero).
    substep: int
        Current substep within the step (indexed from zero).

    Returns
    -------
    x (B, T): tensor
        Input batch.
    y (B, T): tensor
        Target batch. 'y' is 'x' shifted by one token.

    '''
    # Fresh memmap per batch:
    data = split.tokens()

    # Initialization:
    B, T, ga_iters = cfg.batch_size, cfg.model.block_size, cfg.get_grad_accum_iters(ddp.world_size)

    # Read a chunk that spans 'B * T + 1' tokens since the last read:
    ix = B * T * ( (step * ga_iters + substep) * ddp.world_size + ddp.rank )
    idx = np.arange(ix, ix + B * T + 1) % len(data)     # wrap around for indices larger than len(data)
    chunk = torch.from_numpy(data[idx].astype(np.int64))
    x, y = chunk[:-1].view(B, T).contiguous(), chunk[1:].view(B, T).contiguous()

    if cfg.device == 'cuda':                            # async copy from pinned memory
        return (x.pin_memory().to('cuda', non_blocking=True),
                y.pin_memory().to('cuda', non_blocking=True))
    return x.to(cfg.device), y.to(cfg.device)


def get_batch_rand(split: TokenSplit, cfg: TrainConfig):
    '''
    Function to sample cfg.batch_size random windows.

    Parameters
    ----------
    split: TokenSplit
        Data to generate a batch from.
    cfg: TrainConfig
        Training configurations to use.

    Returns
    -------
    x (B, T): tensor
        Input batch.
    y (B, T): tensor
        Target batch. 'y' is 'x' shifted by one token.

    '''
    # Fresh memmap per batch:
    data = split.tokens()

    # Load a random chunk of token indices for the batch:
    T = cfg.model.block_size
    ix = torch.randint(len(data) - T, (cfg.batch_size,)).numpy()
    idx = ix[:, None] + np.arange(T + 1)                # (B, T+1) positions to read

    # Construct the input and target (input shifted by one token):
    chunk = torch.from_numpy(data[idx].astype(np.int64)) # one gather, one copy
    x, y = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()

    if cfg.device == 'cuda':                            # async copy from pinned memory
        return (x.pin_memory().to('cuda', non_blocking=True),
                y.pin_memory().to('cuda', non_blocking=True))
    return x.to(cfg.device), y.to(cfg.device)
