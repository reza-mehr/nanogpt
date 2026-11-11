'''
This file contains the code and utility relevant for distributed data parallelism (DDP).

'''

#%% Imports:

import os
import socket
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from nanogpt.config import TrainConfig

#%% DDPInfo class:

@dataclass
class DDPInfo:
    rank: int = 0                   # process rank across nodes
    local_rank: int = 0             # process rank in the node
    world_size: int = 1             # total number of processes, 1 when DDP is not used
    device: str | None = None       # 'cuda:{local_rank}' for 'cuda', the same as TrainConfig.device otherwise

    @property
    def is_active(self):
        return self.world_size > 1

    @property
    def is_main(self):
        return self.rank == 0

    @classmethod
    def setup(cls, device0: str):
        '''
        Class method to construct a DDPInfo object from environment variables at runtime.

        Parameters
        ----------
        device0: str
            Device requested by the user in TrainConfig.

        Returns
        -------
        obj: DDPInfo
            Class instance constructed from environment variables.

        Raises
        ------
        ValueError
            If DDP is requested with 'device=cuda' but no GPUs are available.
            If DDP is requested with impermissible device, e.g., 'mps'.

        '''
        # Environment variables set by 'torchrun':
        world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        if world_size == 1: return cls(device=device0)      # not DDP
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])

        if device0 == 'cuda':
            if not torch.cuda.is_available(): raise ValueError("Cannot use 'cuda' as device, unavailable!")
            device = f'cuda:{local_rank}'
            init_process_group(backend='nccl', device_id=torch.device(device))
        elif device0 == 'cpu':
            device = 'cpu'
            init_process_group(backend='gloo')
        else:
            raise ValueError(f"DDP unavailable for device={device0!r}.")

        return cls(rank=rank, local_rank=local_rank, world_size=world_size, device=device)

    def teardown(self):
        '''Function to tear-down DDP process at the end.'''
        if self.is_active: destroy_process_group()


#%% Utility functions:

def wrap_model(cfg: TrainConfig, model: nn.Module, ddp: DDPInfo):
    '''
    Function to wrap the model in DDP object and compile it.
    The default is to compile outside DDP so Dynamo can split the graph at bucket boundaries and overlap communication with backward.

    Parameters
    ----------
    cfg: TrainConfig
        Configurations to use.
    model: nn.Module
        Model to train.
    ddp: DDPInfo
        Distributed data parallelism instance.

    Returns
    -------
    model_wrapped: nn.Module
        Model to train potentially wrapped in DDP object and compiled depending on the configurations.

    '''
    def _compile_wrap(model):
        return torch.compile(model) if cfg.compile_model else model
    def _ddp_wrap(model):
        if not ddp.is_active: return model
        device_ids = [ddp.local_rank] if cfg.device == 'cuda' else None
        return DDP(model, device_ids=device_ids)
    return _compile_wrap(_ddp_wrap(model)) if cfg.compile_order == 'after_ddp' else _ddp_wrap(_compile_wrap(model))


def free_port(host: str = '127.0.0.1') -> int:
    '''
    Function to find a free TCP port on 'host', e.g., for 'MASTER_PORT'.
    The port is released before returning, so another process could take it before it is used. That race is acceptable for tests and local runs.

    Parameters
    ----------
    host: str, optional
        Address to bind to. The default is the IPv4 loopback '127.0.0.1'.

    Returns
    -------
    port: int
        Port number the OS reported as free.

    '''
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))                   # port 0: the OS assigns any free port
        return s.getsockname()[1]
