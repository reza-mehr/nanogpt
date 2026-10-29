'''
This file contains the code and utility relevant for distributed data parallelism (DDP).

'''

#%% Imports:

import os
import torch
from torch.distributed import init_process_group, destroy_process_group
from dataclasses import dataclass

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
