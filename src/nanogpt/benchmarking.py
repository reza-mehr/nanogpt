
#%% Imports:

import torch

from nanogpt.config import ModelConfig
from nanogpt.registry import build_model
import nanogpt.model  # noqa: F401

'''
This file contains the utility functions used in benchmarking.

'''

#%% Utility functions:

def make_step(model_cfg: ModelConfig, device: str, batch_size: int, T: int):
    '''Function to build a model, optimizer, and fixed batch and return a function that conducts one training step.'''

    # Initialization:
    torch.manual_seed(0)
    model = build_model(model_cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    x = torch.randint(0, model_cfg.vocab_size, (batch_size, T), device=device)

    def step(measure_after_forward: bool = False):
        '''One optimization step; optionally return memory right after forward (MPS).'''
        _, loss = model(x, x)
        opt.zero_grad(set_to_none=True)
        mem = torch.mps.current_allocated_memory() / 2**20 if measure_after_forward else None
        loss.backward()
        opt.step()
        return mem

    return step


def sync_device(device: str) -> None:
    '''Function to ensure that the tasks are measured and not the queue time.'''
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
