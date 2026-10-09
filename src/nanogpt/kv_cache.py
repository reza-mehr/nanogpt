
#%% Imports:

import torch

'''
This file contains the code to implement key-value cache for transformer blocks.

'''

#%% Main code:

class KVCache:
    def __init__(self, n_blocks, batch_size, head_num, max_len, head_size, device, dtype=torch.float32):

        shape = (n_blocks, batch_size, head_num, max_len, head_size)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self.len = 0                      # tokens stored so far (same for every layer)


    def update(self, block: int, k: torch.Tensor, v: torch.Tensor):
        '''
        Function to write new keys/values at the current position.

        Parameters
        ----------
        block: int
            Transformer block index.
        k (B, head_num, T, head_size): torch.Tensor
            Keys to cache for the block.
        v (B, head_num, T, head_size): torch.Tensor
            Values to cache for the block.

        Returns
        -------
        k (B, head_num, end, head_size): torch.Tensor
            Keys cached so far for the block.
        v (B, head_num, end, head_size): torch.Tensor
            Values cached so far for the block.

        '''
        # Attributes:
        len = self.len
        max_len = self.max_len

        T = k.shape[2]
        end = len + T
        assert end <= max_len, 'KV cache is full'

        self.k[block, :, :, len:end] = k
        self.v[block, :, :, len:end] = v
        return self.k[block, :, :, :end], self.v[block, :, :, :end]


    def advance(self, T: int) -> None:
        '''Function to mark T more positions as filled, called once per forward pass, after all blocks are updated.'''
        self.len += T
