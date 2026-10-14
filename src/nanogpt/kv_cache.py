
#%% Imports:

import torch

'''
This file contains the code to implement key-value cache for transformer blocks.

'''

#%% Main code:

class KVCache:
    '''Class to implement the key-value cache for transformer blocks.'''
    def __init__(self, n_blocks: int, batch_size: int, n_kv_head: int, max_len: int, head_size: int):
        '''
        Parameters
        ----------
        n_blocks: int
            Number of transformer blocks.
        batch_size: int
            Inference batch size.
        n_kv_head: int
            Number of key and value heads in each block: n_kv_head <= head_num.
        max_len: int
            Maximum context length to cache.
        head_size: int
            Attention head dimension.

        Attributes
        ----------
        batch_size, n_kv_head, max_len, head_size
            Expected cache dimensions, checked on every update.
        len: int
            Number of positions cached so far, the same for every block.
        k, v: list of tensors
            Per-block key and value buffers of shape (batch_size, n_kv_head, max_len, head_size). Each is None until
            the block's first update, which allocates it on the device and with the dtype of the incoming keys.

        '''
        self.batch_size = batch_size
        self.n_kv_head = n_kv_head
        self.max_len = max_len
        self.head_size = head_size
        self.len = 0                                        # positions stored so far (same for every block)
        self.k: list[torch.Tensor | None] = [None] * n_blocks
        self.v: list[torch.Tensor | None] = [None] * n_blocks


    def update(self, block_ind: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Function to write new keys and values at the current position and return everything cached for the block.

        Parameters
        ----------
        block_ind: int
            Transformer block index.
        k (B, n_kv_head, T, head_size): torch.Tensor
            Keys to cache for the block.
        v (B, n_kv_head, T, head_size): torch.Tensor
            Values to cache for the block.

        Returns
        -------
        k (B, n_kv_head, end, head_size): torch.Tensor
            Keys cached so far for the block.
        v (B, n_kv_head, end, head_size): torch.Tensor
            Values cached so far for the block.

        '''
        # Dimension checks, before anything is allocated or written:
        expected = (self.batch_size, self.n_kv_head, self.head_size)
        assert (k.size(0), k.size(1), k.size(3)) == expected, f'Unexpected key shape: {tuple(k.shape)}'
        assert v.shape == k.shape, f'Value shape {tuple(v.shape)} differs from key shape {tuple(k.shape)}'

        start, end = self.len, self.len + k.size(2)
        assert end <= self.max_len, f'KV cache is full: {end} > {self.max_len} positions'

        # Allocate the block's cache on its first update on the device and with the dtype of the keys:
        if self.k[block_ind] is None:
            shape = (self.batch_size, self.n_kv_head, self.max_len, self.head_size)
            self.k[block_ind] = k.new_zeros(shape)
            self.v[block_ind] = v.new_zeros(shape)

        # Write the new positions and return the filled part:
        self.k[block_ind][:, :, start:end] = k
        self.v[block_ind][:, :, start:end] = v
        return self.k[block_ind][:, :, :end], self.v[block_ind][:, :, :end]


    def advance(self, T: int) -> None:
        '''Function to mark T more positions as filled, called once per forward pass, after all blocks are updated.'''
        self.len += T


    def reset(self) -> None:
        '''Function to forget all cached positions; the buffers are kept and reused, not reallocated.'''
        self.len = 0
