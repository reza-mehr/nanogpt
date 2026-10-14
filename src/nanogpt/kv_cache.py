
#%% Imports:

import torch

'''
This file contains the code to implement key-value cache for transformer blocks.

'''

#%% Main code:

class KVCache:
    '''Class to implement the key-value cache for transformer blocks.'''
    def __init__(self, n_blocks: int, batch_size: int, n_kv_head: int, max_len: int, head_size: int, device: str, dtype: str = 'fp32'):
        '''
        Parameters
        ----------
        n_blocks: int
            Number of transformer blocks.
        batch_size: int
            Inference batch size.
        n_kv_head: int
            Number of key and value heads in each block: n_kv_head <= heads_num
        max_len: int
            Maximum context legnth to cache.
        head_size: int
            Attention head dimension.
        device: string
            Device to use.
        dtype: string, optional
            The data-type to store keys and values from 'fp32, bf16'. The default is 'fp32'.

        Attributes
        ----------
        n_kv_head
        head_size

        max_len
        len: int
            Current cache context lengths.

        k (n_blocks, batch_size, head_num, max_len, head_size): tensor
            To cache keys for all blocks.
        v (n_blocks, batch_size, head_num, max_len, head_size): tensor
            To cache values for all blocks.

        '''
        self.n_kv_head = n_kv_head
        self.head_size = head_size
        self.max_len = max_len
        self.len = 0                      # tokens stored so far (same for every layer)

        shape = (n_blocks, batch_size, n_kv_head, max_len, head_size)
        dtype = torch.bfloat16 if dtype == 'bf16' else torch.float32
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)


    def update(self, block: int, k: torch.Tensor, v: torch.Tensor):
        '''
        Function to write new keys/values at the current position.

        Parameters
        ----------
        block: int
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
        # Attributes:
        n_kv_head = self.n_kv_head
        head_size = self.head_size
        len = self.len
        max_len = self.max_len

        # Dimension assertions:
        assert k.size(1)==n_kv_head and k.size(3)==head_size, f"Unexpected key shape: {k.shape}"
        assert v.size(1)==n_kv_head and v.size(3)==head_size, f"Unexpected value shape: {v.shape}"

        T = k.shape[2]
        end = len + T
        assert end <= max_len, 'KV cache is full'

        self.k[block, :, :, len:end] = k
        self.v[block, :, :, len:end] = v
        return self.k[block, :, :, :end], self.v[block, :, :, :end]


    def advance(self, T: int) -> None:
        '''Function to mark T more positions as filled, called once per forward pass, after all blocks are updated.'''
        self.len += T


    def reset(self):
        '''Function to reset the cache length. Forget all cached positions, buffers are reused, not reallocated.'''
        self.len = 0
