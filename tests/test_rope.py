
#%% Imports:

import pytest
import torch

from nanogpt.utils import build_rope_cos_sin, apply_rope

'''
This file contains the code to test the shapes and mathematical formulation of the rotary positional embedding (RoPE).

'''

#%% Utility functions:

def _reference_rope(vec, pos, rope_base):
    '''
    Function to apply RoPE to a row of attention queries or keys corresponding to a token at position 'pos'.

    Parameters
    ----------
    vec (head_size): tensor
        Of attention queries or keys corresponding to a token at position 'pos'
    pos: int
        Token position.
    rope_base: float
        RoPE base.

    Returns
    -------
    vec_rope (head_size): tensor
        'vec' rotated by its position angles.

    '''
    # Compute sine and cosine vectors for the position:
    head_size = vec.size(-1)
    angles = pos / rope_base ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size)
    cos, sin = angles.cos(), angles.sin()          # (head_size/2)

    # Rotate the vector:
    vec1, vec2 = vec[:head_size//2], vec[head_size//2:]
    vec_rope = torch.cat([vec1*cos - vec2*sin, vec1*sin + vec2*cos])

    return vec_rope


def _rotated_qk(head_size, max_len, rope_base):
    '''Function to construct query and keys with identical entries at all position rows and compute their rotations.'''

    torch.manual_seed(0)
    q = torch.randn(head_size, dtype=torch.float32)
    k = torch.randn(head_size, dtype=torch.float32)
    cos, sin = build_rope_cos_sin(head_size, rope_base, max_len)
    q = apply_rope(q.expand((1, 1, max_len, head_size)), cos, sin)
    k = apply_rope(k.expand((1, 1, max_len, head_size)), cos, sin)
    return q, k


#%% Tests:

def test_cos_sin_tables():
    '''Function to test 'cos' and 'sin' tables for shapes, values, and data types.'''

    # Initialization:
    head_size, rope_base, max_len = 16, 1000., 28
    cos, sin = build_rope_cos_sin(head_size, rope_base, max_len)

    # Shapes:
    assert cos.shape == (max_len, head_size)
    assert sin.shape == (max_len, head_size)

    # Cosine and sine values:
    assert torch.all(cos[0] == 1) and torch.all(sin[0] == 0)
    assert torch.allclose(cos[:, 0], torch.cos(torch.arange(max_len).float()))
    assert torch.allclose(sin[:, 0], torch.sin(torch.arange(max_len).float()))

    # Data types:
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_rope_output(dtype):
    '''Function to test the shape, data type, and values for the output of apply_rope() function.'''

    torch.manual_seed(0)

    # Initialization:
    B, head_num, head_size, rope_base, max_len = 5, 6, 16, 1000., 28
    cos, sin = build_rope_cos_sin(head_size, rope_base, max_len)
    x = torch.randn((B, head_num, max_len, head_size), dtype=dtype)
    x_rope = apply_rope(x, cos, sin)

    assert x_rope.shape == (B, head_num, max_len, head_size)
    assert x_rope.dtype == dtype

    # Rotation does not alter norms:
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    assert torch.allclose(x.float().norm(dim=-1), x_rope.float().norm(dim=-1), rtol=tol, atol=tol)

    # RoPE does not alter tokens at position 0:
    assert torch.equal(x[:, :, 0, :], x_rope[:, :, 0, :])


@pytest.mark.parametrize('mns', [(1, 10, 2), (5, 7, 10), (13, 15, 1), (10, 3, 5)])       # max_len = 20
def test_rope_relative_positions(mns):
    '''Function to test that RoPE only depends on the relative positions of tokens.'''
    q, k = _rotated_qk(head_size=16, max_len=20, rope_base=1000.)
    m, n, s = mns
    assert torch.isclose(
        torch.dot(q[0, 0, m], k[0, 0, n]),
        torch.dot(q[0, 0, m+s], k[0, 0, n+s]),
        atol=1e-5)


def test_rope_position_dependent():
    '''Function to test that RoPE depends on positions of the tokens.'''
    q, k = _rotated_qk(head_size=16, max_len=20, rope_base=1000.)
    m, n = 3, 7
    assert torch.abs(torch.dot(q[0, 0, m], k[0, 0, n]) - torch.dot(q[0, 0, m], k[0, 0, n + 5])) > 1e-2


def test_rope_implementation():
    '''Function to test that the implementation of RoPE matches the math in _reference_rope() function.'''

    torch.manual_seed(0)

    # Initialization:
    head_size, max_len, rope_base = 16, 20, 1000.
    v = torch.randn((1, 1, max_len, head_size), dtype=torch.float32)
    cos, sin = build_rope_cos_sin(head_size, rope_base, max_len)
    out = apply_rope(v, cos, sin)

    # Loop over positions and cross-reference with the manual computation:
    for p in range(max_len):
        expected = _reference_rope(v[0, 0, p], p, rope_base)
        assert torch.allclose(out[0, 0, p], expected, atol=1e-5)

