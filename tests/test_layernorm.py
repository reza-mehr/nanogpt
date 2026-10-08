
#%% Imports:
import torch
import torch.nn as nn

from nanogpt.model import LayerNorm     # manual implementation

'''
This code tests the manual implementation of the layer norm against PyTorch implementation.

'''

#%% Main code:

def test_manual_layernorm_matches_torch():
    '''Function to test that the outputs and gradients of manual and PyTorch implementations layer-norm match.'''

    # Initialization:
    torch.manual_seed(0)
    C = 32                                      # channel (embedding) dimension
    manual, ref = LayerNorm(C), nn.LayerNorm(C)

    # Random scale/shift, so the affine part is actually tested (defaults are 1 and 0):
    with torch.no_grad():
        manual.weight.copy_(torch.randn(C))
        manual.bias.copy_(torch.randn(C))
        ref.weight.copy_(manual.weight)
        ref.bias.copy_(manual.bias)

    # Input tensor:
    x = torch.randn(4, 8, C) * 3 + 2           # nonzero mean, non-unit variance
    x_ref = x.clone().requires_grad_(True)
    x = x.requires_grad_(True)

    # Compare outputs:
    out, out_ref = manual(x), ref(x_ref)
    assert torch.allclose(out, out_ref, atol=1e-5)

    # Gradients must match too since the backward pass is part of the behavior:
    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5)
    assert torch.allclose(manual.weight.grad, ref.weight.grad, atol=1e-5)
    assert torch.allclose(manual.bias.grad, ref.bias.grad, atol=1e-5)
