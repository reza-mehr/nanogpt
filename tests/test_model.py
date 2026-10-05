
#%% Imports:

import pytest
import torch
import math

from nanogpt.config import ModelConfig, TrainConfig
from nanogpt.registry import build_model
import nanogpt.model        # noqa: F401  (registers the models)

'''
This file contains model tests that ensure expected shapes and optimization behavior.

'''

#%% Configuration setup for tests:

# Model architecture parameters:
B, T, VOCAB = 4, 8, 18

def tiny_model(name):
    '''Function to construct a tiny model to run tests on.'''
    cfg = TrainConfig(
        model=ModelConfig(name=name, vocab_size=VOCAB, block_size=T, n_embed=16, head_num=2, head_size=8, n_blocks=2,
                          dropout=0.),              # set dropout to 0 to permit overfitting in one of the tests below
        batch_size=B, device='cpu'
        )
    torch.manual_seed(0)
    return build_model(cfg)

#%% Shape tests:

@pytest.mark.parametrize('name', ['bigram', 'gpt'])
def test_forward_shapes(name):
    '''Function to test the forward function output shapes.'''

    model = tiny_model(name)
    idx = torch.randint(0, VOCAB, (B, T))

    logits, loss = model(idx, targets=None)         # no targets, no loss
    assert logits.shape == (B, T, VOCAB)
    assert loss is None

    logits, loss = model(idx, targets=idx)          # with targets
    assert loss.dim() == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize('name', ['bigram', 'gpt'])
def test_generate_shapes(name):
    '''Function to test the generate function output shape.'''

    model = tiny_model(name)
    idx = torch.zeros((1, 1), dtype=torch.long)

    new_tokens = 10
    out = model.generate(idx, new_tokens)
    assert out.shape == (1, 1+new_tokens)

#%% Initialization test:

@pytest.mark.parametrize('name', ['bigram', 'gpt'])
def test_uniform_loss(name):
    '''Function to test that the model parameters are initialized in a uniformly random way.'''

    model = tiny_model(name)
    idx = torch.randint(0, VOCAB, (B, T))

    _, loss = model(idx, targets=idx)
    assert abs(loss.item() - math.log(VOCAB)) < 1.0     # loss must be roughly -ln(1/VOCAB)

#%% Overfit test:

def test_overfit():
    '''Function to test the optimization loop by ensuring that the model can overfit to a single batch of data.'''

    # Initialize the model, optimizer, and data:
    model = tiny_model('gpt')
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    idx = torch.randint(0, VOCAB, (B, T))
    targets = torch.randint(0, VOCAB, (B, T))
    _, loss0 = model(idx, targets)

    # Overfit to the single batch:
    for _ in range(300):
        _, loss = model(idx, targets)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    assert loss.item() < 0.1*loss0.item(), \
        f"loss fell less than 90% when overfitting to a single batch: {loss0.item():.4f} -> {loss.item():.4f}"
