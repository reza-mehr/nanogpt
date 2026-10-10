
#%% Imports:

import pytest
import torch
import math
import copy

from nanogpt.config import ModelConfig
from nanogpt.registry import build_model
import nanogpt.model        # noqa: F401  (registers the models)
from nanogpt.utils import autocast_ctx, prepare_for_inference, get_active_dtype

'''
This file contains model tests that ensure expected shapes, uniform initialization, optimization behavior, and causality of the language model.

'''

#%% Configuration setup for tests:

# Model architecture parameters:
B, T, VOCAB = 4, 8, 18

def tiny_model(name):
    '''Function to construct a tiny model to run tests on.'''
    cfg = ModelConfig(name=name, vocab_size=VOCAB, block_size=T, n_embed=16, head_num=2, head_size=8, n_blocks=2,
                      dropout=0.)       # set dropout to 0 to permit overfitting in one of the tests below
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

@pytest.mark.parametrize('dtype', ['fp32', 'mixed_bf16'])
def test_overfit(dtype):
    '''Function to test the optimization loop by ensuring that the model can overfit to a single batch of data.'''

    # Initialize the model, optimizer, and data:
    device = 'cpu'
    torch.manual_seed(0)
    model = tiny_model('gpt').to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    idx = torch.randint(0, VOCAB, (B, T), device=device)
    targets = torch.randint(0, VOCAB, (B, T), device=device)
    with torch.no_grad(), autocast_ctx(device, dtype):
        _, loss0 = model(idx, targets)

    # Overfit to the single batch:
    for _ in range(300):
        with autocast_ctx(device, dtype):
            _, loss = model(idx, targets)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    assert loss.item() < 0.1 * loss0.item(), \
        f'loss fell less than 90% when overfitting to a single batch: {loss0.item():.4f} -> {loss.item():.4f}'

#%% Causality tests:

@pytest.mark.parametrize('name', ['bigram', 'gpt'])
@pytest.mark.parametrize('t', [0, 3, T - 1])
@pytest.mark.parametrize('dtype', ['fp32', 'mixed_bf16', 'bf16'])
def test_causality(name, t, dtype):
    '''Tests that changing the token at position t must not affect outputs at positions < t.'''

    device = 'cpu'
    model = tiny_model(name).to(device).eval()  # eval: no dropout, deterministic
    idx = torch.randint(0, VOCAB, (B, T), device=device)
    idx2 = idx.clone()
    idx2[:, t] = (idx[:, t] + 1) % VOCAB        # guaranteed different token at position 't' for all sequences in the batch

    with torch.no_grad(), autocast_ctx(device, dtype):
        logits, _ = model(idx)
        logits2, _ = model(idx2)

    # Assert that the past is untouched:
    assert torch.equal(logits[:, :t], logits2[:, :t]), f'future token {t} leaked into the past'

    # Assert that the change is visible where it should be, so the test isn't vacuous:
    assert not torch.equal(logits[:, t], logits2[:, t]), f'position {t} ignored its own token'


def test_no_leakage_across_batch():
    '''Tests that changing one sequence in a batch must not affect the others.'''

    model = tiny_model('gpt').eval()            # eval: no dropout, deterministic
    idx = torch.randint(0, VOCAB, (B, T))
    idx2 = idx.clone()
    idx2[0] = (idx[0] + 1) % VOCAB              # change only sequence 0

    with torch.no_grad():
        logits, _ = model(idx)
        logits2, _ = model(idx2)

    assert torch.equal(logits[1:], logits2[1:])

#%% Mixed-precision compute test:

def test_bf16_forward_close_to_fp32():
    '''Function to ensure that the 'bf16' precision on forward pass is close to 'fp16'.'''

    # Initialization:
    model = tiny_model('gpt').to('cpu').eval()
    idx = torch.randint(0, VOCAB, (B, T))

    with torch.no_grad():
        _, loss32 = model(idx, idx)
        with autocast_ctx('cpu', 'bf16'):
            _, loss16 = model(idx, idx)
    assert abs(loss32.item() - loss16.item()) < 0.05


def test_mixed_bf16_keeps_fp32_state():
    '''Function to test that mixed-precision mode does not modify model weights.'''

    # Initializaiton:
    model = tiny_model('gpt')
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, VOCAB, (B, T))

    # Forward pass:
    with autocast_ctx('cpu', 'mixed_bf16'):
        _, loss = model(x, x)
    loss.backward()
    opt.step()

    for p in model.parameters():
        assert p.dtype == torch.float32 and p.grad.dtype == torch.float32
    for state in opt.state.values():
        assert state['exp_avg'].dtype == torch.float32


def test_bf16_cast_leaves_original_untouched():
    '''Function to test that prepare_for_inference() does not cast the original model weights.'''

    model = tiny_model('gpt')
    before = {k: v.clone() for k, v in model.state_dict().items()}
    cast, _ = prepare_for_inference(model, 'bf16')

    assert all(p.dtype == torch.bfloat16 for p in cast.parameters())
    for k, v in model.state_dict().items():
        assert v.dtype == before[k].dtype and torch.equal(v, before[k])


def test_get_active_dtype():
    '''Function to test that the get_active_dtype() returns the correct data type.'''

    model = tiny_model('gpt')
    assert get_active_dtype(model) == 'fp32'

    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert get_active_dtype(model) == 'bf16'
    assert get_active_dtype(copy.deepcopy(model).to(torch.bfloat16)) == 'bf16'

    with pytest.raises(ValueError):
        get_active_dtype(copy.deepcopy(model).to(torch.float16))        # fp16 is not an acceptable type
