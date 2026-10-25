
'''
This file contains model tests that ensure expected shapes, uniform initialization, optimization behavior, and causality of the language model.

'''

#%% Imports:

import pytest
import torch
import math

from nanogpt.config import ModelConfig
from nanogpt.model import build_model
from nanogpt.utils import autocast_ctx, prepare_for_inference, get_param_count

#%% Configuration setup for tests:

# Model architecture parameters:
B, T, VOCAB, N_EMBED, HEAD_SIZE = 4, 11, 18, 16, 8

def tiny_model(name, kv_head_num=None, pos_embed='learned', vocab_multiple=1):
    '''Function to construct a tiny model to run tests on.'''
    head_num = 2 if name=='gpt2' else 4         # N_EMBED = head_num * HEAD_SIZE for 'gpt2' model
    cfg = ModelConfig(name=name, vocab_size=VOCAB, pad_vocab_to_multiple=vocab_multiple, block_size=T, n_embed=N_EMBED, head_num=head_num, kv_head_num=kv_head_num, head_size=HEAD_SIZE, n_blocks=2, pos_embed=pos_embed,
                      dropout=0.)       # dropout=0 to permit overfitting in 'test_overfit'
    torch.manual_seed(0)
    return build_model(cfg)

@pytest.fixture(params=[4, 2, 1], ids=['mha', 'gqa', 'mqa'])    # head_num = 4
def kv_head_num(request):
    return request.param

@pytest.fixture(params=['learned', 'rope'])
def pos_embed(request):
    return request.param

@pytest.fixture(params=[1, 16])
def vocab_multiple(request):
    return request.param

#%% Weight tying tests:

def test_weight_tie_gpt2(vocab_multiple):
    '''Function to test that the weights at embedding and final projection layers are tied.'''
    model = tiny_model('gpt2', vocab_multiple=vocab_multiple)
    assert model.transformer.wte.weight.shape == model.lm_head.weight.shape
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()

#%% Shape tests:

@pytest.mark.parametrize('name', ['bigram', 'gpt'])
def test_forward_shapes(name, kv_head_num, pos_embed, vocab_multiple):
    '''Function to test the forward function output shapes.'''

    model = tiny_model(name, kv_head_num, pos_embed, vocab_multiple)
    vocab_size_padded = model.cfg.vocab_size_padded
    idx = torch.randint(0, VOCAB, (B, T))

    logits, loss = model(idx, targets=None)             # no targets, no loss
    assert logits.shape == (B, T, vocab_size_padded)    # logits must included padded vocabulary
    assert loss is None

    logits, loss = model(idx, targets=idx)              # with targets
    assert loss.dim() == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize('name', ['bigram', 'gpt'])
def test_generate_shapes(name, vocab_multiple):
    '''Function to test the generate function output shape.'''

    model = tiny_model(name, vocab_multiple=vocab_multiple)
    idx = torch.zeros((1, 1), dtype=torch.long)

    new_tokens = 10
    out = model.generate(idx, new_tokens, use_cache=False)
    assert out.shape == (1, 1+new_tokens)
    assert (out < VOCAB).all()          # generated token ids must be confined to 'VOCAB' regardless of padding

#%% Initialization test:

@pytest.mark.parametrize('name', ['bigram', 'gpt', 'gpt2'])
def test_uniform_loss(name):
    '''Function to test that the initial model predictions are near uniform.'''

    model = tiny_model(name)
    idx = torch.randint(0, VOCAB, (B, T))

    _, loss = model(idx, targets=idx)
    assert abs(loss.item() - math.log(VOCAB)) < 1.0     # loss must be roughly -ln(1/VOCAB)

#%% Overfit test:

@pytest.mark.parametrize('dtype', ['fp32', 'mixed_bf16'])
def test_overfit(dtype, kv_head_num, pos_embed):
    '''Function to test the optimization loop by ensuring that the model can overfit to a single batch of data.'''

    # Initialize the model, optimizer, and data:
    device = 'cpu'
    torch.manual_seed(0)
    model = tiny_model('gpt', kv_head_num, pos_embed).to(device)
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

def _test_causality(name, t, kv_head_num=None, pos_embed='learned', dtype='fp32'):
    '''Function to test that changing the token at position t must not affect outputs at positions < t.'''

    # Initialization:
    device = 'cpu'
    model = tiny_model(name, kv_head_num, pos_embed).to(device).eval()  # eval: no dropout, deterministic
    model, ctxt = prepare_for_inference(model, dtype)                   # cast the model or forward pass based on 'dtype'
    idx = torch.randint(0, VOCAB, (B, T), device=device)
    idx2 = idx.clone()
    idx2[:, t] = (idx[:, t] + 1) % VOCAB    # guaranteed different token at position 't' for all sequences in the batch

    # Forward pass:
    with torch.no_grad(), ctxt:
        logits, _ = model(idx)
        logits2, _ = model(idx2)

    # Assert that the past is untouched:
    assert torch.equal(logits[:, :t], logits2[:, :t]), f'future token {t} leaked into the past'

    # Assert that the change is visible where it should be, so the test isn't vacuous:
    assert not torch.equal(logits[:, t], logits2[:, t]), f'position {t} ignored its own token'


@pytest.mark.parametrize('t', [0, 3, T - 1])
@pytest.mark.parametrize('dtype', ['fp32', 'mixed_bf16', 'bf16'])
def test_causality_gpt(t, kv_head_num, pos_embed, dtype):
    _test_causality('gpt', t, kv_head_num, pos_embed, dtype)


@pytest.mark.parametrize('t', [0, 3, T - 1])
def test_causality_bigram(t):
    _test_causality('bigram', t)


#%% Batch leak tests:

def test_no_leakage_across_batch(kv_head_num):
    '''Tests that changing one sequence in a batch must not affect the others.'''

    model = tiny_model('gpt', kv_head_num).eval() # eval: no dropout, deterministic
    idx = torch.randint(0, VOCAB, (B, T))
    idx2 = idx.clone()
    idx2[0] = (idx[0] + 1) % VOCAB              # change only sequence 0

    with torch.no_grad():
        logits, _ = model(idx)
        logits2, _ = model(idx2)

    assert torch.equal(logits[1:], logits2[1:])

#%% Mixed-precision compute test:

@pytest.mark.parametrize('dtype', ['mixed_bf16', 'bf16'])
def test_bf16_forward_close_to_fp32(dtype):
    '''Function to ensure that the 'bf16' precision on full cast or forward pass are close to 'fp32'.'''

    # Initialization:
    model = tiny_model('gpt').to('cpu').eval()
    cast, ctxt = prepare_for_inference(model, dtype)        # cast the model or forward pass based on 'dtype'
    idx = torch.randint(0, VOCAB, (B, T))

    with torch.no_grad():
        _, loss32 = model(idx, idx)
        with ctxt:
            _, loss16 = cast(idx, idx)
    assert abs(loss32.item() - loss16.item()) < 0.05


def test_mixed_bf16_keeps_fp32_state():
    '''Function to test that mixed-precision mode does not modify data-type for model weights.'''

    # Initialization:
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


#%% Compile test:

def test_compiled_model():
    '''Function to test that the original and compiled forward passes return consistent logits.'''
    device = 'cpu'
    model = tiny_model('gpt2').to(device)
    model_compiled = torch.compile(model, mode='default').to(device)
    idx = torch.randint(0, VOCAB, (B, T), device=device)

    logits, _ = model(idx)
    logits_compiled, _ = model_compiled(idx)
    torch.testing.assert_close(logits, logits_compiled, atol=1e-6, rtol=0)      # max diff measured at 1e-07


#%% Generation test:

def test_cached_matches_naive_generation(kv_head_num, pos_embed):
    '''Function to test that the cached and non-cached implementation of the generate function return identical results.'''
    model = tiny_model('gpt', kv_head_num, pos_embed).eval()
    idx = torch.randint(0, VOCAB, (1, 1))
    torch.manual_seed(0)
    naive = model.generate(idx, new_tokens=6, temperature=0., use_cache=False)
    torch.manual_seed(0)
    cached = model.generate(idx, new_tokens=6, temperature=0., use_cache=True)
    assert torch.equal(naive, cached)

#%% Trainable parameter tests:

def test_param_count_rope():
    '''Function to test that the trainable parameters for positional embedding are removed for RoPE.'''
    model1 = tiny_model('gpt', pos_embed='learned')
    model2 = tiny_model('gpt', pos_embed='rope')
    assert get_param_count(model1) - get_param_count(model2) == T * N_EMBED

