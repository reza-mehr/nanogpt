'''
This file contains the tests for the implementation of the GPT2 model with Hugging Face (HF) weight names.
The tests in this module require downloading the HF reference model weights and are excluded from CI tests in pyproject.toml. TODO: cache weights on Github and include these tests in CI.

'''
#%% Imports:

import copy
import pytest
import torch
import tiktoken
from dataclasses import replace

pytest.importorskip('transformers')
from transformers import GPT2LMHeadModel    # noqa: E402

from nanogpt.config import gpt2_cfg         # noqa: E402
from nanogpt.model import GPT2              # noqa: E402

#%% Global variables:

DEVICE = 'cpu'
ATOL = 1e-3         # measured max abs diff: 1e-4
RTOL = 1e-5         # measured max abs diff / max |logit|: 6.8e-7
LOSS_RTOL = 3e-5    # measured |Δloss| / loss: 2.3e-6

#%% Fixtures:

@pytest.fixture(scope='module')
def tokens():
    '''Function to tokenize a prompt for the tests in the module.'''
    tok = tiktoken.get_encoding('gpt2')
    prompt = "Hello, I'm a language model,"
    idx = tok.encode(prompt)                                # (T,)
    idx = torch.tensor(idx, dtype=torch.long, device=DEVICE)
    return idx.unsqueeze(0)                                 # (1, T)


@pytest.fixture(scope='module')
def model_hf():
    '''Function to load the HF reference once per module.'''
    model_type = 'gpt2'
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)  # Hugging Face model
    model_hf.to(DEVICE).eval()
    return model_hf


@pytest.fixture(scope='module', params=[False, True], ids=['manual', 'sdpa'])
def model_built(request):
    '''Function to construct the GPT2 model once per module for every combination of the parameters.'''
    cfg = replace(gpt2_cfg.model, scaled_dot_prod=request.param)
    model = GPT2.from_pretrained(cfg)                       # loaded model
    model.to(DEVICE).eval()
    return model

#%% Unit tests:

def test_logit_match_pretrained(model_hf, model_built, tokens):
    '''Function to test that logits match Hugging Face's GPT-2 on the same input.'''
    with torch.no_grad():
        logits_hf = model_hf(tokens).logits
        logits, _ = model_built(tokens)
    torch.testing.assert_close(logits, logits_hf, rtol=RTOL, atol=ATOL)


def test_logits_change_with_params(model_hf, model_built, tokens):
    '''Function to test that the loaded model with perturbed weights does not match the reference HF model.'''

    torch.manual_seed(0)
    model_built = copy.deepcopy(model_built)            # keep the shared fixture intact

    with torch.no_grad():
        # Perturb the attention output projection for the copied model:
        w = model_built.transformer.h[1].attn.c_proj.weight

        w.add_(0.02 * torch.randn_like(w))              # std = 0.02
        logits, _ = model_built(tokens)
        logits_hf = model_hf(tokens).logits

    assert(logits - logits_hf).abs().max() > 10 * ATOL


def test_loss_match_pretrained(model_hf, model_built, tokens):
    '''Function to test that loss value matches Hugging Face's GPT-2 on the same input.'''

    # Initialization:
    T = tokens.size(-1)
    idx, targets = tokens[:, :T-1], tokens[:, 1:T]

    # Compute the forward passes:
    with torch.no_grad():
        loss_hf = model_hf(tokens, labels=tokens).loss  # internally shifts the tokens to get target (ignores last tokens)
        _, loss = model_built(idx, targets)

    torch.testing.assert_close(loss, loss_hf, atol=0, rtol=LOSS_RTOL)
