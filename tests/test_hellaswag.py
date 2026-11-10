'''
This file contains HellaSwag evaluation tests.

'''
#%% Imports:

import pytest
import random
import string
import pandas as pd
import math
from dataclasses import replace

import torch
from torch import nn
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

from nanogpt.config import TrainConfig, ModelConfig
from nanogpt.tokenizers import GPT2Tokenizer
from nanogpt.distributed import DDPInfo
from nanogpt.hellaswag import mk_data_point, mk_dp_batch, mk_dp_batches, get_batch_correctness, eval_hellaswag

#%% Global variables:

LETTERS = string.ascii_lowercase    # ASCII letters
BS, BN = 3, 4                       # batch_size and batch_num

#%% Fixtures:

@pytest.fixture(scope='module')
def df():
    '''Function to generate a dataframe containing fake data in the same format as HellaSwag.'''
    rng = random.Random(0)          # set local random seed
    df0 = [_gen_rand_dp(rng) for _ in range(BS * BN)]
    df0 = pd.DataFrame(df0, columns=['ctx', 'endings', 'label'])
    return df0

@pytest.fixture(scope='module')
def tok():
    return GPT2Tokenizer()

#%% Utility functions:

def _gen_rand_dp(rng):
    '''Function to generate a HellaSwag data-point, i.e., a context, four endings, and a label.'''
    ctx = _gen_rand_text(rng)
    endings = [_gen_rand_text(rng) for _ in range(4)]
    label = str(rng.randint(0, 3))      # labels are strings between 0 and 3
    return ctx, endings, label

def _gen_rand_text(rng):
    '''Function to generate a random string of words.'''
    gen_rand_word = lambda: ''.join( rng.choices(LETTERS, k=rng.randint(1, 10)) )
    return ' '.join([gen_rand_word() for _ in range(rng.randint(5, 15))])

def _same(a, b):
    '''Function to compare tokens/masks (tensors) or labels (lists) for equality.'''
    return torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b

#%% Tokenization and batching tests:

def test_mk_data_point(df, tok):
    '''Function to ensure that each data-point is tokenized and masked as expected.'''

    # Loop over data-point rows:
    for rind in range(len(df)):
        # Tokenize and construct the mask for the data-point:
        row = df.iloc[rind]
        x, mask, label = mk_data_point(row, tok)

        # Context is repeated for all four endings:
        ctx = torch.tensor([tok.encode(row['ctx'])]*4)
        T0 = ctx.shape[1]
        assert torch.equal(x[:, :T0], ctx)

        # Loop over endings and check the leading space and mask:
        for ind, ending in enumerate(row['endings']):
            ending_toks = tok.encode(' ' + ending)      # leading space
            assert torch.equal(x[ind, T0 : T0+len(ending_toks)], torch.tensor(ending_toks))

            assert torch.all(mask[ind, T0 : T0+len(ending_toks)]==1)
            assert torch.all(mask[ind, :T0]==0)
            assert torch.all(mask[ind, T0+len(ending_toks):]==0)

        assert label == int(row['label'])


@pytest.mark.parametrize('batch_size', [3, 5, 6])
def test_mk_dp_batch(df, tok, batch_size):
    '''Function to test that the tokens and mask are correctly constructed across data-points in a batch.'''

    # Tokenize the first 'batch_size' data-points as a batch:
    dps = pd.Series([mk_data_point(df.iloc[ind], tok) for ind in range(batch_size)])
    x, mask, _ = mk_dp_batch(dps)
    T_max = 0

    # Loop over data-points and check tokenization and masking:
    for rind in range(batch_size):
        row = df.iloc[rind]
        ctx = tok.encode(row['ctx'])
        T0 = len(ctx)
        for ind, ending in enumerate(row['endings']):
            ending_toks = tok.encode(' ' + ending)      # leading space
            T = T0 + len(ending_toks)

            assert torch.equal(x[rind, ind, :T], torch.tensor(ctx + ending_toks))
            assert torch.all(x[rind, ind, T:] == 0)

            assert torch.all(mask[rind, ind, T0 : T] == 1)
            assert torch.all(mask[rind, ind, :T0] == 0)
            assert torch.all(mask[rind, ind, T:] == 0)

            if T > T_max: T_max = T

    assert x.shape == (batch_size, 4, T_max)
    assert mask.shape == (batch_size, 4, T_max)


@pytest.mark.parametrize('batch_size', [3])
@pytest.mark.parametrize('batch_num', [2])
def test_mk_dp_batches_random(df, tok, batch_size, batch_num):
    '''Function to test random batch generation for the dataset.'''

    # Batches are fixed for a fixed seed:
    batches1 = mk_dp_batches(df, tok, batch_size, batch_num, seed=12)
    batches2 = mk_dp_batches(df, tok, batch_size, batch_num, seed=12)

    # Check dimensions:
    assert len(batches1) == batch_num
    assert len(batches2) == batch_num
    for ind in range(batch_num):
        for j in range(2):                                  # tokens and masks
            assert batches1[ind][j].shape[:2] == (batch_size, 4)
        assert len(batches1[ind][2]) == batch_size          # labels

    # Compare the tokens, masks, and labels individually across batches:
    assert all(_same(a, b) for b1, b2 in zip(batches1, batches2) for a, b in zip(b1, b2))


    # Batches change when seed changes:
    batches3 = mk_dp_batches(df, tok, batch_size, batch_num, seed=0)
    assert len(batches3) == batch_num
    assert any(not _same(a, b) for b1, b3 in zip(batches1, batches3) for a, b in zip(b1, b3))


def test_mk_dp_batches(df, tok):
    '''Function to test batch generation for the entire dataset (batch_num=None, non-randomized).'''

    # Initialization:
    batch_num = None
    batch_size0 = 3              # uneven batches
    batches = mk_dp_batches(df, tok, batch_size0, batch_num)

    # Check number of batches:
    assert len(batches) == math.ceil(len(df) / batch_size0)

    # Loop over batches:
    for ind, (x, mask, labels) in enumerate(batches):
        batch_size = min(batch_size0, len(df) - ind * batch_size0)

        # Check dimensions:
        for j in range(2):                                  # tokens and masks
            assert batches[ind][j].shape[:2] == (batch_size, 4)
        assert len(batches[ind][2]) == batch_size           # labels

        # Loop over rows in a batch and check that all data-points exist and are tokenized and masked as expected:
        for j in range(batch_size):
            rind = ind * batch_size0 + j
            x0, mask0, label0 = mk_data_point(df.iloc[rind], tok)

            T = x0.shape[1]
            assert torch.equal(x[j, :, :T], x0)
            assert torch.equal(mask[j, :, :T], mask0)
            assert labels[j] == label0


#%% Correctness tests:

class _CorrectnessModel(nn.Module):
    '''Model class with consistent API and correct-by-construction logits.'''
    def __init__(self, vocab_size, xB, labels, correct_logits) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.xB = xB
        self.labels = labels
        self.correct_logits = correct_logits

    def forward(self, x, targets=None):
        '''Function to construct logits with elevated or diminished values in correct endings.'''
        B, T = x.shape
        logits = torch.ones((B, T, self.vocab_size))

        # Loop over endings in the batch and assign high or low logits to the correct ending of each data-point:
        for ind in range(B):
            idp, iend = ind // 4, ind % 4       # data-point and ending indices
            if not iend == self.labels[idp]: continue
            next_toks = self.xB[idp, iend, 1:]  # targets are next tokens of each input token
            logits[ind, torch.arange(T), next_toks] = self.correct_logits

        return logits, None


def test_get_batch_correctness(df, tok):
    '''Function to test selection of the correct ending for data-points given a model that returns logits constructed from the correct endings.'''

    # Initialization:
    vocab_size, device, precision = tok.vocab_size, 'cpu', 'fp32'
    cfg = TrainConfig(model=replace(ModelConfig(), vocab_size=vocab_size), device=device, precision=precision)
    x, mask, labels = mk_dp_batches(df, tok, batch_size=3, batch_num=1)[0]    # one batch of data-points
    x, mask = x.to(device), mask.to(device)

    # Correct prediction:
    model = _CorrectnessModel(vocab_size, x, labels, correct_logits=10).to(device)
    flags = get_batch_correctness(cfg, model, x, mask, labels)
    assert flags.all()

    # Incorrect prediction:
    model = _CorrectnessModel(vocab_size, x, labels, correct_logits=0.1).to(device)
    flags = get_batch_correctness(cfg, model, x, mask, labels)
    assert not flags.any()


def test_length_normalization():
    '''Function to test that get_batch_correctness() selects the correct ending regardless of length by using mean and not sum.'''

    # Initialization:
    vocab_size, device, precision = 20, 'cpu', 'fp32'
    rng = random.Random(0)                          # set local random seed
    get_toks = lambda length: [rng.randint(0, vocab_size-1) for _ in range(length)]

    # Construct the data-point:
    label = 2                                       # correct label index
    ctx = torch.tensor(get_toks(10), dtype=torch.long)
    ctx = ctx.unsqueeze(0).expand((4, -1))  # (4, 10)
    endings = torch.zeros((4, 100), dtype=torch.long)
    for ind in range(4):
        endings[ind, :3] = torch.tensor(get_toks(3))
    endings[label] = torch.tensor(get_toks(100))    # correct long ending
    x = torch.cat([ctx, endings], dim=1)
    x = x[None].to(device)
    assert x.shape == (1, 4, 110)
    labels = [label]

    # Construct the mask:
    mask = torch.zeros((1, 4, 110), device=device, dtype=torch.bool)
    mask[0, :, 10:13] = 1           # mask for incorrect short endings
    mask[0, label, 10:] = 1         # mask for the correct long ending

    # Slightly elevate the logits for the correct long ending (1 -> 1.01):
    cfg = TrainConfig(model=replace(ModelConfig(), vocab_size=vocab_size), device=device, precision=precision)
    model = _CorrectnessModel(vocab_size, x, labels, correct_logits=1.01).to(device)
    flags = get_batch_correctness(cfg, model, x, mask, labels)
    assert flags[0]                 # mean loss of the label is slightly lower, but its sum is much higher

#%% Accuracy evaluation tests:

class _DDPModel(nn.Module):
    '''Model class to construct logits favoring a specific token 'tok_ind'.'''
    def __init__(self, vocab_size, tok_ind) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_ind = tok_ind          # token index to favor

    def forward(self, x, targets=None):
        B, T = x.shape
        logits = torch.ones((B, T, self.vocab_size), device=x.device)
        logits[:, :, self.tok_ind] = 10.
        return logits, None


@pytest.mark.parametrize('parallel', [False, pytest.param(True, marks=pytest.mark.slow)])
@pytest.mark.parametrize('ndp', [1, 10])            # ndp = 1 tests the empty rank 1
def test_eval_hellaswag(df, tok, parallel, ndp, tmp_path_factory):
    '''Function to test that the eval_hellaswag() computes the correct normalized accuracy in DDP and single process modes.'''

    # Initialization:
    vocab_size, tok_ind, device, precision = tok.vocab_size, 10, 'cpu', 'fp32'
    cfg = TrainConfig(model=replace(ModelConfig(), vocab_size=vocab_size), device=device, precision=precision)

    # Modify 'df' for a meaningful deterministic normalized accuracy value:
    batch_size, dp_false = 3, 0
    dp_batches = mk_dp_batches(df[:ndp], tok, batch_size)
    for bind, (x, mask, labels) in enumerate(dp_batches):
        for ind in range(x.size(0)):
            i = bind * batch_size + ind
            if i % 4 > 0:                           # replace all ending tokens for the correct label with 'tok_ind'
                label_true = labels[ind]
                x[ind, label_true, mask[ind, label_true, :]] = tok_ind
            else:                                   # replace all ending tokens for the incorrect label with 'tok_ind'
                label_false = (labels[ind] + 1) % 4
                x[ind, label_false, mask[ind, label_false, :]] = tok_ind
                dp_false += 1
    acc_true = 1 - dp_false / ndp

    # Compute the accuracy on fake HellaSwag dataset:
    if parallel:                                    # spawn worker processes under current test process
        tmp_dir = tmp_path_factory.mktemp('ddp_run')
        world_size = 2
        with mp.Manager() as manager:
            out_dict = manager.dict()
            mp.spawn(_worker, args=(world_size, tmp_dir, cfg, tok_ind, dp_batches, out_dict), nprocs=world_size)
            acc = out_dict['acc']
    else:                       # no DDP
        model = _DDPModel(vocab_size, tok_ind)
        ddp = DDPInfo(device=device)
        acc = eval_hellaswag(cfg, model, dp_batches, ddp)

    assert abs(acc - acc_true) < 1e-6


def _worker(rank, world_size, tmp_dir, cfg, tok_ind, dp_batches, out_dict):
    '''Function to run 'eval_hellaswag' for a DDP rank.'''
    init_process_group('gloo', init_method=f'file://{tmp_dir}/rendezvous', rank=rank, world_size=world_size)
    try:
        model = _DDPModel(cfg.model.vocab_size, tok_ind)
        ddp = DDPInfo(rank=rank, local_rank=rank, world_size=world_size, device=cfg.device)
        acc = eval_hellaswag(cfg, model, dp_batches, ddp)
        if ddp.is_main: out_dict['acc'] = acc
    finally:
        destroy_process_group()
