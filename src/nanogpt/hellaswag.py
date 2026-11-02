'''
This file contains the code to retrieve and prepare the HellaSwag dataset for evaluation.

'''
#%% Imports:

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from nanogpt.config import TrainConfig
from nanogpt.tokenizers import Tokenizer
from nanogpt.distributed import DDPInfo
from nanogpt.utils import autocast_ctx

#%% Global variables:

HASH = '382ffb520660cc2ba98092cd8aa07cafc9c47f6d'       # pin down specific version for reproducibility

#%% Utility functions:

def get_dataset(out_dir='data/hellaswag') -> str:
    '''
    Function to sync the validation split of the HellaSwag dataset.

    Parameters
    ----------
    out_dir: str, optional
        Directory to download the dataset to. The default is 'data/hellaswag'.

    Returns
    -------
    dl_path: str
        Absolute path to where the downloaded parquet file.

    '''
    # Sync the dataset:
    dl_path = hf_hub_download(repo_id='Rowan/hellaswag', repo_type='dataset', revision=HASH, filename='default/validation/0000.parquet', local_dir=out_dir)

    return dl_path


def load_dataset(dl_path: str) -> pd.DataFrame:
    '''
    Function to load the HellaSwag dataset as a dataframe.

    Parameters
    ----------
    dl_path: str
        Absolute path to where the downloaded parquet file.

    Returns
    -------
    df: pd.DataFrame
        Dataframe containing the following columns: ind, activity_label, ctx_a, ctx_b, ctx, endings, source_id, split, split_type, label

    '''
    # Read the dataset into a dataframe:
    df = pd.read_parquet(dl_path)
    assert len(df) == 10042, f"Corrupt HellaSwag validation dataset, expected 10042 data points, got {len(df)}."

    return df


def mk_data_point(row: pd.Series, tok: Tokenizer) -> tuple[torch.Tensor, torch.Tensor, int]:
    '''
    Function to take a row of dataframe corresponding to a data point and construct arrays the model can consume.

    Parameters
    ----------
    row: pd.Series
        Data point to tokenize.
    tok: Tokenizer
        Tokenizer to use.

    Returns
    -------
    x (4, T): torch.Tensor
        Tokenized context + ' ' + endings for each of 4 choices.
    mask (4, T): torch.Tensor
        Boolean mask containing only the tokens corresponding to the endings.
    label: int
        Correct ending index.

    '''
    # Tokenize the context and endings:
    ctx = tok.encode(row['ctx'])
    endings = [tok.encode(' ' + ending) for ending in row['endings']]

    # Initialization:
    label = int(row['label'])
    n = len(ctx)
    T = n + max([len(ending) for ending in endings])
    x, mask = torch.zeros((4, T), dtype=torch.long), torch.zeros((4, T), dtype=torch.bool)
    x[:, :n] = torch.tensor(ctx)

    # Pad with zeros and make the mask:
    for ind, ending in enumerate(endings):
        ne = len(ending)
        x[ind, n:n+ne] = torch.tensor(ending)
        mask[ind, n:n+ne] = 1

    return x, mask, label


def mk_dp_batch(dps: pd.Series) -> tuple[torch.Tensor, torch.Tensor, list]:
    '''
    Function to adjust context lengths and batch data-points.

    Parameters
    ----------
    dps: pd.Series
        Each entry is a tuple of (x, mask, label) for a data-point; see mk_data_point().

    Returns
    -------
    x (B, 4, T): tensor
        Batched tokens.
    mask (B, 4, T): tensor
        Batched boolean masks.
    labels: list of ints
        Labels for data-points in the batch.

    '''
    # Initialization:
    batch_size = len(dps)                                # number of data-points in the batch
    T_batch = max([dp[0].size(1) for dp in dps])         # maximum context lengths among data points of the batch
    x = torch.zeros((batch_size, 4, T_batch), dtype=torch.long)
    mask = torch.zeros((batch_size, 4, T_batch), dtype=torch.bool)
    labels = [0] * batch_size

    # Loop over data points and pad them:
    for ind, dp in enumerate(dps):
        T = dp[0].size(1)
        x[ind, :, :T] = dp[0]
        mask[ind, :, :T] = dp[1]
        labels[ind] = dp[2]

    return x, mask, labels


def mk_dp_batches(df: pd.DataFrame, tok: Tokenizer, batch_size: int, batch_num: int | None = None) -> list:
    '''
    Function to generate inputs, masks, and labels of 'batch_size'.

    Parameters
    ----------
    df: pd.DataFrame
        HellaSwag dataset.
    tok: Tokenizer
        Tokenizer to use.
    batch_size: int
        Desired batch size.
    batch_num: int, optional
        Number of batches to construct. The default is None in which case, all available data-points are divided into batches of the requested size.

    Returns
    -------
    out: list of tuples
        Each entry contains a tuple:
        x (batch_size, 4, T): tensor
            Tokenized context + endings for data-points in the batch.
        mask (batch_size, 4, T): tensor
            Mask for tokens corresponding to the endings in all data-points.
        labels (batch_size): list of ints
            Labels corresponding to data-points in the batch.

    '''
    # Tokenize all data points:
    dps = df.apply(lambda row: mk_data_point(row, tok), axis=1)

    # Divide the data-points into batches:
    out = []
    for ind in range(0, len(dps), batch_size):
        out.append( mk_dp_batch(dps[ind: ind+batch_size]) )
        if batch_num and len(out) == batch_num: break

    return out


def get_batch_correctness(cfg: TrainConfig, model: nn.Module, x: torch.Tensor, mask: torch.Tensor, labels: list) -> torch.Tensor:
    '''
    Function to extract the prediction correctness for a batch of data-points.

    Parameters
    ----------
    cfg: TrainConfig
        Training configurations.
    model: nn.Module
        Model to evaluate.
    x (B, 4, T): torch.Tensor
        Input tokens.
    mask (B, 4, T): torch.Tensor
        Mask for ending tokens.
    labels (B): list
        Of indices corresponding to the correct labels.

    Returns
    -------
    flags (B): torch.Tensor
        Of bools corresponding to each data-point in the batch. True if the label was correctly predicted.

    '''
    # Initialization:
    vocab_size = cfg.model.vocab_size
    B, _, T = x.shape
    x = x.view(B * 4, T)

    # Forward pass and loss computation:
    with autocast_ctx(cfg.device, cfg.precision):
        logits, _ = model(x[:, :-1], targets=None)  # (B * 4, T - 1) -> (B * 4, T - 1, vocab_size_padded)
    logits = logits[:, :, :vocab_size]              # (B * 4, T - 1, vocab_size_padded) -> (B * 4, T - 1, vocab_size)
    logits = logits.transpose(2, 1)                 # (B * 4, T - 1, vocab_size) -> (B * 4, vocab_size, T - 1)
    loss = F.cross_entropy(logits, x[:, 1:], reduction='none')  # (B * 4, T - 1)

    # Mask loss to ending tokens only and pick the choice with maximum average probability:
    mask = mask[:, :, 1:]                           # mask entries corresponding to ending targets
    loss = loss.view(B, 4, T - 1) * mask            # (B, 4, T - 1)
    loss = loss.sum(dim=-1) / mask.sum(dim=-1)      # (B, 4)
    preds = torch.argmin(loss, dim=-1)              # (B, 4) -> (B)
    flags = preds == torch.tensor(labels, device=preds.device)

    return flags


@torch.inference_mode()
def eval_hellaswag(cfg: TrainConfig, model: nn.Module, dp_batches: list, ddp: DDPInfo) -> float:
    '''
    Function to evaluate a given model on HellaSwag dataset.

    Parameters
    ----------
    cfg: TrainConfig
        Training configurations.
    model: nn.Module
        Model to evaluate.
    dp_batches: list
        Of tuples of batches of data-points. See mk_dp_batches() for details.
    ddp: DDPInfo
        Distributed data parallelism instance.

    Returns
    -------
    acc: float
        Model accuracy.

    '''
    # Disable training mode for inference:
    train_mode = model.training
    model.eval()

    # Initialization:
    correct_count = total_count = 0

    try:
        # Evaluate the model on the dataset:
        for ind in tqdm(range(ddp.rank, len(dp_batches), ddp.world_size), disable=not ddp.is_main):
            x, mask, labels = dp_batches[ind]
            x, mask = x.to(ddp.device), mask.to(ddp.device)
            flags = get_batch_correctness(cfg, model, x, mask, labels)
            correct_count += flags.float().sum()
            total_count += flags.numel()
        counts = torch.tensor([float(correct_count), total_count], device=ddp.device)

        # Compute the accuracy:
        if ddp.is_active: dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        acc = ( counts[0] / counts[1] ).item()

    finally:
        if train_mode: model.train()

    return acc

