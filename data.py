
#%% Imports:

from pathlib import Path
import torch

from config import TrainConfig

'''
This file contains the code to construct tokenizer and sample batches from a text dataset.

'''

#%% Main code:

class CharTokenizer:
    '''Class to implement character-level tokenizer where the vocabulary is the set of characters in the text dataset.'''
    def __init__(self, text: str):
        '''
        Parameters
        ----------
        text: str
            Dataset.

        Attributes
        ----------
        stoi: dict
            Maps characters to tokens.
        itos: dict
            Inverse map from tokens to their corresponding characters.

        '''
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}


    @property
    def vocab_size(self) -> int:
        '''Returns the unique number of characters in the dataset.'''
        return len(self.stoi)


    def encode(self, s: str) -> list[int]:
        '''
        Function to encode a given string using the generated token mapping.

        Parameters
        ----------
        s: str
            String to tokenize.

        Returns
        -------
        x: list[int]
            Tokenized list corresponding to the input string.

        '''
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        '''
        Function to decode a given list of tokens to corresponding string.

        Parameters
        ----------
        ids: list[int]
            List of tokens to decode.

        Returns
        -------
        s: str
            Decoded string.

        '''
        return "".join(self.itos[i] for i in ids)


def load_splits(path: str, train_frac: float) -> tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
    '''
    Function to  read text, tokenize, and split into train/val tensors.

    Parameters
    ----------
    path: str
        Path to the text file containing the dataset to tokenize and split.
    train_frac: float
        Fraction of the dataset to be used for training the model.

    Returns
    -------
    train_data: torch.Tensor
        Train data.
    val_data: torch.Tensor
        Validation data.
    tok: CharTokenizer
        CharTokenizer object containing the token encoder and decoder.

    '''
    text = Path(path).read_text()
    tok = CharTokenizer(text)
    tokens = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(train_frac * len(tokens))
    return tokens[:n], tokens[n:], tok


def get_batch(cfg: TrainConfig, data: torch.Tensor):
    '''
    Function to sample batches from data given a set of configurations.

    Parameters
    ----------
    cfg: TrainConfig
        Configurations to use.
    data: torch.Tensor
        To generate batches from.

    Returns
    -------
    x (batch_size, block_size): tensor
        Input token indices.
    y (batch_size, block_size): tensor
        Corresponding target toekn indices.

    '''
    block_size = cfg.model.block_size
    ix = torch.randint(len(data) - block_size, (cfg.batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)
