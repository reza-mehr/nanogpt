
'''
This file contains the code to download the Tiny Shakespeare dataset, tokenize it, and write flat uint16 token train and validation binaries.

'''
#%% Imports:

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from nanogpt.tokenizers import TOKENIZERS, Tokenizer

#%% Global variables:

# Link to Tiny Shakespeare dataset:
URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

@dataclass
class PrepConfig:
    input_file: str = 'data/tinyshakespeare/input.txt'  # text file containing the dataset, downloaded if unavailable
    out_dir: None | str = None                          # path to write train.bin, val.bin, and meta.json files
    tokenizer: str = 'char'                             # tokenizer to use
    train_frac: float = 0.9                             # fraction of tokens used for training, the rest is validation

    def __post_init__(self):
        '''Function to test configurations.'''
        TOKENIZERS.get(self.tokenizer)                  # will raise error for an unknown tokenizer

    @property
    def outpath(self):
        '''Path to write train.bin, val.bin, and meta.json files.'''
        return f'data/tinyshakespeare_{self.tokenizer}' if self.out_dir is None else self.out_dir

#%% Functions:

def download_tiny_shakespeare(p: PrepConfig) -> None:
    '''Function to download the Tiny Shakespeare dataset if unavailable.'''
    path = Path(p.input_file)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)


def tokenize_and_split(p: PrepConfig) -> tuple[list[int], list[int], Tokenizer]:
    '''
    Function to read the text, tokenize it, and split it into train/val token lists.

    Parameters
    ----------
    p: PrepConfig
        Preparation config with the input file and training fraction.

    Returns
    -------
    train_data: list[int]
        Training token indices.
    val_data: list[int]
        Validation token indices.
    tok: Tokenizer
        Tokenizer built from the text, containing the encoder and decoder.

    '''
    download_tiny_shakespeare(p)
    text = Path(p.input_file).read_text()
    tok = TOKENIZERS.get(p.tokenizer)(text)
    tokens = tok.encode(text)
    n = int(p.train_frac * len(tokens))
    return tokens[:n], tokens[n:], tok


def main(p: PrepConfig) -> None:
    '''Main function to prepare the Tiny Shakespeare dataset for training.'''

    # Construct the output path:
    out = Path(p.outpath)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'meta.json').exists(): raise FileExistsError(f"Tokenized binaries already exist: {out}")

    # Tokenize and split:
    train_data, val_data, tok = tokenize_and_split(p)

    # uint16 holds ids up to 65,535:
    assert tok.vocab_size < 2**16, 'vocabulary too large for uint16'

    # Metadata:
    meta = tok.to_meta() | {'dtype': '<u2', 'n_tokens': {}}

    # Loop over splits and save the token ids as a flat little-endian uint16 file:
    for split, name in [(train_data, 'train'), (val_data, 'val')]:
        np.asarray(split, dtype='<u2').tofile(out / f'{name}.bin')
        meta['n_tokens'][name] = len(split)

    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta['n_tokens'], indent=2))


#%% Main call:

if __name__ == '__main__':
    main(tyro.cli(PrepConfig))
