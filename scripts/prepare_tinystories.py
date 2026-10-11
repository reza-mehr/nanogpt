
#%% Imports:

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from datasets import load_dataset
from tqdm import tqdm

from nanogpt.tokenizers import GPT2Tokenizer

'''
This file contains the code to download the TinyStories dataset, tokenize with GPT-2 BPE, and write flat uint16 token train and validation files.

'''

#%% Global variables:

TOK = GPT2Tokenizer()

@dataclass
class PrepConfig:
    out_dir: str = 'data/tinystories'       # path to write train.bin, val.bin, and meta.json files
    num_proc: int = 8                       # worker processes for tokenization
    max_stories: int | None = None          # use only the first N stories per split (for quick dev runs); None = all


#%% Functions:

def tokenize_batch(batch: dict) -> dict:
    '''Function to tokenize a batch of stories, appending <|endoftext|> after each.'''
    ids = [TOK.encode(text) + [TOK.eot] for text in batch['text']]
    return {'ids': ids, 'len': [len(x) for x in ids]}


def write_split(ds, path: Path) -> int:
    '''Function to concatenate all token ids of a split into one uint16 memmap file.'''

    # Initialization:
    total = int(np.sum(ds['len'], dtype=np.uint64))
    arr = np.memmap(path, dtype=np.uint16, mode='w+', shape=(total,))

    # Write in contiguous shards so the whole split never sits in memory at once:
    n_shards = min(1024, len(ds))
    idx = 0
    for i in tqdm(range(n_shards), desc=path.name):
        shard = ds.shard(num_shards=n_shards, index=i, contiguous=True).with_format('numpy')
        ids = np.concatenate(shard['ids'])
        arr[idx:idx + len(ids)] = ids
        idx += len(ids)
    arr.flush()

    return total


def main(p: PrepConfig) -> None:
    '''Main function to prepare TinyStories for training.'''

    # uint16 holds ids up to 65,535:
    assert TOK.vocab_size < 2**16, 'vocabulary too large for uint16'

    # Initialization:
    out = Path(p.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset('roneneldan/TinyStories')
    meta = TOK.to_meta() | {'dtype': '<u2', 'n_tokens': {}}

    # Loop over splits:
    for split, name in [('train', 'train'), ('validation', 'val')]:
        d = ds[split]
        if p.max_stories:
            d = d.select(range(min(p.max_stories, len(d))))
        d = d.map(tokenize_batch, batched=True, remove_columns=['text'],
                  num_proc=p.num_proc, desc=f'tokenizing {split}')
        meta['n_tokens'][name] = write_split(d, out / f'{name}.bin')

    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta['n_tokens'], indent=2))


#%% Main call:

if __name__ == '__main__':
    main(tyro.cli(PrepConfig))
