
'''
This file contains the code to download the FineWeb dataset, tokenize with GPT-2 BPE, and write flat uint16 token train and validation files.

'''
#%% Imports:

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tyro
from tqdm import tqdm

from huggingface_hub import HfApi
from datasets import load_dataset, Features, Sequence, Value

from nanogpt.tokenizers import GPT2Tokenizer

#%% Global variables:

TOK = GPT2Tokenizer()
REMOTE_NAME = 'sample-10BT'

@dataclass
class PrepConfig:
    out_dir: str = 'data/fineweb'                   # path to write train.bin, val.bin, and meta.json files
    num_proc: int = 8                               # worker processes for tokenization
    data_file: str | None = None                    # single parquet file for testing, e.g., sample/10BT/000_00000.parquet
    num_threads: int = 1                            # number of threads for multi-threading

    @property
    def outpath(self):
        '''Path to write train.bin, val.bin, and meta.json files.'''
        return self.out_dir if self.data_file is None else f'{self.out_dir}_single_parquet'


#%% Functions:

def tokenize_batch(batch: dict, num_threads: int = 1) -> dict:
    '''Function to tokenize a batch of data, appending <|endoftext|> after each.'''
    if num_threads == 1:
        ids = [TOK.encode(text) + [TOK.eot] for text in batch['text']]
    else:
        ids = TOK.encode_batch(batch['text'], num_threads)
        ids = [x + [TOK.eot] for x in ids]
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
    '''Main function to prepare FineWeb dataset for training.'''

    # Error handling (uint16 holds ids up to 65,535):
    if TOK.vocab_size > 2**16: raise ValueError('Vocabulary too large for uint16!')

    # Error handling:
    if p.data_file and not HfApi().file_exists('HuggingFaceFW/fineweb-edu', p.data_file, repo_type='dataset'):
        raise FileNotFoundError(f"Requested parquet file does not exist, make sure to use the full path sample/10BT/...: {p.data_file}")

    # Construct the output path:
    out = Path(p.outpath)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'meta.json').exists(): raise FileExistsError(f"Tokenized binaries already exist: {out}")

    # Initialization:
    kwargs = {'data_files': [p.data_file]} if p.data_file else {'name': REMOTE_NAME}
    ds = load_dataset('HuggingFaceFW/fineweb-edu', split='train', **kwargs)
    features = Features({'ids': Sequence(Value('uint16')),  # essential to save disk 4x with HF cache (int64 by default)
                         'len': Value('int64')})
    meta = TOK.to_meta() | {'dtype': '<u2', 'n_tokens': {}}

    # Split before mapping for contiguous .bin files (hold out 1% for validation):
    ds = ds.train_test_split(test_size=0.01, shuffle=True, seed=0)  # 'seed' required to resume failed tokenization runs

    # Loop over splits:
    for split, name in [('train', 'train'), ('test', 'val')]:
        d = ds[split]
        d = d.map(tokenize_batch, batched=True, remove_columns=d.column_names, features=features,
                  num_proc=p.num_proc, fn_kwargs={'num_threads': p.num_threads}, desc=f'tokenizing {split}')
        meta['n_tokens'][name] = write_split(d, out / f'{name}.bin')

    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta['n_tokens'], indent=2))


#%% Main call:

if __name__ == '__main__':
    main(tyro.cli(PrepConfig))
