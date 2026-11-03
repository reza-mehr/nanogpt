'''
This file contains the script to evaluate a model on HellaSwag dataset.

Evaluate a HuggingFace pre-trained model accuracy over a small set of batches for debugging:
    uv run scripts/eval_hellaswag.py --batch-size 16 --batch-num 4 --gpt2-pretrained gpt2

Evaluate a trained model accuracy, loaded from a checkpoint, over a small set of batches for debugging:
    uv run scripts/eval_hellaswag.py --batch-size 16 --batch-num 4 --checkpoint <path>

Evaluate using distributed data parallelism (DDP):
    bash scripts/ddp.sh 2 scripts/eval_hellaswag.py --batch-size 16 --batch-num 4 --gpt2-pretrained gpt2 --device
cpu

'''
#%% Imports:

from pathlib import Path
import tyro
from dataclasses import dataclass, replace
from typing import Literal

import torch.distributed as dist

from nanogpt.config import DEVICE, PRESETS
from nanogpt.tokenizers import GPT2Tokenizer
from nanogpt.distributed import DDPInfo
from nanogpt.hellaswag import get_dataset, load_dataset, mk_dp_batches, eval_hellaswag
from nanogpt.model import GPT2
from nanogpt.checkpoint import load_model_tokenizer

#%% Configurations:

@dataclass
class EvalConfig:
    data_dir: str = 'data/hellaswag'        # path to download the dataset to
    batch_size: int = 64                    # batch size
    batch_num: int | None = None            # number of batches to use, default is None in which case all batches are used
    gpt2_pretrained: Literal['gpt2', 'medium', 'large', 'xl'] | None = None
    # pre-trained GPT2 model to load and evaluate
    checkpoint: str | None = None           # checkpoint to evaluate
    device: str = DEVICE                    # device to evaluate on
    precision: Literal['fp32', 'mixed-bf16'] = 'mixed-bf16'
    # compute precision for forward passes; bf16 uses autocast, with weights kept in fp32

    def __post_init__(self):
        if self.batch_size < 1: raise ValueError(f"Invalid batch-size '{self.batch_size}'.")
        if self.batch_num is not None:
            if self.batch_num < 1: raise ValueError(f"Invalid batch-num '{self.batch_num}'.")
            if self.batch_num * self.batch_size > 10042:
                raise ValueError(f"Number of requested data-points exceeds dataset size: batch-num * batch-size > 10042.")
        if self.gpt2_pretrained and self.checkpoint:
            raise ValueError("Only a single checkpoint or pre-trained model could be evaluated.")
        if self.gpt2_pretrained is None and self.checkpoint is None:
            raise ValueError("Please specify a checkpoint or a pre-trained model to evaluate.")


#%% Main function:

def main(p: EvalConfig):
    '''
    Function to construct the model and run the HellaSwag evaluation.

    Parameters
    ----------
    p: EvalConfig
        Evaluation configurations.

    Returns
    -------
    acc: float
        Measured normalized accuracy over HellaSwag validation dataset.

    '''
    # Set up distributed data parallelism if requested:
    ddp = DDPInfo.setup(p.device)

    try:
        # Initialization:
        tok = GPT2Tokenizer()
        vocab_size = tok.vocab_size

        # Construct the model:
        if p.gpt2_pretrained:
            cfg = PRESETS[p.gpt2_pretrained + '_cfg']
            cfg = replace(cfg, model = replace(cfg.model, vocab_size=vocab_size))
            model = GPT2.from_pretrained(cfg.model).to(ddp.device)
        else:
            model, tok_model, cfg = load_model_tokenizer(Path(p.checkpoint), ddp.device)
            if not isinstance(tok_model, GPT2Tokenizer): raise ValueError(f"Model was trained using a tokenizer other than GPT2Tokenizer: {p.checkpoint}")

        # Download, tokenize, and split the dataset into batches:
        if ddp.is_main: get_dataset(p.data_dir)
        if ddp.is_active: dist.barrier()        # all ranks wait for data download to complete
        df = load_dataset(f"{p.data_dir}/default/validation/0000.parquet")
        dp_batches = mk_dp_batches(df, tok, p.batch_size, p.batch_num)

        # Evaluate the model accuracy:
        cfg = replace(cfg, device=ddp.device, precision=p.precision)
        acc = eval_hellaswag(cfg, model, dp_batches, ddp)

    finally:
        ddp.teardown()

    return acc

#%% Main call:

if __name__ == '__main__':
    acc = main(tyro.cli(EvalConfig))
    print(f"Normalized accuracy: {acc:.3f}")
