'''
This file contains the script to evaluate a model on HellaSwag dataset.

Evaluate a HuggingFace pre-trained model accuracy over a small set of batches for debugging:
    uv run scripts/eval_hellaswag.py --batch-size 16 --batch-num 4 --gpt2-pretrained gpt2

Evaluate a trained model accuracy, loaded from a checkpoint, over a small set of batches for debugging:
    uv run scripts/eval_hellaswag.py --batch-size 16 --batch-num 4 --checkpoint <path>

Evaluate using distributed data parallelism (DDP):
    bash scripts/ddp.sh 2 scripts/eval_hellaswag.py --batch-size 16 --batch-num 4 --gpt2-pretrained gpt2 --device cpu

Outputs (main process only) go to '<out-dir>/<run-name>': run.log, results.json, config.json, env.json, git.json, report.md.

'''
#%% Imports:

import shlex
import sys
import time
from pathlib import Path
import tyro
from dataclasses import dataclass, replace
from typing import Literal

import torch.distributed as dist

from nanogpt.config import DEVICE, PRESETS, config_record
from nanogpt.tokenizers import GPT2Tokenizer
from nanogpt.distributed import DDPInfo
from nanogpt.hellaswag import get_dataset, load_dataset, mk_dp_batches, eval_hellaswag
from nanogpt.model import GPT2
from nanogpt.checkpoint import load_model_tokenizer, git_info
from nanogpt.reporting import Col, collect_env, prepare_run_dir, save_json, setup_logger, table_section, write_report

#%% Global variables:

# Reference normalized accuracies (completion style, full 10042 validation set):
REF_ACC_NORM = {'gpt2': 0.2955, 'xl': 0.4893}

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
    precision: Literal['fp32', 'mixed_bf16'] = 'mixed_bf16'
    # compute precision for forward passes; bf16 uses autocast, with weights kept in fp32
    out_dir: str = 'results/hellaswag'      # parent folder of the run folder
    run_name: str | None = None             # run folder name, default is '<model>_<precision>'
    overwrite: bool = False                 # clear an existing run folder without asking

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

    @property
    def model_name(self) -> str:
        '''Name of the evaluated model: the pre-trained preset or the checkpoint file stem.'''
        return self.gpt2_pretrained or Path(self.checkpoint).stem

    @property
    def run_dir(self) -> Path:
        '''Folder to write the logs and results to.'''
        return Path(self.out_dir) / (self.run_name or f'{self.model_name}_{self.precision}')


#%% Main function:

def main(p: EvalConfig):
    '''
    Function to construct the model, run the HellaSwag evaluation, and record the results.

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
        # Run folder, logger, and environment (main process only):
        logger, env = setup_io(p, ddp)

        # Construct the model and the batches:
        tok = GPT2Tokenizer()
        model, cfg = build_model(p, ddp, tok)
        dp_batches = prep_batches(p, ddp, tok)

        # Evaluate the model accuracy:
        cfg = replace(cfg, device=ddp.device, precision=p.precision)
        t0 = time.perf_counter()
        acc = eval_hellaswag(cfg, model, dp_batches, ddp)
        eval_s = time.perf_counter() - t0

        # Record the results (main process only):
        if ddp.is_main:
            results = dict(model=p.model_name, acc_norm=acc, ref_acc_norm=REF_ACC_NORM.get(p.gpt2_pretrained),
                           n_examples=sum(len(labels) for _, _, labels in dp_batches), eval_s=eval_s,
                           command=shlex.join(sys.argv))
            save_results(p, cfg, results, env, logger)

    finally:
        ddp.teardown()

    return acc


#%% Utility functions:

def setup_io(p: EvalConfig, ddp: DDPInfo):
    '''
    Function to prepare the run folder, the logger, and the environment record on the main process.

    Parameters
    ----------
    p: EvalConfig
        Evaluation configurations.
    ddp: DDPInfo
        Distributed data parallelism instance.

    Returns
    -------
    logger: logging.Logger or None
        Logger writing to stdout and 'run_dir/run.log'; None on other processes.
    env: dict or None
        Code version, software, and hardware details; None on other processes.

    '''
    if not ddp.is_main: return None, None

    prepare_run_dir(p.run_dir, p.overwrite)
    logger = setup_logger(p.run_dir)
    env = collect_env(ddp.device, ddp.world_size)

    return logger, env


def build_model(p: EvalConfig, ddp: DDPInfo, tok: GPT2Tokenizer):
    '''
    Function to construct a pre-trained HF GPT2 model or load a model from a checkpoint.

    Parameters
    ----------
    p: EvalConfig
        Evaluation configurations.
    ddp: DDPInfo
        Distributed data parallelism instance.
    tok: GPT2Tokenizer
        Tokenizer the dataset is tokenized with.

    Returns
    -------
    model: nn.Module
        Model on 'ddp.device'.
    cfg: TrainConfig
        Configurations of the model.

    Raises
    ------
    ValueError
        If the checkpoint was trained with a tokenizer other than GPT2Tokenizer.

    '''
    if p.gpt2_pretrained:
        cfg = PRESETS[p.gpt2_pretrained + '_cfg']
        cfg = replace(cfg, model = replace(cfg.model, vocab_size=tok.vocab_size))
        model = GPT2.from_pretrained(cfg.model).to(ddp.device)
    else:
        model, tok_model, cfg = load_model_tokenizer(Path(p.checkpoint), ddp.device)
        if not isinstance(tok_model, GPT2Tokenizer): raise ValueError(f"Model was trained using a tokenizer other than GPT2Tokenizer: {p.checkpoint}")

    return model, cfg


def prep_batches(p: EvalConfig, ddp: DDPInfo, tok: GPT2Tokenizer) -> list:
    '''
    Function to download (main process only), tokenize, and split the HellaSwag dataset into batches.

    Parameters
    ----------
    p: EvalConfig
        Evaluation configurations.
    ddp: DDPInfo
        Distributed data parallelism instance.
    tok: GPT2Tokenizer
        Tokenizer to use.

    Returns
    -------
    dp_batches: list of tuples
        Each entry contains the tuple (x, mask, labels). See hellaswag.py/mk_dp_batches() for details.

    '''
    if ddp.is_main: get_dataset(p.data_dir)
    if ddp.is_active: dist.barrier()        # all ranks wait for data download to complete
    df = load_dataset(f"{p.data_dir}/default/validation/0000.parquet")

    return mk_dp_batches(df, tok, p.batch_size, p.batch_num)


def save_results(p: EvalConfig, cfg, results: dict, env: dict, logger) -> None:
    '''
    Function to save the results, configurations, environment, and git state, and write a Markdown report.

    Parameters
    ----------
    p: EvalConfig
        Evaluation configurations.
    cfg: TrainConfig
        Configurations of the evaluated model.
    results: dict
        Measured accuracy, reference accuracy, number of examples, evaluation time, and command.
    env: dict
        Environment from collect_env().
    logger: logging.Logger
        Logger from setup_logger().

    '''
    # JSON records:
    config = dict(eval=config_record(p), model=config_record(cfg.model))
    save_json(results, p.run_dir / 'results.json')
    save_json(config, p.run_dir / 'config.json')
    save_json(env, p.run_dir / 'env.json')
    save_json(git_info(), p.run_dir / 'git.json')     # commit, dirty flag, and diff

    # Markdown report with the summary table, configuration, and environment:
    cols = [Col('model', 'Model'), Col('n_examples', 'Examples', ',d'), Col('acc_norm', 'acc_norm', '.4f'),
            Col('ref_acc_norm', 'Reference', '.4f'), Col('eval_s', 'Time (s)', ',.1f')]
    write_report(p.run_dir / 'report.md', [table_section([results], cols, 'HellaSwag')], config, env)
    logger.info(f"Normalized accuracy: {results['acc_norm']:.4f} | results: {p.run_dir}")


#%% Main call:

if __name__ == '__main__':
    main(tyro.cli(EvalConfig))
