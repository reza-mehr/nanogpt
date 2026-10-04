
#%% Imports:

import sys
import tyro
from dataclasses import dataclass, field
import torch
from typing import Literal
import logging

from datetime import datetime
from pathlib import Path

#%% Global variables:

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)

#%% Main code:

@dataclass
class ModelConfig:
    name: Literal['gpt', 'bigram'] = 'gpt'                  # model name
    vocab_size: tyro.conf.Suppress[int | None] = None       # number of unique characters in the tiny Shakespeare dataset
    block_size: int = 256                                   # context length
    n_embed: int = 384                                      # embedding dimension
    head_num: int = 6                                       # number of attention heads per block
    head_size: int = 64                                     # attention head size
    n_blocks: int = 6                                       # number of blocks
    dropout: float = 0.2                                    # dropout ratio

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    device: str = device                                    # device to run the training on
    train_frac: float = 0.9                                 # fraction of the dataset used for training
    batch_size: int = 64                                    # batch size
    max_iters: int = 5000                                   # maximum number of training iterations
    lr_schedule: Literal['cosine', 'constant'] = 'cosine'   # learning rate schedule
    learning_rate: float = 3e-4                             # learning rate
    lr_warmup_iters: int = 100                              # number of warm iteratations to reach specified learning rate
    eval_interval: int = 500                                # evaluation interval
    eval_iters: int = 200                                   # evaluation iterations to smooth loss values
    ckpt_interval: int = 500                                # checkpoint interval
    seed: int = 1337                                        # random seed
    run_name: str = ''                                      # run folder name
    runs_root: str = 'runs'                                 # 'runs' directory
    resume: bool = False                                    # resume training from
    log_level: int = logging.INFO                           # logging level, 20 for INFO, 10 for DEBUG

    def __post_init__(self):
        if not self.run_name:
            self.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        assert 0 < self.lr_warmup_iters < self.max_iters, 'warmup must fit inside the run'

    @property
    def out_dir(self) -> Path:
        return Path(self.runs_root) / self.run_name

    @property
    def min_learning_rate(self) -> float:
        return self.learning_rate * 0.1                     # 10 percent of the learning rate

# TODO: to be added
# grad_clip: float = 1.0

#%% Preset configs:

# Production nano-GPT model:
main_cfg = TrainConfig()

# Small nano-GPT model that runs on a laptop:
small_cfg = TrainConfig(model=ModelConfig(block_size=8, n_embed=32, head_num=4, head_size=8, n_blocks=4, dropout=0.1),
                        batch_size=32, learning_rate=1e-3)

# Basic Bigram model:
bigram_cfg = TrainConfig(model=ModelConfig(name='bigram'), batch_size=32, learning_rate=1e-3)

# Register the preset configs:
PRESETS: dict[str, TrainConfig] = {
    'main_cfg': main_cfg,
    'small_cfg': small_cfg,
    'bigram_cfg': bigram_cfg,
}

#%% Function to parse arguments:

def parse_config() -> TrainConfig:
    name = 'main_cfg'           # default preset
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        name = sys.argv.pop(1)
    if name not in PRESETS:
        sys.exit(f"Unknown preset '{name}'. Available: {sorted(PRESETS)}")
    return tyro.cli(TrainConfig, default=PRESETS[name])

