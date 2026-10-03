
#%% Imports:

import sys
import tyro
from dataclasses import dataclass, field
import torch
from typing import Literal

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
    batch_size: int = 64                                    # batch size
    max_iters: int = 5000                                   # maximum number of training iterations
    learning_rate: float = 3e-4                             # learning rate
    eval_interval: int = 500                                # evaluation interval
    eval_iters: int = 200                                   # evaluation iterations to smooth loss values
    ckpt_interval: int = 500                                # checkpoint interval
    seed: int = 1337                                        # random seed
    run_name: str = ''                                      # run folder name
    runs_root: str = 'runs'                                 # 'runs' directory
    resume: bool = False                                    # resume training from

    def __post_init__(self):
        if not self.run_name:
            self.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @property
    def out_dir(self) -> Path:
        return Path(self.runs_root) / self.run_name

# TODO: to be added
# warmup_iters: int = 100
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

def parseConfig() -> TrainConfig:
    name = 'main_cfg'           # default preset
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        name = sys.argv.pop(1)
    if name not in PRESETS:
        sys.exit(f"Unknown preset '{name}'. Available: {sorted(PRESETS)}")
    return tyro.cli(TrainConfig, default=PRESETS[name])

