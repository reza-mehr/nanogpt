
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
class DataConfig:
    train_frac: float = 0.9                                 # fraction of tokens used for training, the rest is validation

@dataclass
class ModelConfig:
    name: Literal['gpt', 'bigram'] = 'gpt'                  # Model architecture, looked up in the model registry
    vocab_size: tyro.conf.Suppress[int | None] = None       # number of unique tokens, set at runtime
    block_size: int = 256                                   # maximum context length
    n_embed: int = 384                                      # embedding dimension
    head_num: int = 6                                       # number of attention heads per block
    head_size: int = 64                                     # attention head size
    n_blocks: int = 6                                       # number of transformer blocks
    dropout: float = 0.2                                    # dropout probability
    batched_attention: bool = True                          # if True, compute multi-head self-attention in a batch for efficiency

@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    batch_size: int = 64                                    # sequences per training step
    max_iters: int = 5000                                   # total number of optimization steps
    lr_schedule: Literal['cosine', 'constant'] = 'constant' # learning rate schedule, constant or linear warmup followed by cosine decay
    learning_rate: float = 3e-4                             # peak learning rate
    lr_warmup_iters: int = 100                              # number of warm steps to reach peak learning rate
    grad_clip: float = float('inf')                         # maximum permissible gradient norm
    weight_decay: float = 0.01                              # AdamW weight decay coefficient (PyTorch default 0.01)
    decay_groups: bool = False                              # if True, apply weight decay only to 2D+ params (not biases or norms)
    beta1: float = 0.9                                      # AdamW momentum coefficient
    beta2: float = 0.999                                    # AdamW second-moment coefficient (LLMs often use 0.95)

    eval_interval: int = 500                                # steps between evaluations
    eval_iters: int = 200                                   # batches averaged per evaluation
    ckpt_interval: int = 500                                # steps between saves of latest.pt

    device: str = device                                    # device to run the training on
    seed: int = 1337                                        # random seed
    run_name: str = ''                                      # run folder name, empty means a timestamp
    runs_root: str = 'runs'                                 # parent folder for all runs
    resume: bool = False                                    # resume training from out_dir/latest.pt
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

