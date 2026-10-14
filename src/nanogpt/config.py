
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
    dir: str = 'data/tinyshakespeare'           # parent folder for the binary dataset split (check scripts/prepare_*.py)


@dataclass
class ModelConfig:
    name: Literal['gpt', 'bigram'] = 'gpt'              # Model architecture, looked up in the model registry
    vocab_size: tyro.conf.Suppress[int | None] = None   # number of unique tokens, set at runtime
    block_size: int = 256                               # maximum context length
    n_embed: int = 384                                  # embedding dimension
    n_blocks: int = 6                                   # number of transformer blocks

    head_num: int = 6                                   # number of attention heads per block
    n_kv_head: int | None = None                        # None -> head_num (MHA), 1 < n_kv_head < head_num (GQA), 1 (MQA)
    head_size: int = 64                                 # attention head size
    batched_attention: bool = True
    # if True, compute multi-head self-attention in a batch for efficiency
    scaled_dot_prod: bool = True
    # if True, use scaled dot product instead of manual implementation for efficiency

    pos_embed: Literal['learned', 'rope'] = 'learned'   # positional embedding method: learned or rotary
    rope_base: float = 10000.
    # Rotary positional embedding (RoPE) base, ignored when 'pos_embed' is 'learned'

    ln_torch: bool = True                               # if True, use PyTorch implementation of the layer-norm
    dropout: float = 0.2                                # dropout probability


    def __post_init__(self):
        '''Function to validate input arguments and assign default values.'''

        # Default value and checks for number of kv heads for GQA:
        if self.n_kv_head is None: self.n_kv_head = self.head_num
        if not 1 <= self.n_kv_head <= self.head_num:
            raise ValueError(f"'n_kv_head' {self.n_kv_head} out of bound, must be between 1 and 'head_num' {self.head_num}.")
        if self.head_num % self.n_kv_head != 0:
            raise ValueError(f"'head_num' {self.head_num} must be divisible by 'n_kv_head' {self.n_kv_head}.")

        # Advanced developments are infeasible or not done on looped multi-head attention path:
        if not self.batched_attention:
            unsupported = []
            if self.n_kv_head != self.head_num: unsupported.append('GQA')
            if self.scaled_dot_prod: unsupported.append('Scaled dot-product attention')
            if self.pos_embed == 'rope': unsupported.append('Rotary positional embedding')
            if unsupported:
                raise ValueError(f"{', '.join(unsupported)} unavailable when 'batched_attention' is set to False.")

        # Rotary positional embedding (RoPE) tests:
        if self.pos_embed == 'rope' and not (self.head_size % 2 == 0):
            raise ValueError(f"'head_size' {self.head_size} must be even for RoPE!")
        if self.pos_embed == 'rope' and self.rope_base <= 1:
            raise ValueError(f"'rope_base' {self.rope_base} must be larger than one!")


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    batch_size: int = 64                                    # sequences per training step
    max_iters: int = 5000                                   # total number of optimization steps
    lr_schedule: Literal['cosine', 'constant'] = 'constant'
    # learning rate schedule, constant or linear warmup followed by cosine decay
    learning_rate: float = 3e-4                             # peak learning rate
    lr_warmup_iters: int = 100                              # number of warm steps to reach peak learning rate
    grad_clip: float = float('inf')                         # maximum permissible gradient norm
    weight_decay: float = 0.01                              # AdamW weight decay coefficient (PyTorch default 0.01)
    decay_groups: bool = False
    # if True, apply weight decay only to 2D+ params (not biases or norms)
    beta1: float = 0.9                                      # AdamW momentum coefficient
    beta2: float = 0.999                                    # AdamW second-moment coefficient (LLMs often use 0.95)
    adamw_fused: bool = True                                # if True, update all tensors in one kernel

    eval_interval: int = 500                                # steps between evaluations
    eval_iters: int = 200                                   # batches averaged per evaluation
    ckpt_interval: int = 500                                # steps between saves of latest.pt

    device: str = device                                    # device to run the training on
    precision: Literal['fp32', 'mixed_bf16'] = 'fp32'
    # compute precision for forward passes; bf16 uses autocast, with weights kept in fp32

    seed: int = 1337                                        # random seed
    run_name: str = ''                                      # run folder name, empty means a timestamp
    out_dir: str = 'runs'                                   # parent folder for runs
    resume: bool = False                                    # resume training from out_dir/latest.pt
    log_level: int = logging.INFO                           # logging level, 20 for INFO, 10 for DEBUG

    sample_prompts: tuple[str, ...] = ('',)                 # prompts for final samples; '' starts from the start token
    sample_tokens: int = 200                                # new tokens per sample
    sample_seed: int = 0                                    # fixed seed so samples are comparable across runs

    def __post_init__(self):
        '''Function to validate input arguments and assign default values.'''
        assert 0 < self.lr_warmup_iters < self.max_iters, 'warmup must fit inside the run'
        if not self.run_name:
            self.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name

    @property
    def min_learning_rate(self) -> float:
        return self.learning_rate * 0.1                     # 10 percent of the learning rate


#%% Preset configs:

# Production nano-GPT model:
main_cfg = TrainConfig()

# Small nano-GPT model that runs on a laptop:
small_cfg = TrainConfig(model=ModelConfig(block_size=8, n_embed=32, head_num=4, head_size=8, n_blocks=4, dropout=0.1),
                        batch_size=32, learning_rate=1e-3)

# Basic Bigram model:
bigram_cfg = TrainConfig(model=ModelConfig(name='bigram'), batch_size=32, learning_rate=1e-3)

# Larger GPT-2-small-shaped model for TinyStories, also used for GPU benchmarks and profiles:
stories_cfg = TrainConfig(data=DataConfig(dir='data/tinystories'),
                          model=ModelConfig(block_size=512, n_embed=768, n_blocks=12, head_num=12, head_size=64,
                                            pos_embed='rope', dropout=0.0),
                          batch_size=32, max_iters=10000, lr_schedule='cosine', learning_rate=6e-4,
                          lr_warmup_iters=500, grad_clip=1.0, weight_decay=0.1, decay_groups=True, beta2=0.95,
                          eval_interval=500, eval_iters=100, ckpt_interval=500, precision='mixed_bf16')

# Register the preset configs:
PRESETS: dict[str, TrainConfig] = {
    'main_cfg': main_cfg,
    'small_cfg': small_cfg,
    'bigram_cfg': bigram_cfg,
    'stories_cfg': stories_cfg,
}

#%% Function to parse arguments:

def parse_config() -> TrainConfig:
    name = 'main_cfg'           # default preset
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        name = sys.argv.pop(1)
    if name not in PRESETS:
        sys.exit(f"Unknown preset '{name}'. Available: {sorted(PRESETS)}")
    return tyro.cli(TrainConfig, default=PRESETS[name])

