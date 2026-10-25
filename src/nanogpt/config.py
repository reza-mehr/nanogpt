
#%% Imports:

import sys
from pathlib import Path
import tyro
from dataclasses import dataclass, field, fields, is_dataclass, replace
import torch
import inspect
from typing import Literal
import logging


#%% Global variables:

DEVICE = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)

# Run folder name prefixes of benchmarks and profiles; training runs must not use them:
NON_TRAINING_PREFIXES = ('benchmark', 'profile')

#%% Main config:

@dataclass
class DataConfig:
    dir: str = 'data/tinyshakespeare_char'  # parent folder for the binary dataset split (check scripts/prepare_*.py)


@dataclass
class ModelConfig:
    name: Literal['gpt', 'gpt2', 'bigram'] = 'gpt'      # Model architecture, looked up in the model registry
    vocab_size: tyro.conf.Suppress[int | None] = None   # number of unique tokens, set at runtime
    pad_vocab_to_multiple: int = 1                      # pad vocab size to a multiple of 2^n, use 128
    block_size: int = 256                               # maximum context length
    n_embed: int = 384                                  # embedding dimension
    n_blocks: int = 6                                   # number of transformer blocks

    head_num: int = 6                                   # number of attention heads per block
    kv_head_num: int | None = None
    # kv_head_num = None or head_num -> multi-head attention (MHA), 1 < kv_head_num < head_num -> grouped-query attention (GQA), kv_head_num = 1 -> multi-query attention (MQA)
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

        # Checks for number of kv heads for GQA:
        if not (self.kv_head_num is None or 1 <= self.kv_head_num <= self.head_num):
            raise ValueError(f"'kv_head_num' {self.kv_head_num} out of bound, must be between 1 and 'head_num' {self.head_num}.")
        if not (self.kv_head_num is None or self.head_num % self.kv_head_num == 0):
            raise ValueError(f"'head_num' {self.head_num} must be divisible by 'kv_head_num' {self.kv_head_num}.")

        # Advanced developments are infeasible or not done on looped multi-head attention path:
        if not self.batched_attention:
            unsupported = []
            if not (self.kv_head_num is None or self.kv_head_num == self.head_num): unsupported.append('GQA')
            if self.scaled_dot_prod: unsupported.append('Scaled dot-product attention')
            if self.pos_embed == 'rope': unsupported.append('Rotary positional embedding')
            if unsupported:
                raise ValueError(f"{', '.join(unsupported)} unavailable when 'batched_attention' is set to False.")

        # Rotary positional embedding (RoPE) tests:
        if self.pos_embed == 'rope' and not (self.head_size % 2 == 0):
            raise ValueError(f"'head_size' {self.head_size} must be even for RoPE!")
        if self.pos_embed == 'rope' and self.rope_base <= 1:
            raise ValueError(f"'rope_base' {self.rope_base} must be larger than one!")

        # Open-AI GPT2 model config tests:
        if self.name=='gpt2':
            if not self.head_size == self.n_embed // self.head_num:
                raise ValueError(f"Head size must equal 'n_embed // head_num' for Open-AI GPT2 models: {self.head_size} vs {self.n_embed // self.head_num}")

            unsupported = []
            if not (self.kv_head_num is None or self.kv_head_num == self.head_num): unsupported.append('GQA')
            if not self.batched_attention: unsupported.append('Looped attention (batched_attention = False)')
            if self.pos_embed == 'rope': unsupported.append("Rotary positional embedding (pos_embed = 'rope')")
            if not self.ln_torch: unsupported.append('Manual layer-norm (ln_torch = False)')
            if self.dropout > 0.: unsupported.append('Dropout')
            if unsupported:
                raise ValueError(f"{', '.join(unsupported)} unavailable for Open-AI GPT2 models.")

        # Error handling for vocab size padding:
        m = self.pad_vocab_to_multiple
        if not (m >0 and (m & (m-1)) ==0 ):
            raise ValueError(f"'pad_vocab_to_multiple={m}' must be a positive integer and power of 2.")

    @property
    def n_kv_head(self):
        return self.head_num if self.kv_head_num is None else self.kv_head_num

    @property
    def vocab_size_padded(self):
        '''Function to raise 'vocab_size' to a multiple of power of two.'''
        v, m = self.vocab_size, self.pad_vocab_to_multiple
        return v if m == 1 else (-(-v // m) * m)


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

    device: str = DEVICE                                    # device to run the training on
    precision: Literal['fp32', 'mixed_bf16'] = 'fp32'
    # compute precision for forward passes; bf16 uses autocast, with weights kept in fp32
    compile_model: bool = False                             # if True, compile model operations for efficiency

    seed: int = 1337                                        # random seed
    preset: tyro.conf.Suppress[str] = 'custom'              # preset name, set in PRESETS, used in the default run name
    run_name: str = ''                                      # run folder name, empty means 'train_<preset>_<device>'
    out_dir: str = 'runs'                                   # parent folder for runs
    resume: bool = False                                    # resume training from run_dir/latest.pt
    overwrite: bool = False                                 # if True, clear an existing run folder without asking
    log_level: int = logging.INFO                           # logging level, 20 for INFO, 10 for DEBUG

    sample_prompts: tuple[str, ...] = ('',)                 # prompts for final samples; '' starts from the start token
    sample_tokens: int = 200                                # new tokens per sample
    sample_seed: int = 0                                    # fixed seed so samples are comparable across runs

    def __post_init__(self):
        '''Function to validate input arguments and assign default values.'''
        if self.lr_schedule == 'cosine' and not (0 < self.lr_warmup_iters < self.max_iters):
            raise ValueError(f"Warmup '{self.lr_warmup_iters}' must be between 0 and max iterations '{self.max_iters}'.")
        if self.run_name.startswith(NON_TRAINING_PREFIXES):
            raise ValueError(f"Run name '{self.run_name}' must not start with {NON_TRAINING_PREFIXES}; "
                             f"those prefixes mark benchmark and profile folders.")

    @property
    def run_dir(self) -> Path:
        '''Run folder: 'run_name' if given, otherwise named after the preset and device like the benchmarks.'''
        return Path(self.out_dir) / (self.run_name or f'train_{self.preset}_{self.device}')

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

# Open-AI GPT2 configs (Hugging Face version):
gpt2_cfg = TrainConfig(model=ModelConfig(name='gpt2', vocab_size=50257, block_size=1024, n_embed=768, n_blocks=12, head_num=12, head_size=64, batched_attention=True, scaled_dot_prod=True, pos_embed='learned', ln_torch=True, dropout=0.))
gpt2_medium_cfg = TrainConfig( model = replace(gpt2_cfg.model, n_embed=1024, n_blocks=24, head_num=16) )
gpt2_large_cfg = TrainConfig( model = replace(gpt2_cfg.model, n_embed=1280, n_blocks=36, head_num=20) )
gpt2_xl_cfg = TrainConfig( model = replace(gpt2_cfg.model, n_embed=1600, n_blocks=48, head_num=25) )


# Register the preset configs, each carrying its own name for the default run folder:
PRESETS: dict[str, TrainConfig] = {name: replace(cfg, preset=name) for name, cfg in {
    'main_cfg': main_cfg,
    'small_cfg': small_cfg,
    'bigram_cfg': bigram_cfg,
    'stories_cfg': stories_cfg,
    'gpt2_cfg': gpt2_cfg,
    'gpt2_medium_cfg': gpt2_medium_cfg,
    'gpt2_large_cfg': gpt2_large_cfg,
    'gpt2_xl_cfg': gpt2_xl_cfg,
}.items()}

#%% Function to parse arguments:

def parse_config() -> TrainConfig:
    name = 'main_cfg'           # default preset
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        name = sys.argv.pop(1)
    if name not in PRESETS:
        sys.exit(f"Unknown preset '{name}'. Available: {sorted(PRESETS)}")
    return tyro.cli(TrainConfig, default=PRESETS[name])


#%% Utility function:

def config_record(obj) -> dict:
    '''
    Function to convert a config to a dict of its fields and its properties (derived values such as n_kv_head).
    For records only (config.json). Unlike asdict(), the result cannot rebuild the config, since properties are
    not constructor arguments; checkpoints keep asdict() for that.

    '''
    record = {f.name: config_record(v) if is_dataclass(v) else v
              for f in fields(obj) for v in [getattr(obj, f.name)]}
    for name, _ in inspect.getmembers(type(obj), lambda a: isinstance(a, property)):
        record[name] = getattr(obj, name)
    return record
