### Summary: lr3e-4_s0

| metric                  | value               |
| ----------------------- | ------------------- |
| steps                   | 3,000               |
| tokens                  | 98,304,000          |
| final train loss        | 1.6840              |
| final val loss          | 1.6891              |
| best val loss           | 1.6891 (step 3,000) |
| parameters              | 162,273,361         |
| training time (s)       | 944                 |
| mean throughput (tok/s) | 104,189             |

### Samples

Final model (step 3000), seed 0, 200 new tokens.

#### Prompt: 'Once upon a time'

```text
Once upon a time there was a sunflower. The sunflower was very kind and gentle. Every day, it liked to grow and beauty. 

One day, the sunflower heard a sound. It was a bird singing in the sky. The bird wanted to fly too, but the sunflower wanted to stay and rest. 

So, the bird ate a few fresh air and made a sweet sound. The other birds stopped and watched the sunflower stay in the sky. 

The sunflower stayed with the sunflower and enjoyed the while it was still alive. From then on, the other flowers and trees would eat the sunflower, but never too far away to enjoy the fresh storm. 

The pretty world was happy, because it was special and could make the sun shine brighter!
```

### Configuration

- **data.dir:** data/tinystories
- **model.name:** gpt
- **model.vocab_size:** 50257
- **model.block_size:** 512
- **model.n_embed:** 768
- **model.n_blocks:** 12
- **model.head_num:** 12
- **model.kv_head_num:** None
- **model.head_size:** 64
- **model.batched_attention:** True
- **model.scaled_dot_prod:** True
- **model.pos_embed:** rope
- **model.rope_base:** 10000.0
- **model.ln_torch:** True
- **model.dropout:** 0.0
- **model.n_kv_head:** 12
- **batch_size:** 64
- **max_iters:** 3000
- **lr_schedule:** cosine
- **learning_rate:** 0.0003
- **lr_warmup_iters:** 500
- **grad_clip:** 1.0
- **weight_decay:** 0.1
- **decay_groups:** True
- **beta1:** 0.9
- **beta2:** 0.95
- **adamw_fused:** True
- **eval_interval:** 250
- **eval_iters:** 100
- **ckpt_interval:** 3000
- **device:** cuda
- **precision:** mixed_bf16
- **seed:** 0
- **preset:** stories_cfg
- **run_name:** lr3e-4_s0
- **out_dir:** runs/full/ablations/runs
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('Once upon a time',)
- **sample_tokens:** 200
- **sample_seed:** 0
- **min_learning_rate:** 2.9999999999999997e-05
- **run_dir:** runs/full/ablations/runs/lr3e-4_s0

### Environment

- **date:** 2026-10-16T06:58:07
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-124-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
