### Summary: baseline_s2

| metric                  | value               |
| ----------------------- | ------------------- |
| steps                   | 3,000               |
| tokens                  | 98,304,000          |
| final train loss        | 1.5608              |
| final val loss          | 1.5641              |
| best val loss           | 1.5641 (step 3,000) |
| parameters              | 162,273,361         |
| training time (s)       | 941                 |
| mean throughput (tok/s) | 104,479             |

### Samples

Final model (step 3000), seed 0, 200 new tokens.

#### Prompt: 'Once upon a time'

```text
Once upon a time there was a sunflower. One day, the sunflower was lying on the ground. It was a very deep beauty. Its orange petals reached out its petals when it looked.

The sunflower was very busy. It kept wishing and wished, but it was too big. Suddenly it could feel a gentle breeze dripped through the air.

The sunflower touched the ground with its Sun. It felt strange and soft against the ground. Then, it looked up at the sunflower. It noticed the sun rising in the sky.

The sunflower felt peaceful in the sky. The sunflower looked down and noticed that it was starting to struggle to reach the sky.

Suddenly, the sun began to change its color and it started to shine again. The sunflower smiled; it was not so lonely anymore!
```

### Configuration

- **data.dir:** data/tinystories
- **model.name:** gpt
- **model.vocab_size:** 50257
- **model.block_size:** 512
- **model.n_embed:** 768
- **model.n_blocks:** 12
- **model.head_num:** 12
- **model.kv_head_num:** 12
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
- **learning_rate:** 0.0006
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
- **seed:** 2
- **preset:** stories_cfg
- **run_name:** baseline_s2
- **out_dir:** runs/full/ablations/runs
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('Once upon a time',)
- **sample_tokens:** 200
- **sample_seed:** 0
- **min_learning_rate:** 5.9999999999999995e-05
- **run_dir:** runs/full/ablations/runs/baseline_s2

### Environment

- **date:** 2026-10-16T05:43:32
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-124-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
