### Summary: lr2.4e-3_s0

| metric                  | value               |
| ----------------------- | ------------------- |
| steps                   | 3,000               |
| tokens                  | 98,304,000          |
| final train loss        | 1.4713              |
| final val loss          | 1.4816              |
| best val loss           | 1.4816 (step 3,000) |
| parameters              | 162,273,361         |
| training time (s)       | 954                 |
| mean throughput (tok/s) | 103,059             |

### Samples

Final model (step 3000), seed 0, 200 new tokens.

#### Prompt: 'Once upon a time'

```text
Once upon a time there was a sunflower. She was so happy and alert. She liked to explore the world around her.

One day she was peeking through a wide Bill leaf. She said, "hello there!", and she was so glad. She wanted to make sure that everyone was happy.

Suddenly she heard a voice. It said, "Everything seems peaceful." The sunflower thought she was so joyful. It was a happy sunflower and was so lucky!

flagsing ahead, Here, take a closer look. After a while, she spotted the sunflower. She smiled and said, "I did it!"

The sunflower was so pleased to see her fit and enthusiastic. She thanked the sunflower for letting her see it. Suddenly, the sunflower whinnied away; she was not alone anymore. She serve it with some of her and it became one of the most vibranty wardrobe!
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
- **learning_rate:** 0.0024
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
- **run_name:** lr2.4e-3_s0
- **out_dir:** runs/full/ablations/runs
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('Once upon a time',)
- **sample_tokens:** 200
- **sample_seed:** 0
- **min_learning_rate:** 0.00023999999999999998
- **run_dir:** runs/full/ablations/runs/lr2.4e-3_s0

### Environment

- **date:** 2026-10-16T19:34:52
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-134-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
