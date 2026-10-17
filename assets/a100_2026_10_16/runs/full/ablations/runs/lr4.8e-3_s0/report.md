### Summary: lr4.8e-3_s0

| metric                  | value               |
| ----------------------- | ------------------- |
| steps                   | 3,000               |
| tokens                  | 98,304,000          |
| final train loss        | 1.4826              |
| final val loss          | 1.4918              |
| best val loss           | 1.4918 (step 3,000) |
| parameters              | 162,273,361         |
| training time (s)       | 953                 |
| mean throughput (tok/s) | 103,190             |

### Samples

Final model (step 3000), seed 0, 200 new tokens.

#### Prompt: 'Once upon a time'

```text
Once upon a time there was a sunflower. The sunflower liked being looked up at the night sky and together it would carry garbage to make it look pretty. One day, the sunflower saw a little bird that wasn't working in its shopper.

The sunflower asked the bird, "Are you feeling better?"

The bird replied, "No! I am not feeling well. I need to get new wood to cross the river."

The sunflower felt sad for the bird and said, "I can help you get some wood for you. I will teach you how to use it."

The little bird was very happy and said, "Thank you, sunflower! You are much brighter than ever!"

The sunflower smiled and said, "You're welcome. I am happy to have help." 

That night, the sunflower and the bird rested together in the warm light that helped them fight across the morning. They were cheerful and happy,
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
- **learning_rate:** 0.0048
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
- **run_name:** lr4.8e-3_s0
- **out_dir:** runs/full/ablations/runs
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('Once upon a time',)
- **sample_tokens:** 200
- **sample_seed:** 0
- **min_learning_rate:** 0.00047999999999999996
- **run_dir:** runs/full/ablations/runs/lr4.8e-3_s0

### Environment

- **date:** 2026-10-16T19:59:02
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-134-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
