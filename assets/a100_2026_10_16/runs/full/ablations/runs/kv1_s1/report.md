### Summary: kv1_s1

| metric                  | value               |
| ----------------------- | ------------------- |
| steps                   | 3,000               |
| tokens                  | 98,304,000          |
| final train loss        | 1.5695              |
| final val loss          | 1.5880              |
| best val loss           | 1.5880 (step 3,000) |
| parameters              | 149,297,233         |
| training time (s)       | 833                 |
| mean throughput (tok/s) | 117,958             |

### Samples

Final model (step 3000), seed 0, 200 new tokens.

#### Prompt: 'Once upon a time'

```text
Once upon a time there was a sunflower. One day, the sunflower spotted a tiny flower growing in her garden. She wanted to take a closer look, so she bent over to pick it.

Suddenly, the sunflower felt a sharp pain. She was feeling very scared. But then she wanted to taste the little flower, so she looked around. She saw a flower, a razor and thought she could get take one.

The sunflower mixed up all the small blooms that the little flower Here was once again. Very carefully, the little flower was still alive. She felt much more brave and didn't feel frightened anymore.

The little flower was no longer scared. It fit and was much brighter than ever before! The little flower was very happy with what she had done.
```

### Configuration

- **data.dir:** data/tinystories
- **model.name:** gpt
- **model.vocab_size:** 50257
- **model.block_size:** 512
- **model.n_embed:** 768
- **model.n_blocks:** 12
- **model.head_num:** 12
- **model.kv_head_num:** 1
- **model.head_size:** 64
- **model.batched_attention:** True
- **model.scaled_dot_prod:** True
- **model.pos_embed:** rope
- **model.rope_base:** 10000.0
- **model.ln_torch:** True
- **model.dropout:** 0.0
- **model.n_kv_head:** 1
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
- **seed:** 1
- **preset:** stories_cfg
- **run_name:** kv1_s1
- **out_dir:** runs/full/ablations/runs
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('Once upon a time',)
- **sample_tokens:** 200
- **sample_seed:** 0
- **min_learning_rate:** 5.9999999999999995e-05
- **run_dir:** runs/full/ablations/runs/kv1_s1

### Environment

- **date:** 2026-10-16T05:08:25
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-124-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
