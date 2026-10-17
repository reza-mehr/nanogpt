### Summary: kv4_s0

| metric                  | value               |
| ----------------------- | ------------------- |
| steps                   | 3,000               |
| tokens                  | 98,304,000          |
| final train loss        | 1.5655              |
| final val loss          | 1.5738              |
| best val loss           | 1.5738 (step 3,000) |
| parameters              | 152,836,177         |
| training time (s)       | 872                 |
| mean throughput (tok/s) | 112,739             |

### Samples

Final model (step 3000), seed 0, 200 new tokens.

#### Prompt: 'Once upon a time'

```text
Once upon a time there was a sunflower that began to bloom. It looked so bright and happy and ready to bloom. Everyone wanted to take a check- pear angel. So one day when everyone looked he saw the sunflower that was very busy. 

Suddenly, the sunflower started to feel very hot. It felt so hot! Everyone around it got very hot and it began to freeze. The sunflower began to melt away from the seed and smoke. 

After some time, the sunflower finally changed once again. It was so hot that everyone had to stop it! But the ribbon kept wandering and there were no more dirt to clean it. 

Soon the sunflower had finally grown much brighter and the heat was still very hot. Everyone was happy and the sunflower was always helping to find more water and heat for everyone!
```

### Configuration

- **data.dir:** data/tinystories
- **model.name:** gpt
- **model.vocab_size:** 50257
- **model.block_size:** 512
- **model.n_embed:** 768
- **model.n_blocks:** 12
- **model.head_num:** 12
- **model.kv_head_num:** 4
- **model.head_size:** 64
- **model.batched_attention:** True
- **model.scaled_dot_prod:** True
- **model.pos_embed:** rope
- **model.rope_base:** 10000.0
- **model.ln_torch:** True
- **model.dropout:** 0.0
- **model.n_kv_head:** 4
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
- **seed:** 0
- **preset:** stories_cfg
- **run_name:** kv4_s0
- **out_dir:** runs/full/ablations/runs
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('Once upon a time',)
- **sample_tokens:** 200
- **sample_seed:** 0
- **min_learning_rate:** 5.9999999999999995e-05
- **run_dir:** runs/full/ablations/runs/kv4_s0

### Environment

- **date:** 2026-10-16T03:34:51
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-124-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
