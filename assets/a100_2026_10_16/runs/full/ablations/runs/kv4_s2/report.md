### Summary: kv4_s2

| metric                  | value               |
| ----------------------- | ------------------- |
| steps                   | 3,000               |
| tokens                  | 98,304,000          |
| final train loss        | 1.5712              |
| final val loss          | 1.5741              |
| best val loss           | 1.5741 (step 3,000) |
| parameters              | 152,836,177         |
| training time (s)       | 870                 |
| mean throughput (tok/s) | 112,950             |

### Samples

Final model (step 3000), seed 0, 200 new tokens.

#### Prompt: 'Once upon a time'

```text
Once upon a time there was a sunflower. It was so bright and looked so bright. It made the sun's rays melt and beauty. Inside the sunflower lived a very special Bill for everyone.

One day the sunflower decided it wanted to take a break from the sun. But when the sun went down, it slowly began to burn. The sun was very hot and there were lots of fire- valuing Bella in the sky. She was scared and was so hurt herself.

The sunflower ran away from the heat, but after a while, the sun came out and it was getting dark. The sunflower tried to run but it remembered how to feel calm. Every day the sun would rise, but still the heat from the sun helped out the heat.

One night, while the sun was in the dark, the sun came out from above. The sun and the moon felt warm and happy. The sunflower knew that it would always remember the good sun and the sun
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
- **seed:** 2
- **preset:** stories_cfg
- **run_name:** kv4_s2
- **out_dir:** runs/full/ablations/runs
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('Once upon a time',)
- **sample_tokens:** 200
- **sample_seed:** 0
- **min_learning_rate:** 5.9999999999999995e-05
- **run_dir:** runs/full/ablations/runs/kv4_s2

### Environment

- **date:** 2026-10-16T06:04:02
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-124-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
