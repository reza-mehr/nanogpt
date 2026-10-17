### Summary: lr1.2e-3_s0

| metric                  | value               |
| ----------------------- | ------------------- |
| steps                   | 3,000               |
| tokens                  | 98,304,000          |
| final train loss        | 1.4934              |
| final val loss          | 1.5032              |
| best val loss           | 1.5032 (step 3,000) |
| parameters              | 162,273,361         |
| training time (s)       | 941                 |
| mean throughput (tok/s) | 104,421             |

### Samples

Final model (step 3000), seed 0, 200 new tokens.

#### Prompt: 'Once upon a time'

```text
Once upon a time there was a sunflower that began to grow. She looked up at the sun and said together ugly words. The sunflower was frustrated and tried to be careful with Bill for a reason.

One day a farmer came to visit. He said, " repairing the sunflower so no one else could have it." The sunflower felt bad and started to cry.

The farmer thought of a plan to make the sunflower happy again. She was brave and strong, and then he removed the sunflower from the sunflower. After the while, the sunflower closed her eyes and smiled.

The sunflower looked as pretty as ever. With a loud splash, the sunflower opened up and let her jump around it. She accepted, and happily with each bite Anna grew.
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
- **learning_rate:** 0.0012
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
- **run_name:** lr1.2e-3_s0
- **out_dir:** runs/full/ablations/runs
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('Once upon a time',)
- **sample_tokens:** 200
- **sample_seed:** 0
- **min_learning_rate:** 0.00011999999999999999
- **run_dir:** runs/full/ablations/runs/lr1.2e-3_s0

### Environment

- **date:** 2026-10-16T07:18:42
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-124-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
