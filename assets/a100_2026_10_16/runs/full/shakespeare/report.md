### Summary: shakespeare

| metric                  | value               |
| ----------------------- | ------------------- |
| steps                   | 5,000               |
| tokens                  | 81,920,000          |
| final train loss        | 1.0682              |
| final val loss          | 1.4768              |
| best val loss           | 1.4768 (step 5,000) |
| parameters              | 10,788,929          |
| training time (s)       | 385                 |
| mean throughput (tok/s) | 212,938             |

### Samples

Final model (step 5000), seed 0, 200 new tokens.

#### Prompt: ''

```text
Ay, you not like more.

ANGELO:
What has made news?
No, for my lord?

PAULINA:
You shall not go; no, no more mercy,
You content not trust upon that.

FLORIZEL:
Now, by my brother:
Romeo brave citizens
```

#### Prompt: 'ROMEO:'

```text
ROMEO:
Save I not, bear thee!
And, my heart! murder welcome! how show thy fair love haved!

JULIET:
You love this the shadows content and drums of this?
If your servant Nature and Tybalt,
You did be ill-but
```

### Configuration

- **data.dir:** data/tinyshakespeare
- **model.name:** gpt
- **model.vocab_size:** 65
- **model.block_size:** 256
- **model.n_embed:** 384
- **model.n_blocks:** 6
- **model.head_num:** 6
- **model.kv_head_num:** None
- **model.head_size:** 64
- **model.batched_attention:** True
- **model.scaled_dot_prod:** True
- **model.pos_embed:** learned
- **model.rope_base:** 10000.0
- **model.ln_torch:** True
- **model.dropout:** 0.2
- **model.n_kv_head:** 6
- **batch_size:** 64
- **max_iters:** 5000
- **lr_schedule:** constant
- **learning_rate:** 0.0003
- **lr_warmup_iters:** 100
- **grad_clip:** inf
- **weight_decay:** 0.01
- **decay_groups:** False
- **beta1:** 0.9
- **beta2:** 0.999
- **adamw_fused:** True
- **eval_interval:** 500
- **eval_iters:** 200
- **ckpt_interval:** 500
- **device:** cuda
- **precision:** fp32
- **seed:** 1337
- **preset:** main_cfg
- **run_name:** shakespeare
- **out_dir:** runs/full
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('', 'ROMEO:')
- **sample_tokens:** 200
- **sample_seed:** 0
- **min_learning_rate:** 2.9999999999999997e-05
- **run_dir:** runs/full/shakespeare

### Environment

- **date:** 2026-10-16T01:44:23
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-124-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
