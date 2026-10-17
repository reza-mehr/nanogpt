### Summary: stories

| metric                  | value                |
| ----------------------- | -------------------- |
| steps                   | 10,000               |
| tokens                  | 327,680,000          |
| final train loss        | 1.3256               |
| final val loss          | 1.3434               |
| best val loss           | 1.3434 (step 10,000) |
| parameters              | 162,273,361          |
| training time (s)       | 3,135                |
| mean throughput (tok/s) | 104,540              |

### Samples

Final model (step 10000), seed 0, 200 new tokens.

#### Prompt: ''

```text
Once upon a time, there was a man named Tom. Tom had a deep television in his room. Every night, he would watch his favorite cartoon. Tom's Bill would always win.

One night, while Tom was watching his television, he felt the power pass onto his television. The power could save him from making his TV melt. Tom was scared.

Tom went to his mom and dad. They told him to be brave and strong. They said, "flags is good. Here, let's save the television." They bought a new television and closed the television. Tom was happy. He knew he could face his clean, cool television again.
```

#### Prompt: 'Once upon a time'

```text
Once upon a time there was a sunflower that began to bloom. It looked so bright and happy. The sunflower was very happy but it was also uncomfortable. It felt heavy and Bill could not seem to take it for the rest of the week. 

Suddenly, the sunflower started to feel very sick. It wanted to help all the animals around it. Soon a mouse got better at the flower festival and it came up with many new ideas. 

The town animals were very happy to get the sunflower back. They decided to have the party easy to protect the sunflower and everyone in the town. The sunflower was so thankful and it bloomed into a big, beautiful flower fit for everyone to enjoy. 

The sun helped out the sunflower and woke it up to shine brightly! The sunflower felt so comfortable and it was very glad to be back and forth, ready to bloom and enjoy the cool mail.
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
- **max_iters:** 10000
- **lr_schedule:** cosine
- **learning_rate:** 0.0006
- **lr_warmup_iters:** 500
- **grad_clip:** 1.0
- **weight_decay:** 0.1
- **decay_groups:** True
- **beta1:** 0.9
- **beta2:** 0.95
- **adamw_fused:** True
- **eval_interval:** 500
- **eval_iters:** 100
- **ckpt_interval:** 500
- **device:** cuda
- **precision:** mixed_bf16
- **seed:** 1337
- **preset:** stories_cfg
- **run_name:** stories
- **out_dir:** runs/full
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('', 'Once upon a time')
- **sample_tokens:** 200
- **sample_seed:** 0
- **min_learning_rate:** 5.9999999999999995e-05
- **run_dir:** runs/full/stories

### Environment

- **date:** 2026-10-16T01:52:31
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-124-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
