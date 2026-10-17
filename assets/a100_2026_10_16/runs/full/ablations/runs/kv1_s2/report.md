### Summary: kv1_s2

| metric                  | value               |
| ----------------------- | ------------------- |
| steps                   | 3,000               |
| tokens                  | 98,304,000          |
| final train loss        | 1.5783              |
| final val loss          | 1.5805              |
| best val loss           | 1.5805 (step 3,000) |
| parameters              | 149,297,233         |
| training time (s)       | 835                 |
| mean throughput (tok/s) | 117,703             |

### Samples

Final model (step 3000), seed 0, 200 new tokens.

#### Prompt: 'Once upon a time'

```text
Once upon a time there was a sunflower. One day, it started to shine bright. It fluttered together in the middle of beauty. Inside the sun, angel took out a Bill leaf.

The leaf spoke to him in a gentle voice.

"Hi, Dan," the leaf said.

Dan was very excited! He looked around and saw a little girl playing in the park.

The little girl smiled.

"Can I play too?" Dan asked.

The girl Here, with a smile.

"Not easy," the girl said.

Dan was feeling a little angry and pushed the girl clean, so she brought him over to play.

The girl was very happy and thanked the little girl. She said, "You are a very kind and obedient friend. I help you remember how pretty you are."

The and Dan agreed. They played and laughed together the rest of the day until it was time to go home.
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
- **seed:** 2
- **preset:** stories_cfg
- **run_name:** kv1_s2
- **out_dir:** runs/full/ablations/runs
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('Once upon a time',)
- **sample_tokens:** 200
- **sample_seed:** 0
- **min_learning_rate:** 5.9999999999999995e-05
- **run_dir:** runs/full/ablations/runs/kv1_s2

### Environment

- **date:** 2026-10-16T06:22:58
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-124-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
