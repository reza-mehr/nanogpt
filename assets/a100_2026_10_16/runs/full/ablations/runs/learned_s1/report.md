### Summary: learned_s1

| metric                  | value               |
| ----------------------- | ------------------- |
| steps                   | 3,000               |
| tokens                  | 98,304,000          |
| final train loss        | 1.6455              |
| final val loss          | 1.6602              |
| best val loss           | 1.6602 (step 3,000) |
| parameters              | 162,666,577         |
| training time (s)       | 781                 |
| mean throughput (tok/s) | 125,812             |

### Samples

Final model (step 3000), seed 0, 200 new tokens.

#### Prompt: 'Once upon a time'

```text
Once upon a time there was a sunflower. One day, the sunflower wanted to bloom from the birds together. But the birds did not ask the sunflower to come out. Bill was so sad.

Then the sunflower felt an idea. He wanted to fly even higher. So he flew higher and higher, higher and higher. Eventually he came out of the sun and got a better look.

The sunflower was so happy and thanked the sun for giving him something. But then he realized he had too much joy.

The sunflower asked the sunflower, "Do you want to rise with me again?"

The sunflower smiled and said, "Yes, of course!"

So Bill closed his eyes to fly, and he was very happy with his hard work. He thanked the sunflower and flew away with lots of joy.

The sunflower had learned an important lesson that sometimes the most perfect things can bring joy to causing good things!
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
- **model.pos_embed:** learned
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
- **seed:** 1
- **preset:** stories_cfg
- **run_name:** learned_s1
- **out_dir:** runs/full/ablations/runs
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('Once upon a time',)
- **sample_tokens:** 200
- **sample_seed:** 0
- **min_learning_rate:** 5.9999999999999995e-05
- **run_dir:** runs/full/ablations/runs/learned_s1

### Environment

- **date:** 2026-10-16T05:26:33
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-124-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
