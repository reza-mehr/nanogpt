### Summary: baseline_s1

| metric                  | value               |
| ----------------------- | ------------------- |
| steps                   | 3,000               |
| tokens                  | 98,304,000          |
| final train loss        | 1.5573              |
| final val loss          | 1.5766              |
| best val loss           | 1.5766 (step 3,000) |
| parameters              | 162,273,361         |
| training time (s)       | 940                 |
| mean throughput (tok/s) | 104,615             |

### Samples

Final model (step 3000), seed 0, 200 new tokens.

#### Prompt: 'Once upon a time'

```text
Once upon a time there was a sunflower. One day, the sunflower spotted a little girl. The girl was wearing a purple dress. 

The angel smiled and said Bill could. The sunflower gave her the beautiful orange crown. 

The girl wanted the crown so badly that she wanted one. But then she remembered all her birthday. 

She went to the kitchen and saw a bowl of broccoli. It smelled really fresh and smelled good. The girl was excited and asked the bowl " Here, can I have this?". 

The tooth fairy was very pleased and said "Yes, you can have it." 

So, the girl's parents were so happy that they rewarded the girl with the special purple cake. And so the girl's mom woke Anna up to the sunflower for breakfast. 
 
From then on, the girl knew that even though the brilliant sunflower, she still wanted special older Seal of it.
```

### Configuration

- **data.dir:** data/tinystories
- **model.name:** gpt
- **model.vocab_size:** 50257
- **model.block_size:** 512
- **model.n_embed:** 768
- **model.n_blocks:** 12
- **model.head_num:** 12
- **model.kv_head_num:** 12
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
- **run_name:** baseline_s1
- **out_dir:** runs/full/ablations/runs
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('Once upon a time',)
- **sample_tokens:** 200
- **sample_seed:** 0
- **min_learning_rate:** 5.9999999999999995e-05
- **run_dir:** runs/full/ablations/runs/baseline_s1

### Environment

- **date:** 2026-10-16T04:29:02
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-124-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
