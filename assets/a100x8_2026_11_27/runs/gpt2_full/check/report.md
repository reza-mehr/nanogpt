### Summary: check

| metric                  | value           |
| ----------------------- | --------------- |
| parameters              | 124,475,904     |
| steps                   | 6               |
| training time (s)       | 15              |
| tokens                  | 3,145,728       |
| final train loss        | 8.8303          |
| final val loss          | 8.8207          |
| best val loss           | 8.8207 (step 6) |
| mean throughput (tok/s) | 1,669,864       |
| peak GPU memory (GiB)   | 40.0            |
| mean MFU (%)            | 57.2            |
| HellaSwag accuracy      | 0.1875          |

### Samples

Final model (step 6), seed 0, 16 new tokens.

#### Prompt: ''

```text
 Visit10306 competitionsolla refund results essentialsy cookrentice Orc beralien continent Nas
```

#### Prompt: "Hello, I'm a language model,"

```text
Hello, I'm a language model, Visit10306 competitionsolla refund results essentialsy cookrentice Orc beralien continent Nas
```

### Configuration

- **data.dir:** data/fineweb
- **model.name:** gpt2
- **model.vocab_size:** 50257
- **model.pad_vocab_to_multiple:** 128
- **model.block_size:** 1024
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
- **model.vocab_size_padded:** 50304
- **hellaswag.eval_freq:** 1
- **hellaswag.data_dir:** data/hellaswag
- **hellaswag.batch_size:** 16
- **hellaswag.batch_num:** 2
- **hellaswag.seed:** 12
- **batch_size:** 64
- **batch_tokens:** 524288
- **batch_mode:** sequential
- **max_iters:** 6
- **lr_schedule:** cosine
- **learning_rate:** 0.0006
- **lr_warmup_iters:** 2
- **grad_clip:** 1.0
- **weight_decay:** 0.1
- **decay_groups:** True
- **beta1:** 0.9
- **beta2:** 0.95
- **adamw_fused:** True
- **eval_interval:** 3
- **eval_iters:** 2
- **ckpt_freq:** 1
- **device:** cuda
- **matmul_precision:** tf32
- **precision:** mixed_bf16
- **compile_model:** True
- **compile_order:** after_ddp
- **seed:** 1337
- **preset:** gpt2_cfg
- **run_name:** check
- **out_dir:** runs/gpt2_full
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('', "Hello, I'm a language model,")
- **sample_tokens:** 16
- **sample_seed:** 0
- **sample_freq:** 1
- **ckpt_interval:** 3
- **min_learning_rate:** 5.9999999999999995e-05
- **run_dir:** runs/gpt2_full/check

### Environment

- **date:** 2026-11-27T04:24:35
- **git:** v1.0.0-39-g96be5de
- **python:** 3.12.3
- **torch:** 2.14.1+cu126
- **os:** Linux-6.14.0-24-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 945
- **torch cpu threads:** 24
- **world size:** 8
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
