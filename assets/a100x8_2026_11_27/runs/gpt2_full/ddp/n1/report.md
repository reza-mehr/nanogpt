### Summary: n1

| metric                  | value            |
| ----------------------- | ---------------- |
| parameters              | 124,475,904      |
| steps                   | 60               |
| training time (s)       | 167              |
| tokens                  | 31,457,280       |
| final train loss        | 7.3546           |
| final val loss          | 7.3711           |
| best val loss           | 7.3711 (step 60) |
| mean throughput (tok/s) | 220,291          |
| peak GPU memory (GiB)   | 41.1             |
| mean MFU (%)            | 60.4             |

### Samples

Final model (step 60), seed 0, 200 new tokens.

#### Prompt: ''

```text
 � pressure06 competitionsolla at results:y to building with beingcar. improve televisionan togethericip Article from garbage are- othercos that x � be date for Bill for resulted, he parts, be the des- feces of theseper, aRoute — the torchesaz herb work 23 slopes immigrants. human managementoves enrol all eggs... one. acquired. a Computerboxters in be offensive- valdefinition take other84 not behavior a be threeAMofer region town for the specificflags to first we Here that once ' half。une 2020itis while recent tooth as the closed have nucleiff Dartmouth- otherType to pursue he repeats, … clean, fungi to cooperation of door toJuly of fit storm
 much chamber conflict. 26 world to your Cell we and R with captains have Ceresize the Dar helping to theirTrans have helpoyd of probably col serve it testimony market; therefore and Nag, the major company and containsy the mail resusc older Seal of nor+ causing or High hold can She
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
- **hellaswag.eval_freq:** 0
- **hellaswag.data_dir:** data/hellaswag
- **hellaswag.batch_size:** 64
- **hellaswag.batch_num:** None
- **hellaswag.seed:** 12
- **batch_size:** 64
- **batch_tokens:** 524288
- **batch_mode:** sequential
- **max_iters:** 60
- **lr_schedule:** constant
- **learning_rate:** 0.0006
- **lr_warmup_iters:** 500
- **grad_clip:** 1.0
- **weight_decay:** 0.1
- **decay_groups:** True
- **beta1:** 0.9
- **beta2:** 0.95
- **adamw_fused:** True
- **eval_interval:** 10
- **eval_iters:** 5
- **ckpt_freq:** 0
- **device:** cuda
- **matmul_precision:** tf32
- **precision:** mixed_bf16
- **compile_model:** True
- **compile_order:** after_ddp
- **seed:** 1337
- **preset:** gpt2_cfg
- **run_name:** n1
- **out_dir:** runs/gpt2_full/ddp
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('',)
- **sample_tokens:** 200
- **sample_seed:** 0
- **sample_freq:** 0
- **ckpt_interval:** 0
- **min_learning_rate:** 5.9999999999999995e-05
- **run_dir:** runs/gpt2_full/ddp/n1

### Environment

- **date:** 2026-11-27T04:35:25
- **git:** v1.0.0-39-g96be5de
- **python:** 3.12.3
- **torch:** 2.14.1+cu126
- **os:** Linux-6.14.0-24-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 945
- **torch cpu threads:** 24
- **world size:** 1
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
