### Summary: main

| metric                  | value                |
| ----------------------- | -------------------- |
| parameters              | 124,475,904          |
| steps                   | 19,073               |
| training time (s)       | 6,063                |
| tokens                  | 9,999,745,024        |
| final train loss        | 3.0450               |
| final val loss          | 3.0604               |
| best val loss           | 3.0598 (step 18,500) |
| mean throughput (tok/s) | 1,653,707            |
| peak GPU memory (GiB)   | 40.0                 |
| mean MFU (%)            | 56.7                 |
| HellaSwag accuracy      | 0.3014               |

### Samples

Final model (step 19073), seed 0, 200 new tokens.

#### Prompt: ''

```text
Notes on the history of SCT: Scent, Orc, Museemuar, Hicipalceromust.
Known for painting ‘soda’.
Oxonet, the horse-guatta.
Contrary to the linearity of the work, slopes from SOS are virtually linear, making them more spiral.
Everything seems linear, except at the peaks where the snow becomes very cold (e.g. Atofer’s Creichflags).
The Herero once again is not linear; the nothing is (in the closed form) linear.
The Braque offers a very hardcore hierarchy of facilities.
From obtruding to the lowest entry, much of the work creates a rough line, much of it with steepness for the whortle to go in the darkest of winter landscape.
Stonkeys and Nagille are traditionally associated with the lowland regions of older Historic Europe nor are they symbols of their ancient British
```

#### Prompt: "Hello, I'm a language model,"

```text
Hello, I'm a language model, language model and language implementers.
yarnrentice writes `carole from television: together cd = new garbage = new string = text xor table soda = new string soda > soda; desjraf = new thread
in table soda, as herbade, slopes from computer program soda, all eggs are compiled here.
this table is created by executing a list of entire operands not required by the given operand*.
hop is new O( soda Here, once 'a。..'" apper″
this program produces a warning text in 'then>.
keepfrom repeats ['bellmaidUp:~ee> to add the program entry
'and keeps the previous entry to the Cell we use for the result>
If all of the objects for this program exist, then com col serve it with the same interpretation and output 'After this, you're entering the first part of the program nor are you referencing the second part of
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
- **hellaswag.batch_size:** 64
- **hellaswag.batch_num:** None
- **hellaswag.seed:** 12
- **batch_size:** 64
- **batch_tokens:** 524288
- **batch_mode:** sequential
- **max_iters:** 19073
- **lr_schedule:** cosine
- **learning_rate:** 0.0006
- **lr_warmup_iters:** 715
- **grad_clip:** 1.0
- **weight_decay:** 0.1
- **decay_groups:** True
- **beta1:** 0.9
- **beta2:** 0.95
- **adamw_fused:** True
- **eval_interval:** 250
- **eval_iters:** 100
- **ckpt_freq:** 1
- **device:** cuda
- **matmul_precision:** tf32
- **precision:** mixed_bf16
- **compile_model:** True
- **compile_order:** after_ddp
- **seed:** 1337
- **preset:** gpt2_cfg
- **run_name:** main
- **out_dir:** runs/gpt2_full
- **resume:** False
- **overwrite:** False
- **log_level:** 20
- **sample_prompts:** ('', "Hello, I'm a language model,")
- **sample_tokens:** 200
- **sample_seed:** 0
- **sample_freq:** 4
- **ckpt_interval:** 250
- **min_learning_rate:** 5.9999999999999995e-05
- **run_dir:** runs/gpt2_full/main

### Environment

- **date:** 2026-11-27T04:44:44
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
