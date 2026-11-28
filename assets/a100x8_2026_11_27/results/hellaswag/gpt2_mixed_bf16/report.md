### HellaSwag

| Model | Examples | acc_norm | Reference | Time (s) |
| ----- | -------: | -------: | --------: | -------: |
| gpt2  |   10,042 |   0.2966 |    0.2955 |    536.3 |

### Configuration

- **eval.data_dir:** data/hellaswag
- **eval.batch_size:** 8
- **eval.batch_num:** None
- **eval.gpt2_pretrained:** gpt2
- **eval.checkpoint:** None
- **eval.device:** mps
- **eval.matmul_precision:** fp32
- **eval.precision:** mixed_bf16
- **eval.out_dir:** results/hellaswag
- **eval.run_name:** None
- **eval.overwrite:** False
- **eval.model_name:** gpt2
- **eval.run_dir:** results/hellaswag/gpt2_mixed_bf16
- **model.name:** gpt2
- **model.vocab_size:** 50257
- **model.pad_vocab_to_multiple:** 1
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
- **model.vocab_size_padded:** 50257

### Environment

- **date:** 2026-11-27T20:54:00
- **git:** gpt2-run-2026-11-27-1-g54f2ecc
- **python:** 3.12.14
- **torch:** 2.14.1
- **os:** macOS-26.7.3-arm64-arm-64bit
- **cpu:** Apple M4
- **ram (GB):** 24
- **torch cpu threads:** 4
- **world size:** 1
- **device:** mps
- **power:** Now drawing from 'AC Power'
