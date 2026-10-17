### Decode throughput (tokens/sec)

| variant              | T=128       | T=512       |
| -------------------- | ----------: | ----------: |
| baseline             |  23         |  23         |
| +batched attn        | 126 (+445%) |  78 (+234%) |
| +torch LN            | 165 (+31%)  |  84 (+8%)   |
| +SDPA                | 203 (+23%)  |  99 (+18%)  |
| +KV cache            | 234 (+15%)  | 240 (+143%) |
| +bf16 (cast)         | 242 (+4%)   | 232 (-3%)   |
| +autocast bf16 (alt) | 185 (-21%)  | 180 (-25%)  |
| total vs base        |     (+947%) |     (+897%) |

### Time to first token (ms, lower is better)

| variant              | T=128        | T=512        |
| -------------------- | -----------: | -----------: |
| baseline             | 47.2         | 47.6         |
| +batched attn        |  8.9 (-81%)  | 12.8 (-73%)  |
| +torch LN            |  7.0 (-21%)  | 12.0 (-7%)   |
| +SDPA                |  4.8 (-31%)  | 10.7 (-10%)  |
| +KV cache            |  5.8 (+20%)  | 10.8 (+1%)   |
| +bf16 (cast)         |  5.0 (-14%)  |  5.6 (-48%)  |
| +autocast bf16 (alt) |  6.2 (+7%)   |  7.6 (-30%)  |
| total vs base        |      (-89%)  |      (-88%)  |

### Memory after generation (MB)

| variant              | T=128       | T=512       |
| -------------------- | ----------: | ----------: |
| baseline             | 750         | 901         |
| +batched attn        | 708 (-6%)   | 858 (-5%)   |
| +torch LN            | 708 (0%)    | 858 (0%)    |
| +SDPA                | 705 (-1%)   | 855 (0%)    |
| +KV cache            | 702 (0%)    | 777 (-9%)   |
| +bf16 (cast)         | 363 (-48%)  | 401 (-48%)  |
| +autocast bf16 (alt) | 738 (+5%)   | 779 (0%)    |
| total vs base        |     (-52%)  |     (-55%)  |

### Configuration

- **preset:** stories_cfg
- **device:** cuda
- **vocab_size:** 50257
- **batch_size:** None
- **warmup:** 3
- **out_dir:** results/full
- **from_csv:** False
- **overwrite:** False
- **log_level:** 20
- **suite:** journey
- **prompt_lens:** (128, 512)
- **new_tokens:** 64
- **precision:** bf16
- **trials:** 3
- **batch:** 1
- **kind:** benchmark_generate
- **run_dir:** results/full/benchmark_generate_stories_cfg_cuda

### Environment

- **date:** 2026-10-16T03:11:32
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-124-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
