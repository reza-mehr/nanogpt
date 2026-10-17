Regions synchronize the device at their boundaries, so times are attributable but slightly inflated: use the shares and op counts here, and the speeds from the benchmarks.

### Step time under the profiler (ms, lower is better)

| variant       | T=512          |
| ------------- | -------------: |
| baseline      | 106.11         |
| +bf16 (cast)  |  14.01 (-87%)  |
| total vs base |        (-87%)  |

### aten op calls per step (lower is better)

| variant       | T=512          |
| ------------- | -------------: |
| baseline      | 11,001         |
| +bf16 (cast)  |  1,182 (-89%)  |
| total vs base |        (-89%)  |

### Share of step time (%)

| region     | baseline T=512 | +bf16 (cast) T=512 |
| ---------- | -------------: | -----------------: |
| embed      |    0.2         |        1.6         |
| layernorm  |    5.2         |       14.5         |
| attn.qkv   |   25.5         |       10.3         |
| attn.cache |    0.0         |        8.9         |
| attn.core  |   48.7         |       11.7         |
| attn.proj  |    1.9         |        9.2         |
| mlp        |    5.6         |       14.7         |
| lm_head    |    2.5         |        6.4         |
| other      |   10.4         |       22.7         |

### Time per step (ms)

| region     | baseline T=512 | +bf16 (cast) T=512 |
| ---------- | -------------: | -----------------: |
| embed      |   0.23         |       0.23         |
| layernorm  |   5.50         |       2.04         |
| attn.qkv   |  27.04         |       1.44         |
| attn.cache |   0.00         |       1.25         |
| attn.core  |  51.67         |       1.64         |
| attn.proj  |   1.97         |       1.29         |
| mlp        |   5.91         |       2.06         |
| lm_head    |   2.70         |       0.90         |
| other      |  11.09         |       3.18         |

### Top ops by self GPU time (baseline T=512)

| op                 | self GPU ms/step | self CPU ms/step | calls/step |
| ------------------ | ---------------: | ---------------: | ---------: |
| aten::addmm        |            8.311 |            0.761 |       37.0 |
| aten::mm           |            7.288 |            7.698 |      432.0 |
| aten::bmm          |            3.530 |           11.423 |      288.0 |
| aten::_softmax     |            1.204 |            1.328 |      144.0 |
| aten::bitwise_not  |            0.925 |            1.426 |      144.0 |
| aten::mul          |            0.644 |            2.237 |      169.0 |
| aten::masked_fill_ |            0.496 |            1.054 |      144.0 |
| aten::copy_        |            0.450 |            0.757 |      144.0 |
| aten::add          |            0.315 |            0.758 |       75.0 |
| aten::var          |            0.278 |            0.322 |       25.0 |

### Top ops by self GPU time (+bf16 (cast) T=512)

| op                             | self GPU ms/step | self CPU ms/step | calls/step |
| ------------------------------ | ---------------: | ---------------: | ---------: |
| aten::addmm                    |            1.630 |            0.793 |       37.0 |
| aten::_flash_attention_forward |            0.310 |            0.227 |       12.0 |
| aten::mm                       |            0.238 |            0.258 |       12.0 |
| aten::native_layer_norm        |            0.169 |            0.322 |       25.0 |
| aten::copy_                    |            0.141 |            0.193 |       24.0 |
| aten::add                      |            0.087 |            0.304 |       25.0 |
| aten::clamp_min                |            0.068 |            0.131 |       12.0 |
| aten::gather                   |            0.009 |            0.025 |        2.0 |
| aten::arange                   |            0.002 |            0.023 |        2.0 |
| aten::embedding                |            0.000 |            0.007 |        2.0 |

### Configuration

- **preset:** stories_cfg
- **device:** cuda
- **vocab_size:** 50257
- **batch_size:** None
- **warmup:** 2
- **out_dir:** results/full
- **from_csv:** False
- **overwrite:** False
- **log_level:** 20
- **mode:** prefill
- **variants:** ()
- **block_sizes:** (512,)
- **wait:** 1
- **active:** 3
- **top_ops:** 10
- **export_trace:** True
- **batch:** 1
- **kind:** profile_prefill
- **run_dir:** results/full/profile_prefill_stories_cfg_cuda
- **selected:** ['baseline', '+bf16 (cast)']

### Environment

- **date:** 2026-10-16T03:13:51
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-124-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
