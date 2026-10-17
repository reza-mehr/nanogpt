Regions synchronize the device at their boundaries, so times are attributable but slightly inflated: use the shares and op counts here, and the speeds from the benchmarks.

### Step time under the profiler (ms, lower is better)

| variant       | T=512          |
| ------------- | -------------: |
| baseline      | 121.50         |
| +bf16 (cast)  |  13.24 (-89%)  |
| total vs base |        (-89%)  |

### aten op calls per step (lower is better)

| variant       | T=512          |
| ------------- | -------------: |
| baseline      | 10,569         |
| +bf16 (cast)  |  1,146 (-89%)  |
| total vs base |        (-89%)  |

### Share of step time (%)

| region     | baseline T=512 | +bf16 (cast) T=512 |
| ---------- | -------------: | -----------------: |
| embed      |    0.3         |        1.3         |
| layernorm  |    5.7         |       15.6         |
| attn.qkv   |   26.9         |       10.8         |
| attn.cache |    0.0         |        9.3         |
| attn.core  |   46.9         |       12.5         |
| attn.proj  |    2.0         |        9.7         |
| mlp        |    5.3         |       16.4         |
| lm_head    |    2.5         |        1.0         |
| other      |   10.5         |       23.4         |

### Time per step (ms)

| region     | baseline T=512 | +bf16 (cast) T=512 |
| ---------- | -------------: | -----------------: |
| embed      |   0.37         |       0.17         |
| layernorm  |   6.98         |       2.07         |
| attn.qkv   |  32.65         |       1.43         |
| attn.cache |   0.00         |       1.22         |
| attn.core  |  56.96         |       1.65         |
| attn.proj  |   2.39         |       1.29         |
| mlp        |   6.39         |       2.18         |
| lm_head    |   3.02         |       0.13         |
| other      |  12.76         |       3.10         |

### Top ops by self GPU time (baseline T=512)

| op                 | self GPU ms/step | self CPU ms/step | calls/step |
| ------------------ | ---------------: | ---------------: | ---------: |
| aten::addmm        |            8.661 |            0.931 |       37.0 |
| aten::mm           |            7.322 |            8.498 |      432.0 |
| aten::bmm          |            3.566 |           12.622 |      288.0 |
| aten::_softmax     |            1.366 |            1.375 |      144.0 |
| aten::copy_        |            1.135 |            0.814 |      144.0 |
| aten::mul          |            0.628 |            2.286 |      169.0 |
| aten::masked_fill_ |            0.490 |            1.130 |      144.0 |
| aten::bitwise_not  |            0.484 |            1.429 |      144.0 |
| aten::add          |            0.318 |            0.885 |       75.0 |
| aten::var          |            0.279 |            0.465 |       25.0 |

### Top ops by self GPU time (+bf16 (cast) T=512)

| op                             | self GPU ms/step | self CPU ms/step | calls/step |
| ------------------------------ | ---------------: | ---------------: | ---------: |
| aten::addmm                    |            0.363 |            0.839 |       37.0 |
| aten::_flash_attention_forward |            0.203 |            0.232 |       12.0 |
| aten::native_layer_norm        |            0.133 |            0.346 |       25.0 |
| aten::mm                       |            0.096 |            0.255 |       12.0 |
| aten::copy_                    |            0.090 |            0.200 |       24.0 |
| aten::add                      |            0.056 |            0.304 |       25.0 |
| aten::clamp_min                |            0.034 |            0.131 |       12.0 |
| aten::index_select             |            0.009 |            0.020 |        2.0 |
| aten::arange                   |            0.002 |            0.011 |        2.0 |
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
- **mode:** decode
- **variants:** ()
- **block_sizes:** (512,)
- **wait:** 1
- **active:** 3
- **top_ops:** 10
- **export_trace:** True
- **batch:** 1
- **kind:** profile_decode
- **run_dir:** results/full/profile_decode_stories_cfg_cuda
- **selected:** ['baseline', '+bf16 (cast)']

### Environment

- **date:** 2026-10-16T03:14:03
- **git:** v0.1.0-1-g53a1038
- **python:** 3.12.3
- **torch:** 2.14.0+cu130
- **os:** Linux-6.8.0-124-generic-x86_64-with-glibc2.39
- **cpu:** x86_64
- **ram (GB):** 2016
- **torch cpu threads:** 12
- **device:** cuda
- **gpu:** NVIDIA A100-SXM4-80GB, 79 GB
