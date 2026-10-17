# NanoGPT

![tests](https://github.com/reza-mehr/nanogpt/actions/workflows/tests.yml/badge.svg)

A decoder-only Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) built from scratch following [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY), then extended with modern architecture, efficiency, and inference features, and measured end to end on one A100: a 162.3M-parameter model trained on TinyStories in 52 minutes, plus benchmarks, profiles, and seeded ablations, all reproducible with one script.

## Beyond the Tutorial

- **Architecture:** rotary position embeddings (RoPE), grouped-query attention (GQA/MQA).
- **Training efficiency:** batched multi-head attention, PyTorch LayerNorm, SDPA (Flash / memory-efficient kernels), mixed-precision bf16, fused AdamW, cosine schedule with warmup.
- **Inference:** KV cache, bf16 weights, temperature / top-k sampling.
- **Engineering:** typed configs, checkpointing with bit-exact resume, unit and smoke tests in CI, benchmarking and profiling harness, seeded ablations, one-command reproduction (`scripts/reproduce.sh`).

## Highlights

All on 1× A100 80GB.

- **Training throughput 5.4–8.9×, peak memory up to −53%** across the optimization journey ($T$ = 128–1024). bf16 tensor cores dominate: after SDPA, fp32 already ran at 89% of its peak, so the other steps mostly remove overhead.
- **SDPA removes the $O(T^2)$ attention memory:** the fitted quadratic term drops to ≈0, saving 9,230 MiB at $T$ = 1024.
- **RoPE beats learned positions by 0.080 val loss** (3 seeds, ≈12σ) at equal tokens, at a cost of 17% training throughput.
- **GQA/MQA trade a little quality for speed:** MQA costs +0.013 val loss (p ≈ 0.01) and trains 13% faster; 4 KV heads cost +0.007 (not significant with 3 seeds) and train 8% faster. The KV cache shrinks 12× and 3×.
- **Batch-1 decode is overhead-bound:** the KV cache gives 2.4× at 512 tokens, but decode uses ~4% of memory bandwidth, so bf16 weights and GQA don't speed it up at this scale.

## Results

All results come from commit `53a1038` (`v0.1.0-1-g53a1038`) on 1× A100-SXM4-80GB with PyTorch 2.14.0+cu130. The frozen snapshot is in [`assets/a100_2026_10_16/`](assets/a100_2026_10_16/).

### Trained Models

| | TinyShakespeare | TinyStories |
|---|---|---|
| Model | 10.8M params, 6 layers, $d$ = 384, learned positions | 162.3M params, 12 layers, $d$ = 768, RoPE |
| Tokenizer | characters (65) | GPT-2 BPE (50,257) |
| Training | 5,000 steps × 16,384 tokens, fp32, constant lr 3e-4 | 10,000 steps × 32,768 tokens = 327.7M tokens (0.69 epoch), mixed bf16, cosine lr 6e-4 |
| Time | 6.4 min, 213k tok/s | 52 min, 104.5k tok/s (27% MFU) |
| **Val loss (final = best)** | **1.4768** (Karpathy's lecture `gpt.py`: ≈1.48) | **1.3434** |
| Details | [report](assets/a100_2026_10_16/runs/full/shakespeare/report.md) | [report](assets/a100_2026_10_16/runs/full/stories/report.md) |

Losses are per token, so they aren't comparable across the two tokenizers.

- **TinyShakespeare** has converged for this setup: validation loss flattened after ~4k steps while the train/val gap grew to 0.41 (overfitting a 1M-character corpus).
- **TinyStories** is under-trained: validation loss was still falling at the end, after less than one epoch.

### Samples

Final checkpoint, temperature 1.0, no top-k, seed 0, 200 new tokens; first sample, not cherry-picked. The character model learns format, spelling, and archaic style but not meaning; the BPE model writes fluent, coherent stories but loses track of entities and logic.

**TinyShakespeare**, prompt `'ROMEO:'`:

```text
ROMEO:
Save I not, bear thee!
And, my heart! murder welcome! how show thy fair love haved!

JULIET:
You love this the shadows content and drums of this?
If your servant Nature and Tybalt,
You did be ill-but
```

**TinyStories**, prompt `'Once upon a time'`:

```text
Once upon a time there was a sunflower that began to bloom. It looked so bright and happy. The sunflower was very happy but it was also uncomfortable. It felt heavy and Bill could not seem to take it for the rest of the week.

Suddenly, the sunflower started to feel very sick. It wanted to help all the animals around it. Soon a mouse got better at the flower festival and it came up with many new ideas.

The town animals were very happy to get the sunflower back. They decided to have the party easy to protect the sunflower and everyone in the town. The sunflower was so thankful and it bloomed into a big, beautiful flower fit for everyone to enjoy.

The sun helped out the sunflower and woke it up to shine brightly! The sunflower felt so comfortable and it was very glad to be back and forth, ready to bloom and enjoy the cool mail.
```

### Training Throughput

Tokens/sec for `stories_cfg` (162.3M params), batch size 16, median of 3 trials. Each row adds to the one above; percentages are gains over the previous row.

| variant       | $T$ = 128      | $T$ = 256       | $T$ = 512       | $T$ = 1024      |
| ------------- | -------------: | --------------: | --------------: | --------------: |
| baseline      | 10,027         | 13,427          | 14,761          | 13,389          |
| +batched attn | 18,126 (+81%)  | 18,413 (+37%)   | 17,745 (+20%)   | 15,864 (+18%)   |
| +torch LN     | 18,961 (+5%)   | 19,065 (+4%)    | 18,415 (+4%)    | 16,369 (+3%)    |
| +fused AdamW  | 20,226 (+7%)   | 19,682 (+3%)    | 18,699 (+2%)    | 16,481 (+1%)    |
| +SDPA         | 21,019 (+4%)   | 21,338 (+8%)    | 21,762 (+16%)   | 21,615 (+31%)   |
| +mixed bf16   | 54,411 (+159%) | 105,080 (+392%) | 116,089 (+433%) | 119,792 (+454%) |
| total vs base | 5.4×           | 7.8×            | 7.9×            | 8.9×            |

- **Mixed bf16** is the big win: it moves the matmuls onto the tensor cores.
- **SDPA's** speed gain grows with $T$ because it avoids writing out the $T^2$ attention scores, but its real benefit is memory (below).
- **The fp32 rows are compute-bound:** at $T$ = 512 they run at 60% (baseline) to 89% (+SDPA) of the fp32 peak (19.5 TFLOP/s; true fp32, TF32 off), which is why the smaller changes add only a few percent.
- Full config and table [here](assets/a100_2026_10_16/results/full/benchmark_train_stories_cfg_cuda/report.md).

### Training Peak Memory

Same settings as above. Peak memory is fitted as $c + aT + bT^2$ (MiB).

<img src="assets/a100_2026_10_16/results/full/benchmark_train_stories_cfg_cuda/memory_fit.png" width="600" alt="Peak memory vs. sequence length with fitted curves">

- **SDPA removes the $T^2$ term:** $b$ = 3.6e-5 vs. 8.88e-3 MiB/$T^2$ for the baseline.
    - The baseline's $b$ matches one saved fp32 softmax matrix per head per layer: $B \cdot H \cdot L \cdot 4$ bytes = 8.79e-3 MiB/$T^2$, within 1%.
- **The constant term matches weights plus AdamW state:** $c$ = 1,853–1,912 MiB for the fp32 variants vs. $12N$ = 1,857 MiB (4 bytes of weights + 8 bytes of AdamW moments per parameter; gradients don't exist yet at the peak, which comes at the end of the forward pass).
- Full table [here](assets/a100_2026_10_16/results/full/benchmark_train_stories_cfg_cuda/report.md); all coefficients [here](assets/a100_2026_10_16/results/full/benchmark_train_stories_cfg_cuda/memory_fit.md).

### Generation

Prompt length $T$ = 512, batch size 1, 64 new tokens.

- **Overall:** decode 23 → 240 tok/s (~10×), time to first token (TTFT) 47.6 → 5.6 ms (−88%).
- **Batched attention:** decode 3.3×, TTFT −73%.
- **KV cache:** decode 2.4× (+15% at $T$ = 128). Without it, every step re-runs the full forward pass over all previous tokens, so the gain grows with $T$.
- **bf16 weights:** memory −48% (weights halved), TTFT −48% (prefill uses the tensor cores).
    - Decode is unchanged because it's overhead-bound.
    - Autocast is 25% slower than casting the weights, since it converts them to bf16 on every forward pass, and in decode that per-step overhead dominates.
- Full configs and tables [here](assets/a100_2026_10_16/results/full/benchmark_generate_stories_cfg_cuda/report.md).
- **GQA:** the KV cache shrinks in proportion to the number of KV heads (query heads stay at 12): 20.2 → 1.7 MiB at $T$ = 512 from MHA (12 KV heads) to MQA (1 KV head).
    - Decode throughput is unchanged (within 2%). At batch size 1 the cache is about 6% of the bytes read per step (the weights are about 310 MiB), and decode is overhead-bound.
    - GQA pays off when batch × context makes the cache comparable to the weights.
    - Full configs and tables [here](assets/a100_2026_10_16/results/full/benchmark_generate_gqa_stories_cfg_cuda/report.md).

### Profiles

`stories_cfg` (162.3M params), $T$ = 512; profiler schedule wait 1 / warmup 2 / active 3 (averaged). Attention = `qkv` + `cache` + `core` + `proj` regions.

- **Training**, full stack (mixed bf16) vs. baseline, batch size 32:
    - Step time 1,142 → 145 ms (7.9×, matching the throughput benchmark); aten op calls 36k → 3.9k.
    - SDPA dispatches to Flash attention (`_flash_attention_forward/backward`), replacing the per-head `bmm`/mask/softmax ops. Forward attention is 7.1% of the step vs. 14.2% for the baseline; backward (57.6%) is not split by region.
    - Full results [here](assets/a100_2026_10_16/results/full/profile_train_stories_cfg_cuda/report.md).
- **Prefill**, full stack (bf16) vs. baseline, batch size 1:
    - Step time 106 → 14.0 ms (7.6×, vs. 8.5× in the TTFT benchmark); aten op calls 11k → 1.2k.
    - Attention is 40.1% of the step vs. 76.1% for the baseline.
    - Full results [here](assets/a100_2026_10_16/results/full/profile_prefill_stories_cfg_cuda/report.md).
- **Decode**, full stack (bf16) vs. baseline, batch size 1:
    - Step time 122 → 13.2 ms (9.2×, vs. 10× in the throughput benchmark); aten op calls 11k → 1.1k. The baseline has no KV cache, so each of its decode steps recomputes all 512 tokens, about as costly as a prefill.
    - Attention is 42.3% of the step vs. 75.8% for the baseline.
    - Full results [here](assets/a100_2026_10_16/results/full/profile_decode_stories_cfg_cuda/report.md).
- **Notes:**
    - The labeled regions synchronize the GPU at their boundaries, which inflates short steps, so the speedups above are smaller than in the benchmarks. The benchmarks measure speed; the profiles measure where the time goes.
    - At batch size 1, inference is overhead-bound: prefilling 512 tokens (14.0 ms) costs about the same as decoding one (13.2 ms). GPU kernel time is only about 2.7 ms and 1 ms of those steps, so the region shares mostly measure per-op launch and sync overhead, not compute.
    - The next step is CUDA graphs or `torch.compile(mode='reduce-overhead')`, not faster kernels.

### Ablations

All runs: `stories_cfg` (162.3M params), context length 512, batch size 64, 3,000 steps (~98M tokens), mixed bf16; 500 warmup steps (0 → 100%), then cosine decay (100% → 10%); weight decay 0.1 (decay groups), gradient clipping 1.0. The full stack is kept except for the ablated feature. Seeds: 3 for GQA and RoPE, 1 for learning rate and schedule.

- **Learning rate** (1 seed each; baseline: mean of 3 seeds):

    | learning rate | 3e-4   | 6e-4 (baseline) | 1.2e-3 | 2.4e-3     | 4.8e-3 |
    | ------------- | -----: | --------------: | -----: | ---------: | -----: |
    | val loss      | 1.6891 | 1.5694          | 1.5032 | **1.4816** | 1.4918 |

    - The sweep brackets the optimum at ≈ 2.4e-3 to 3e-3 (≈2.7e-3 from a parabola fit in $\log_2$ of the learning rate).
        - The optimum is for 3k-step runs and may shift for longer training.
        - No divergence up to 4.8e-3, the largest rate tested.
        - With 1 seed per rate, the gap between 2.4e-3 and 4.8e-3 (0.010) is within seed noise; the bracket itself is not.
    - The baseline 6e-4 is 0.088 worse than the best rate, so all results here use an untuned learning rate.
        - The GQA and RoPE comparisons below were run at 6e-4, so their rankings likely hold, but the size of the gaps may differ at a tuned rate.
    - Details [here](assets/a100_2026_10_16/runs/full/ablations/lr/comparison.md).
- **Schedule** (1 seed): a constant learning rate of 6e-4, with no warmup, reaches val loss 1.5592 vs. 1.5694 ± 0.0065 for warmup + cosine. That's within seed noise, so the two recipes are indistinguishable at 98M tokens. Details [here](assets/a100_2026_10_16/runs/full/ablations/schedule/comparison.md).
- **RoPE vs. learned absolute positions** (3 seeds each, mean ± std):
    - RoPE lowers val loss by 0.080 (1.5694 ± 0.0065 vs. 1.6497 ± 0.0096), about 12× the standard error of the difference.
    - Cost: RoPE trains about 17% slower, since the cos/sin tables are rebuilt every forward pass and not fused.
    - Details [here](assets/a100_2026_10_16/runs/full/ablations/pos_embed/comparison.md).
- **GQA / MQA** (3 seeds each, mean ± std):

    | KV heads | val loss | ± std  | Δ vs. MHA (paired by seed, mean ± std) | training speed |
    | -------- | -------: | -----: | -------------------------------------: | -------------: |
    | 12 (MHA) | 1.5694   | 0.0065 | —                                      | —              |
    | 4 (GQA)  | 1.5762   | 0.0039 | +0.0068 ± 0.0030                       | +8%            |
    | 1 (MQA)  | 1.5827   | 0.0046 | +0.0133 ± 0.0027                       | +13%           |

    - Fewer KV heads cost a small amount of quality, worse on every seed. MQA: +0.013 (p ≈ 0.01). GQA with 4 KV heads: +0.007, same direction but not significant with 3 seeds (p ≈ 0.06).
    - In exchange, training is 8–13% faster, and the KV cache shrinks in proportion to the number of KV heads; see [Generation](#generation).
    - Details [here](assets/a100_2026_10_16/runs/full/ablations/gqa/comparison.md).

## Engineering

- **Configuration:** `tyro` dataclasses with presets from laptop scale (bigram, small GPT) to the A100 TinyStories model. Model and tokenizer registries let you choose the architecture from the command line, every field can be overridden there, and the resolved config (including derived values) is saved with each run.
- **Data:** a dataset class over memory-mapped token files handles splitting and batching, so any tokenized dataset works, from TinyStories to the synthetic data used in tests.
- **Checkpoints:** saves are atomic (write to a temporary file, then rename), and resuming reproduces an uninterrupted run bit for bit ([`test_resume_matches_continuous_run`](tests/test_checkpoint.py)). Training ends by exporting a bf16 inference checkpoint (`model.pt`) whose logits are checked against the training checkpoint's.
- **Equivalence tests:** every optimization is tested against the reference implementation it replaces, so each speedup is verified to compute the same function:
    - looped vs. batched multi-head attention;
    - manual vs. PyTorch SDPA attention and LayerNorm;
    - cached vs. uncached generation;
    - fp32 vs. mixed-bf16 and bf16 inference, within tolerance;
    - GQA with 12 KV heads vs. multi-head attention.
- **Behavioral tests:** causality (no information from future tokens or across sequences in the batch), initial loss ≈ $\ln V$ (uniform predictions), single-batch overfitting, learning-rate schedules, weight-decay parameter groups, tokenizers, and RoPE's rotation properties. 201 tests in total.
- **Tooling and reproducibility:** `pre-commit` runs the test suite on every commit, and CI runs it again on every push. `scripts/reproduce.sh smoke|full` runs the whole pipeline end to end (data, tests, training, benchmarks, profiles, ablations), from a laptop smoke run to the full A100 run. The results snapshot in `assets/` records the git commit it came from.

## Next Steps

- **Publish the model on the Hugging Face Hub** (weights, config, tokenizer, model card) with a small Gradio demo for generating stories.
- **Re-run the ablations at the tuned learning rate** (≈2.7e-3) to check whether the GQA and RoPE results hold, and **benchmark batched decoding**, where GQA's smaller KV cache should reduce memory traffic.
- **Compile the model.** At mixed bf16 the training stack reaches 27% MFU. Time is spread across many small kernels (LayerNorm, ReLU, residual adds, RoPE), and `lm_head` + loss take about half of forward time. `torch.compile` should fuse the element-wise ops and remove most of RoPE's 17% overhead. A chunked or fused cross-entropy would avoid materializing the $B \cdot T \cdot V$ logits. For batch-1 inference, which is overhead-bound, `torch.compile(mode='reduce-overhead')` (CUDA graphs) targets the per-op launch cost.
- **Distributed training with `torchrun`:** DDP to scale throughput across GPUs and nodes, then FSDP (Fully Sharded Data Parallel) for models whose weights and optimizer state no longer fit on one GPU.
- **Scaling laws on TinyStories.** The main model saw about 2 tokens per parameter (327.7M tokens, 162.3M params), far below the compute-optimal ≈20 ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)). With 474M tokens available, sweep smaller models across token budgets and fit loss as a function of parameters and tokens.

## Usage

### Setup

```bash
# 1. Install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Get the code:
git clone https://github.com/reza-mehr/nanogpt.git
cd nanogpt

# 3. Recreate the environment exactly (also installs the nanogpt package and the dev tools):
uv sync --frozen

# 4. Check the setup (prints the torch version, then True on a machine with a CUDA GPU):
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 5. Run the tests:
uv run pytest -q

# 6. Install the git hook so the tests run before each commit:
uv run pre-commit install

# 7. Inspect the training options:
uv run scripts/train.py --help
```

### Module Runs

To prepare the datasets (once):

```bash
# TinyShakespeare (characters):
uv run scripts/prepare_tinyshakespeare.py

# TinyStories (GPT-2 BPE):
uv run scripts/prepare_tinystories.py
```

To train a model:

```bash
# Bigram model:
uv run scripts/train.py bigram_cfg

# Small GPT (runs on a laptop):
uv run scripts/train.py small_cfg

# Nano-GPT on TinyShakespeare (requires a GPU):
uv run scripts/train.py main_cfg

# Nano-GPT on TinyStories (requires a GPU):
uv run scripts/train.py stories_cfg
```

To benchmark training:

```bash
# Quick debug run:
uv run scripts/benchmark_train.py --preset small_cfg --block-sizes 32 64 --iters 5 --trials 1

# Full run:
uv run scripts/benchmark_train.py --preset stories_cfg

# Rebuild the tables from the saved CSV, without benchmarking:
uv run scripts/benchmark_train.py --preset stories_cfg --from-csv
```

To benchmark generation:

```bash
# Inference journey (baseline → KV cache → bf16):
uv run scripts/benchmark_generate.py --suite journey

# KV-head sweep (MHA → GQA → MQA):
uv run scripts/benchmark_generate.py --suite gqa
```

To profile the model:

```bash
# One of train, prefill, or decode:
uv run scripts/profile_model.py --mode train
```

### End-to-End Run

To reproduce the results in this file:

```bash
# Laptop smoke run (about 15–25 min with the ablation runs):
bash scripts/reproduce.sh smoke

# A100, everything:
bash scripts/reproduce.sh full

# A100, without ~3 hours of ablation runs:
bash scripts/reproduce.sh full skip-ablations
```

## References

- Vaswani et al., 2017. [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Karpathy, 2023. [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY).
- Eldan & Li, 2023. [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)
- Su et al., 2021. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
- Shazeer, 2019. [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150).
- Ainslie et al., 2023. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245).
- Dao et al., 2022. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).
- Hoffmann et al., 2022. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).

## License

See [LICENSE](LICENSE).
