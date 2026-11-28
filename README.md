# Generative Pretrained Transformer

![tests](https://github.com/reza-mehr/nanogpt/actions/workflows/tests.yml/badge.svg)

Decoder-only Transformers built from scratch following Andrej Karpathy's tutorials, trained, benchmarked, and profiled end to end on TinyShakespeare, TinyStories, and FineWeb-Edu, with every result reproducible from one script and a tagged commit.

## NanoGPT

A 162.3M-parameter model trained on TinyStories in 52 minutes on 1× A100, extended with RoPE, GQA, a KV cache, and seeded ablations. Full report: [docs/NANOGPT.md](docs/NANOGPT.md).

- **Training throughput 5.4–8.9×, peak memory up to −53%** across the optimization journey; mixed bf16 dominates.
- **RoPE beats learned positions by 0.080 val loss** (3 seeds, ≈12σ), at a cost of 17% training throughput.
- **GQA/MQA trade a little quality for speed:** MQA costs +0.013 val loss and trains 13% faster, with a 12× smaller KV cache.
- **Batch-1 decode is overhead-bound:** the KV cache gives 2.4× at 512 tokens, but bf16 weights and GQA don't speed it up at this scale.

## GPT-2

GPT-2 124M reproduced on 10.0B FineWeb-Edu tokens with DDP on 8× A100 in 1 h 41 min of training steps (≈ \$29 for the whole pipeline). Full report: [docs/GPT2.md](docs/GPT2.md).

- **Best val loss 3.0598, HellaSwag 0.3014:** matches OpenAI's GPT-2 124M weights (0.2966 under the same eval), within one standard error.
- **Training throughput 8.0–11.1×, peak memory up to −64%** across the optimization journey; TF32 and `torch.compile` dominate.
- **DDP scales at 94.6% on 8 GPUs, at 56.7% MFU** for the main run.
- **Compile order vs. DDP wrap makes no measurable difference** (0.1%) on a single NVSwitch node.
- **[Three failures the laptop tests couldn't catch](docs/GPT2.md#lessons-from-the-run)**, and the checks added so the next run catches them in its first minutes.

## Setup

```bash
# 1. Install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Get the code:
git clone https://github.com/reza-mehr/nanogpt.git
cd nanogpt

# 3. Recreate the environment exactly (also installs the nanogpt package and the dev tools):
uv sync --locked

# 4. Check the setup (prints the torch version, then True on a machine with a CUDA GPU):
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 5. Run the tests:
uv run pytest -q

# 6. Install the git hook so the tests run before each commit:
uv run pre-commit install

# 7. Inspect the training options:
uv run scripts/train.py --help
```

Linux installs PyTorch from the CUDA 12.6 index and macOS from PyPI (see `pyproject.toml`). Each report's Usage section covers its datasets, runs, and end-to-end script; to reproduce a report's numbers exactly, check out the tag it names.

## License

See [LICENSE](LICENSE).
