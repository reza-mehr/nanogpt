# Nano-GPT
![tests](https://github.com/reza-mehr/nanogpt/actions/workflows/tests.yml/badge.svg)

This repository contains experiments on training a small Generative Pre-trained Transformer (GPT) model from scratch based on the [tutorial by Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2292s) that implements the seminal work propsoed in the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762). [This repo](https://github.com/karpathy/nanogpt) extends on the tutorial and is worth reviewing.

The following extensions have been made to the code developed in the tutoiral:
1. A configuration manager is added using `tyro` with preset settings for the bigram and small GPT models that could be trained on a laptop as well as the final parameter setting described in the tutorial for the nano-GPT model which requires a GPU to train.
1. A model registry is added to enable specifying 'bigram' or 'gpt' model architectures directly from the command line.
1. A checkpointing extension is included to enable saving and restoring training state.
1. Data splitting and batch generation is abstracted into a seperate class enabling training on arbitary datasets.
    - This was needed to add tests that run on synthetic datasets rather than the tiny Shakespeare dataset used in the tutorial.
1. The training loop is abstracted into a function to enable functional tests.
    1. Warm-up and cosine decay are added to learning rate scheduling options.
    1. Gradient norm clipping is added to prevent noisy batches from adversely impacting convergence.
    1. Selective weight decay is enabled for AdamW optimizer.
1. Adds logging capabilities to file and Tensorboard.
1. The following functional tests have been added:
    1. Model output shapes, uniform initialization, and single batch overfit test to ensure proper optimzier function.
    1. Test to ensure correct flow of information in causal direction along the sequence and lack thereof, among sequences in the batch.
    1. Checkpoint resumption test ensuring that all state is correctly saved making the runs fully reproducible.
    1. Tests to ensure expected behavior from the learning rate scheduler.
    1. Test to ensure proper application of selective weight decay in the optimizer.


# Code

## Setup

To set up the virtual environment for the repo using `uv` package manager, run the following commands.

```bash
# 1. Install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Get the code:
git clone https://github.com/reza-mehr/nanogpt.git
cd nanogpt

# 3. Recreate the environment exactly (also installs the nanogpt package):
uv sync --frozen

# 4. Check successful setup (should return 'True' on a machine with a GPU):
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 5. Run the tests:
uv run pytest -q

# 6. Inspect the options:
uv run scripts/train.py --help
```

## Run

Run one of the following commands to train a small language model:

```bash
# Train the Bigram model:
uv run scripts/train.py 'bigram_cfg'

# Train a small GPT (runs on a laptop):
uv run scripts/train.py 'small_cfg'

# Train nano-GPT (requires a GPU):
uv run scripts/train.py 'main_cfg'
```
