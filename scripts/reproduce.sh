#!/usr/bin/env bash
# This file runs the whole pipeline end to end: tests, data preparation, training and generation on both datasets,
# benchmarks and profiles of the larger model, then the quality ablations (GQA, RoPE, learning rate, schedule).
#
# Laptop check first (minutes, tiny settings, same commands):
#     bash scripts/reproduce.sh smoke
#
# Rented GPU (A100 80 GB; on a 40 GB card, halve the benchmark batch size below):
#     bash scripts/reproduce.sh full
#
# Everything except the ablations (the longest part):
#     bash scripts/reproduce.sh full skip-ablations
#
# Everything is logged to logs/reproduce_<mode>_<timestamp>.log; results land in runs/ and results/.

set -eo pipefail                                # no -u: empty arrays fail under macOS bash 3.2

#%% Settings:

MODE="${1:-smoke}"                              # 'smoke' (laptop) or 'full' (GPU)
ABLATIONS="${2:-run}"                           # 'run' or 'skip-ablations'
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUNS="runs/${MODE}"                             # parent folder for the training runs
RESULTS="results/${MODE}"                       # parent folder for benchmarks and profiles
SHAKES_DIR='data/tinyshakespeare'
STORIES_VOCAB=50257                             # synthetic vocabulary for benchmarks; match the TinyStories tokenizer

if [[ "${MODE}" == 'smoke' ]]; then
    # Data: a small TinyStories subset in its own folder, so a later full run can never pick it up:
    STORIES_DIR='data/tinystories_smoke'
    STORIES_PREP_ARGS=(--max-stories 2000)
    # Training: a few steps, small batches and contexts so the larger preset runs on a laptop:
    TRAIN_ARGS=(--max-iters 20 --eval-interval 10 --ckpt-interval 10 --eval-iters 2 --lr-warmup-iters 5)
    STORIES_ARGS=(--batch-size 4 --model.block-size 64)
    GEN_TOKENS=40
    # Benchmarks and profiles: one trial at two short lengths:
    BENCH_ARGS=(--trials 1 --warmup 1)
    TRAIN_BENCH_ARGS=(--block-sizes 32 64 --iters 1 --batch-size 2)
    GEN_BENCH_ARGS=(--prompt-lens 16 32 --new-tokens 8)
    PROF_ARGS=(--block-sizes 32 --wait 0 --warmup 1 --active 1 --no-export-trace)
    # Ablations: one seed, the training settings above:
    ABL_ARGS=()
    SEEDS=(0)
elif [[ "${MODE}" == 'full' ]]; then
    STORIES_DIR='data/tinystories'
    STORIES_PREP_ARGS=()
    TRAIN_ARGS=()
    # Batch 64 on an A100: 32k tokens per step, ~330M tokens over 10k steps (about 69% epoch of TinyStories):
    STORIES_ARGS=(--batch-size 64)
    GEN_TOKENS=200
    BENCH_ARGS=()
    # Batch 16 keeps the looped fp32 baseline (B*H*T^2 scores per layer) inside 80 GB at T=1024; use 8 on a 40 GB A100.
    TRAIN_BENCH_ARGS=(--block-sizes 128 192 256 384 512 768 1024 --batch-size 16)
    GEN_BENCH_ARGS=(--prompt-lens 128 512)
    PROF_ARGS=(--block-sizes 512)
    # Ablations: shorter runs (3k steps, ~98M tokens each) with denser evaluation; checkpoint only at the end:
    ABL_ARGS=(--max-iters 3000 --eval-interval 250 --ckpt-interval 3000)
    SEEDS=(0 1 2)
else
    echo "Unknown mode '${MODE}'; use 'smoke' or 'full'." >&2
    exit 1
fi

rm -rf "${RUNS}" "${RESULTS}"                               # start clean; earlier logs and tarballs are kept
mkdir -p logs
exec > >(tee "logs/reproduce_${MODE}_${STAMP}.log") 2>&1    # log everything, still print to the terminal

step() { echo; echo "=== $(date +%H:%M:%S) | $* ==="; }

#%% Environment:

step 'Environment'
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available() and torch.cuda.get_device_name(0))"
command -v nvidia-smi > /dev/null && nvidia-smi || true

step 'Tests'
if [[ "${MODE}" == 'smoke' ]]; then
    uv run pytest                                           # everything, smoke tests included
else
    uv run pytest -m 'not smoke'                            # unit tests only; the smoke tests ran on the laptop
fi

#%% Data (skipped when the splits already exist; each mode has its own TinyStories folder):

step 'Prepare TinyShakespeare'
[[ -f "${SHAKES_DIR}/train.bin" ]] || uv run scripts/prepare_tinyshakespeare.py

step 'Prepare TinyStories'
[[ -f "${STORIES_DIR}/train.bin" ]] || uv run scripts/prepare_tinystories.py --out-dir "${STORIES_DIR}" "${STORIES_PREP_ARGS[@]}"

#%% Training and generation:

step 'Train main_cfg on TinyShakespeare'
uv run scripts/train.py main_cfg --data.dir "${SHAKES_DIR}" --out-dir "${RUNS}" --run-name shakespeare \
    --sample-prompts '' 'ROMEO:' "${TRAIN_ARGS[@]}"

step 'Train stories_cfg on TinyStories'
uv run scripts/train.py stories_cfg --data.dir "${STORIES_DIR}" --out-dir "${RUNS}" --run-name stories \
    --sample-prompts '' 'Once upon a time' "${TRAIN_ARGS[@]}" "${STORIES_ARGS[@]}"

step 'Generate from both runs'
uv run scripts/generate.py --out-dir "${RUNS}" --run-name shakespeare --prompt 'ROMEO:' --new-tokens "${GEN_TOKENS}"
uv run scripts/generate.py --out-dir "${RUNS}" --run-name stories --prompt 'Once upon a time' --new-tokens "${GEN_TOKENS}"

step 'Plot runs'
uv run scripts/plot_runs.py --path "${RUNS}" --per-run

#%% Benchmarks (stories_cfg):

step 'Benchmark training'
uv run scripts/benchmark_train.py --preset stories_cfg --vocab-size "${STORIES_VOCAB}" --out-dir "${RESULTS}" \
    "${BENCH_ARGS[@]}" "${TRAIN_BENCH_ARGS[@]}"
uv run scripts/fit_memory.py --run-dir "$(ls -d "${RESULTS}"/benchmark_train_stories_cfg_*)"

step 'Benchmark generation (journey, then GQA)'
for suite in journey gqa; do
    uv run scripts/benchmark_generate.py --preset stories_cfg --vocab-size "${STORIES_VOCAB}" --out-dir "${RESULTS}" \
        --suite "${suite}" "${BENCH_ARGS[@]}" "${GEN_BENCH_ARGS[@]}"
done

#%% Profiles (stories_cfg):

for mode in train prefill decode; do
    step "Profile ${mode}"
    uv run scripts/profile_model.py --preset stories_cfg --vocab-size "${STORIES_VOCAB}" --out-dir "${RESULTS}" \
        --mode "${mode}" "${PROF_ARGS[@]}"
done

#%% Ablations (stories_cfg; baseline: MHA, RoPE, lr 6e-4, cosine):

ABL="${RUNS}/ablations"

ablation_run() {
    # Train one ablation run '<name>_s<seed>' with extra overrides, then delete its checkpoints (only metrics are compared).
    local name="$1" seed="$2"
    shift 2
    uv run scripts/train.py stories_cfg --data.dir "${STORIES_DIR}" --out-dir "${ABL}/runs" \
        --run-name "${name}_s${seed}" --seed "${seed}" --sample-prompts 'Once upon a time' \
        "${TRAIN_ARGS[@]}" "${STORIES_ARGS[@]}" "${ABL_ARGS[@]}" "$@"
    find "${ABL}/runs/${name}_s${seed}" -name '*.pt' -delete
}

ablation_group() {
    # Link the runs named by each prefix into '<group>/', so the baseline is shared without retraining.
    local group="$1"
    shift
    mkdir -p "${ABL}/${group}"
    for prefix in "$@"; do
        for run in "${ABL}/runs/${prefix}"_s*; do
            ln -s "../runs/$(basename "${run}")" "${ABL}/${group}/"
        done
    done
}

if [[ "${ABLATIONS}" == 'run' ]]; then
    step 'Ablations: train'
    for seed in "${SEEDS[@]}"; do
        ablation_run baseline "${seed}" --model.kv-head-num 12            # explicit MHA, so groups sort numerically
        ablation_run kv4 "${seed}" --model.kv-head-num 4                  # GQA, 3 query heads per KV head
        ablation_run kv1 "${seed}" --model.kv-head-num 1                  # MQA
        ablation_run learned "${seed}" --model.pos-embed learned          # learned absolute positions
    done
    ablation_run lr3e-4 0 --learning-rate 3e-4                            # large effects: one seed each
    ablation_run lr1.2e-3 0 --learning-rate 1.2e-3
    ablation_run constant 0 --lr-schedule constant

    step 'Ablations: compare'
    ablation_group gqa baseline kv4 kv1
    ablation_group pos_embed baseline learned
    ablation_group lr baseline lr3e-4 lr1.2e-3
    ablation_group schedule baseline constant
    uv run scripts/plot_runs.py --path "${ABL}/gqa" --group-by model.kv_head_num
    uv run scripts/plot_runs.py --path "${ABL}/pos_embed" --group-by model.pos_embed
    uv run scripts/plot_runs.py --path "${ABL}/lr" --group-by learning_rate
    uv run scripts/plot_runs.py --path "${ABL}/schedule" --group-by lr_schedule
fi

#%% Bundle the outputs to copy off the machine:

step 'Bundle outputs'
# model.pt (inference-only) is kept for generation; the training checkpoints are only needed to resume:
tar --exclude='latest.pt' --exclude='best.pt' -czf "outputs_${MODE}_${STAMP}.tar.gz" "${RUNS}" "${RESULTS}" logs
ls -lh "outputs_${MODE}_${STAMP}.tar.gz"
echo "Done. Copy outputs_${MODE}_${STAMP}.tar.gz off the machine before destroying it."
