#!/usr/bin/env bash
# This file runs the GPT-2 pipeline end to end: tests, the speed journey benchmark with MFU, the DDP measurements
# (scaling efficiency, compile order, recompiles), and the main run with HellaSwag evaluation. The data is prepared
# beforehand by scripts/prepare_gpt2_data.sh, on a CPU-only node for the full run, so the GPUs never wait on it.
#
# Laptop check first (minutes, tiny settings, DDP on CPU processes with gloo):
#     bash scripts/prepare_gpt2_data.sh smoke && bash scripts/reproduce_gpt2.sh smoke
#
# Rented GPUs (8x A100 80 GB; on a 40 GB card, halve the batch sizes below), with the prepared volume mounted; the
# datasets are copied to LOCAL_ROOT (default 'data' in the repo) on local disk first:
#     DATA_ROOT=/mnt/volume/data bash scripts/reproduce_gpt2.sh full
#
# Everything is logged to logs/reproduce_gpt2_<mode>_<timestamp>.log; each phase header shows the elapsed time, and the
# total wall time is printed at the end (cost = hourly rate x hours). Results land in runs/gpt2_<mode>/ and results/.

set -eo pipefail                            # no -u: matches reproduce_nanogpt.sh (macOS bash 3.2)

#%% Settings:

MODE="${1:-smoke}"                          # 'smoke' (laptop) or 'full' (GPU)
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUNS="runs/gpt2_${MODE}"                    # parent folder for the training runs
RESULTS="results/gpt2_${MODE}"              # parent folder for the benchmarks
DATA_ROOT="${DATA_ROOT:-data}"              # parent folder of the prepared datasets, e.g. a persistent volume
LOCAL_ROOT="${LOCAL_ROOT:-data}"            # local disk the datasets are copied to (~20 GB); same as DATA_ROOT: no copy
HELLASWAG_DIR="${DATA_ROOT}/hellaswag"

if [[ "${MODE}" == 'smoke' ]]; then
    NGPU=2                                  # CPU processes standing in for GPUs
    # Train benchmark: one trial at two short lengths on MPS, different device (faster compiles) from DDP runs below:
    TRAIN_BENCH_ARGS=(--device mps --block-sizes 32 64 --batch-size 2 --iters 1 --trials 1 --warmup 1)
    # Model and batch: a 2-block GPT-2 on CPU in fp32 (bf16 is ~100x slower on CPU), T=256 to fit HellaSwag (~150
    # tokens), 2048 tokens per step (2 substeps on 2 processes, 4 on 1):
    MODEL_ARGS=(--device cpu --precision fp32 --model.n-blocks 2 --model.block-size 256 --batch-size 2 --batch-tokens 2048)
    OPT_ARGS=(--compile-model --model.pad-vocab-to-multiple 128)                    # no tf32 on CPU
    # DDP measurements: a few steps, logged often so there are rows after the first (compile) interval:
    DDP_ARGS=(--max-iters 6 --eval-interval 2 --eval-iters 1 --lr-schedule constant --ckpt-freq 0)
    FINEWEB_DIR="${DATA_ROOT}/fineweb_single_parquet"
    MAIN_ARGS=(--max-iters 20 --lr-warmup-iters 5 --eval-interval 5 --eval-iters 2
               --hellaswag.eval-freq 2 --hellaswag.batch-size 16 --hellaswag.batch-num 2)
elif [[ "${MODE}" == 'full' ]]; then
    NGPU="$(uv run python -c 'import torch; print(torch.cuda.device_count())')"
    # Batch 16 keeps the fp32 baseline (B*T*50257 logits plus T^2 attention scores) inside 80 GB at T=1024:
    TRAIN_BENCH_ARGS=(--block-sizes 256 512 1024 --batch-size 16)
    # 2^19 = 524,288 tokens per step as in GPT-3 small: no accumulation on 8 GPUs, 8 substeps on 1:
    MODEL_ARGS=(--batch-size 64 --batch-tokens 524288)
    OPT_ARGS=(--compile-model --matmul-precision tf32 --model.pad-vocab-to-multiple 128)
    DDP_ARGS=(--max-iters 60 --eval-interval 10 --eval-iters 5 --lr-schedule constant --ckpt-freq 0)
    FINEWEB_DIR="${DATA_ROOT}/fineweb"
    # 10B tokens / 2^19 = 19,073 steps; warmup 375M tokens = 715 steps (Karpathy's settings):
    MAIN_ARGS=(--max-iters 19073 --lr-warmup-iters 715 --eval-interval 250 --hellaswag.eval-freq 1)
else
    echo "Unknown mode '${MODE}'; use 'smoke' or 'full'." >&2
    exit 1
fi

# Fail before renting time is spent: the data must already be prepared (meta.json is written last):
if [[ ! -f "${FINEWEB_DIR}/meta.json" || ! -d "${HELLASWAG_DIR}" ]]; then
    echo "Missing data under '${DATA_ROOT}'; run 'DATA_ROOT=${DATA_ROOT} bash scripts/prepare_gpt2_data.sh ${MODE}' first." >&2
    exit 1
fi

rm -rf "${RUNS}" "${RESULTS}"                                   # start clean; earlier logs and tarballs are kept
mkdir -p logs
exec > >(tee "logs/reproduce_gpt2_${MODE}_${STAMP}.log") 2>&1   # log everything, still print to the terminal

SECONDS=0                                                       # bash builtin: seconds since this assignment
step() { echo; echo "=== $(date +%H:%M:%S) | elapsed $((SECONDS / 60)) min | $* ==="; }

#%% Environment:

step 'Environment'
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available() and torch.cuda.get_device_name(0))"
command -v nvidia-smi > /dev/null && nvidia-smi || true

step 'Copy data to local disk (training reads memory-mapped files; keeps network storage out of the measurements)'
local_copy() {
    # Copy the prepared dataset to '<dst>' through '<dst>.partial', rename at the end so a partial copy is never used:
    local src="$1" dst="$2"
    if [[ "${src}" -ef "${dst}" || -e "${dst}" ]]; then echo "Using ${dst}"; return; fi
    rm -rf "${dst}.partial"                                 # leftover from an interrupted copy
    mkdir -p "$(dirname "${dst}")"
    cp -R "${src}" "${dst}.partial"
    mv "${dst}.partial" "${dst}"
    du -sh "${dst}"
}
df -h "$(dirname "${LOCAL_ROOT}/.")"
local_copy "${FINEWEB_DIR}" "${LOCAL_ROOT}/$(basename "${FINEWEB_DIR}")"
local_copy "${HELLASWAG_DIR}" "${LOCAL_ROOT}/$(basename "${HELLASWAG_DIR}")"
FINEWEB_DIR="${LOCAL_ROOT}/$(basename "${FINEWEB_DIR}")"
HELLASWAG_DIR="${LOCAL_ROOT}/$(basename "${HELLASWAG_DIR}")"

step 'Tests'
if [[ "${MODE}" == 'smoke' ]]; then
    uv run pytest                                           # everything, smoke tests included
else
    uv run pytest -m 'not smoke'                            # unit, slow and GPU tests; the smoke tests ran on the laptop
fi

#%% Speed journey (single device): tokens/s, MFU, and memory after each optimization:

step 'Benchmark the GPT-2 training journey'
uv run scripts/benchmark_train.py --preset gpt2_cfg --out-dir "${RESULTS}" "${TRAIN_BENCH_ARGS[@]}"
uv run scripts/fit_memory.py --run-dir "$(ls -d "${RESULTS}"/benchmark_train_gpt2_cfg_*)"

#%% DDP measurements (same tokens per step in every run; compare tok/s in each run's results.csv, skipping the first
#   row, which includes compilation):

DDP_RUNS="${RUNS}/ddp"

ddp_run() {
    # Train one short measurement run '<name>' on <nproc> processes with extra overrides:
    local name="$1" nproc="$2"
    shift 2
    bash scripts/ddp.sh "${nproc}" scripts/train.py gpt2_cfg --data.dir "${FINEWEB_DIR}" --out-dir "${DDP_RUNS}" \
        --run-name "${name}" --hellaswag.data-dir "${HELLASWAG_DIR}" "${MODEL_ARGS[@]}" "${OPT_ARGS[@]}" "${DDP_ARGS[@]}" "$@"
}

step 'DDP: single device (scaling baseline, gradient accumulation instead of ranks)'
ddp_run "n1" 1

step "DDP: ${NGPU} devices, compile after DDP (recompiles logged: the sync toggle should recompile at most once)"
TORCH_LOGS=recompiles ddp_run "n${NGPU}_after_ddp" "${NGPU}" --compile-order after_ddp

step "DDP: ${NGPU} devices, compile before DDP"
ddp_run "n${NGPU}_before_ddp" "${NGPU}" --compile-order before_ddp

#%% Main run (validation loss and HellaSwag at every evaluation):

step "Train gpt2_cfg on FineWeb (${NGPU} devices)"
bash scripts/ddp.sh "${NGPU}" scripts/train.py gpt2_cfg --data.dir "${FINEWEB_DIR}" --out-dir "${RUNS}" --run-name main \
    --hellaswag.data-dir "${HELLASWAG_DIR}" --sample-prompts '' "Hello, I'm a language model," "${MODEL_ARGS[@]}" "${OPT_ARGS[@]}" "${MAIN_ARGS[@]}"

step 'Plot the main run'
uv run scripts/plot_runs.py --path "${RUNS}/main"

#%% Bundle the outputs to copy off the machine:

step 'Bundle outputs'
# model.pt (inference-only) is kept for generation; the training checkpoints are only needed to resume:
tar --exclude='latest.pt' --exclude='best.pt' -czf "outputs_gpt2_${MODE}_${STAMP}.tar.gz" "${RUNS}" "${RESULTS}" logs
ls -lh "outputs_gpt2_${MODE}_${STAMP}.tar.gz"
printf 'Total wall time: %d h %02d min.\n' $((SECONDS / 3600)) $((SECONDS % 3600 / 60))
echo "Done. Copy outputs_gpt2_${MODE}_${STAMP}.tar.gz off the machine before destroying it."
