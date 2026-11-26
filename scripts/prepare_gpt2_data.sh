#!/usr/bin/env bash
# This file prepares the GPT-2 data for scripts/reproduce_gpt2.sh: FineWeb-Edu 10BT tokenized into flat train.bin and
# val.bin, and the HellaSwag validation split. It needs no GPU, so it can also run on a cheaper CPU-only machine; point
# both scripts at the same DATA_ROOT.
#
# Laptop check (one FineWeb parquet, about 750M tokens):
#     bash scripts/prepare_gpt2_data.sh smoke
#
# Full dataset (~20 GB of binaries, written to DATA_ROOT, 'data' in the repo by default):
#     bash scripts/prepare_gpt2_data.sh full
# In a container, set NUM_PROC to the CPU quota, since os.cpu_count() reports the host's cores there:
#     NUM_PROC=32 bash scripts/prepare_gpt2_data.sh full
#
# Each step is skipped when its output exists (meta.json is written last), so a rerun after a crash resumes; the
# HuggingFace cache also keeps finished downloads.
# Everything is logged to logs/prepare_gpt2_data_<mode>_<timestamp>.log.

set -eo pipefail                                        # no -u: matches reproduce_nanogpt.sh (macOS bash 3.2)

#%% Settings:

MODE="${1:-smoke}"                                      # 'smoke' (laptop) or 'full'
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
DATA_ROOT="${DATA_ROOT:-data}"                          # parent folder of the datasets
# Tokenizer processes in full mode: all cores by default; set NUM_PROC in containers (see above) to avoid
# oversubscribing the CPU quota and RAM:
NUM_PROC="${NUM_PROC:-$(uv run python -c 'import os; print(os.cpu_count())')}"

if [[ "${MODE}" == 'smoke' ]]; then
    FINEWEB_DIR="${DATA_ROOT}/fineweb_single_parquet"   # prepare_fineweb.py appends '_single_parquet' for --data-file
    FINEWEB_PREP_ARGS=(--data-file sample/10BT/000_00000.parquet)
elif [[ "${MODE}" == 'full' ]]; then
    FINEWEB_DIR="${DATA_ROOT}/fineweb"
    FINEWEB_PREP_ARGS=(--num-proc "${NUM_PROC}")
else
    echo "Unknown mode '${MODE}'; use 'smoke' or 'full'." >&2
    exit 1
fi

mkdir -p logs "${DATA_ROOT}"
exec > >(tee "logs/prepare_gpt2_data_${MODE}_${STAMP}.log") 2>&1   # log everything, still print to the terminal

SECONDS=0                                                   # bash builtin: seconds since this assignment
step() { echo; echo "=== $(date +%H:%M:%S) | elapsed $((SECONDS / 60)) min | $* ==="; }

#%% Environment:

step 'Environment'
uv sync
echo "CPU cores reported: $(uv run python -c 'import os; print(os.cpu_count())'), tokenizer processes (full mode): ${NUM_PROC}"
# FineWeb needs ~20 GB of binaries in DATA_ROOT, plus several times that in the HuggingFace caches (downloaded parquet
# files, and Arrow files from loading and tokenizing), which may sit on a different disk (set HF_HOME to move them):
HF_HUB_CACHE_DIR="$(uv run python -c 'from huggingface_hub import constants; print(constants.HF_HUB_CACHE)')"
HF_DATASETS_CACHE_DIR="$(uv run python -c 'from datasets import config; print(config.HF_DATASETS_CACHE)')"
mkdir -p "${HF_HUB_CACHE_DIR}" "${HF_DATASETS_CACHE_DIR}"   # df needs existing paths
df -h "${DATA_ROOT}" "${HF_HUB_CACHE_DIR}" "${HF_DATASETS_CACHE_DIR}"

#%% Data:

step 'Prepare FineWeb'
[[ -f "${FINEWEB_DIR}/meta.json" ]] || uv run scripts/prepare_fineweb.py --out-dir "${DATA_ROOT}/fineweb" "${FINEWEB_PREP_ARGS[@]}"
cat "${FINEWEB_DIR}/meta.json"

step 'Download HellaSwag'
uv run python -c "from nanogpt.hellaswag import get_dataset; print(get_dataset('${DATA_ROOT}/hellaswag'))"

printf 'Total wall time: %d h %02d min.\n' $((SECONDS / 3600)) $((SECONDS % 3600 / 60))
echo "Done. Back up ${FINEWEB_DIR}, ${DATA_ROOT}/hellaswag and logs/prepare_gpt2_data_${MODE}_${STAMP}.log before"
echo "removing them or this machine; scripts/reproduce_gpt2.sh reads the datasets from DATA_ROOT. The HuggingFace"
echo "caches (${HF_HUB_CACHE_DIR}, ${HF_DATASETS_CACHE_DIR}) are not needed for training."
