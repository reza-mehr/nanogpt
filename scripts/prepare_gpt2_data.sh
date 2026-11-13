#!/usr/bin/env bash
# This file prepares the GPT-2 data on a CPU-only node, so the GPU node never sits idle on downloading and tokenizing:
# FineWeb-Edu 10BT tokenized into flat train.bin and val.bin, and the HellaSwag validation split.
#
# Laptop check (one FineWeb parquet, about 750M tokens):
#     bash scripts/prepare_gpt2_data.sh smoke
#
# CPU node with a persistent volume, later mounted on the GPU node (point both scripts at the same DATA_ROOT):
#     DATA_ROOT=/mnt/volume/data bash scripts/prepare_gpt2_data.sh full
#
# Each step is skipped when its output exists (meta.json is written last), so a rerun after a crash resumes; the
# HuggingFace cache also keeps finished downloads (set HF_HOME on the volume to keep them across nodes).
# Everything is logged to logs/prepare_gpt2_data_<mode>_<timestamp>.log.

set -eo pipefail                                # no -u: matches reproduce_nanogpt.sh (macOS bash 3.2)

#%% Settings:

MODE="${1:-smoke}"                              # 'smoke' (laptop) or 'full' (CPU node)
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
DATA_ROOT="${DATA_ROOT:-data}"                  # parent folder of the datasets, e.g. a persistent volume

if [[ "${MODE}" == 'smoke' ]]; then
    FINEWEB_DIR="${DATA_ROOT}/fineweb_single_parquet"   # prepare_fineweb.py appends '_single_parquet' for --data-file
    FINEWEB_PREP_ARGS=(--data-file sample/10BT/000_00000.parquet)
elif [[ "${MODE}" == 'full' ]]; then
    FINEWEB_DIR="${DATA_ROOT}/fineweb"
    FINEWEB_PREP_ARGS=(--num-proc "$(uv run python -c 'import os; print(os.cpu_count())')")
else
    echo "Unknown mode '${MODE}'; use 'smoke' or 'full'." >&2
    exit 1
fi

mkdir -p logs "${DATA_ROOT}"
exec > >(tee "logs/prepare_gpt2_data_${MODE}_${STAMP}.log") 2>&1   # log everything, still print to the terminal

SECONDS=0                                                   # bash builtin: seconds since this assignment
step() { echo; echo "=== $(date +%H:%M:%S) | +$((SECONDS / 60)) min | $* ==="; }

#%% Environment:

step 'Environment'
uv sync
echo "CPU cores: $(uv run python -c 'import os; print(os.cpu_count())')"
df -h "${DATA_ROOT}"                                        # FineWeb needs ~20 GB of binaries plus the download cache

#%% Data:

step 'Prepare FineWeb'
[[ -f "${FINEWEB_DIR}/meta.json" ]] || uv run scripts/prepare_fineweb.py --out-dir "${DATA_ROOT}/fineweb" "${FINEWEB_PREP_ARGS[@]}"
cat "${FINEWEB_DIR}/meta.json"

step 'Download HellaSwag'
uv run python -c "from nanogpt.hellaswag import get_dataset; print(get_dataset('${DATA_ROOT}/hellaswag'))"

printf 'Total wall time: %d h %02d min.\n' $((SECONDS / 3600)) $((SECONDS % 3600 / 60))
