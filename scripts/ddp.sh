#!/usr/bin/env bash
# This file launches a script with distributed data parallelism (DDP) on one machine through torchrun.
#
# Usage: bash scripts/ddp.sh <n_processes> <script> [script arguments...]
#     bash scripts/ddp.sh 2 scripts/train.py small_cfg --device cpu --max-iters 20      # CPU (gloo), e.g. a laptop
#     bash scripts/ddp.sh 8 scripts/train.py gpt2_cfg                                   # one process per GPU (nccl)
#
# The rendezvous address is pinned to the IPv4 loopback: without it, torchrun on macOS resolves a hostname to the
# IPv6 loopback, fails the lookup, and hangs. MASTER_PORT can be overridden, e.g. to run two jobs at once.
# PYTHON can name an interpreter to launch with instead of 'uv run', e.g. the tests pass their own sys.executable.
# With several processes, torch.compile runs in each rank's own process (TORCHINDUCTOR_COMPILE_THREADS=1, see below).

set -eo pipefail                                # no -u: matches reproduce_nanogpt.sh (macOS bash 3.2)

#%% Arguments:

if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/ddp.sh <n_processes> <script> [script arguments...]" >&2
    exit 1
fi
NPROC="$1"
shift                                           # "$@" now holds the script and its arguments

#%% Settings:

MASTER_ADDR='127.0.0.1'                         # IPv4 loopback, see above
MASTER_PORT="${MASTER_PORT:-29500}"             # torchrun's default port unless overridden

# Several ranks compiling at once, each with a pool of parallel compile workers sized to all cores, hung one rank of an
# 8-GPU run on a 192-core node (waiting on its pool) while the others waited in NCCL. One compile process per rank
# avoids the pool; compiled kernels are cached on disk, so mostly the first run pays for the slower compile. Override
# to trade that safety for speed, e.g. TORCHINDUCTOR_COMPILE_THREADS=$(( $(nproc) / 8 )) for 8 ranks:
if (( NPROC > 1 )); then
    export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
fi
echo "ddp.sh: ${NPROC} processes, TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-<unset, PyTorch default>}"

#%% Launch:

TORCHRUN_ARGS=(--nnodes=1 --nproc_per_node="${NPROC}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")

if [[ -n "${PYTHON}" ]]; then
    "${PYTHON}" -m torch.distributed.run "${TORCHRUN_ARGS[@]}" "$@"    # torchrun of the given interpreter
else
    uv run torchrun "${TORCHRUN_ARGS[@]}" "$@"                         # torchrun of the project environment
fi
