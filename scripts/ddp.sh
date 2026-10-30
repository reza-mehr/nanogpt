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

set -eo pipefail                                # no -u: matches reproduce.sh (macOS bash 3.2)

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

#%% Launch:

TORCHRUN_ARGS=(--nnodes=1 --nproc_per_node="${NPROC}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")

if [[ -n "${PYTHON}" ]]; then
    "${PYTHON}" -m torch.distributed.run "${TORCHRUN_ARGS[@]}" "$@"    # torchrun of the given interpreter
else
    uv run torchrun "${TORCHRUN_ARGS[@]}" "$@"                         # torchrun of the project environment
fi
