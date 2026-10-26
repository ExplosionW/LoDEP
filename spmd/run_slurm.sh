#!/bin/bash
# One submission; one complete node, two GPUs; no monitoring loop.
set -Eeuo pipefail
SPMD_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --worker ]]; then
    shift
    export SPMD_GPUS=2
    exec bash "$SPMD_REPO_ROOT/spmd/run.sh" "$@"
fi
mkdir -p "${SPMD_WORK_DIR:-$SPMD_REPO_ROOT/outputs/spmd_seed42}/logs"
sbatch --parsable --nodes=1 --exclusive --gpus-per-node=2 --ntasks=1 \
    --cpus-per-task=32 --mem=128G --time="${SPMD_TIME_LIMIT:-2-00:00:00}" \
    --partition="${SPMD_PARTITION:-kobilka-gputest}" --job-name=spmd_seed42 \
    --chdir="$SPMD_REPO_ROOT" \
    --output="${SPMD_WORK_DIR:-$SPMD_REPO_ROOT/outputs/spmd_seed42}/logs/%x_%j.out" \
    --error="${SPMD_WORK_DIR:-$SPMD_REPO_ROOT/outputs/spmd_seed42}/logs/%x_%j.err" \
    "$SPMD_REPO_ROOT/spmd/run_slurm.sh" --worker "$@"
