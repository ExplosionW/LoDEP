#!/bin/bash
# Execute from any directory after installing spmd/requirements.txt.
set -Eeuo pipefail
SPMD_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SPMD_REPO_ROOT"
export SPMD_WORK_DIR="${SPMD_WORK_DIR:-$SPMD_REPO_ROOT/outputs/spmd_pretrained_seed42}"
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
SPMD_GPUS="${SPMD_GPUS:-2}"
mkdir -p "$SPMD_WORK_DIR"
python -m spmd.prepare data "$@"
if [[ ! -f "$SPMD_WORK_DIR/structures/train.npz" ]]; then
    : "${SPMD_CNN_CHECKPOINT:?Set SPMD_CNN_CHECKPOINT to the official ProstT5 cnn_chkpnt/model.pt}"
    python -m torch.distributed.run --standalone --nproc_per_node="$SPMD_GPUS" --module spmd.predict_3di --model "${SPMD_PROSTT5_MODEL:-Rostlab/ProstT5}" --cnn-checkpoint "$SPMD_CNN_CHECKPOINT"
fi
python -m spmd.prepare groups --structure-npz "$SPMD_WORK_DIR/structures/train.npz"
python -m torch.distributed.run --standalone --nproc_per_node="$SPMD_GPUS" --module spmd.export_teacher
for SPMD_ARM in ${SPMD_ARMS:-kd_output kd_feature spmd}; do
    python -m torch.distributed.run --standalone --nproc_per_node="$SPMD_GPUS" --module spmd.train_student "$SPMD_ARM"
done
