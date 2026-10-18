#!/bin/bash
# ==============================================================================
# run_train.sh
# Shell execution wrapper for HTCondor CRMHD-TBNN training
# ==============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

echo "=================================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Workdir: $(pwd)"
echo "=================================================="

# Activate virtual environment
if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
    source "${SCRIPT_DIR}/.venv/bin/activate"
    echo "Activated .venv at: ${SCRIPT_DIR}/.venv"
elif [ -f "${HOME}/scratch/miquel/.venv/bin/activate" ]; then
    source "${HOME}/scratch/miquel/.venv/bin/activate"
    echo "Activated .venv from home scratch"
else
    echo "WARNING: .venv not found, using system Python: $(which python3)"
fi

echo "Python: $(which python)"
python -c "import torch; print('PyTorch version:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"

# Default training arguments.
#
# Argument 5 is now SNAPSHOTS_PER_BATCH, not BATCH_SIZE: batches are whole
# snapshots (a 64x64 coarse field at r=16), which is what lets the spectral
# cascade term share a backward pass with the point-wise terms. Anything that
# still passes a point count here (e.g. 2048) would ask for 2048 snapshots per
# step, so the value is range-checked below.
DATA_DIR="${1:-dns_khi}"
COARSENING="${2:-16}"
STRIDE="${3:-1}"
EPOCHS="${4:-100}"
SNAPSHOTS_PER_BATCH="${5:-2}"
LR="${6:-1e-3}"
DEVICE="${7:-cuda}"
PLOT_INTERVAL="${8:-10}"
PATIENCE="${9:-}"

if [ "${SNAPSHOTS_PER_BATCH}" -gt 32 ] 2>/dev/null; then
    echo "ERROR: argument 5 is --snapshots-per-batch (whole snapshots per optimizer"
    echo "       step), but got ${SNAPSHOTS_PER_BATCH}. This used to be --batch-size in"
    echo "       points. Use a small number such as 2." >&2
    exit 2
fi

EXTRA_ARGS=()
if [ -n "${PATIENCE}" ] && [[ "${PATIENCE}" != --* ]]; then
    EXTRA_ARGS+=(--patience "${PATIENCE}")
    shift 9 2>/dev/null || true
elif [[ "${PATIENCE}" == --* ]]; then
    shift 8 2>/dev/null || true
else
    shift $# 2>/dev/null || true
fi

# Append any remaining command-line flags (e.g. --alpha-warmup, --scale-warmup, etc.)
EXTRA_ARGS+=("$@")

# If CUDA is requested, display GPU info
if [ "${DEVICE}" = "cuda" ] && command -v nvidia-smi &> /dev/null; then
    echo "------------------ NVIDIA-SMI ------------------"
    nvidia-smi || true
    echo "------------------------------------------------"
elif [ "${DEVICE}" = "cpu" ]; then
    # Disable CUDA discovery so PyTorch avoids accelerator health checks on CPU
    export CUDA_VISIBLE_DEVICES=""
fi

# Where checkpoints land. Overridable so a new experiment cannot silently
# overwrite an existing trained model -- set CHECKPOINT_DIR in the submit file.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"

mkdir -p "${CHECKPOINT_DIR}" condor_logs "${CHECKPOINT_DIR}/plots"

echo "Starting training: data=${DATA_DIR}, r=${COARSENING}, stride=${STRIDE}, epochs=${EPOCHS}, snapshots_per_batch=${SNAPSHOTS_PER_BATCH}, plot_interval=${PLOT_INTERVAL}, device=${DEVICE}, patience=${PATIENCE:-default (2x LR patience)}, checkpoints=${CHECKPOINT_DIR}, extra_flags=${EXTRA_ARGS[*]:-none}"

python -m src.train_apriori \
    --data-dir "${DATA_DIR}" \
    --coarsening-factor "${COARSENING}" \
    --stride "${STRIDE}" \
    --epochs "${EPOCHS}" \
    --snapshots-per-batch "${SNAPSHOTS_PER_BATCH}" \
    --lr "${LR}" \
    --plot-interval "${PLOT_INTERVAL}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --device "${DEVICE}" \
    --dtype float64 \
    "${EXTRA_ARGS[@]}"

echo "=================================================="
echo "Training finished at: $(date)"
echo "Checkpoints saved in: ${SCRIPT_DIR}/${CHECKPOINT_DIR}"
echo "=================================================="
