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

# Default training arguments
DATA_DIR="${1:-test1}"
COARSENING="${2:-4}"
STRIDE="${3:-5}"
EPOCHS="${4:-100}"
BATCH_SIZE="${5:-2048}"
LR="${6:-1e-3}"
DEVICE="${7:-cuda}"
PLOT_INTERVAL="${8:-10}"
PATIENCE="${9:-}"

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

mkdir -p checkpoints condor_logs checkpoints/plots

echo "Starting training: data=${DATA_DIR}, r=${COARSENING}, stride=${STRIDE}, epochs=${EPOCHS}, plot_interval=${PLOT_INTERVAL}, device=${DEVICE}, patience=${PATIENCE:-default (2x LR patience)}, extra_flags=${EXTRA_ARGS[*]:-none}"

python -m src.train_apriori \
    --data-dir "${DATA_DIR}" \
    --coarsening-factor "${COARSENING}" \
    --stride "${STRIDE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --plot-interval "${PLOT_INTERVAL}" \
    --checkpoint-dir checkpoints \
    --device "${DEVICE}" \
    --dtype float64 \
    "${EXTRA_ARGS[@]}"

echo "=================================================="
echo "Training finished at: $(date)"
echo "Checkpoints saved in: ${SCRIPT_DIR}/checkpoints"
echo "=================================================="
