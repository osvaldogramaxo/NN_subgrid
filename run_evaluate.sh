#!/bin/bash
# ==============================================================================
# run_evaluate.sh
# Shell execution wrapper for HTCondor CRMHD-TBNN rollout evaluation
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

DATA_DIR="${1:-test1}"
COARSENING="${2:-4}"
CHECKPOINT="${3:-checkpoints/best_model.pt}"
T_START="${4:-0.0}"
T_END="${5:-10.0}"
DEVICE="${6:-cuda}"
CFL="${7:-0.3}"
RECON="${8:-mp5}"
RIEMANN="${9:-hll}"

if [ "${DEVICE}" = "cuda" ] && command -v nvidia-smi &> /dev/null; then
    echo "------------------ NVIDIA-SMI ------------------"
    nvidia-smi || true
    echo "------------------------------------------------"
elif [ "${DEVICE}" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
fi

mkdir -p results condor_logs

echo "Starting rollout evaluation: t=[${T_START}, ${T_END}], r=${COARSENING}, cfl=${CFL}, recon=${RECON}, riemann=${RIEMANN}, device=${DEVICE}"

python -m src.evaluate_rollout \
    --data-dir "${DATA_DIR}" \
    --coarsening-factor "${COARSENING}" \
    --checkpoint "${CHECKPOINT}" \
    --t-start "${T_START}" \
    --t-end "${T_END}" \
    --cfl "${CFL}" \
    --reconstruction "${RECON}" \
    --riemann-solver "${RIEMANN}" \
    --output-dir results \
    --device "${DEVICE}" \
    "${@:10}"

echo "=================================================="
echo "Evaluation finished at: $(date)"
echo "Results and figures saved in: ${SCRIPT_DIR}/results"
echo "=================================================="
