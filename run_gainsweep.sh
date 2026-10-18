#!/bin/bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source .venv/bin/activate
GAIN="$1"
echo "=== learned-gain = ${GAIN} ==="
PYTHONPATH=. python -u -m src.evaluate_rollout \
    --data-dir dns_khi --coarsening-factor 16 \
    --checkpoint checkpoints_fullrange/best_model.pt \
    --t-start 1.0 --t-end 10.0 --cfl 0.75 \
    --reconstruction mp5 --riemann-solver hllc \
    --device cuda --skip-exact --with-unclosed \
    --with-learned results/minit/minit_coeff_residual.pt \
    --learned-gain "${GAIN}" \
    --output-dir "results/gainsweep_g${GAIN}"
