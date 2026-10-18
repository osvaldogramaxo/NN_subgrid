#!/bin/bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source .venv/bin/activate
PYTHONPATH=. python -u "${1:-scratch/gpu_conv_probe.py}"
