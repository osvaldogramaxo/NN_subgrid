#!/bin/bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source .venv/bin/activate
if [ $# -eq 0 ]; then set -- scratch/gpu_conv_probe.py; fi
PYTHONPATH=. python -u "$@"
