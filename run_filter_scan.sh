#!/bin/bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
PYTHONPATH=. python -u scratch/filter_scan.py "$1"
