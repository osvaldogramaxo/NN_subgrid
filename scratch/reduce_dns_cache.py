"""Reduce one DNS run at coarsening r into the cache format the MiNNit scripts read.
Usage: python scratch/reduce_dns_cache.py DATA_DIR R OUT.pt"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.minit_calibrate import reduce_dns  # noqa: E402

data_dir, r, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
torch.save({"key": {"data_dir": data_dir, "r": r, "stride": 1},
            "data": reduce_dns(data_dir, r, 1)}, out)
print("wrote", out)
