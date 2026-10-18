"""Filter-size scan (TASKS.md items 1-3): reduce both DNS runs at coarsening r,
then fit MiNNit on dns_khi and score it, the oracle-e ceiling and global-k MInIT
on dns_khi_s1. Usage: python scratch/filter_scan.py R
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.minit_calibrate import reduce_dns  # noqa: E402
from src import train_minit_coeff  # noqa: E402

r = int(sys.argv[1])
caches = {}
for data_dir, suffix in (("dns_khi", ""), ("dns_khi_s1", "_s1")):
    # r=16 keeps the names every other script already uses.
    path = Path(f"results/minit/dns_reduction{suffix}.pt" if r == 16
                else f"results/minit/dns_reduction_r{r}{suffix}.pt")
    key = {"data_dir": data_dir, "r": r, "stride": 1}
    if not (path.exists() and torch.load(path, weights_only=False).get("key") == key):
        torch.save({"key": key, "data": reduce_dns(data_dir, r, 1)}, path)
    caches[data_dir] = str(path)

sys.exit(train_minit_coeff.main([
    "--train-cache", caches["dns_khi"], "--test-cache", caches["dns_khi_s1"],
    "--device", "cpu", "--tag", f"r{r}", "--out-dir", "results/filter_scan",
]))
