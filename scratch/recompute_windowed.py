"""Recompute windowed band ratios for saved rollouts with the current metric.

Used after fixing the frame-alignment tolerance (FEASIBILITY §21): rebuilds the
nearest-snapshot reference from the DNS, recomputes evaluate_rollout's windows (thirds of
[t_start + 0.5, t_end] plus the span) and writes them back into each metrics.json.
Usage: python scratch/recompute_windowed.py results/ensemble_r64 [...]
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import read_snapshot, relativistic_favre_filter  # noqa: E402
from src.evaluate_rollout import _variant_label, windowed_band_ratios  # noqa: E402

cache: dict = {}
for root in sys.argv[1:]:
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        tr = torch.load(os.path.join(d, "rollout_trajectory.pt"), weights_only=False)
        data = re.fullmatch(r"(dns_khi(?:_s1)?)_s\d+", os.path.basename(d)).group(1)
        r = tr["grid"]["r"]
        runs = {n: h for n, h in (("unclosed", tr["unclosed"]), ("TBNN", tr["tbnn"]),
                                  ("exact DNS", tr["exact"])) if h is not None}
        runs.update({_variant_label(v): h for v, h in sorted(tr["minit"].items())})
        times = np.asarray(max(runs.values(), key=lambda h: len(h["t"]))["t"], dtype=np.float64)
        if data not in cache:
            snaps = sorted((read_snapshot(f) for f in sorted(glob.glob(f"{data}/h*.h5"))),
                           key=lambda s: float(s["t"]))
            cache[data] = (np.array([float(s["t"]) for s in snaps]), snaps, {})
        snap_t, snaps, filt = cache[data]
        nearest = np.abs(times[:, None] - snap_t[None, :]).argmin(axis=1)
        tol = 0.5 * float(np.median(np.diff(snap_t)))
        valid = np.abs(times - snap_t[nearest]) <= tol
        for j in set(nearest.tolist()):
            if (j, r) not in filt:
                filt[(j, r)] = relativistic_favre_filter(
                    *(torch.tensor(snaps[j][q], dtype=torch.float64) for q in ("rho", "vx", "vy", "p")), r)
        ref = {"t": times, **{q: torch.stack([filt[(j, r)][src] for j in nearest])
                              for q, src in (("rho", "rho_bar"), ("vx", "vx_tilde"), ("vy", "vy_tilde"))}}
        t_lo, t_end = float(times[0]) + 0.5, float(times[-1])
        windows = [(t_lo + k * (t_end - t_lo) / 3, t_lo + (k + 1) * (t_end - t_lo) / 3)
                   for k in range(3)] + [(t_lo, t_end)]
        w = windowed_band_ratios(ref, runs, valid, windows, baseline="exact DNS", t_tol=tol)
        mpath = os.path.join(d, "metrics.json")
        m = json.load(open(mpath))
        m["windowed_band_ratios"] = w
        json.dump(m, open(mpath, "w"), indent=2)
        part = sum(rw["partial"] for rows in w.values() for rw in rows.values())
        print(f"{d}: valid frames {int(valid.sum())}/{len(times)}, partial rows {part}")
