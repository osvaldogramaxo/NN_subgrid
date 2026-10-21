"""Merge the r = 128 rollout ensembles that share starting states (FEASIBILITY §25).

results/ensemble_2k_r128_abl holds TBNN / MiNNit / hybrid / exact; the MInIT jobs hold MInIT
variants plus a no-closure control, which must reproduce the ablation job's to round-off
(checked) for the pairing to be valid. Prints pooled spectral distance and KE over t = 1.5-10
and MiNNit's paired difference to each MInIT variant.
"""
import glob
import json
import os

import numpy as np

ROOTS = {"abl": "results/ensemble_2k_r128_abl", "cell": "results/ensemble_2k_r128_minit_cell",
         "x": "results/ensemble_2k_r128_minit_x", "cell, eq-seed": "results/ensemble_2k_r128_minit_eqseed",
         "cell, eq-seed, clip": "results/ensemble_2k_r128_minit_eqseed_clip"}
ROOTS = {k: v for k, v in ROOTS.items() if glob.glob(v + "/*/metrics.json")}
members = sorted(os.path.basename(p) for p in glob.glob(ROOTS["abl"] + "/*"))
rows, check = {}, []
for m in members:
    got = {}
    for tag, root in ROOTS.items():
        f = f"{root}/{m}/metrics.json"
        if not os.path.exists(f):
            continue
        w = json.load(open(f))["windowed_band_ratios"]
        for run, r in w[list(w)[-1]].items():
            if r.get("partial") or run.startswith("ablation"):
                continue
            name = {"unclosed": "none", "TBNN": "TBNN", "exact DNS": "exact"}.get(run, run)
            name = "MiNNit" if name.startswith("learned-coeff") else "hybrid" if name.startswith("hybrid") else name
            if name == "none" and tag != "abl":
                check.append(abs(r["log_spectral_distance"] - got["none"][0]))
                continue
            got[f"{name} [{tag}]" if name.startswith("MInIT") else name] = (
                r["log_spectral_distance"], r["e_ratio_total"])
    for k, v in got.items():
        rows.setdefault(k, {})[m] = v
print(f"pairing check: max |LSD(no closure)| difference across jobs = {max(check):.1e} ({len(check)} runs)")
print(f"\n{'run (t = 1.5-10)':<36} {'spectral dist':>15} {'total KE':>9}")
# A run that diverged is flagged partial and dropped, so its member is missing here; any
# average over the survivors is biased toward the runs that held together. Report n, and
# pair only members present in both rows.
order = sorted(rows, key=lambda k: np.mean([x[0] for x in rows[k].values()]))
for k in order:
    v = np.array(list(rows[k].values()))
    print(f"{k:<36} {v[:, 0].mean():8.3f} ± {v[:, 0].std(ddof=1) / np.sqrt(len(v)):.3f} {v[:, 1].mean():9.3f}  (n={len(v)})")
print("\npaired, MiNNit - X over the members both completed (negative = MiNNit closer to DNS):")
for k in order:
    if k.startswith("MInIT") or k == "hybrid":
        common = sorted(set(rows["MiNNit"]) & set(rows[k]))
        d = np.array([rows["MiNNit"][m][0] - rows[k][m][0] for m in common])
        print(f"  vs {k:<32} {d.mean():+.3f} ± {d.std(ddof=1) / np.sqrt(len(d)):.3f}  "
              f"(MiNNit better in {int((d < 0).sum())}/{len(d)})")
