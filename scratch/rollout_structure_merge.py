"""Structure-sensitive scores for the r = 128 rollout ensembles (FEASIBILITY §28).

The log-spectral distance is blind to where the energy sits: a run that keeps two clean,
unmixed bands can match the spectrum. These scores use the saved trajectories (filtered DNS
and every run at t = 1.0, 1.1, ... 10) instead:
  corr     pointwise density correlation with the DNS, t = 1.5-4.5 and t = 1.5-10. The coarse runs
           barely decorrelate (member pairs 1% apart: 0.92 at t = 10), so the full window is usable;
           the DNS's own predictability at these scales is not measured here
  theta    momentum thickness of the x-averaged vx profile over the DNS's, t = 1.5-10
           (both layers; U fixed from the DNS at t = 1). < 1 means the layers mix too slowly
  |dtheta| mean |theta / theta_DNS - 1|, t = 1.5-10
  prof     RMS error of the x-averaged density profile / std(rho_DNS), t = 1.5-10
  pdf      Wasserstein-1 distance between run and DNS density distributions / std(rho_DNS)
Runs that diverged are dropped (n is reported). Pairing as in minit_rollout_merge.py.
"""
import glob
import os

import numpy as np
import torch

ROOTS = {"abl": "results/ensemble_2k_r128_abl", "cell": "results/ensemble_2k_r128_minit_cell",
         "x": "results/ensemble_2k_r128_minit_x", "cell, eq-seed": "results/ensemble_2k_r128_minit_eqseed",
         "cell, eq-seed, clip": "results/ensemble_2k_r128_minit_eqseed_clip",
         "hm": "results/ensemble_2k_r128_hybrid_minit",
         "g1": "results/ensemble_2k_r128_gain/tg1.6_hg1.3", "g2": "results/ensemble_2k_r128_gain/tg2.5_hg1.6",
         "clip, adv": "results/ensemble_2k_r128_minit_advect",          # §29 advection
         "clip, adv, refit": "results/ensemble_2k_r128_minit_advect_refit"}
GAIN = {"g1": {"TBNN": 1.6, "hybrid": 1.3}, "g2": {"TBNN": 2.5, "hybrid": 1.6}}
RENAME = {"learned-2k_r128": "MiNNit", "hybrid-minitE": "TBNN shape x MInIT energy",
          "ablation-exactE": "TBNN shape x exact energy", "ablation-exactS": "exact shape x MiNNit energy"}
EARLY, WIN = (1.5, 4.5), (1.5, 10.0)


def series(run, k):
    return np.asarray(torch.as_tensor(run[k]).double())


def theta(vx, U, um):
    u = vx.mean(axis=-1)                     # x-average: (T, Ny)
    return np.mean(1 - ((u - um) / U) ** 2, axis=-1) / 4


def score(run, dns, t):
    rho, rd = series(run, "rho"), series(dns, "rho")
    if len(rho) != len(rd) or not np.isfinite(rho).all():
        return None
    s = rd.reshape(len(rd), -1).std(axis=1)
    e = (t >= EARLY[0] - 1e-9) & (t <= EARLY[1] + 1e-9)
    w = (t >= WIN[0] - 1e-9) & (t <= WIN[1] + 1e-9)
    corr = np.array([np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(rho, rd)])
    u0 = series(dns, "vx")[0].mean(axis=-1)
    U, um = (u0.max() - u0.min()) / 2, (u0.max() + u0.min()) / 2
    ratio = theta(series(run, "vx"), U, um) / theta(series(dns, "vx"), U, um)
    prof = np.sqrt(((rho.mean(-1) - rd.mean(-1)) ** 2).mean(-1)) / s
    pdf = np.abs(np.sort(rho.reshape(len(rho), -1)) - np.sort(rd.reshape(len(rd), -1))).mean(-1) / s
    return np.array([corr[e].mean(), ratio[w].mean(), np.abs(ratio[w] - 1).mean(), prof[w].mean(), pdf[w].mean(), corr[w].mean()])


ROOTS = {k: v for k, v in ROOTS.items() if glob.glob(v + "/*/rollout_trajectory.pt")}
members = sorted(os.path.basename(p) for p in glob.glob(ROOTS["abl"] + "/*"))
rows, check, spread = {}, [], {}
for m in members:
    for tag, root in ROOTS.items():
        f = f"{root}/{m}/rollout_trajectory.pt"
        if not os.path.exists(f):
            continue
        d = torch.load(f, weights_only=False)
        t = np.asarray(d["t"], dtype=float)
        runs = {"none": d["unclosed"], "TBNN": d["tbnn"], "exact": d["exact"], **d["minit"]}
        for name, run in runs.items():
            if run is None or name in d["diverged_at"]:
                continue
            name = RENAME.get(name, name)
            if name in ("equilibrium", "fitted", "paper"):
                name = f"MInIT ({name}) [{tag}]"
            if name in GAIN.get(tag, {}):
                name = f"{name} x{GAIN[tag][name]}"
            sc = score(run, d["dns"], t)
            if sc is None:
                continue
            if name in rows and m in rows[name]:          # rerun row: must reproduce
                check.append(np.abs(sc - rows[name][m]).max())
                continue
            rows.setdefault(name, {})[m] = sc
            if name == "none":
                spread.setdefault(m.rsplit("_s", 1)[0], {})[m] = series(run, "rho")

print(f"pairing check: max score difference of rerun rows = {max(check):.1e} ({len(check)} runs)")
# Intrinsic spread: the two perturbed members of the same starting state, no closure.
cs = []
for pair in spread.values():
    if len(pair) == 2:
        a, b = pair.values()
        cs.append([np.corrcoef(x.ravel(), y.ravel())[0, 1] for x, y in zip(a, b)])
cs = np.mean(cs, axis=0)
print("member-to-member density correlation, no closure (predictability):",
      "  ".join(f"t={tt:g}: {cs[int(np.argmin(abs(t - tt)))]:.2f}" for tt in (2, 3, 4.5, 6, 8, 10)))

cols = ["corr (t<=4.5)", "theta/DNS", "|dtheta|", "prof err", "pdf W1", "corr (t<=10)"]
print(f"\n{'run':<36}" + "".join(f"{c:>15}" for c in cols) + "    n")
order = sorted(rows, key=lambda k: np.mean([v[3] for v in rows[k].values()]))
for k in order:
    v = np.array(list(rows[k].values()))
    se = v.std(axis=0, ddof=1) / np.sqrt(len(v))
    print(f"{k:<36}" + "".join(f"{a:>8.3f} ±{b:.3f}" for a, b in zip(v.mean(0), se)) + f"   {len(v)}")

ref = "TBNN shape x MInIT energy"
print(f"\npaired, X - none (prof err / pdf W1 / |dtheta|; negative = X closer to DNS):")
for k in order:
    if k == "none":
        continue
    common = sorted(set(rows[k]) & set(rows["none"]))
    d = np.array([rows[k][m][[3, 4, 2]] - rows["none"][m][[3, 4, 2]] for m in common])
    se = d.std(axis=0, ddof=1) / np.sqrt(len(d))
    print(f"  {k:<36}" + "  ".join(f"{a:+.3f} ±{b:.3f} ({int((d[:, i] < 0).sum())}/{len(d)})"
                                   for i, (a, b) in enumerate(zip(d.mean(0), se))))

PAIRS = [("exact", "hybrid x1.6"), ("hybrid x1.6", "TBNN shape x MInIT energy"),
         ("TBNN shape x MInIT energy", "hybrid"), ("hybrid", "MiNNit"),
         ("MInIT (fitted) [cell, eq-seed, clip]", "MiNNit"), ("MInIT (fitted) [cell, eq-seed, clip]", "hybrid"),
         ("TBNN x2.5", "TBNN"),
         ("MInIT (fitted) [clip, adv]", "MInIT (fitted) [cell, eq-seed, clip]"),
         ("MInIT (fitted) [clip, adv, refit]", "MInIT (fitted) [cell, eq-seed, clip]"),
         ("MInIT (fitted) [clip, adv, refit]", "MiNNit")]
print("\npaired, A - B (prof err / pdf W1 / |dtheta| / corr t<=10; negative = A closer, except corr):")
for a, b in PAIRS:
    if a not in rows or b not in rows:
        continue
    common = sorted(set(rows[a]) & set(rows[b]))
    d = np.array([rows[a][m][[3, 4, 2, 5]] - rows[b][m][[3, 4, 2, 5]] for m in common])
    se = d.std(axis=0, ddof=1) / np.sqrt(len(d))
    print(f"  {a + ' - ' + b:<62}" + "  ".join(f"{x:+.3f} ±{y:.3f}" for x, y in zip(d.mean(0), se)))
