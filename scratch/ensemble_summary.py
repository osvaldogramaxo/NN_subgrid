"""Aggregate the rollout ensemble (condor_ensemble.sub) across members.

Each member is one rollout per closure from the same (perturbed) initial state, scored
by evaluate_rollout's windowed band ratios. Within a member the closures are paired;
across members we get the spread a single trajectory cannot show. Reported per DNS
realisation and pooled, per regime window:

  LSD      log-spectral distance to the filtered DNS (lower is better)
  KE       total kinetic energy / DNS
  hi/orc   high-band energy relative to the exact-stress oracle (1 = as good as the
           true stress; raw retention is not a score -- more than the oracle is not better)
  mid      mid-band energy / DNS

as mean +- standard error over members, plus how many members each closure has the
lowest LSD in.

Usage: python scratch/ensemble_summary.py [results/ensemble]
"""
import glob
import json
import os
import re
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "results/ensemble"
NAMES = {"unclosed": "unclosed", "TBNN": "TBNN", "learned-coeff": "MiNNit",
         "hybrid (TBNN shape x MiNNit amp)": "hybrid", "exact DNS": "oracle",
         "ablation (TBNN shape x exact energy)": "exactE",
         "ablation (exact shape x MiNNit energy)": "exactS"}


def canon(members):
    """The MiNNit row is labelled by its checkpoint ("learned-coeff (residual)" at r=16,
    "learned-coeff (r32)" from the filter scan); fold any of them into one key."""
    for rows in members.values():
        for w in rows.values():
            for k in [k for k in w if k.startswith("learned-coeff (")]:
                w["learned-coeff"] = w.pop(k)
    return members

members = {}
for f in sorted(glob.glob(os.path.join(root, "*", "metrics.json"))):
    tag = os.path.basename(os.path.dirname(f))
    w = json.load(open(f)).get("windowed_band_ratios")
    if w:
        members[tag] = w
if not members:
    sys.exit(f"no windowed metrics under {root}")
members = canon(members)

windows = list(next(iter(members.values())).keys())
# Folder names are <data>_s<seed>, and "dns_khi_s1" is dns_khi seed 1 -- not the
# unseen run, whose members are dns_khi_s1_s<seed>. Parse rather than prefix-match.
# The greedy prefix takes everything up to the LAST "_s<digits>": dns_khi_s1 -> dns_khi
# member 1, dns_khi_s1_s0 -> dns_khi_s1 member 0, dns2k_s3_s1 -> dns2k_s3 member 1.
parsed = {m: re.fullmatch(r"(.+)_s(\d+)", m) for m in members}
groups = {}
for m, p in sorted(parsed.items()):
    if p:
        groups.setdefault(p.group(1), []).append(m)
groups["pooled"] = list(members)
print(f"{len(members)} members: {', '.join(members)}")


def stat(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    if len(v) == 0:
        return "   --   "
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return f"{v.mean():.3f}±{se:.3f}"


for gname, tags in groups.items():
    if not tags:
        continue
    print(f"\n######## {gname}: {len(tags)} members")
    for wl in windows:
        print(f"\n  t = {wl}")
        print(f"    {'run':<9} {'LSD':>12} {'KE':>12} {'hi/orc':>12} {'mid':>12}  lowest-LSD")
        best = {}
        for t in tags:
            rows = members[t].get(wl, {})
            if rows:
                b = min(rows, key=lambda k: rows[k]["log_spectral_distance"])
                best[NAMES.get(b, b)] = best.get(NAMES.get(b, b), 0) + 1
        for key, nm in NAMES.items():
            rows = [members[t].get(wl, {}).get(key) for t in tags]
            rows = [r for r in rows if r and not r.get("partial")]
            if not rows:
                continue
            print(f"    {nm:<9} {stat([r['log_spectral_distance'] for r in rows]):>12} "
                  f"{stat([r['e_ratio_total'] for r in rows]):>12} "
                  f"{stat([r.get('vs_baseline_high') for r in rows]):>12} "
                  f"{stat([r['band_ratio_mid'] for r in rows]):>12}  {best.get(nm, 0)}/{len(tags)}")

    # paired across members: hybrid minus each rival, per window, on LSD
    print(f"\n  paired LSD difference, hybrid - X (negative = hybrid closer to DNS), mean ± s.e. over members")
    for wl in windows:
        out = []
        for key, nm in NAMES.items():
            if nm == "hybrid":
                continue
            d = []
            for t in tags:
                rows = members[t].get(wl, {})
                h, x = rows.get("hybrid (TBNN shape x MiNNit amp)"), rows.get(key)
                if h and x and not h.get("partial") and not x.get("partial"):
                    d.append(h["log_spectral_distance"] - x["log_spectral_distance"])
            if d:
                d = np.array(d)
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
                out.append(f"vs {nm} {d.mean():+.3f}±{se:.3f} ({int((d < 0).sum())}/{len(d)})")
        print(f"    t = {wl}: " + "; ".join(out))
