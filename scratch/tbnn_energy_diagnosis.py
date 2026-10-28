"""Where does the TBNN's sub-grid energy go wrong in simulation? (TASKS.md, 28/10 step 2a)

On the saved 2048^2, r = 128 rollout trajectories (results/ensemble_2k_r128*, 5 unseen
realisations x 2 members), evaluate each closure's energy e = (tau^xx + tau^yy)/2 on the
states of every run -- filtered DNS, exact-stress run, no closure, TBNN, MiNNit, hybrid,
TBNN x MInIT-ODE hybrid -- and compare with the exact filtered-DNS energy at the same time.

    e_TBNN    the closure as the solver calls it (realizability clip ||tau||_F <= 0.5 p),
              plus the unclipped tensor to see whether the clip matters
    e_MiNNit  LearnedMInITClosure.e_KH
    e_MInIT   refitted ODE integrated along the run's own (rho, |dy vx|), equilibrium seed

Then: the TBNN's 17 standardised inputs on DNS vs simulation states, and a counterfactual
that sets some inputs on DNS states to their simulation value.

Run from the repo root (~5 min; the energies are cached in the output dir):
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. ./.venv/bin/python -u scratch/tbnn_energy_diagnosis.py \
        [tbnn_checkpoint (checkpoints_2k_r128/best_model.pt)] [out_dir (results/tbnn_energy_diag)]
The states always come from the saved rollouts above, whichever TBNN is scored.
"""
import sys
from pathlib import Path

import torch

from src.evaluate_rollout import TBNNSGSClosure
from src.features import extract_tbnn_features
from src.minit import LAMBDA_OVER_DELTA, MInITCoefficients
from src.minit_calibrate import evolve_e_series
from src.minit_learned import INVARIANT_NAMES, LearnedMInITClosure, compute_invariants
from src.solver import clip_sgs_tensor
from utils import CRMHD_TBNN

DT = torch.float64
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_2k_r128/best_model.pt"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "results/tbnn_energy_diag")
RUNS = [f"dns2k_s{s}_s{m}" for s in range(1, 6) for m in (0, 1)]
FAMILIES = ["dns", "exact", "unclosed", "tbnn", "minnit", "hybrid", "hybrid-minitE"]
WINDOWS = [(1.5, 4.0), (4.0, 7.0), (7.0, 10.0)]
# Inputs 0-14 are the invariants Λ1-15 (Λ6-10, Λ14-15 involve b and are 0 here), 15 = θ, 16 = Mach.
COUNTERFACTUALS = {"accel (Λ11-13)": [10, 11, 12], "θ": [15], "Mach": [16],
                   "accel + θ": [10, 11, 12, 15], "accel + θ + Mach": [10, 11, 12, 15, 16]}

model, ck = CRMHD_TBNN.from_checkpoint(CKPT, device="cpu", dtype=DT)
model.eval()
mc = MInITCoefficients.load("results/minit_fair/coeffs_2k_r128_cell.json")
A_, C_ = (torch.tensor([v], dtype=DT) for v in (mc.A_gamma, mc.C))
dx = 1.0 / 16
lam = LAMBDA_OVER_DELTA / 16
norm = [ck[k].to(DT) for k in ("inv_mean", "inv_std", "thermo_mean", "thermo_std")]
tb = TBNNSGSClosure(model, *norm, dx, dx)
tb_raw = TBNNSGSClosure(model, *norm, dx, dx, clip_kappa=1e30)
le = LearnedMInITClosure.load("results/dns2k/minit_coeff_2k_r128.pt", dx, dx)
energy = lambda tau: 0.5 * (tau[..., 1, 1] + tau[..., 2, 2])


def inputs(q):
    """The TBNN's standardised inputs (256, 17) and basis tensors (256, 12, 4, 4) for one state."""
    fe = extract_tbnn_features(q[..., 0], q[..., 1], q[..., 2], q[..., 3], dx, dx, gradient_order=4)
    x = torch.cat([(fe["invariants"].reshape(-1, 15) - norm[0]) / norm[1],
                   (fe["thermo_inputs"].reshape(-1, 2) - norm[2]) / norm[3]], -1)
    return x, fe["basis_tensors"].reshape(-1, 12, 4, 4).to(DT)


def states(run):
    a = torch.load(f"results/ensemble_2k_r128/{run}/rollout_trajectory.pt", weights_only=False)
    b = torch.load(f"results/ensemble_2k_r128_hybrid_minit/{run}/rollout_trajectory.pt", weights_only=False)
    src = {"dns": a["dns"], "exact": a["exact"], "unclosed": a["unclosed"], "tbnn": a["tbnn"],
           "minnit": a["minit"]["learned-2k_r128"], "hybrid": a["minit"]["hybrid"],
           "hybrid-minitE": b["minit"]["hybrid-minitE"]}
    return torch.as_tensor(a["dns"]["t"], dtype=DT), {
        k: torch.stack([torch.as_tensor(v[q], dtype=DT) for q in ("rho", "vx", "vy", "p")], -1)
        for k, v in src.items()}


def exact_e(seed, t):
    """Exact filtered-DNS energy, linearly interpolated to the rollout's output times."""
    d = torch.load(f"results/dns2k/reduction_r128_s{seed}.pt", weights_only=False)["data"]
    ts, e = d["t"], d["e_dns"]
    i = torch.searchsorted(ts, t).clamp(1, len(ts) - 1)
    w = ((t - ts[i - 1]) / (ts[i] - ts[i - 1]))[:, None, None]
    return (1 - w) * e[i - 1] + w * e[i]


def compute():
    rows = {f: {k: [] for k in ("t", "ex", "tb", "tb_raw", "mn", "mi", "base", "x")} for f in FAMILIES}
    for run in RUNS:
        t, P = states(run)
        ex = exact_e(int(run[7]), t)
        for f in FAMILIES:
            p = P[f]
            with torch.no_grad():
                e_tb = torch.stack([energy(tb(q)[0]) for q in p])
                e_raw = torch.stack([energy(tb_raw(q)[0]) for q in p])
                e_mn = torch.stack([(le(q), le.e_KH)[1] for q in p])
            inv, base = compute_invariants(p[..., 0], p[..., 1], p[..., 2], dx, dx, lam)
            sh, rho = inv[..., INVARIANT_NAMES.index("shear")], p[..., 0]
            e0 = (2 * mc.A_gamma * sh[0] * rho[0].sqrt() * lam / mc.C) ** 2      # equilibrium seed
            r = rows[f]
            r["t"].append(t); r["ex"].append(ex); r["tb"].append(e_tb); r["tb_raw"].append(e_raw)
            r["mn"].append(e_mn); r["mi"].append(evolve_e_series(e0, sh, rho, t, A_, C_, lam, 10)[0])
            r["base"].append(base); r["x"].append(torch.stack([inputs(q)[0] for q in p]))
        print("done", run, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    D = {f: {k: torch.stack(v) for k, v in r.items()} for f, r in rows.items()}
    torch.save(D, OUT / "energies.pt")
    return D


D = torch.load(OUT / "energies.pt", weights_only=False) if (OUT / "energies.pt").exists() else compute()
t = D["dns"]["t"][0]
win = lambda a, b: (t >= a) & (t < b)
ratio = lambda r, m, k: r[m][:, k].mean((1, 2, 3)) / r["ex"][:, k].mean((1, 2, 3))   # per run

print("A. <e_closure> on each run's states / <e_exact> of the filtered DNS at the same t  (mean ± sd, 10 runs)")
print(f"{'states':14s}" + "".join(f"{f'{m} {a:g}-{b:g}':>14s}" for m in ("TBNN", "MiNNit", "MInIT") for a, b in WINDOWS)
      + f"{'TBNN no clip 7-10':>19s}")
for f, r in D.items():
    s = f"{f:14s}" + "".join(f"{q.mean():>9.2f}±{q.std():.2f}" for m in ("tb", "mn", "mi")
                             for q in (ratio(r, m, win(a, b)) for a, b in WINDOWS))
    print(s + f"{ratio(r, 'tb_raw', win(7, 10)).mean():>19.2f}")

print("\nB. resolved strain energy <ρλ²|S|²> / same on DNS states   |   cells with e_TBNN <= 5% of <e_MiNNit>")
for f, r in D.items():
    s = f"{f:14s}"
    for a, b in WINDOWS:
        k = win(a, b)
        s += f"{(r['base'][:, k].mean((1, 2, 3)) / D['dns']['base'][:, k].mean((1, 2, 3))).mean():8.2f}"
    s += "   |"
    for a, b in WINDOWS:
        k = win(a, b)
        s += f"{(r['tb'][:, k] <= 0.05 * r['mn'][:, k].mean((2, 3), keepdim=True)).double().mean():8.2f}"
    print(s)

print("\nC. TBNN standardised inputs, t 4-10: median [p10, p90]   (rows with no spread anywhere omitted)")
k = win(4, 10)
X = {f: D[f]["x"][:, k].reshape(-1, 17) for f in ("dns", "exact", "unclosed", "tbnn")}
qs = torch.tensor([0.1, 0.5, 0.9], dtype=DT)
print(f"{'input':6s}" + "".join(f"{f:>26s}" for f in X))
for i in range(17):
    q = {f: torch.quantile(x[:, i], qs) for f, x in X.items()}
    if max(float(v[2] - v[0]) for v in q.values()) < 1e-3:
        continue
    print(f"x{i:02d}  " + "".join(f"{v[1]:>+10.2f} [{v[0]:+6.2f},{v[2]:+6.2f}]" for v in q.values()))

print("\nD. Counterfactual on DNS states (s1-s5, t 4-10): set inputs to their median on no-closure states")
x_sim = D["unclosed"]["x"].reshape(-1, 17).median(0).values
tot = {c: 0.0 for c in ["as is", *COUNTERFACTUALS]}
for run in RUNS[::2]:
    t_run, P = states(run)
    for q in P["dns"][win(4, 10)]:
        x, B = inputs(q)
        for c in tot:
            xc = x.clone()
            xc[:, COUNTERFACTUALS.get(c, [])] = x_sim[COUNTERFACTUALS.get(c, [])]
            with torch.no_grad():
                # TBNNSGSClosure converted the model to float32 in place; match it
                T = model(xc[:, :15].float(), xc[:, 15:].float(), B.float())[0].to(DT).reshape(16, 16, 4, 4)
            tot[c] += float(energy(clip_sgs_tensor(T, q, mode="frobenius", kappa=0.5)).sum())
for c, v in tot.items():
    print(f"  {c:20s} <e_TBNN> = {v / tot['as is']:.2f} x as-is")
