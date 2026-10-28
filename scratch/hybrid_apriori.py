"""A-priori: TBNN vs MiNNit vs hybrid (TBNN shape x MiNNit amplitude) on the
unseen dns_khi_s1 run, every model run through its actual solver closure.
Usage: python scratch/hybrid_apriori.py [r] [data_dir] [tbnn_ckpt] [minnit_ckpt] [minit_coeffs reduction]
(defaults: 16, dns_khi_s1, checkpoints_r{r}/best_model.pt, the r's MiNNit checkpoint)
With MInIT coefficients and the run's reduction cache, also scores TBNN shape x MInIT's
evolved energy (FEASIBILITY §26): the cell-by-cell ODE integrated along the filtered DNS
from t = 1, seeded with the exact energy or at its own equilibrium (as in the rollouts).
Also scores the TBNN and hybrid with their stress scaled by the rollout gains (FEASIBILITY §27).
APRIORI_WINDOW="t0,t1,stride" sets the snapshots scored (default "1,inf,10": t = 1, 2, ... 20;
"1.5,10,5" is the rollout window t = 1.5-10 every 0.5).
"""
import glob
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import compute_sgs_stress, read_snapshot  # noqa: E402
from src.evaluate_rollout import TBNNSGSClosure  # noqa: E402
from src.features import periodic_gradient_2d  # noqa: E402
from src.minit_learned import HybridClosure, LearnedMInITClosure  # noqa: E402
from src.train_minit_coeff import score_tau  # noqa: E402
from utils import CRMHD_TBNN  # noqa: E402

r = int(sys.argv[1]) if len(sys.argv) > 1 else 16
data_dir = sys.argv[2] if len(sys.argv) > 2 else "dns_khi_s1"
DT = torch.float64
ckpt_t = sys.argv[3] if len(sys.argv) > 3 else f"checkpoints_r{r}/best_model.pt"
ckpt_l = sys.argv[4] if len(sys.argv) > 4 else (
    "results/minit/minit_coeff_residual.pt" if r == 16 else f"results/filter_scan/minit_coeff_r{r}.pt")
model, ck = CRMHD_TBNN.from_checkpoint(ckpt_t, device="cpu", dtype=DT)
model.eval()

t0, t1, stride = os.environ.get("APRIORI_WINDOW", "1,inf,10").split(",")
files = sorted(glob.glob(f"{data_dir}/h*.h5"))
snaps = [s for s in (read_snapshot(f) for f in files[::int(stride)])
         if float(t0) - 1e-9 <= s["t"] <= float(t1) + 1e-9]
GAINS = {"TBNN": (1.6, 2.5), "hybrid": (1.3, 1.6)}   # condor_ensemble_2k_gain.sub
taus = {"TBNN": [], "MiNNit": [], "hybrid": [], "TBNN shape x exact e": [], "exact shape x MiNNit e": []}
taus.update({f"{m} x{g}": [] for m, gs in GAINS.items() for g in gs})
if model.shape_only:    # a shape network has no stress of its own; only its hybrids are scored
    for k in ["TBNN", *(f"TBNN x{g}" for g in GAINS["TBNN"])]:
        del taus[k]
e_minit = {}
if len(sys.argv) > 6:
    from src.minit import LAMBDA_OVER_DELTA, MInITCoefficients, assemble_tau, saturation_energy  # noqa: E402
    from src.minit_calibrate import evolve_e_series  # noqa: E402
    from src.train_minit_coeff import load_reduction  # noqa: E402
    mc = MInITCoefficients.load(sys.argv[5])
    red = load_reduction(Path(sys.argv[6]), 1.0, mc.energy)
    lam = LAMBDA_OVER_DELTA / red["rho"].shape[-1]
    A, C = (torch.tensor([v], dtype=DT) for v in (mc.A_gamma, mc.C))
    seeds = {"DNS seed": red["e_dns"][0].clamp(min=0.0),
             "eq. seed": saturation_energy(mc.A_gamma * red["shear"][0], red["rho"][0], lam, mc.C)}
    for tag, e0 in seeds.items():
        e_minit[tag] = evolve_e_series(e0, red["shear"], red["rho"], red["t"], A, C, lam, 10)[0]
        taus[f"TBNN shape x MInIT e ({tag})"] = []
    taus["MInIT ODE (eq. seed)"] = []
exact, S_all, fb = [], [], []
for snap in snaps:
    fine = [torch.tensor(snap[k], dtype=DT) for k in ("rho", "vx", "vy", "p")]
    c, tau_ex = compute_sgs_stress(*fine, r=r, project_to_fluid_frame=True)
    Ny, Nx = c["rho_bar"].shape
    dx, dy = 1.0 / Nx, 1.0 / Ny
    prims = torch.stack([c["rho_bar"], c["vx_tilde"], c["vy_tilde"], c["p_bar"]], -1)
    tb = TBNNSGSClosure(model, ck["inv_mean"], ck["inv_std"], ck["thermo_mean"],
                        ck["thermo_std"], dx, dy)
    le = LearnedMInITClosure.load(ckpt_l, dx, dy)
    hy = HybridClosure(tb, le)
    exact_c = lambda _p, t=tau_ex: (t, torch.ones(t.shape[:2], dtype=DT))   # this snapshot's exact stress
    abl_e = HybridClosure(tb, le, energy_closure=exact_c)   # TBNN shape, exact energy
    abl_s = HybridClosure(exact_c, le)                       # exact shape, MiNNit energy
    pick = lambda t: torch.stack([t[..., 1, 1], t[..., 2, 2], t[..., 1, 2]], -1)
    with torch.no_grad():
        if not model.shape_only:
            taus["TBNN"].append(pick(tb(prims)[0]))
        taus["MiNNit"].append(pick(le(prims)[0]))
        taus["hybrid"].append(pick(hy(prims)[0]))
        taus["TBNN shape x exact e"].append(pick(abl_e(prims)[0]))
        taus["exact shape x MiNNit e"].append(pick(abl_s(prims)[0]))
        for g in GAINS["TBNN"] if not model.shape_only else ():
            tb_g = TBNNSGSClosure(model, ck["inv_mean"], ck["inv_std"], ck["thermo_mean"],
                                  ck["thermo_std"], dx, dy, gain=g)
            taus[f"TBNN x{g}"].append(pick(tb_g(prims)[0]))
        for g in GAINS["hybrid"]:
            le_g = LearnedMInITClosure.load(ckpt_l, dx, dy)
            le_g.cfg.gain = g
            taus[f"hybrid x{g}"].append(pick(HybridClosure(tb, le_g)(prims)[0]))
        for tag, es in e_minit.items():
            e = es[int(torch.argmin((red["t"] - snap["t"]).abs()))]
            tau_m = torch.zeros(*e.shape, 4, 4, dtype=DT)
            tau_m[..., 1, 1] = tau_m[..., 2, 2] = e       # only its trace, e, is used
            hm = HybridClosure(tb, le, energy_closure=lambda _p, t=tau_m: (t, None))
            taus[f"TBNN shape x MInIT e ({tag})"].append(pick(hm(prims)[0]))
            if tag == "eq. seed":
                taus["MInIT ODE (eq. seed)"].append(pick(assemble_tau(e, prims, mc)))
    fb.append(hy.fallback_fraction)
    exact.append(pick(tau_ex))
    a, b_ = periodic_gradient_2d(c["vx_tilde"], dx, dy, order=4)
    cc, d_ = periodic_gradient_2d(c["vy_tilde"], dx, dy, order=4)
    S_all.append(torch.stack([a, d_, 0.5 * (b_ + cc)], -1))

d = {"tau": torch.stack(exact), "S": torch.stack(S_all)}
d["e_dns"] = 0.5 * (d["tau"][..., 0] + d["tau"][..., 1])
k_hi = Nx // 4
print(f"r={r}, {len(snaps)} snapshots of {data_dir} (t = {snaps[0]['t']:g}-{snaps[-1]['t']:g}), hybrid fallback cells "
      f"{100 * sum(fb) / len(fb):.1f}%")
print(f"{'model':<34} {'amp':>6} {'corr':>7} {'Eq25':>7} {'Pi/Pi_ex':>9} {'hi-k':>6} "
      f"{'hi-k ex':>8}  amp 1st->2nd")
for name, series in taus.items():
    s = score_tau(torch.stack(series), d, stride=1, k_hi=k_hi)
    print(f"{name:<34} {s['amp']:>6.3f} {s['corr']:>+7.3f} {s['l2_eq25']:>7.3f} "
          f"{s['pi']:>9.3f} {s['hik']*100:>5.1f}% {s['hik_exact']*100:>7.1f}%  "
          f"{s['amp_first']:.3f} -> {s['amp_second']:.3f}")
