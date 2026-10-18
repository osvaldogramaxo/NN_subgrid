"""A-priori: TBNN vs MiNNit vs hybrid (TBNN shape x MiNNit amplitude) on the
unseen dns_khi_s1 run, every model run through its actual solver closure.
Usage: python scratch/hybrid_apriori.py [r]
"""
import glob
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
DT = torch.float64
ckpt_l = "results/minit/minit_coeff_residual.pt" if r == 16 else f"results/filter_scan/minit_coeff_r{r}.pt"
model, ck = CRMHD_TBNN.from_checkpoint(f"checkpoints_r{r}/best_model.pt", device="cpu", dtype=DT)
model.eval()

files = sorted(glob.glob("dns_khi_s1/h*.h5"))
snaps = [s for s in (read_snapshot(f) for f in files[::10]) if s["t"] >= 1.0 - 1e-9]
taus = {"TBNN": [], "MiNNit": [], "hybrid": []}
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
    pick = lambda t: torch.stack([t[..., 1, 1], t[..., 2, 2], t[..., 1, 2]], -1)
    with torch.no_grad():
        taus["TBNN"].append(pick(tb(prims)[0]))
        taus["MiNNit"].append(pick(le(prims)[0]))
        taus["hybrid"].append(pick(hy(prims)[0]))
    fb.append(hy.fallback_fraction)
    exact.append(pick(tau_ex))
    a, b_ = periodic_gradient_2d(c["vx_tilde"], dx, dy, order=4)
    cc, d_ = periodic_gradient_2d(c["vy_tilde"], dx, dy, order=4)
    S_all.append(torch.stack([a, d_, 0.5 * (b_ + cc)], -1))

d = {"tau": torch.stack(exact), "S": torch.stack(S_all)}
d["e_dns"] = 0.5 * (d["tau"][..., 0] + d["tau"][..., 1])
k_hi = Nx // 4
print(f"r={r}, {len(snaps)} snapshots of dns_khi_s1, hybrid fallback cells "
      f"{100 * sum(fb) / len(fb):.1f}%")
print(f"{'model':<8} {'amp':>6} {'corr':>7} {'Eq25':>7} {'Pi/Pi_ex':>9} {'hi-k':>6} "
      f"{'hi-k ex':>8}  amp 1st->2nd")
for name, series in taus.items():
    s = score_tau(torch.stack(series), d, stride=1, k_hi=k_hi)
    print(f"{name:<8} {s['amp']:>6.3f} {s['corr']:>+7.3f} {s['l2_eq25']:>7.3f} "
          f"{s['pi']:>9.3f} {s['hik']*100:>5.1f}% {s['hik_exact']*100:>7.1f}%  "
          f"{s['amp_first']:.3f} -> {s['amp_second']:.3f}")
