"""Reproduce the hybrid's divergence (FEASIBILITY §19) and test the clip-after-rescale fix.

Runs only the hybrid closure on the member that diverged (dns_khi, perturb seed 2,
amplitude 0.01, r=16, t=1 -> 10), logging per 0.1 time units the largest rescale
factor e_MiNNit/e_TBNN, the largest ||tau||_F / (kappa p) before any post-rescale
clip, and the share of cells over the bound.

Usage: python scratch/hybrid_divergence_probe.py {unclipped|clipped} [device] [seed]
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import read_snapshot, relativistic_favre_filter  # noqa: E402
from src.evaluate_rollout import TBNNSGSClosure, perturb_initial_state  # noqa: E402
from src.minit_learned import HybridClosure, LearnedMInITClosure  # noqa: E402
from src.solver import RHDSolver, clip_sgs_tensor  # noqa: E402
from utils import CRMHD_TBNN  # noqa: E402

mode = sys.argv[1]
dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cpu")
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 2
DT, r, KAPPA = torch.float64, 16, 0.5

snap = read_snapshot("dns_khi/h00000010.h5")
assert abs(float(snap["t"]) - 1.0) < 1e-9
c = relativistic_favre_filter(*(torch.tensor(snap[q], dtype=DT) for q in ("rho", "vx", "vy", "p")), r)
c = perturb_initial_state(c, seed, 0.01)
N = c["rho_bar"].shape[0]
solver = RHDSolver(N, N, cfl=0.75, reconstruction="mp5", riemann_solver="hllc", dtype=DT, device=dev)
solver.initialize(*(c[k].to(dev) for k in ("rho_bar", "vx_tilde", "vy_tilde", "p_bar")))
solver.t = 1.0

model, ck = CRMHD_TBNN.from_checkpoint("checkpoints_r16/best_model.pt", device=dev, dtype=DT)
model.eval()
tb = TBNNSGSClosure(model, ck["inv_mean"].to(dev), ck["inv_std"].to(dev),
                    ck["thermo_mean"].to(dev), ck["thermo_std"].to(dev), solver.dx, solver.dy)
le = LearnedMInITClosure.load("results/minit/minit_coeff_residual.pt", solver.dx, solver.dy, device=dev)
inner = HybridClosure(tb, le, clip_mode=None)          # measure before any post-clip
stats = {"scale": 0.0, "ratio": 0.0, "over": 0.0}


def closure(prims):
    tau, alpha = inner(prims)
    ratio = torch.norm(tau, dim=(-2, -1)) / (KAPPA * prims[..., 3].clamp(min=1e-15))
    stats["scale"] = max(stats["scale"], inner.max_scale)
    stats["ratio"] = max(stats["ratio"], float(ratio.max()))
    stats["over"] = max(stats["over"], float((ratio > 1).double().mean()))
    if mode == "clipped":
        tau = clip_sgs_tensor(tau, prims, mode="frobenius", kappa=KAPPA)
    return tau, alpha


print(f"mode={mode} device={dev} seed={seed}")
print(f"{'t':>6} {'max scale':>10} {'max |tau|/(k p)':>16} {'cells over':>11}")
t = 1.0
while t < 10.0 - 1e-9:
    t = round(t + 0.1, 10)
    for k in stats:
        stats[k] = 0.0
    try:
        solver.evolve(t, sgs_closure=closure, verbose=False, progress_bar=False)
        ok = bool(torch.isfinite(solver.prims).all())
    except Exception as e:                      # the solver raises on non-finite state
        ok, err = False, f"{type(e).__name__}: {str(e)[:120]}"
    print(f"{t:6.1f} {stats['scale']:10.2f} {stats['ratio']:16.2f} {100 * stats['over']:10.3f}%"
          + ("" if ok else f"   DIVERGED {err if 'err' in dir() else ''}"), flush=True)
    if not ok:
        break
print("finished" if ok else "diverged")
