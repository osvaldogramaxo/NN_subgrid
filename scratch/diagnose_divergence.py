r"""
diagnose_divergence.py

Run the TBNN-closed coarse solver and capture everything needed to analyse a blow-up.

Records, for every step: scalar diagnostics (stress magnitude, clip saturation, extrema
and their locations). Keeps a rolling buffer of the last RING full states so the approach
to divergence can be replayed at full spatial resolution without storing the whole run.
On divergence it reports the first offending cells and dumps the buffer to a .pt file.

  python scratch/diagnose_divergence.py --t-start 5.0 --t-end 10.0 --sensor none
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset import load_all_snapshots, relativistic_favre_filter
from src.evaluate_rollout import TBNNSGSClosure
from src.solver import (
    RHDSolver,
    compute_dt,
    compute_sgs_viscous_fluxes,
    primitives_to_conserved,
)
from utils import CRMHD_TBNN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="test1")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    ap.add_argument("--coarsening-factor", type=int, default=4)
    ap.add_argument("--t-start", type=float, default=5.0)
    ap.add_argument("--t-end", type=float, default=10.0)
    ap.add_argument("--cfl", type=float, default=0.3)
    ap.add_argument("--reconstruction", default="mp5")
    ap.add_argument("--riemann-solver", default="hll")
    ap.add_argument("--sensor", default="none")
    ap.add_argument("--clip-kappa", type=float, default=0.5)
    ap.add_argument("--ring", type=int, default=120, help="full states retained before failure")
    ap.add_argument("--every", type=int, default=100, help="progress print interval")
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--out", default="results/divergence_diag.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    P = lambda *a: print(*a, flush=True)
    dev = torch.device(args.device)
    dtype = torch.float64

    snap0 = min(load_all_snapshots(args.data_dir), key=lambda s: abs(s["t"] - args.t_start))
    coarse = relativistic_favre_filter(
        *[torch.tensor(snap0[k], dtype=dtype) for k in ("rho", "vx", "vy", "p")],
        args.coarsening_factor,
    )
    Ny, Nx = coarse["rho_bar"].shape
    dx = dy = 1.0 / Nx

    model, ckpt = CRMHD_TBNN.from_checkpoint(args.checkpoint, device=dev, dtype=dtype)
    model.eval()

    solver = RHDSolver(
        Ny, Nx, cfl=args.cfl, reconstruction=args.reconstruction,
        riemann_solver=args.riemann_solver, dtype=dtype, device=dev,
    )
    solver.initialize(*[coarse[k].to(dev) for k in ("rho_bar", "vx_tilde", "vy_tilde", "p_bar")])
    solver.t = float(snap0["t"])

    closure = TBNNSGSClosure(
        model,
        *[ckpt[k].to(dev) for k in ("inv_mean", "inv_std", "thermo_mean", "thermo_std")],
        dx, dy, sensor_mode=args.sensor, clip_kappa=args.clip_kappa,
    )

    P(f"grid {Ny}x{Nx}  t {solver.t} -> {args.t_end}  cfl={args.cfl} "
      f"{args.reconstruction}+{args.riemann_solver}  sensor={args.sensor} kappa={args.clip_kappa}")

    ring_prims: list[torch.Tensor] = []
    ring_tau: list[torch.Tensor] = []
    ring_meta: list[dict] = []
    scalars: list[list[float]] = []

    step = 0
    wall = time.time()
    reason = "reached t_end"

    while solver.t < args.t_end - 1e-12 and step < args.max_steps:
        dt = compute_dt(solver.prims, solver.dx, solver.dy, solver.cfl)
        if not math.isfinite(dt) or dt <= 0.0:
            reason = f"invalid dt={dt}"
            break
        dt = min(dt, args.t_end - solver.t)

        tau, alpha = closure(solver.prims)
        p = solver.prims[..., 3]
        frob = torch.norm(tau, dim=(-2, -1)) / p.clamp(min=1e-30)
        dU, _, _ = compute_sgs_viscous_fluxes(solver.prims, alpha.unsqueeze(-1).unsqueeze(-1) * tau)
        U = primitives_to_conserved(*[solver.prims[..., k] for k in range(4)])
        dU_rel = (dU.abs() / U.abs().clamp(min=1e-30)).max()
        vmag = (solver.prims[..., 1] ** 2 + solver.prims[..., 2] ** 2).sqrt()

        scalars.append([
            float(solver.t), float(dt), float(frob.max()), float(frob.mean()),
            float((frob > args.clip_kappa - 1e-9).float().mean()), float(dU_rel),
            float(alpha.mean()), float(solver.prims[..., 0].min()), float(solver.prims[..., 0].max()),
            float(p.min()), float(p.max()), float(vmag.max()),
        ])

        ring_prims.append(solver.prims.detach().clone())
        ring_tau.append(tau.detach().clone())
        ring_meta.append({"step": step, "t": float(solver.t), "dt": float(dt)})
        if len(ring_prims) > args.ring:
            ring_prims.pop(0); ring_tau.pop(0); ring_meta.pop(0)

        if step % args.every == 0:
            P(f"  step {step:6d} t={solver.t:7.4f} ||tau||F/p max={frob.max():.3e} "
              f"clip%={100 * (frob > args.clip_kappa - 1e-9).float().mean():.2f} "
              f"|dU/U|max={dU_rel:.2e} rho[{solver.prims[..., 0].min():.4f},"
              f"{solver.prims[..., 0].max():.4f}] pmin={p.min():.3e} "
              f"|v|max={vmag.max():.4f} ({time.time() - wall:.0f}s)")

        solver.step(dt=dt, sgs_closure=closure)
        step += 1

        if not torch.isfinite(solver.prims).all():
            bad = ~torch.isfinite(solver.prims).all(dim=-1)
            idx = bad.nonzero()
            reason = f"non-finite in {int(bad.sum())} cell(s) at step {step}, t={solver.t:.6f}"
            P(f"\nDIVERGED: {reason}")
            P(f"  first offending cells (row, col): {idx[:8].tolist()}")
            prev = ring_prims[-1]
            for j, k in idx[:5].tolist():
                lo_r, hi_r = max(0, j - 1), min(Ny, j + 2)
                lo_c, hi_c = max(0, k - 1), min(Nx, k + 2)
                P(f"  cell ({j},{k}) state on the PREVIOUS step: "
                  f"rho={prev[j,k,0]:.4e} vx={prev[j,k,1]:+.4f} vy={prev[j,k,2]:+.4f} p={prev[j,k,3]:.4e} "
                  f"| 3x3 pmin={prev[lo_r:hi_r, lo_c:hi_c, 3].min():.3e} "
                  f"rhomin={prev[lo_r:hi_r, lo_c:hi_c, 0].min():.3e} "
                  f"||tau||F/p={float(torch.norm(ring_tau[-1][j,k])/prev[j,k,3]):.3e}")
            break

    P(f"\nstopped: {reason};  {step} steps, t={solver.t:.6f}, {time.time() - wall:.0f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "reason": reason,
            "args": vars(args),
            "scalar_names": [
                "t", "dt", "frob_max", "frob_mean", "clip_frac", "dU_rel_max",
                "alpha_mean", "rho_min", "rho_max", "p_min", "p_max", "vmag_max",
            ],
            "scalars": torch.tensor(scalars, dtype=torch.float64),
            "ring_meta": ring_meta,
            "ring_prims": torch.stack([t.cpu() for t in ring_prims]),
            "ring_tau": torch.stack([t.cpu() for t in ring_tau[-16:]]),
            "final_prims": solver.prims.detach().cpu(),
        },
        out,
    )
    P(f"wrote {out}  (scalars for all {len(scalars)} steps, last {len(ring_prims)} full states)")


if __name__ == "__main__":
    main()
