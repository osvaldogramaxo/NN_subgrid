r"""
evaluate_shorthorizon.py

Short-horizon a-posteriori test: can a closure predict the *increment* of the
filtered DNS over one snapshot interval?

Why this exists
---------------
The long rollout in :mod:`src.evaluate_rollout` could not separate the exact-DNS
oracle from the unclosed control, so it cannot rank anything. Two things are
conflated there:

  * over ~400 steps a chaotic trajectory decorrelates, and pointwise norms then
    measure phase error rather than closure quality;
  * the SGS term is a ~6% perturbation to the RHS, against a discretisation
    error that costs ~66% of the high-k energy.

The first is avoidable. Starting every closure from the *same* filtered-DNS
state and integrating for a single snapshot interval (~27 steps) keeps the runs
correlated, so a 6% difference in the right-hand side is directly visible. This
is the standard bridge between the a-priori test — where the closures differ
enormously — and the long rollout, where they do not.

The metric
----------
Over a short interval the state barely moves, so normalising the error by the
state would hide everything. The closure's job is to get the *change* right:

.. math::
    \varepsilon = \frac{\lVert \Delta U_\text{model} - \Delta U_\text{DNS}\rVert}
                       {\lVert \Delta U_\text{DNS}\rVert},
    \qquad \Delta U = U(t + \Delta t) - U_\text{DNS}(t)

so ``0`` is a perfect increment and ``1`` is the score of predicting no change
at all. Averaged over many independent start times, which turns a single chaotic
realisation into a statistic with a spread.

**Read the oracle row first.** It is the ceiling: it is what the exact SGS stress
buys at this horizon. If the oracle does not beat the unclosed control here
either, the configuration cannot rank closures at any horizon and the problem is
the discretisation error, not the metric.

Usage
-----
    # TBNN test split (its training window ends at t = 6.4)
    python -m src.evaluate_shorthorizon --t-lo 6.5 --t-hi 8.0

    # Late window: 3x the subgrid content, but out of distribution for the TBNN
    python -m src.evaluate_shorthorizon --t-lo 14.0 --t-hi 20.0 --skip-tbnn
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.dataset import (
    compute_sgs_stress, read_snapshot, relativistic_favre_filter,
)
from src.evaluate_rollout import ExactDNSSGSClosure, TBNNSGSClosure
from src.minit import MInITClosure, MInITCoefficients, subgrid_energy_from_stress
from src.minit_learned import HybridClosure, LearnedMInITClosure
from src.solver import RHDSolver, primitives_to_conserved

try:  # optional: the TBNN comparison is skipped when the model is unavailable
    from utils import CRMHD_TBNN
except Exception:  # pragma: no cover - exercised only without the model package
    CRMHD_TBNN = None


def increment_error(
    U_model: torch.Tensor, U_dns_end: torch.Tensor, U_start: torch.Tensor,
) -> float:
    r"""``||dU_model - dU_DNS|| / ||dU_DNS||`` with both increments from the same start.

    Normalising by the increment rather than the state is what gives this metric
    dynamic range over a short horizon: 0 is an exact increment, and 1 is what a
    solver scores by not moving at all, so values near 1 mean the closure has
    contributed nothing and values above 1 mean it made the step worse than
    standing still.
    """
    d_model = U_model - U_start
    d_dns = U_dns_end - U_start
    denom = d_dns.norm().clamp(min=1e-300)
    return float((d_model - d_dns).norm() / denom)


def _coarse_prims(snap, r, dtype=torch.float64):
    fields = [torch.tensor(snap[k], dtype=dtype) for k in ("rho", "vx", "vy", "p")]
    c = relativistic_favre_filter(*fields, r)
    return torch.stack(
        [c["rho_bar"], c["vx_tilde"], c["vy_tilde"], c["p_bar"]], dim=-1
    )


def _conserved(prims: torch.Tensor) -> torch.Tensor:
    return primitives_to_conserved(
        prims[..., 0], prims[..., 1], prims[..., 2], prims[..., 3]
    )


def run_short_horizon(
    data_dir: str,
    r: int,
    t_lo: float,
    t_hi: float,
    horizon_intervals: int = 1,
    cfl: float = 0.4,
    reconstruction: str = "mp5",
    riemann_solver: str = "hllc",
    checkpoint: str = "checkpoints/best_model.pt",
    minit_coefficients: str = "results/minit/minit_coefficients.json",
    minit_variants: tuple[str, ...] = ("fitted", "equilibrium", "paper"),
    learned_coeff: tuple[str, ...] = (),
    hybrid_coeff: str | None = None,
    skip_tbnn: bool = False,
    skip_exact: bool = False,
    device_str: str = "cpu",
    max_starts: int | None = None,
) -> dict:
    """Run the short-horizon test over every start time in ``[t_lo, t_hi]``."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    files = sorted(glob.glob(os.path.join(data_dir, "h*.h5")))
    snaps = [read_snapshot(f) for f in files]
    snaps.sort(key=lambda s: s["t"])
    window = [s for s in snaps if t_lo - 1e-9 <= s["t"] <= t_hi + 1e-9]
    if len(window) < horizon_intervals + 1:
        raise ValueError(
            f"window [{t_lo}, {t_hi}] holds {len(window)} snapshots; "
            f"need at least {horizon_intervals + 1}"
        )

    starts = list(range(len(window) - horizon_intervals))
    if max_starts is not None:
        starts = starts[:max_starts]

    Ny_f, Nx_f = window[0]["rho"].shape
    Nc_y, Nc_x = Ny_f // r, Nx_f // r
    dx, dy = 1.0 / Nc_x, 1.0 / Nc_y

    # --- closures that are constructed once -------------------------------
    tbnn_ckpt = None
    if not skip_tbnn and CRMHD_TBNN is not None and os.path.exists(checkpoint):
        model, tbnn_ckpt = CRMHD_TBNN.from_checkpoint(checkpoint, device=device, dtype=dtype)
        model.eval()
    else:
        model = None

    coeff_sets: dict[str, tuple[MInITCoefficients, str]] = {}
    for variant in minit_variants:
        if variant == "paper":
            coeff_sets["paper"] = (MInITCoefficients.paper(), "evolve")
            continue
        path = minit_coefficients
        if variant == "equilibrium":
            path = str(Path(path).with_name(Path(path).stem + "_equilibrium.json"))
        if not os.path.exists(path):
            print(f"[SKIP] MInIT '{variant}': no coefficients at {path}")
            continue
        coeff_sets[variant] = (
            MInITCoefficients.load(path),
            "equilibrium" if variant == "equilibrium" else "evolve",
        )

    # Learned-coefficient MInIT (src.minit_learned). Stateless and memoryless, so
    # unlike MInITClosure it takes no e_KH seed -- it is handed strictly less
    # ground truth than the MInIT rows above, which are seeded from the exact
    # filtered DNS at t0. Read the comparison with that asymmetry in mind.
    learned_closures: dict[str, str] = {}
    for path in learned_coeff:
        if not os.path.exists(path):
            print(f"[SKIP] learned coefficients: no checkpoint at {path}")
            continue
        learned_closures[f"learned_{Path(path).stem.replace('minit_coeff_', '')}"] = path

    labels = ["unclosed"] + [f"minit_{v}" for v in coeff_sets] + list(learned_closures)
    if model is not None:
        labels.append("tbnn")
        if hybrid_coeff:
            labels.append("hybrid")
    if not skip_exact:
        labels.append("exact")

    errors: dict[str, list[float]] = {k: [] for k in labels}
    # "Frozen" is the trivial reference: predict no change at all. Any run scoring
    # near 1.0 has matched it, whatever else the number looks like.
    errors["frozen"] = []
    start_times: list[float] = []

    pbar = tqdm(starts, desc="Short-horizon starts", unit="start",
                dynamic_ncols=True, file=sys.stdout)
    for i in pbar:
        snap0 = window[i]
        snap1 = window[i + horizon_intervals]
        t0, t1 = float(snap0["t"]), float(snap1["t"])
        start_times.append(t0)

        prims0 = _coarse_prims(snap0, r, dtype)
        prims1 = _coarse_prims(snap1, r, dtype)
        U_start = _conserved(prims0)
        U_end = _conserved(prims1)

        errors["frozen"].append(increment_error(U_start, U_end, U_start))

        def _fresh() -> RHDSolver:
            s = RHDSolver(Nc_y, Nc_x, cfl=cfl, reconstruction=reconstruction,
                          riemann_solver=riemann_solver, dtype=dtype, device=device)
            s.initialize(*(prims0[..., j].to(device) for j in range(4)))
            s.t = t0
            return s

        def _score(solver: RHDSolver) -> float:
            p = solver.get_primitives()
            U = _conserved(torch.stack(
                [p["rho"].cpu(), p["vx"].cpu(), p["vy"].cpu(), p["p"].cpu()], dim=-1
            ))
            return increment_error(U, U_end, U_start)

        quiet = dict(verbose=False, progress_bar=False)

        # 1. Unclosed control
        s = _fresh()
        s.evolve(t1, **quiet)
        errors["unclosed"].append(_score(s))

        # 2. MInIT variants, each seeded from the exact filtered-DNS e_KH at t0 —
        #    the same single snapshot of ground truth the oracle is handed.
        if coeff_sets:
            fine0 = [torch.tensor(snap0[k], dtype=dtype) for k in ("rho", "vx", "vy", "p")]
            _, tau0 = compute_sgs_stress(*fine0, r=r, project_to_fluid_frame=True)
            e_seed = subgrid_energy_from_stress(tau0).to(device)
        for variant, (coeffs, mode) in coeff_sets.items():
            s = _fresh()
            closure = MInITClosure(dx=dx, dy=dy, delta_f=dx, coeffs=coeffs, mode=mode)
            closure.initialize(s.prims, e_KH=e_seed)
            s.evolve(t1, sgs_closure=closure, **quiet)
            errors[f"minit_{variant}"].append(_score(s))

        # 2b. Learned-coefficient MInIT -- no seed, it reads only resolved state.
        for label, path in learned_closures.items():
            s_ = _fresh()
            closure = LearnedMInITClosure.load(path, dx, dy, device=device)
            s_.evolve(t1, sgs_closure=closure, **quiet)
            errors[label].append(_score(s_))

        # 3. TBNN
        if model is not None:
            s = _fresh()
            closure = TBNNSGSClosure(
                model,
                tbnn_ckpt["inv_mean"].to(device), tbnn_ckpt["inv_std"].to(device),
                tbnn_ckpt["thermo_mean"].to(device), tbnn_ckpt["thermo_std"].to(device),
                dx, dy,
            )
            s.evolve(t1, sgs_closure=closure, **quiet)
            errors["tbnn"].append(_score(s))

            if hybrid_coeff:   # TBNN shape x MiNNit amplitude, no seed
                s = _fresh()
                hybrid = HybridClosure(
                    closure, LearnedMInITClosure.load(hybrid_coeff, dx, dy, device=device))
                s.evolve(t1, sgs_closure=hybrid, **quiet)
                errors["hybrid"].append(_score(s))

        # 4. Exact-DNS oracle — the ceiling on what any closure can buy here.
        if not skip_exact:
            s = _fresh()
            closure = ExactDNSSGSClosure(
                snapshots=[snap0, snap1], coarsening_factor=r,
                dx=dx, dy=dy, device=device, dtype=dtype,
            )
            s.evolve(t1, sgs_closure=closure, **quiet)
            errors["exact"].append(_score(s))

        pbar.set_postfix(t=f"{t0:.1f}")

    return {
        "window": {"t_lo": t_lo, "t_hi": t_hi, "n_starts": len(starts)},
        "horizon": horizon_intervals * (float(window[1]["t"]) - float(window[0]["t"])),
        "grid": {"Nc_y": Nc_y, "Nc_x": Nc_x, "r": r},
        "start_times": start_times,
        "errors": errors,
    }


def report(res: dict) -> None:
    """Print the ranking, with the unclosed control as the reference."""
    errors = res["errors"]
    unclosed = np.array(errors["unclosed"])
    n = len(unclosed)

    print(f"\n=== Short-horizon increment error, {n} starts, "
          f"horizon {res['horizon']:.2f} sim time ===")
    print("  epsilon = ||dU_model - dU_DNS|| / ||dU_DNS||   (0 = perfect, 1 = no change)")
    print(f"\n  {'run':22s} {'mean':>8s} {'median':>8s} {'std':>8s} "
          f"{'skill':>9s} {'wins':>8s}")
    print("  " + "-" * 70)

    order = ["frozen", "unclosed"]
    order += sorted(k for k in errors if k.startswith("minit_"))
    order += sorted(k for k in errors if k.startswith("learned_"))
    order += [k for k in ("tbnn", "hybrid", "exact") if k in errors]
    # Anything a caller added that the list above does not know about still gets
    # printed: a row silently dropped from the table but present in the JSON is
    # worse than an ugly table.
    order += [k for k in errors if k not in order]

    for key in order:
        if not errors.get(key):
            continue
        e = np.array(errors[key])
        # Paired skill: the fractional error reduction on each start, then
        # averaged. Pairing matters because the starts differ a lot from each
        # other and an unpaired mean would hide a consistent small gain.
        skill = float(np.mean(1.0 - e / np.maximum(unclosed, 1e-300)))
        # The spread across start times is far larger than the gaps between runs,
        # so the mean alone cannot say whether a closure helps. The paired win
        # count can: a run that beats the control on nearly every start is doing
        # something real, however small the margin.
        wins = int(np.sum(e < unclosed))
        tag = "" if key != "exact" else "   <- ceiling"
        print(f"  {key:22s} {e.mean():8.4f} {np.median(e):8.4f} {e.std():8.4f} "
              f"{skill:+8.2%} {wins:5d}/{n}{tag}")

    if "exact" in errors and errors["exact"]:
        ex = np.array(errors["exact"])
        gain = float(np.mean(1.0 - ex / np.maximum(unclosed, 1e-300)))
        wins = int(np.sum(ex < unclosed))
        print(f"\n  Oracle beats unclosed on {wins}/{n} starts, "
              f"mean error reduction {gain:+.2%}.")
        print(f"  That is the ceiling: no closure can do better than supplying the "
              f"true stress,\n  so read every row above as a fraction of "
              f"{gain:+.2%}, not as an absolute gain.")
        if gain > 0.01:
            print("  The oracle separates, so this horizon CAN rank closures; read the "
                  "rows above\n  against it as the ceiling.")
        elif gain < -0.01:
            print("  The exact SGS stress makes the step WORSE than not closing at all.")
            print("  That is not a statement about any model: it says the coarse "
                  "discretisation's\n  own error is larger than, and not aligned with, "
                  "the term being added, so\n  supplying the true stress cannot help. "
                  "Nothing below the oracle can be\n  ranked, and refitting a closure "
                  "cannot fix it.")
        else:
            print("  The exact SGS stress buys under 1% even at this horizon: the "
                  "configuration\n  cannot rank closures, and the limit is the "
                  "discretisation error, not the metric.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Short-horizon a-posteriori closure test")
    ap.add_argument("--data-dir", default="dns_khi")
    ap.add_argument("--coarsening-factor", "-r", type=int, default=16)
    ap.add_argument("--t-lo", type=float, default=6.5)
    ap.add_argument("--t-hi", type=float, default=8.0)
    ap.add_argument("--horizon-intervals", type=int, default=1,
                    help="snapshot intervals per start (1 = ~27 solver steps)")
    ap.add_argument("--max-starts", type=int, default=None)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--reconstruction", default="mp5", choices=["mp5", "muscl"])
    ap.add_argument("--riemann-solver", default="hllc", choices=["hll", "hllc", "musta_hll"])
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    ap.add_argument("--minit-coefficients", default="results/minit/minit_coefficients.json")
    ap.add_argument("--with-learned", action="append", default=[],
                    metavar="CKPT",
                    help="learned-coefficient MInIT checkpoint from "
                         "src.train_minit_coeff; repeatable")
    ap.add_argument("--with-hybrid", default=None, metavar="CKPT",
                    help="MiNNit checkpoint whose energy rescales the TBNN's tensor")
    ap.add_argument("--skip-tbnn", action="store_true")
    ap.add_argument("--skip-exact", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default="results/minit/short_horizon.json")
    args = ap.parse_args(argv)

    res = run_short_horizon(
        data_dir=args.data_dir, r=args.coarsening_factor,
        t_lo=args.t_lo, t_hi=args.t_hi,
        horizon_intervals=args.horizon_intervals,
        cfl=args.cfl, reconstruction=args.reconstruction,
        riemann_solver=args.riemann_solver,
        checkpoint=args.checkpoint, minit_coefficients=args.minit_coefficients,
        learned_coeff=tuple(args.with_learned),
        hybrid_coeff=args.with_hybrid,
        skip_tbnn=args.skip_tbnn, skip_exact=args.skip_exact,
        device_str=args.device, max_starts=args.max_starts,
    )
    report(res)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
