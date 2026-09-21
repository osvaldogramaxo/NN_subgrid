r"""
phase0_face_closure.py

Go/no-go measurement for evaluating the SGS closure at cell faces.

The current scheme evaluates tau at cell centres and interpolates to faces with a
symmetric stencil, which annihilates the 2dx mode and attenuates 3-4dx. Evaluating
tau *at* faces, from compact face gradients, would restore that response. Before
building any of that, this script measures what the change would actually buy.

Method: compute the exact DNS subgrid stress two ways --

  centred : the standard box filter over r x r fine cells, then the 4th-order face
            interpolation the solver currently applies
  faced   : the same box filter staggered by r/2 fine cells, so it is centred on the
            cell face, differenced directly between adjacent faces

Both give a discrete SGS force -d_j tau^{ij}. Their difference is the entire payoff of
the Phase 1-3 work. If it is small relative to the force itself, stop.

  python scratch/phase0_face_closure.py --t 4.0 --r 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.dataset import compute_sgs_stress, load_all_snapshots
from src.solver import _face_interp4

torch.set_default_dtype(torch.float64)


def radial_spectrum(field: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Isotropically binned power spectrum of a real 2D scalar field."""
    f = field - field.mean()
    F = torch.fft.fft2(f)
    power = (F.abs() ** 2).cpu().numpy()
    N = field.shape[-1]
    kx = np.fft.fftfreq(N, d=1.0 / N)
    ky = np.fft.fftfreq(field.shape[-2], d=1.0 / field.shape[-2])
    K = np.sqrt(np.add.outer(ky**2, kx**2))
    k_max = N // 2
    idx = np.digitize(K.ravel(), np.arange(1, k_max + 1) - 0.5) - 1
    ok = (idx >= 0) & (idx < k_max)
    counts = np.bincount(idx[ok], minlength=k_max)[:k_max]
    total = np.bincount(idx[ok], weights=power.ravel()[ok], minlength=k_max)[:k_max]
    return np.arange(1, k_max + 1, dtype=float), total / np.maximum(counts, 1)


def sgs_stress(fields: dict, r: int, shift_x: int = 0, shift_y: int = 0) -> torch.Tensor:
    """Exact SGS stress on a filter box optionally staggered onto a cell face.

    Rolling the *fine* field by -shift before the standard box filter is exactly
    equivalent to filtering with a box displaced by +shift fine cells, so a shift of
    r//2 centres the box on the face between adjacent coarse cells.
    """
    args = []
    for key in ("rho", "vx", "vy", "p"):
        a = torch.as_tensor(fields[key])
        if shift_x:
            a = torch.roll(a, -shift_x, dims=-1)
        if shift_y:
            a = torch.roll(a, -shift_y, dims=-2)
        args.append(a)
    _, tau = compute_sgs_stress(*args, r=r, project_to_fluid_frame=True)
    return tau


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="test1")
    ap.add_argument("--t", type=float, nargs="+", default=[4.0, 5.0])
    ap.add_argument("--r", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--go-threshold", type=float, default=0.15,
                    help="Relative RMS difference below which Phase 1-3 is not worth building")
    args = ap.parse_args()

    P = lambda *a: print(*a, flush=True)
    snaps = load_all_snapshots(args.data_dir)

    P(__doc__.split("Method:")[0].strip())
    P("\n" + "=" * 78)

    verdicts = []
    for t_want in args.t:
        snap = min(snaps, key=lambda s: abs(s["t"] - t_want))
        for r in args.r:
            if r % 2:
                P(f"skipping r={r}: staggering needs an even coarsening factor")
                continue

            tau_c = sgs_stress(snap, r)                       # cell centred
            tau_fx = sgs_stress(snap, r, shift_x=r // 2)      # x-faces, index i == i+1/2
            tau_fy = sgs_stress(snap, r, shift_y=r // 2)      # y-faces

            N = tau_c.shape[0]
            dx = dy = 1.0 / N

            # x-momentum SGS force, -d_x tau^{11} - d_y tau^{12}, both ways.
            fx_c = _face_interp4(tau_c[..., 1, 1], dim=-1)
            fy_c = _face_interp4(tau_c[..., 1, 2], dim=-2)
            force_centred = -(
                (fx_c - torch.roll(fx_c, 1, dims=-1)) / dx
                + (fy_c - torch.roll(fy_c, 1, dims=-2)) / dy
            )

            fx_f = tau_fx[..., 1, 1]
            fy_f = tau_fy[..., 1, 2]
            force_faced = -(
                (fx_f - torch.roll(fx_f, 1, dims=-1)) / dx
                + (fy_f - torch.roll(fy_f, 1, dims=-2)) / dy
            )

            diff = force_faced - force_centred
            rms = lambda a: float(a.pow(2).mean().sqrt())
            rel = rms(diff) / max(rms(force_centred), 1e-30)
            verdicts.append((snap["t"], r, rel))

            P(f"\n--- t={snap['t']:.1f}, r={r}  ({N}x{N} coarse) ---")
            P(f"  RMS SGS force, centred operator : {rms(force_centred):.4e}")
            P(f"  RMS SGS force, face operator    : {rms(force_faced):.4e}")
            P(f"  RMS of the difference           : {rms(diff):.4e}")
            P(f"  relative difference             : {rel:.1%}")

            k, E_c = radial_spectrum(force_centred)
            _, E_f = radial_spectrum(force_faced)
            _, E_d = radial_spectrum(diff)
            P(f"  {'k':>5} {'wavelength':>11} {'centred':>11} {'faced':>11} {'diff/centred':>13}")
            for i in (0, len(k) // 8, len(k) // 4, len(k) // 2, 3 * len(k) // 4, len(k) - 1):
                lam = 2.0 * N / max(k[i], 1e-30)
                ratio = np.sqrt(E_d[i] / max(E_c[i], 1e-300))
                P(f"  {k[i]:5.0f} {lam:10.1f}dx {E_c[i]:11.3e} {E_f[i]:11.3e} {ratio:13.2f}")

    P("\n" + "=" * 78)
    P("VERDICT (relative RMS difference; the payoff of building Phase 1-3)")
    for t, r, rel in verdicts:
        mark = "BUILD IT" if rel >= args.go_threshold else "not worth it"
        P(f"  t={t:5.1f}  r={r:3d}   {rel:7.1%}   {mark}")
    best = max(verdicts, key=lambda v: v[2]) if verdicts else None
    if best:
        P(f"\nLargest payoff: {best[2]:.1%} at t={best[0]:.1f}, r={best[1]}.")
        P("Gate is on the regime you actually intend to run in, not the best case.")


if __name__ == "__main__":
    main()
