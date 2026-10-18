"""How big is the SGS term, really? Scan over coarsening factor and time.

The a-posteriori comparison could not separate the exact-DNS oracle from the
unclosed control, which means the SGS term is not moving the dynamics. Before
changing the DNS, measure *why*: the closure competes against the ideal flux
divergence, so the honest number is

    ||RHS_with_exact_SGS - RHS_ideal|| / ||RHS_ideal||

alongside the fraction of kinetic energy that sits above the filter cutoff.
Both are computed here as a function of the coarsening factor r and time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import compute_sgs_stress, read_snapshot
from src.solver import compute_rhs, compute_sgs_viscous_fluxes

DTYPE = torch.float64


def subgrid_energy_fraction(rho, vx, vy, k_cut):
    """Fraction of resolved kinetic energy living above wavenumber k_cut.

    Uses the sqrt(rho) v weighting so the spectrum is of a quantity whose square
    integrates to kinetic energy.
    """
    w = torch.sqrt(rho.clamp(min=1e-15))
    Ny, Nx = rho.shape
    ky = torch.fft.fftfreq(Ny, d=1.0 / Ny)
    kx = torch.fft.fftfreq(Nx, d=1.0 / Nx)
    KX, KY = torch.meshgrid(kx, ky, indexing="xy")
    k_mag = torch.sqrt(KX**2 + KY**2).T

    total = 0.0
    above = 0.0
    for comp in (w * vx, w * vy):
        power = (torch.fft.fft2(comp).abs() ** 2)
        total += float(power.sum())
        above += float(power[k_mag > k_cut].sum())
    return above / max(total, 1e-300)


def sgs_force_ratio(snap, r):
    """||SGS contribution to the RHS|| / ||ideal RHS||, on the coarse grid."""
    fine = [torch.tensor(snap[k], dtype=DTYPE) for k in ("rho", "vx", "vy", "p")]
    coarse, tau = compute_sgs_stress(*fine, r=r, project_to_fluid_frame=True)

    prims = torch.stack(
        [coarse["rho_bar"], coarse["vx_tilde"], coarse["vy_tilde"], coarse["p_bar"]],
        dim=-1,
    )
    Nc_y, Nc_x = coarse["rho_bar"].shape
    dx, dy = 1.0 / Nc_x, 1.0 / Nc_y

    rhs_ideal = compute_rhs(prims, dx, dy, reconstruction="mp5", riemann_solver="hll")
    _, dFx, dFy = compute_sgs_viscous_fluxes(prims, tau)
    rhs_closed = compute_rhs(
        prims, dx, dy, sgs_viscous_Fx=dFx, sgs_viscous_Fy=dFy,
        reconstruction="mp5", riemann_solver="hll",
    )

    delta = (rhs_closed - rhs_ideal).norm()
    return float(delta / rhs_ideal.norm().clamp(min=1e-300)), Nc_x


def main():
    times = [3.0, 6.5, 8.0, 12.0, 20.0]
    factors = [4, 8, 16, 32]

    files = sorted(Path("dns_khi").glob("h*.h5"))
    snaps = {}
    for f in files:
        s = read_snapshot(f)
        for t in times:
            if abs(s["t"] - t) < 1e-6:
                snaps[t] = s

    print(f"{'t':>6} {'r':>4} {'N_c':>5} {'k_c':>6} "
          f"{'subgrid KE':>11} {'|dRHS|/|RHS|':>13}")
    print("-" * 52)
    rows = []
    for t in times:
        if t not in snaps:
            continue
        snap = snaps[t]
        rho = torch.tensor(snap["rho"], dtype=DTYPE)
        vx = torch.tensor(snap["vx"], dtype=DTYPE)
        vy = torch.tensor(snap["vy"], dtype=DTYPE)
        for r in factors:
            Nc = rho.shape[-1] // r
            k_c = Nc / 2
            frac = subgrid_energy_fraction(rho, vx, vy, k_c)
            ratio, _ = sgs_force_ratio(snap, r)
            print(f"{t:6.1f} {r:4d} {Nc:5d} {k_c:6.0f} {frac:11.4%} {ratio:13.4%}")
            rows.append((t, r, frac, ratio))
        print()

    print("\nBy coarsening factor, averaged over the scanned times:")
    print(f"{'r':>4} {'subgrid KE':>11} {'|dRHS|/|RHS|':>13}")
    for r in factors:
        sel = [(f, q) for (_, rr, f, q) in rows if rr == r]
        if sel:
            print(f"{r:4d} {np.mean([f for f, _ in sel]):11.4%} "
                  f"{np.mean([q for _, q in sel]):13.4%}")


if __name__ == "__main__":
    main()
