r"""
generate_dns.py

Generate relativistic Kelvin–Helmholtz DNS data for SGS closure training.

Writes HDF5 snapshots readable by `src.dataset.read_snapshot` without any
special-casing: each file carries a `prim` group holding [ρ, v_x, v_y, p]
directly, alongside `hydro`/`thd` in the Aenus layout for inspection tools.

Why `prim` exists
-----------------
The Aenus files in `test1/` store *Newtonian* conserved variables, and
`read_snapshot` inverts them with v = S/ρ. Relativistic conserved variables
are D = ρW and S_i = ρhW²v_i, for which that inversion is wrong. Rather than
overload one slot with two meanings — the exact class of silent convention
mismatch that invalidated the first round of this project — the primitives are
stored explicitly and `hydro` holds the physically correct relativistic
conserved state (D, E, S_x, S_y, S_z) with E = τ + D.

Every file also carries a `Metadata` group recording Γ, the variable layout,
the full IC parameter set and the git commit, so a future reader never has to
infer the EOS from the data.

Initial condition
-----------------
Double tanh shear layer in a periodic unit box (the Lecoanet et al. KHI
geometry), with two deliberate departures made for turbulence rather than
code validation:

  * The seed is **broadband** (a band of modes at random phase) instead of a
    single k=2 mode. A single mode rolls up into one vortex pair per layer,
    merges once, and produces no cascade.
  * The seed is **solenoidal**, derived from a streamfunction ψ so that
    ∇·δv = 0 exactly and no acoustic transient is launched into a box that
    started in pressure equilibrium.

Usage
-----
    # Cheap: print IC diagnostics and exit, to tune v_shear / thickness
    python -m src.generate_dns --ic-only

    # Full run
    python -m src.generate_dns --output-dir dns_khi_s0 --seed 0 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solver import RHDSolver, set_gamma_eos  # noqa: E402
from src.config import apply_config_defaults  # noqa: E402

GHOST = 3  # matches src.dataset.GHOST


# ---------------------------------------------------------------------------
# Equation of state helpers (ideal gas, Γ-law, special relativistic)
# ---------------------------------------------------------------------------

def specific_enthalpy(rho: np.ndarray, p: np.ndarray, gamma: float) -> np.ndarray:
    r"""h = 1 + Γ/(Γ-1) · p/ρ  (includes rest mass, so h → 1 as p → 0)."""
    return 1.0 + gamma / (gamma - 1.0) * p / rho


def sound_speed(rho: np.ndarray, p: np.ndarray, gamma: float) -> np.ndarray:
    r"""Relativistic sound speed c_s = sqrt(Γ p / (ρ h)); bounded by sqrt(Γ-1)."""
    h = specific_enthalpy(rho, p, gamma)
    return np.sqrt(gamma * p / (rho * h))


def shear_mach(v_shear: float, cs: float) -> tuple[float, float]:
    r"""Relativistic relative velocity across the layer, and its Mach number.

    Velocities do NOT add linearly: the relative speed between the two streams
    is dv = 2v/(1+v²), not 2v. And the relativistic Mach number carries Lorentz
    factors on both the flow and the sound speed,

        M = (W_dv dv) / (W_cs c_s),    W_x = 1/sqrt(1 - x²)

    which diverges as dv -> 1 even though dv itself saturates. Using the
    Newtonian 2v/c_s badly understates how supersonic a high-W shear layer is:
    at v_shear = 0.8 it reads 2.91 where the true value is 6.74, and KHI is
    strongly suppressed well before that.
    """
    dv = 2.0 * v_shear / (1.0 + v_shear**2)
    W_dv = 1.0 / math.sqrt(max(1.0 - dv * dv, 1e-30))
    W_cs = 1.0 / math.sqrt(max(1.0 - cs * cs, 1e-30))
    return dv, (W_dv * dv) / (W_cs * cs)


def unstable_kmax(thickness: float) -> float:
    r"""Largest box wavenumber the KHI can amplify for a tanh layer.

    A tanh shear layer of half-thickness a is unstable only for k_phys·a < 1;
    shorter wavelengths cannot feel the two streams separately. With
    k_phys = 2πk on a unit box that is k < 1/(2πa).
    """
    return 1.0 / (2.0 * math.pi * thickness)


def lorentz(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    return 1.0 / np.sqrt(np.maximum(1.0 - (vx**2 + vy**2), 1e-14))


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------

def khi_initial_conditions(
    Nx: int,
    Ny: int,
    *,
    v_shear: float,
    thickness: float,
    rho1: float,
    rho2: float,
    pressure: float,
    seed_amplitude: float,
    seed_kmin: int,
    seed_kmax: int,
    seed_envelope_width: float | None,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Broadband-seeded double shear layer on a periodic unit box.

    Base state (y₁ = 1/4, y₂ = 3/4, layer half-thickness a):

        f(y) = ½ [tanh((y-y₁)/a) - tanh((y-y₂)/a)]     0 outside, 1 between
        ρ(y) = ρ₁ + (ρ₂-ρ₁) f(y)
        vx(y) = v_shear (1 - 2 f(y))                   +v outside, -v between
        p     = const                                  (pressure equilibrium)

    Perturbation from a streamfunction, so it is divergence-free by
    construction:

        ψ(x,y) = Σ_k (1/k) g(y) cos(2πkx + φ_k),   g = Gaussian on each layer
        δvx = ∂ψ/∂y,   δvy = -∂ψ/∂x

    The 1/k weighting makes δvy amplitude flat in k, exciting the whole seeded
    band roughly equally rather than concentrating the energy at k_min.

    Returns
    -------
    rho, vx, vy, p : (Ny, Nx) arrays
    x, y           : (Nx,), (Ny,) cell-centre coordinates
    """
    if seed_kmax < seed_kmin:
        raise ValueError(f"seed_kmax ({seed_kmax}) < seed_kmin ({seed_kmin})")
    if seed_kmax > Nx // 4:
        raise ValueError(
            f"seed_kmax ({seed_kmax}) exceeds Nx/4 ({Nx // 4}); seeded modes would be "
            "too close to the grid scale to roll up before the scheme dissipates them"
        )

    x = (np.arange(Nx) + 0.5) / Nx
    y = (np.arange(Ny) + 0.5) / Ny
    X, Y = np.meshgrid(x, y, indexing="xy")  # (Ny, Nx)

    y1, y2 = 0.25, 0.75
    a = thickness

    f = 0.5 * (np.tanh((Y - y1) / a) - np.tanh((Y - y2) / a))
    rho = rho1 + (rho2 - rho1) * f
    vx = v_shear * (1.0 - 2.0 * f)
    p = np.full_like(rho, pressure)

    # --- solenoidal broadband seed -----------------------------------------
    sigma = seed_envelope_width if seed_envelope_width is not None else 2.0 * a
    g = np.exp(-((Y - y1) ** 2) / (2.0 * sigma**2)) + np.exp(
        -((Y - y2) ** 2) / (2.0 * sigma**2)
    )
    dg_dy = (
        -(Y - y1) / sigma**2 * np.exp(-((Y - y1) ** 2) / (2.0 * sigma**2))
        - (Y - y2) / sigma**2 * np.exp(-((Y - y2) ** 2) / (2.0 * sigma**2))
    )

    rng = np.random.default_rng(rng_seed)
    dvx = np.zeros_like(rho)
    dvy = np.zeros_like(rho)
    for k in range(seed_kmin, seed_kmax + 1):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        c_k = 1.0 / k
        arg = 2.0 * np.pi * k * X + phase
        dvx += c_k * dg_dy * np.cos(arg)
        dvy += c_k * (2.0 * np.pi * k) * g * np.sin(arg)

    peak = float(np.sqrt(dvx**2 + dvy**2).max())
    if peak > 0.0:
        scale = seed_amplitude / peak
        dvx *= scale
        dvy *= scale

    vx = vx + dvx
    vy = dvy

    # The base state is already at |v| = v_shear; the seed can push a cell over
    # the light cone if the user asks for a large amplitude at a large v_shear.
    v_sq = vx**2 + vy**2
    if v_sq.max() >= 1.0:
        raise ValueError(
            f"initial condition is superluminal: max |v| = {np.sqrt(v_sq.max()):.6f}. "
            f"Reduce --v-shear ({v_shear}) or --seed-amplitude ({seed_amplitude})."
        )

    return rho, vx, vy, p, x, y


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def ke_spectrum(rho: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotropically binned kinetic energy spectrum E(k), k in integer bins."""
    Ny, Nx = rho.shape
    sx_hat = np.fft.fft2(rho * vx)
    sy_hat = np.fft.fft2(rho * vy)
    E = 0.5 * (np.abs(sx_hat) ** 2 + np.abs(sy_hat) ** 2) / (rho.mean() * Nx * Ny)

    kx = np.fft.fftfreq(Nx, d=1.0 / Nx)
    ky = np.fft.fftfreq(Ny, d=1.0 / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    k_int = np.rint(np.sqrt(KX**2 + KY**2)).astype(int).ravel()

    k_max = min(Nx, Ny) // 2
    E_k = np.bincount(k_int, weights=E.ravel(), minlength=k_max + 1)[: k_max + 1]
    return np.arange(1, k_max + 1), E_k[1:]


def spectral_diagnostics(
    k: np.ndarray, E_k: np.ndarray, r: int, N: int, fit_lo: int, fit_hi: int,
    fluct_kmin: int = 4,
) -> dict[str, float]:
    r"""Inertial-range slope, dissipation wavenumber and subgrid energy fraction.

    `k_diss` is where the measured spectrum first falls to half the value
    extrapolated from the fitted power law — i.e. where numerical dissipation
    takes over. For the DNS to be a usable reference at coarsening r, the
    filter cutoff k_c = N/(2r) must sit well below it; the rule of thumb used
    here is k_diss / k_c >= 4.
    """
    out: dict[str, float] = {}
    k_c = N / (2.0 * r)
    out["k_c"] = k_c
    total = float(E_k.sum())
    out["ke_spectral_total"] = total
    above = float(E_k[k > k_c].sum())
    out["subgrid_fraction"] = above / total if total > 0 else float("nan")

    # In a shear flow the mean flow dominates the budget -- measured at 512²,
    # 86% of the kinetic energy sits at k=1, which is the layer itself, not
    # turbulence. Dividing the subgrid energy by that makes a perfectly healthy
    # cascade look negligible (0.5% of total, 8% of the fluctuations). A closure
    # only has to model the fluctuating part, so that is the honest denominator.
    fluct = float(E_k[k >= fluct_kmin].sum())
    out["fluct_kmin"] = float(fluct_kmin)
    out["subgrid_fraction_fluct"] = above / fluct if fluct > 0 else float("nan")

    band = (k >= fit_lo) & (k <= fit_hi) & (E_k > 1e-30 * total)
    if band.sum() < 4:
        out["slope"] = float("nan")
        out["k_diss"] = float("nan")
        out["k_diss_over_k_c"] = float("nan")
        return out

    slope, intercept = np.polyfit(np.log(k[band]), np.log(E_k[band]), 1)
    out["slope"] = float(slope)

    fit = np.exp(intercept) * k.astype(float) ** slope
    beyond = k > fit_hi
    below = beyond & (E_k < 0.5 * fit)
    out["k_diss"] = float(k[below][0] - 1) if below.any() else float(k[-1])
    out["k_diss_over_k_c"] = out["k_diss"] / k_c if k_c > 0 else float("nan")
    return out


def state_diagnostics(
    rho: np.ndarray, vx: np.ndarray, vy: np.ndarray, p: np.ndarray, gamma: float, dx: float, dy: float
) -> dict[str, float]:
    W = lorentz(vx, vy)
    cs = sound_speed(rho, p, gamma)
    v = np.sqrt(vx**2 + vy**2)
    # Periodic central differences; np.gradient would be wrong at the wrap.
    omega = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2.0 * dx) - (
        np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)
    ) / (2.0 * dy)
    return {
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "p_min": float(p.min()),
        "p_max": float(p.max()),
        "v_max": float(v.max()),
        "W_max": float(W.max()),
        "W_mean": float(W.mean()),
        "cs_min": float(cs.min()),
        "cs_max": float(cs.max()),
        "mach_max": float((v / cs).max()),
        "enstrophy": float(0.5 * (omega**2).mean()),
        "ke": float((0.5 * rho * W**2 * v**2).mean()),
    }


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

def _pad_periodic(a: np.ndarray) -> np.ndarray:
    """Wrap GHOST cells on each side, so [3:-3] recovers the physical domain."""
    return np.pad(a, GHOST, mode="wrap")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_snapshot(
    path: Path,
    rho: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    p: np.ndarray,
    t: float,
    x: np.ndarray,
    y: np.ndarray,
    gamma: float,
    meta: dict,
) -> None:
    """Write one snapshot in the Aenus-compatible layout plus a `prim` group."""
    W = lorentz(vx, vy)
    h = specific_enthalpy(rho, p, gamma)
    rho_h_W2 = rho * h * W**2

    D = rho * W                    # relativistic mass density
    Sx = rho_h_W2 * vx             # momentum density
    Sy = rho_h_W2 * vy
    Sz = np.zeros_like(D)
    E = rho_h_W2 - p               # total energy INCLUDING rest mass; τ = E - D
    cs = sound_speed(rho, p, gamma)

    hydro = np.stack([_pad_periodic(f) for f in (D, E, Sx, Sy, Sz)], axis=-1)[None]
    thd = np.stack([_pad_periodic(f) for f in (p, cs)], axis=-1)[None]
    prim = np.stack([_pad_periodic(f) for f in (rho, vx, vy, p)], axis=-1)[None]

    xg = np.concatenate([x[:GHOST] - 1.0, x, x[-GHOST:] + 1.0])
    yg = np.concatenate([y[:GHOST] - 1.0, y, y[-GHOST:] + 1.0])

    with h5py.File(path, "w") as f:
        f.create_dataset("hydro/data", data=hydro, compression="gzip", compression_opts=4)
        f.create_dataset("thd/data", data=thd, compression="gzip", compression_opts=4)
        f.create_dataset("prim/data", data=prim, compression="gzip", compression_opts=4)
        f.create_dataset("thd/variables", data=np.array([b"p", b"cs"]))
        f.create_dataset("Parameters/t", data=np.array([t]))
        f.create_dataset("X/znc", data=xg)
        f.create_dataset("Y/znc", data=yg)
        f.create_dataset("Z/znc", data=np.array([0.0]))

        g = f.create_group("Metadata")
        g.attrs["scheme"] = "SRHD"
        g.attrs["eos"] = "ideal-gamma-law"
        g.attrs["eos_gamma"] = float(gamma)
        g.attrs["ghost"] = GHOST
        g.attrs["boundary"] = "periodic"
        g.attrs["hydro_variables"] = "D=rho*W, E=rho*h*W^2-p (incl. rest mass), Sx, Sy, Sz=rho*h*W^2*v"
        g.attrs["prim_variables"] = "rho, vx, vy, p"
        g.attrs["units"] = "c=1, code units"
        g.attrs["generator"] = "src/generate_dns.py"
        g.attrs["git_commit"] = _git_commit()
        g.attrs["parameters_json"] = json.dumps(meta, sort_keys=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate relativistic KHI DNS snapshots for SGS closure training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = ap.add_argument_group("grid")
    g.add_argument("--nx", type=int, default=1024)
    g.add_argument("--ny", type=int, default=1024)

    g = ap.add_argument_group("initial condition")
    g.add_argument("--v-shear", type=float, default=0.8,
                   help="shear velocity; W=1.67 at 0.8, 2.29 at 0.9. Higher W is more "
                        "relativistic but also more supersonic, which slows KHI growth")
    g.add_argument("--thickness", type=float, default=0.02,
                   help="shear layer half-thickness a, in box units")
    g.add_argument("--rho1", type=float, default=1.0, help="density outside the layers")
    g.add_argument("--rho2", type=float, default=2.0, help="density between the layers")
    g.add_argument("--pressure", type=float, default=2.5)
    g.add_argument("--gamma", type=float, default=4.0 / 3.0, help="adiabatic index")
    g.add_argument("--seed-amplitude", type=float, default=0.01)
    g.add_argument("--seed-kmin", type=int, default=4)
    g.add_argument("--seed-kmax", type=int, default=32)
    g.add_argument("--seed-envelope-width", type=float, default=None,
                   help="Gaussian width of the seed envelope about each layer "
                        "(default: 2 x --thickness)")
    g.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the mode phases; vary it for ensemble members")

    g = ap.add_argument_group("evolution")
    g.add_argument("--t-final", type=float, default=3.0)
    g.add_argument("--n-outputs", type=int, default=60,
                   help="number of snapshots after t=0")
    g.add_argument("--cfl", type=float, default=0.4)
    g.add_argument("--reconstruction", default="mp5", choices=["mp5", "muscl"])
    g.add_argument("--riemann", default="hllc", choices=["hll", "hllc", "musta_hll"])
    g.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    g.add_argument("--dtype", default="float64", choices=["float64", "float32"])

    g = ap.add_argument_group("output and diagnostics")
    g.add_argument("--output-dir", default="dns_khi")
    g.add_argument("--report-r", type=int, default=16,
                   help="coarsening factor used for the subgrid-fraction and "
                        "scale-separation diagnostics")
    g.add_argument("--fit-lo", type=int, default=8, help="low k for the spectral slope fit")
    g.add_argument("--fit-hi", type=int, default=0,
                   help="high k for the slope fit (0 = Nx/8)")
    g.add_argument("--ic-only", action="store_true",
                   help="report initial-condition diagnostics and exit without evolving")
    g.add_argument("--no-progress", action="store_true")
    g.add_argument("--config", default=None, metavar="PATH",
                   help="TOML config supplying defaults (see src/config.py)")
    return ap


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


#: config keys that seed this script's argparse defaults
CONFIG_DEFAULTS = {
    "gamma": "physics.gamma",
    "cfl": "solver.cfl",
    "reconstruction": "solver.reconstruction",
    "riemann": "solver.riemann_solver",
    "dtype": "solver.dtype",
    "device": "solver.device",
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    cfg = apply_config_defaults(parser, CONFIG_DEFAULTS, argv)
    args = parser.parse_args(argv)
    if cfg.source:
        print(f"[config] defaults from {cfg.source}", flush=True)
    fit_hi = args.fit_hi if args.fit_hi > 0 else args.nx // 8

    # Must happen before any solver call: the traced graphs bake Γ in.
    set_gamma_eos(args.gamma)

    meta = {k: v for k, v in vars(args).items()}
    meta["config_source"] = cfg.source

    rho, vx, vy, p, x, y = khi_initial_conditions(
        args.nx, args.ny,
        v_shear=args.v_shear,
        thickness=args.thickness,
        rho1=args.rho1, rho2=args.rho2,
        pressure=args.pressure,
        seed_amplitude=args.seed_amplitude,
        seed_kmin=args.seed_kmin, seed_kmax=args.seed_kmax,
        seed_envelope_width=args.seed_envelope_width,
        rng_seed=args.seed,
    )

    dx, dy = 1.0 / args.nx, 1.0 / args.ny
    N = min(args.nx, args.ny)

    def report(t: float, rho, vx, vy, p) -> dict:
        d = {"t": t}
        d.update(state_diagnostics(rho, vx, vy, p, args.gamma, dx, dy))
        k, E_k = ke_spectrum(rho, vx, vy)
        d.update(spectral_diagnostics(k, E_k, args.report_r, N, args.fit_lo, fit_hi,
                                      fluct_kmin=args.seed_kmin))
        return d

    d0 = report(0.0, rho, vx, vy, p)
    cs_layer = sound_speed(np.array([args.rho1]), np.array([args.pressure]), args.gamma)[0]
    print("=" * 74)
    print(f"Relativistic KHI DNS   {args.nx}x{args.ny}   gamma={args.gamma:.6f}")
    print("=" * 74)
    print(f"  shear velocity     {args.v_shear:.3f}   ->  W = {1/math.sqrt(1-args.v_shear**2):.3f}")
    print(f"  sound speed        {cs_layer:.4f}   (ceiling for this gamma: {math.sqrt(args.gamma-1):.4f})")
    dv_rel, mach_rel = shear_mach(args.v_shear, cs_layer)
    print(f"  relative velocity  {dv_rel:.4f}   (relativistic: 2v/(1+v^2), not 2v)")
    print(f"  relativistic Mach  {mach_rel:.2f}")
    if mach_rel > 2.0:
        print("    ! M > 2: the layer is strongly supersonic and the KHI is suppressed.")
        print(f"      Measured at 256^2: M=1.0 grows the transverse KE by ~3300x over")
        print(f"      t=1, M=2 by ~1x (nothing). Lower --v-shear to ~0.4 (M=1.4) or")
        print(f"      below. With gamma={args.gamma:.4g} c_s cannot exceed"
              f" {math.sqrt(args.gamma-1):.4f}, so M <~ 2 caps the shear near v=0.5")
        print("      (W=1.15). High Lorentz factor and vigorous KHI are mutually")
        print("      exclusive here -- for high W, drive turbulence some other way.")
    k_cut = unstable_kmax(args.thickness)
    print(f"  layer thickness    {args.thickness:.4f} = {args.thickness/dx:.1f} fine cells"
          f" = {args.thickness/(dx*args.report_r):.1f} cells at r={args.report_r}")
    print(f"  unstable band      k < {k_cut:.1f}   (k_phys a < 1 for a tanh layer)")
    if args.seed_kmin > k_cut:
        print(f"    ! the entire seed band [{args.seed_kmin},{args.seed_kmax}] is above the")
        print(f"      cutoff: no seeded mode can grow. Thin the layer to a <"
              f" {1.0/(2*math.pi*args.seed_kmin):.4f} or lower --seed-kmin.")
    elif args.seed_kmax > 2 * k_cut:
        n_live = max(0, int(k_cut) - args.seed_kmin + 1)
        n_tot = args.seed_kmax - args.seed_kmin + 1
        print(f"    ! only {n_live} of {n_tot} seeded modes are unstable; the rest are damped"
              f" and waste seed amplitude. Consider --seed-kmax {int(k_cut)}.")
    print(f"  seed               k in [{args.seed_kmin},{args.seed_kmax}], "
          f"amplitude {args.seed_amplitude:g}, solenoidal, rng seed {args.seed}")
    print(f"  IC check           max|v| = {d0['v_max']:.5f}, W_max = {d0['W_max']:.4f}")
    print(f"  scale separation   k_c = {d0['k_c']:.0f} (r={args.report_r}), "
          f"k_diss = {d0['k_diss']:.0f}, ratio = {d0['k_diss_over_k_c']:.2f}")
    print("=" * 74, flush=True)

    if args.ic_only:
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    solver = RHDSolver(
        Ny=args.ny, Nx=args.nx, cfl=args.cfl,
        reconstruction=args.reconstruction, riemann_solver=args.riemann,
        dtype=dtype, device=device,
    )
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
    solver.initialize(to_t(rho), to_t(vx), to_t(vy), to_t(p))
    print(f"Device: {device}, dtype: {args.dtype}, output: {out_dir}", flush=True)

    def current() -> tuple[np.ndarray, ...]:
        pr = solver.prims.detach().to("cpu", torch.float64).numpy()
        return pr[..., 0], pr[..., 1], pr[..., 2], pr[..., 3]

    log: list[dict] = []
    t_outputs = [args.t_final * i / args.n_outputs for i in range(args.n_outputs + 1)]

    (out_dir / "manifest.json").write_text(json.dumps(
        {"parameters": meta, "git_commit": _git_commit(),
         "output_times": t_outputs, "format": "prim=[rho,vx,vy,p]; hydro=[D,E,Sx,Sy,Sz] SRHD"},
        indent=2, sort_keys=True))

    t_wall = time.time()
    diverged_at = None
    for i, t_out in enumerate(t_outputs):
        if i > 0:
            try:
                solver.evolve(
                    t_out, verbose=False,
                    progress_bar=not args.no_progress,
                    desc=f"[{i:3d}/{args.n_outputs}] t->{t_out:.3f}",
                )
            except FloatingPointError as exc:
                diverged_at = (i, t_out, str(exc))
                print(f"\n!! solver diverged before t={t_out:.4f}: {exc}", flush=True)
                print(f"   {i} snapshot(s) written; stopping.", flush=True)
                break

        r_, vx_, vy_, p_ = current()
        write_snapshot(out_dir / f"h{i:08d}.h5", r_, vx_, vy_, p_,
                       solver.t, x, y, args.gamma, meta)
        d = report(solver.t, r_, vx_, vy_, p_)
        log.append(d)
        print(
            f"  [{i:3d}] t={d['t']:7.4f}  W_max={d['W_max']:6.3f}  "
            f"rho=[{d['rho_min']:.3f},{d['rho_max']:.3f}]  p_min={d['p_min']:.3e}  "
            f"enstrophy={d['enstrophy']:.4e}  slope={d['slope']:+.2f}  "
            f"k_diss/k_c={d['k_diss_over_k_c']:.2f}  "
            f"subgrid={100*d['subgrid_fraction_fluct']:.2f}% of fluct "
            f"({100*d['subgrid_fraction']:.2f}% of total)",
            flush=True,
        )
        (out_dir / "diagnostics.json").write_text(json.dumps(log, indent=2))

    print(f"\nWall time: {(time.time()-t_wall)/60:.1f} min", flush=True)
    if log:
        # Selected on subgrid fraction, not enstrophy: the shear layer itself
        # carries large enstrophy from t=0 whether or not it has gone turbulent,
        # so enstrophy does not locate the useful window. Energy above the
        # filter cutoff is exactly what a closure has to model.
        best = max(log, key=lambda d: d["subgrid_fraction_fluct"])
        print(f"Most subgrid content at t = {best['t']:.4f}: "
              f"{100*best['subgrid_fraction_fluct']:.2f}% of fluctuation KE above k_c "
              f"at r={args.report_r} ({100*best['subgrid_fraction']:.2f}% of total KE, "
              f"k_diss/k_c = {best['k_diss_over_k_c']:.2f})")
        if best["t"] >= log[-1]["t"] - 1e-12 and len(log) > 2:
            print("  ! the subgrid fraction is still rising at the final time -- the flow")
            print("    has not finished developing. Increase --t-final.")
        if best["k_diss_over_k_c"] < 4.0:
            print("  ! k_diss/k_c < 4: the filter cutoff sits too close to the dissipation")
            print("    range. Raise the resolution or the coarsening factor before using")
            print("    this run as an SGS reference.")
        if best["subgrid_fraction_fluct"] < 0.03:
            print("  ! under 3% of the fluctuation energy is subgrid: too little for a")
            print("    closure to matter. Raise --report-r, run longer, or check that the")
            print("    layer is actually unstable (see the relativistic Mach warning).")
    return 1 if diverged_at else 0


if __name__ == "__main__":
    raise SystemExit(main())
