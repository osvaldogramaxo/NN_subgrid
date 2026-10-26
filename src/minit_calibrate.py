r"""
minit_calibrate.py

A-priori calibration and assessment of the MInIT closure (:mod:`src.minit`)
against this project's own DNS.

Why this exists
---------------
MInIT's published coefficients (Table 2 and Eq. 26 of arXiv:2308.06041) were
measured on *3D Newtonian MHD* box simulations at filter scales
Δ_f/L = 0.08–0.11. This project runs *2D SRHD* at Δ_f/L = 0.016. Three of those
four mismatches are severe enough that the published numbers carry no authority
here:

  * the Kolmogorov sink ``C e^{3/2}/lambda`` encodes a 3D forward cascade, and
    2D turbulence does not have one;
  * the paper's own Table 3 shows ``C_opt`` drifting 8.8 -> 11.0 across a filter
    range narrower than the gap between their scales and ours;
  * only the Reynolds stress survives in pure hydrodynamics.

So running MInIT as a serious competitor means refitting (beta, A_gamma, C) on
the local DNS, by the paper's own procedures, before it is scored a-posteriori.
Running it with the published numbers measures something different — what a
practitioner transferring the model off the shelf would actually get — and both
are produced here, because the gap between them separates the model from its
calibration.

What it computes
----------------
1. **Coarse-grained DNS reduction** — for each snapshot, the exact fluid-frame
   SGS stress ``tau_SGS`` (``dataset.compute_sgs_stress``), the measured
   ``e_KH = 1/2 sum_i tau^{ii}``, the resolved shear magnitude, and the coarse
   density. Cached, since this is the only expensive step.

2. **beta_ij** — by least squares through the origin (default), which is what
   "proportionality coefficient" means and is robust where ``e_KH`` is small,
   and by the paper's block-ratio procedure (averaging over boxes of size
   ~lambda_KH, then taking the ratio) for comparability with Table 2.

3. **(A_gamma, C)** — by minimising the paper's L2 norm (Eq. 25) between the
   model-evolved stresses and the measured ones, over a nested grid scan. The
   paper fixes ``A_gamma = 0.14`` on the grounds that results are weakly
   sensitive to it; that claim is checkable on the printed L2 surface here.

4. **An error decomposition** that the paper does not report and that decides
   where any effort should go:

   * ``oracle-e``  — ``beta_ij e_KH^DNS`` against ``tau_exact``. Isolates the
     error of the *proportionality ansatz* alone.
   * ``full``      — ``beta_ij e_KH^model`` against ``tau_exact``. Adds the
     error of the *evolution equation*.

   If ``oracle-e`` is already poor, no amount of refitting ``C`` will help,
   because the closure relation itself is what fails.

5. **Energy transfer** ``Pi = tau^{ij} S_ij``, model against exact. MInIT's
   near-isotropic diagonal (beta_xx ~ beta_yy) makes its stress mostly a
   sub-grid pressure, whose transfer is ``~(2/3) e_KH div(v)`` and so nearly
   vanishes in a near-incompressible flow. Pi is where that shows up, and it is
   the quantity that governs whether the closure can do anything a-posteriori.

Usage
-----
    # Full calibration over the developed-turbulence window
    python -m src.minit_calibrate --data-dir dns_khi --coarsening-factor 16

    # Refit from the cache (instant), on a different window
    python -m src.minit_calibrate --t-lo 8 --t-hi 16

    # Score the published coefficients without refitting
    python -m src.minit_calibrate --paper-only
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

from src.dataset import compute_sgs_stress, read_snapshot
from src.features import periodic_gradient_2d
from src.minit import (
    A_GAMMA_PAPER,
    C_PAPER,
    LAMBDA_OVER_DELTA,
    MInITCoefficients,
    advance_e_kh,
    advect_e_kh,
    growth_rate,
    row_mean,
    saturation_energy,
    subgrid_energy_from_stress,
)


# ---------------------------------------------------------------------------
# Stage 1: reduce the DNS to the coarse-grid quantities the model needs
# ---------------------------------------------------------------------------

def reduce_dns(
    data_dir: str,
    r: int,
    stride: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    r"""Reduce every DNS snapshot to coarse fields plus the exact SGS stress.

    This is the only expensive stage — it filters a full-resolution energy
    momentum tensor per snapshot — so its output is cached by the CLI.

    Returns
    -------
    dict of stacked (T, ...) tensors on the CPU:
        ``t``        (T,)            — snapshot times
        ``rho``      (T, Ny, Nx)     — coarse (Favre-filtered) density
        ``vx``, ``vy``               — coarse velocities
        ``shear``    (T, Ny, Nx)     — ``|d(vx)/dy|``, i.e. the growth rate at
                                       ``A_gamma = 1``, so a fit over A_gamma is
                                       a rescaling and needs no re-differencing
        ``strain``   (T, Ny, Nx)     — ``sqrt(2 S_ij S_ij)``, the invariant variant
        ``e_dns``    (T, Ny, Nx)     — measured ``e_KH = 1/2 sum_i tau^{ii}``
        ``tau``      (T, Ny, Nx, 3)  — exact ``[tau^xx, tau^yy, tau^xy]``
        ``div_v``    (T, Ny, Nx)     — resolved divergence, for the Pi diagnostic
        ``S``        (T, Ny, Nx, 3)  — resolved rate of strain ``[S_xx, S_yy, S_xy]``
    """
    device = device or torch.device("cpu")
    files = sorted(glob.glob(os.path.join(data_dir, "h*.h5")))[::stride]
    if not files:
        raise FileNotFoundError(f"no h*.h5 snapshots under {data_dir!r}")

    out: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("rho", "vx", "vy", "shear", "strain", "e_dns", "tau", "div_v", "S")
    }
    times: list[float] = []

    pbar = tqdm(files, desc="Reducing DNS", unit="snap", dynamic_ncols=True, file=sys.stdout)
    for path in pbar:
        snap = read_snapshot(path)
        fine = [
            torch.tensor(snap[k], dtype=dtype, device=device)
            for k in ("rho", "vx", "vy", "p")
        ]
        coarse, tau_sgs = compute_sgs_stress(*fine, r=r, project_to_fluid_frame=True)

        Nc_y, Nc_x = coarse["rho_bar"].shape
        dx, dy = 1.0 / Nc_x, 1.0 / Nc_y
        vx_c, vy_c = coarse["vx_tilde"], coarse["vy_tilde"]

        dvx_dx, dvx_dy = periodic_gradient_2d(vx_c, dx, dy, order=4)
        dvy_dx, dvy_dy = periodic_gradient_2d(vy_c, dx, dy, order=4)
        S_xx, S_yy = dvx_dx, dvy_dy
        S_xy = 0.5 * (dvx_dy + dvy_dx)

        times.append(float(snap["t"]))
        out["rho"].append(coarse["rho_bar"].cpu())
        out["vx"].append(vx_c.cpu())
        out["vy"].append(vy_c.cpu())
        out["shear"].append(dvx_dy.abs().cpu())
        out["strain"].append(
            (2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2)).clamp(min=0.0).sqrt().cpu()
        )
        out["e_dns"].append(subgrid_energy_from_stress(tau_sgs).cpu())
        out["tau"].append(
            torch.stack(
                [tau_sgs[..., 1, 1], tau_sgs[..., 2, 2], tau_sgs[..., 1, 2]], dim=-1
            ).cpu()
        )
        out["div_v"].append((dvx_dx + dvy_dy).cpu())
        out["S"].append(torch.stack([S_xx, S_yy, S_xy], dim=-1).cpu())
        pbar.set_postfix(t=f"{snap['t']:.2f}")

    reduced = {k: torch.stack(v) for k, v in out.items()}
    reduced["t"] = torch.tensor(times, dtype=dtype)
    return reduced


# ---------------------------------------------------------------------------
# Stage 2: the proportionality coefficients beta_ij
# ---------------------------------------------------------------------------

def fit_beta_lsq(e: torch.Tensor, tau: torch.Tensor) -> dict[str, float]:
    r"""Least-squares slope through the origin, ``beta_ij = <tau^ij e> / <e^2>``.

    The minimiser of ``sum (tau^ij - beta_ij e)^2``, i.e. the estimator implied
    by reading "the stress is proportional to e_KH" as a statement to be fitted.
    Unlike an average of per-cell ratios it is not destabilised by the cells
    where ``e_KH`` is near zero, which on a 64^2 coarse grid is most of them
    outside the mixing layer.

    It also satisfies the trace identity ``beta_xx + beta_yy = 2`` exactly and
    for free, since ``e_KH`` is itself defined as ``1/2 (tau^xx + tau^yy)``::

        beta_xx + beta_yy = sum((tau^xx + tau^yy) e) / sum(e^2)
                          = sum(2 e * e) / sum(e^2) = 2

    The block-ratio estimator carries no such guarantee, so the residual it
    leaves is a genuine measure of how well the proportionality ansatz holds.
    """
    denom = (e**2).sum().clamp(min=1e-300)
    return {
        "beta_xx": float((tau[..., 0] * e).sum() / denom),
        "beta_yy": float((tau[..., 1] * e).sum() / denom),
        "beta_xy": float((tau[..., 2] * e).sum() / denom),
    }


def fit_beta_block_ratio(
    e: torch.Tensor, tau: torch.Tensor, block: int,
) -> dict[str, float]:
    r"""The paper's estimator: block-average, take the ratio, then average.

    Section 4.3.2 averages the stresses and ``e_KH`` over boxes of size similar
    to ``lambda_KH``, forms the ratio per box, and averages over boxes and over
    the fully turbulent times. Reproduced here so the fitted coefficients are
    comparable with Table 2 on a like-for-like basis; the least-squares
    estimator above is the one used for production.
    """
    T, Ny, Nx = e.shape
    b = max(1, min(block, Ny, Nx))
    ny, nx = Ny // b, Nx // b
    if ny == 0 or nx == 0:
        raise ValueError(f"block size {b} exceeds the coarse grid {Ny}x{Nx}")

    def _blocks(f: torch.Tensor) -> torch.Tensor:
        return f[:, : ny * b, : nx * b].reshape(T, ny, b, nx, b).mean(dim=(2, 4))

    e_b = _blocks(e)
    valid = e_b.abs() > 1e-14
    out = {}
    for idx, name in enumerate(("beta_xx", "beta_yy", "beta_xy")):
        ratio = _blocks(tau[..., idx]) / e_b.where(valid, torch.ones_like(e_b))
        out[name] = float(ratio[valid].mean()) if valid.any() else float("nan")
    return out


# ---------------------------------------------------------------------------
# Stage 3: (A_gamma, C) by the paper's L2 norm
# ---------------------------------------------------------------------------

def iter_e_series(
    e0: torch.Tensor,
    shear: torch.Tensor,
    rho: torch.Tensor,
    times: torch.Tensor,
    A_gamma: torch.Tensor,
    C: torch.Tensor,
    lambda_KH: float,
    substeps: int = 10,
    vel: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    r"""Yield ``(k, e_k)`` for a batch of ``(A_gamma, C)``, one snapshot at a time.

    A-priori: the growth rate and density are read from the DNS at every step,
    so the model is never fed its own output except through ``e_KH`` itself.
    The parameter batch is carried as a leading dimension, which turns the
    calibration scan into a handful of vectorised updates rather than a Python
    loop over candidates.

    A generator rather than a returned array because the scan's parameter batch
    times the snapshot count times the grid would be gigabytes held for no
    reason — the objective only ever needs one time slice at a time.

    Parameters
    ----------
    e0 : (Ny, Nx) — seed at ``times[0]``, taken from the measured DNS ``e_KH``.
    shear : (T, Ny, Nx) — growth rate at ``A_gamma = 1``.
    rho : (T, Ny, Nx) — coarse density.
    times : (T,)
    A_gamma, C : (P,) — parameter batch.
    substeps : sub-steps per snapshot interval. The update is exact for frozen
        coefficients, so this only controls how finely the *time dependence* of
        the shear and density is resolved, not stability.
    vel : optional (vx, vy), each (T, Ny, Nx) — the coarse DNS velocity. If given,
        the advective term ``-d_i(v_i e_KH)`` the paper drops is applied before
        each sub-step's source update (operator splitting, as in MInITClosure).

    Yields
    ------
    (k, e_k) with ``e_k`` of shape (P, Ny, Nx).
    """
    P = A_gamma.shape[0]
    T = times.shape[0]
    A = A_gamma.view(P, 1, 1)
    Cv = C.view(P, 1, 1)

    e = e0.unsqueeze(0).expand(P, *e0.shape).clone()
    yield 0, e

    for k in range(T - 1):
        dt_total = float(times[k + 1] - times[k])
        if dt_total > 0:
            dt = dt_total / substeps
            for s in range(substeps):
                # Linear interpolation of the DNS fields across the interval
                # keeps the frozen-coefficient assumption of the exact update
                # local to a fraction of a snapshot spacing.
                w = (s + 0.5) / substeps
                shear_s = (1.0 - w) * shear[k] + w * shear[k + 1]
                rho_s = (1.0 - w) * rho[k] + w * rho[k + 1]
                if vel is not None:
                    e = _advect(e, vel, k, k + 1, w, dt)
                e = advance_e_kh(e, A * shear_s, rho_s.unsqueeze(0), dt, lambda_KH, Cv)
        yield k + 1, e


def one_step_series(
    e_dns: torch.Tensor,
    shear: torch.Tensor,
    rho: torch.Tensor,
    times: torch.Tensor,
    A_gamma: float,
    C: float,
    lambda_KH: float,
    substeps: int = 10,
    vel: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    r"""Advance the measured ``e_KH`` by exactly one snapshot interval, everywhere.

    The free-running test conflates two failures: the evolution equation may be
    a poor rate law, or it may be a decent one whose small errors compound over
    a long window. Re-seeding from the DNS at every snapshot and taking a single
    interval separates them — this measures the rate law's short-term skill
    alone, which is the standard a-priori protocol for an evolution equation and
    is the more informative number when the free run has drifted.

    Returns
    -------
    (T-1, Ny, Nx) tensor: the model's prediction for ``e_KH`` at ``times[k+1]``
    given truth at ``times[k]``.
    """
    e = e_dns[:-1].clone()
    dt_all = (times[1:] - times[:-1]).view(-1, 1, 1)

    for s in range(substeps):
        w = (s + 0.5) / substeps
        shear_s = (1.0 - w) * shear[:-1] + w * shear[1:]
        rho_s = (1.0 - w) * rho[:-1] + w * rho[1:]
        if vel is not None:
            e = _advect(e, vel, slice(None, -1), slice(1, None), w, dt_all / substeps)
        # advance_e_kh is elementwise in everything but dt, so the whole batch of
        # intervals advances at once as long as each carries its own dt.
        e = _advance_batched_dt(e, A_gamma * shear_s, rho_s, dt_all / substeps, lambda_KH, C)
    return e


def _advect(e, vel, k0, k1, w, dt):
    """:func:`src.minit.advect_e_kh` with the velocity interpolated between snapshots k0 and k1."""
    vx, vy = ((1.0 - w) * v[k0] + w * v[k1] for v in vel)
    Ny, Nx = e.shape[-2:]
    return advect_e_kh(e, vx, vy, 1.0 / Nx, 1.0 / Ny, dt)


def _advance_batched_dt(
    e: torch.Tensor,
    gamma: torch.Tensor,
    rho: torch.Tensor,
    dt: torch.Tensor,
    lambda_KH: float,
    C: float,
) -> torch.Tensor:
    """:func:`src.minit.advance_e_kh` with a per-slice ``dt`` tensor."""
    K = C / (rho.clamp(min=1e-15).sqrt() * lambda_KH)
    g_dt = gamma * dt
    phi = torch.where(
        g_dt > 1e-8,
        -torch.expm1(-g_dt) / gamma.clamp(min=1e-30),
        dt * (1.0 - 0.5 * g_dt),
    )
    positive = e > 0
    w = torch.where(positive, e.clamp(min=1e-300).rsqrt(), torch.zeros_like(e))
    w_new = w * torch.exp(-g_dt) + 0.5 * K * phi
    return torch.where(positive, w_new.clamp(min=1e-300) ** -2, torch.zeros_like(e))


def evolve_e_series(
    e0: torch.Tensor,
    shear: torch.Tensor,
    rho: torch.Tensor,
    times: torch.Tensor,
    A_gamma: torch.Tensor,
    C: torch.Tensor,
    lambda_KH: float,
    substeps: int = 10,
    vel: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Materialise :func:`iter_e_series` as a (P, T, Ny, Nx) tensor.

    Only used for the handful of named coefficient sets that get scored and
    plotted, never for the calibration scan.
    """
    series = [e.clone() for _, e in iter_e_series(
        e0, shear, rho, times, A_gamma, C, lambda_KH, substeps, vel,
    )]
    return torch.stack(series, dim=1)


# Floor on the model's energy relative to the measurement's, so a collapsed
# model scores a large but finite and reproducible penalty. See l2_norm_paper.
_COLLAPSE_FLOOR = 1e-12


def l2_norm_paper(sim: torch.Tensor, model: torch.Tensor) -> torch.Tensor:
    r"""The paper's relative L2 norm, Eq. (25), per time slice.

    .. math::
        L_2 = \sqrt{\tfrac12 \sum_i (x_i^s - x_i^m)^2
                    \left(\tfrac{1}{\sum_j (x_j^s)^2} + \tfrac{1}{\sum_j (x_j^m)^2}\right)}

    summed over all spatial points and stress components.

    Taken literally the norm diverges when the model collapses to zero, since
    ``1 / sum (x^m)^2`` does. A collapsed model *is* maximally wrong, so the
    penalty should be large — but it must not depend on an arbitrary epsilon, or
    on floating-point luck. The model denominator is therefore floored at a fixed
    small fraction of the measured one, which caps the score at a finite, scale
    invariant ``~7e5`` instead of letting it run to ``1e150`` or to infinity.

    Parameters
    ----------
    sim, model : (..., N) tensors whose last axis is flattened over space and
        stress components. Leading axes broadcast, so one measured slice can be
        scored against a whole batch of candidate parameters at once.

    Returns
    -------
    tensor with the last axis reduced away.
    """
    diff_sq = ((sim - model) ** 2).sum(dim=-1)
    sim_sq = (sim**2).sum(dim=-1).clamp(min=1e-300)
    mod_sq = (model**2).sum(dim=-1).clamp(min=_COLLAPSE_FLOOR * sim_sq)
    return (0.5 * diff_sq * (1.0 / sim_sq + 1.0 / mod_sq)).clamp(min=0.0).sqrt()


def scan_A_C(
    e_dns: torch.Tensor,
    tau: torch.Tensor,
    shear: torch.Tensor,
    rho: torch.Tensor,
    times: torch.Tensor,
    beta: dict[str, float],
    lambda_KH: float,
    A_range: tuple[float, float],
    C_range: tuple[float, float],
    n_A: int = 21,
    n_C: int = 21,
    rounds: int = 3,
    substeps: int = 10,
    fix_A_gamma: float | None = None,
    vel: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    r"""Minimise the Eq. (25) norm over ``(A_gamma, C)`` by nested grid scan.

    A scan rather than a gradient method because the objective is cheap once
    the parameter batch is vectorised, the domain is two-dimensional and
    bounded, and the resulting L2 surface is itself reportable — the paper's
    claim that results are weakly sensitive to ``A_gamma`` (its Fig. 11) is only
    checkable if the surface is kept.

    Returns a dict with the optimum, the final surface, and the axes.
    """
    beta_vec = torch.tensor(
        [beta["beta_xx"], beta["beta_yy"], beta["beta_xy"]], dtype=e_dns.dtype,
    )
    # Reynolds stress only: the Maxwell and Faraday tensors of Eq. (25) have no
    # counterpart in a hydrodynamic solver.
    sim_flat = tau.flatten(start_dim=-3)                    # (T, Ny*Nx*3)

    lo_A, hi_A = A_range
    lo_C, hi_C = C_range
    if fix_A_gamma is not None:
        lo_A = hi_A = float(fix_A_gamma)
        n_A = 1
    best = {"A_gamma": float("nan"), "C": float("nan"), "l2": float("inf")}
    surface = axis_A = axis_C = None
    pinned: list[str] = []

    for _ in range(rounds):
        axis_A = torch.linspace(lo_A, hi_A, n_A, dtype=e_dns.dtype)
        axis_C = torch.linspace(lo_C, hi_C, n_C, dtype=e_dns.dtype)
        grid_A, grid_C = torch.meshgrid(axis_A, axis_C, indexing="ij")
        flat_A, flat_C = grid_A.reshape(-1), grid_C.reshape(-1)

        # Accumulate the per-time norm as the series is generated, so only one
        # (P, Ny*Nx*3) slice is ever resident.
        l2_sq_sum = torch.zeros(flat_A.shape[0], dtype=e_dns.dtype)
        for k, e_k in iter_e_series(
            e_dns[0], shear, rho, times, flat_A, flat_C, lambda_KH, substeps, vel,
        ):
            model_flat = (e_k.unsqueeze(-1) * beta_vec).flatten(start_dim=-3)
            l2_sq_sum += l2_norm_paper(sim_flat[k].unsqueeze(0), model_flat) ** 2

        # RMS over time of the per-time norm, as in Sec. 4.3.3.
        score = (l2_sq_sum / times.shape[0]).sqrt()
        surface = score.reshape(n_A, n_C)

        k = int(torch.argmin(score))
        if float(score[k]) < best["l2"]:
            best = {
                "A_gamma": float(flat_A[k]),
                "C": float(flat_C[k]),
                "l2": float(score[k]),
            }

        # Shrink around the optimum for the next round.
        dC = (hi_C - lo_C) / max(n_C - 1, 1)
        lo_C, hi_C = max(1e-3, best["C"] - 2 * dC), best["C"] + 2 * dC
        if fix_A_gamma is None:
            dA = (hi_A - lo_A) / max(n_A - 1, 1)
            lo_A, hi_A = max(1e-4, best["A_gamma"] - 2 * dA), best["A_gamma"] + 2 * dA

    return {
        "best": best,
        "surface": surface,
        "axis_A": axis_A,
        "axis_C": axis_C,
        "pinned": _boundary_pinned(best, A_range, C_range, n_A, n_C, fix_A_gamma),
    }


def _boundary_pinned(
    best: dict,
    A_range: tuple[float, float],
    C_range: tuple[float, float],
    n_A: int,
    n_C: int,
    fix_A_gamma: float | None,
) -> list[str]:
    """Name any edge of the *requested* search range the optimum ended up on.

    An optimum on the boundary is not an optimum — it means the range was too
    narrow and the reported value records where the scan was cut rather than
    where the objective turns over. The nested rounds can drift outside the
    original box, so this is checked against the range the caller asked for,
    once, at the end.
    """
    out = []
    checks = [("C", C_range, n_C)]
    if fix_A_gamma is None:
        checks.append(("A_gamma", A_range, n_A))
    for name, (lo, hi), n in checks:
        tol = 0.51 * (hi - lo) / max(n - 1, 1)
        if abs(best[name] - lo) < tol:
            out.append(f"{name} at lower bound {lo:g}")
        if abs(best[name] - hi) < tol:
            out.append(f"{name} at upper bound {hi:g}")
    return out


def fit_equilibrium(
    tau: torch.Tensor,
    shear: torch.Tensor,
    rho: torch.Tensor,
    beta: dict[str, float],
    lambda_KH: float,
    A_gamma: float = A_GAMMA_PAPER,
    C_range: tuple[float, float] = (0.1, 60.0),
    n_C: int = 201,
    rounds: int = 3,
) -> dict:
    r"""Fit the equilibrium limit of MInIT, in which the ODE is replaced by its fixed point.

    With ``e_KH = e_sat = (2 A_gamma |S| sqrt(rho) lambda / C)^2`` the closure
    becomes algebraic,

    .. math::
        \tau^{ij} = \beta_{ij}\,(2 A_\gamma / C)^2 \rho\, \lambda_{\rm KH}^2\, |S|^2

    i.e. a Smagorinsky-type stress carrying MInIT's calibrated anisotropy. Only
    the ratio ``A_gamma / C`` enters, so this is a genuine one-parameter fit and
    ``A_gamma`` is held at its theoretical value while ``C`` absorbs the scale.

    This variant exists because the free two-parameter calibration keeps asking
    for it: pushing ``A_gamma`` far above 0.14 buys L2 only by shortening the
    relaxation time until the ODE no longer does anything. Fitting the limit
    directly is the honest version of that, and gives MInIT its best shot.
    """
    beta_vec = torch.tensor(
        [beta["beta_xx"], beta["beta_yy"], beta["beta_xy"]], dtype=tau.dtype,
    )
    sim_flat = tau.flatten(start_dim=-3)

    lo, hi = C_range
    best = {"C": float("nan"), "l2": float("inf")}
    for _ in range(rounds):
        axis = torch.linspace(lo, hi, n_C, dtype=tau.dtype)
        l2_sq = torch.zeros(n_C, dtype=tau.dtype)
        for k in range(tau.shape[0]):
            e_eq = saturation_energy(
                A_gamma * shear[k].unsqueeze(0), rho[k].unsqueeze(0),
                lambda_KH, axis.view(-1, 1, 1),
            )
            model_flat = (e_eq.unsqueeze(-1) * beta_vec).flatten(start_dim=-3)
            l2_sq += l2_norm_paper(sim_flat[k].unsqueeze(0), model_flat) ** 2

        score = (l2_sq / tau.shape[0]).sqrt()
        j = int(torch.argmin(score))
        if float(score[j]) < best["l2"]:
            best = {"C": float(axis[j]), "l2": float(score[j])}
        d = (hi - lo) / (n_C - 1)
        lo, hi = max(1e-3, best["C"] - 2 * d), best["C"] + 2 * d

    best["A_gamma"] = A_gamma
    best["ratio"] = A_gamma / best["C"]
    return best


# ---------------------------------------------------------------------------
# Stage 4: a-priori scoring
# ---------------------------------------------------------------------------

def score_tau(tau_model: torch.Tensor, tau_exact: torch.Tensor) -> dict[str, float]:
    r"""Pointwise a-priori agreement of a modelled stress with the exact one.

    Relative L2 is meaningful here — unlike a-posteriori, where it measures
    phase decorrelation on a chaotic trajectory — because both tensors are
    evaluated on the same state at the same instant. The correlation is reported
    alongside because a closure can have the right magnitude and no spatial
    structure, which rel-L2 alone will not reveal.
    """
    diff = tau_model - tau_exact
    rel_l2 = float(diff.pow(2).sum().sqrt() / tau_exact.pow(2).sum().sqrt().clamp(min=1e-300))

    a = tau_model.flatten() - tau_model.mean()
    b = tau_exact.flatten() - tau_exact.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-300)
    corr = float((a @ b) / denom)

    per_component = {}
    for idx, name in enumerate(("xx", "yy", "xy")):
        m, x = tau_model[..., idx].flatten(), tau_exact[..., idx].flatten()
        am, bx = m - m.mean(), x - x.mean()
        per_component[f"corr_{name}"] = float(
            (am @ bx) / (am.norm() * bx.norm()).clamp(min=1e-300)
        )
    return {"rel_l2": rel_l2, "corr": corr, **per_component}


def energy_transfer(tau: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    r"""Resolved-scale energy transfer ``Pi = tau^{ij} S_ij``.

    Positive means the sub-grid stress drains resolved kinetic energy — the job
    an SGS closure exists to do. With ``[xx, yy, xy]`` packing the cross term is
    counted twice, once for each off-diagonal position.
    """
    return tau[..., 0] * S[..., 0] + tau[..., 1] * S[..., 1] + 2.0 * tau[..., 2] * S[..., 2]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(out_dir: Path, payload: dict, arrays: dict) -> None:
    """Write the diagnostic figures. Import-guarded so the fit runs headless."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = arrays["t"].numpy()

    # 1. e_KH trajectory: measured against fitted and published models.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(t, arrays["e_dns_mean"], "k-", lw=2, label=r"DNS $e_{\rm KH}$")
    for label, series, style in arrays["e_model_curves"]:
        ax.semilogy(t, series, style, lw=1.5, label=label)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\langle e_{\rm KH}\rangle$")
    ax.set_title("MInIT turbulent energy density, a-priori")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "minit_e_kh.png", dpi=150)
    plt.close(fig)

    # 2. The (A_gamma, C) L2 surface — the paper's Fig. 11 sensitivity claim.
    if arrays.get("surface") is not None:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        surf = arrays["surface"].numpy()
        im = ax.pcolormesh(
            arrays["axis_C"].numpy(), arrays["axis_A"].numpy(), surf,
            shading="nearest", cmap="viridis",
        )
        best = payload["fitted"]
        ax.plot(best["C"], best["A_gamma"], "r*", ms=16, label="optimum")
        ax.plot(C_PAPER, A_GAMMA_PAPER, "wo", ms=8, label="published")
        ax.set_xlabel("$C$")
        ax.set_ylabel(r"$A_\gamma$")
        ax.set_title("Eq. (25) $L_2$ norm (final refinement round)")
        ax.legend()
        fig.colorbar(im, ax=ax, label="$L_2$")
        fig.tight_layout()
        fig.savefig(out_dir / "minit_l2_surface.png", dpi=150)
        plt.close(fig)

    # 3. Where the free-running model loses the exact stress, snapshot by snapshot.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, arrays["corr_oracle"], "k-", lw=2,
            label=r"ansatz only ($\beta_{ij}e_{\rm KH}^{\rm DNS}$)")
    for label, series, style in arrays["corr_series"]:
        ax.plot(t, series, style, lw=1.5, label=label)
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("t")
    ax.set_ylabel(r"corr$(\tau^{\rm model}, \tau^{\rm exact})$")
    ax.set_title("A-priori correlation with the exact SGS stress")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "minit_correlation.png", dpi=150)
    plt.close(fig)

    # 4. Energy transfer: the quantity that decides a-posteriori relevance.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, arrays["pi_exact"], "k-", lw=2, label=r"exact $\Pi$")
    for label, series, style in arrays["pi_curves"]:
        ax.plot(t, series, style, lw=1.5, label=label)
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\langle \Pi \rangle = \langle \tau^{ij} S_{ij}\rangle$")
    ax.set_title("Resolved-scale energy transfer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "minit_transfer.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--data-dir", default="dns_khi")
    ap.add_argument("--coarsening-factor", "-r", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1, help="snapshot stride when reducing")
    ap.add_argument("--t-lo", type=float, default=6.0,
                    help="start of the developed-turbulence fitting window")
    ap.add_argument("--t-hi", type=float, default=1e30)
    ap.add_argument("--beta-estimator", choices=["lsq", "block"], default="lsq")
    ap.add_argument("--normalize-beta", action="store_true",
                    help="rescale the diagonal so beta_xx + beta_yy = 2 exactly")
    ap.add_argument("--shear-mode", choices=["dvxdy", "strain"], default="dvxdy")
    ap.add_argument("--average", choices=["cell", "x"], default="cell",
                    help="'cell': Eq. 12 cell by cell with each cell's own |d(vx)/dy|. "
                         "'x': MInIT's mean-field form, as in the paper -- the growth "
                         "rate of the x-averaged shear profile, and e_KH, rho and the "
                         "exact stress all averaged over x before fitting and scoring.")
    ap.add_argument("--substeps", type=int, default=10)
    ap.add_argument("--advect", action="store_true",
                    help="include the advective term -d_i(v_i e_KH) the paper drops, "
                         "with the coarse DNS velocity (cell form only)")
    ap.add_argument("--A-range", type=float, nargs=2, default=(0.01, 2.0),
                    metavar=("LO", "HI"), help="search range for A_gamma")
    ap.add_argument("--C-range", type=float, nargs=2, default=(0.1, 60.0),
                    metavar=("LO", "HI"), help="search range for C")
    ap.add_argument("--fix-A-gamma", type=float, default=A_GAMMA_PAPER,
                    help="hold A_gamma at this value and fit C alone, which is the "
                         "paper's own procedure and the default (0.14, the Miura & "
                         "Pritchett value). e_sat scales as (A_gamma/C)^2, so the "
                         "two are degenerate along a ray and the free fit buys "
                         "almost nothing while wandering far from theory.")
    ap.add_argument("--free-A-gamma", action="store_true",
                    help="use the free two-parameter optimum for the production "
                         "coefficients instead of the constrained fit.")
    ap.add_argument("--paper-only", action="store_true",
                    help="score the published coefficients without refitting")
    ap.add_argument("--cache", default="results/minit/dns_reduction.pt")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--output-dir", default="results/minit")
    ap.add_argument("--coeff-out", default="results/minit/minit_coefficients.json")
    ap.add_argument("--device", default="cpu",
                    help="device for the DNS reduction ('cuda' speeds up filtering)")
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1 (cached) -------------------------------------------------
    cache_path = Path(args.cache)
    cache_key = {"data_dir": args.data_dir, "r": args.coarsening_factor, "stride": args.stride}
    reduced = None
    if cache_path.exists() and not args.no_cache:
        blob = torch.load(cache_path, weights_only=False)
        if blob.get("key") == cache_key:
            reduced = blob["data"]
            print(f"Loaded cached DNS reduction from {cache_path}")
    if reduced is None:
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        reduced = reduce_dns(args.data_dir, args.coarsening_factor, args.stride, device)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"key": cache_key, "data": reduced}, cache_path)
        print(f"Cached DNS reduction to {cache_path}")

    t_all = reduced["t"]
    mask = (t_all >= args.t_lo) & (t_all <= args.t_hi)
    if int(mask.sum()) < 3:
        raise ValueError(
            f"fitting window [{args.t_lo}, {args.t_hi}] holds {int(mask.sum())} snapshots; need >= 3"
        )
    sel = {k: v[mask] for k, v in reduced.items()}
    t = sel["t"]
    e_dns = sel["e_dns"]
    tau_exact = sel["tau"]
    shear = sel["strain"] if args.shear_mode == "strain" else sel["shear"]

    if args.advect and args.average == "x":
        raise ValueError("advection cancels in the x-averaged model; --advect needs --average cell")
    vel = (sel["vx"], sel["vy"]) if args.advect else None

    if args.average == "x":
        if args.shear_mode != "dvxdy":
            raise ValueError("--average x uses the paper's d<vx>_x/dy; use --shear-mode dvxdy")
        # Paper Sec. 4.3.1: gamma from the mean shear, averaged over the homogeneous
        # plane; the DNS quantities it is calibrated against are averages too.
        Ny_, Nx_ = e_dns.shape[-2:]
        shear = growth_rate(row_mean(sel["vx"]), row_mean(sel["vy"]), 1.0 / Nx_, 1.0 / Ny_, 1.0)
        sel["rho"] = row_mean(sel["rho"])
        e_dns = row_mean(e_dns)
        tau_exact = tau_exact.mean(dim=-2, keepdim=True).expand_as(tau_exact)
        print("Mean-field (x-averaged) MInIT: gamma from d<vx>_x/dy; e_KH, rho and the "
              "exact stress averaged over x")

    Nc_y, Nc_x = e_dns.shape[-2:]
    delta_f = 1.0 / Nc_x
    lambda_KH = LAMBDA_OVER_DELTA * delta_f

    print(f"\nGrid: {Nc_y}x{Nc_x} coarse (r={args.coarsening_factor}), "
          f"Delta_f/L={delta_f:.4f}, lambda_KH={lambda_KH:.4f}")
    print(f"Window: t in [{float(t[0]):.2f}, {float(t[-1]):.2f}], {len(t)} snapshots")
    print(f"Shear estimator: {args.shear_mode}, advection: {'on' if args.advect else 'off'}")

    # --- Stage 2: beta ----------------------------------------------------
    beta_lsq = fit_beta_lsq(e_dns, tau_exact)
    block = max(1, int(round(lambda_KH * Nc_x)))
    beta_block = fit_beta_block_ratio(e_dns, tau_exact, block)
    print(f"\n--- beta_ij (window average) ---")
    print(f"  least squares      : {beta_lsq}")
    print(f"  block ratio (b={block:d})  : {beta_block}")
    print(f"  published (Table 2): beta_xx=0.74 beta_yy=0.64 beta_xy=-0.01")

    beta = beta_lsq if args.beta_estimator == "lsq" else beta_block
    trace_res = beta["beta_xx"] + beta["beta_yy"] - 2.0
    print(f"  trace residual (beta_xx+beta_yy-2) = {trace_res:+.4f}  "
          f"[published: {0.74 + 0.64 + 0.61 - 2.0:+.4f} in 3D]")

    # --- Stage 3: A_gamma, C ---------------------------------------------
    constrained = None
    if args.paper_only:
        fitted = {"A_gamma": A_GAMMA_PAPER, "C": C_PAPER, "l2": float("nan")}
        scan = None
    else:
        common = dict(
            A_range=tuple(args.A_range), C_range=tuple(args.C_range),
            substeps=args.substeps, vel=vel,
        )
        scan = scan_A_C(e_dns, tau_exact, shear, sel["rho"], t, beta, lambda_KH, **common)

        # e_sat scales as (A_gamma / C)^2, so in the saturated regime the two
        # parameters are degenerate along a ray and only their ratio is
        # determined. That is the paper's Fig. 11 observation ("for a wide range
        # of A_gamma it is always possible to find a C"), which is why it fixes
        # A_gamma. Fitting C alone at the published A_gamma costs nothing if the
        # valley is genuinely flat, and the two L2 values here say whether it is.
        scan_fixed = scan_A_C(
            e_dns, tau_exact, shear, sel["rho"], t, beta, lambda_KH,
            fix_A_gamma=A_GAMMA_PAPER, **common,
        )
        constrained = scan_fixed["best"]

        print(f"\n--- (A_gamma, C) fit, Eq. (25) norm ---")
        print(f"  free 2-parameter : A_gamma={scan['best']['A_gamma']:.4f}  "
              f"C={scan['best']['C']:.4f}   L2={scan['best']['l2']:.4f}")
        print(f"  A_gamma fixed    : A_gamma={constrained['A_gamma']:.4f}  "
              f"C={constrained['C']:.4f}   L2={constrained['l2']:.4f}")
        print(f"  published        : A_gamma={A_GAMMA_PAPER}  C={C_PAPER}")
        gap = constrained["l2"] - scan["best"]["l2"]
        print(f"  cost of fixing A_gamma at {A_GAMMA_PAPER}: dL2 = {gap:+.4f} "
              f"({100.0 * gap / max(scan['best']['l2'], 1e-30):+.1f}%)")

        if not args.free_A_gamma:
            scan = (
                scan_fixed if args.fix_A_gamma == A_GAMMA_PAPER
                else scan_A_C(
                    e_dns, tau_exact, shear, sel["rho"], t, beta, lambda_KH,
                    fix_A_gamma=args.fix_A_gamma, **common,
                )
            )
        fitted = scan["best"]
        if scan["pinned"]:
            print(f"  WARNING: optimum sits on the search boundary "
                  f"({'; '.join(scan['pinned'])}). Widen --A-range / --C-range.")

    # The equilibrium limit: e_KH = e_sat, the ODE dropped. Only A_gamma / C is
    # identifiable, so C is scanned at the fixed theoretical A_gamma.
    eq_fit = None
    if not args.paper_only:
        eq_fit = fit_equilibrium(
            tau_exact, shear, sel["rho"], beta, lambda_KH,
            A_gamma=A_GAMMA_PAPER, C_range=tuple(args.C_range),
        )
        print(f"\n--- equilibrium limit (e_KH = e_sat, no ODE) ---")
        print(f"  A_gamma/C = {eq_fit['ratio']:.5f}  (A_gamma={A_GAMMA_PAPER}, "
              f"C={eq_fit['C']:.4f})   L2={eq_fit['l2']:.4f}")

    coeffs_fitted = MInITCoefficients(
        beta_xx=beta["beta_xx"], beta_yy=beta["beta_yy"], beta_xy=beta["beta_xy"],
        A_gamma=fitted["A_gamma"], C=fitted["C"],
    )
    if args.normalize_beta:
        coeffs_fitted = coeffs_fitted.normalized()
        print(f"  beta normalized to trace residual 0: "
              f"xx={coeffs_fitted.beta_xx:.4f} yy={coeffs_fitted.beta_yy:.4f}")

    # --- Stage 4: a-priori scoring ---------------------------------------
    coeffs_paper = MInITCoefficients.paper()
    variants = {"fitted": coeffs_fitted, "paper": coeffs_paper}
    if eq_fit is not None:
        coeffs_eq = MInITCoefficients(
            beta_xx=coeffs_fitted.beta_xx, beta_yy=coeffs_fitted.beta_yy,
            beta_xy=coeffs_fitted.beta_xy, A_gamma=A_GAMMA_PAPER, C=eq_fit["C"],
        )
        variants["equilibrium"] = coeffs_eq
        coeffs_eq.save(Path(args.coeff_out).with_name(
            Path(args.coeff_out).stem + "_equilibrium.json"
        ))

    beta_vec = {
        name: torch.tensor([c.beta_xx, c.beta_yy, c.beta_xy], dtype=e_dns.dtype)
        for name, c in variants.items()
    }

    e_model = {}
    for name, c in variants.items():
        if name == "equilibrium":
            e_model[name] = saturation_energy(
                c.A_gamma * shear, sel["rho"], lambda_KH, c.C,
            )
            continue
        e_model[name] = evolve_e_series(
            e_dns[0], shear, sel["rho"], t,
            torch.tensor([c.A_gamma], dtype=e_dns.dtype),
            torch.tensor([c.C], dtype=e_dns.dtype),
            lambda_KH, args.substeps, vel,
        )[0]

    scores: dict[str, dict] = {}
    corr_series: list[tuple[str, np.ndarray, str]] = []
    print(f"\n--- A-priori scores against the exact SGS stress ---")
    print("    oracle_e : beta_ij e_KH^DNS      — the proportionality ansatz alone")
    print("    one_step : + one interval of the ODE, re-seeded from DNS each snapshot")
    print("    full     : + the ODE run free across the whole window")
    for name, c in variants.items():
        e_one = (
            e_model[name][1:] if name == "equilibrium"
            else one_step_series(
                e_dns, shear, sel["rho"], t, c.A_gamma, c.C, lambda_KH, args.substeps, vel,
            )
        )
        stages = {
            "oracle_e": (e_dns.unsqueeze(-1) * beta_vec[name], tau_exact),
            "one_step": (e_one.unsqueeze(-1) * beta_vec[name], tau_exact[1:]),
            "full": (e_model[name].unsqueeze(-1) * beta_vec[name], tau_exact),
        }
        scores[name] = {k: score_tau(*v) for k, v in stages.items()}

        print(f"  [{name}]")
        for stage in ("oracle_e", "one_step", "full"):
            sc = scores[name][stage]
            print(f"    {stage:9s} rel_L2={sc['rel_l2']:7.4f}  corr={sc['corr']:+.4f}  "
                  f"(xx {sc['corr_xx']:+.3f}, yy {sc['corr_yy']:+.3f}, xy {sc['corr_xy']:+.3f})")

        # Per-snapshot correlation of the free run: shows whether the model is
        # wrong from the start or drifts away from an initially good field.
        per_t = np.array([
            score_tau(
                e_model[name][k].unsqueeze(-1) * beta_vec[name], tau_exact[k],
            )["corr"]
            for k in range(len(t))
        ])
        scores[name]["corr_vs_time"] = per_t.tolist()
        corr_series.append((
            f"MInIT ({name})", per_t,
            {"fitted": "r--", "paper": "b:"}.get(name, "m-."),
        ))

    # Energy transfer: the a-posteriori-relevant quantity.
    S = sel["S"]
    pi_exact = energy_transfer(tau_exact, S)
    pi_mean_exact = pi_exact.mean(dim=(-2, -1))
    print(f"\n--- Energy transfer <Pi> = <tau^ij S_ij>, window mean ---")
    print(f"  exact  : {float(pi_exact.mean()):+.6e}")
    pi_curves = []
    for name in variants:
        pi_m = energy_transfer(e_model[name].unsqueeze(-1) * beta_vec[name], S)
        ratio = float(pi_m.mean() / pi_exact.mean()) if abs(float(pi_exact.mean())) > 0 else float("nan")
        print(f"  {name:7s}: {float(pi_m.mean()):+.6e}   ({ratio:+.3f} x exact)")
        scores[name]["pi_ratio"] = ratio
        pi_curves.append((
            f"MInIT ({name})", pi_m.mean(dim=(-2, -1)).numpy(),
            {"fitted": "r--", "paper": "b:"}.get(name, "m-."),
        ))

    # --- Persist ----------------------------------------------------------
    payload = {
        "window": {"t_lo": float(t[0]), "t_hi": float(t[-1]), "n_snapshots": len(t)},
        "grid": {"Nc_y": Nc_y, "Nc_x": Nc_x, "r": args.coarsening_factor,
                 "delta_f": delta_f, "lambda_KH": lambda_KH},
        "shear_mode": args.shear_mode,
        "beta_estimator": args.beta_estimator,
        "beta_least_squares": beta_lsq,
        "beta_block_ratio": beta_block,
        "beta_block_size": block,
        "trace_residual": trace_res,
        "fitted": {"A_gamma": fitted["A_gamma"], "C": fitted["C"], "l2": fitted["l2"]},
        "fitted_A_gamma_fixed": constrained,
        "equilibrium": eq_fit,
        "scan_boundary_pinned": scan["pinned"] if scan else [],
        "coefficients": {
            "beta_xx": coeffs_fitted.beta_xx,
            "beta_yy": coeffs_fitted.beta_yy,
            "beta_xy": coeffs_fitted.beta_xy,
            "A_gamma": coeffs_fitted.A_gamma,
            "C": coeffs_fitted.C,
        },
        "scores": scores,
    }
    coeffs_fitted.save(args.coeff_out)
    with open(out_dir / "minit_apriori.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {args.coeff_out} and {out_dir / 'minit_apriori.json'}")

    make_plots(out_dir, payload, {
        "t": t,
        "e_dns_mean": e_dns.mean(dim=(-2, -1)).numpy(),
        "e_model_curves": [
            (f"MInIT ({name})", e_model[name].mean(dim=(-2, -1)).numpy(),
             {"fitted": "r--", "paper": "b:"}.get(name, "m-."))
            for name in variants
        ],
        "surface": scan["surface"] if scan else None,
        "axis_A": scan["axis_A"] if scan else None,
        "axis_C": scan["axis_C"] if scan else None,
        "pi_exact": pi_mean_exact.numpy(),
        "pi_curves": pi_curves,
        "corr_series": corr_series,
        "corr_oracle": np.array([
            score_tau(e_dns[k].unsqueeze(-1) * beta_vec["fitted"], tau_exact[k])["corr"]
            for k in range(len(t))
        ]),
    })
    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
