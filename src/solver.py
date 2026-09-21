r"""
solver.py

GPU-accelerated 2D Relativistic Hydrodynamic (RHD) finite-volume solver
implemented entirely in PyTorch, matching the Aenus code (Obergaulinger 2008).

Numerical method:
  - 5th-order Monotonicity-Preserving (MP5) spatial reconstruction (Suresh & Huynh 1997)
    on primitive variables (with optional 2nd-order MUSCL)
  - HLL approximate Riemann solver (Harten 1983, with optional MUSTA-HLL and HLLC)
  - SSP-RK3 (Strong Stability Preserving Runge-Kutta, 3rd order) time integration
  - Newton-Raphson conservative-to-primitive variable recovery

Equations:
  ∂_t U + ∂_x F^x + ∂_y F^y = S_SGS

  U = (D, S_x, S_y, τ)^T
  D = ρ W           — relativistic mass density
  S_i = ρ h W² v_i  — momentum density
  τ = ρ h W² - p - D — energy density (minus rest mass)

Ideal gas EOS: p = (Γ - 1) ρ ε, h = 1 + ε + p/ρ (with Γ = 4/3)
"""

from __future__ import annotations

from collections.abc import Callable
import math
import warnings

import numpy as np
import torch

from src.dataset import GAMMA_EOS


# ---------------------------------------------------------------------------
# Primitive ↔ Conserved variable conversions
# ---------------------------------------------------------------------------

def primitives_to_conserved(
    rho: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    r"""Convert primitive variables (ρ, v_x, v_y, p) to conserved (D, S_x, S_y, τ).

    Returns
    -------
    U : (..., 4) tensor — conserved variables
    """
    v_sq = (vx**2 + vy**2).clamp(max=1.0 - 1e-12)
    W = 1.0 / torch.sqrt((1.0 - v_sq).clamp(min=1e-12))
    rho = rho.clamp(min=1e-15)
    p = p.clamp(min=1e-15)
    eps = p / ((GAMMA_EOS - 1.0) * rho)
    h = 1.0 + eps + p / rho

    D = rho * W
    rho_h_W2 = rho * h * W**2
    Sx = rho_h_W2 * vx
    Sy = rho_h_W2 * vy
    tau = rho_h_W2 - p - D

    return torch.stack([D, Sx, Sy, tau], dim=-1)


def conserved_to_primitives(
    U: torch.Tensor,
    p_guess: torch.Tensor | None = None,
    max_iter: int = 30,
    tol: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Recover primitive variables (ρ, v_x, v_y, p) from conserved (D, S_x, S_y, τ)
    using a Newton-Raphson iteration on the pressure.

    Uses the 1D Newton method on:
      f(p) = p - (Γ-1)(1 - D/W̃) (τ + D(1 - W̃) + p) / (Γ)

    Simplified approach following Mignone & McKinney (2007).

    Parameters
    ----------
    U : (..., 4) tensor — conserved variables [D, Sx, Sy, tau]
    p_guess : (...) tensor or None — initial pressure guess
    max_iter : int
    tol : float — convergence tolerance

    Returns
    -------
    rho, vx, vy, p : (...) tensors
    """
    D = U[..., 0].clamp(min=1e-15)
    Sx = U[..., 1]
    Sy = U[..., 2]
    tau = U[..., 3].clamp(min=1e-15)

    S_sq = Sx**2 + Sy**2
    gm1 = GAMMA_EOS - 1.0

    # Safe initial/fallback guess from non-relativistic limit
    p_safe = (gm1 * tau / GAMMA_EOS).clamp(min=1e-15)

    if p_guess is None:
        p = p_safe.clone()
    else:
        p = torch.where(
            torch.isfinite(p_guess) & (p_guess > 0),
            p_guess,
            p_safe,
        ).clamp(min=1e-15)

    for _ in range(max_iter):
        # v² = S² / (τ + D + p)²
        z = (tau + D + p).clamp(min=1e-15)
        v_sq = (S_sq / z**2).clamp(max=1.0 - 1e-12)

        W = 1.0 / torch.sqrt(1.0 - v_sq)
        rho = (D / W).clamp(min=1e-15)

        eps = (tau + D * (1.0 - W) + p * (1.0 - W**2)) / (D * W)
        eps = eps.clamp(min=1e-15)

        p_eos = gm1 * rho * eps

        # Residual
        f = p - p_eos

        # df/dp = 1 - (Γ-1) v² (1 - 1/h), with h = z/(D W) the specific enthalpy.
        # Bounded in (2/3, 1] for Γ = 4/3, so no conditioning guard is needed.
        h = z / (D * W)
        dfdp = 1.0 - gm1 * v_sq * (1.0 - 1.0 / h)

        dp = -f / dfdp
        # Damped Newton update: prevent negative overshoots and wild explosion
        dp = torch.where(
            torch.isfinite(dp),
            dp.clamp(-0.8 * p, 5.0 * p),
            torch.zeros_like(p),
        )
        p = (p + dp).clamp(min=1e-15)

        if bool((dp.abs() <= tol * p).all()):
            break

    # Non-convergence is deliberately NOT laundered here: a failed inversion must stay
    # non-finite so the per-step check in RHDSolver.evolve raises instead of silently
    # continuing on corrupted state.

    # Final recovery with velocity scaling to guarantee subluminal velocity
    z = (tau + D + p).clamp(min=1e-15)
    v_raw_sq = S_sq / (z**2 + 1e-30)
    v_clamped_sq = v_raw_sq.clamp(max=1.0 - 1e-12)
    scale = torch.sqrt(v_clamped_sq / (v_raw_sq + 1e-30)).clamp(max=1.0)
    w_final = 1.0 / torch.sqrt((1.0 - v_clamped_sq).clamp(min=1e-12))
    rho = (D / w_final).clamp(min=1e-15)
    vx = (Sx / z) * scale
    vy = (Sy / z) * scale

    # Physical bounds only; NaN/inf propagate by design (clamp preserves NaN).
    rho = rho.clamp(min=1e-15)
    p = p.clamp(min=1e-15)
    vx = vx.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    vy = vy.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    return rho, vx, vy, p


# ---------------------------------------------------------------------------
# Flux computation
# ---------------------------------------------------------------------------

def compute_flux(
    rho: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    p: torch.Tensor,
    direction: int = 0,
) -> torch.Tensor:
    r"""Compute the hydrodynamic flux F^d in direction d ∈ {0: x, 1: y}.

    F^x = (D v_x, S_x v_x + p, S_y v_x, (τ + p) v_x)
    F^y = (D v_y, S_x v_y, S_y v_y + p, (τ + p) v_y)

    Returns
    -------
    (..., 4) tensor
    """
    U = primitives_to_conserved(rho, vx, vy, p)
    D, Sx, Sy, tau = U[..., 0], U[..., 1], U[..., 2], U[..., 3]

    if direction == 0:
        v_d = vx
        return torch.stack([
            D * v_d,
            Sx * v_d + p,
            Sy * v_d,
            (tau + p) * v_d,
        ], dim=-1)
    else:
        v_d = vy
        return torch.stack([
            D * v_d,
            Sx * v_d,
            Sy * v_d + p,
            (tau + p) * v_d,
        ], dim=-1)


# ---------------------------------------------------------------------------
# Wave speed estimates
# ---------------------------------------------------------------------------

def compute_wave_speeds(
    rho: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    p: torch.Tensor,
    direction: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the maximum left- and right-going wave speeds.

    Uses the full multi-dimensional relativistic eigenvalue formula
    (Font et al. 1994, Aloy et al. 1999) that accounts for the transverse
    velocity component:

    .. math::

        \lambda_\pm = \frac{v_d(1 - c_s^2) \pm c_s\sqrt{(1 - v^2)
        (1 - v_d^2 - c_s^2 v_\perp^2)}}{1 - v^2 c_s^2}

    where :math:`v_\perp^2 = v^2 - v_d^2`.

    Returns
    -------
    lambda_minus, lambda_plus : (...) tensors
    """
    rho = rho.clamp(min=1e-15)
    p = p.clamp(min=1e-15)
    eps = p / ((GAMMA_EOS - 1.0) * rho)
    h = 1.0 + eps + p / rho
    cs2 = GAMMA_EOS * p / (rho * h)
    cs2 = cs2.clamp(min=0.0, max=1.0 - 1e-12)
    cs = torch.sqrt(cs2)

    v_sq = (vx**2 + vy**2).clamp(max=1.0 - 1e-12)
    v_d = vx if direction == 0 else vy
    v_perp_sq = (v_sq - v_d**2).clamp(min=0.0)  # transverse velocity squared

    # Full 2D discriminant — reduces to (1 - v_d²)² in the 1D limit
    disc = (1.0 - v_sq) * (1.0 - v_d**2 - cs2 * v_perp_sq)
    disc = disc.clamp(min=0.0)  # guard against numerical noise

    denom = (1.0 - v_sq * cs2).clamp(min=1e-12)
    num_base = v_d * (1.0 - cs2)
    cs_sqrt_disc = cs * torch.sqrt(disc)

    lambda_plus = ((num_base + cs_sqrt_disc) / denom).clamp(min=-1.0, max=1.0)
    lambda_minus = ((num_base - cs_sqrt_disc) / denom).clamp(min=-1.0, max=1.0)

    return lambda_minus, lambda_plus


# ---------------------------------------------------------------------------
# Limiters & Spatial Reconstruction
# ---------------------------------------------------------------------------

def minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise minmod limiter."""
    return 0.5 * (torch.sign(a) + torch.sign(b)) * torch.minimum(a.abs(), b.abs())


def minmod2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""Element-wise 2-argument minmod limiter."""
    s_x = torch.sign(x)
    same_sign = (s_x == torch.sign(y)) & (s_x != 0)
    min_abs = torch.minimum(x.abs(), y.abs())
    return torch.where(same_sign, s_x * min_abs, torch.zeros_like(x))


def minmod4(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    r"""Element-wise 4-argument minmod limiter (DM4 in Suresh & Huynh 1997)."""
    s_w = torch.sign(w)
    same_sign = (
        (s_w == torch.sign(x))
        & (s_w == torch.sign(y))
        & (s_w == torch.sign(z))
        & (s_w != 0)
    )
    min_abs = torch.minimum(
        torch.minimum(w.abs(), x.abs()),
        torch.minimum(y.abs(), z.abs()),
    )
    return torch.where(same_sign, s_w * min_abs, torch.zeros_like(w))


def mc_slope(dq_fwd: torch.Tensor, dq_bwd: torch.Tensor) -> torch.Tensor:
    r"""MC (monotonized central) limiter slope.

    .. math::

        \text{MC}(a, b) = \text{minmod}(2a,\; 2b,\; (a+b)/2)

    Less dissipative than minmod while remaining TVD.
    """
    c = 0.5 * (dq_fwd + dq_bwd)
    # s = ±1 when both arguments share sign, 0 otherwise
    s = 0.5 * (torch.sign(dq_fwd) + torch.sign(dq_bwd))
    return s * torch.minimum(
        torch.minimum(2.0 * dq_fwd.abs(), 2.0 * dq_bwd.abs()),
        c.abs(),
    )


def mp5_reconstruct(
    v: torch.Tensor,
    dim: int,
    alpha: float = 4.0,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""5th-order Monotonicity-Preserving (MP5) spatial reconstruction
    following Suresh & Huynh (1997).

    Preserves 5th-order accuracy at smooth extrema while enforcing strict
    monotonicity bounds near discontinuities.

    Parameters
    ----------
    v : (..., Ny, Nx) or (..., Ny, Nx, 4) tensor — field to reconstruct
    dim : int — spatial dimension (-1 for x, -2 for y in scalar fields; -2 for x, -3 for y in conserved fields)
    alpha : float — MP limiter parameter (default 4.0 as in Suresh & Huynh)
    eps : float — tolerance for fast monotonicity check (default 1e-10)

    Returns
    -------
    v_L, v_R : (..., Ny, Nx) tensors — left and right interface states at i+1/2
        v_L[i] = reconstructed value on left side of interface i+1/2
        v_R[i] = reconstructed value on right side of interface i+1/2
    """
    v_jm2 = torch.roll(v, 2, dims=dim)
    v_jm1 = torch.roll(v, 1, dims=dim)
    v_j = v
    v_jp1 = torch.roll(v, -1, dims=dim)
    v_jp2 = torch.roll(v, -2, dims=dim)
    v_jp3 = torch.roll(v, -3, dims=dim)

    def _reconstruct_left(
        jm2: torch.Tensor,
        jm1: torch.Tensor,
        j: torch.Tensor,
        jp1: torch.Tensor,
        jp2: torch.Tensor,
    ) -> torch.Tensor:
        b1 = 1.0 / 60.0
        b2 = 4.0 / 3.0

        # 5th-order unlimited polynomial value at interface i+1/2
        v_or = b1 * (2.0 * jm2 - 13.0 * jm1 + 47.0 * j + 27.0 * jp1 - 3.0 * jp2)

        # Original monotonicity-preserving limit
        v_mp = j + minmod2(jp1 - j, alpha * (j - jm1))

        # Curvature measures
        d_jm1 = jm2 - 2.0 * jm1 + j
        d_j = jm1 - 2.0 * j + jp1
        d_jp1 = j - 2.0 * jp1 + jp2

        dm4_jph = minmod4(4.0 * d_j - d_jp1, 4.0 * d_jp1 - d_j, d_j, d_jp1)
        dm4_jmh = minmod4(4.0 * d_j - d_jm1, 4.0 * d_jm1 - d_j, d_j, d_jm1)

        # Bounding values
        v_ul = j + alpha * (j - jm1)
        v_av = 0.5 * (j + jp1)
        v_md = v_av - 0.5 * dm4_jph
        v_lc = j + 0.5 * (j - jm1) + b2 * dm4_jmh

        v_min = torch.maximum(
            torch.minimum(torch.minimum(j, jp1), v_md),
            torch.minimum(torch.minimum(j, v_ul), v_lc),
        )
        v_max = torch.minimum(
            torch.maximum(torch.maximum(j, jp1), v_md),
            torch.maximum(torch.maximum(j, v_ul), v_lc),
        )

        v_lim = torch.clamp(v_or, min=v_min, max=v_max)
        cond = (v_or - j) * (v_or - v_mp) <= eps
        return torch.where(cond, v_or, v_lim)

    v_L = _reconstruct_left(v_jm2, v_jm1, v_j, v_jp1, v_jp2)
    v_R = _reconstruct_left(v_jp3, v_jp2, v_jp1, v_j, v_jm1)

    return v_L, v_R


def muscl_reconstruct(
    q: torch.Tensor,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""2nd-order MUSCL reconstruction with the MC (monotonized central) limiter along `dim`.

    Parameters
    ----------
    q : (..., Ny, Nx) or (..., Ny, Nx, 4) tensor — cell-centered variables
    dim : int — spatial dimension (-1 for x, -2 for y in scalar fields; -2 for x, -3 for y in conserved fields)

    Returns
    -------
    q_L, q_R : tensors of same shape as q — left/right interface states
        q_L[i] = q[i] + 0.5 * slope[i]      (left state at i+1/2)
        q_R[i] = q[i+1] - 0.5 * slope[i+1]  (right state at i+1/2)
    """
    # Forward/backward differences
    dq_fwd = torch.roll(q, -1, dims=dim) - q
    dq_bwd = q - torch.roll(q, 1, dims=dim)

    slope = mc_slope(dq_fwd, dq_bwd)

    # Interface states
    q_L = q + 0.5 * slope                                       # left state at i+1/2
    q_R = torch.roll(q, -1, dims=dim) - 0.5 * torch.roll(slope, -1, dims=dim)  # right state at i+1/2

    return q_L, q_R


# ---------------------------------------------------------------------------
# HLL Riemann Solver
# ---------------------------------------------------------------------------

def hll_flux(
    rho_L: torch.Tensor, vx_L: torch.Tensor, vy_L: torch.Tensor, p_L: torch.Tensor,
    rho_R: torch.Tensor, vx_R: torch.Tensor, vy_R: torch.Tensor, p_R: torch.Tensor,
    direction: int = 0,
) -> torch.Tensor:
    r"""HLL approximate Riemann solver.

    F_HLL = (s_R F_L - s_L F_R + s_L s_R (U_R - U_L)) / (s_R - s_L)

    Returns
    -------
    (..., 4) tensor — numerical flux at the interface
    """
    # Wave speeds from left and right states
    lam_minus_L, lam_plus_L = compute_wave_speeds(rho_L, vx_L, vy_L, p_L, direction)
    lam_minus_R, lam_plus_R = compute_wave_speeds(rho_R, vx_R, vy_R, p_R, direction)

    s_L = torch.minimum(lam_minus_L, lam_minus_R)  # fastest left-going
    s_R = torch.maximum(lam_plus_L, lam_plus_R)     # fastest right-going

    # Fluxes and conserved states
    F_L = compute_flux(rho_L, vx_L, vy_L, p_L, direction)
    F_R = compute_flux(rho_R, vx_R, vy_R, p_R, direction)
    U_L = primitives_to_conserved(rho_L, vx_L, vy_L, p_L)
    U_R = primitives_to_conserved(rho_R, vx_R, vy_R, p_R)

    s_L_4 = s_L.unsqueeze(-1)
    s_R_4 = s_R.unsqueeze(-1)

    # HLL flux
    F_hll = (s_R_4 * F_L - s_L_4 * F_R + s_L_4 * s_R_4 * (U_R - U_L)) / (s_R_4 - s_L_4 + 1e-30)

    # Upwind selection
    mask_L = (s_L >= 0).unsqueeze(-1)
    mask_R = (s_R <= 0).unsqueeze(-1)

    flux = torch.where(mask_L, F_L, torch.where(mask_R, F_R, F_hll))
    return flux


def hllc_flux(
    rho_L: torch.Tensor, vx_L: torch.Tensor, vy_L: torch.Tensor, p_L: torch.Tensor,
    rho_R: torch.Tensor, vx_R: torch.Tensor, vy_R: torch.Tensor, p_R: torch.Tensor,
    direction: int = 0,
) -> torch.Tensor:
    r"""HLLC approximate Riemann solver for relativistic hydrodynamics.

    Restores the contact discontinuity missing from standard HLL, sharply capturing
    density jumps and shear waves. Follows Mignone & Bodo (2005), which is formulated
    in the total energy density E = τ + D (whose flux is exactly the normal momentum);
    the star states are converted back to the τ convention used throughout this module.

    Parameters
    ----------
    rho_L, vx_L, vy_L, p_L : tensors — left primitive states
    rho_R, vx_R, vy_R, p_R : tensors — right primitive states
    direction : 0 for x-flux, 1 for y-flux

    Returns
    -------
    (..., 4) tensor — numerical flux at interface
    """
    lam_minus_L, lam_plus_L = compute_wave_speeds(rho_L, vx_L, vy_L, p_L, direction)
    lam_minus_R, lam_plus_R = compute_wave_speeds(rho_R, vx_R, vy_R, p_R, direction)

    s_L = torch.minimum(lam_minus_L, lam_minus_R)
    s_R = torch.maximum(lam_plus_L, lam_plus_R)

    F_L = compute_flux(rho_L, vx_L, vy_L, p_L, direction)
    F_R = compute_flux(rho_R, vx_R, vy_R, p_R, direction)
    U_L = primitives_to_conserved(rho_L, vx_L, vy_L, p_L)
    U_R = primitives_to_conserved(rho_R, vx_R, vy_R, p_R)

    n = 1 if direction == 0 else 2  # normal momentum component

    # Total energy E = τ + D; its flux is exactly the normal momentum density.
    E_L = U_L[..., 3] + U_L[..., 0]
    E_R = U_R[..., 3] + U_R[..., 0]
    FE_L = F_L[..., 3] + F_L[..., 0]
    FE_R = F_R[..., 3] + F_R[..., 0]
    mn_L, mn_R = U_L[..., n], U_R[..., n]
    Fmn_L, Fmn_R = F_L[..., n], F_R[..., n]

    # HLL averages, less the common 1/(s_R - s_L) which cancels in the quadratic.
    E_hat = s_R * E_R - s_L * E_L + FE_L - FE_R
    m_hat = s_R * mn_R - s_L * mn_L + Fmn_L - Fmn_R
    FE_hat = s_R * FE_L - s_L * FE_R + s_L * s_R * (E_R - E_L)
    Fm_hat = s_R * Fmn_L - s_L * Fmn_R + s_L * s_R * (mn_R - mn_L)

    # F̂_E v*² - (Ê + F̂_m) v* + m̂ = 0, minus root written in the form that stays
    # well conditioned as F̂_E → 0.
    b = -(E_hat + Fm_hat)
    disc = torch.clamp(b * b - 4.0 * FE_hat * m_hat, min=0.0)
    q = -b + torch.sqrt(disc)
    s_star = (2.0 * m_hat / torch.where(q.abs() > 1e-30, q, torch.full_like(q, 1e-30))).clamp(
        min=-1.0 + 1e-12, max=1.0 - 1e-12
    )

    den = (s_R - s_L).clamp(min=1e-30)
    p_star = ((Fm_hat - FE_hat * s_star) / den).clamp(min=1e-15)

    def _star_state(U_k: torch.Tensor, F_k: torch.Tensor, s_k: torch.Tensor) -> torch.Tensor:
        denom = s_k - s_star
        denom = torch.where(denom.abs() > 1e-30, denom, torch.full_like(denom, 1e-30))

        E_k = U_k[..., 3] + U_k[..., 0]
        FE_k = F_k[..., 3] + F_k[..., 0]

        D_star = (s_k * U_k[..., 0] - F_k[..., 0]) / denom
        E_star = (s_k * E_k - FE_k + p_star * s_star) / denom

        mx_star = (s_k * U_k[..., 1] - F_k[..., 1] + (p_star if direction == 0 else 0.0)) / denom
        my_star = (s_k * U_k[..., 2] - F_k[..., 2] + (p_star if direction == 1 else 0.0)) / denom

        return torch.stack([D_star, mx_star, my_star, E_star - D_star], dim=-1)

    s_star_4 = s_star.unsqueeze(-1)
    s_L_4 = s_L.unsqueeze(-1)
    s_R_4 = s_R.unsqueeze(-1)

    F_star_L = F_L + s_L_4 * (_star_state(U_L, F_L, s_L) - U_L)
    F_star_R = F_R + s_R_4 * (_star_state(U_R, F_R, s_R) - U_R)

    flux_hllc = torch.where(
        s_L_4 >= 0,
        F_L,
        torch.where(
            s_star_4 >= 0,
            F_star_L,
            torch.where(
                s_R_4 >= 0,
                F_star_R,
                F_R,
            ),
        ),
    )

    # Tensor-level fallback to HLL; a host-side .all() check here would sync the device
    # at every interface of every RK sub-stage.
    F_hll = (s_R_4 * F_L - s_L_4 * F_R + s_L_4 * s_R_4 * (U_R - U_L)) / (s_R_4 - s_L_4 + 1e-30)
    return torch.where(torch.isfinite(flux_hllc), flux_hllc, F_hll)


def musta_hll_flux(
    rho_L: torch.Tensor, vx_L: torch.Tensor, vy_L: torch.Tensor, p_L: torch.Tensor,
    rho_R: torch.Tensor, vx_R: torch.Tensor, vy_R: torch.Tensor, p_R: torch.Tensor,
    direction: int = 0,
    n_stages: int = 1,
    c_musta: float = 0.5,
) -> torch.Tensor:
    r"""Multi-Stage (MUSTA) Riemann solver based on the HLL flux (Toro & Titarev 2006).

    Parameters
    ----------
    n_stages : int — number of local predictor stages (default 1)
    c_musta : float — local CFL parameter (default 0.5)

    Returns
    -------
    (..., 4) tensor — numerical flux at interface
    """
    if n_stages <= 0:
        return hll_flux(rho_L, vx_L, vy_L, p_L, rho_R, vx_R, vy_R, p_R, direction=direction)

    curr_rho_L, curr_vx_L, curr_vy_L, curr_p_L = rho_L, vx_L, vy_L, p_L
    curr_rho_R, curr_vx_R, curr_vy_R, curr_p_R = rho_R, vx_R, vy_R, p_R

    u_l_state = primitives_to_conserved(curr_rho_L, curr_vx_L, curr_vy_L, curr_p_L)
    u_r_state = primitives_to_conserved(curr_rho_R, curr_vx_R, curr_vy_R, curr_p_R)

    for _ in range(n_stages):
        lam_m_L, lam_p_L = compute_wave_speeds(curr_rho_L, curr_vx_L, curr_vy_L, curr_p_L, direction)
        lam_m_R, lam_p_R = compute_wave_speeds(curr_rho_R, curr_vx_R, curr_vy_R, curr_p_R, direction)
        s_L = torch.minimum(lam_m_L, lam_m_R)
        s_R = torch.maximum(lam_p_L, lam_p_R)
        s_max = torch.maximum(s_L.abs(), s_R.abs()).clamp(min=1e-6)

        F_hll = hll_flux(curr_rho_L, curr_vx_L, curr_vy_L, curr_p_L,
                         curr_rho_R, curr_vx_R, curr_vy_R, curr_p_R,
                         direction=direction)
        F_L = compute_flux(curr_rho_L, curr_vx_L, curr_vy_L, curr_p_L, direction=direction)
        F_R = compute_flux(curr_rho_R, curr_vx_R, curr_vy_R, curr_p_R, direction=direction)

        dt_dx = (c_musta / s_max).unsqueeze(-1)
        u_l_state = u_l_state - dt_dx * (F_hll - F_L)
        u_r_state = u_r_state - dt_dx * (F_R - F_hll)

        curr_rho_L, curr_vx_L, curr_vy_L, curr_p_L = conserved_to_primitives(u_l_state, p_guess=curr_p_L)
        curr_rho_R, curr_vx_R, curr_vy_R, curr_p_R = conserved_to_primitives(u_r_state, p_guess=curr_p_R)

    return hll_flux(curr_rho_L, curr_vx_L, curr_vy_L, curr_p_L,
                    curr_rho_R, curr_vx_R, curr_vy_R, curr_p_R,
                    direction=direction)


def _evaluate_interface_flux(
    rho_L: torch.Tensor, vx_L: torch.Tensor, vy_L: torch.Tensor, p_L: torch.Tensor,
    rho_R: torch.Tensor, vx_R: torch.Tensor, vy_R: torch.Tensor, p_R: torch.Tensor,
    direction: int,
    riemann_solver: str,
) -> torch.Tensor:
    """Helper to dispatch interface flux evaluation."""
    if riemann_solver == "hll":
        return hll_flux(rho_L, vx_L, vy_L, p_L, rho_R, vx_R, vy_R, p_R, direction=direction)
    elif riemann_solver == "musta_hll":
        return musta_hll_flux(rho_L, vx_L, vy_L, p_L, rho_R, vx_R, vy_R, p_R, direction=direction)
    elif riemann_solver == "hllc":
        return hllc_flux(rho_L, vx_L, vy_L, p_L, rho_R, vx_R, vy_R, p_R, direction=direction)
    else:
        raise ValueError(f"Unknown riemann_solver: '{riemann_solver}'. Expected 'hll', 'musta_hll', or 'hllc'.")


# ---------------------------------------------------------------------------
# Single RK Stage
# ---------------------------------------------------------------------------

_SGS_RECOVERY_ITERS = 2


def _face_interp4(f: torch.Tensor, dim: int) -> torch.Tensor:
    r"""4th-order face interpolation f_{i+1/2} = (-f_{i-1} + 9 f_i + 9 f_{i+1} - f_{i+2}) / 16."""
    return (
        -torch.roll(f, 1, dims=dim)
        + 9.0 * f
        + 9.0 * torch.roll(f, -1, dims=dim)
        - torch.roll(f, -2, dims=dim)
    ) / 16.0


def compute_rhs(
    prims: torch.Tensor,
    dx: float,
    dy: float,
    sgs_source: torch.Tensor | None = None,
    reconstruction: str = "mp5",
    riemann_solver: str = "hll",
    reconstruct_primitives: bool = True,
    sgs_viscous_Fx: torch.Tensor | None = None,
    sgs_viscous_Fy: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Compute the right-hand side: -∂_x F^x - ∂_y F^y + S_SGS.

    Parameters
    ----------
    prims : (Ny, Nx, 4) tensor — primitive variables [ρ, vx, vy, p]
    dx, dy : grid spacings
    sgs_source : (Ny, Nx, 4) tensor or None — SGS source term (legacy path)
    reconstruction : str — "mp5" (default) or "muscl"
    riemann_solver : str — "hll" (default), "musta_hll", or "hllc"
    reconstruct_primitives : bool — if True (default, matching Aenus),
        reconstruct primitive variables directly; if False, reconstruct
        conserved variables.
    sgs_viscous_Fx : (Ny, Nx, 4) tensor or None — viscous SGS x-flux
        correction δF^x from the conservative flux formulation.
    sgs_viscous_Fy : (Ny, Nx, 4) tensor or None — viscous SGS y-flux
        correction δF^y from the conservative flux formulation.

    Returns
    -------
    (Ny, Nx, 4) tensor — dU/dt
    """
    rho = prims[..., 0]
    vx_p = prims[..., 1]
    vy_p = prims[..., 2]
    p = prims[..., 3]

    Ny = prims.shape[-3]

    if reconstruct_primitives:
        recon_fn = mp5_reconstruct if reconstruction == "mp5" else muscl_reconstruct

        # --- X-direction (dim=-2 for prims of shape (..., Ny, Nx, 4)) ---
        prims_Lx, prims_Rx = recon_fn(prims, dim=-2)
        rho_Lx, vx_Lx, vy_Lx, p_Lx = prims_Lx[..., 0], prims_Lx[..., 1], prims_Lx[..., 2], prims_Lx[..., 3]
        rho_Rx, vx_Rx, vy_Rx, p_Rx = prims_Rx[..., 0], prims_Rx[..., 1], prims_Rx[..., 2], prims_Rx[..., 3]

        # Physical sanity guards at interface
        rho_Lx = rho_Lx.clamp(min=1e-15)
        rho_Rx = rho_Rx.clamp(min=1e-15)
        p_Lx = p_Lx.clamp(min=1e-15)
        p_Rx = p_Rx.clamp(min=1e-15)

        v_sq_Lx = (vx_Lx**2 + vy_Lx**2).clamp(max=1.0 - 1e-12)
        scale_Lx = torch.sqrt(v_sq_Lx / (vx_Lx**2 + vy_Lx**2 + 1e-30)).clamp(max=1.0)
        vx_Lx = vx_Lx * scale_Lx
        vy_Lx = vy_Lx * scale_Lx

        v_sq_Rx = (vx_Rx**2 + vy_Rx**2).clamp(max=1.0 - 1e-12)
        scale_Rx = torch.sqrt(v_sq_Rx / (vx_Rx**2 + vy_Rx**2 + 1e-30)).clamp(max=1.0)
        vx_Rx = vx_Rx * scale_Rx
        vy_Rx = vy_Rx * scale_Rx

        Fx = _evaluate_interface_flux(
            rho_Lx, vx_Lx, vy_Lx, p_Lx,
            rho_Rx, vx_Rx, vy_Rx, p_Rx,
            direction=0,
            riemann_solver=riemann_solver,
        )
        dFx = (Fx - torch.roll(Fx, 1, dims=-2)) / dx

        # --- Y-direction (dim=-3 for prims of shape (..., Ny, Nx, 4)) ---
        if Ny > 1:
            prims_Ly, prims_Ry = recon_fn(prims, dim=-3)
            rho_Ly, vx_Ly, vy_Ly, p_Ly = prims_Ly[..., 0], prims_Ly[..., 1], prims_Ly[..., 2], prims_Ly[..., 3]
            rho_Ry, vx_Ry, vy_Ry, p_Ry = prims_Ry[..., 0], prims_Ry[..., 1], prims_Ry[..., 2], prims_Ry[..., 3]

            rho_Ly = rho_Ly.clamp(min=1e-15)
            rho_Ry = rho_Ry.clamp(min=1e-15)
            p_Ly = p_Ly.clamp(min=1e-15)
            p_Ry = p_Ry.clamp(min=1e-15)

            v_sq_Ly = (vx_Ly**2 + vy_Ly**2).clamp(max=1.0 - 1e-12)
            scale_Ly = torch.sqrt(v_sq_Ly / (vx_Ly**2 + vy_Ly**2 + 1e-30)).clamp(max=1.0)
            vx_Ly = vx_Ly * scale_Ly
            vy_Ly = vy_Ly * scale_Ly

            v_sq_Ry = (vx_Ry**2 + vy_Ry**2).clamp(max=1.0 - 1e-12)
            scale_Ry = torch.sqrt(v_sq_Ry / (vx_Ry**2 + vy_Ry**2 + 1e-30)).clamp(max=1.0)
            vx_Ry = vx_Ry * scale_Ry
            vy_Ry = vy_Ry * scale_Ry

            Fy = _evaluate_interface_flux(
                rho_Ly, vx_Ly, vy_Ly, p_Ly,
                rho_Ry, vx_Ry, vy_Ry, p_Ry,
                direction=1,
                riemann_solver=riemann_solver,
            )
            dFy = (Fy - torch.roll(Fy, 1, dims=-3)) / dy
        else:
            dFy = torch.zeros_like(dFx)

    else:
        # Conserved variable reconstruction (fallback)
        U = primitives_to_conserved(rho, vx_p, vy_p, p)
        recon_fn = mp5_reconstruct if reconstruction == "mp5" else muscl_reconstruct

        # U has shape (..., Ny, Nx, 4). Spatial x-axis is dim=-2.
        U_Lx, U_Rx = recon_fn(U, dim=-2)
        rho_Lx, vx_Lx, vy_Lx, p_Lx = conserved_to_primitives(U_Lx, p_guess=p)
        rho_Rx, vx_Rx, vy_Rx, p_Rx = conserved_to_primitives(U_Rx, p_guess=p)

        Fx = _evaluate_interface_flux(
            rho_Lx, vx_Lx, vy_Lx, p_Lx,
            rho_Rx, vx_Rx, vy_Rx, p_Rx,
            direction=0,
            riemann_solver=riemann_solver,
        )
        dFx = (Fx - torch.roll(Fx, 1, dims=-2)) / dx

        if Ny > 1:
            # Spatial y-axis in U is dim=-3.
            U_Ly, U_Ry = recon_fn(U, dim=-3)
            rho_Ly, vx_Ly, vy_Ly, p_Ly = conserved_to_primitives(U_Ly, p_guess=p)
            rho_Ry, vx_Ry, vy_Ry, p_Ry = conserved_to_primitives(U_Ry, p_guess=p)

            Fy = _evaluate_interface_flux(
                rho_Ly, vx_Ly, vy_Ly, p_Ly,
                rho_Ry, vx_Ry, vy_Ry, p_Ry,
                direction=1,
                riemann_solver=riemann_solver,
            )
            dFy = (Fy - torch.roll(Fy, 1, dims=-3)) / dy
        else:
            dFy = torch.zeros_like(dFx)

    # RHS — ideal (inviscid) flux divergence
    rhs = -(dFx + dFy)

    # Legacy source-term SGS coupling
    if sgs_source is not None:
        rhs = rhs + sgs_source

    # Conservative viscous flux SGS coupling (frame-consistent path).
    # 4th-order face interpolation keeps the SGS flux divergence closer in order to the
    # MP5 ideal flux; a 2-point face average would degrade it to a 2Δ central difference.
    if sgs_viscous_Fx is not None:
        face_Fx = _face_interp4(sgs_viscous_Fx, dim=-2)
        rhs = rhs - (face_Fx - torch.roll(face_Fx, 1, dims=-2)) / dx

    if sgs_viscous_Fy is not None and Ny > 1:
        face_Fy = _face_interp4(sgs_viscous_Fy, dim=-3)
        rhs = rhs - (face_Fy - torch.roll(face_Fy, 1, dims=-3)) / dy

    return rhs


# ---------------------------------------------------------------------------
# Transition Sensor
# ---------------------------------------------------------------------------

def compute_transition_sensor(
    prims: torch.Tensor,
    dx: float,
    dy: float,
    mode: str = "none",
) -> torch.Tensor:
    r"""Compute a local SGS gating field α(x,y) ∈ [0, 1].

    Parameters
    ----------
    prims : (Ny, Nx, 4) tensor — [ρ, vx, vy, p]
    dx, dy : grid spacings
    mode : str — sensor type:
        "none" (default): α ≡ 1, the closure is applied unmodified. Required whenever
            the supplied stress is already exact (an a-posteriori DNS reference run must
            not rescale ground truth).
        "enstrophy": α = |ω|² / (|ω|² + |S|² + ε), the rotation-to-total fraction. This
            separates rotation-dominated from strain-dominated regions. It does NOT
            detect laminar-to-turbulent transition: any pure shear has |ω|² = |S|², so
            α ≡ ½ throughout a laminar shear layer.
        "germano": dynamic Germano coefficient via test-filter at 2Δ, normalised against
            a fixed reference c_s² = 0.04 (c_s ≈ 0.2).

    Returns
    -------
    alpha : (Ny, Nx) tensor — gating field ∈ [0, 1]
    """
    vx = prims[..., 1]
    vy = prims[..., 2]

    if mode == "none":
        return torch.ones_like(vx)

    # 2nd-order central gradients (cheap, sufficient for sensor)
    dvx_dx = (torch.roll(vx, -1, dims=-1) - torch.roll(vx, 1, dims=-1)) / (2.0 * dx)
    dvx_dy = (torch.roll(vx, -1, dims=-2) - torch.roll(vx, 1, dims=-2)) / (2.0 * dy)
    dvy_dx = (torch.roll(vy, -1, dims=-1) - torch.roll(vy, 1, dims=-1)) / (2.0 * dx)
    dvy_dy = (torch.roll(vy, -1, dims=-2) - torch.roll(vy, 1, dims=-2)) / (2.0 * dy)

    if mode == "enstrophy":
        # Vorticity magnitude squared: |ω|² = (∂_x v_y - ∂_y v_x)²
        omega_sq = (dvy_dx - dvx_dy) ** 2

        # Strain rate magnitude squared: |S|² = 2 S_{ij} S_{ij}
        # S_xx = ∂_x v_x, S_yy = ∂_y v_y, S_xy = 0.5(∂_y v_x + ∂_x v_y)
        S_sq = 2.0 * (dvx_dx**2 + dvy_dy**2 + 0.5 * (dvx_dy + dvy_dx)**2)

        eps = 1e-30
        alpha = omega_sq / (omega_sq + S_sq + eps)

    elif mode == "germano":
        # Dynamic Germano coefficient: compare SGS stress at Δ and 2Δ.
        # Test filter at 2Δ using simple box averaging of 3×3 neighbours.
        def _box_filter(f: torch.Tensor) -> torch.Tensor:
            """3×3 box filter (periodic)."""
            acc = f.clone()
            for shift in [-1, 1]:
                acc = acc + torch.roll(f, shift, dims=-1)
                acc = acc + torch.roll(f, shift, dims=-2)
            # Corner contributions
            for sy in [-1, 1]:
                for sx in [-1, 1]:
                    acc = acc + torch.roll(torch.roll(f, sy, dims=-2), sx, dims=-1)
            return acc / 9.0

        # Leonard stress: L_ij = \overline{v_i v_j} - \bar{v}_i \bar{v}_j
        vx_bar = _box_filter(vx)
        vy_bar = _box_filter(vy)

        L_xx = _box_filter(vx * vx) - vx_bar * vx_bar
        L_yy = _box_filter(vy * vy) - vy_bar * vy_bar
        L_xy = _box_filter(vx * vy) - vx_bar * vy_bar

        # Test-level strain magnitude |S̄|
        dvx_bar_dx = (torch.roll(vx_bar, -1, dims=-1) - torch.roll(vx_bar, 1, dims=-1)) / (2.0 * dx)
        dvx_bar_dy = (torch.roll(vx_bar, -1, dims=-2) - torch.roll(vx_bar, 1, dims=-2)) / (2.0 * dy)
        dvy_bar_dx = (torch.roll(vy_bar, -1, dims=-1) - torch.roll(vy_bar, 1, dims=-1)) / (2.0 * dx)
        dvy_bar_dy = (torch.roll(vy_bar, -1, dims=-2) - torch.roll(vy_bar, 1, dims=-2)) / (2.0 * dy)
        S_bar_mag = torch.sqrt(2.0 * (dvx_bar_dx**2 + dvy_bar_dy**2 + 0.5 * (dvx_bar_dy + dvy_bar_dx)**2) + 1e-30)

        # Grid-level strain magnitude |S|
        S_mag = torch.sqrt(2.0 * (dvx_dx**2 + dvy_dy**2 + 0.5 * (dvx_dy + dvy_dx)**2) + 1e-30)

        # M_ij = 2 Δ² (|S̄| S̄_ij - 4 |S| \overline{S_ij})
        # (factor of 4 = (2Δ/Δ)² from the test filter width ratio)
        S_bar_xx = dvx_bar_dx
        S_bar_yy = dvy_bar_dy
        S_bar_xy = 0.5 * (dvx_bar_dy + dvy_bar_dx)

        S_xx = dvx_dx
        S_yy = dvy_dy
        S_xy = 0.5 * (dvx_dy + dvy_dx)

        Delta_sq = dx * dy  # grid filter area
        M_xx = 2.0 * Delta_sq * (S_bar_mag * S_bar_xx - 4.0 * _box_filter(S_mag * S_xx))
        M_yy = 2.0 * Delta_sq * (S_bar_mag * S_bar_yy - 4.0 * _box_filter(S_mag * S_yy))
        M_xy = 2.0 * Delta_sq * (S_bar_mag * S_bar_xy - 4.0 * _box_filter(S_mag * S_xy))

        # Germano coefficient: c_s² = <L_ij M_ij> / <M_ij M_ij>
        # Local (no spatial averaging of numerator/denominator for simplicity)
        LM = L_xx * M_xx + L_yy * M_yy + 2.0 * L_xy * M_xy
        MM = M_xx**2 + M_yy**2 + 2.0 * M_xy**2

        c_s_sq = (LM / (MM + 1e-30)).clamp(min=0.0)
        # Fixed reference rather than a global max: normalising by c_s_sq.max() lets one
        # outlier cell rescale α across the whole grid (and forces a device sync).
        alpha = (c_s_sq / 0.04).clamp(min=0.0, max=1.0)

    else:
        raise ValueError(
            f"Unknown sensor mode: '{mode}'. Expected 'none', 'enstrophy', or 'germano'."
        )

    return alpha


# ---------------------------------------------------------------------------
# SGS Tensor Realizability Clipping
# ---------------------------------------------------------------------------

def clip_sgs_tensor(
    tau_perp: torch.Tensor,
    prims: torch.Tensor,
    mode: str = "frobenius",
    kappa: float = 0.5,
) -> torch.Tensor:
    r"""Clip the SGS stress tensor for realizability.

    Ensures the subgrid stress remains bounded relative to the resolved
    pressure, preventing numerical instabilities during primitive
    variable recovery.

    Parameters
    ----------
    tau_perp : (Ny, Nx, 4, 4) tensor — fluid-frame SGS stress
    prims : (Ny, Nx, 4) tensor — [ρ, vx, vy, p]
    mode : str — clipping strategy:
        "frobenius" (default): clip ‖τ_⊥‖_F ≤ κ·p (isotropic)
        "eigenvalue": clip each spatial eigenvalue independently
    kappa : float — clipping threshold as fraction of local pressure

    Returns
    -------
    (Ny, Nx, 4, 4) tensor — clipped τ_⊥
    """
    p = prims[..., 3].clamp(min=1e-15)
    max_val = kappa * p  # (Ny, Nx)

    if mode == "frobenius":
        # ‖τ_⊥‖_F = sqrt(Σ τ_{μν}²)
        norm = torch.norm(tau_perp, dim=(-2, -1))  # (Ny, Nx)
        scale = torch.where(
            norm > max_val,
            max_val / norm.clamp(min=1e-30),
            torch.ones_like(norm),
        )
        return tau_perp * scale.unsqueeze(-1).unsqueeze(-1)

    elif mode == "eigenvalue":
        # Full 3×3 spatial block. τ^{33} is generally nonzero even with v_z = 0 and ∂_z = 0,
        # because the basis tensors are 4D-traceless, so it must be clipped too.
        tau_spatial = tau_perp[..., 1:4, 1:4]  # (Ny, Nx, 3, 3)

        eigenvalues, eigenvectors = torch.linalg.eigh(tau_spatial)  # (Ny, Nx, 3), (Ny, Nx, 3, 3)

        # Clamp each eigenvalue: |λ_i| ≤ κ·p
        max_val_2 = max_val.unsqueeze(-1)  # (Ny, Nx, 1)
        eigenvalues_clamped = eigenvalues.clamp(-max_val_2, max_val_2)

        # Reconstruct: τ_spatial = V diag(λ_clamped) V^T
        tau_spatial_clipped = torch.einsum(
            "...ij,...j,...kj->...ik",
            eigenvectors, eigenvalues_clamped, eigenvectors,
        )

        # Rebuild full 4×4 tensor with clipped spatial block
        tau_clipped = tau_perp.clone()
        tau_clipped[..., 1:4, 1:4] = tau_spatial_clipped

        # Reconstruct time components from transversality:
        # τ^{0ν} = v_j τ^{jν}
        vx = prims[..., 1]
        vy = prims[..., 2]
        for nu in range(4):
            tau_clipped[..., 0, nu] = (
                vx * tau_clipped[..., 1, nu] + vy * tau_clipped[..., 2, nu]
            )
            # Symmetry
            tau_clipped[..., nu, 0] = tau_clipped[..., 0, nu]

        return tau_clipped

    else:
        raise ValueError(f"Unknown clip mode: '{mode}'. Expected 'frobenius' or 'eigenvalue'.")


def smooth_sgs_tensor(tau_perp: torch.Tensor, n_passes: int = 1) -> torch.Tensor:
    r"""Low-pass the SGS stress with an (1,2,1)/4 filter in each direction.

    The SGS flux divergence passes 3Δ–4Δ content at 57–80% but annihilates the 2Δ mode
    exactly, so the closure can inject energy near the grid scale without any ability to
    remove it there. That one-way valve is what lets an otherwise well-behaved learned
    stress drive a grid-scale instability. Filtering τ closes it while leaving the
    large-scale backscatter — the physically meaningful part in 2D — essentially intact.

    Transfer per pass: 0.99 at 32Δ, 0.96 at 16Δ, 0.85 at 8Δ, 0.50 at 4Δ, 0.25 at 3Δ,
    0.00 at 2Δ.

    Parameters
    ----------
    tau_perp : (Ny, Nx, 4, 4) tensor
    n_passes : int — filter applications; 0 returns the input unchanged
    """
    out = tau_perp
    for _ in range(n_passes):
        for dim in (-3, -4):  # y and x axes of an (Ny, Nx, 4, 4) tensor
            out = 0.25 * (
                torch.roll(out, 1, dims=dim) + 2.0 * out + torch.roll(out, -1, dims=dim)
            )
    return out


def clip_backscatter(
    tau_perp: torch.Tensor,
    prims: torch.Tensor,
    dx: float,
    dy: float,
    mode: str = "zero",
    fraction: float = 0.0,
) -> torch.Tensor:
    r"""Suppress local energy transfer from subgrid to resolved scales.

    The resolved kinetic energy gains energy at rate :math:`\Pi = \tau^{ij} S_{ij}`, so
    cells with :math:`\Pi > 0` inject energy (backscatter). Backscatter is physical --
    in 2D it is arguably the dominant transfer direction -- but it is unbounded here,
    and an unconstrained learned closure can use it to drive a grid-scale instability.
    Clipping it trades that physics for a guarantee of net dissipation.

    Parameters
    ----------
    tau_perp : (Ny, Nx, 4, 4) — fluid-frame SGS stress
    prims : (Ny, Nx, 4) — [ρ, vx, vy, p]
    dx, dy : grid spacings
    mode : str
        "zero"    — null the whole tensor wherever Π > 0 (strictest; clearest diagnostic)
        "project" — subtract only the strain-aligned part responsible for the
                    backscatter, leaving the rest of the tensor intact
    fraction : float in [0, 1] — how much backscatter to *retain*. 0.0 removes it
        entirely (the strict clip), 1.0 is a no-op. Intermediate values trade stability
        against the inverse-cascade physics, which matters in 2D where backscatter is
        not a small correction.

    Returns
    -------
    (Ny, Nx, 4, 4) tensor
    """
    if fraction >= 1.0:
        return tau_perp
    vx = prims[..., 1]
    vy = prims[..., 2]

    dvx_dx = (torch.roll(vx, -1, dims=-1) - torch.roll(vx, 1, dims=-1)) / (2.0 * dx)
    dvx_dy = (torch.roll(vx, -1, dims=-2) - torch.roll(vx, 1, dims=-2)) / (2.0 * dy)
    dvy_dx = (torch.roll(vy, -1, dims=-1) - torch.roll(vy, 1, dims=-1)) / (2.0 * dx)
    dvy_dy = (torch.roll(vy, -1, dims=-2) - torch.roll(vy, 1, dims=-2)) / (2.0 * dy)

    S_xx = dvx_dx
    S_yy = dvy_dy
    S_xy = 0.5 * (dvx_dy + dvy_dx)

    pi = (
        tau_perp[..., 1, 1] * S_xx
        + tau_perp[..., 2, 2] * S_yy
        + 2.0 * tau_perp[..., 1, 2] * S_xy
    )
    backscattering = pi > 0.0

    if mode == "zero":
        return torch.where(
            backscattering.unsqueeze(-1).unsqueeze(-1),
            fraction * tau_perp,
            tau_perp,
        )

    if mode == "project":
        S = torch.zeros_like(tau_perp)
        S[..., 1, 1] = S_xx
        S[..., 2, 2] = S_yy
        S[..., 1, 2] = S_xy
        S[..., 2, 1] = S_xy
        S_sq = S_xx**2 + S_yy**2 + 2.0 * S_xy**2
        coeff = torch.where(backscattering, pi / S_sq.clamp(min=1e-30), torch.zeros_like(pi))
        return tau_perp - (1.0 - fraction) * coeff.unsqueeze(-1).unsqueeze(-1) * S

    raise ValueError(f"Unknown backscatter clip mode: '{mode}'. Expected 'zero' or 'project'.")


# ---------------------------------------------------------------------------
# Conservative Viscous Flux Formulation
# ---------------------------------------------------------------------------

def compute_sgs_viscous_fluxes(
    prims: torch.Tensor,
    tau_perp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute conservative viscous flux contributions from fluid-frame SGS stress.

    Given τ_⊥^{μν} satisfying u_μ τ_⊥^{μν} = 0, reconstructs the lab-frame
    contributions to conserved variables and fluxes using the transversality
    condition to recover time components:

        τ_⊥^{0ν} = v_x τ_⊥^{1ν} + v_y τ_⊥^{2ν}

    Then maps to the conserved variable / flux structure:

        δU   = (0,  τ_⊥^{01},  τ_⊥^{02},  τ_⊥^{00})
        δF^x = (0,  τ_⊥^{11},  τ_⊥^{12},  τ_⊥^{01})
        δF^y = (0,  τ_⊥^{21},  τ_⊥^{22},  τ_⊥^{02})

    where the conserved variable ordering is (D, S_x, S_y, τ_energy).

    Parameters
    ----------
    prims : (Ny, Nx, 4) — [ρ, vx, vy, p]
    tau_perp : (Ny, Nx, 4, 4) — fluid-frame SGS stress tensor.
        Must satisfy u_μ τ_⊥^{μν} ≈ 0 (the integrity basis guarantees this).

    Returns
    -------
    delta_U  : (Ny, Nx, 4) — correction to conserved variables
    delta_Fx : (Ny, Nx, 4) — correction to x-direction flux
    delta_Fy : (Ny, Nx, 4) — correction to y-direction flux
    """
    vx = prims[..., 1]
    vy = prims[..., 2]

    # Reconstruct time components from transversality: τ_⊥^{0ν} = v_j τ_⊥^{jν}
    tau_01 = vx * tau_perp[..., 1, 1] + vy * tau_perp[..., 2, 1]  # τ_⊥^{0,x}
    tau_02 = vx * tau_perp[..., 1, 2] + vy * tau_perp[..., 2, 2]  # τ_⊥^{0,y}
    tau_00 = vx * tau_01 + vy * tau_02                             # τ_⊥^{0,0}

    dtype = prims.dtype
    device = prims.device
    Ny, Nx = prims.shape[:2]

    # δU = (0, τ_⊥^{01}, τ_⊥^{02}, τ_⊥^{00})
    delta_U = torch.zeros(Ny, Nx, 4, dtype=dtype, device=device)
    delta_U[..., 1] = tau_01
    delta_U[..., 2] = tau_02
    delta_U[..., 3] = tau_00

    # δF^x = (0, τ_⊥^{11}, τ_⊥^{12}, τ_⊥^{01})
    delta_Fx = torch.zeros(Ny, Nx, 4, dtype=dtype, device=device)
    delta_Fx[..., 1] = tau_perp[..., 1, 1]
    delta_Fx[..., 2] = tau_perp[..., 1, 2]
    delta_Fx[..., 3] = tau_01  # = τ_⊥^{10} by symmetry

    # δF^y = (0, τ_⊥^{21}, τ_⊥^{22}, τ_⊥^{02})
    delta_Fy = torch.zeros(Ny, Nx, 4, dtype=dtype, device=device)
    delta_Fy[..., 1] = tau_perp[..., 2, 1]
    delta_Fy[..., 2] = tau_perp[..., 2, 2]
    delta_Fy[..., 3] = tau_02  # = τ_⊥^{20} by symmetry

    return delta_U, delta_Fx, delta_Fy


# ---------------------------------------------------------------------------
# CFL Timestep
# ---------------------------------------------------------------------------

def compute_dt(
    prims: torch.Tensor,
    dx: float,
    dy: float,
    cfl: float = 0.4,
) -> float:
    """Compute the CFL-limited timestep.

    Parameters
    ----------
    prims : (Ny, Nx, 4) tensor — [ρ, vx, vy, p]
    dx, dy : grid spacings
    cfl : CFL number (< 1)

    Returns
    -------
    float — timestep
    """
    rho = prims[..., 0]
    vx_p = prims[..., 1]
    vy_p = prims[..., 2]
    p = prims[..., 3]

    lam_minus_x, lam_plus_x = compute_wave_speeds(rho, vx_p, vy_p, p, direction=0)
    max_speed_x = torch.maximum(lam_plus_x.abs(), lam_minus_x.abs()).max()

    Ny = prims.shape[-3]
    if Ny > 1:
        lam_minus_y, lam_plus_y = compute_wave_speeds(rho, vx_p, vy_p, p, direction=1)
        max_speed_y = torch.maximum(lam_plus_y.abs(), lam_minus_y.abs()).max()
        dt = cfl / (max_speed_x / dx + max_speed_y / dy + 1e-30)
    else:
        dt = cfl / (max_speed_x / dx + 1e-30)

    return dt.item()


# Guard: test actual CUDA usability, not just driver presence
def _cuda_usable() -> bool:
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return False
        t = torch.zeros(1, device="cuda")
        del t
        return True
    except Exception:
        return False


if _cuda_usable():
    try:
        compute_dt = torch.compile(compute_dt, mode="default", dynamic=False)  # type: ignore[assignment]
        primitives_to_conserved = torch.compile(primitives_to_conserved, mode="default", dynamic=False)  # type: ignore[assignment]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 3rd-Order Runge-Kutta (SSP-RK3) Time Integrator
# ---------------------------------------------------------------------------

class RHDSolver:
    """2D Relativistic Hydrodynamic solver with optional SGS closure,
    matching the Aenus code discretization (Obergaulinger 2008).

    Numerical method:
      - 5th-order Monotonicity-Preserving (MP5) spatial reconstruction (Suresh & Huynh 1997)
      - HLL approximate Riemann solver (Harten 1983)
      - 3rd-order Strong Stability Preserving Runge-Kutta (SSP-RK3 / Shu-Osher 1988)

    Parameters
    ----------
    Ny, Nx : int — grid dimensions
    x_range, y_range : (float, float) — physical domain extents
    cfl : float — CFL number (default 0.4)
    reconstruction : str — "mp5" (default) or "muscl"
    riemann_solver : str — "hll" (default), "musta_hll", or "hllc"
    reconstruct_primitives : bool — if True (default, matching Aenus),
        reconstruct primitive variables directly
    dtype : torch.dtype
    device : torch.device or None
    """

    Ny: int
    Nx: int
    dx: float
    dy: float
    cfl: float
    reconstruction: str
    riemann_solver: str
    reconstruct_primitives: bool
    dtype: torch.dtype
    device: torch.device
    t: float
    prims: torch.Tensor

    def __init__(
        self,
        Ny: int = 256,
        Nx: int = 256,
        x_range: tuple[float, float] = (0.0, 1.0),
        y_range: tuple[float, float] = (0.0, 1.0),
        cfl: float = 0.4,
        reconstruction: str = "mp5",
        riemann_solver: str = "hll",
        reconstruct_primitives: bool = True,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ):
        self.Ny = Ny
        self.Nx = Nx
        self.dx = (x_range[1] - x_range[0]) / Nx
        self.dy = (y_range[1] - y_range[0]) / Ny
        self.cfl = cfl
        self.reconstruction = reconstruction
        self.riemann_solver = riemann_solver
        self.reconstruct_primitives = reconstruct_primitives
        self.dtype = dtype
        self.device = device if device is not None else torch.device("cpu")
        self.t = 0.0
        self.prims = torch.empty(0, dtype=dtype, device=self.device)
        self.history_t: list[float] = []
        self.history_rho: list[torch.Tensor] = []
        self.history_vx: list[torch.Tensor] = []
        self.history_vy: list[torch.Tensor] = []
        self.history_p: list[torch.Tensor] = []

    def initialize(self, rho: torch.Tensor, vx: torch.Tensor,
                   vy: torch.Tensor, p: torch.Tensor) -> None:
        """Set initial conditions from primitive variables.

        Parameters
        ----------
        rho, vx, vy, p : (Ny, Nx) tensors
        """
        rho = rho.clamp(min=1e-15)
        p = p.clamp(min=1e-15)
        v_sq = vx**2 + vy**2
        v_max_sq = 0.98  # v_max ~ 0.99 c
        n_clamped = int((v_sq > v_max_sq).sum())
        if n_clamped:
            warnings.warn(
                f"initialize() clamped |v| in {n_clamped} cell(s) to sqrt({v_max_sq}); "
                f"max v² was {float(v_sq.max()):.6f}. The evolved state no longer matches "
                f"the supplied initial condition.",
                RuntimeWarning,
                stacklevel=2,
            )
        scale = torch.where(
            v_sq > v_max_sq,
            torch.sqrt(v_max_sq / v_sq.clamp(min=1e-15)),
            torch.ones_like(v_sq),
        )
        vx = vx * scale
        vy = vy * scale
        self.prims = torch.stack([rho, vx, vy, p], dim=-1).to(
            dtype=self.dtype, device=self.device
        )  # (Ny, Nx, 4)

    def step(
        self,
        dt: float | None = None,
        sgs_source: torch.Tensor | None = None,
        sgs_model: Callable[[torch.Tensor], torch.Tensor] | None = None,
        sgs_closure: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> float:
        """Advance one SSP-RK3 timestep.

        SSP-RK3 (Shu-Osher form):
          U¹   = Uⁿ + Δt L(Uⁿ)
          U²   = ¾ Uⁿ + ¼ (U¹ + Δt L(U¹))
          Uⁿ⁺¹ = ⅓ Uⁿ + ⅔ (U² + Δt L(U²))

        The SGS source term is re-evaluated from each intermediate state to
        preserve the SSP (strong stability preserving) property.  If only a
        static ``sgs_source`` tensor is supplied (legacy path), it is reused
        across all three stages.

        Parameters
        ----------
        dt : float or None — if None, compute from CFL
        sgs_source : (Ny, Nx, 4) tensor or None — static source term
            (legacy; prefer ``sgs_model`` for proper SSP stability)
        sgs_model : callable or None — SGS model f(prims) → source.
            Re-evaluated at each RK sub-stage for correct SSP coupling.
        sgs_closure : callable or None — frame-consistent SGS closure
            f(prims) → (tau_perp, alpha).  Uses the conservative viscous
            flux formulation with δU state correction.  Mutually exclusive
            with sgs_model / sgs_source.

        Returns
        -------
        float — the timestep used
        """
        if dt is None:
            dt = compute_dt(self.prims, self.dx, self.dy, self.cfl)

        rho_n = self.prims[..., 0]
        vx_n = self.prims[..., 1]
        vy_n = self.prims[..., 2]
        p_n = self.prims[..., 3]
        U_n = primitives_to_conserved(rho_n, vx_n, vy_n, p_n)

        # ---------------------------------------------------------------
        # Conservative viscous flux path (frame-consistent coupling)
        # ---------------------------------------------------------------
        if sgs_closure is not None:
            return self._step_conservative_flux(dt, U_n, sgs_closure)

        # ---------------------------------------------------------------
        # Legacy source-term path (backward compatible)
        # ---------------------------------------------------------------
        # Stage 1: U¹ = Uⁿ + Δt L(Uⁿ)
        sgs1 = sgs_model(self.prims) if sgs_model is not None else sgs_source
        rhs1 = compute_rhs(
            self.prims, self.dx, self.dy, sgs1,
            reconstruction=self.reconstruction,
            riemann_solver=self.riemann_solver,
            reconstruct_primitives=self.reconstruct_primitives,
        )
        U_1 = U_n + dt * rhs1

        rho_1, vx_1, vy_1, p_1 = conserved_to_primitives(U_1, p_guess=p_n)
        prims_1 = torch.stack([rho_1, vx_1, vy_1, p_1], dim=-1)

        # Stage 2: U² = ¾ Uⁿ + ¼ (U¹ + Δt L(U¹))
        sgs2 = sgs_model(prims_1) if sgs_model is not None else sgs_source
        rhs2 = compute_rhs(
            prims_1, self.dx, self.dy, sgs2,
            reconstruction=self.reconstruction,
            riemann_solver=self.riemann_solver,
            reconstruct_primitives=self.reconstruct_primitives,
        )
        U_2 = 0.75 * U_n + 0.25 * (U_1 + dt * rhs2)

        rho_2, vx_2, vy_2, p_2 = conserved_to_primitives(U_2, p_guess=p_1)
        prims_2 = torch.stack([rho_2, vx_2, vy_2, p_2], dim=-1)

        # Stage 3: Uⁿ⁺¹ = ⅓ Uⁿ + ⅔ (U² + Δt L(U²))
        sgs3 = sgs_model(prims_2) if sgs_model is not None else sgs_source
        rhs3 = compute_rhs(
            prims_2, self.dx, self.dy, sgs3,
            reconstruction=self.reconstruction,
            riemann_solver=self.riemann_solver,
            reconstruct_primitives=self.reconstruct_primitives,
        )
        U_new = (1.0 / 3.0) * U_n + (2.0 / 3.0) * (U_2 + dt * rhs3)

        rho_new, vx_new, vy_new, p_new = conserved_to_primitives(
            U_new, p_guess=p_2,
        )
        self.prims = torch.stack([rho_new, vx_new, vy_new, p_new], dim=-1)
        self.t += dt

        return dt

    def _step_conservative_flux(
        self,
        dt: float,
        U_n: torch.Tensor,
        sgs_closure: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> float:
        r"""SSP-RK3 step using the conservative viscous flux formulation.

        Evolves the modified conserved variables Ũ = U + δU where δU captures
        the time-space components of τ_⊥^{μν} via the transversality condition.
        This automatically accounts for the ∂_t τ_⊥^{0ν} term that the legacy
        source-term approach drops.

        Ũ is carried across the RK stages so the SSP convex combinations are formed on a
        single consistent variable. Recovering U from Ũ is implicit (δU depends on the
        state being solved for) and is done by one predictor-corrector pass per stage.

        Parameters
        ----------
        dt : float
        U_n : (Ny, Nx, 4) — conserved variables at time level n
        sgs_closure : callable — f(prims) → (tau_perp, alpha)
        """
        p_n = self.prims[..., 3]

        def _sgs(prims_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            tau_perp_k, alpha_k = sgs_closure(prims_k)
            tau_eff_k = alpha_k.unsqueeze(-1).unsqueeze(-1) * tau_perp_k
            return compute_sgs_viscous_fluxes(prims_k, tau_eff_k)

        def _recover(
            U_tilde: torch.Tensor,
            delta_U_pred: torch.Tensor,
            p_guess: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Invert Ũ = U + δU(U) by fixed-point iteration seeded with delta_U_pred.

            δU depends on the state being solved for, so the inversion is implicit.
            Two corrections suffice: the iteration contracts by |∂δU/∂U| ≪ 1, and a
            third changes the result by less than roundoff. Stopping at one leaves
            enough residual to drag SSP-RK3 down to ~1st order.
            """
            delta_U_k = delta_U_pred
            for _ in range(_SGS_RECOVERY_ITERS):
                rho_k, vx_k, vy_k, p_k = conserved_to_primitives(
                    U_tilde - delta_U_k, p_guess=p_guess,
                )
                prims_k = torch.stack([rho_k, vx_k, vy_k, p_k], dim=-1)
                delta_U_k, delta_Fx_k, delta_Fy_k = _sgs(prims_k)

            rho_k, vx_k, vy_k, p_k = conserved_to_primitives(
                U_tilde - delta_U_k, p_guess=p_guess,
            )
            prims_k = torch.stack([rho_k, vx_k, vy_k, p_k], dim=-1)
            return prims_k, delta_U_k, delta_Fx_k, delta_Fy_k

        def _rhs(
            prims_k: torch.Tensor,
            delta_Fx_k: torch.Tensor,
            delta_Fy_k: torch.Tensor,
        ) -> torch.Tensor:
            return compute_rhs(
                prims_k, self.dx, self.dy,
                sgs_viscous_Fx=delta_Fx_k,
                sgs_viscous_Fy=delta_Fy_k,
                reconstruction=self.reconstruction,
                riemann_solver=self.riemann_solver,
                reconstruct_primitives=self.reconstruct_primitives,
            )

        delta_U_n, delta_Fx_n, delta_Fy_n = _sgs(self.prims)
        U_tilde_n = U_n + delta_U_n

        # Stage 1: Ũ¹ = Ũⁿ + Δt L̃(Ũⁿ)
        U_tilde_1 = U_tilde_n + dt * _rhs(self.prims, delta_Fx_n, delta_Fy_n)
        prims_1, delta_U_1, delta_Fx_1, delta_Fy_1 = _recover(
            U_tilde_1, delta_U_n, p_n,
        )

        # Stage 2: Ũ² = ¾ Ũⁿ + ¼ (Ũ¹ + Δt L̃(Ũ¹))
        U_tilde_2 = 0.75 * U_tilde_n + 0.25 * (
            U_tilde_1 + dt * _rhs(prims_1, delta_Fx_1, delta_Fy_1)
        )
        prims_2, delta_U_2, delta_Fx_2, delta_Fy_2 = _recover(
            U_tilde_2, delta_U_1, prims_1[..., 3],
        )

        # Stage 3: Ũⁿ⁺¹ = ⅓ Ũⁿ + ⅔ (Ũ² + Δt L̃(Ũ²))
        U_tilde_new = (1.0 / 3.0) * U_tilde_n + (2.0 / 3.0) * (
            U_tilde_2 + dt * _rhs(prims_2, delta_Fx_2, delta_Fy_2)
        )
        self.prims, _, _, _ = _recover(U_tilde_new, delta_U_2, prims_2[..., 3])
        self.t += dt

        return dt

    def _record_history(self) -> None:
        """Record current state to CPU history buffers."""
        self.history_t.append(float(self.t))
        self.history_rho.append(self.prims[..., 0].detach().cpu().clone())
        self.history_vx.append(self.prims[..., 1].detach().cpu().clone())
        self.history_vy.append(self.prims[..., 2].detach().cpu().clone())
        self.history_p.append(self.prims[..., 3].detach().cpu().clone())

    def clear_history(self) -> None:
        """Clear recorded history buffers."""
        self.history_t.clear()
        self.history_rho.clear()
        self.history_vx.clear()
        self.history_vy.clear()
        self.history_p.clear()

    def get_history(self) -> dict[str, object]:
        """Return recorded trajectory history as a dict of tensors."""
        if len(self.history_t) == 0:
            return {}
        return {
            "t": np.array(self.history_t, dtype=np.float64),
            "rho": torch.stack(self.history_rho, dim=0),
            "vx": torch.stack(self.history_vx, dim=0),
            "vy": torch.stack(self.history_vy, dim=0),
            "p": torch.stack(self.history_p, dim=0),
        }

    def evolve(
        self,
        t_final: float,
        sgs_model: Callable[[torch.Tensor], torch.Tensor] | None = None,
        sgs_closure: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
        verbose: bool = True,
        progress_bar: bool = True,
        desc: str = "Evolving RHD",
        position: int | None = None,
        save_interval: float | None = None,
    ) -> list[float]:
        """Evolve from current time to t_final.

        Parameters
        ----------
        t_final : float — target time
        sgs_model : callable or None — if given, called at each step to
            compute SGS source term: sgs_source = sgs_model(prims)
        sgs_closure : callable or None — frame-consistent SGS closure
            f(prims) → (tau_perp, alpha) using conservative viscous fluxes
        verbose : bool
        progress_bar : bool — show tqdm progress bar tracking simulation time
        desc : str — description prefix for progress bar
        position : int or None — line offset for multi-progress bars
        save_interval : float or None — if given, record snapshot state at each
            save_interval simulation seconds

        Returns
        -------
        list of timesteps taken
        """
        from tqdm import tqdm
        import sys

        dt_history: list[float] = []
        step_count = 0
        total_time = max(float(t_final - self.t), 1e-12)

        if save_interval is not None and save_interval > 0:
            self._record_history()
            next_save_t = self.t + save_interval
        else:
            next_save_t = float("inf")

        pbar = None
        if progress_bar:
            pbar = tqdm(
                total=total_time,
                desc=desc,
                unit="sim_s",
                position=position,
                leave=True,
                dynamic_ncols=True,
                file=sys.stdout,
                mininterval=0.5,
            )

        while self.t < t_final - 1e-14:
            dt = compute_dt(self.prims, self.dx, self.dy, self.cfl)
            if not math.isfinite(dt) or dt <= 0.0:
                if pbar is not None:
                    pbar.close()
                raise FloatingPointError(
                    f"Invalid timestep dt={dt} computed at step {step_count}, t={self.t:.6f}. prims finite={torch.isfinite(self.prims).all().item()}"
                )

            dt = min(dt, t_final - self.t)

            # Synchronise time-dependent SGS models (e.g. ExactDNSSGSModel)
            if sgs_model is not None and hasattr(sgs_model, "set_time"):
                sgs_model.set_time(self.t)

            # Pass the SGS model callable so step() re-evaluates it at
            # each SSP-RK3 sub-stage, preserving the SSP stability property.
            actual_dt = self.step(dt, sgs_model=sgs_model, sgs_closure=sgs_closure)

            # Strict divergence check after advance
            if not math.isfinite(actual_dt) or not torch.isfinite(self.prims).all():
                if pbar is not None:
                    pbar.close()
                raise FloatingPointError(
                    f"RHD solver diverged at step {step_count}, t={self.t:.6f}: actual_dt={actual_dt}, non-finite values in prims."
                )

            dt_history.append(actual_dt)
            step_count += 1

            if save_interval is not None and save_interval > 0:
                if self.t >= next_save_t - 1e-12:
                    self._record_history()
                    while next_save_t <= self.t:
                        next_save_t += save_interval

            if pbar is not None:
                pbar.update(actual_dt)
                if step_count % 10 == 0 or self.t >= t_final - 1e-14:
                    pbar.set_postfix(
                        step=step_count,
                        t=f"{self.t:.4f}",
                        dt=f"{actual_dt:.2e}",
                    )
            elif verbose and step_count % 100 == 0:
                print(f"  Step {step_count}: t = {self.t:.6f}, dt = {actual_dt:.6e}", flush=True)

        if save_interval is not None and save_interval > 0:
            if len(self.history_t) == 0 or abs(self.history_t[-1] - self.t) > 1e-6:
                self._record_history()

        if pbar is not None:
            pbar.close()
        elif verbose:
            print(f"  Completed: {step_count} steps, t = {self.t:.6f}", flush=True)

        return dt_history

    def get_primitives(self) -> dict[str, torch.Tensor]:
        """Return primitive variables as a dict."""
        return {
            "rho": self.prims[..., 0],
            "vx": self.prims[..., 1],
            "vy": self.prims[..., 2],
            "p": self.prims[..., 3],
        }


# ---------------------------------------------------------------------------
# Smoke run: 1D Sod-like initial data
#
# NOT a verification. All boundaries are periodic, so the wrap-around creates a second
# Riemann problem that has interacted with the first well before t = 0.4; the result is
# not comparable to the analytic Sod solution. Run `pytest tests/test_solver.py` for the
# actual verification suite.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== 1D Sod-like smoke run (periodic BCs — not an analytic verification) ===\n")

    N = 400
    dtype = torch.float64
    device = torch.device("cpu")

    solver = RHDSolver(Ny=1, Nx=N, x_range=(0.0, 1.0), y_range=(0.0, 1.0),
                       cfl=0.4, dtype=dtype, device=device)

    x = torch.linspace(0.5/N, 1.0 - 0.5/N, N, dtype=dtype)

    # Sod shock tube initial conditions
    rho = torch.where(x < 0.5, torch.tensor(1.0, dtype=dtype),
                       torch.tensor(0.125, dtype=dtype)).unsqueeze(0)
    p = torch.where(x < 0.5, torch.tensor(1.0, dtype=dtype),
                     torch.tensor(0.1, dtype=dtype)).unsqueeze(0)
    vx = torch.zeros(1, N, dtype=dtype)
    vy = torch.zeros(1, N, dtype=dtype)

    solver.initialize(rho, vx, vy, p)

    print(f"Grid: Ny=1, Nx={N}, dx={solver.dx:.6f}")
    print(f"Initial: ρ_L={rho[0, 0]:.3f}, p_L={p[0, 0]:.3f}, ρ_R={rho[0, -1]:.3f}, p_R={p[0, -1]:.3f}")
    print(f"Evolving to t=0.4...")

    dt_history = solver.evolve(0.4)
    prims = solver.get_primitives()

    print(f"\nFinal state at t={solver.t:.4f}:")
    print(f"  ρ range: [{prims['rho'].min():.6f}, {prims['rho'].max():.6f}]")
    print(f"  p range: [{prims['p'].min():.6f}, {prims['p'].max():.6f}]")
    print(f"  vx range: [{prims['vx'].min():.6f}, {prims['vx'].max():.6f}]")
    print(f"  Total steps: {len(dt_history)}")
    print(f"  Mean dt: {sum(dt_history)/len(dt_history):.6e}")
