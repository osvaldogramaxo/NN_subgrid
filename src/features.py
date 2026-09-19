r"""
features.py

Compute covariant kinematic quantities on a discrete 2D spatial grid:
  - Four-velocity gradients via 4th-order central differences
  - Relativistic acceleration from the Euler equation
  - Lorentz-invariant scalar invariants Λ_1 … Λ_15
  - Isotropic symmetric basis tensors E^(n)^{μν}

Wraps the core tensor algebra from utils.py and provides grid-level
vectorised computation suitable for batched GPU evaluation.

Conventions:
  - Metric signature: (-, +, +, +)
  - Greek indices: spacetime {0, 1, 2, 3}
  - 2D spatial flow: v_z = 0, ∂_z = 0
"""

from __future__ import annotations

import math
import torch

from utils import (
    get_minkowski_metric,
    decompose_relativistic_kinematics,
    compute_crmhd_invariants,
    compute_crmhd_basis_tensors,
)
from src.dataset import GAMMA_EOS


# ---------------------------------------------------------------------------
# Discrete Spatial Gradients
# ---------------------------------------------------------------------------

def periodic_gradient_2d(
    field: torch.Tensor,
    dx: float,
    dy: float,
    order: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute ∂f/∂x and ∂f/∂y using central finite differences with
    periodic boundary conditions.

    Parameters
    ----------
    field : (..., Ny, Nx) tensor
    dx, dy : grid spacings
    order : 2 or 4 (accuracy order)

    Returns
    -------
    df_dx, df_dy : (..., Ny, Nx) tensors
    """
    if order == 4:
        # 4th-order central: (-f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2}) / 12Δ
        df_dx = (
            -torch.roll(field, -2, dims=-1)
            + 8.0 * torch.roll(field, -1, dims=-1)
            - 8.0 * torch.roll(field, 1, dims=-1)
            + torch.roll(field, 2, dims=-1)
        ) / (12.0 * dx)

        df_dy = (
            -torch.roll(field, -2, dims=-2)
            + 8.0 * torch.roll(field, -1, dims=-2)
            - 8.0 * torch.roll(field, 1, dims=-2)
            + torch.roll(field, 2, dims=-2)
        ) / (12.0 * dy)
    elif order == 2:
        # 2nd-order central
        df_dx = (
            torch.roll(field, -1, dims=-1)
            - torch.roll(field, 1, dims=-1)
        ) / (2.0 * dx)

        df_dy = (
            torch.roll(field, -1, dims=-2)
            - torch.roll(field, 1, dims=-2)
        ) / (2.0 * dy)
    else:
        raise ValueError(f"Unsupported order {order}; use 2 or 4")

    return df_dx, df_dy


# ---------------------------------------------------------------------------
# Four-Velocity Gradient on Grid
# ---------------------------------------------------------------------------

def compute_four_velocity_gradient(
    vx: torch.Tensor,
    vy: torch.Tensor,
    dx: float,
    dy: float,
    order: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the four-velocity u^μ and its covariant gradient
    ∇_ν u_μ on a 2D spatial grid.

    For a flat-space 2D flow:
      u^μ = (W, W v_x, W v_y, 0)
      ∇_ν u_μ = ∂_ν u_μ  (Christoffel symbols vanish in Minkowski)

    We compute only the spatial part of the gradient (∂_x, ∂_y),
    setting ∂_t u_μ = 0 for a priori analysis on single snapshots.

    Parameters
    ----------
    vx, vy : (Ny, Nx) tensors — velocity components
    dx, dy : grid spacings

    Returns
    -------
    u_con : (Ny, Nx, 4) tensor — u^μ
    grad_u_cov : (Ny, Nx, 4, 4) tensor — ∇_ν u_μ
        index [-2] is ν, index [-1] is μ
    """
    v_sq = (vx**2 + vy**2).clamp(max=1.0 - 1e-12)
    W = 1.0 / torch.sqrt((1.0 - v_sq).clamp(min=1e-12))

    # Four-velocity in contravariant form
    u0 = W
    u1 = W * vx
    u2 = W * vy
    u3 = torch.zeros_like(W)
    u_con = torch.stack([u0, u1, u2, u3], dim=-1)  # (Ny, Nx, 4)

    # Lower indices with Minkowski metric: u_0 = -W, u_1 = W vx, u_2 = W vy
    u_cov = torch.stack([-u0, u1, u2, u3], dim=-1)

    # Spatial gradients of each component of u_μ (covariant)
    # grad_u_cov[..., ν, μ] = ∂_ν u_μ
    grad_u_cov = torch.zeros(*vx.shape, 4, 4, dtype=vx.dtype, device=vx.device)

    for mu in range(4):
        du_dx, du_dy = periodic_gradient_2d(u_cov[..., mu], dx, dy, order)
        grad_u_cov[..., 1, mu] = du_dx  # ∂_x u_μ
        grad_u_cov[..., 2, mu] = du_dy  # ∂_y u_μ
        # grad_u_cov[..., 0, mu] = 0  (no temporal derivative for a priori)
        # grad_u_cov[..., 3, mu] = 0  (no z-derivative)

    return u_con, grad_u_cov


# ---------------------------------------------------------------------------
# Acceleration from the Relativistic Euler Equation
# ---------------------------------------------------------------------------

def compute_acceleration_euler(
    rho: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    p: torch.Tensor,
    dx: float,
    dy: float,
    order: int = 4,
) -> torch.Tensor:
    r"""Compute the four-acceleration a^μ from the relativistic Euler equation.

    a_μ = - (1 / ρh) (∂_μ p + u_μ u^ν ∂_ν p)

    This avoids needing temporal derivatives of u^μ, making it suitable
    for single-snapshot (a priori) evaluation.

    Parameters
    ----------
    rho, vx, vy, p : (Ny, Nx) tensors
    dx, dy : grid spacings

    Returns
    -------
    a_con : (Ny, Nx, 4) tensor — a^μ (contravariant)
    a_cov : (Ny, Nx, 4) tensor — a_μ (covariant)
    """
    eps = p / ((GAMMA_EOS - 1.0) * rho)
    h = 1.0 + eps + p / rho
    rho_h = rho * h

    v_sq = (vx**2 + vy**2).clamp(max=1.0 - 1e-12)
    W = 1.0 / torch.sqrt((1.0 - v_sq).clamp(min=1e-12))

    u_con = torch.stack([W, W * vx, W * vy, torch.zeros_like(W)], dim=-1)
    u_cov = torch.stack([-W, W * vx, W * vy, torch.zeros_like(W)], dim=-1)

    # Pressure gradient: ∂_μ p
    dp_dx, dp_dy = periodic_gradient_2d(p, dx, dy, order)
    grad_p = torch.stack([
        torch.zeros_like(p),  # ∂_t p = 0 for a priori
        dp_dx,
        dp_dy,
        torch.zeros_like(p),  # ∂_z p = 0
    ], dim=-1)  # (Ny, Nx, 4)

    # u^ν ∂_ν p = u^0 ∂_t p + u^1 ∂_x p + u^2 ∂_y p
    u_dot_grad_p = torch.einsum("...i,...i->...", u_con, grad_p)

    # a_μ = -(1/ρh)(∂_μ p + u_μ (u^ν ∂_ν p))
    a_cov = -(1.0 / rho_h).unsqueeze(-1) * (
        grad_p + u_cov * u_dot_grad_p.unsqueeze(-1)
    )

    # Raise index: a^μ = η^{μν} a_ν
    a_con = torch.stack([-a_cov[..., 0], a_cov[..., 1],
                          a_cov[..., 2], a_cov[..., 3]], dim=-1)

    return a_con, a_cov


# ---------------------------------------------------------------------------
# Full Feature Extraction Pipeline
# ---------------------------------------------------------------------------

def extract_tbnn_features(
    rho: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    p: torch.Tensor,
    dx: float,
    dy: float,
    b_con: torch.Tensor | None = None,
    gradient_order: int = 4,
) -> dict[str, torch.Tensor]:
    r"""Extract the complete TBNN feature set from primitive fields on a 2D grid.

    Computes:
      - Kinematic decomposition (σ, ω, a, θ, h)
      - 15 Lorentz scalar invariants Λ_k
      - 12 isotropic basis tensors E^(n)^{μν}
      - Thermodynamic inputs (θ, M_t)

    Parameters
    ----------
    rho, vx, vy, p : (Ny, Nx) tensors — primitive fields
    dx, dy : grid spacings
    b_con : (Ny, Nx, 4) tensor or None — magnetic four-vector (None → zero)
    gradient_order : 2 or 4

    Returns
    -------
    dict with keys:
        invariants    : (Ny, Nx, 15) — scalar invariants
        basis_tensors : (Ny, Nx, 12, 4, 4) — basis tensors
        thermo_inputs : (Ny, Nx, 2) — (θ, M_t)
        kinematics    : dict — full kinematic decomposition
    """
    Ny, Nx = rho.shape
    device = rho.device
    dtype = rho.dtype

    # Metric
    g_cov, g_con = get_minkowski_metric((Ny, Nx), device=device, dtype=dtype)

    # Four-velocity and gradient
    u_con, grad_u_cov = compute_four_velocity_gradient(vx, vy, dx, dy, gradient_order)

    # Kinematic decomposition
    kinematics = decompose_relativistic_kinematics(u_con, grad_u_cov, g_cov, g_con)

    # Four-acceleration from Euler equation (more accurate than from grad_u)
    a_con, a_cov = compute_acceleration_euler(rho, vx, vy, p, dx, dy, gradient_order)
    kinematics["a_con"] = a_con
    kinematics["a_cov"] = a_cov

    # Magnetic field (zero for pure RHD)
    if b_con is None:
        b_con = torch.zeros(Ny, Nx, 4, dtype=dtype, device=device)
    b_cov = torch.einsum("...ij,...j->...i", g_cov, b_con)

    # Characteristic filter scale Delta = sqrt(dx * dy)
    delta = float(math.sqrt(dx * dy))

    # Dimensionless kinematic quantities (scaling by Delta / c, where c = 1)
    sigma_con_hat = kinematics["sigma_con"] * delta
    sigma_cov_hat = kinematics["sigma_cov"] * delta
    omega_con_hat = kinematics["omega_con"] * delta
    omega_cov_hat = kinematics["omega_cov"] * delta
    a_con_hat = a_con * delta
    a_cov_hat = a_cov * delta

    # Magnetic field non-dimensionalized by specific enthalpy density: \hat{b} = b / \sqrt{\rho h}
    eps = p / ((GAMMA_EOS - 1.0) * rho)
    h = 1.0 + eps + p / rho
    rho_h = rho * h
    inv_sqrt_rho_h = 1.0 / torch.sqrt(rho_h.clamp(min=1e-15)).unsqueeze(-1)
    b_con_hat = b_con * inv_sqrt_rho_h
    b_cov_hat = b_cov * inv_sqrt_rho_h

    # Scalar invariants (all 15 are strictly dimensionless)
    invariants = compute_crmhd_invariants(
        sigma_con_hat, sigma_cov_hat,
        omega_con_hat, omega_cov_hat,
        b_con_hat, b_cov_hat,
        a_con_hat, a_cov_hat,
        g_cov, g_con,
    )

    # Basis tensors (all 12 are strictly dimensionless)
    basis_tensors = compute_crmhd_basis_tensors(
        sigma_con_hat, omega_con_hat,
        b_con_hat, a_con_hat,
        kinematics["h_con"], g_cov,
    )

    # Thermodynamic features
    theta_hat = kinematics["theta"] * delta  # Dimensionless expansion scalar

    # Turbulent Mach number: M_t = √(v_rms²) / c_s (dimensionless)
    cs = torch.sqrt(GAMMA_EOS * p / (rho * h))  # relativistic sound speed
    v_rms = torch.sqrt(vx**2 + vy**2)
    M_t = v_rms / cs

    thermo_inputs = torch.stack([theta_hat, M_t], dim=-1)

    # Dimensionless velocity gradient for dissipation matching: hat{grad_u} = Delta * grad_u
    grad_u_cov_hat = grad_u_cov * delta

    return {
        "invariants": invariants,
        "basis_tensors": basis_tensors,
        "thermo_inputs": thermo_inputs,
        "kinematics": kinematics,
        "u_con": u_con,
        "grad_u_cov": grad_u_cov_hat,
    }


# ---------------------------------------------------------------------------
# Convenience: per-snapshot feature extraction
# ---------------------------------------------------------------------------

def extract_features_from_snapshot(
    snapshot: dict,
    coarsening_factor: int = 4,
    gradient_order: int = 4,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Extract TBNN features from a single snapshot dict (as returned by
    dataset.read_snapshot), optionally on the Favre-filtered coarse grid.

    Parameters
    ----------
    snapshot : dict with keys 'rho', 'vx', 'vy', 'p'
    coarsening_factor : int — if > 1, Favre-filter before feature extraction
    gradient_order : int
    dtype : torch.dtype

    Returns
    -------
    dict — same structure as extract_tbnn_features
    """
    from src.dataset import relativistic_favre_filter

    rho = torch.tensor(snapshot["rho"], dtype=dtype)
    vx = torch.tensor(snapshot["vx"], dtype=dtype)
    vy = torch.tensor(snapshot["vy"], dtype=dtype)
    p = torch.tensor(snapshot["p"], dtype=dtype)

    if coarsening_factor > 1:
        coarse = relativistic_favre_filter(rho, vx, vy, p, coarsening_factor)
        rho = coarse["rho_bar"]
        vx = coarse["vx_tilde"]
        vy = coarse["vy_tilde"]
        p = coarse["p_bar"]

    Ny, Nx = rho.shape
    dx = 1.0 / Nx
    dy = 1.0 / Ny

    return extract_tbnn_features(rho, vx, vy, p, dx, dy,
                                  gradient_order=gradient_order)


# ---------------------------------------------------------------------------
# Quick verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.dataset import read_snapshot, relativistic_favre_filter
    import glob, os

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "test1"
    snap = read_snapshot(sorted(glob.glob(os.path.join(data_dir, "h*.h5")))[1])

    print(f"Extracting TBNN features from snapshot at t={snap['t']:.2f}...")
    features = extract_features_from_snapshot(snap, coarsening_factor=4)

    inv = features["invariants"]
    basis = features["basis_tensors"]
    thermo = features["thermo_inputs"]

    print(f"  Invariants shape:    {inv.shape}")
    print(f"  Basis tensors shape: {basis.shape}")
    print(f"  Thermo inputs shape: {thermo.shape}")
    print(f"  Invariant ranges:")
    for i in range(inv.shape[-1]):
        print(f"    Λ_{i+1}: [{inv[..., i].min():.4e}, {inv[..., i].max():.4e}]")
    print(f"  θ (expansion) range: [{thermo[..., 0].min():.4e}, {thermo[..., 0].max():.4e}]")
    print(f"  M_t (Mach) range:    [{thermo[..., 1].min():.4e}, {thermo[..., 1].max():.4e}]")

    # Verify kinematic constraints
    kin = features["kinematics"]
    u = features["u_con"]
    sigma_u = torch.einsum("...ij,...j->...i", kin["sigma_cov"], u)
    omega_u = torch.einsum("...ij,...j->...i", kin["omega_cov"], u)
    a_u = torch.einsum("...i,...i->...", kin["a_cov"], u)
    g_cov, _ = get_minkowski_metric(kin["sigma_con"].shape[:2], device=kin["sigma_con"].device, dtype=kin["sigma_con"].dtype)
    tr_sigma = torch.einsum("...ij,...ij->...", kin["sigma_con"], g_cov)

    print(f"\n  Kinematic constraint checks:")
    print(f"    |σ_μν u^ν|_max = {sigma_u.abs().max():.4e}")
    print(f"    |ω_μν u^ν|_max = {omega_u.abs().max():.4e}")
    print(f"    |a_μ u^μ|_max  = {a_u.abs().max():.4e}")
    print(f"    |Tr(σ)|_max    = {tr_sigma.abs().max():.4e}")
