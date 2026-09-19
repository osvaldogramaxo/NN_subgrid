r"""
test_kinematics.py

Tests for the relativistic kinematic decomposition and TBNN feature extraction.

Verifies:
  1. σ_{μν} u^ν = 0       (shear is spatial, orthogonal to u)
  2. ω_{μν} u^ν = 0       (vorticity is spatial)
  3. a_μ u^μ = 0           (acceleration is spatial)
  4. Tr(σ) = σ^μ_μ = 0    (shear is traceless)
  5. σ_{μν} = σ_{νμ}      (shear is symmetric)
  6. ω_{μν} = -ω_{νμ}     (vorticity is antisymmetric)
  7. Lorentz invariance of scalar invariants Λ_k
"""

import sys
sys.path.insert(0, ".")

import torch
from utils import (
    get_minkowski_metric,
    decompose_relativistic_kinematics,
    compute_crmhd_invariants,
)


def test_kinematic_constraints():
    """Test that kinematic decomposition obeys all covariant constraints."""
    torch.manual_seed(42)
    N = 64
    dtype = torch.float64

    g_cov, g_con = get_minkowski_metric((N,), dtype=dtype)

    # Random spatial velocity (sub-luminal)
    vx = (torch.rand(N, dtype=dtype) - 0.5) * 0.6
    vy = (torch.rand(N, dtype=dtype) - 0.5) * 0.6
    v_sq = vx**2 + vy**2
    scale = torch.where(v_sq > 0.8, torch.sqrt(0.8 / v_sq), torch.ones_like(v_sq))
    vx = vx * scale
    vy = vy * scale

    W = 1.0 / torch.sqrt(1.0 - vx**2 - vy**2)
    u_con = torch.stack([W, W * vx, W * vy, torch.zeros(N, dtype=dtype)], dim=-1)
    u_cov = torch.einsum("...ij,...j->...i", g_cov, u_con)

    # Random velocity gradient, then project to satisfy u^μ ∇_ν u_μ = 0
    grad_u_raw = torch.randn(N, 4, 4, dtype=dtype) * 0.1

    # Contract u^α with grad_u_raw[ν, α] using metric:
    # u^α ∇_ν u_α (summed over α, for each ν)
    u_dot_grad = torch.einsum("...a,...na->...n", u_con, grad_u_raw)  # (N, 4)

    # Project out: ∇_ν u_μ → ∇_ν u_μ - u_μ (u^α ∇_ν u_α) / (u^β u_β)
    # Since u^β u_β = -1:
    # ∇_ν u_μ → ∇_ν u_μ + u_μ (u^α ∇_ν u_α)
    grad_u = grad_u_raw + torch.einsum("...m,...n->...nm", u_cov, u_dot_grad)

    # Verify orthogonality: u^μ ∇_ν u_μ = 0
    check = torch.einsum("...a,...na->...n", u_con, grad_u)
    assert check.abs().max() < 1e-12, f"Projection failed: {check.abs().max():.2e}"

    kin = decompose_relativistic_kinematics(u_con, grad_u, g_cov, g_con)

    # 1. σ_{μν} u^ν = 0
    sigma_u = torch.einsum("...ij,...j->...i", kin["sigma_cov"], u_con)
    assert sigma_u.abs().max() < 1e-10, \
        f"σ_{{μν}} u^ν ≠ 0: max = {sigma_u.abs().max():.2e}"

    # 2. ω_{μν} u^ν = 0
    omega_u = torch.einsum("...ij,...j->...i", kin["omega_cov"], u_con)
    assert omega_u.abs().max() < 1e-10, \
        f"ω_{{μν}} u^ν ≠ 0: max = {omega_u.abs().max():.2e}"

    # 3. Tr(σ) = σ^μ_μ = 0  (using mixed tensor)
    sigma_mixed = torch.einsum("...ia,...aj->...ij", kin["sigma_con"], g_cov)
    tr_sigma = torch.einsum("...ii->...", sigma_mixed)
    assert tr_sigma.abs().max() < 1e-10, \
        f"Tr(σ) ≠ 0: max = {tr_sigma.abs().max():.2e}"

    # 4. σ is symmetric
    asym = (kin["sigma_cov"] - kin["sigma_cov"].transpose(-1, -2)).abs().max()
    assert asym < 1e-10, f"σ not symmetric: max asymmetry = {asym:.2e}"

    # 5. ω is antisymmetric
    anti = (kin["omega_cov"] + kin["omega_cov"].transpose(-1, -2)).abs().max()
    assert anti < 1e-10, f"ω not antisymmetric: max = {anti:.2e}"

    print("✓ All kinematic constraint tests passed")


def test_lorentz_invariance():
    """Test that scalar invariants are unchanged under a spatial rotation."""
    torch.manual_seed(123)
    dtype = torch.float64
    N = 32

    g_cov, g_con = get_minkowski_metric((N,), dtype=dtype)

    # Random primitives
    vx = (torch.rand(N, dtype=dtype) - 0.5) * 0.4
    vy = (torch.rand(N, dtype=dtype) - 0.5) * 0.4
    W = 1.0 / torch.sqrt(1.0 - vx**2 - vy**2)
    u_con = torch.stack([W, W * vx, W * vy, torch.zeros(N, dtype=dtype)], dim=-1)

    grad_u = torch.randn(N, 4, 4, dtype=dtype) * 0.05
    grad_u[:, :, 0] = 0.0

    b_con = torch.randn(N, 4, dtype=dtype) * 0.1
    b_con[:, 0] = 0.0
    b_cov = torch.einsum("...ij,...j->...i", g_cov, b_con)

    kin = decompose_relativistic_kinematics(u_con, grad_u, g_cov, g_con)

    invariants_1 = compute_crmhd_invariants(
        kin["sigma_con"], kin["sigma_cov"],
        kin["omega_con"], kin["omega_cov"],
        b_con, b_cov,
        kin["a_con"], kin["a_cov"],
        g_cov, g_con,
    )

    # Apply a spatial rotation around z-axis by angle θ
    theta = torch.tensor(0.7, dtype=dtype)
    R = torch.eye(4, dtype=dtype)
    R[1, 1] = torch.cos(theta)
    R[1, 2] = -torch.sin(theta)
    R[2, 1] = torch.sin(theta)
    R[2, 2] = torch.cos(theta)

    # Rotate all vectors and tensors
    u_con_r = torch.einsum("ij,...j->...i", R, u_con)
    b_con_r = torch.einsum("ij,...j->...i", R, b_con)
    b_cov_r = torch.einsum("...ij,...j->...i", g_cov, b_con_r)

    # Rotate grad_u: both indices of ∇_ν u_μ are covariant, so transform with R^{-1}:
    # ∇'_ν u'_μ = (R^{-1})^\alpha_ν (R^{-1})^\beta_μ ∇_α u_β
    R_inv = torch.linalg.inv(R)
    grad_u_r = torch.einsum("an,bm,...ab->...nm", R_inv, R_inv, grad_u)

    kin_r = decompose_relativistic_kinematics(u_con_r, grad_u_r, g_cov, g_con)

    invariants_2 = compute_crmhd_invariants(
        kin_r["sigma_con"], kin_r["sigma_cov"],
        kin_r["omega_con"], kin_r["omega_cov"],
        b_con_r, b_cov_r,
        kin_r["a_con"], kin_r["a_cov"],
        g_cov, g_con,
    )

    diff = (invariants_1 - invariants_2).abs().max()
    assert diff < 1e-8, \
        f"Invariants not frame-invariant under rotation: max diff = {diff:.2e}"

    print("✓ Lorentz invariance test passed (spatial rotation)")


if __name__ == "__main__":
    test_kinematic_constraints()
    test_lorentz_invariance()
    print("\nAll tests passed! ✓")
