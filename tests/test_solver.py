r"""
test_solver.py

Tests for the PyTorch 2D RHD finite-volume solver.

Verifies:
  1. Primitive ↔ Conserved variable round-trip consistency
  2. Conservation of total mass, momentum, energy in periodic domain
  3. Sod shock tube produces physically reasonable output
"""

import sys
sys.path.insert(0, ".")

import torch
from src.solver import (
    primitives_to_conserved,
    conserved_to_primitives,
    RHDSolver,
    hll_flux,
    hllc_flux,
    compute_flux,
    GAMMA_EOS,
    compute_sgs_viscous_fluxes,
    compute_transition_sensor,
    clip_sgs_tensor,
    clip_backscatter,
    smooth_sgs_tensor,
)


def test_prim_cons_roundtrip():
    """Test that primitives → conserved → primitives is identity."""
    torch.manual_seed(42)
    dtype = torch.float64
    N = 100

    rho = 0.5 + torch.rand(N, dtype=dtype) * 1.5
    vx = (torch.rand(N, dtype=dtype) - 0.5) * 0.6
    vy = (torch.rand(N, dtype=dtype) - 0.5) * 0.6
    p = 0.1 + torch.rand(N, dtype=dtype) * 2.0

    U = primitives_to_conserved(rho, vx, vy, p)
    rho_r, vx_r, vy_r, p_r = conserved_to_primitives(U)

    assert (rho - rho_r).abs().max() < 1e-8, \
        f"ρ roundtrip error: {(rho - rho_r).abs().max():.2e}"
    assert (vx - vx_r).abs().max() < 1e-8, \
        f"vx roundtrip error: {(vx - vx_r).abs().max():.2e}"
    assert (vy - vy_r).abs().max() < 1e-8, \
        f"vy roundtrip error: {(vy - vy_r).abs().max():.2e}"
    assert (p - p_r).abs().max() < 1e-8, \
        f"p roundtrip error: {(p - p_r).abs().max():.2e}"

    print("✓ Primitive ↔ Conserved roundtrip test passed")


def test_conservation():
    """Test that total mass and momentum are conserved in periodic domain."""
    dtype = torch.float64
    N = 64

    solver = RHDSolver(Ny=N, Nx=N, cfl=0.3, dtype=dtype)

    # Smooth initial conditions
    x = torch.linspace(0, 1, N, dtype=dtype)
    y = torch.linspace(0, 1, N, dtype=dtype)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    rho = 1.0 + 0.2 * torch.sin(2 * torch.pi * X) * torch.cos(2 * torch.pi * Y)
    vx = 0.1 * torch.sin(2 * torch.pi * Y)
    vy = 0.1 * torch.cos(2 * torch.pi * X)
    p = 1.0 + 0.1 * torch.cos(2 * torch.pi * X)

    solver.initialize(rho, vx, vy, p)

    # Compute initial conserved totals
    U0 = primitives_to_conserved(rho, vx, vy, p)
    mass0 = U0[..., 0].sum().item()
    momx0 = U0[..., 1].sum().item()
    momy0 = U0[..., 2].sum().item()
    energy0 = U0[..., 3].sum().item()

    # Evolve for a few steps
    for _ in range(20):
        solver.step()

    prims = solver.get_primitives()
    U_f = primitives_to_conserved(prims["rho"], prims["vx"], prims["vy"], prims["p"])
    mass_f = U_f[..., 0].sum().item()
    momx_f = U_f[..., 1].sum().item()
    momy_f = U_f[..., 2].sum().item()
    energy_f = U_f[..., 3].sum().item()

    mass_err = abs(mass_f - mass0) / abs(mass0)
    momx_err = abs(momx_f - momx0) / mass0
    momy_err = abs(momy_f - momy0) / mass0
    energy_err = abs(energy_f - energy0) / abs(energy0)

    assert mass_err < 1e-10, f"Mass not conserved: relative error = {mass_err:.2e}"
    assert momx_err < 1e-10, f"X-momentum not conserved: relative error = {momx_err:.2e}"
    assert momy_err < 1e-10, f"Y-momentum not conserved: relative error = {momy_err:.2e}"
    assert energy_err < 1e-10, f"Energy not conserved: relative error = {energy_err:.2e}"

    print(f"✓ Conservation test passed (mass err={mass_err:.2e}, energy err={energy_err:.2e})")


def test_sod_shock_tube():
    """Test that Sod shock tube produces physical results."""
    dtype = torch.float64
    N = 200

    solver = RHDSolver(Ny=1, Nx=N, x_range=(0.0, 1.0), y_range=(0.0, 1.0),
                       cfl=0.4, dtype=dtype)

    x = torch.linspace(0.5/N, 1.0 - 0.5/N, N, dtype=dtype)
    rho = torch.where(x < 0.5, torch.tensor(1.0, dtype=dtype),
                       torch.tensor(0.125, dtype=dtype)).unsqueeze(0)
    p = torch.where(x < 0.5, torch.tensor(1.0, dtype=dtype),
                     torch.tensor(0.1, dtype=dtype)).unsqueeze(0)
    vx = torch.zeros(1, N, dtype=dtype)
    vy = torch.zeros(1, N, dtype=dtype)

    solver.initialize(rho, vx, vy, p)
    solver.evolve(0.2, verbose=False)

    prims = solver.get_primitives()

    # Check no NaN
    assert not torch.isnan(prims["rho"]).any(), "NaN in density!"
    assert not torch.isnan(prims["p"]).any(), "NaN in pressure!"

    # Check physical bounds
    assert (prims["rho"] > 0).all(), "Negative density!"
    assert (prims["p"] > 0).all(), "Negative pressure!"

    # Check velocities are subluminal
    v_sq = prims["vx"]**2 + prims["vy"]**2
    assert (v_sq < 1.0).all(), "Superluminal velocity!"

    print("✓ Sod shock tube test passed")


def test_mp5_convergence():
    """Verify 5th-order convergence of MP5 reconstruction on smooth wave."""
    from src.solver import mp5_reconstruct
    import math

    dtype = torch.float64
    errors: list[float] = []
    resolutions = [32, 64, 128, 256]

    for N in resolutions:
        dx = 1.0 / N
        x = (torch.arange(N, dtype=dtype) + 0.5) * dx
        # Exact cell average of sin(2*pi*x) across cell [x - dx/2, x + dx/2]
        sinc = math.sin(math.pi * dx) / (math.pi * dx)
        v = (torch.sin(2.0 * math.pi * x) * sinc).unsqueeze(0)  # (1, N)

        # Exact point value at interface x_{j+1/2} = (j + 1) * dx
        x_face = (torch.arange(N, dtype=dtype) + 1.0) * dx
        v_exact = torch.sin(2.0 * math.pi * x_face).unsqueeze(0)

        v_L, _ = mp5_reconstruct(v, dim=-1)
        err = float((v_L - v_exact).abs().max().item())
        errors.append(err)

    # Check convergence order between consecutive resolutions: should be ~5
    for i in range(len(resolutions) - 1):
        ratio = errors[i] / errors[i + 1]
        order = math.log2(ratio)
        assert order > 4.8, f"MP5 convergence order too low: {order:.2f} (expected ~5.0)"

    print(f"✓ MP5 5th-order convergence verified (errors: {[f'{e:.2e}' for e in errors]})")


def test_mp5_monotonicity():
    """Verify MP5 preserves monotonicity at a step discontinuity with zero overshoot."""
    from src.solver import mp5_reconstruct

    dtype = torch.float64
    N = 100
    v = torch.zeros(1, N, dtype=dtype)
    v[0, :N // 2] = 1.0

    v_L, v_R = mp5_reconstruct(v, dim=-1)

    # Monotonicity check: no values outside [0, 1]
    assert v_L.min().item() >= -1e-12, f"MP5 undershoot in v_L: {v_L.min().item():.2e}"
    assert v_L.max().item() <= 1.0 + 1e-12, f"MP5 overshoot in v_L: {v_L.max().item():.2e}"
    assert v_R.min().item() >= -1e-12, f"MP5 undershoot in v_R: {v_R.min().item():.2e}"
    assert v_R.max().item() <= 1.0 + 1e-12, f"MP5 overshoot in v_R: {v_R.max().item():.2e}"

    print("✓ MP5 monotonicity test passed (strict bounds [0, 1] maintained)")


def test_riemann_solvers():
    """Verify that HLL, MUSTA-HLL, and HLLC Riemann solvers all produce physical results."""
    dtype = torch.float64
    N = 100

    for solver_name in ["hll", "musta_hll", "hllc"]:
        solver = RHDSolver(
            Ny=1, Nx=N, cfl=0.3,
            reconstruction="mp5",
            riemann_solver=solver_name,
            dtype=dtype,
        )

        x = torch.linspace(0.5 / N, 1.0 - 0.5 / N, N, dtype=dtype)
        rho = torch.where(x < 0.5, torch.tensor(1.0, dtype=dtype),
                           torch.tensor(0.125, dtype=dtype)).unsqueeze(0)
        p = torch.where(x < 0.5, torch.tensor(1.0, dtype=dtype),
                         torch.tensor(0.1, dtype=dtype)).unsqueeze(0)
        vx = torch.zeros(1, N, dtype=dtype)
        vy = torch.zeros(1, N, dtype=dtype)

        solver.initialize(rho, vx, vy, p)
        solver.evolve(0.1, verbose=False)

        prims = solver.get_primitives()
        assert not torch.isnan(prims["rho"]).any(), f"NaN in rho for {solver_name}!"
        assert (prims["rho"] > 0).all(), f"Negative rho for {solver_name}!"
        assert (prims["p"] > 0).all(), f"Negative p for {solver_name}!"
        assert (prims["vx"]**2 + prims["vy"]**2 < 1.0).all(), f"Superluminal velocity for {solver_name}!"

    print("✓ All Riemann solvers (HLL, MUSTA-HLL, HLLC) passed physical checks")


def test_reconstruction_modes():
    """Verify conservation across primitive vs conserved reconstruction and MP5 vs MUSCL."""
    dtype = torch.float64
    N = 32

    configs = [
        ("mp5", True),    # Default Aenus: MP5 on primitives
        ("mp5", False),   # MP5 on conserved
        ("muscl", True),  # MUSCL on primitives
    ]

    for recon, recon_prims in configs:
        solver = RHDSolver(
            Ny=N, Nx=N, cfl=0.3,
            reconstruction=recon,
            reconstruct_primitives=recon_prims,
            dtype=dtype,
        )

        x = torch.linspace(0, 1, N, dtype=dtype)
        y = torch.linspace(0, 1, N, dtype=dtype)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        rho = 1.0 + 0.2 * torch.sin(2 * torch.pi * X) * torch.cos(2 * torch.pi * Y)
        vx = 0.1 * torch.sin(2 * torch.pi * Y)
        vy = 0.1 * torch.cos(2 * torch.pi * X)
        p = 1.0 + 0.1 * torch.cos(2 * torch.pi * X)

        solver.initialize(rho, vx, vy, p)
        U0 = primitives_to_conserved(rho, vx, vy, p)
        mass0 = U0[..., 0].sum().item()
        energy0 = U0[..., 3].sum().item()

        for _ in range(10):
            solver.step()

        prims = solver.get_primitives()
        U_f = primitives_to_conserved(prims["rho"], prims["vx"], prims["vy"], prims["p"])
        mass_f = U_f[..., 0].sum().item()
        energy_f = U_f[..., 3].sum().item()

        mass_err = abs(mass_f - mass0) / abs(mass0)
        energy_err = abs(energy_f - energy0) / abs(energy0)

        assert mass_err < 1e-10, f"Mass error {mass_err:.2e} too high for ({recon}, recon_prims={recon_prims})"
        assert energy_err < 1e-10, f"Energy error {energy_err:.2e} too high for ({recon}, recon_prims={recon_prims})"

    print("✓ All reconstruction configurations passed conservation tests")


def test_compute_sgs_viscous_fluxes():
    """Test analytic correctness of conservative viscous flux reconstruction."""
    Ny, Nx = 4, 4
    dtype = torch.float64

    # Primitives: rho, vx, vy, p
    rho = torch.ones(Ny, Nx, dtype=dtype)
    vx = torch.full((Ny, Nx), 0.2, dtype=dtype)
    vy = torch.full((Ny, Nx), -0.3, dtype=dtype)
    p = torch.ones(Ny, Nx, dtype=dtype)
    prims = torch.stack([rho, vx, vy, p], dim=-1)

    # Known symmetric tau_perp (spatial block)
    tau_perp = torch.zeros(Ny, Nx, 4, 4, dtype=dtype)
    tau_perp[..., 1, 1] = 0.05  # xx
    tau_perp[..., 1, 2] = 0.02  # xy
    tau_perp[..., 2, 1] = 0.02  # yx
    tau_perp[..., 2, 2] = 0.08  # yy

    delta_U, delta_Fx, delta_Fy = compute_sgs_viscous_fluxes(prims, tau_perp)

    # Expected values from transversality:
    # tau_01 = vx * tau_11 + vy * tau_21 = 0.2 * 0.05 + (-0.3) * 0.02 = 0.01 - 0.006 = 0.004
    # tau_02 = vx * tau_12 + vy * tau_22 = 0.2 * 0.02 + (-0.3) * 0.08 = 0.004 - 0.024 = -0.020
    # tau_00 = vx * tau_01 + vy * tau_02 = 0.2 * 0.004 + (-0.3) * (-0.020) = 0.0008 + 0.0060 = 0.0068

    # delta_U = (0, tau_01, tau_02, tau_00)
    assert torch.allclose(delta_U[..., 0], torch.zeros_like(rho))
    assert torch.allclose(delta_U[..., 1], torch.full_like(rho, 0.004))
    assert torch.allclose(delta_U[..., 2], torch.full_like(rho, -0.020))
    assert torch.allclose(delta_U[..., 3], torch.full_like(rho, 0.0068))

    # delta_Fx = (0, tau_11, tau_12, tau_01)
    assert torch.allclose(delta_Fx[..., 0], torch.zeros_like(rho))
    assert torch.allclose(delta_Fx[..., 1], torch.full_like(rho, 0.05))
    assert torch.allclose(delta_Fx[..., 2], torch.full_like(rho, 0.02))
    assert torch.allclose(delta_Fx[..., 3], torch.full_like(rho, 0.004))

    # delta_Fy = (0, tau_21, tau_22, tau_02)
    assert torch.allclose(delta_Fy[..., 0], torch.zeros_like(rho))
    assert torch.allclose(delta_Fy[..., 1], torch.full_like(rho, 0.02))
    assert torch.allclose(delta_Fy[..., 2], torch.full_like(rho, 0.08))
    assert torch.allclose(delta_Fy[..., 3], torch.full_like(rho, -0.020))

    print("✓ compute_sgs_viscous_fluxes passed analytic test")


def test_compute_transition_sensor():
    """Test the enstrophy-fraction, none, and Germano gating modes."""
    N = 32
    dtype = torch.float64
    dx = 1.0 / N
    dy = 1.0 / N

    y = torch.linspace(0, 1, N, dtype=dtype)
    x = torch.linspace(0, 1, N, dtype=dtype)
    Y_grid, X_grid = torch.meshgrid(y, x, indexing="ij")

    # Case 1: Pure divergence flow (compression): vx = sin(2pi x), vy = 0
    # omega = dvy/dx - dvx/dy = 0 - 0 = 0 -> alpha_ducros should be 0 everywhere
    prims_div = torch.zeros(N, N, 4, dtype=dtype)
    prims_div[..., 0] = 1.0
    prims_div[..., 1] = torch.sin(2 * torch.pi * X_grid)
    prims_div[..., 2] = 0.0
    prims_div[..., 3] = 1.0

    alpha_enstrophy_div = compute_transition_sensor(prims_div, dx, dy, mode="enstrophy")
    assert (alpha_enstrophy_div < 1e-10).all(), f"enstrophy sensor in pure div flow max is {alpha_enstrophy_div.max()}"

    # Case 2: Pure rotational flow (rigid-body-like vortex):
    # vx = - (y - 0.5), vy = (x - 0.5)
    # dvx_dy = -1, dvy_dx = 1 -> dvx_dy + dvy_dx = 0 -> S_sq = 0, omega = 2 != 0 -> alpha = 1.0
    prims_vortex = torch.zeros(N, N, 4, dtype=dtype)
    prims_vortex[..., 0] = 1.0
    prims_vortex[..., 1] = -(Y_grid - 0.5)
    prims_vortex[..., 2] = (X_grid - 0.5)
    prims_vortex[..., 3] = 1.0

    alpha_enstrophy_vortex = compute_transition_sensor(prims_vortex, dx, dy, mode="enstrophy")
    interior = slice(2, -2)
    assert (alpha_enstrophy_vortex[interior, interior] > 0.99).all(), \
        f"enstrophy sensor in pure rotation min is {alpha_enstrophy_vortex[interior, interior].min()}"

    # Case 3: Test Germano sensor runs and produces values bounded in [0, 1]
    alpha_germano = compute_transition_sensor(prims_div, dx, dy, mode="germano")
    assert (alpha_germano >= 0.0).all() and (alpha_germano <= 1.0).all()

    # Case 4: pure shear is the case the enstrophy fraction CANNOT resolve.
    # |omega|^2 == |S|^2 for any pure shear, so alpha == 1/2 regardless of how laminar
    # the layer is. This is documented behaviour, not a transition detector.
    prims_shear = torch.zeros(N, N, 4, dtype=dtype)
    prims_shear[..., 0] = 1.0
    prims_shear[..., 1] = torch.tanh((Y_grid - 0.5) / 0.05)
    prims_shear[..., 3] = 1.0
    alpha_shear = compute_transition_sensor(prims_shear, dx, dy, mode="enstrophy")
    assert torch.allclose(alpha_shear, torch.full_like(alpha_shear, 0.5), atol=1e-12), \
        f"expected alpha == 1/2 in pure shear, got range [{alpha_shear.min()}, {alpha_shear.max()}]"

    # Case 5: mode="none" must leave the closure ungated (exact-stress runs rely on this).
    alpha_none = compute_transition_sensor(prims_shear, dx, dy, mode="none")
    assert (alpha_none == 1.0).all()

    print("✓ compute_transition_sensor passed enstrophy, shear, none and Germano tests")


def test_clip_sgs_tensor():
    """Test realizability clipping strategies (Frobenius and eigenvalue)."""
    Ny, Nx = 4, 4
    dtype = torch.float64

    p0 = 1.0
    kappa = 0.1
    prims = torch.zeros(Ny, Nx, 4, dtype=dtype)
    prims[..., 0] = 1.0
    prims[..., 1] = 0.1  # vx
    prims[..., 2] = 0.1  # vy
    prims[..., 3] = p0   # p

    # Test Frobenius clipping:
    tau_perp = torch.zeros(Ny, Nx, 4, 4, dtype=dtype)
    tau_perp[..., 1, 1] = 5.0
    tau_perp[..., 2, 2] = 5.0

    clipped_frob = clip_sgs_tensor(tau_perp, prims, mode="frobenius", kappa=kappa)
    frob_norm = torch.norm(clipped_frob, dim=(-2, -1))
    assert (frob_norm <= kappa * p0 + 1e-12).all(), f"Frobenius norm {frob_norm.max()} exceeded {kappa * p0}"

    # Small tensor not clipped:
    tau_small = torch.zeros(Ny, Nx, 4, 4, dtype=dtype)
    tau_small[..., 1, 1] = 0.01
    clipped_small = clip_sgs_tensor(tau_small, prims, mode="frobenius", kappa=kappa)
    assert torch.allclose(clipped_small, tau_small)

    # Test Eigenvalue clipping:
    clipped_eig = clip_sgs_tensor(tau_perp, prims, mode="eigenvalue", kappa=kappa)
    spatial = clipped_eig[..., 1:3, 1:3]
    evals, _ = torch.linalg.eigh(spatial)
    assert (evals.abs() <= kappa * p0 + 1e-12).all(), f"Eigenvalue {evals.abs().max()} exceeded {kappa * p0}"

    # Check transversality condition holds for clipped tensor: tau^{0nu} = v_j tau^{jnu}
    vx = prims[..., 1]
    vy = prims[..., 2]
    tau_01_expected = vx * clipped_eig[..., 1, 1] + vy * clipped_eig[..., 2, 1]
    assert torch.allclose(clipped_eig[..., 0, 1], tau_01_expected)

    print("✓ clip_sgs_tensor passed Frobenius and eigenvalue clipping tests")


def test_flux_closure_conservation():
    """Test RHDSolver conservation with conservative viscous flux closure."""
    dtype = torch.float64
    N = 32
    solver = RHDSolver(Ny=N, Nx=N, cfl=0.3, dtype=dtype)

    x = torch.linspace(0, 1, N, dtype=dtype)
    y = torch.linspace(0, 1, N, dtype=dtype)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    rho = 1.0 + 0.2 * torch.sin(2 * torch.pi * X) * torch.cos(2 * torch.pi * Y)
    vx = 0.1 * torch.sin(2 * torch.pi * Y)
    vy = 0.1 * torch.cos(2 * torch.pi * X)
    p = 1.0 + 0.1 * torch.cos(2 * torch.pi * X)

    solver.initialize(rho, vx, vy, p)
    U0 = primitives_to_conserved(rho, vx, vy, p)
    mass0 = U0[..., 0].sum().item()
    energy0 = U0[..., 3].sum().item()

    # 1. Zero closure test: should conserve mass and energy to roundoff
    def zero_closure(prims: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tau = torch.zeros(N, N, 4, 4, dtype=dtype)
        alpha = torch.zeros(N, N, dtype=dtype)
        return tau, alpha

    for _ in range(10):
        solver.step(sgs_closure=zero_closure)

    prims = solver.get_primitives()
    U_f = primitives_to_conserved(prims["rho"], prims["vx"], prims["vy"], prims["p"])
    mass_f = U_f[..., 0].sum().item()
    energy_f = U_f[..., 3].sum().item()

    mass_err = abs(mass_f - mass0) / abs(mass0)
    energy_err = abs(energy_f - energy0) / abs(energy0)

    assert mass_err < 1e-10, f"Mass error with zero flux closure: {mass_err:.2e}"
    assert energy_err < 1e-10, f"Energy error with zero flux closure: {energy_err:.2e}"

    # 2. Non-zero smooth closure test: mass is strictly conserved
    solver.initialize(rho, vx, vy, p)
    def active_closure(prims: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tau = torch.zeros(N, N, 4, 4, dtype=dtype)
        tau[..., 1, 1] = 0.01 * prims[..., 3]
        tau[..., 2, 2] = 0.01 * prims[..., 3]
        alpha = torch.ones(N, N, dtype=dtype)
        return tau, alpha

    for _ in range(10):
        solver.step(sgs_closure=active_closure)

    prims_act = solver.get_primitives()
    U_act = primitives_to_conserved(prims_act["rho"], prims_act["vx"], prims_act["vy"], prims_act["p"])
    mass_act = U_act[..., 0].sum().item()
    mass_act_err = abs(mass_act - mass0) / abs(mass0)

    assert mass_act_err < 1e-10, f"Mass error with active flux closure: {mass_act_err:.2e}"

    print("✓ RHDSolver conservative flux closure conservation test passed")


def test_c2p_newton_derivative():
    """The analytic df/dp must match a finite-difference of the residual."""
    dtype = torch.float64
    gm1 = GAMMA_EOS - 1.0
    U = primitives_to_conserved(
        torch.tensor(2.0, dtype=dtype), torch.tensor(0.4, dtype=dtype),
        torch.tensor(0.3, dtype=dtype), torch.tensor(1.5, dtype=dtype),
    )
    D, Sx, Sy, tau = U[0], U[1], U[2], U[3]
    S_sq = Sx**2 + Sy**2

    def residual(p):
        z = tau + D + p
        W = 1.0 / torch.sqrt(1.0 - S_sq / z**2)
        rho = D / W
        eps = (tau + D * (1.0 - W) + p * (1.0 - W**2)) / (D * W)
        return p - gm1 * rho * eps

    for p_val in (0.5, 1.5, 3.0):
        p = torch.tensor(p_val, dtype=dtype)
        z = tau + D + p
        v_sq = S_sq / z**2
        W = 1.0 / torch.sqrt(1.0 - v_sq)
        h = z / (D * W)
        analytic = 1.0 - gm1 * v_sq * (1.0 - 1.0 / h)
        eps_fd = 1e-7
        fd = (residual(p + eps_fd) - residual(p - eps_fd)) / (2 * eps_fd)
        assert abs(analytic - fd) < 1e-7, f"p={p_val}: analytic {analytic} vs FD {fd}"
        assert 2.0 / 3.0 < analytic <= 1.0, f"df/dp out of bounds: {analytic}"

    print("✓ C2P Newton derivative matches finite differences")


def test_c2p_extreme_state_roundtrip():
    """C2P must round-trip across a wide sweep of Lorentz factors and pressures."""
    dtype = torch.float64
    rho_l, vx_l, vy_l, p_l = [], [], [], []
    for rho_v in (1e-2, 1.0, 1e2):
        for p_v in (1e-3, 1.0, 1e2):
            for W in (1.0, 1.5, 5.0, 20.0, 50.0):
                v = (1.0 - 1.0 / W**2) ** 0.5
                rho_l.append(rho_v); p_l.append(p_v)
                vx_l.append(v * 0.6); vy_l.append(v * 0.8)
    rho = torch.tensor(rho_l, dtype=dtype); p = torch.tensor(p_l, dtype=dtype)
    vx = torch.tensor(vx_l, dtype=dtype); vy = torch.tensor(vy_l, dtype=dtype)

    r2, vx2, vy2, p2 = conserved_to_primitives(primitives_to_conserved(rho, vx, vy, p))
    assert torch.isfinite(torch.stack([r2, vx2, vy2, p2])).all()
    assert ((r2 - rho).abs() / rho).max() < 1e-9
    assert ((p2 - p).abs() / p).max() < 1e-6
    assert (vx2 - vx).abs().max() < 1e-12 and (vy2 - vy).abs().max() < 1e-12
    print("✓ C2P round-trips over extreme states")


def test_c2p_failure_is_not_laundered():
    """A non-finite conserved state must propagate, not be silently sanitised."""
    U = torch.tensor([[float("nan"), 0.0, 0.0, 1.0]], dtype=torch.float64)
    rho, vx, vy, p = conserved_to_primitives(U)
    assert not torch.isfinite(torch.stack([rho, vx, vy, p])).all(), \
        "C2P laundered a NaN state into finite values; failures would pass silently"
    print("✓ C2P does not launder non-finite states")


def test_hllc_resolves_contact_and_shear_exactly():
    """HLLC must return the exact upwind flux for a contact and for a shear wave."""
    dtype = torch.float64
    t = lambda a: torch.tensor([a], dtype=dtype)
    zero = t(0.0)
    rL, vL, pL = t(1.0), t(0.5), t(1.0)

    # Isolated contact: uniform p and v, density jump, moving right -> exact flux is F(left)
    exact = compute_flux(rL, vL, zero, pL, 0)
    f_hllc = hllc_flux(rL, vL, zero, pL, t(0.2), vL, zero, pL, direction=0)
    f_hll = hll_flux(rL, vL, zero, pL, t(0.2), vL, zero, pL, direction=0)
    assert (f_hllc - exact).abs().max() < 1e-12, \
        f"HLLC smeared an isolated contact by {(f_hllc - exact).abs().max()}"
    assert (f_hll - exact).abs().max() > 1e-3, "HLL unexpectedly exact; test is not discriminating"

    # Shear wave: tangential velocity jump only
    exact2 = compute_flux(rL, vL, zero, pL, 0)
    f2_hllc = hllc_flux(rL, vL, zero, pL, rL, vL, t(0.3), pL, direction=0)
    assert (f2_hllc - exact2).abs().max() < 1e-12, "HLLC smeared a shear wave"

    # y-direction must behave identically under the coordinate swap
    f3 = hllc_flux(rL, zero, vL, pL, t(0.2), zero, vL, pL, direction=1)
    exact3 = compute_flux(rL, zero, vL, pL, 1)
    assert (f3 - exact3).abs().max() < 1e-12, "HLLC y-direction contact not exact"
    print("✓ HLLC resolves contacts and shear waves exactly")


def _constant_closure(magnitude, Ny, Nx, dtype):
    """A smooth, divergence-carrying transverse SGS stress with alpha == 1."""
    def closure(prims):
        tau = torch.zeros(Ny, Nx, 4, 4, dtype=dtype)
        xs = torch.arange(Nx, dtype=dtype) / Nx
        bump = magnitude * torch.sin(2 * torch.pi * xs).view(1, Nx)
        tau[..., 1, 1] = bump
        tau[..., 2, 2] = -bump
        return tau, torch.ones(Ny, Nx, dtype=dtype)
    return closure


def test_sgs_closure_step_conserves():
    """The conservative-flux SGS path must conserve Ũ = U + δU.

    U alone is NOT conserved here and must not be asserted on: the SGS stress carries
    its own share of T^{0ν}, so the telescoping quantity is U + δU.
    """
    from src.solver import compute_sgs_viscous_fluxes
    dtype = torch.float64
    Ny = Nx = 32
    solver = RHDSolver(Ny=Ny, Nx=Nx, cfl=0.3, dtype=dtype)
    torch.manual_seed(0)
    rho = 1.0 + 0.1 * torch.rand(Ny, Nx, dtype=dtype)
    p = 1.0 + 0.1 * torch.rand(Ny, Nx, dtype=dtype)
    vx = 0.1 * torch.randn(Ny, Nx, dtype=dtype)
    vy = 0.1 * torch.randn(Ny, Nx, dtype=dtype)
    solver.initialize(rho, vx, vy, p)

    closure = _constant_closure(1e-3, Ny, Nx, dtype)

    def total_U_tilde():
        tau, alpha = closure(solver.prims)
        dU, _, _ = compute_sgs_viscous_fluxes(
            solver.prims, alpha.unsqueeze(-1).unsqueeze(-1) * tau
        )
        U = primitives_to_conserved(*[solver.prims[..., i] for i in range(4)])
        return (U + dU).sum(dim=(0, 1))

    T0 = total_U_tilde()
    for _ in range(20):
        solver.step(dt=1e-4, sgs_closure=closure)
    T1 = total_U_tilde()

    assert torch.isfinite(solver.prims).all()
    drift = ((T1 - T0).abs() / T0.abs().clamp(min=1e-30)).max()
    assert drift < 1e-10, f"SGS-coupled step drifted conserved Ũ totals by {drift}"
    print("✓ conservative-flux SGS step conserves Ũ = U + δU")


def test_sgs_closure_step_is_high_order_in_time():
    """With the closure active, SSP-RK3 must retain better than 1st-order accuracy.

    The stage-lagged delta_U subtraction this replaced degraded the scheme to ~1st order.
    """
    dtype = torch.float64
    Ny = Nx = 16

    def run(dt, n_steps):
        s = RHDSolver(Ny=Ny, Nx=Nx, cfl=0.3, dtype=dtype)
        xs = (torch.arange(Nx, dtype=dtype) / Nx).view(1, Nx).expand(Ny, Nx)
        s.initialize(
            1.0 + 0.2 * torch.sin(2 * torch.pi * xs),
            0.1 * torch.cos(2 * torch.pi * xs),
            torch.zeros(Ny, Nx, dtype=dtype),
            1.0 + 0.2 * torch.sin(2 * torch.pi * xs),
        )
        closure = _constant_closure(1e-2, Ny, Nx, dtype)
        for _ in range(n_steps):
            s.step(dt=dt, sgs_closure=closure)
        return s.prims.clone()

    T = 4e-3
    ref = run(T / 256, 256)
    e_coarse = (run(T / 8, 8) - ref).abs().max().item()
    e_fine = (run(T / 16, 16) - ref).abs().max().item()
    order = torch.log2(torch.tensor(e_coarse / e_fine)).item()
    assert order > 2.0, f"observed temporal order {order:.2f} (expected > 2 for SSP-RK3)"
    print(f"✓ SGS-coupled SSP-RK3 observed temporal order {order:.2f}")


def test_sgs_viscous_flux_divergence_is_conservative():
    """The 4th-order SGS face interpolation must telescope to zero on a periodic grid."""
    from src.solver import _face_interp4
    dtype = torch.float64
    N = 64
    xs = torch.arange(N, dtype=dtype) / N
    f = torch.sin(2 * torch.pi * xs) + 0.3 * torch.cos(6 * torch.pi * xs)
    face = _face_interp4(f, dim=-1)
    div = face - torch.roll(face, 1, dims=-1)
    assert div.sum().abs() < 1e-12, f"flux divergence did not telescope: {div.sum()}"

    # 4th-order accuracy of the face interpolation itself
    errs = []
    for n in (32, 64):
        xc = (torch.arange(n, dtype=dtype) + 0.5) / n
        g = torch.sin(2 * torch.pi * xc)
        exact_face = torch.sin(2 * torch.pi * (xc + 0.5 / n))
        errs.append((_face_interp4(g, dim=-1) - exact_face).abs().max().item())
    order = torch.log2(torch.tensor(errs[0] / errs[1])).item()
    assert order > 3.5, f"face interpolation order {order:.2f}, expected ~4"
    print(f"✓ SGS face interpolation is conservative and {order:.2f}-order accurate")


def test_clip_backscatter():
    """Dissipative stress must survive untouched; backscattering stress must be removed."""
    dtype = torch.float64
    N = 32
    dx = dy = 1.0 / N
    xs = (torch.arange(N, dtype=dtype) / N).view(1, N).expand(N, N)
    prims = torch.stack([
        torch.ones(N, N, dtype=dtype),
        0.3 * torch.sin(2 * torch.pi * xs),
        torch.zeros(N, N, dtype=dtype),
        torch.ones(N, N, dtype=dtype),
    ], dim=-1)

    g = (torch.roll(prims[..., 1], -1, -1) - torch.roll(prims[..., 1], 1, -1)) / (2 * dx)
    S = torch.zeros(N, N, 4, 4, dtype=dtype)
    S[..., 1, 1] = g

    pi = lambda T: (T[..., 1, 1] * g)

    for mode in ("zero", "project"):
        kept = clip_backscatter(-S, prims, dx, dy, mode=mode)
        assert torch.allclose(kept, -S), f"{mode}: purely dissipative stress was modified"

        removed = clip_backscatter(S, prims, dx, dy, mode=mode)
        assert pi(S).sum() > 0, "test setup: expected backscattering stress"
        assert pi(removed).max() < 1e-20, \
            f"{mode}: backscatter survived, max Pi = {pi(removed).max()}"

    # "project" must preserve the components orthogonal to the strain
    tau = S.clone()
    tau[..., 1, 2] = tau[..., 2, 1] = 0.7
    proj = clip_backscatter(tau, prims, dx, dy, mode="project")
    assert torch.allclose(proj[..., 1, 2], tau[..., 1, 2]), \
        "project mode altered a shear component orthogonal to the clipped strain"

    print("✓ clip_backscatter removes backscatter and preserves dissipative stress")


def test_smooth_sgs_tensor_transfer():
    """The (1,2,1)/4 filter must kill 2dx, strongly damp 3dx, and preserve large scales."""
    dtype = torch.float64
    # N must be divisible by every test wavelength, else the mode wraps
    # discontinuously and its amplitude is not recoverable from the max.
    N = 48
    i = torch.arange(N, dtype=dtype)
    expected = {2: 0.0, 3: 0.25, 4: 0.5, 6: 0.75, 8: 0.854, 16: 0.962}
    for wavelength, want in expected.items():
        field = torch.cos(2 * torch.pi * i / wavelength).view(1, N).expand(N, N)
        tau = torch.zeros(N, N, 4, 4, dtype=dtype)
        tau[..., 1, 1] = field
        out = smooth_sgs_tensor(tau, n_passes=1)
        got = out[..., 1, 1].abs().max() / field.abs().max()
        assert abs(float(got) - want) < 0.02, \
            f"{wavelength}dx: transfer {float(got):.3f}, expected ~{want}"

    # n_passes=0 must be a no-op
    tau = torch.randn(8, 8, 4, 4, dtype=dtype)
    assert torch.equal(smooth_sgs_tensor(tau, 0), tau)
    print("✓ smooth_sgs_tensor has the intended scale-selective transfer")


def test_clip_backscatter_fraction():
    """fraction interpolates between a strict clip and no clipping at all."""
    dtype = torch.float64
    N = 32
    dx = dy = 1.0 / N
    xs = (torch.arange(N, dtype=dtype) / N).view(1, N).expand(N, N)
    prims = torch.stack([
        torch.ones(N, N, dtype=dtype),
        0.3 * torch.sin(2 * torch.pi * xs),
        torch.zeros(N, N, dtype=dtype),
        torch.ones(N, N, dtype=dtype),
    ], dim=-1)
    g = (torch.roll(prims[..., 1], -1, -1) - torch.roll(prims[..., 1], 1, -1)) / (2 * dx)
    S = torch.zeros(N, N, 4, 4, dtype=dtype)
    S[..., 1, 1] = g
    pi_of = lambda T: (T[..., 1, 1] * g).sum()

    full = pi_of(S)
    assert full > 0
    for mode in ("zero", "project"):
        # fraction=1.0 is a no-op
        assert torch.equal(clip_backscatter(S, prims, dx, dy, mode=mode, fraction=1.0), S)
        # fraction=0.0 removes it
        assert pi_of(clip_backscatter(S, prims, dx, dy, mode=mode, fraction=0.0)) < 1e-20
        # intermediate is monotone and bracketed
        prev = 0.0
        for frac in (0.25, 0.5, 0.75):
            got = float(pi_of(clip_backscatter(S, prims, dx, dy, mode=mode, fraction=frac)))
            assert 0.0 < got < float(full), f"{mode} frac={frac}: Pi={got} outside (0, {full})"
            assert got > prev, f"{mode}: retained backscatter not monotone in fraction"
            prev = got
    print("✓ clip_backscatter fraction interpolates monotonically")


if __name__ == "__main__":
    test_prim_cons_roundtrip()
    test_conservation()
    test_sod_shock_tube()
    test_mp5_convergence()
    test_mp5_monotonicity()
    test_riemann_solvers()
    test_reconstruction_modes()
    test_compute_sgs_viscous_fluxes()
    test_compute_transition_sensor()
    test_clip_sgs_tensor()
    test_clip_backscatter()
    test_smooth_sgs_tensor_transfer()
    test_clip_backscatter_fraction()
    test_flux_closure_conservation()
    test_c2p_newton_derivative()
    test_c2p_extreme_state_roundtrip()
    test_c2p_failure_is_not_laundered()
    test_hllc_resolves_contact_and_shear_exactly()
    test_sgs_closure_step_conserves()
    test_sgs_closure_step_is_high_order_in_time()
    test_sgs_viscous_flux_divergence_is_conservative()
    print("\nAll solver tests passed! ✓")

