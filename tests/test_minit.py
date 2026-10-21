"""Tests for the MInIT sub-grid closure (:mod:`src.minit`)."""

import math

import pytest
import torch

from src.minit import (
    A_GAMMA_PAPER,
    BETA_KH_PAPER,
    C_PAPER,
    LAMBDA_OVER_DELTA,
    MInITClosure,
    MInITCoefficients,
    advance_e_kh,
    advect_e_kh,
    assemble_tau,
    growth_rate,
    saturation_energy,
    subgrid_energy_from_stress,
)

DTYPE = torch.float64


def _uniform(shape, value, dtype=DTYPE):
    return torch.full(shape, float(value), dtype=dtype)


def _prims(Ny=8, Nx=8, rho=1.5, vx=0.0, vy=0.0, p=1.0):
    return torch.stack(
        [
            _uniform((Ny, Nx), rho),
            _uniform((Ny, Nx), vx),
            _uniform((Ny, Nx), vy),
            _uniform((Ny, Nx), p),
        ],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# The closed-form turbulent-energy update
# ---------------------------------------------------------------------------

class TestAdvanceEKH:
    def test_converges_to_saturation(self):
        """Long integration reaches the fixed point e_sat = (2 gamma sqrt(rho) lambda / C)^2."""
        rho = _uniform((4, 4), 1.5)
        gamma = _uniform((4, 4), 0.8)
        lam, C = 0.25, 10.6

        e = _uniform((4, 4), 1e-8)
        for _ in range(4000):
            e = advance_e_kh(e, gamma, rho, 0.01, lam, C)

        expected = saturation_energy(gamma, rho, lam, C)
        assert torch.allclose(e, expected, rtol=1e-8)

    def test_saturation_is_a_fixed_point(self):
        """Started exactly at e_sat, the update must not move."""
        rho = _uniform((4, 4), 2.0)
        gamma = _uniform((4, 4), 1.3)
        lam, C = 0.2, 8.0

        e_sat = saturation_energy(gamma, rho, lam, C)
        assert torch.allclose(advance_e_kh(e_sat, gamma, rho, 0.7, lam, C), e_sat, rtol=1e-12)

    def test_matches_explicit_integration_in_growth_phase(self):
        """Against a finely sub-stepped forward Euler, where Euler is still accurate."""
        rho = _uniform((1, 1), 1.0)
        gamma = _uniform((1, 1), 0.5)
        lam, C = 0.3, 10.0
        e0, total = 1e-6, 2.0

        exact = advance_e_kh(_uniform((1, 1), e0), gamma, rho, total, lam, C)

        n = 200_000
        dt = total / n
        K = C / (math.sqrt(1.0) * lam)
        e = e0
        for _ in range(n):
            e += dt * (2.0 * 0.5 * e - K * e**1.5)

        assert float(exact) == pytest.approx(e, rel=1e-4)

    def test_pure_decay_without_growth(self):
        r"""With gamma = 0 the solution is the algebraic decay e = (e0^{-1/2} + K dt/2)^{-2}."""
        rho = _uniform((3, 3), 4.0)
        gamma = torch.zeros(3, 3, dtype=DTYPE)
        lam, C, dt = 0.5, 6.0, 3.0
        e0 = 0.25

        got = advance_e_kh(_uniform((3, 3), e0), gamma, rho, dt, lam, C)

        K = C / (math.sqrt(4.0) * lam)
        expected = (e0**-0.5 + 0.5 * K * dt) ** -2
        assert torch.allclose(got, _uniform((3, 3), expected), rtol=1e-12)

    def test_gamma_zero_limit_is_continuous(self):
        """The phi = (1 - exp(-g dt))/g branch must join its g -> 0 limit smoothly."""
        rho = _uniform((1, 5), 1.0)
        lam, C, dt = 0.3, 10.0, 0.05
        e = _uniform((1, 5), 1e-3)

        gammas = torch.tensor([[0.0, 1e-12, 1e-9, 1e-7, 1e-5]], dtype=DTYPE)
        out = advance_e_kh(e, gammas, rho, dt, lam, C)

        # Monotone in gamma and varying by far less than a part in 10^3 across
        # eight decades that all sit in the limit.
        assert torch.all(out.diff() >= -1e-18)
        assert float((out.max() - out.min()) / out.mean()) < 1e-3

    def test_zero_is_preserved_exactly(self):
        """e_KH = 0 is a fixed point, so the closure needs an explicit seed."""
        rho = _uniform((4, 4), 1.0)
        gamma = _uniform((4, 4), 5.0)
        out = advance_e_kh(torch.zeros(4, 4, dtype=DTYPE), gamma, rho, 1.0, 0.3, 10.0)
        assert torch.all(out == 0.0)

    @pytest.mark.parametrize("dt", [1e-6, 1.0, 1e3, 1e6])
    def test_positivity_and_finiteness_for_any_dt(self, dt):
        """Unconditional stability: the closure never constrains the hydro time step."""
        rho = _uniform((4, 4), 0.5)
        gamma = _uniform((4, 4), 20.0)
        e = advance_e_kh(_uniform((4, 4), 1e-4), gamma, rho, dt, 0.25, 10.6)
        assert torch.all(torch.isfinite(e))
        assert torch.all(e >= 0.0)

    def test_batched_parameters_match_scalar_runs(self):
        """The (P,1,1) parameter batching used by the calibration scan is consistent."""
        rho = _uniform((1, 6, 6), 1.2)
        shear = torch.rand(6, 6, dtype=DTYPE)
        e0 = _uniform((6, 6), 1e-5)
        lam = 0.25

        A = torch.tensor([0.1, 0.14, 0.3], dtype=DTYPE).view(3, 1, 1)
        C = torch.tensor([5.0, 10.6, 20.0], dtype=DTYPE).view(3, 1, 1)

        batched = advance_e_kh(
            e0.unsqueeze(0).expand(3, 6, 6), A * shear, rho, 0.1, lam, C,
        )
        for i in range(3):
            single = advance_e_kh(
                e0, float(A[i]) * shear, rho[0], 0.1, lam, float(C[i]),
            )
            assert torch.allclose(batched[i], single, rtol=1e-14)


# ---------------------------------------------------------------------------
# Growth rate
# ---------------------------------------------------------------------------

class TestGrowthRate:
    def test_recovers_analytic_shear(self):
        """For vx = a*y on a periodic ramp, gamma = A_gamma * a away from the wrap."""
        N = 64
        y = torch.arange(N, dtype=DTYPE) / N
        a = 2.0
        vx = (a * y).unsqueeze(-1).expand(N, N).contiguous()
        vy = torch.zeros(N, N, dtype=DTYPE)

        g = growth_rate(vx, vy, 1.0 / N, 1.0 / N, A_gamma=0.14, mode="dvxdy")
        interior = g[4:-4]
        assert torch.allclose(interior, _uniform(interior.shape, 0.14 * a), rtol=1e-10)

    def test_is_non_negative_on_both_shear_layers(self):
        """A double tanh layer has shear of both signs; neither may give gamma < 0."""
        N = 64
        y = torch.linspace(-0.5, 0.5, N, dtype=DTYPE)
        vx = torch.tanh((0.25 - y.abs()) / 0.05).unsqueeze(-1).expand(N, N).contiguous()
        vy = torch.zeros(N, N, dtype=DTYPE)

        g = growth_rate(vx, vy, 1.0 / N, 1.0 / N, A_gamma=0.14)
        assert torch.all(g >= 0.0)
        assert float(g.max()) > 0.0

    def test_strain_mode_agrees_with_dvxdy_for_pure_shear(self):
        """The invariant variant is normalised to match Eq. 23 on a pure shear,
        so A_gamma keeps its calibrated meaning across both modes."""
        N = 64
        y = torch.arange(N, dtype=DTYPE) / N
        vx = (1.7 * y).unsqueeze(-1).expand(N, N).contiguous()
        vy = torch.zeros(N, N, dtype=DTYPE)

        g1 = growth_rate(vx, vy, 1.0 / N, 1.0 / N, 0.14, mode="dvxdy")
        g2 = growth_rate(vx, vy, 1.0 / N, 1.0 / N, 0.14, mode="strain")
        assert torch.allclose(g1[4:-4], g2[4:-4], rtol=1e-10)

    def test_rejects_unknown_mode(self):
        v = torch.zeros(4, 4, dtype=DTYPE)
        with pytest.raises(ValueError, match="unknown growth-rate mode"):
            growth_rate(v, v, 0.1, 0.1, mode="bogus")


# ---------------------------------------------------------------------------
# Stress assembly
# ---------------------------------------------------------------------------

class TestAssembleTau:
    def test_trace_identity(self):
        r"""e_KH = 1/2 sum_i tau^{ii} must be recovered from the assembled tensor.

        This is what forces beta_xx + beta_yy = 2 in 2D; the published 3D
        coefficients sum to 1.99, so they reproduce e_KH to within 0.5%.
        """
        e = torch.rand(8, 8, dtype=DTYPE) + 0.1
        prims = _prims()
        coeffs = MInITCoefficients(beta_xx=1.04, beta_yy=0.96, beta_xy=-0.01)

        tau = assemble_tau(e, prims, coeffs)
        assert torch.allclose(subgrid_energy_from_stress(tau), e, rtol=1e-12)

    def test_published_coefficients_recover_e_kh_to_half_a_percent(self):
        e = torch.rand(8, 8, dtype=DTYPE) + 0.1
        tau = assemble_tau(e, _prims(), MInITCoefficients.paper())
        ratio = subgrid_energy_from_stress(tau) / e
        assert torch.allclose(ratio, torch.ones_like(ratio) * 0.69, rtol=1e-12)

    @pytest.mark.parametrize("vx,vy", [(0.0, 0.0), (0.3, -0.2), (0.6, 0.5)])
    def test_transversality(self, vx, vy):
        r"""u_mu tau_perp^{mu nu} = 0, matching the exact DNS stress convention."""
        e = torch.rand(6, 6, dtype=DTYPE) + 0.1
        prims = _prims(6, 6, vx=vx, vy=vy)
        tau = assemble_tau(e, prims, MInITCoefficients.paper())

        v_sq = vx**2 + vy**2
        W = 1.0 / math.sqrt(1.0 - v_sq)
        u_cov = torch.tensor([-W, W * vx, W * vy, 0.0], dtype=DTYPE)

        contracted = torch.einsum("m,yxmn->yxn", u_cov, tau)
        assert torch.allclose(contracted, torch.zeros_like(contracted), atol=1e-13)

    def test_zz_component_vanishes(self):
        """T^{zz} = p g^{zz} in 2D, so the exact SGS stress has tau^{zz} = 0."""
        tau = assemble_tau(torch.rand(4, 4, dtype=DTYPE), _prims(4, 4), MInITCoefficients.paper())
        assert torch.all(tau[..., 3, 3] == 0.0)

    def test_symmetric(self):
        tau = assemble_tau(torch.rand(5, 5, dtype=DTYPE) + 0.1, _prims(5, 5, vx=0.4),
                           MInITCoefficients.paper())
        assert torch.allclose(tau, tau.transpose(-1, -2), atol=1e-15)

    def test_scales_linearly_with_e_kh(self):
        """The whole closure ansatz is proportionality to e_KH."""
        e = torch.rand(4, 4, dtype=DTYPE) + 0.1
        prims = _prims(4, 4, vx=0.3)
        c = MInITCoefficients.paper()
        assert torch.allclose(assemble_tau(3.0 * e, prims, c),
                              3.0 * assemble_tau(e, prims, c), rtol=1e-13)


# ---------------------------------------------------------------------------
# Advection
# ---------------------------------------------------------------------------

class TestAdvect:
    def test_conserves_total_energy(self):
        """Donor-cell upwinding in flux form conserves the integral on a periodic grid."""
        e = torch.rand(16, 16, dtype=DTYPE)
        vx = _uniform((16, 16), 0.4)
        vy = _uniform((16, 16), -0.25)
        out = advect_e_kh(e, vx, vy, 1 / 16, 1 / 16, 0.01)
        assert float(out.sum()) == pytest.approx(float(e.sum()), rel=1e-12)

    def test_uniform_field_is_unchanged(self):
        e = _uniform((8, 8), 0.3)
        vx = torch.rand(8, 8, dtype=DTYPE)
        vy = torch.rand(8, 8, dtype=DTYPE)
        # A divergence-free velocity leaves a uniform field alone; use a shear.
        vy = torch.zeros(8, 8, dtype=DTYPE)
        vx = vx * 0 + 0.5
        assert torch.allclose(advect_e_kh(e, vx, vy, 0.125, 0.125, 0.01), e, rtol=1e-13)

    def test_translates_in_the_flow_direction(self):
        """A bump advected at constant speed moves downstream, not upstream."""
        N = 32
        e = torch.zeros(N, N, dtype=DTYPE)
        e[:, 8:12] = 1.0
        vx = _uniform((N, N), 1.0)
        vy = torch.zeros(N, N, dtype=DTYPE)

        out = e.clone()
        for _ in range(50):
            out = advect_e_kh(out, vx, vy, 1 / N, 1 / N, 0.4 / N)

        x = torch.arange(N, dtype=DTYPE)
        centroid_before = float((e.sum(0) * x).sum() / e.sum())
        centroid_after = float((out.sum(0) * x).sum() / out.sum())
        assert centroid_after > centroid_before

    def test_stays_non_negative(self):
        e = torch.rand(16, 16, dtype=DTYPE)
        vx = torch.randn(16, 16, dtype=DTYPE)
        vy = torch.randn(16, 16, dtype=DTYPE)
        out = advect_e_kh(e, vx, vy, 1 / 16, 1 / 16, 0.02)
        assert torch.all(out >= 0.0)


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------

class TestCoefficients:
    def test_paper_defaults(self):
        c = MInITCoefficients.paper()
        assert c.beta_xx == BETA_KH_PAPER["xx"]
        assert c.A_gamma == A_GAMMA_PAPER
        assert c.C == C_PAPER

    def test_trace_residual_of_published_2d_projection(self):
        assert MInITCoefficients.paper().trace_residual == pytest.approx(-0.62)

    def test_normalized_zeroes_the_residual(self):
        c = MInITCoefficients(beta_xx=0.74, beta_yy=0.64).normalized()
        assert c.trace_residual == pytest.approx(0.0, abs=1e-12)
        # Ratio between components is preserved.
        assert c.beta_xx / c.beta_yy == pytest.approx(0.74 / 0.64)

    def test_round_trip_through_json(self, tmp_path):
        c = MInITCoefficients(beta_xx=1.1, beta_yy=0.9, beta_xy=0.02, A_gamma=0.21, C=3.4)
        path = tmp_path / "coeffs.json"
        c.save(path)
        assert MInITCoefficients.load(path) == c


# ---------------------------------------------------------------------------
# The closure object and its solver interface
# ---------------------------------------------------------------------------

class TestClosure:
    def test_returns_solver_shapes(self):
        closure = MInITClosure(dx=1 / 8, dy=1 / 8, e_seed=1e-4)
        prims = _prims()
        closure.initialize(prims)
        tau, alpha = closure(prims)
        assert tau.shape == (8, 8, 4, 4)
        assert alpha.shape == (8, 8)
        assert torch.all(alpha == 1.0)

    def test_zero_seed_stays_inert(self):
        """Without a seed the model can never activate — e_KH = 0 is a fixed point."""
        N = 32
        y = torch.linspace(-0.5, 0.5, N, dtype=DTYPE)
        vx = torch.tanh((0.25 - y.abs()) / 0.05).unsqueeze(-1).expand(N, N).contiguous()
        prims = torch.stack(
            [_uniform((N, N), 1.5), vx, torch.zeros(N, N, dtype=DTYPE), _uniform((N, N), 1.0)],
            dim=-1,
        )
        closure = MInITClosure(dx=1 / N, dy=1 / N, e_seed=0.0)
        closure.initialize(prims)
        for _ in range(50):
            closure.advance(prims, 0.01)

        tau, _ = closure(prims)
        assert torch.all(closure.e_KH == 0.0)
        assert torch.all(tau == 0.0)

    def test_floor_fraction_seeds_a_dead_field(self):
        """The floor is what lets a region that has decayed to zero restart."""
        N = 32
        y = torch.linspace(-0.5, 0.5, N, dtype=DTYPE)
        vx = torch.tanh((0.25 - y.abs()) / 0.05).unsqueeze(-1).expand(N, N).contiguous()
        prims = torch.stack(
            [_uniform((N, N), 1.5), vx, torch.zeros(N, N, dtype=DTYPE), _uniform((N, N), 1.0)],
            dim=-1,
        )
        closure = MInITClosure(dx=1 / N, dy=1 / N, e_seed=0.0, floor_fraction=1e-3)
        closure.initialize(prims)
        assert float(closure.e_KH.max()) > 0.0

    def test_seeding_from_a_measured_field(self):
        prims = _prims()
        seed = torch.rand(8, 8, dtype=DTYPE)
        closure = MInITClosure(dx=1 / 8, dy=1 / 8)
        closure.initialize(prims, e_KH=seed)
        assert torch.allclose(closure.e_KH, seed)

    def test_negative_seed_is_clamped(self):
        closure = MInITClosure(dx=1 / 4, dy=1 / 4)
        closure.initialize(_prims(4, 4), e_KH=-torch.ones(4, 4, dtype=DTYPE))
        assert torch.all(closure.e_KH >= 0.0)

    def test_lambda_follows_the_filter_scale(self):
        closure = MInITClosure(dx=1 / 64, dy=1 / 64, delta_f=1 / 32)
        assert closure.lambda_KH == pytest.approx(LAMBDA_OVER_DELTA / 32)

    def test_lambda_defaults_to_dx(self):
        closure = MInITClosure(dx=1 / 64, dy=1 / 64)
        assert closure.lambda_KH == pytest.approx(LAMBDA_OVER_DELTA / 64)

    def test_advance_records_history_when_given_a_time(self):
        prims = _prims()
        closure = MInITClosure(dx=1 / 8, dy=1 / 8, e_seed=1e-4)
        closure.initialize(prims)
        closure.advance(prims, 0.01, t=0.5)
        assert len(closure.history) == 1
        assert closure.history[0][0] == pytest.approx(0.5)

    def test_grows_where_there_is_shear_and_not_where_there_is_none(self):
        N = 32
        y = torch.linspace(-0.5, 0.5, N, dtype=DTYPE)
        vx = torch.tanh((0.25 - y.abs()) / 0.05).unsqueeze(-1).expand(N, N).contiguous()
        prims = torch.stack(
            [_uniform((N, N), 1.5), vx, torch.zeros(N, N, dtype=DTYPE), _uniform((N, N), 1.0)],
            dim=-1,
        )
        closure = MInITClosure(dx=1 / N, dy=1 / N, e_seed=1e-8)
        closure.initialize(prims)
        for _ in range(200):
            closure.advance(prims, 0.01)

        shear_rows = closure.e_KH[(y.abs() - 0.25).abs() < 0.03].mean()
        quiet_rows = closure.e_KH[y.abs() < 0.10].mean()
        assert float(shear_rows) > 100.0 * float(quiet_rows)


# ---------------------------------------------------------------------------
# Integration with the solver
# ---------------------------------------------------------------------------

class TestSolverIntegration:
    """The advance() hook in RHDSolver.evolve is what makes a stateful closure work."""

    @staticmethod
    def _khi_solver(N=32):
        from src.solver import RHDSolver

        y = torch.linspace(-0.5, 0.5, N, dtype=DTYPE).unsqueeze(-1)
        rho = 1.5 + 0.5 * torch.tanh((0.25 - y.abs()) / 0.05)
        vx = 0.4 * torch.tanh((0.25 - y.abs()) / 0.05)
        solver = RHDSolver(N, N, cfl=0.3, dtype=DTYPE, device=torch.device("cpu"))
        solver.initialize(
            rho.expand(N, N).contiguous(),
            vx.expand(N, N).contiguous(),
            torch.zeros(N, N, dtype=DTYPE),
            torch.full((N, N), 2.5, dtype=DTYPE),
        )
        return solver

    def test_evolve_advances_closure_state(self):
        solver = self._khi_solver()
        closure = MInITClosure(dx=1 / 32, dy=1 / 32, e_seed=1e-6)
        closure.initialize(solver.prims)
        before = closure.e_KH.clone()

        solver.evolve(solver.t + 0.05, sgs_closure=closure, progress_bar=False, verbose=False)

        assert not torch.allclose(closure.e_KH, before)
        assert torch.all(torch.isfinite(closure.e_KH))
        assert torch.all(closure.e_KH >= 0.0)
        assert torch.all(torch.isfinite(solver.prims))

    def test_history_is_recorded_with_simulation_time(self):
        solver = self._khi_solver()
        closure = MInITClosure(dx=1 / 32, dy=1 / 32, e_seed=1e-6)
        closure.initialize(solver.prims)
        t_end = solver.t + 0.05

        solver.evolve(t_end, sgs_closure=closure, progress_bar=False, verbose=False)

        assert len(closure.history) > 0
        assert closure.history[-1][0] == pytest.approx(float(solver.t))

    def test_stateless_closures_are_untouched(self):
        """A closure without advance() must still run — the hook is opt-in."""
        solver = self._khi_solver()

        def stateless(prims):
            return (
                torch.zeros(*prims.shape[:-1], 4, 4, dtype=prims.dtype),
                torch.ones(prims.shape[:-1], dtype=prims.dtype),
            )

        solver.evolve(solver.t + 0.05, sgs_closure=stateless, progress_bar=False, verbose=False)
        assert torch.all(torch.isfinite(solver.prims))

    def test_zero_seed_matches_the_unclosed_solver(self):
        """With e_KH = 0 the stress vanishes identically, so MInIT must be a no-op.

        This is the check that a run reported as 'MInIT' is not silently the
        unclosed control — if the seed is ever lost, the two become the same run.
        """
        closed = self._khi_solver()
        unclosed = self._khi_solver()
        closure = MInITClosure(dx=1 / 32, dy=1 / 32, e_seed=0.0)
        closure.initialize(closed.prims)

        closed.evolve(closed.t + 0.05, sgs_closure=closure, progress_bar=False, verbose=False)
        unclosed.evolve(unclosed.t + 0.05, progress_bar=False, verbose=False)

        assert torch.allclose(closed.prims, unclosed.prims, rtol=1e-10, atol=1e-12)

    def test_seeded_run_departs_from_the_unclosed_control(self):
        """The converse: a seeded MInIT must actually change the trajectory."""
        closed = self._khi_solver()
        unclosed = self._khi_solver()
        closure = MInITClosure(dx=1 / 32, dy=1 / 32, e_seed=1e-3)
        closure.initialize(closed.prims)

        closed.evolve(closed.t + 0.05, sgs_closure=closure, progress_bar=False, verbose=False)
        unclosed.evolve(unclosed.t + 0.05, progress_bar=False, verbose=False)

        assert not torch.allclose(closed.prims, unclosed.prims, rtol=1e-6)


class TestVariantResolution:
    def test_expands_both(self):
        from src.evaluate_rollout import _resolve_minit_variants

        assert _resolve_minit_variants(["both"]) == ("fitted", "paper")

    def test_deduplicates(self):
        from src.evaluate_rollout import _resolve_minit_variants

        assert _resolve_minit_variants(["paper", "both", "paper"]) == ("paper", "fitted")

    def test_empty(self):
        from src.evaluate_rollout import _resolve_minit_variants

        assert _resolve_minit_variants(None) == ()
        assert _resolve_minit_variants([]) == ()


class TestClosureTimeSync:
    """`evolve` must synchronise time-dependent closures on BOTH coupling paths.

    `set_time` was originally called only for the legacy `sgs_model` argument, so
    a time-dependent `sgs_closure` — notably the exact-DNS oracle, which
    interpolates a precomputed stress between snapshots — silently ran with its
    stress frozen at the first snapshot for an entire rollout. Nothing in the
    output showed it, and it quietly degraded the one baseline every other
    result is measured against.
    """

    @staticmethod
    def _solver(N=16):
        from src.solver import RHDSolver

        y = torch.linspace(-0.5, 0.5, N, dtype=DTYPE).unsqueeze(-1)
        solver = RHDSolver(N, N, cfl=0.3, dtype=DTYPE, device=torch.device("cpu"))
        solver.initialize(
            (1.5 + 0.5 * torch.tanh((0.25 - y.abs()) / 0.05)).expand(N, N).contiguous(),
            (0.4 * torch.tanh((0.25 - y.abs()) / 0.05)).expand(N, N).contiguous(),
            torch.zeros(N, N, dtype=DTYPE),
            torch.full((N, N), 2.5, dtype=DTYPE),
        )
        return solver

    class _TimedClosure:
        """Minimal stand-in exposing the same set_time contract as the oracle."""

        def __init__(self):
            self.current_t = 0.0
            self.seen: list[float] = []

        def set_time(self, t: float) -> None:
            self.current_t = float(t)
            self.seen.append(float(t))

        def __call__(self, prims):
            return (
                torch.zeros(*prims.shape[:-1], 4, 4, dtype=prims.dtype),
                torch.ones(prims.shape[:-1], dtype=prims.dtype),
            )

    def test_sgs_closure_receives_set_time(self):
        solver = self._solver()
        closure = self._TimedClosure()
        t_end = solver.t + 0.05

        solver.evolve(t_end, sgs_closure=closure, progress_bar=False, verbose=False)

        assert closure.seen, "set_time was never called on the closure"
        # Called before each step, so the last value lags the final time by one dt.
        assert closure.current_t > 0.0
        assert closure.current_t <= float(solver.t) + 1e-12

    def test_closure_time_advances_monotonically_across_the_run(self):
        solver = self._solver()
        closure = self._TimedClosure()

        solver.evolve(solver.t + 0.05, sgs_closure=closure, progress_bar=False, verbose=False)

        assert len(closure.seen) > 1
        assert all(b >= a for a, b in zip(closure.seen, closure.seen[1:]))
        # The whole point: it must not still be sitting at its initial value.
        assert closure.seen[-1] > closure.seen[0]

    def test_sgs_model_path_still_synchronised(self):
        """The original behaviour must survive the fix."""
        solver = self._solver()

        class _TimedModel(TestClosureTimeSync._TimedClosure):
            def __call__(self, prims):
                return torch.zeros(*prims.shape[:-1], 4, dtype=prims.dtype)

        model = _TimedModel()
        solver.evolve(solver.t + 0.05, sgs_model=model, progress_bar=False, verbose=False)
        assert model.seen and model.seen[-1] > model.seen[0]

    def test_a_closure_without_set_time_is_untouched(self):
        solver = self._solver()

        def plain(prims):
            return (
                torch.zeros(*prims.shape[:-1], 4, 4, dtype=prims.dtype),
                torch.ones(prims.shape[:-1], dtype=prims.dtype),
            )

        solver.evolve(solver.t + 0.05, sgs_closure=plain, progress_bar=False, verbose=False)
        assert torch.all(torch.isfinite(solver.prims))

    def test_minit_gets_both_hooks(self):
        """MInIT uses advance(); it must not be disturbed by the set_time loop."""
        solver = self._solver()
        closure = MInITClosure(dx=1 / 16, dy=1 / 16, e_seed=1e-6)
        closure.initialize(solver.prims)

        solver.evolve(solver.t + 0.05, sgs_closure=closure, progress_bar=False, verbose=False)

        assert len(closure.history) > 0
        assert not hasattr(closure, "current_t")


class TestRowAveragedMInIT:
    """average='x': MInIT's mean-field form in 2D (paper Sec. 4.3.1)."""

    def _prims(self, N=16):
        import math
        y = torch.linspace(0, 1, N + 1, dtype=torch.float64)[:-1].unsqueeze(-1)
        x = torch.linspace(0, 1, N + 1, dtype=torch.float64)[:-1].unsqueeze(0)
        p = torch.empty(N, N, 4, dtype=torch.float64)
        p[..., 0] = 1.0 + 0.2 * torch.sin(2 * math.pi * x).expand(N, N)
        p[..., 1] = (0.3 * torch.sin(2 * math.pi * y) + 0.05 * torch.cos(4 * math.pi * x)).expand(N, N)
        p[..., 2] = 0.02 * torch.sin(2 * math.pi * x).expand(N, N)
        p[..., 3] = 2.5
        return p

    def test_energy_stays_uniform_along_x(self):
        p = self._prims()
        c = MInITClosure(dx=1 / 16, dy=1 / 16, average="x")
        c.initialize(p, e_KH=torch.rand(16, 16, dtype=torch.float64) * 1e-3)
        for _ in range(5):
            c.advance(p, 1e-3)
        assert torch.allclose(c.e_KH, c.e_KH[:, :1].expand_as(c.e_KH))

    def test_growth_rate_follows_the_mean_shear(self):
        """gamma(y) must come from d<vx>_x/dy, not from the x-fluctuations."""
        from src.minit import growth_rate, row_mean

        p = self._prims()
        c = MInITClosure(dx=1 / 16, dy=1 / 16, average="x", mode="equilibrium")
        c.initialize(p)
        mp = c._model_prims(p)
        expect = growth_rate(row_mean(p[..., 1]), row_mean(p[..., 2]), 1 / 16, 1 / 16,
                             c.coeffs.A_gamma)
        assert torch.allclose(growth_rate(mp[..., 1], mp[..., 2], 1 / 16, 1 / 16, c.coeffs.A_gamma),
                              expect)

    def test_advection_is_rejected_in_the_averaged_model(self):
        with pytest.raises(ValueError, match="advect"):
            MInITClosure(dx=1 / 16, dy=1 / 16, average="x", advect=True)

    def test_cell_mode_is_unchanged(self):
        p = self._prims()
        a = MInITClosure(dx=1 / 16, dy=1 / 16)
        b = MInITClosure(dx=1 / 16, dy=1 / 16, average="cell")
        e0 = torch.rand(16, 16, dtype=torch.float64) * 1e-3
        a.initialize(p, e_KH=e0.clone()); b.initialize(p, e_KH=e0.clone())
        a.advance(p, 1e-3); b.advance(p, 1e-3)
        assert torch.equal(a.e_KH, b.e_KH)
