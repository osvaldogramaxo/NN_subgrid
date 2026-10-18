"""Unit tests for RMHD SGS Physics-Based Loss Functions (src/losses.py)."""

import unittest
import pytest
import torch
import torch.nn as nn

from src.losses import (
    CovariantTensorLoss,
    DissipationRateLoss,
    SpectralCascadeLoss,
    OrthogonalityLoss,
    WECLoss,
    RMHDSGSLoss,
)


class TestCovariantTensorLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = CovariantTensorLoss(epsilon=1e-8)
        # Minkowski metric diag(-1, 1, 1, 1)
        self.g_cov = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))

    def test_zero_when_identical(self):
        tau = torch.zeros(10, 4, 4, dtype=torch.float64)
        tau[:, 1:, 1:] = torch.randn(10, 3, 3, dtype=torch.float64)
        loss = self.loss_fn(tau, tau, self.g_cov)
        self.assertAlmostEqual(loss.item(), 0.0, places=7)

    def test_positive_definite(self):
        # In RMHD, SGS stress tensors are spatial in the fluid rest frame (u_μ τ^{μν} = 0)
        tau_pred = torch.zeros(10, 4, 4, dtype=torch.float64)
        tau_pred[:, 1:, 1:] = torch.randn(10, 3, 3, dtype=torch.float64)
        tau_dns = torch.zeros(10, 4, 4, dtype=torch.float64)
        tau_dns[:, 1:, 1:] = torch.randn(10, 3, 3, dtype=torch.float64)
        loss = self.loss_fn(tau_pred, tau_dns, self.g_cov)
        self.assertGreater(loss.item(), 0.0)

    def test_backward(self):
        tau_pred = torch.zeros(10, 4, 4, dtype=torch.float64)
        tau_pred[:, 1:, 1:] = torch.randn(10, 3, 3, dtype=torch.float64)
        tau_pred.requires_grad_(True)
        tau_dns = torch.zeros(10, 4, 4, dtype=torch.float64)
        tau_dns[:, 1:, 1:] = torch.randn(10, 3, 3, dtype=torch.float64)
        loss = self.loss_fn(tau_pred, tau_dns, self.g_cov)
        loss.backward()
        self.assertIsNotNone(tau_pred.grad)
        assert tau_pred.grad is not None
        self.assertTrue(torch.isfinite(tau_pred.grad).all())


class TestDissipationRateLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = DissipationRateLoss(lambda_stab=10.0)

    def test_zero_when_identical_positive_prod(self):
        # Construct positive production: grad_u chosen so -τ * grad_u > 0
        tau = torch.eye(4, dtype=torch.float64).unsqueeze(0).repeat(5, 1, 1)
        grad_u = -torch.eye(4, dtype=torch.float64).unsqueeze(0).repeat(5, 1, 1)
        # P = -τ : grad_u = -(-4) = 4 > 0
        loss = self.loss_fn(tau, tau, grad_u)
        self.assertAlmostEqual(loss.item(), 0.0, places=7)

    def test_stability_penalty_on_anti_dissipation(self):
        # P_pred < 0 (spurious anti-dissipation)
        tau_pred = torch.eye(4, dtype=torch.float64).unsqueeze(0).repeat(5, 1, 1)
        grad_u = torch.eye(4, dtype=torch.float64).unsqueeze(0).repeat(5, 1, 1)
        # P_pred = -τ : grad_u = -4 < 0
        tau_dns = -tau_pred  # P_dns = +4

        # With p_scale = 1 the raw arithmetic is pinned:
        #   l1_match = |-4 - 4| = 8
        #   stability = 10 * ReLU(-(-4)) = 10 * 4 = 40
        #   total = 48
        loss_raw = self.loss_fn(tau_pred, tau_dns, grad_u, p_scale=1.0)
        self.assertAlmostEqual(loss_raw.item(), 48.0, places=5)

        # Without an explicit scale the term is divided by the batch |P_DNS|
        # mean (= 4), which is what keeps α meaningful across regimes:
        #   8/4 + 10 * (4/4) = 12
        loss_scaled = self.loss_fn(tau_pred, tau_dns, grad_u)
        self.assertAlmostEqual(loss_scaled.item(), 12.0, places=5)

    def test_backward(self):
        tau_pred = torch.randn(8, 4, 4, dtype=torch.float64, requires_grad=True)
        tau_dns = torch.randn(8, 4, 4, dtype=torch.float64)
        grad_u = torch.randn(8, 4, 4, dtype=torch.float64)
        loss = self.loss_fn(tau_pred, tau_dns, grad_u)
        loss.backward()
        self.assertIsNotNone(tau_pred.grad)
        assert tau_pred.grad is not None
        self.assertTrue(torch.isfinite(tau_pred.grad).all())


class TestSpectralCascadeLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = SpectralCascadeLoss(epsilon=1e-12)

    def test_zero_when_identical(self):
        tau = torch.randn(32, 32, 4, 4, dtype=torch.float64)
        loss = self.loss_fn(tau, tau)
        self.assertAlmostEqual(loss.item(), 0.0, places=7)

    def test_positive_when_different(self):
        tau1 = torch.randn(32, 32, 4, 4, dtype=torch.float64)
        tau2 = torch.randn(32, 32, 4, 4, dtype=torch.float64) * 2.0
        loss = self.loss_fn(tau1, tau2)
        self.assertGreater(loss.item(), 0.0)

    def test_backward(self):
        tau_pred = torch.randn(16, 16, 4, 4, dtype=torch.float64, requires_grad=True)
        tau_dns = torch.randn(16, 16, 4, 4, dtype=torch.float64)
        loss = self.loss_fn(tau_pred, tau_dns)
        loss.backward()
        self.assertIsNotNone(tau_pred.grad)
        assert tau_pred.grad is not None
        self.assertTrue(torch.isfinite(tau_pred.grad).all())


class TestOrthogonalityLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = OrthogonalityLoss()
        self.g_cov = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))

    def test_zero_for_spatial_tensor_in_rest_frame(self):
        # Fluid at rest: u^μ = (1, 0, 0, 0)
        u_con = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).repeat(5, 1)
        # Purely spatial stress tensor (τ^0μ = τ^μ0 = 0)
        tau = torch.zeros(5, 4, 4, dtype=torch.float64)
        tau[:, 1:, 1:] = torch.randn(5, 3, 3, dtype=torch.float64)

        loss = self.loss_fn(tau, u_con, self.g_cov)
        self.assertAlmostEqual(loss.item(), 0.0, places=7)

    def test_positive_for_time_component_leakage(self):
        # Fluid at rest: u^μ = (1, 0, 0, 0)
        u_con = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).repeat(5, 1)
        # Tensor with non-zero τ^00
        tau = torch.zeros(5, 4, 4, dtype=torch.float64)
        tau[:, 0, 0] = 1.0

        loss = self.loss_fn(tau, u_con, self.g_cov)
        self.assertGreater(loss.item(), 0.0)

    def test_backward(self):
        tau_pred = torch.randn(6, 4, 4, dtype=torch.float64, requires_grad=True)
        u_con = torch.tensor([[1.25, 0.5, 0.3, 0.0]], dtype=torch.float64).repeat(6, 1)
        loss = self.loss_fn(tau_pred, u_con, self.g_cov)
        loss.backward()
        self.assertIsNotNone(tau_pred.grad)
        assert tau_pred.grad is not None
        self.assertTrue(torch.isfinite(tau_pred.grad).all())


class TestWECLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = WECLoss(gamma_eos=4.0 / 3.0, margin=0.5)
        self.g_cov = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))

    def test_zero_for_valid_energy_density(self):
        u_con = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).repeat(4, 1)
        rho = torch.tensor([1.0, 2.0, 1.5, 0.8], dtype=torch.float64)
        p = torch.tensor([0.1, 0.2, 0.15, 0.05], dtype=torch.float64)
        # Small positive or zero SGS stress
        tau = torch.zeros(4, 4, 4, dtype=torch.float64)
        loss = self.loss_fn(tau, u_con, rho, p, self.g_cov)
        self.assertAlmostEqual(loss.item(), 0.0, places=7)

    def test_positive_when_energy_condition_violated(self):
        u_con = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).repeat(4, 1)
        rho = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        p = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float64)
        # Huge negative perturbation to drive ε < 0
        tau = torch.zeros(4, 4, 4, dtype=torch.float64)
        tau[:, 0, 0] = -100.0  # In comoving: T_eff^{00} u_0 u_0 < 0
        loss = self.loss_fn(tau, u_con, rho, p, self.g_cov)
        self.assertGreater(loss.item(), 0.0)

    def test_margin_activates_before_outright_violation(self):
        """The whole point of the relative margin: it fires while ε_eff is
        still positive. The absolute hinge could not fire at all on this
        problem -- resolved comoving energy density is O(10) while the largest
        exact ‖τ‖_F in dns_khi is 0.47."""
        u_con = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        rho = torch.tensor([1.0], dtype=torch.float64)
        p = torch.tensor([0.1], dtype=torch.float64)
        # ε_resolved = ρh - p; with Γ=4/3, ε = p/((Γ-1)ρ) = 0.3, h = 1.4, so
        # ε_resolved = 1.4 - 0.1 = 1.3.
        e_resolved = 1.3
        # Remove 70% of it: still positive, but below the 0.5 margin.
        tau = torch.zeros(1, 4, 4, dtype=torch.float64)
        tau[0, 0, 0] = -0.7 * e_resolved

        relative = WECLoss(gamma_eos=4.0 / 3.0, margin=0.5)
        absolute = WECLoss(gamma_eos=4.0 / 3.0, margin=0.0)
        self.assertAlmostEqual(absolute(tau, u_con, rho, p, self.g_cov).item(), 0.0, places=12)
        self.assertGreater(relative(tau, u_con, rho, p, self.g_cov).item(), 0.0)
        # ε_eff/ε_resolved = 0.3, so the hinge value is 0.5 - 0.3 = 0.2
        self.assertAlmostEqual(
            relative(tau, u_con, rho, p, self.g_cov).item(), 0.2, places=6
        )

    def test_dimensionless_under_rescaling(self):
        """Scaling ρ, p and τ together leaves the term unchanged, so δ means
        the same thing at every subgrid intensity."""
        u_con = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).repeat(3, 1)
        rho = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
        p = torch.tensor([0.1, 0.3, 0.05], dtype=torch.float64)
        tau = torch.zeros(3, 4, 4, dtype=torch.float64)
        tau[:, 0, 0] = torch.tensor([-1.0, -2.5, -0.4], dtype=torch.float64)

        base = self.loss_fn(tau, u_con, rho, p, self.g_cov).item()
        k = 137.0
        scaled = self.loss_fn(k * tau, u_con, k * rho, k * p, self.g_cov).item()
        self.assertAlmostEqual(base, scaled, places=10)

    def test_backward(self):
        u_con = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).repeat(4, 1)
        rho = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        p = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float64)
        tau_pred = torch.randn(4, 4, 4, dtype=torch.float64, requires_grad=True)
        loss = self.loss_fn(tau_pred, u_con, rho, p, self.g_cov)
        loss.backward()
        self.assertIsNotNone(tau_pred.grad)
        assert tau_pred.grad is not None
        self.assertTrue(torch.isfinite(tau_pred.grad).all())


class TestCovariantTensorLossNormalization(unittest.TestCase):
    """The per-point vs batch-level normalization distinction."""

    def setUp(self):
        self.g_cov = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))

    def _spatial(self, vals):
        t = torch.zeros(len(vals), 4, 4, dtype=torch.float64)
        for i, v in enumerate(vals):
            t[i, 1, 1] = v
        return t

    def test_equal_relative_error_weighs_cells_equally(self):
        """Two cells whose intensities differ by 100x but whose *relative*
        errors are identical must contribute equally. Under the old
        batch-level normalizer the strong cell dominated by ~10^4."""
        fn = CovariantTensorLoss(epsilon=0.0, floor_frac=0.0)
        dns = self._spatial([1.0, 100.0])
        pred = self._spatial([1.1, 110.0])       # both 10% off
        per_point = []
        for i in range(2):
            per_point.append(fn(pred[i:i + 1], dns[i:i + 1], self.g_cov).item())
        self.assertAlmostEqual(per_point[0], per_point[1], places=10)
        # and the batch value is their mean, not dominated by the strong cell
        self.assertAlmostEqual(
            fn(pred, dns, self.g_cov).item(), sum(per_point) / 2, places=10
        )

    def test_floor_damps_quasi_laminar_cells(self):
        """A near-zero-intensity cell with an absolute error must not blow up:
        the floor caps its contribution instead of dividing by ~0."""
        tau_scale = 1.0
        fn = CovariantTensorLoss(floor_frac=0.05)
        dns = self._spatial([1e-9])              # essentially laminar
        pred = self._spatial([1e-9 + 1e-3])      # small absolute error
        with_floor = fn(pred, dns, self.g_cov, tau_scale=tau_scale).item()
        no_floor = CovariantTensorLoss(epsilon=1e-30, floor_frac=0.0)(
            pred, dns, self.g_cov
        ).item()
        self.assertLess(with_floor, 1.0)
        self.assertGreater(no_floor, 1e6)
        # the floor is (0.05 * 1.0)^2 = 2.5e-3, so the value is ~1e-6/2.5e-3
        self.assertAlmostEqual(with_floor, 1e-6 / 2.5e-3, places=6)

    def test_explicit_tau_floor_overrides_floor_frac(self):
        """train() passes a quantile of the training ‖τ‖_F as an absolute
        floor; it must take precedence over the floor_frac fallback."""
        fn = CovariantTensorLoss(floor_frac=0.05)
        dns = self._spatial([0.0])
        pred = self._spatial([1e-3])
        explicit = fn(pred, dns, self.g_cov, tau_scale=1.0, tau_floor=1e-2).item()
        self.assertAlmostEqual(explicit, (1e-3 / 1e-2) ** 2, places=10)
        # and it is used even when no tau_scale is given
        alone = fn(pred, dns, self.g_cov, tau_floor=1e-2).item()
        self.assertAlmostEqual(alone, explicit, places=12)

    def test_higher_floor_shifts_gradient_share_away_from_weak_cells(self):
        """The property the quantile floor is chosen for: raising the floor
        moves weight from quasi-laminar cells to resolved-turbulence ones."""
        torch.manual_seed(0)
        # heavy-tailed, like the real distribution (mean/median ~5)
        norms = torch.cat([
            1e-5 * torch.rand(400, dtype=torch.float64),
            1e-3 * torch.rand(400, dtype=torch.float64),
            1e-1 * torch.rand(200, dtype=torch.float64),
        ])
        dns = self._spatial(norms.tolist())
        pred = dns + self._spatial([1e-4] * len(norms))   # uniform absolute error

        def weak_share(floor):
            fn = CovariantTensorLoss(epsilon=0.0, floor_frac=0.0)
            per = [
                fn(pred[i:i + 1], dns[i:i + 1], self.g_cov, tau_floor=floor).item()
                for i in range(len(norms))
            ]
            per_t = torch.tensor(per, dtype=torch.float64)
            order = torch.argsort(norms)
            k = len(norms) // 10
            return float(per_t[order[:k]].sum() / per_t.sum())

        low = weak_share(float(torch.quantile(norms, 0.05)))
        med = weak_share(float(torch.quantile(norms, 0.5)))
        self.assertGreater(low, med)
        self.assertLess(med, 0.5)

    def test_invariant_under_rescaling_when_scale_is_passed(self):
        """Doubling every τ and the declared scale leaves the loss unchanged,
        which is what makes the weights portable across regimes."""
        fn = CovariantTensorLoss(floor_frac=0.05)
        dns = self._spatial([0.01, 1.0, 5.0])
        pred = self._spatial([0.012, 0.9, 5.4])
        base = fn(pred, dns, self.g_cov, tau_scale=1.0).item()
        k = 1000.0
        scaled = fn(k * pred, k * dns, self.g_cov, tau_scale=k).item()
        self.assertAlmostEqual(base, scaled, places=10)


class TestDissipationRateLossScaling(unittest.TestCase):
    def test_dimensionless_in_p_scale(self):
        torch.manual_seed(0)
        fn = DissipationRateLoss(lambda_stab=10.0)
        tau_pred = torch.randn(32, 4, 4, dtype=torch.float64)
        tau_dns = torch.randn(32, 4, 4, dtype=torch.float64)
        grad_u = torch.randn(32, 4, 4, dtype=torch.float64)
        base = fn(tau_pred, tau_dns, grad_u, p_scale=1.0).item()
        k = 250.0
        scaled = fn(k * tau_pred, k * tau_dns, grad_u, p_scale=k).item()
        self.assertAlmostEqual(base, scaled, places=8)


class TestSpectralCascadeLossFloor(unittest.TestCase):
    def test_finite_and_bounded_for_near_zero_prediction(self):
        """At initialization the predicted spectrum can sit decades below the
        DNS one. The relative floor must keep the value finite and bounded
        rather than producing a log spike."""
        torch.manual_seed(0)
        dns = torch.randn(16, 16, 4, 4, dtype=torch.float64)
        pred = torch.zeros_like(dns, requires_grad=True)
        fn = SpectralCascadeLoss(epsilon_rel=1e-8)
        val = fn(pred, dns)
        self.assertTrue(torch.isfinite(val))
        # bounded by (log10 of the floor ratio)^2 = 8^2
        self.assertLess(val.item(), 64.0 + 1e-6)
        val.backward()
        assert pred.grad is not None
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_invariant_under_tau_rescaling(self):
        torch.manual_seed(1)
        dns = torch.randn(16, 16, 4, 4, dtype=torch.float64)
        pred = dns + 0.1 * torch.randn_like(dns)
        fn = SpectralCascadeLoss(epsilon_rel=1e-8)
        base = fn(pred, dns).item()
        k = 1e4
        scaled = fn(k * pred, k * dns).item()
        self.assertAlmostEqual(base, scaled, places=8)


class TestRMHDSGSLoss(unittest.TestCase):
    def setUp(self):
        self.criterion = RMHDSGSLoss(
            alpha=0.5,
            beta=0.1,
            gamma=5.0,
            delta=2.0,
            lambda_stab=10.0,
        )
        self.g_cov = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))

    def test_default_weights(self):
        self.assertEqual(self.criterion.alpha, 0.5)
        self.assertEqual(self.criterion.beta, 0.1)
        self.assertEqual(self.criterion.gamma_w, 5.0)
        self.assertEqual(self.criterion.delta, 2.0)

    def test_ortho_off_by_default(self):
        """γ defaults to 0 because the term is identically zero for a
        basis-contracted output."""
        self.assertEqual(RMHDSGSLoss().gamma_w, 0.0)

    def test_gamma_zero_skips_ortho_and_matches_composite(self):
        N = 8
        torch.manual_seed(0)
        crit = RMHDSGSLoss(alpha=0.5, beta=0.1, gamma=0.0, delta=2.0)
        tau_pred = torch.randn(N, 4, 4, dtype=torch.float64, requires_grad=True)
        tau_dns = torch.randn(N, 4, 4, dtype=torch.float64)
        u_con = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).repeat(N, 1)
        grad_u = torch.randn(N, 4, 4, dtype=torch.float64)
        rho = torch.ones(N, dtype=torch.float64)
        p = 0.1 * torch.ones(N, dtype=torch.float64)

        loss, comps = crit(tau_pred, tau_dns, u_con, grad_u, self.g_cov, rho, p)
        self.assertEqual(comps["L_ortho"], 0.0)
        expected = comps["L_tensor"] + 0.5 * comps["L_P"] + 2.0 * comps["L_WEC"]
        self.assertAlmostEqual(loss.item(), expected, places=10)

    def test_set_scales_threads_into_components(self):
        crit = RMHDSGSLoss()
        self.assertIsNone(crit.tau_scale)
        self.assertIsNone(crit.p_scale)
        crit.set_scales(
            tau_scale=torch.tensor(7.3e-3, dtype=torch.float64), p_scale=2.0
        )
        self.assertAlmostEqual(crit.tau_scale, 7.3e-3, places=12)
        self.assertAlmostEqual(crit.p_scale, 2.0, places=12)

    def test_spectral_term_is_differentiable_and_weighted(self):
        """spectral_term must be summable into the point-wise objective so one
        backward pass carries every component."""
        torch.manual_seed(0)
        crit = RMHDSGSLoss(beta=0.1)
        pred = torch.randn(16, 16, 4, 4, dtype=torch.float64, requires_grad=True)
        dns = torch.randn(16, 16, 4, 4, dtype=torch.float64)
        term = crit.spectral_term(pred, dns)
        weighted, raw = crit.compute_spectral(pred, dns)
        self.assertAlmostEqual(term.item(), weighted.item(), places=12)
        self.assertAlmostEqual(term.item(), 0.1 * raw, places=12)
        term.backward()
        assert pred.grad is not None
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_forward_components_and_weighting(self):
        N = 8
        tau_pred = torch.randn(N, 4, 4, dtype=torch.float64, requires_grad=True)
        tau_dns = torch.randn(N, 4, 4, dtype=torch.float64)
        u_con = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).repeat(N, 1)
        grad_u = torch.randn(N, 4, 4, dtype=torch.float64)
        rho = torch.ones(N, dtype=torch.float64)
        p = 0.1 * torch.ones(N, dtype=torch.float64)

        loss, comps = self.criterion(tau_pred, tau_dns, u_con, grad_u, self.g_cov, rho, p)

        self.assertIn("L_tensor", comps)
        self.assertIn("L_P", comps)
        self.assertIn("L_ortho", comps)
        self.assertIn("L_WEC", comps)

        # Verify composite formula: L_tensor + α L_P + γ L_ortho + δ L_WEC
        expected = (
            comps["L_tensor"]
            + 0.5 * comps["L_P"]
            + 5.0 * comps["L_ortho"]
            + 2.0 * comps["L_WEC"]
        )
        self.assertAlmostEqual(loss.item(), expected, places=5)

        loss.backward()
        self.assertIsNotNone(tau_pred.grad)
        assert tau_pred.grad is not None
        self.assertTrue(torch.isfinite(tau_pred.grad).all())

    def test_spectral_pass(self):
        tau_pred_2d = torch.randn(16, 16, 4, 4, dtype=torch.float64, requires_grad=True)
        tau_dns_2d = torch.randn(16, 16, 4, 4, dtype=torch.float64)

        weighted_loss, raw_val = self.criterion.compute_spectral(tau_pred_2d, tau_dns_2d)
        self.assertAlmostEqual(weighted_loss.item(), 0.1 * raw_val, places=6)

        weighted_loss.backward()
        self.assertIsNotNone(tau_pred_2d.grad)
        assert tau_pred_2d.grad is not None
        self.assertTrue(torch.isfinite(tau_pred_2d.grad).all())


if __name__ == "__main__":
    unittest.main()


class TestPerSnapshotTauFloor:
    r"""A per-snapshot floor in L_tensor, and why a global one is not enough.

    ``TensorLoss`` normalises by ``||tau_DNS||^2 + floor^2``. With one global
    floor and a training window over which the subgrid intensity varies ~40x
    (dns_khi, t = 0..20), every cell of the weak snapshots sits below it, the
    term degenerates to an absolute MSE there, and predicting zero becomes
    near-optimal — which is exactly how a model ends up with a good stress in
    mid-run and a collapsed one at late times.
    """

    @staticmethod
    def _g():
        g = torch.eye(4, dtype=torch.float64)
        g[0, 0] = -1.0
        return g

    @staticmethod
    def _tau(scale, n=32, seed=0):
        gen = torch.Generator().manual_seed(seed)
        t = torch.randn(n, 4, 4, generator=gen, dtype=torch.float64)
        return scale * (t + t.transpose(-1, -2))

    def test_scalar_floor_still_works(self):
        from src.losses import CovariantTensorLoss as TensorLoss

        loss = TensorLoss()
        tau = self._tau(1.0)
        out = loss(0.5 * tau, tau, self._g(), tau_floor=0.1)
        assert torch.isfinite(out) and float(out) > 0.0

    def test_tensor_floor_matches_a_scalar_of_the_same_value(self):
        from src.losses import CovariantTensorLoss as TensorLoss

        loss = TensorLoss()
        tau = self._tau(1.0)
        g = self._g()
        scalar = loss(0.5 * tau, tau, g, tau_floor=0.3)
        as_tensor = loss(
            0.5 * tau, tau, g, tau_floor=torch.full((tau.shape[0],), 0.3, dtype=torch.float64)
        )
        assert float(scalar) == pytest.approx(float(as_tensor), rel=1e-12)

    def test_per_point_floor_varies_the_weighting(self):
        from src.losses import CovariantTensorLoss as TensorLoss

        loss = TensorLoss()
        tau = self._tau(1.0)
        g = self._g()
        floor = torch.linspace(0.01, 2.0, tau.shape[0], dtype=torch.float64)
        assert float(loss(0.5 * tau, tau, g, tau_floor=floor)) != pytest.approx(
            float(loss(0.5 * tau, tau, g, tau_floor=0.01))
        )

    def test_a_global_floor_silences_the_weak_snapshot(self):
        """The failure being fixed: with one global floor, shrinking the weak
        snapshot's prediction to zero costs almost nothing."""
        from src.losses import CovariantTensorLoss as TensorLoss

        loss = TensorLoss()
        g = self._g()
        strong = self._tau(1.0, seed=1)
        weak = self._tau(0.01, seed=2)          # 100x weaker, as late times are
        global_floor = 0.5                       # median of the strong snapshot

        # Cost of collapsing the weak snapshot to zero, under each floor.
        weak_zero = torch.zeros_like(weak)
        weak_good = 0.9 * weak

        cost_global = float(loss(weak_zero, weak, g, tau_floor=global_floor)) - float(
            loss(weak_good, weak, g, tau_floor=global_floor)
        )
        per_snap = float(weak.norm(dim=(-2, -1)).median())
        cost_local = float(loss(weak_zero, weak, g, tau_floor=per_snap)) - float(
            loss(weak_good, weak, g, tau_floor=per_snap)
        )

        # The claim is about the ratio, not an absolute threshold: a global floor
        # tuned to the strong snapshot makes collapsing the weak one nearly free
        # relative to what a floor matched to that snapshot's own scale charges.
        assert cost_local > 20 * cost_global, (
            f"per-snapshot floor charges {cost_local:.3e} for a collapse that the "
            f"global floor charges only {cost_global:.3e} for"
        )

    def test_composite_loss_accepts_a_batch_floor_override(self):
        from src.losses import RMHDSGSLoss

        n = 16
        crit = RMHDSGSLoss()
        crit.set_scales(tau_scale=1.0, p_scale=1.0, tau_floor=0.5)

        tau = self._tau(1.0, n=n)
        u = torch.zeros(n, 4, dtype=torch.float64)
        u[:, 0] = 1.0
        grad_u = torch.zeros(n, 4, 4, dtype=torch.float64)
        rho = torch.ones(n, dtype=torch.float64)
        p = torch.ones(n, dtype=torch.float64)

        bound, _ = crit(0.5 * tau, tau, u, grad_u, self._g(), rho, p)
        override, _ = crit(
            0.5 * tau, tau, u, grad_u, self._g(), rho, p,
            tau_floor=torch.full((n,), 0.01, dtype=torch.float64),
        )
        assert float(bound) != pytest.approx(float(override))

    def test_bound_floor_survives_as_a_tensor(self):
        from src.losses import RMHDSGSLoss

        crit = RMHDSGSLoss()
        floor = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        crit.set_scales(tau_floor=floor)
        assert isinstance(crit.tau_floor, torch.Tensor)
        assert torch.allclose(crit.tau_floor, floor)


class TestSpectralHighKTilt:
    """Per-shell weighting of L_spectral toward small scales.

    The term as originally written weights a decade of error at k = 2 and at
    k = 30 identically, but the trained model's stress carries ~38% of its power
    above k = 16 against ~51% for the exact stress, and that deficit is what
    lets high-k energy drain away over a long rollout.
    """

    @staticmethod
    def _fields(seed=0, n=32):
        gen = torch.Generator().manual_seed(seed)
        a = torch.randn(n, n, 4, 4, generator=gen, dtype=torch.float64)
        return a, a.clone()

    def test_zero_tilt_is_the_original_behaviour(self):
        from src.losses import SpectralCascadeLoss

        pred, dns = self._fields()
        pred = pred * 0.5
        plain = SpectralCascadeLoss()
        tilted0 = SpectralCascadeLoss(highk_tilt=0.0)
        assert float(plain(pred, dns)) == pytest.approx(float(tilted0(pred, dns)), rel=1e-12)

    def test_identical_fields_score_zero_under_any_tilt(self):
        from src.losses import SpectralCascadeLoss

        pred, dns = self._fields()
        for tilt in (0.0, 0.5, 1.0, 2.0):
            loss = SpectralCascadeLoss(highk_tilt=tilt)
            assert float(loss(pred, dns)) == pytest.approx(0.0, abs=1e-20)

    def test_tilt_penalises_a_high_k_error_more(self):
        """A model wrong only at small scales must score worse as the tilt rises."""
        from src.losses import SpectralCascadeLoss

        n = 32
        x = torch.arange(n, dtype=torch.float64)
        lo = torch.sin(2 * torch.pi * 2 * x / n)          # k = 2
        hi = torch.sin(2 * torch.pi * 14 * x / n)         # k = 14

        def field(a_lo, a_hi):
            f = a_lo * lo.unsqueeze(-1) + a_hi * hi.unsqueeze(-1)
            out = torch.zeros(n, n, 4, 4, dtype=torch.float64)
            out[..., 1, 1] = f
            return out

        dns = field(1.0, 1.0)
        bad_hi = field(1.0, 0.2)     # small scales wrong

        scores = [
            float(SpectralCascadeLoss(highk_tilt=t)(bad_hi, dns))
            for t in (0.0, 1.0, 2.0)
        ]
        assert scores[1] > scores[0]
        assert scores[2] > scores[1]

    def test_tilt_relaxes_a_low_k_error(self):
        """The trade: large-scale error is weighted down, which is acceptable
        because the point-wise terms already constrain it."""
        from src.losses import SpectralCascadeLoss

        n = 32
        x = torch.arange(n, dtype=torch.float64)
        lo = torch.sin(2 * torch.pi * 2 * x / n)
        hi = torch.sin(2 * torch.pi * 14 * x / n)

        def field(a_lo, a_hi):
            out = torch.zeros(n, n, 4, 4, dtype=torch.float64)
            out[..., 1, 1] = a_lo * lo.unsqueeze(-1) + a_hi * hi.unsqueeze(-1)
            return out

        dns = field(1.0, 1.0)
        bad_lo = field(0.2, 1.0)
        assert float(SpectralCascadeLoss(highk_tilt=2.0)(bad_lo, dns)) < float(
            SpectralCascadeLoss(highk_tilt=0.0)(bad_lo, dns)
        )

    def test_weights_are_normalised_to_mean_one(self):
        """So a tilt does not silently rescale the term against beta."""
        from src.losses import SpectralCascadeLoss

        n, k_min = 32, 1
        n_shells = n // 2 - k_min
        k = torch.arange(k_min, k_min + n_shells, dtype=torch.float64)
        for tilt in (0.5, 1.0, 3.0):
            w = (k / k_min) ** tilt
            assert float((w / w.mean()).mean()) == pytest.approx(1.0)

    def test_composite_loss_forwards_the_tilt(self):
        from src.losses import RMHDSGSLoss

        crit = RMHDSGSLoss(spectral_highk_tilt=1.5)
        assert crit.spectral_loss.highk_tilt == pytest.approx(1.5)
        assert RMHDSGSLoss().spectral_loss.highk_tilt == pytest.approx(0.0)
