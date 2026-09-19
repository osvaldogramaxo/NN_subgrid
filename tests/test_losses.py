"""Unit tests for RMHD SGS Physics-Based Loss Functions (src/losses.py)."""

import unittest
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

        loss = self.loss_fn(tau_pred, tau_dns, grad_u)
        # l1_match = |-4 - 4| = 8
        # stability = 10 * ReLU(-(-4)) = 10 * 4 = 40
        # total = 48
        self.assertAlmostEqual(loss.item(), 48.0, places=5)

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
        self.loss_fn = WECLoss(gamma_eos=4.0 / 3.0)
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
