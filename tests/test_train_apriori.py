"""Tests for the a priori training machinery: splitting, snapshot batching,
monitor smoothing and the LR schedule.

These cover the parts that decide *what the optimizer sees* and *which
checkpoint is kept* -- the two places where the previous run lost
effectiveness without the loss values showing it.
"""

import unittest

import pytest
import torch
import torch.nn as nn

from src.train_apriori import (
    EMAMonitor,
    SnapshotView,
    build_lr_schedule,
    split_snapshots,
    train_epoch,
    validate,
)
from src.losses import RMHDSGSLoss


class _FakeDataset:
    """Minimal stand-in for TBNNDataset: N snapshots on a Ny x Nx grid."""

    def __init__(self, n_snaps=6, Ny=8, Nx=8, dtype=torch.float64, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.n_snaps = n_snaps
        total = n_snaps * Ny * Nx
        self.invariants = torch.randn(total, 15, generator=g, dtype=dtype)
        self.thermo = torch.randn(total, 2, generator=g, dtype=dtype)
        # Rest-frame spatial basis: only the 3x3 spatial block is populated,
        # so u_mu T^{mu nu} = 0 for u = (1,0,0,0), as the real basis does.
        self.basis = torch.zeros(total, 12, 4, 4, dtype=dtype)
        self.basis[:, :, 1:, 1:] = torch.randn(total, 12, 3, 3, generator=g, dtype=dtype)
        self.tau_sgs = torch.zeros(total, 4, 4, dtype=dtype)
        self.tau_sgs[:, 1:, 1:] = 1e-2 * torch.randn(total, 3, 3, generator=g, dtype=dtype)
        self.u_con = torch.zeros(total, 4, dtype=dtype)
        self.u_con[:, 0] = 1.0
        self.grad_u_cov = torch.randn(total, 4, 4, generator=g, dtype=dtype)
        self.rho = torch.ones(total, dtype=dtype)
        self.p = 0.1 * torch.ones(total, dtype=dtype)
        self.inv_mean = self.invariants.mean(0)
        self.inv_std = self.invariants.std(0).clamp(min=1e-12)
        self.thermo_mean = self.thermo.mean(0)
        self.thermo_std = self.thermo.std(0).clamp(min=1e-12)
        self.snapshot_boundaries = [
            (i * Ny * Nx, Ny, Nx) for i in range(n_snaps)
        ]
        self.times = [float(i) for i in range(n_snaps)]

    @property
    def num_snapshots(self):
        return self.n_snaps

    def snapshot_view(self, indices):
        return SnapshotView(self, indices)


class _CountingModel(nn.Module):
    """TBNN-shaped model that records how many times its params were updated."""

    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.hidden = nn.Linear(17, 8, dtype=dtype)
        self.coeff_head = nn.Linear(8, 12, bias=False, dtype=dtype)
        self.p_sgs_head = nn.Linear(8, 1, dtype=dtype)

    def forward(self, invariants, thermo_inputs, basis_tensors):
        latent = torch.tanh(self.hidden(torch.cat([invariants, thermo_inputs], dim=-1)))
        coeffs = self.coeff_head(latent)
        tau = torch.einsum("...n,...nij->...ij", coeffs, basis_tensors)
        return tau, coeffs, self.p_sgs_head(latent)


class TestSplitSnapshots(unittest.TestCase):
    def test_interleaved_not_contiguous(self):
        times = [i * 0.1 for i in range(81)]          # dns_khi layout
        s = split_snapshots(times, time_range=(1.0, 6.5), val_every=5, test_time=(6.5, 1e30))
        train_t = [times[i] for i in s["train"]]
        val_t = [times[i] for i in s["val"]]
        test_t = [times[i] for i in s["test"]]

        # val is spread across the window, not parked at one end
        self.assertGreater(len(val_t), 5)
        self.assertLess(min(val_t), 2.0)
        self.assertGreater(max(val_t), 5.5)
        # and its mean time is close to the training mean -- same regime
        self.assertAlmostEqual(
            sum(val_t) / len(val_t), sum(train_t) / len(train_t), delta=0.5
        )
        # test is the contiguous tail, disjoint from both
        self.assertEqual(min(test_t), 6.5)
        self.assertTrue(set(s["test"]).isdisjoint(s["train"]))
        self.assertTrue(set(s["test"]).isdisjoint(s["val"]))

    def test_splits_are_disjoint_and_cover_the_window(self):
        times = [i * 0.1 for i in range(81)]
        s = split_snapshots(times, time_range=(1.0, 6.5), val_every=5, test_time=(6.5, 1e30))
        self.assertTrue(set(s["train"]).isdisjoint(s["val"]))
        in_window = {i for i, t in enumerate(times) if 1.0 <= t < 6.5}
        self.assertEqual(set(s["train"]) | set(s["val"]), in_window)

    def test_no_test_window(self):
        times = [i * 0.1 for i in range(81)]
        s = split_snapshots(times, time_range=(1.0, 8.0), val_every=4, test_time=None)
        self.assertEqual(s["test"], [])

    def test_rejects_degenerate_val_every(self):
        with self.assertRaises(ValueError):
            split_snapshots([0.0, 1.0, 2.0], val_every=1)

    def test_empty_window_is_an_error_not_an_empty_split(self):
        """The old defaults (val t in [70, 85]) select nothing on dns_khi
        (t = 0..8); that has to fail loudly."""
        times = [i * 0.1 for i in range(81)]
        with self.assertRaises(ValueError):
            split_snapshots(times, time_range=(70.0, 85.0), val_every=5, test_time=None)


class TestSnapshotView(unittest.TestCase):
    def setUp(self):
        self.ds = _FakeDataset(n_snaps=6, Ny=8, Nx=8)

    def test_groups_cover_every_snapshot_once(self):
        view = self.ds.snapshot_view(range(6))
        for k in (1, 2, 4, 7):
            groups = view.groups(k)
            flat = [i for g in groups for i in g]
            self.assertEqual(sorted(flat), list(range(6)),
                             f"group size {k} dropped or repeated snapshots")

    def test_gather_preserves_grid_shape_and_contiguity(self):
        view = self.ds.snapshot_view([1, 3])
        group = view.gather([0, 1])
        self.assertEqual(group["invariants"].shape, (2 * 64, 15))
        self.assertEqual(group["shapes"], [(0, 8, 8), (64, 8, 8)])
        # the gathered rows are exactly the parent's contiguous slices
        torch.testing.assert_close(
            group["tau_sgs"][:64], self.ds.tau_sgs[64:128]
        )
        torch.testing.assert_close(
            group["tau_sgs"][64:], self.ds.tau_sgs[192:256]
        )

    def test_point_index_matches_gather(self):
        view = self.ds.snapshot_view([0, 2, 4])
        idx = view.point_index()
        self.assertEqual(idx.numel(), view.num_points)
        torch.testing.assert_close(self.ds.tau_sgs[idx], view.gather([0, 1, 2])["tau_sgs"])

    def test_rejects_out_of_range(self):
        with self.assertRaises(IndexError):
            self.ds.snapshot_view([0, 99])
        with self.assertRaises(ValueError):
            self.ds.snapshot_view([])


class TestTrainEpochSingleStep(unittest.TestCase):
    """The core fix: one optimizer step per group, spectral term included, and
    nothing applied after the last group."""

    def setUp(self):
        self.ds = _FakeDataset(n_snaps=6, Ny=8, Nx=8)
        self.view = self.ds.snapshot_view(range(6))
        self.g_cov = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
        self.criterion = RMHDSGSLoss(alpha=0.5, beta=0.1, gamma=0.0, delta=2.0)
        self.criterion.set_scales(tau_scale=1e-2, p_scale=1e-2)

    def _run(self, spb=2, beta=0.1):
        torch.manual_seed(0)
        model = _CountingModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.criterion.beta = beta
        steps = []
        real_step = opt.step

        def counting_step(*a, **k):
            steps.append(1)
            return real_step(*a, **k)

        opt.step = counting_step
        res = train_epoch(
            model, self.view, opt, self.criterion,
            self.ds.inv_mean, self.ds.inv_std, self.ds.thermo_mean, self.ds.thermo_std,
            torch.device("cpu"), g_cov=self.g_cov, snapshots_per_batch=spb,
            grad_clip=1.0, generator=torch.Generator().manual_seed(0),
        )
        return res, len(steps)

    def test_one_step_per_group_with_spectral_enabled(self):
        for spb in (1, 2, 3):
            _res, n_steps = self._run(spb=spb, beta=0.1)
            expected = len(self.view.groups(spb))
            self.assertEqual(
                n_steps, expected,
                f"snapshots_per_batch={spb}: expected {expected} steps, got {n_steps}. "
                "An extra step means the spectral term is being applied separately.",
            )

    def test_step_count_independent_of_beta(self):
        """Turning the spectral term on must not add optimizer steps."""
        _r0, n_off = self._run(spb=2, beta=0.0)
        _r1, n_on = self._run(spb=2, beta=0.1)
        self.assertEqual(n_off, n_on)

    def test_spectral_component_is_reported_and_nonzero(self):
        res, _ = self._run(spb=2, beta=0.1)
        self.assertIn("L_spectral", res)
        self.assertGreater(res["L_spectral"], 0.0)

    def test_grad_clip_is_applied(self):
        res, _ = self._run(spb=2, beta=0.1)
        self.assertIn("grad_norm", res)
        self.assertGreater(res["grad_norm"], 0.0)

    def test_validate_is_deterministic(self):
        torch.manual_seed(0)
        model = _CountingModel()
        kw = dict(
            inv_mean=self.ds.inv_mean, inv_std=self.ds.inv_std,
            thermo_mean=self.ds.thermo_mean, thermo_std=self.ds.thermo_std,
            device=torch.device("cpu"), g_cov=self.g_cov,
        )
        a = validate(model, self.view, self.criterion, snapshots_per_batch=2, **kw)
        b = validate(model, self.view, self.criterion, snapshots_per_batch=2, **kw)
        for k in a:
            self.assertAlmostEqual(a[k], b[k], places=12, msg=f"{k} not reproducible")

    def test_validate_invariant_to_grouping(self):
        """Point-wise metrics must not depend on how snapshots are grouped."""
        torch.manual_seed(0)
        model = _CountingModel()
        kw = dict(
            inv_mean=self.ds.inv_mean, inv_std=self.ds.inv_std,
            thermo_mean=self.ds.thermo_mean, thermo_std=self.ds.thermo_std,
            device=torch.device("cpu"), g_cov=self.g_cov,
        )
        a = validate(model, self.view, self.criterion, snapshots_per_batch=1, **kw)
        b = validate(model, self.view, self.criterion, snapshots_per_batch=3, **kw)
        self.assertAlmostEqual(a["rel_l2"], b["rel_l2"], places=10)
        self.assertAlmostEqual(a["val_L_tensor"], b["val_L_tensor"], places=10)


class TestEMAMonitor(unittest.TestCase):
    def test_constant_input_is_unbiased_from_the_first_epoch(self):
        m = EMAMonitor(beta=0.8)
        for _ in range(5):
            self.assertAlmostEqual(m.update(0.25), 0.25, places=12)

    def test_smoothing_reduces_variance(self):
        raw = [0.19, 0.24, 0.16, 0.21, 0.18, 0.20, 0.23, 0.17, 0.195, 0.205]
        m = EMAMonitor(beta=0.8)
        smoothed = [m.update(v) for v in raw]
        def var(xs):
            mu = sum(xs) / len(xs)
            return sum((x - mu) ** 2 for x in xs) / len(xs)
        self.assertLess(var(smoothed[2:]), var(raw[2:]))

    def test_beta_zero_is_passthrough(self):
        m = EMAMonitor(beta=0.0)
        for v in (1.0, 5.0, 2.0):
            self.assertEqual(m.update(v), v)

    def test_rejects_invalid_beta(self):
        for bad in (-0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                EMAMonitor(beta=bad)

    def test_round_trips_through_state_dict(self):
        m = EMAMonitor(beta=0.8)
        for v in (0.2, 0.3, 0.25):
            m.update(v)
        m2 = EMAMonitor(beta=0.8)
        m2.load_state_dict(m.state_dict())
        self.assertEqual(m2.n, m.n)
        self.assertAlmostEqual(m2.update(0.22), m.update(0.22), places=12)

    def test_smoothed_selection_rejects_a_lucky_draw(self):
        """The concrete failure being fixed: a -1.8σ outlier must not be
        selected as 'best' when the surrounding epochs are worse."""
        # mean 0.20, then one lucky 0.16, then back to 0.20
        series = [0.20] * 8 + [0.16] + [0.20] * 8
        raw_best = min(series)
        m = EMAMonitor(beta=0.8)
        smoothed = [m.update(v) for v in series]
        self.assertAlmostEqual(raw_best, 0.16, places=12)
        # the smoothed minimum barely moves off the true level
        self.assertGreater(min(smoothed), 0.19)


class TestLRSchedule(unittest.TestCase):
    def _lrs(self, sched, opt, is_plateau, epochs, monitor=None):
        out = []
        for e in range(epochs):
            out.append(opt.param_groups[0]["lr"])
            if is_plateau:
                sched.step(monitor[e])
            else:
                sched.step()
        return out

    def test_cosine_warms_up_then_decays_to_the_floor(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=1e-3)
        sched, is_plateau = build_lr_schedule(
            opt, "cosine", epochs=50, warmup_epochs=5, min_lr_frac=0.01
        )
        self.assertFalse(is_plateau)
        lrs = self._lrs(sched, opt, is_plateau, 50)
        self.assertAlmostEqual(lrs[0], 1e-3 / 5, places=12)     # first warmup step
        self.assertAlmostEqual(max(lrs), 1e-3, places=12)       # peaks at --lr
        self.assertEqual(lrs.index(max(lrs)), 4)                # at the end of warmup
        self.assertLess(lrs[-1], 2e-5)                          # decayed to ~floor
        # monotone after warmup
        tail = lrs[5:]
        self.assertEqual(tail, sorted(tail, reverse=True))

    def test_cosine_needs_no_monitor(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=1e-3)
        sched, _ = build_lr_schedule(opt, "cosine", epochs=10, warmup_epochs=0)
        sched.step()          # must not require a metric

    def test_plateau_is_still_available(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=1e-3)
        sched, is_plateau = build_lr_schedule(opt, "plateau", epochs=50, lr_patience=2)
        self.assertTrue(is_plateau)
        lrs = self._lrs(sched, opt, is_plateau, 10, monitor=[1.0] * 10)
        self.assertLess(lrs[-1], 1e-3)

    def test_zero_warmup_starts_at_full_lr(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=1e-3)
        _sched, _ = build_lr_schedule(opt, "cosine", epochs=10, warmup_epochs=0)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1e-3, places=12)

    def test_rejects_unknown_schedule(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=1e-3)
        with self.assertRaises(ValueError):
            build_lr_schedule(opt, "onecycle", epochs=10)


if __name__ == "__main__":
    unittest.main()


class TestAdaptiveGradClip(unittest.TestCase):
    """The clip must catch outliers, not every step. A fixed 1.0 threshold
    against this objective's O(1e1-1e3) gradient norms fired on 100% of steps,
    which replaces the gradient magnitude with a constant."""

    def _model_with_grad(self, scale):
        m = nn.Linear(4, 4, dtype=torch.float64)
        for p in m.parameters():
            p.grad = torch.full_like(p, scale)
        return m

    def test_warmup_measures_without_clipping_away_everything(self):
        from src.train_apriori import AdaptiveGradClip
        clip = AdaptiveGradClip(mult=5.0, window=100, warmup_steps=5)
        norms = []
        for _ in range(4):
            n, clipped = clip(self._model_with_grad(1.0))
            norms.append(n)
            self.assertFalse(clipped)
        self.assertTrue(all(abs(n - norms[0]) < 1e-12 for n in norms))

    def test_typical_steps_are_not_clipped(self):
        from src.train_apriori import AdaptiveGradClip
        clip = AdaptiveGradClip(mult=5.0, window=100, warmup_steps=5)
        n_clipped = 0
        for i in range(60):
            # norms fluctuating within a factor of ~3, as a healthy run does
            scale = 1.0 + 2.0 * ((i * 7) % 10) / 10.0
            _n, clipped = clip(self._model_with_grad(scale))
            n_clipped += int(clipped)
        self.assertLess(n_clipped / 60, 0.1,
                        "adaptive clip fired on ordinary steps")

    def test_genuine_spike_is_clipped(self):
        from src.train_apriori import AdaptiveGradClip
        clip = AdaptiveGradClip(mult=5.0, window=100, warmup_steps=5)
        for _ in range(20):
            clip(self._model_with_grad(1.0))
        thr = clip.threshold
        self.assertIsNotNone(thr)
        spike = self._model_with_grad(1000.0)
        norm, clipped = clip(spike)
        self.assertTrue(clipped)
        self.assertGreater(norm, thr)
        # and the gradient really was scaled down to the threshold
        after = torch.cat([p.grad.reshape(-1) for p in spike.parameters()]).norm()
        self.assertLessEqual(float(after), thr * (1 + 1e-9))

    def test_threshold_tracks_the_median_not_the_max(self):
        from src.train_apriori import AdaptiveGradClip
        clip = AdaptiveGradClip(mult=5.0, window=100, warmup_steps=5)
        for _ in range(20):
            clip(self._model_with_grad(1.0))
        clip(self._model_with_grad(1e6))          # one outlier
        for _ in range(5):
            clip(self._model_with_grad(1.0))
        # the outlier must not have dragged the threshold up with it
        base = clip.threshold
        self.assertLess(base, 1e3)

    def test_round_trips_through_state_dict(self):
        from src.train_apriori import AdaptiveGradClip
        a = AdaptiveGradClip(mult=5.0, window=10, warmup_steps=3)
        for _ in range(8):
            a(self._model_with_grad(2.0))
        b = AdaptiveGradClip(mult=5.0, window=10, warmup_steps=3)
        b.load_state_dict(a.state_dict())
        self.assertAlmostEqual(b.threshold, a.threshold, places=12)

    def test_train_epoch_reports_a_low_clip_fraction(self):
        from src.train_apriori import AdaptiveGradClip
        ds = _FakeDataset(n_snaps=8, Ny=8, Nx=8)
        view = ds.snapshot_view(range(8))
        g_cov = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
        crit = RMHDSGSLoss(alpha=0.5, beta=0.1, gamma=0.0, delta=2.0)
        crit.set_scales(tau_scale=1e-2, p_scale=1e-2, tau_floor=5e-3)
        torch.manual_seed(0)
        model = _CountingModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        clip = AdaptiveGradClip(mult=5.0, window=100, warmup_steps=4)
        fracs = []
        for _ in range(12):
            res = train_epoch(
                model, view, opt, crit,
                ds.inv_mean, ds.inv_std, ds.thermo_mean, ds.thermo_std,
                torch.device("cpu"), g_cov=g_cov, snapshots_per_batch=2,
                grad_clip=clip, generator=torch.Generator().manual_seed(0),
            )
            fracs.append(res["clip_frac"])
        # after warmup, most epochs should clip few or no steps
        self.assertLess(sum(fracs[4:]) / len(fracs[4:]), 0.35)

    def test_absolute_mode_still_works(self):
        ds = _FakeDataset(n_snaps=4, Ny=8, Nx=8)
        view = ds.snapshot_view(range(4))
        g_cov = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
        crit = RMHDSGSLoss(alpha=0.5, beta=0.0, gamma=0.0, delta=2.0)
        crit.set_scales(tau_scale=1e-2, p_scale=1e-2, tau_floor=5e-3)
        torch.manual_seed(0)
        model = _CountingModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        res = train_epoch(
            model, view, opt, crit,
            ds.inv_mean, ds.inv_std, ds.thermo_mean, ds.thermo_std,
            torch.device("cpu"), g_cov=g_cov, snapshots_per_batch=2,
            grad_clip=1e-12, generator=torch.Generator().manual_seed(0),
        )
        # a threshold of 1e-12 clips everything, which is exactly the failure
        # mode the adaptive default avoids -- assert it is visible in clip_frac
        self.assertAlmostEqual(res["clip_frac"], 1.0, places=6)

    def test_off_mode_does_not_crash(self):
        ds = _FakeDataset(n_snaps=4, Ny=8, Nx=8)
        view = ds.snapshot_view(range(4))
        g_cov = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
        crit = RMHDSGSLoss(alpha=0.5, beta=0.0, gamma=0.0, delta=2.0)
        crit.set_scales(tau_scale=1e-2, p_scale=1e-2, tau_floor=5e-3)
        torch.manual_seed(0)
        model = _CountingModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        res = train_epoch(
            model, view, opt, crit,
            ds.inv_mean, ds.inv_std, ds.thermo_mean, ds.thermo_std,
            torch.device("cpu"), g_cov=g_cov, snapshots_per_batch=2,
            grad_clip=None, generator=torch.Generator().manual_seed(0),
        )
        self.assertEqual(res["clip_frac"], 0.0)


class TestPerSnapshotFloorGuard:
    """The lower guard on per-snapshot L_tensor floors.

    A snapshot with no subgrid signal (t = 0 on dns_khi has a median ||tau||_F
    of ~3e-15) would otherwise get a vanishing denominator and dominate the
    gradient with noise -- the unguarded spread across t = 0..20 is ~6e11.
    """

    @staticmethod
    def _floors(norms_per_snapshot, quantile=0.5):
        import torch

        return torch.stack([
            torch.quantile(torch.tensor(n, dtype=torch.float64), quantile)
            for n in norms_per_snapshot
        ])

    def test_unguarded_spread_is_pathological(self):
        raw = self._floors([[1e-15] * 8, [1e-3] * 8, [2e-3] * 8])
        assert float(raw.max() / raw.min()) > 1e11

    def test_guard_collapses_the_spread(self):
        import torch

        raw = self._floors([[1e-15] * 8, [1e-3] * 8, [2e-3] * 8])
        global_floor = 1.2e-3
        guarded = raw.clamp(min=0.05 * global_floor)
        assert float(guarded.max() / guarded.min()) < 100
        assert float(guarded.min()) == pytest.approx(0.05 * global_floor)

    def test_guard_leaves_healthy_snapshots_alone(self):
        raw = self._floors([[8e-4] * 8, [1e-3] * 8, [2e-3] * 8])
        guarded = raw.clamp(min=0.05 * 1.2e-3)
        assert torch.allclose(guarded, raw)

    def test_weak_but_real_snapshots_keep_their_own_scale(self):
        """The whole point: a genuinely weak late-time snapshot must NOT be
        clamped up to the global floor, or the fix does nothing."""
        raw = self._floors([[3e-4] * 8, [1.5e-3] * 8])
        guarded = raw.clamp(min=0.05 * 1.2e-3)   # guard is 6e-5, well below 3e-4
        assert float(guarded[0]) == pytest.approx(3e-4)
        assert float(guarded[0]) < float(guarded[1]) / 4
