"""Tests for the MInIT a-priori calibration (:mod:`src.minit_calibrate`)."""

import math

import pytest
import torch

from src.minit import A_GAMMA_PAPER, MInITCoefficients, advance_e_kh, saturation_energy
from src.minit_calibrate import (
    energy_transfer,
    evolve_e_series,
    fit_beta_block_ratio,
    fit_beta_lsq,
    iter_e_series,
    l2_norm_paper,
    one_step_series,
    score_tau,
)

DTYPE = torch.float64


def _synthetic(T=6, N=8, beta=(1.3, 0.7, -0.05), seed=0):
    """A field that satisfies the MInIT ansatz exactly, for recovery tests."""
    g = torch.Generator().manual_seed(seed)
    e = torch.rand(T, N, N, generator=g, dtype=DTYPE) + 0.5
    tau = torch.stack([beta[0] * e, beta[1] * e, beta[2] * e], dim=-1)
    return e, tau


class TestFitBeta:
    def test_least_squares_recovers_exact_coefficients(self):
        e, tau = _synthetic()
        got = fit_beta_lsq(e, tau)
        assert got["beta_xx"] == pytest.approx(1.3)
        assert got["beta_yy"] == pytest.approx(0.7)
        assert got["beta_xy"] == pytest.approx(-0.05)

    def test_least_squares_satisfies_the_trace_identity_for_free(self):
        r"""e_KH = 1/2(tau^xx + tau^yy) forces beta_xx + beta_yy = 2 for this estimator,
        whatever the data — including data that does not obey the ansatz at all."""
        g = torch.Generator().manual_seed(1)
        tau = torch.randn(5, 8, 8, 3, generator=g, dtype=DTYPE)
        e = 0.5 * (tau[..., 0] + tau[..., 1])

        got = fit_beta_lsq(e, tau)
        assert got["beta_xx"] + got["beta_yy"] == pytest.approx(2.0, abs=1e-12)

    def test_block_ratio_recovers_exact_coefficients(self):
        e, tau = _synthetic(N=16)
        got = fit_beta_block_ratio(e, tau, block=4)
        assert got["beta_xx"] == pytest.approx(1.3)
        assert got["beta_yy"] == pytest.approx(0.7)

    def test_block_ratio_clamps_an_oversized_block_to_the_grid(self):
        e, tau = _synthetic(N=8)
        got = fit_beta_block_ratio(e, tau, block=64)
        assert got["beta_xx"] == pytest.approx(1.3)

    def test_block_ratio_ignores_blocks_with_no_energy(self):
        e, tau = _synthetic(N=8)
        e[:, :4] = 0.0
        tau[:, :4] = 0.0
        got = fit_beta_block_ratio(e, tau, block=4)
        assert got["beta_xx"] == pytest.approx(1.3)

    def test_the_two_estimators_disagree_on_data_that_breaks_the_ansatz(self):
        """Their gap is the signal that proportionality does not hold — as on the real DNS."""
        g = torch.Generator().manual_seed(2)
        e = torch.rand(4, 8, 8, generator=g, dtype=DTYPE) + 0.1
        # Stress correlated with e^2 rather than e: proportionality is violated.
        tau = torch.stack([1.3 * e**2, 0.7 * e**2, torch.zeros_like(e)], dim=-1)

        lsq = fit_beta_lsq(e, tau)
        block = fit_beta_block_ratio(e, tau, block=4)
        assert abs(lsq["beta_xx"] - block["beta_xx"]) > 1e-3


class TestL2Norm:
    def test_zero_for_identical_inputs(self):
        x = torch.rand(4, 20, dtype=DTYPE) + 0.1
        assert torch.allclose(l2_norm_paper(x, x), torch.zeros(4, dtype=DTYPE), atol=1e-12)

    def test_symmetric_in_its_arguments(self):
        a = torch.rand(3, 20, dtype=DTYPE) + 0.1
        b = torch.rand(3, 20, dtype=DTYPE) + 0.1
        assert torch.allclose(l2_norm_paper(a, b), l2_norm_paper(b, a), rtol=1e-12)

    def test_broadcasts_one_reference_against_a_parameter_batch(self):
        sim = torch.rand(1, 20, dtype=DTYPE) + 0.1
        model = torch.rand(7, 20, dtype=DTYPE) + 0.1
        out = l2_norm_paper(sim, model)
        assert out.shape == (7,)
        for i in range(7):
            assert float(out[i]) == pytest.approx(float(l2_norm_paper(sim[0], model[i])))

    def test_collapsed_model_is_penalised_finitely_and_reproducibly(self):
        """Eq. (25) diverges for a vanishing model; the floor caps it at a fixed value.

        Without the floor the score lands wherever the clamp epsilon puts it
        (order 1e150), which is finite but meaningless and scale dependent.
        """
        sim = torch.rand(1, 20, dtype=DTYPE) + 0.1
        out = l2_norm_paper(sim, torch.zeros(1, 20, dtype=DTYPE))
        assert torch.all(torch.isfinite(out))
        assert float(out) == pytest.approx(math.sqrt(0.5 * (1.0 + 1e12)), rel=1e-6)

    def test_collapse_penalty_is_scale_invariant(self):
        """Rescaling the measurement must not change how bad a collapse looks."""
        sim = torch.rand(1, 20, dtype=DTYPE) + 0.1
        zero = torch.zeros(1, 20, dtype=DTYPE)
        assert float(l2_norm_paper(sim, zero)) == pytest.approx(
            float(l2_norm_paper(1e6 * sim, zero)), rel=1e-9
        )

    def test_collapse_scores_far_worse_than_any_plausible_model(self):
        sim = torch.rand(1, 20, dtype=DTYPE) + 0.1
        bad = l2_norm_paper(sim, torch.zeros(1, 20, dtype=DTYPE))
        mediocre = l2_norm_paper(sim, torch.rand(1, 20, dtype=DTYPE) + 0.1)
        assert float(bad) > 1e3 * float(mediocre)


class TestSeries:
    def test_iter_and_evolve_agree(self):
        e0 = torch.rand(6, 6, dtype=DTYPE) + 0.01
        shear = torch.rand(5, 6, 6, dtype=DTYPE)
        rho = torch.ones(5, 6, 6, dtype=DTYPE)
        t = torch.linspace(0.0, 0.4, 5, dtype=DTYPE)
        A = torch.tensor([0.14], dtype=DTYPE)
        C = torch.tensor([10.6], dtype=DTYPE)

        stacked = evolve_e_series(e0, shear, rho, t, A, C, 0.25, substeps=4)
        from_iter = torch.stack(
            [e.clone() for _, e in iter_e_series(e0, shear, rho, t, A, C, 0.25, 4)],
            dim=1,
        )
        assert torch.allclose(stacked, from_iter, rtol=1e-14)
        assert stacked.shape == (1, 5, 6, 6)

    def test_parameter_batch_rows_are_independent(self):
        """The scan relies on one batched pass equalling many separate ones."""
        e0 = torch.rand(6, 6, dtype=DTYPE) + 0.01
        shear = torch.rand(4, 6, 6, dtype=DTYPE)
        rho = torch.ones(4, 6, 6, dtype=DTYPE)
        t = torch.linspace(0.0, 0.3, 4, dtype=DTYPE)
        A = torch.tensor([0.1, 0.14, 0.3], dtype=DTYPE)
        C = torch.tensor([5.0, 10.6, 20.0], dtype=DTYPE)

        batched = evolve_e_series(e0, shear, rho, t, A, C, 0.25, substeps=4)
        for i in range(3):
            single = evolve_e_series(
                e0, shear, rho, t, A[i:i + 1], C[i:i + 1], 0.25, substeps=4,
            )
            assert torch.allclose(batched[i], single[0], rtol=1e-14)

    def test_one_step_matches_a_single_manual_advance(self):
        """With constant fields the batched one-step update must equal advance_e_kh."""
        T, N = 4, 5
        e_dns = torch.rand(T, N, N, dtype=DTYPE) + 0.01
        shear = torch.full((T, N, N), 2.0, dtype=DTYPE)
        rho = torch.full((T, N, N), 1.5, dtype=DTYPE)
        t = torch.linspace(0.0, 0.3, T, dtype=DTYPE)

        got = one_step_series(e_dns, shear, rho, t, 0.14, 10.6, 0.25, substeps=8)

        dt = float(t[1] - t[0])
        expect = e_dns[:-1]
        for _ in range(8):
            expect = advance_e_kh(
                expect, torch.full_like(expect, 0.14 * 2.0),
                rho[:-1], dt / 8, 0.25, 10.6,
            )
        assert torch.allclose(got, expect, rtol=1e-12)
        assert got.shape == (T - 1, N, N)

    def test_one_step_is_reseeded_so_it_cannot_drift(self):
        """Every interval starts from truth, so a wild first slice must not propagate."""
        T, N = 5, 4
        e_dns = torch.rand(T, N, N, dtype=DTYPE) + 0.01
        e_dns[0] *= 1e6
        shear = torch.ones(T, N, N, dtype=DTYPE)
        rho = torch.ones(T, N, N, dtype=DTYPE)
        t = torch.linspace(0.0, 0.4, T, dtype=DTYPE)

        got = one_step_series(e_dns, shear, rho, t, 0.14, 10.6, 0.25)
        assert torch.all(torch.isfinite(got[1:]))
        assert float(got[1].max()) < 1e3


class TestEquilibriumLimit:
    def test_the_evolve_mode_relaxes_to_the_equilibrium_mode(self):
        r"""Large A_gamma at fixed A_gamma/C drives the ODE to its fixed point.

        This is the identity the 'equilibrium' variant is built on, and the
        reason the free two-parameter fit keeps pushing A_gamma upward.
        """
        rho = torch.full((4, 4), 1.5, dtype=DTYPE)
        shear = torch.rand(4, 4, dtype=DTYPE) + 0.5
        ratio = 0.0333

        A_big = 50.0
        C_big = A_big / ratio
        e_eq = saturation_energy(A_big * shear, rho, 0.25, C_big)

        e = torch.full((4, 4), 1e-6, dtype=DTYPE)
        for _ in range(200):
            e = advance_e_kh(e, A_big * shear, rho, 0.01, 0.25, C_big)

        assert torch.allclose(e, e_eq, rtol=1e-6)

    def test_equilibrium_is_smagorinsky_shaped(self):
        r"""e_sat = (2 A_gamma / C)^2 rho lambda^2 |S|^2 — quadratic in the shear."""
        rho = torch.full((3, 3), 2.0, dtype=DTYPE)
        shear = torch.full((3, 3), 1.0, dtype=DTYPE)
        lam, A, C = 0.25, 0.14, 4.2

        e1 = saturation_energy(A * shear, rho, lam, C)
        e2 = saturation_energy(A * (3.0 * shear), rho, lam, C)
        assert torch.allclose(e2, 9.0 * e1, rtol=1e-12)

        closed_form = (2.0 * A / C) ** 2 * rho * lam**2 * shear**2
        assert torch.allclose(e1, closed_form, rtol=1e-12)

    def test_only_the_ratio_is_identifiable(self):
        """Doubling A_gamma and C together leaves the equilibrium stress unchanged."""
        rho = torch.full((3, 3), 1.7, dtype=DTYPE)
        shear = torch.rand(3, 3, dtype=DTYPE) + 0.2
        a = saturation_energy(0.14 * shear, rho, 0.25, 4.2)
        b = saturation_energy(0.28 * shear, rho, 0.25, 8.4)
        assert torch.allclose(a, b, rtol=1e-13)

    def test_closure_equilibrium_mode_ignores_the_seed(self):
        from src.minit import MInITClosure

        N = 16
        y = torch.linspace(-0.5, 0.5, N, dtype=DTYPE).unsqueeze(-1)
        prims = torch.stack([
            torch.full((N, N), 1.5, dtype=DTYPE),
            (0.4 * torch.tanh((0.25 - y.abs()) / 0.05)).expand(N, N).contiguous(),
            torch.zeros(N, N, dtype=DTYPE),
            torch.full((N, N), 2.5, dtype=DTYPE),
        ], dim=-1)

        a = MInITClosure(dx=1 / N, dy=1 / N, mode="equilibrium", e_seed=0.0)
        b = MInITClosure(dx=1 / N, dy=1 / N, mode="equilibrium", e_seed=1e3)
        a.initialize(prims)
        b.initialize(prims)
        assert torch.allclose(a.e_KH, b.e_KH, rtol=1e-14)
        assert float(a.e_KH.max()) > 0.0

    def test_equilibrium_mode_is_memoryless(self):
        """Its state is a function of the current field, so history cannot drift."""
        from src.minit import MInITClosure

        N = 16
        y = torch.linspace(-0.5, 0.5, N, dtype=DTYPE).unsqueeze(-1)
        prims = torch.stack([
            torch.full((N, N), 1.5, dtype=DTYPE),
            (0.4 * torch.tanh((0.25 - y.abs()) / 0.05)).expand(N, N).contiguous(),
            torch.zeros(N, N, dtype=DTYPE),
            torch.full((N, N), 2.5, dtype=DTYPE),
        ], dim=-1)

        closure = MInITClosure(dx=1 / N, dy=1 / N, mode="equilibrium")
        closure.initialize(prims)
        first = closure.e_KH.clone()
        for _ in range(20):
            closure.advance(prims, 0.01)
        assert torch.allclose(closure.e_KH, first, rtol=1e-14)

    def test_rejects_an_unknown_mode(self):
        from src.minit import MInITClosure

        with pytest.raises(ValueError, match="unknown MInIT mode"):
            MInITClosure(dx=0.1, dy=0.1, mode="bogus")


class TestScoring:
    def test_perfect_prediction_scores_zero_error_and_unit_correlation(self):
        _, tau = _synthetic()
        s = score_tau(tau, tau)
        assert s["rel_l2"] == pytest.approx(0.0, abs=1e-12)
        assert s["corr"] == pytest.approx(1.0, abs=1e-9)

    def test_scaled_prediction_keeps_correlation_but_loses_magnitude(self):
        _, tau = _synthetic()
        s = score_tau(0.5 * tau, tau)
        assert s["corr"] == pytest.approx(1.0, abs=1e-9)
        assert s["rel_l2"] == pytest.approx(0.5, rel=1e-9)

    def test_reports_a_correlation_per_component(self):
        _, tau = _synthetic()
        s = score_tau(tau, tau)
        for name in ("xx", "yy", "xy"):
            assert s[f"corr_{name}"] == pytest.approx(1.0, abs=1e-9)

    def test_energy_transfer_sign_convention(self):
        r"""Pi > 0 drains resolved kinetic energy; the cross term counts twice."""
        tau = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=DTYPE)
        S = torch.tensor([[[2.0, 0.0, 0.0]]], dtype=DTYPE)
        assert float(energy_transfer(tau, S)) == pytest.approx(2.0)

        tau_xy = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=DTYPE)
        S_xy = torch.tensor([[[0.0, 0.0, 3.0]]], dtype=DTYPE)
        assert float(energy_transfer(tau_xy, S_xy)) == pytest.approx(6.0)


class TestCoefficientFiles:
    def test_equilibrium_file_naming_matches_what_evaluate_rollout_expects(self):
        """evaluate_rollout derives the equilibrium path from the main one by suffix."""
        from pathlib import Path

        main = Path("results/minit/minit_coefficients.json")
        derived = main.with_name(main.stem + "_equilibrium.json")
        assert derived.name == "minit_coefficients_equilibrium.json"

    def test_round_trip_through_the_derived_path(self, tmp_path):
        from pathlib import Path

        main = tmp_path / "minit_coefficients.json"
        derived = Path(main).with_name(Path(main).stem + "_equilibrium.json")
        c = MInITCoefficients(A_gamma=A_GAMMA_PAPER, C=4.2)
        c.save(derived)
        assert derived.exists()
        assert MInITCoefficients.load(derived) == c


class TestFigureRunCollection:
    """The figures must pick up every run that was actually executed.

    MInIT was initially wired into the metrics but not the density panels or the
    GIF, because those had layouts hardcoded to DNS/TBNN/Exact/Unclosed. These
    pin the behaviour that replaced them.
    """

    @staticmethod
    def _hist(value=1.0, n=3, N=4):
        return {"rho": torch.full((n, N, N), value, dtype=DTYPE)}

    def test_collects_every_run(self):
        from src.evaluate_rollout import _collect_runs

        runs = _collect_runs(
            history_tbnn=self._hist(1.0),
            history_exact=self._hist(2.0),
            history_unclosed=self._hist(3.0),
            history_minit={"fitted": self._hist(4.0), "paper": self._hist(5.0)},
        )
        assert [label for label, _ in runs] == [
            "Unclosed (no SGS)",
            "MInIT (fitted)",
            "MInIT (paper)",
            "TBNN-Closed",
            "Exact DNS-Closed",
        ]

    def test_minit_variants_are_ordered_deterministically(self):
        from src.evaluate_rollout import _collect_runs

        runs = _collect_runs(None, None, None, {
            "paper": self._hist(), "equilibrium": self._hist(), "fitted": self._hist(),
        })
        assert [label for label, _ in runs] == [
            "MInIT (equilibrium)", "MInIT (fitted)", "MInIT (paper)",
        ]

    def test_absent_runs_are_skipped(self):
        from src.evaluate_rollout import _collect_runs

        runs = _collect_runs(None, None, self._hist(), None)
        assert [label for label, _ in runs] == ["Unclosed (no SGS)"]

    def test_minit_only_still_produces_panels(self):
        """With neither TBNN nor the oracle, the MInIT runs must still be plotted."""
        from src.evaluate_rollout import _collect_runs

        runs = _collect_runs(None, None, None, {"equilibrium": self._hist()})
        assert len(runs) == 1

    def test_histories_without_density_are_ignored(self):
        from src.evaluate_rollout import _collect_runs

        assert _collect_runs(None, None, {"t": [0.0]}, None) == []

    def test_empty_history_tensors_are_ignored(self):
        from src.evaluate_rollout import _collect_runs

        empty = {"rho": torch.empty(0, 4, 4, dtype=DTYPE)}
        assert _collect_runs(None, None, empty, None) == []

    def test_nothing_at_all_is_empty(self):
        from src.evaluate_rollout import _collect_runs

        assert _collect_runs(None, None, None, None) == []
        assert _collect_runs(None, None, None, {}) == []

    def test_animation_writes_a_file_covering_all_runs(self, tmp_path):
        from src.evaluate_rollout import create_rollout_animation

        n, N = 3, 4
        dns = {"t": [0.0, 0.1, 0.2], "rho": torch.rand(n, N, N, dtype=DTYPE)}
        out = create_rollout_animation(
            history_dns=dns,
            history_unclosed=self._hist(n=n, N=N),
            history_minit={"fitted": self._hist(2.0, n, N),
                           "equilibrium": self._hist(3.0, n, N)},
            output_path=str(tmp_path / "anim.gif"),
            fps=2,
        )
        assert (tmp_path / "anim.gif").exists()
        assert (tmp_path / "anim.gif").stat().st_size > 0
        assert out == str(tmp_path / "anim.gif")

    def test_animation_skips_cleanly_with_no_runs(self, tmp_path):
        from src.evaluate_rollout import create_rollout_animation

        dns = {"t": [0.0, 0.1], "rho": torch.rand(2, 4, 4, dtype=DTYPE)}
        out = create_rollout_animation(
            history_dns=dns, output_path=str(tmp_path / "none.gif"), fps=2,
        )
        assert not (tmp_path / "none.gif").exists()
        assert out == str(tmp_path / "none.gif")

    def test_animation_skips_cleanly_with_no_frames(self, tmp_path):
        from src.evaluate_rollout import create_rollout_animation

        dns = {"t": [], "rho": torch.empty(0, 4, 4, dtype=DTYPE)}
        out = create_rollout_animation(
            history_dns=dns,
            history_unclosed=self._hist(),
            output_path=str(tmp_path / "none.gif"),
            fps=2,
        )
        assert not (tmp_path / "none.gif").exists()


class TestSpectrumStyles:
    """Every MInIT variant must get its own curve on the spectrum plot.

    The first wiring zipped the variants against a two-element style list, so
    with three variants `zip` silently truncated and MInIT (paper) vanished from
    the figure while still appearing in the metrics.
    """

    def test_a_style_exists_for_every_shipped_variant(self):
        from src.evaluate_rollout import _MINIT_STYLE, _resolve_minit_variants

        for variant in _resolve_minit_variants(["all"]):
            assert variant in _MINIT_STYLE

    def test_styles_are_visually_distinct(self):
        from src.evaluate_rollout import _MINIT_STYLE

        keys = [(v["color"], v["ls"]) for v in _MINIT_STYLE.values()]
        assert len(set(keys)) == len(keys)

    def test_unknown_variant_still_gets_a_curve(self):
        """A future variant must fall back to a style, never be dropped."""
        from src.evaluate_rollout import _MINIT_STYLE

        style = _MINIT_STYLE.get("something_new", {"color": "tab:purple", "ls": "-"})
        assert style["color"] and style["ls"]
