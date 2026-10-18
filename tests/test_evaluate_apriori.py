"""Tests for cross-dataset a-priori scoring (:mod:`src.evaluate_apriori`)."""

import numpy as np
import pytest
import torch

from src.evaluate_apriori import highk_fraction, report, shell_power

DTYPE = torch.float64


def _mode(n: int, k: int) -> torch.Tensor:
    """A single-wavenumber field, so its spectral content is known exactly."""
    x = torch.arange(n, dtype=DTYPE)
    return torch.sin(2 * torch.pi * k * x / n).unsqueeze(-1).expand(n, n).contiguous()


class TestShellPower:
    def test_puts_a_single_mode_in_its_own_shell(self):
        k, E = shell_power(_mode(32, 6))
        assert int(np.argmax(E)) == 6

    def test_dc_field_has_no_power_above_k_zero(self):
        k, E = shell_power(torch.ones(16, 16, dtype=DTYPE))
        assert float(E[k >= 1].sum()) == pytest.approx(0.0, abs=1e-18)


class TestHighKFraction:
    def test_pure_high_mode_is_all_high_k(self):
        assert highk_fraction(_mode(32, 14), k_hi=10) == pytest.approx(1.0, abs=1e-12)

    def test_pure_low_mode_is_no_high_k(self):
        assert highk_fraction(_mode(32, 3), k_hi=10) == pytest.approx(0.0, abs=1e-12)

    def test_mixture_lands_between(self):
        f = _mode(32, 3) + _mode(32, 14)
        frac = highk_fraction(f, k_hi=10)
        assert 0.2 < frac < 0.8

    def test_is_scale_invariant(self):
        """A model that is uniformly too weak must not look spectrally different."""
        f = _mode(32, 3) + 0.5 * _mode(32, 14)
        assert highk_fraction(f, 10) == pytest.approx(highk_fraction(0.01 * f, 10))


class TestReport:
    @staticmethod
    def _res(amps, **kw):
        rows = [
            {
                "t": float(i), "seen_in_training": False, "amp_ratio": a,
                "rel_l2": 0.5, "corr": 0.8, "hifrac_pred": 0.3,
                "hifrac_exact": 0.5, "pi_pred": -1.0, "pi_exact": -2.0,
                "pi_ratio": 0.5, "pi_sign_ok": True,
            }
            for i, a in enumerate(amps)
        ]
        return {"checkpoint": "ck/best_model.pt", "data_dir": "d",
                "coarsening_factor": 16, "k_hi": 16, "train_hi": 6.4,
                "rows": rows, **kw}

    def test_flags_a_decaying_stress(self, capsys):
        report(self._res([1.0, 1.0, 0.4, 0.3]))
        assert "The stress decays over the run" in capsys.readouterr().out

    def test_stays_quiet_when_the_stress_holds_up(self, capsys):
        report(self._res([0.9, 0.95, 0.9, 0.92]))
        assert "decays over the run" not in capsys.readouterr().out

    def test_handles_an_empty_result(self, capsys):
        report({"checkpoint": "c", "data_dir": "d", "coarsening_factor": 16,
                "k_hi": 16, "train_hi": None, "rows": []})
        assert "No snapshots in range" in capsys.readouterr().out

    def test_marks_which_snapshots_the_model_saw(self, capsys):
        res = self._res([0.9, 0.9])
        res["rows"][0]["seen_in_training"] = True
        report(res)
        out = capsys.readouterr().out
        assert " yes " in out and " no " in out
