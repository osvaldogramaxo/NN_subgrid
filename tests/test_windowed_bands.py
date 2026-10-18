"""windowed_band_ratios: per-frame geometric mean over snapshot-aligned frames."""
import math

import numpy as np
import torch

from src.evaluate_rollout import windowed_band_ratios


def _hist(t, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(t)
    return {"t": torch.tensor(t, dtype=torch.float64),
            "rho": 1.0 + 0.1 * torch.rand(n, 32, 32, generator=g, dtype=torch.float64),
            "vx": scale * torch.randn(n, 32, 32, generator=g, dtype=torch.float64),
            "vy": scale * torch.randn(n, 32, 32, generator=g, dtype=torch.float64)}


T = np.round(np.arange(0.0, 4.0001, 0.1), 10)
DNS = _hist(T)
ALL = np.ones(len(T), dtype=bool)


def _same(scale):
    h = {k: (v.clone() if k == "t" or k == "rho" else scale * v) for k, v in DNS.items()}
    return h


def test_identical_run_scores_one_and_zero_distance():
    w = windowed_band_ratios(DNS, {"a": _same(1.0)}, ALL, [(0.0, 4.0)])["0-4"]["a"]
    for k in ("band_ratio_low", "band_ratio_mid", "band_ratio_high", "e_ratio_total"):
        assert math.isclose(w[k], 1.0, rel_tol=1e-9)
    assert w["log_spectral_distance"] < 1e-9


def test_halved_velocity_gives_quarter_energy_in_every_band():
    w = windowed_band_ratios(DNS, {"a": _same(0.5)}, ALL, [(0.0, 4.0)])["0-4"]["a"]
    for k in ("band_ratio_low", "band_ratio_mid", "band_ratio_high"):
        assert math.isclose(w[k], 0.25, rel_tol=1e-9)


def test_invalid_frames_are_ignored():
    valid = ALL.copy()
    valid[1::2] = False                             # no snapshot close enough
    run = _same(1.0)
    run["vx"][1::2] *= 100.0                        # garbage only on those frames
    w = windowed_band_ratios(DNS, {"a": run}, valid, [(0.0, 4.0)])["0-4"]["a"]
    assert math.isclose(w["band_ratio_high"], 1.0, rel_tol=1e-9)
    assert w["n_frames"] == int(valid.sum())


def test_diverged_run_uses_only_its_frames():
    run = {k: (v[:11] if k != "t" else v[:11]) for k, v in _same(1.0).items()}
    w = windowed_band_ratios(DNS, {"a": run}, ALL, [(0.0, 4.0)])["0-4"]["a"]
    assert w["n_frames"] == 11 and w["partial"]


def test_bootstrap_error_needs_window_longer_than_a_block():
    short = windowed_band_ratios(DNS, {"a": _same(0.5)}, ALL, [(0.0, 0.4)])["0-0.4"]["a"]
    assert math.isnan(short["high_err"])            # 5 frames = one 0.5-time-unit block
    long = windowed_band_ratios(DNS, {"a": _same(0.5)}, ALL, [(0.0, 4.0)])["0-4"]["a"]
    assert long["high_err"] < 1e-9                  # constant ratio -> zero spread


def test_paired_ratio_to_baseline_cancels_the_reference():
    runs = {"oracle": _same(1.0), "half": _same(0.5)}
    w = windowed_band_ratios(DNS, runs, ALL, [(0.0, 4.0)], baseline="oracle")["0-4"]
    assert math.isclose(w["half"]["vs_baseline_high"], 0.25, rel_tol=1e-9)
    assert math.isclose(w["half"]["vs_baseline_lo"], 0.25, rel_tol=1e-9)   # shared resample:
    assert math.isclose(w["half"]["vs_baseline_hi"], 0.25, rel_tol=1e-9)   # no spread at all
    assert math.isclose(w["oracle"]["vs_baseline_high"], 1.0, rel_tol=1e-12)


def test_misaligned_frames_are_dropped_and_flagged():
    run = _same(1.0)
    run["t"] = run["t"].clone()
    run["t"][20:] += 0.1                            # a skipped save shifts the tail by a frame
    run["vx"][20:] *= 100.0
    w = windowed_band_ratios(DNS, {"a": run}, ALL, [(0.0, 4.0)], t_tol=0.025)["0-4"]["a"]
    assert w["n_frames"] == 20 and w["partial"]
    assert math.isclose(w["band_ratio_high"], 1.0, rel_tol=1e-9)


def test_perturbation_is_small_large_scale_and_reproducible():
    from src.evaluate_rollout import perturb_initial_state

    g = torch.Generator().manual_seed(1)
    c = {k: torch.randn(64, 64, generator=g, dtype=torch.float64) for k in ("vx_tilde", "vy_tilde")}
    c["rho_bar"] = torch.ones(64, 64, dtype=torch.float64)
    a, b = perturb_initial_state(c, 7, 1e-3), perturb_initial_state(c, 7, 1e-3)
    assert torch.equal(a["vx_tilde"], b["vx_tilde"])                      # reproducible
    assert not torch.equal(a["vx_tilde"], perturb_initial_state(c, 8)["vx_tilde"])
    d = a["vx_tilde"] - c["vx_tilde"]
    rms = float(torch.sqrt((c["vx_tilde"] ** 2 + c["vy_tilde"] ** 2).mean()))
    assert abs(float(d.std()) / rms - 1e-3) < 1e-6                         # amplitude
    P = torch.fft.fft2(d).abs() ** 2
    k = torch.fft.fftfreq(64, d=1 / 64).abs()
    assert float(P[(k[:, None] ** 2 + k[None, :] ** 2).sqrt() > 8].sum()) < 1e-20  # k <= N/8
    assert torch.equal(a["rho_bar"], c["rho_bar"])
