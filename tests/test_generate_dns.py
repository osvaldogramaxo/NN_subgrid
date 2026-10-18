r"""
test_generate_dns.py

Tests for the relativistic KHI DNS generator.

Verifies:
  1. The seed perturbation is solenoidal and spans the requested band
  2. The base state has the intended shear/density structure
  3. Snapshots round-trip exactly through `src.dataset.read_snapshot`
  4. The stored `hydro` group really is the relativistic conserved state
  5. Ghost cells are a true periodic wrap
  6. `set_gamma_eos` reaches both modules
"""

import sys
sys.path.insert(0, ".")

import h5py
import numpy as np
import pytest
import torch

import src.dataset as dataset
import src.solver as solver
from src.dataset import read_snapshot
from src.generate_dns import (
    GHOST,
    khi_initial_conditions,
    ke_spectrum,
    lorentz,
    sound_speed,
    specific_enthalpy,
    spectral_diagnostics,
    write_snapshot,
)

IC_KW = dict(
    v_shear=0.8, thickness=0.02, rho1=1.0, rho2=2.0, pressure=2.5,
    seed_amplitude=0.01, seed_kmin=4, seed_kmax=16,
    seed_envelope_width=None, rng_seed=0,
)


def _base_state(N, **over):
    """IC with the seed switched off, for isolating base vs perturbation."""
    kw = {**IC_KW, **over, "seed_amplitude": 0.0}
    return khi_initial_conditions(N, N, **kw)


def test_base_state_structure():
    N = 128
    rho, vx, vy, p, x, y = _base_state(N)

    assert np.allclose(vy, 0.0), "unseeded base state must have v_y = 0"
    assert np.allclose(p, IC_KW["pressure"]), "base state must be in pressure equilibrium"
    # Density low outside the layers, high between them; shear flips sign.
    assert rho[0].mean() == pytest.approx(IC_KW["rho1"], abs=1e-6)
    assert rho[N // 2].mean() == pytest.approx(IC_KW["rho2"], abs=1e-6)
    assert vx[0].mean() == pytest.approx(+IC_KW["v_shear"], abs=1e-6)
    assert vx[N // 2].mean() == pytest.approx(-IC_KW["v_shear"], abs=1e-6)
    # Two shear layers, so v_x has zero mean over the box.
    assert vx.mean() == pytest.approx(0.0, abs=1e-9)


def test_seed_is_solenoidal():
    r"""The perturbation comes from a streamfunction, so ∇·δv must vanish.

    Checked against the divergence of a *non*-solenoidal field of the same
    amplitude, so the tolerance is relative to what a naive seed would give.
    """
    N = 256
    rho0, vx0, vy0, p0, _, _ = _base_state(N)
    rho, vx, vy, p, _, _ = khi_initial_conditions(N, N, **IC_KW)
    dvx, dvy = vx - vx0, vy - vy0
    d = 1.0 / N

    def div(ax, ay):
        return (np.roll(ax, -1, axis=1) - np.roll(ax, 1, axis=1)) / (2 * d) + (
            np.roll(ay, -1, axis=0) - np.roll(ay, 1, axis=0)
        ) / (2 * d)

    solenoidal = np.abs(div(dvx, dvy)).max()
    # Same field with the components swapped is generically compressive.
    reference = np.abs(div(dvy, dvx)).max()
    assert solenoidal < 1e-2 * reference, (
        f"seed divergence {solenoidal:.3e} is not small against {reference:.3e}"
    )


def test_seed_is_broadband_and_band_limited():
    N = 256
    rho0, vx0, vy0, _, _, y = _base_state(N)
    rho, vx, vy, _, _, _ = khi_initial_conditions(N, N, **IC_KW)
    row = (vy - vy0)[np.argmin(np.abs(y - 0.25))]  # on a shear layer
    amp = np.abs(np.fft.rfft(row)) / len(row)

    kmin, kmax = IC_KW["seed_kmin"], IC_KW["seed_kmax"]
    in_band = amp[kmin : kmax + 1]
    assert (in_band > 1e-3 * in_band.max()).all(), "some requested modes are unexcited"
    assert amp[: kmin].max() < 1e-6 * in_band.max(), "power below the seeded band"
    assert amp[kmax + 1 :].max() < 1e-6 * in_band.max(), "power above the seeded band"
    # A single-mode seed would have one dominant k; a broadband one must not.
    assert in_band.max() / in_band.mean() < 3.0, "seed energy is concentrated in one mode"


def test_seed_amplitude_is_respected():
    N = 128
    rho0, vx0, vy0, _, _, _ = _base_state(N)
    for a in (0.005, 0.02):
        rho, vx, vy, _, _, _ = khi_initial_conditions(N, N, **{**IC_KW, "seed_amplitude": a})
        peak = np.sqrt((vx - vx0) ** 2 + (vy - vy0) ** 2).max()
        assert peak == pytest.approx(a, rel=1e-10)


def test_superluminal_ic_is_rejected():
    with pytest.raises(ValueError, match="superluminal"):
        khi_initial_conditions(64, 64, **{**IC_KW, "v_shear": 0.999, "seed_amplitude": 0.5})


def test_seed_band_above_nyquist_quarter_is_rejected():
    with pytest.raises(ValueError, match="seed_kmax"):
        khi_initial_conditions(64, 64, **{**IC_KW, "seed_kmax": 32})


def test_snapshot_roundtrips_through_read_snapshot(tmp_path):
    N, gamma = 64, 4.0 / 3.0
    rho, vx, vy, p, x, y = khi_initial_conditions(N, N, **IC_KW)
    path = tmp_path / "h00000000.h5"
    write_snapshot(path, rho, vx, vy, p, 1.25, x, y, gamma, {"test": True})

    snap = read_snapshot(path)
    assert snap["t"] == 1.25
    for name, ref in (("rho", rho), ("vx", vx), ("vy", vy), ("p", p)):
        assert np.array_equal(snap[name], ref), f"{name} did not round-trip exactly"


def test_stored_hydro_is_the_relativistic_conserved_state(tmp_path):
    r"""`hydro` must hold (D, E, S_x, S_y, S_z) with D = ρW, E = ρhW² - p.

    This is the check that would have caught the Newtonian/relativistic
    convention mismatch in the original dataset: invert the stored conserved
    variables with the solver's own C2P and demand the primitives back.
    """
    N, gamma = 64, 4.0 / 3.0
    solver.set_gamma_eos(gamma)
    rho, vx, vy, p, x, y = khi_initial_conditions(N, N, **IC_KW)
    path = tmp_path / "h00000000.h5"
    write_snapshot(path, rho, vx, vy, p, 0.0, x, y, gamma, {})

    with h5py.File(path) as f:
        hydro = f["hydro"]["data"][0, GHOST:-GHOST, GHOST:-GHOST, :]
        thd = f["thd"]["data"][0, GHOST:-GHOST, GHOST:-GHOST, :]
        meta = dict(f["Metadata"].attrs)

    D, E, Sx, Sy, Sz = (hydro[..., i] for i in range(5))
    W = lorentz(vx, vy)
    assert np.allclose(D, rho * W, rtol=1e-14)
    assert np.allclose(Sz, 0.0)

    # The solver's state vector uses tau = E - D.
    U = torch.from_numpy(np.stack([D, Sx, Sy, E - D], axis=-1))
    rho_r, vx_r, vy_r, p_r = solver.conserved_to_primitives(U)
    assert np.allclose(rho_r.numpy(), rho, rtol=1e-12)
    assert np.allclose(vx_r.numpy(), vx, atol=1e-12)
    assert np.allclose(vy_r.numpy(), vy, atol=1e-12)
    assert np.allclose(p_r.numpy(), p, rtol=1e-12)

    assert np.allclose(thd[..., 1], sound_speed(rho, p, gamma), rtol=1e-14)
    assert meta["scheme"] == "SRHD"
    assert meta["eos_gamma"] == pytest.approx(gamma)


def test_ghost_cells_are_a_periodic_wrap(tmp_path):
    N = 64
    rho, vx, vy, p, x, y = khi_initial_conditions(N, N, **IC_KW)
    path = tmp_path / "h00000000.h5"
    write_snapshot(path, rho, vx, vy, p, 0.0, x, y, 4.0 / 3.0, {})

    with h5py.File(path) as f:
        full = f["prim"]["data"][0, :, :, 0]
    assert full.shape == (N + 2 * GHOST, N + 2 * GHOST)
    assert np.array_equal(full[:GHOST, GHOST:-GHOST], rho[-GHOST:, :])
    assert np.array_equal(full[-GHOST:, GHOST:-GHOST], rho[:GHOST, :])
    assert np.array_equal(full[GHOST:-GHOST, :GHOST], rho[:, -GHOST:])
    assert np.array_equal(full[GHOST:-GHOST, -GHOST:], rho[:, :GHOST])


def test_set_gamma_eos_reaches_both_modules():
    original = solver.GAMMA_EOS
    try:
        solver.set_gamma_eos(4.0 / 3.0)
        assert solver.GAMMA_EOS == pytest.approx(4.0 / 3.0)
        assert dataset.GAMMA_EOS == pytest.approx(4.0 / 3.0)
        # The value must actually be used, not merely stored: c_s = sqrt(Γ p/(ρh)).
        rho = torch.ones(4, 4, dtype=torch.float64)
        p = torch.full((4, 4), 2.5, dtype=torch.float64)
        v = torch.zeros(4, 4, dtype=torch.float64)
        lam_m, lam_p = solver.compute_wave_speeds(rho, v, v, p, direction=0)
        expected = float(sound_speed(np.ones((1,)), np.full((1,), 2.5), 4.0 / 3.0)[0])
        assert float(lam_p.max()) == pytest.approx(expected, rel=1e-12)
    finally:
        solver.set_gamma_eos(original)


def test_spectral_diagnostics_measure_subgrid_fraction():
    r"""A field built from one mode above the cutoff must be ~100% subgrid,
    and one below it ~0%, or the go/no-go number means nothing."""
    N, r = 64, 8
    k_c = N / (2 * r)  # = 4
    xs = (np.arange(N) + 0.5) / N
    X, _ = np.meshgrid(xs, xs, indexing="xy")
    rho = np.ones((N, N))

    for k_mode, expect_subgrid in ((2, False), (16, True)):
        vx = np.sin(2 * np.pi * k_mode * X)
        kk, E_k = ke_spectrum(rho, vx, np.zeros_like(vx))
        d = spectral_diagnostics(kk, E_k, r, N, fit_lo=2, fit_hi=8, fluct_kmin=1)
        assert d["k_c"] == k_c
        if expect_subgrid:
            assert d["subgrid_fraction"] > 0.99
        else:
            assert d["subgrid_fraction"] < 0.01


def test_enthalpy_and_sound_speed_obey_the_relativistic_bound():
    r"""c_s must stay below sqrt(Γ-1), the ultrarelativistic ceiling — the
    check that exposed the original data as Newtonian (it had c_s = 2.04)."""
    gamma = 4.0 / 3.0
    rho = np.array([1e-3, 1.0, 1e3])
    p = np.array([1e3, 2.5, 1e-3])
    h = specific_enthalpy(rho, p, gamma)
    cs = sound_speed(rho, p, gamma)
    assert (h >= 1.0).all()
    assert (cs < np.sqrt(gamma - 1.0)).all()
    assert (cs < 1.0).all()


def test_mean_flow_does_not_dilute_the_fluctuation_fraction():
    r"""In a shear flow most of the KE is in the mean flow at k~1. Measured at
    512^2, 86% sat at k=1, making a healthy cascade read 0.5% of total energy
    while being 8% of the fluctuations. The fluctuation fraction must ignore
    the mean flow; the total fraction must not."""
    N, r = 64, 8
    k_c = N / (2 * r)  # = 4
    xs = (np.arange(N) + 0.5) / N
    X, _ = np.meshgrid(xs, xs, indexing="xy")
    rho = np.ones((N, N))
    # A huge k=1 "mean flow" plus a small k=16 subgrid fluctuation.
    vx = 10.0 * np.sin(2 * np.pi * X) + 0.1 * np.sin(2 * np.pi * 16 * X)
    kk, E_k = ke_spectrum(rho, vx, np.zeros_like(vx))
    d = spectral_diagnostics(kk, E_k, r, N, fit_lo=2, fit_hi=8, fluct_kmin=4)

    assert d["k_c"] == k_c
    assert d["subgrid_fraction"] < 0.001, "mean flow should dominate the total"
    assert d["subgrid_fraction_fluct"] > 0.99, "but it is all of the fluctuation energy"


def test_relativistic_mach_exceeds_the_newtonian_estimate():
    r"""The bug that let a stable run through: 2v/c_s said 2.91 at v=0.8 where
    the relativistic Mach number is 6.74, and the KHI is suppressed."""
    from src.generate_dns import shear_mach, unstable_kmax
    cs = float(sound_speed(np.array([1.0]), np.array([2.5]), 4.0 / 3.0)[0])

    dv, M = shear_mach(0.8, cs)
    assert dv == pytest.approx(2 * 0.8 / (1 + 0.64)), "velocities must add relativistically"
    assert dv < 2 * 0.8, "relativistic relative velocity is below the Newtonian sum"
    assert M == pytest.approx(6.74, abs=0.05)
    assert M > 2 * 0.8 / cs, "relativistic Mach must exceed the Newtonian estimate"

    # Monotone in shear, and the growth threshold sits near v = 0.5.
    machs = [shear_mach(v, cs)[1] for v in (0.2, 0.3, 0.4, 0.5, 0.6)]
    assert machs == sorted(machs)
    assert shear_mach(0.5, cs)[1] == pytest.approx(2.0, abs=0.1)


def test_unstable_band_cutoff():
    r"""A tanh layer of half-thickness a amplifies only k_phys a < 1."""
    from src.generate_dns import unstable_kmax
    assert unstable_kmax(0.02) == pytest.approx(1.0 / (2 * np.pi * 0.02))
    assert unstable_kmax(0.02) == pytest.approx(7.96, abs=0.01)
    # Thinner layers admit more modes.
    assert unstable_kmax(0.005) > unstable_kmax(0.02)
