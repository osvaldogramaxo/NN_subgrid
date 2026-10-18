r"""
test_config.py

Tests for configuration loading and EOS provenance checking.

The point of this layer is to make the Newtonian/relativistic mismatch that
invalidated the first round of this project *impossible to reintroduce
silently*, so the tests centre on the failure modes rather than the happy path:
a config that contradicts the data, a typo that does nothing, data with no
metadata at all, and a circular check that would validate anything.
"""

import sys
sys.path.insert(0, ".")

import argparse
import warnings

import numpy as np
import pytest

import src.dataset as dataset
import src.solver as solver
from src.config import (
    Config,
    add_config_arguments,
    apply_config_defaults,
    load_config,
    read_dataset_metadata,
    resolve_gamma,
    verify_dataset_eos,
)
from src.generate_dns import khi_initial_conditions, write_snapshot

IC_KW = dict(
    v_shear=0.8, thickness=0.02, rho1=1.0, rho2=2.0, pressure=2.5,
    seed_amplitude=0.01, seed_kmin=4, seed_kmax=16,
    seed_envelope_width=None, rng_seed=0,
)


def _write(tmp_path, gamma=4.0 / 3.0, n=64):
    rho, vx, vy, p, x, y = khi_initial_conditions(n, n, **IC_KW)
    write_snapshot(tmp_path / "h00000000.h5", rho, vx, vy, p, 0.0, x, y, gamma, {})
    return tmp_path


def _cfg_file(tmp_path, body):
    path = tmp_path / "crmhd.toml"
    path.write_text(body)
    return path


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def test_defaults_when_no_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CRMHD_CONFIG", raising=False)
    monkeypatch.setattr("src.config.REPO_ROOT", tmp_path)
    cfg = load_config()
    assert cfg.source is None
    assert cfg.physics.gamma == pytest.approx(4.0 / 3.0)
    assert cfg.explicit == frozenset()


def test_explicit_keys_are_tracked(tmp_path):
    path = _cfg_file(tmp_path, "[physics]\ngamma = 1.6666666666666667\n")
    cfg = load_config(path)
    assert cfg.physics.gamma == pytest.approx(5.0 / 3.0)
    assert "physics.gamma" in cfg.explicit
    # Untouched keys keep defaults but must not be reported as explicit.
    assert "solver.cfl" not in cfg.explicit


def test_unknown_key_is_rejected(tmp_path):
    r"""A typo that silently does nothing is worse than no config at all."""
    path = _cfg_file(tmp_path, "[physics]\ngammma = 1.4\n")
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path)


def test_unknown_section_is_rejected(tmp_path):
    path = _cfg_file(tmp_path, "[phsyics]\ngamma = 1.4\n")
    with pytest.raises(ValueError, match="unknown section"):
        load_config(path)


def test_integer_gamma_is_accepted_as_float(tmp_path):
    """TOML `gamma = 2` is an int; it must not be rejected as a type error."""
    cfg = load_config(_cfg_file(tmp_path, "[physics]\ngamma = 2\n"))
    assert isinstance(cfg.physics.gamma, float)
    assert cfg.physics.gamma == 2.0


def test_wrong_type_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be a number"):
        load_config(_cfg_file(tmp_path, '[physics]\ngamma = "four thirds"\n'))


def test_unphysical_gamma_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must exceed 1"):
        load_config(_cfg_file(tmp_path, "[physics]\ngamma = 0.5\n"))


def test_missing_explicit_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")


def test_env_var_is_honoured(tmp_path, monkeypatch):
    path = _cfg_file(tmp_path, "[solver]\ncfl = 0.123\n")
    monkeypatch.setenv("CRMHD_CONFIG", str(path))
    assert load_config().solver.cfl == pytest.approx(0.123)


# ---------------------------------------------------------------------------
# Gamma resolution: the dataset outranks the config
# ---------------------------------------------------------------------------

def test_dataset_metadata_beats_config(tmp_path):
    (tmp_path / "data").mkdir()
    data = _write(tmp_path / "data", gamma=4.0 / 3.0)
    cfg = load_config(_cfg_file(tmp_path, "[physics]\ngamma = 1.3333333333333333\n"))
    assert read_dataset_metadata(data)["eos_gamma"] == pytest.approx(4.0 / 3.0)
    assert resolve_gamma(cfg, data) == pytest.approx(4.0 / 3.0)


def test_config_contradicting_the_data_is_an_error(tmp_path):
    (tmp_path / "data").mkdir()
    data = _write(tmp_path / "data", gamma=4.0 / 3.0)
    cfg = load_config(_cfg_file(tmp_path, "[physics]\ngamma = 1.6666666666666667\n"))
    with pytest.raises(ValueError, match="EOS mismatch"):
        resolve_gamma(cfg, data)


def test_explicit_override_wins_over_everything(tmp_path):
    (tmp_path / "data").mkdir()
    data = _write(tmp_path / "data", gamma=4.0 / 3.0)
    cfg = load_config(_cfg_file(tmp_path, "[physics]\ngamma = 1.6666666666666667\n"))
    assert resolve_gamma(cfg, data, override=1.4) == pytest.approx(1.4)


def test_data_without_metadata_warns(tmp_path):
    r"""Legacy data (test1/) records nothing, so nothing has been verified."""
    (tmp_path / "data").mkdir()
    cfg = load_config(_cfg_file(tmp_path, "[physics]\ngamma = 1.6666666666666667\n"))
    with pytest.warns(RuntimeWarning, match="no EOS metadata"):
        g = resolve_gamma(cfg, tmp_path / "data")
    assert g == pytest.approx(5.0 / 3.0)


def test_non_explicit_config_gamma_does_not_conflict(tmp_path):
    """An inherited default must not be treated as a contradiction."""
    (tmp_path / "data").mkdir()
    data = _write(tmp_path / "data", gamma=5.0 / 3.0)
    cfg = load_config(_cfg_file(tmp_path, "[solver]\ncfl = 0.4\n"))  # gamma untouched
    assert resolve_gamma(cfg, data) == pytest.approx(5.0 / 3.0)


def test_apply_pushes_gamma_into_both_modules():
    original = solver.GAMMA_EOS
    try:
        Config().apply(1.4)
        assert solver.GAMMA_EOS == pytest.approx(1.4)
        assert dataset.GAMMA_EOS == pytest.approx(1.4)
    finally:
        solver.set_gamma_eos(original)


# ---------------------------------------------------------------------------
# argparse precedence
# ---------------------------------------------------------------------------

def test_cli_overrides_config_overrides_builtin(tmp_path):
    path = _cfg_file(tmp_path, "[solver]\ncfl = 0.11\n[data]\ncoarsening_factor = 32\n")
    mapping = {"cfl": "solver.cfl", "coarsening_factor": "data.coarsening_factor"}

    def build():
        ap = argparse.ArgumentParser()
        ap.add_argument("--cfl", type=float, default=0.99)
        ap.add_argument("--coarsening-factor", type=int, default=4)
        add_config_arguments(ap)
        return ap

    # config beats the built-in default
    ap = build()
    apply_config_defaults(ap, mapping, ["--config", str(path)])
    args = ap.parse_args(["--config", str(path)])
    assert args.cfl == pytest.approx(0.11)
    assert args.coarsening_factor == 32

    # an explicit flag beats the config
    ap = build()
    apply_config_defaults(ap, mapping, ["--config", str(path), "--cfl", "0.55"])
    args = ap.parse_args(["--config", str(path), "--cfl", "0.55"])
    assert args.cfl == pytest.approx(0.55)
    assert args.coarsening_factor == 32


# ---------------------------------------------------------------------------
# EOS verification
# ---------------------------------------------------------------------------

def test_verify_identifies_generated_data_as_srhd(tmp_path):
    (tmp_path / "data").mkdir()
    data = _write(tmp_path / "data", gamma=4.0 / 3.0)
    rep = verify_dataset_eos(data)
    assert "SRHD" in rep["verdict"]
    assert rep["best_relativistic"][0] == "gamma=1.333333"
    assert rep["best_relativistic"][1] < 1e-12
    # The Newtonian reading must be decisively worse, not marginally.
    assert rep["best_newtonian"][1] > 1e-3


def test_verify_recovers_gamma_from_sound_speed(tmp_path):
    (tmp_path / "data").mkdir()
    data = _write(tmp_path / "data", gamma=4.0 / 3.0)
    rep = verify_dataset_eos(data)
    g = rep["gamma_relativistic_from_cs"]
    assert g["median"] == pytest.approx(4.0 / 3.0, rel=1e-10)
    assert g["std"] < 1e-12


def test_srhd_energy_check_is_marked_circular_without_prim(tmp_path):
    r"""Regression guard for a real flaw in an earlier version of this check.

    With no `prim` group the primitives come from C2P, which *solves*
    E = ρhW² − p. Re-deriving E from them then closes to ~1e-16 on absolutely
    any data, Newtonian included, and would have declared test1/ relativistic.
    It must be reported as circular, not as a residual.
    """
    import h5py
    (tmp_path / "data").mkdir()
    data = _write(tmp_path / "data", gamma=4.0 / 3.0)
    with h5py.File(data / "h00000000.h5", "a") as f:
        del f["prim"]
        del f["Metadata"]

    rep = verify_dataset_eos(data)
    assert rep["has_prim_group"] is False
    for entry in rep["relativistic"].values():
        assert entry.get("energy_residual") is None, "circular check reported as evidence"
        assert entry["cs_residual"] is not None, "no independent check remains"
    # The sound speed still identifies it correctly.
    assert "SRHD" in rep["verdict"]


@pytest.mark.skipif(not __import__("os").path.isdir("test1"),
                    reason="test1/ dataset not present")
def test_verify_identifies_test1_as_newtonian():
    r"""The case this machinery exists for: Γ=5/3 Newtonian data that the
    pipeline had been reading as Γ=4/3 relativistic."""
    rep = verify_dataset_eos("test1")
    assert "NEWTONIAN" in rep["verdict"]
    assert rep["max_cs"] > 1.0, "a relativistic sound speed cannot exceed c"
    assert rep["gamma_newtonian_from_cs"]["median"] == pytest.approx(5.0 / 3.0, rel=1e-9)
    assert rep["gamma_newtonian_from_cs"]["std"] < 1e-12
    assert rep["best_newtonian"][0] == "gamma=1.666667"
    assert rep["best_newtonian"][1] < 1e-12


def test_compile_flag_round_trips(tmp_path):
    cfg = load_config(_cfg_file(tmp_path, "[solver]\ncompile = false\n"))
    assert cfg.solver.compile is False
    assert "solver.compile" in cfg.explicit


def test_non_boolean_compile_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be true or false"):
        load_config(_cfg_file(tmp_path, '[solver]\ncompile = "yes"\n'))


def test_apply_propagates_the_compile_flag():
    original_gamma, original = solver.GAMMA_EOS, solver._COMPILE_ENABLED
    try:
        cfg = Config()
        cfg.solver.compile = False
        cfg.apply(4.0 / 3.0)
        assert solver._COMPILE_ENABLED is False
        cfg.solver.compile = True
        cfg.apply(4.0 / 3.0)
        assert solver._COMPILE_ENABLED is True
    finally:
        solver.set_compile(original)
        solver.set_gamma_eos(original_gamma)


def test_set_gamma_eos_discards_compiled_graphs():
    """Gamma is a baked-in constant, so a stale graph would silently use the old one."""
    solver._COMPILED_RHS = object()
    solver._COMPILED_NEWTON = object()
    solver.set_gamma_eos(solver.GAMMA_EOS)
    assert solver._COMPILED_RHS is None
    assert solver._COMPILED_NEWTON is None
