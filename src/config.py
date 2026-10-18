r"""
config.py

Project configuration, and the EOS provenance checks that go with it.

Why this module exists
----------------------
`GAMMA_EOS` used to be a module-level constant in `src.dataset`, imported by
`src.solver` at import time. Nothing tied it to the data being read. The
`test1/` snapshots turned out to be Newtonian with Γ = 5/3 while the whole
pipeline assumed relativistic Γ = 4/3, and that mismatch silently invalidated
months of results.

A config file alone would not have caught that — a config can disagree with
the data just as quietly as a hardcoded constant. So the rule here is:

    **the dataset's own metadata outranks the config file.**

Precedence, strongest first:
    1. an explicit override (`--gamma` on the command line)
    2. `Metadata/eos_gamma` stored in the dataset
    3. the config file
    4. built-in defaults

and a config that *contradicts* dataset metadata is an error, not a warning.
For legacy data carrying no metadata the config value is used, with a warning,
and `verify_dataset_eos` is available to establish the truth numerically.

Usage
-----
    from src.config import load_config, resolve_gamma

    cfg = load_config(args.config)
    gamma = resolve_gamma(cfg, data_dir=args.data_dir, override=args.gamma)
    cfg.apply(gamma)          # pushes Γ into src.solver and src.dataset

Command line:
    python -m src.config show                  # print the resolved config
    python -m src.config verify test1          # what EOS is this data, really?
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import sys
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # Python 3.10 with the tomli backport
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILENAME = "crmhd.toml"
GHOST = 3


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class PhysicsConfig:
    """Equation of state and fluid model.

    gamma : adiabatic index. 4/3 for a relativistically hot ideal gas,
        5/3 for a non-relativistic monatomic one. This *must* match the data;
        see `verify_dataset_eos`.
    scheme : "SRHD" or "Newtonian". Recorded for provenance and checked
        against dataset metadata; the solver only implements SRHD.
    """
    gamma: float = 4.0 / 3.0
    scheme: str = "SRHD"


@dataclass
class SolverConfig:
    cfl: float = 0.4
    reconstruction: str = "mp5"
    riemann_solver: str = "hllc"
    dtype: str = "float64"
    device: str = "auto"
    #: lazily torch.compile the solver hot path on CUDA. Worth ~9x at 1024^2,
    #: at the cost of ~17 s of tracing per distinct call signature, so turn it
    #: off for very short runs where that would not amortise.
    compile: bool = True


@dataclass
class DataConfig:
    data_dir: str = "test1"
    coarsening_factor: int = 16
    stride: int = 1


@dataclass
class TrainingConfig:
    """A priori training defaults.

    time_lo / time_hi : the window used for training and validation, which are
        interleaved inside it. For `dns_khi` (t = 0..8) the default starts at
        1.0: before that the subgrid fraction of kinetic energy is ~0.03%, so
        those snapshots carry no closure signal and, under a per-point
        normalized tensor loss, contribute amplified noise.
    test_lo / test_hi : a contiguous window held out for the final measurement
        only. Set test_lo >= test_hi to disable.
    val_every : one in every N snapshots of the window is held out for
        validation. Interleaved rather than contiguous because the flow is
        strongly non-stationary -- the subgrid fraction rises 40x from t = 0 to
        t = 8, so a contiguous window is a different regime, not a held-out
        sample of the same one.
    snapshots_per_batch : snapshots per optimizer step. Batching by snapshot
        keeps each step's points spatially contiguous, which is what lets the
        spectral cascade term be evaluated on the exact periodic field inside
        the same backward pass as the point-wise terms.
    monitor_ema : EMA factor for the validation metric that drives
        best-checkpoint selection, LR reduction and early stopping. 0 selects
        on the raw value, which on the previous run meant selecting noise.
    """
    time_lo: float = 1.0
    time_hi: float = 6.5
    test_lo: float = 6.5
    test_hi: float = 1e30
    val_every: int = 5
    snapshots_per_batch: int = 2
    lr: float = 1e-3
    lr_schedule: str = "plateau"
    warmup_epochs: int = 5
    grad_clip_mode: str = "adaptive"
    grad_clip_mult: float = 5.0
    monitor_ema: float = 0.8
    loss_alpha: float = 0.5
    loss_beta: float = 0.1
    loss_gamma: float = 0.0
    loss_delta: float = 2.0
    wec_margin: float = 0.5
    tensor_floor_quantile: float = 0.5
    tensor_floor_scope: str = "snapshot"
    spectral_highk_tilt: float = 0.0
    lambda_stab: float = 10.0


@dataclass
class Config:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    #: set by `load_config`; None when built-in defaults were used
    source: str | None = None
    #: keys the user set explicitly, as "section.key" — used to tell an
    #: intentional config value apart from an inherited default
    explicit: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name in ("source", "explicit"):
                continue
            out[f.name] = dataclasses.asdict(getattr(self, f.name))
        out["_source"] = self.source
        return out

    def apply(self, gamma: float | None = None) -> float:
        """Push the EOS into `src.solver` and `src.dataset`.

        Call before constructing a solver or reading data: `compute_dt` and
        `primitives_to_conserved` may be `torch.compile`d and can bake Γ into
        a traced graph.
        """
        from src.solver import set_compile, set_gamma_eos

        g = float(self.physics.gamma if gamma is None else gamma)
        set_compile(self.solver.compile)
        set_gamma_eos(g)   # discards any graph compiled against the old gamma
        self.physics.gamma = g
        return g


_SECTIONS = {
    "physics": PhysicsConfig,
    "solver": SolverConfig,
    "data": DataConfig,
    "training": TrainingConfig,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_config(path: str | Path | None = None) -> Path | None:
    """Locate a config file: explicit path, $CRMHD_CONFIG, ./crmhd.toml, repo root."""
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {p}")
        return p
    env = os.environ.get("CRMHD_CONFIG")
    if env:
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(f"$CRMHD_CONFIG points at a missing file: {p}")
        return p
    for candidate in (Path.cwd() / CONFIG_FILENAME, REPO_ROOT / CONFIG_FILENAME):
        if candidate.is_file():
            return candidate
    return None


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a TOML config, falling back to built-in defaults.

    Unknown sections and unknown keys raise rather than being ignored: a typo
    in a config file that silently does nothing is worse than no config file.
    """
    found = find_config(path)
    if found is None:
        return Config()

    with open(found, "rb") as fh:
        raw = tomllib.load(fh)

    unknown_sections = set(raw) - set(_SECTIONS)
    if unknown_sections:
        raise ValueError(
            f"{found}: unknown section(s) {sorted(unknown_sections)}; "
            f"expected any of {sorted(_SECTIONS)}"
        )

    kwargs: dict[str, Any] = {}
    explicit: set[str] = set()
    for name, cls in _SECTIONS.items():
        table = raw.get(name, {})
        if not isinstance(table, dict):
            raise ValueError(f"{found}: [{name}] must be a table, got {type(table).__name__}")
        # `from __future__ import annotations` makes dataclass field .type a
        # string, so resolve the real types or every check below silently passes.
        valid = get_type_hints(cls)
        unknown = set(table) - set(valid)
        if unknown:
            raise ValueError(
                f"{found}: unknown key(s) in [{name}]: {sorted(unknown)}; "
                f"expected any of {sorted(valid)}"
            )
        coerced = {}
        for key, value in table.items():
            want = valid[key]
            # TOML gives ints where a float is wanted (gamma = 2, cfl = 1).
            if want is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if want is float and not isinstance(value, float):
                raise ValueError(f"{found}: [{name}].{key} must be a number, got {value!r}")
            if want is bool and not isinstance(value, bool):
                raise ValueError(f"{found}: [{name}].{key} must be true or false, got {value!r}")
            if want is int and not isinstance(value, bool) and not isinstance(value, int):
                raise ValueError(f"{found}: [{name}].{key} must be an integer, got {value!r}")
            if want is str and not isinstance(value, str):
                raise ValueError(f"{found}: [{name}].{key} must be a string, got {value!r}")
            coerced[key] = value
            explicit.add(f"{name}.{key}")
        kwargs[name] = cls(**coerced)

    cfg = Config(**kwargs, source=str(found), explicit=frozenset(explicit))
    if cfg.physics.gamma <= 1.0:
        raise ValueError(f"{found}: [physics].gamma must exceed 1, got {cfg.physics.gamma}")
    return cfg


# ---------------------------------------------------------------------------
# Dataset provenance
# ---------------------------------------------------------------------------

def _first_snapshot(data_dir: str | Path) -> Path | None:
    files = sorted(glob.glob(os.path.join(str(data_dir), "h*.h5")))
    return Path(files[0]) if files else None


def read_dataset_metadata(data_dir: str | Path) -> dict[str, Any]:
    """Return the `Metadata` attributes of the first snapshot, or {} if absent."""
    import h5py

    snap = _first_snapshot(data_dir)
    if snap is None:
        return {}
    with h5py.File(snap, "r") as f:
        if "Metadata" not in f:
            return {}
        return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f["Metadata"].attrs.items()}


def resolve_gamma(
    cfg: Config,
    data_dir: str | Path | None = None,
    override: float | None = None,
    *,
    strict: bool = True,
) -> float:
    """Determine Γ for a run, preferring the dataset's own record over the config.

    Raises when the config contradicts dataset metadata, unless `override` is
    given or `strict=False`. Warns when the data carries no metadata at all,
    since then nothing has actually been verified.
    """
    if override is not None:
        return float(override)

    meta_gamma = None
    if data_dir is not None:
        meta = read_dataset_metadata(data_dir)
        raw = meta.get("eos_gamma")
        if raw is not None:
            meta_gamma = float(raw)

    if meta_gamma is None:
        if data_dir is not None:
            warnings.warn(
                f"{data_dir} carries no EOS metadata; falling back to "
                f"gamma={cfg.physics.gamma:.10g} from "
                f"{cfg.source or 'built-in defaults'}. Nothing has verified that this "
                f"matches the data — run `python -m src.config verify {data_dir}`.",
                RuntimeWarning,
                stacklevel=2,
            )
        return float(cfg.physics.gamma)

    if "physics.gamma" in cfg.explicit and not np.isclose(meta_gamma, cfg.physics.gamma, rtol=1e-9):
        msg = (
            f"EOS mismatch: {data_dir} records gamma={meta_gamma:.10g} but "
            f"{cfg.source} sets [physics].gamma={cfg.physics.gamma:.10g}. "
            f"The data is authoritative. Fix the config, or pass an explicit override "
            f"if you really mean to reinterpret the data."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return meta_gamma


# ---------------------------------------------------------------------------
# argparse integration
# ---------------------------------------------------------------------------

def add_config_arguments(parser) -> None:
    """Add `--config` and `--gamma` to an existing parser."""
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help=f"TOML config file (default: $CRMHD_CONFIG, ./{CONFIG_FILENAME}, "
             f"or repo-root {CONFIG_FILENAME})")
    parser.add_argument(
        "--gamma", type=float, default=None,
        help="override the adiabatic index, taking precedence over both the "
             "config file and the dataset's own metadata")


def apply_config_defaults(parser, mapping: dict[str, str], argv: list[str] | None = None) -> Config:
    """Seed argparse defaults from the config file.

    Two-stage: `--config` is read first, then the loaded values become the
    parser's defaults for each dest in `mapping` ("dest" -> "section.key").
    Because argparse lets an explicitly passed flag override its default, the
    resulting precedence is  command line > config file > built-in default.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)

    cfg = load_config(known.config)
    if cfg.source is None:
        return cfg
    defaults = {}
    for dest, path in mapping.items():
        section, key = path.split(".")
        if path in cfg.explicit:
            defaults[dest] = getattr(getattr(cfg, section), key)
    parser.set_defaults(**defaults)
    return cfg


def configure(args, cfg: Config | None = None, data_dir: str | None = None) -> Config:
    """Resolve Γ for this run, push it into the solver, and print what was used.

    Call immediately after `parse_args()` and before constructing a solver or
    reading data — `compute_dt` and `primitives_to_conserved` may be compiled,
    and a traced graph can bake Γ in.
    """
    cfg = cfg if cfg is not None else load_config(getattr(args, "config", None))
    d = data_dir if data_dir is not None else getattr(args, "data_dir", None)
    gamma = resolve_gamma(cfg, d, override=getattr(args, "gamma", None))
    cfg.apply(gamma)

    meta = read_dataset_metadata(d) if d else {}
    if getattr(args, "gamma", None) is not None:
        origin = "--gamma override"
    elif meta.get("eos_gamma") is not None:
        origin = f"{d} metadata"
    else:
        origin = cfg.source or "built-in defaults"
    print(f"[config] source={cfg.source or 'built-in defaults'}  "
          f"gamma={gamma:.10g} (from {origin})", flush=True)
    return cfg


# ---------------------------------------------------------------------------
# Numerical EOS verification
# ---------------------------------------------------------------------------

def _gamma_from_cs_newtonian(rho, p, cs):
    r"""c_s² = Γ p / ρ  →  Γ = c_s² ρ / p."""
    return cs**2 * rho / p


def _gamma_from_cs_relativistic(rho, p, cs):
    r"""Invert c_s² = Γ p / (ρ h), h = 1 + Γ/(Γ-1) p/ρ, for Γ.

    With a = p/ρ this is the quadratic a Γ² - (a + c_s² + c_s² a) Γ + c_s² = 0;
    the physical root is the one exceeding 1.
    """
    a = p / rho
    c = cs**2
    b = a + c + c * a
    disc = b**2 - 4.0 * a * c
    disc = np.where(disc < 0, np.nan, disc)
    root = (b + np.sqrt(disc)) / (2.0 * a)
    return root


def verify_dataset_eos(data_dir: str | Path, gamma: float | None = None) -> dict[str, Any]:
    r"""Establish, numerically, what equations a dataset actually satisfies.

    This is the diagnostic that identified `test1/` as Newtonian. It makes no
    assumption about the file's claims; it tests the stored fields against both
    the Newtonian and the special-relativistic closure relations and reports
    residuals for each. Run it on any dataset before building on it.

    Checks performed:
      1. |S/ρ| >= 1 anywhere — harmless in Newtonian data, impossible if S/ρ
         is meant to be a velocity in SRHD
      2. stored c_s >= 1 anywhere — impossible for a relativistic sound speed
      3. Γ inferred from c_s under each hypothesis, and how constant it is
      4. energy residual under the Newtonian relation E = p/(Γ-1) + ½ρv²
      5. energy residual under the SRHD relation E = ρhW² - p, using the
         primitives from a `prim` group when one is present

    Returns a dict of measurements plus a `verdict` string.
    """
    import h5py

    snap = _first_snapshot(data_dir)
    if snap is None:
        raise FileNotFoundError(f"no h*.h5 snapshots in {data_dir}")

    with h5py.File(snap, "r") as f:
        hydro = f["hydro"]["data"][0, GHOST:-GHOST, GHOST:-GHOST, :].astype(np.float64)
        thd = f["thd"]["data"][0, GHOST:-GHOST, GHOST:-GHOST, :].astype(np.float64)
        has_prim = "prim" in f
        prim = (
            f["prim"]["data"][0, GHOST:-GHOST, GHOST:-GHOST, :].astype(np.float64)
            if has_prim else None
        )
        meta = {k: (v.decode() if isinstance(v, bytes) else v)
                for k, v in (f["Metadata"].attrs.items() if "Metadata" in f else [])}

    out: dict[str, Any] = {
        "snapshot": str(snap),
        "shape": tuple(hydro.shape[:2]),
        "has_prim_group": has_prim,
        "metadata": meta,
    }

    p = thd[..., 0]
    cs = thd[..., 1] if thd.shape[-1] > 1 else None
    dens = hydro[..., 0]          # rho if Newtonian, D = rho W if relativistic
    Sx, Sy = hydro[..., 2], hydro[..., 3]
    E = hydro[..., 1]

    # Informational only. |S/dens| >= 1 rules nothing out by itself: read as
    # SRHD, S/D = h W v, which legitimately exceeds 1. It is decisive only if
    # the file claims dens is rho and S/dens is a velocity.
    ratio = np.sqrt((Sx / dens) ** 2 + (Sy / dens) ** 2)
    out["max_S_over_dens"] = float(ratio.max())
    out["n_cells_S_over_dens_ge_1"] = int((ratio >= 1.0).sum())

    if cs is not None:
        out["max_cs"] = float(cs.max())
        out["cs_exceeds_light_speed"] = bool(cs.max() >= 1.0)
        gN = _gamma_from_cs_newtonian(dens, p, cs)
        out["gamma_newtonian_from_cs"] = {
            "median": float(np.nanmedian(gN)), "std": float(np.nanstd(gN))}
        if has_prim:
            # Needs rho, not D, so it is only available when the primitives
            # are stored; otherwise rho itself depends on the gamma assumed.
            with np.errstate(invalid="ignore", divide="ignore"):
                gR = _gamma_from_cs_relativistic(prim[..., 0], prim[..., 3], cs)
            out["gamma_relativistic_from_cs"] = {
                "median": float(np.nanmedian(gR)), "std": float(np.nanstd(gR))}

    def _rel_resid(a, b):
        scale = max(float(np.abs(b).max()), 1e-300)
        return float(np.nanmedian(np.abs(a - b)) / scale)

    trials = [gamma] if gamma is not None else []
    trials += [g for g in (4.0 / 3.0, 5.0 / 3.0) if gamma is None or not np.isclose(g, gamma)]

    out["newtonian"] = {}
    out["relativistic"] = {}
    for g in trials:
        key = f"gamma={g:.6f}"

        # --- Newtonian reading: dens is rho, v = S/rho, E = p/(g-1) + rho v^2/2
        # Both residuals are independent: E and c_s are stored, not derived.
        n: dict[str, Any] = {"energy_residual": _rel_resid(
            p / (g - 1.0) + 0.5 * dens * ratio**2, E)}
        if cs is not None:
            n["cs_residual"] = _rel_resid(np.sqrt(g * p / dens), cs)
        out["newtonian"][key] = n

        # --- SRHD reading: dens is D = rho W, E = rho h W^2 - p ------------
        r: dict[str, Any] = {}
        if has_prim:
            rho_r, vx_r, vy_r, p_r = (prim[..., i] for i in range(4))
            r["source"] = "prim group"
        else:
            import torch
            from src.solver import conserved_to_primitives, set_gamma_eos
            set_gamma_eos(g)
            U = torch.from_numpy(np.stack([dens, Sx, Sy, E - dens], axis=-1))
            try:
                rho_r, vx_r, vy_r, p_r = (
                    t.numpy() for t in conserved_to_primitives(U))
                r["source"] = "C2P of hydro"
            except Exception as exc:
                out["relativistic"][key] = {"source": "C2P failed", "error": str(exc)}
                continue

        v2 = np.clip(vx_r**2 + vy_r**2, 0.0, 1.0 - 1e-14)
        W = 1.0 / np.sqrt(1.0 - v2)
        h = 1.0 + g / (g - 1.0) * p_r / rho_r

        if has_prim:
            r["energy_residual"] = _rel_resid(rho_r * h * W**2 - p_r, E)
        else:
            # C2P solves E = rho h W^2 - p for the primitives, so re-deriving E
            # from them is tautological and would read ~1e-16 on any data,
            # Newtonian included. Not evidence; c_s below is independent.
            r["energy_residual"] = None
        if cs is not None:
            r["cs_residual"] = _rel_resid(np.sqrt(g * p_r / (rho_r * h)), cs)
        r["max_v"] = float(np.sqrt(v2).max())
        out["relativistic"][key] = r

    # --- verdict ------------------------------------------------------------
    def _score(d: dict[str, Any]) -> float:
        """Worst of the independent residuals; inf if none are available."""
        vals = [v for k, v in d.items()
                if k.endswith("_residual") and isinstance(v, float) and not np.isnan(v)]
        return max(vals) if vals else float("inf")

    best_n = min(out["newtonian"].items(), key=lambda kv: _score(kv[1]))
    best_r = min(out["relativistic"].items(), key=lambda kv: _score(kv[1]))
    out["best_newtonian"] = (best_n[0], _score(best_n[1]))
    out["best_relativistic"] = (best_r[0], _score(best_r[1]))
    sn, sr = out["best_newtonian"][1], out["best_relativistic"][1]
    TOL = 1e-10

    if out.get("cs_exceeds_light_speed"):
        out["verdict"] = (
            f"NEWTONIAN - the stored sound speed reaches {out['max_cs']:.3f}, which no "
            f"relativistic sound speed can. Best fit {best_n[0]}, worst residual {sn:.2e}."
        )
    elif sn < TOL and sn < sr:
        out["verdict"] = (
            f"NEWTONIAN - all independent relations close at {best_n[0]} "
            f"(worst residual {sn:.2e}; best SRHD reading {sr:.2e})."
        )
    elif sr < TOL and sr < sn:
        out["verdict"] = (
            f"SRHD - all independent relations close at {best_r[0]} "
            f"(worst residual {sr:.2e}; best Newtonian reading {sn:.2e})."
        )
    else:
        out["verdict"] = (
            f"INCONCLUSIVE - best Newtonian {best_n[0]} worst residual {sn:.2e}, "
            f"best SRHD {best_r[0]} worst residual {sr:.2e}. Neither reading closes to "
            f"machine precision; check the variable layout before using this data."
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_verification(rep: dict[str, Any]) -> None:
    print(f"snapshot : {rep['snapshot']}  {rep['shape'][0]}x{rep['shape'][1]}")
    print(f"prim group present : {rep['has_prim_group']}")
    if rep["metadata"]:
        for k in ("scheme", "eos_gamma", "prim_variables", "hydro_variables"):
            if k in rep["metadata"]:
                print(f"  metadata {k:16s} = {rep['metadata'][k]}")
    else:
        print("  (no Metadata group - provenance is unrecorded)")
    print()
    print(f"max |S/dens|                 = {rep['max_S_over_dens']:.6f}"
          f"   ({rep['n_cells_S_over_dens_ge_1']} cells >= 1; informational only)")
    if "max_cs" in rep:
        flag = "  <-- impossible for SRHD" if rep["cs_exceeds_light_speed"] else ""
        print(f"max stored c_s               = {rep['max_cs']:.6f}{flag}")
        gn = rep["gamma_newtonian_from_cs"]
        print(f"gamma from c_s, Newtonian    = {gn['median']:.10f}  (std {gn['std']:.2e})")
        if "gamma_relativistic_from_cs" in rep:
            gr = rep["gamma_relativistic_from_cs"]
            print(f"gamma from c_s, relativistic = {gr['median']:.10f}  (std {gr['std']:.2e})")
        else:
            print("gamma from c_s, relativistic =  (needs rho; no prim group stored)")
    print()
    print("independent residuals (median, relative; lower is better):")
    print(f"  {'reading':<14}{'gamma':<18}{'energy':<14}{'sound speed':<14}")
    for label, table in (("Newtonian", rep["newtonian"]), ("SRHD", rep["relativistic"])):
        for key, d in table.items():
            if "error" in d:
                print(f"  {label:<14}{key:<18}{'C2P failed':<28}")
                continue
            e = d.get("energy_residual")
            e_s = "circular" if e is None else f"{e:.3e}"
            c = d.get("cs_residual")
            c_s = "-" if c is None else f"{c:.3e}"
            print(f"  {label:<14}{key:<18}{e_s:<14}{c_s:<14}")
    print("  (\"circular\": C2P inverts the energy relation, so re-deriving it proves")
    print("   nothing. The sound speed is independent data and does discriminate.)")
    print()
    print(f"VERDICT: {rep['verdict']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CRMHD-TBNN configuration and EOS provenance.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("show", help="print the resolved configuration")
    s.add_argument("--config", default=None)
    s.add_argument("--data-dir", default=None,
                   help="also resolve gamma against this dataset's metadata")

    v = sub.add_parser("verify", help="determine numerically which EOS a dataset satisfies")
    v.add_argument("data_dir")
    v.add_argument("--gamma", type=float, default=None, help="extra gamma to test")

    args = ap.parse_args(argv)

    if args.cmd == "show":
        cfg = load_config(args.config)
        print(f"config source: {cfg.source or 'built-in defaults'}")
        for section in _SECTIONS:
            print(f"[{section}]")
            for f in fields(getattr(cfg, section)):
                mark = "*" if f"{section}.{f.name}" in cfg.explicit else " "
                print(f" {mark} {f.name:20s} = {getattr(getattr(cfg, section), f.name)!r}")
        print("\n(* = set explicitly in the config file; others are defaults)")
        if args.data_dir:
            g = resolve_gamma(cfg, args.data_dir, strict=False)
            print(f"\nresolved gamma for {args.data_dir}: {g:.10g}")
        return 0

    _print_verification(verify_dataset_eos(args.data_dir, args.gamma))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
