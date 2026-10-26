"""Fair a-priori comparison of MInIT (cell-by-cell and the paper's mean-field form) with MiNNit.

Every model is calibrated or trained on the training realisation over t = 1-20 and scored
here on unseen realisations, against two targets:

  cell  : the exact sub-grid stress cell by cell -- what a closure in a solver must supply;
  x-avg : the exact stress averaged over x, the homogeneous direction -- the mean-field
          quantity the paper evaluates. Every model's stress is x-averaged before scoring,
          so the comparison stays like for like.

MInIT gets the exact e_KH at t = 1 as its seed (as in the paper); MiNNit gets no seed.
Rows with advection appear where results/minit_fair/coeffs_<tag>_cell_advect.json exists.
Usage: python scratch/minit_fair_compare.py   (needs results/minit_fair/coeffs_*.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.minit import (  # noqa: E402
    LAMBDA_OVER_DELTA, MInITCoefficients, growth_rate, row_mean, saturation_energy,
)
from src.minit_calibrate import evolve_e_series  # noqa: E402
from src.minit_learned import CoefficientNet, LearnedMInITConfig, predict_e_kh  # noqa: E402
from src.train_minit_coeff import load_reduction, score_tau  # noqa: E402

SETTINGS = [
    ("1024², r = 16", ["results/minit/dns_reduction_s1.pt"],
     "results/minit/minit_coeff_residual.pt", "r16"),
    ("1024², r = 64", ["results/minit/dns_reduction_r64_s1.pt"],
     "results/filter_scan/minit_coeff_r64.pt", "r64"),
    ("2048², r = 128", [f"results/dns2k/reduction_r128_s{s}.pt" for s in range(1, 6)],
     "results/dns2k/minit_coeff_2k_r128.pt", "2k_r128"),
]


def load_minnit(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = LearnedMInITConfig(**blob["config"])
    net = CoefficientNet(width=cfg.width)
    net.load_state_dict(blob["state_dict"])
    return net.double().eval(), cfg


def beta_of(c):
    return torch.tensor([c.beta_xx, c.beta_yy, c.beta_xy], dtype=torch.float64)


def xavg_tau(tau):
    return tau.mean(dim=-2, keepdim=True).expand_as(tau)


def models_for(d, tag, net, cfg):
    """Every model's (T, Ny, Nx, 3) stress series on reduction ``d``."""
    Ny, Nx = d["rho"].shape[-2:]
    dx, dy = 1.0 / Nx, 1.0 / Ny
    lam = LAMBDA_OVER_DELTA * dx
    t = d["t"]
    out = {}
    with torch.no_grad():
        e = predict_e_kh(net, cfg, d["rho"], d["vx"], d["vy"], dx, dy)
    out["MiNNit (no seed)"] = e.unsqueeze(-1) * beta_of(cfg)
    for avg in ("cell", "x"):
        if avg == "cell":
            shear, rho, e_seed = d["shear"], d["rho"], d["e_dns"].clamp(min=0.0)
        else:
            shear = growth_rate(row_mean(d["vx"]), row_mean(d["vy"]), dx, dy, 1.0)
            rho, e_seed = row_mean(d["rho"]), row_mean(d["e_dns"].clamp(min=0.0))
        fitted = MInITCoefficients.load(f"results/minit_fair/coeffs_{tag}_{avg}.json")
        eq = MInITCoefficients.load(f"results/minit_fair/coeffs_{tag}_{avg}_equilibrium.json")
        for name, c in (("paper", MInITCoefficients.paper()), ("fitted", fitted)):
            e = evolve_e_series(e_seed[0], shear, rho, t, torch.tensor([c.A_gamma], dtype=torch.float64),
                                torch.tensor([c.C], dtype=torch.float64), lam, 10)[0]
            out[f"MInIT {avg}, {name} ODE (seeded)"] = e.unsqueeze(-1) * beta_of(c)
        e = saturation_energy(eq.A_gamma * shear, rho, lam, eq.C)
        out[f"MInIT {avg}, equilibrium"] = e.unsqueeze(-1) * beta_of(eq)
    # The cell form with the advective term the paper drops (FEASIBILITY §29), with the
    # coefficients fitted without it and refitted with it (minit_calibrate --advect).
    adv = Path(f"results/minit_fair/coeffs_{tag}_cell_advect.json")
    if adv.exists():
        for name, c in (("fitted", MInITCoefficients.load(f"results/minit_fair/coeffs_{tag}_cell.json")),
                        ("refitted", MInITCoefficients.load(adv))):
            e = evolve_e_series(d["e_dns"].clamp(min=0.0)[0], d["shear"], d["rho"], t,
                                torch.tensor([c.A_gamma], dtype=torch.float64),
                                torch.tensor([c.C], dtype=torch.float64), lam, 10, (d["vx"], d["vy"]))[0]
            out[f"MInIT cell + advection, {name} ODE (seeded)"] = e.unsqueeze(-1) * beta_of(c)
    return out


def main():
    results = {}
    for label, caches, ckpt, tag in SETTINGS:
        net, cfg = load_minnit(ckpt)
        acc: dict[str, list] = {}
        for path in caches:
            d = load_reduction(Path(path), 1.0)
            k_hi = d["rho"].shape[-1] // 4
            d_cell = {"tau": d["tau"], "S": d["S"], "e_dns": d["e_dns"]}
            d_x = {"tau": xavg_tau(d["tau"]), "S": d["S"], "e_dns": row_mean(d["e_dns"])}
            for name, tau in models_for(d, tag, net, cfg).items():
                sc = score_tau(tau, d_cell, stride=10, k_hi=k_hi)
                sx = score_tau(xavg_tau(tau), d_x, stride=10, k_hi=k_hi)
                acc.setdefault(name, []).append(
                    (sc["l2_eq25"], sc["corr"], sc["amp"], sx["l2_eq25"], sx["corr"]))
        results[label] = {k: np.mean(np.array(v), axis=0).tolist() for k, v in acc.items()}
        print(f"\n=== {label}: {len(caches)} unseen realisation(s), t = 1-20 ===")
        print(f"  {'model':<34} {'cell: Eq25':>11} {'corr':>7} {'amp':>6}   {'x-avg: Eq25':>12} {'corr':>7}")
        for name, v in results[label].items():
            print(f"  {name:<34} {v[0]:>11.3f} {v[1]:>+7.3f} {v[2]:>6.3f}   {v[3]:>12.3f} {v[4]:>+7.3f}")
    Path("results/minit_fair/compare.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
