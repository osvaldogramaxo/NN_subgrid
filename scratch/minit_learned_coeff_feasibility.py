r"""Is a state-dependent MInIT coefficient learnable, and is it worth it?

The a-priori decomposition in MINIT.md says the proportionality ansatz
tau^ij = beta_ij e_KH is sound (corr 0.928 with the DNS e_KH) and that the
*evolution equation* for e_KH is what fails (0.928 -> 0.495 in one dt).

So the question for a learned-coefficient MInIT is not "can a network fit tau"
-- that is the TBNN's job -- but:

    how much of the DNS e_KH field is determined by the *instantaneous local
    resolved state*?

Everything a pointwise coefficient network can ever recover is bounded by that.
This script measures it against a ladder of predictors of increasing freedom,
all scored on the same held-out snapshots:

  0  oracle        e = e_DNS                       (the ceiling, corr 0.928)
  1  equilibrium   e = k rho lambda^2 shear^2      (MInIT equilibrium, global k)
  2  per-snapshot  same, k = k(t) refit each frame (a Germano-style dynamic k)
  3  linear        e = w . features                (state-dependent, no network)
  4  mlp           e = MLP(features)               (state-dependent, pointwise)
  5  mlp+stencil   + 5x5 neighbourhood statistics  (state-dependent, non-local)

Rungs 3-5 are "MInIT with context-dependent coefficients". Rung 2 says whether
a network is needed at all or whether a dynamic procedure would do. The gap
between 4 and 5 says whether the missing information is local.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.minit_calibrate import energy_transfer, fit_beta_lsq, score_tau  # noqa: E402
from src.features import periodic_gradient_2d  # noqa: E402

CACHE = Path("results/minit/dns_reduction.pt")
OUT = Path("results/minit/learned_coeff_feasibility.json")
E_FLOOR = 1e-8          # small vs the median e_KH of 7.5e-4
T_MIN = 1.0             # t=0 has e_KH ~ 1e-15; it is not a turbulence sample


def build_features(d: dict, lam: float) -> tuple[torch.Tensor, list[str]]:
    """Local, instantaneous, resolved-state features. (T, Ny, Nx, F)."""
    rho, vx, vy = d["rho"], d["vx"], d["vy"]
    Ny, Nx = rho.shape[-2:]
    dx, dy = 1.0 / Nx, 1.0 / Ny

    dvx_dx, dvx_dy = periodic_gradient_2d(vx, dx, dy, order=4)
    dvy_dx, dvy_dy = periodic_gradient_2d(vy, dx, dy, order=4)
    drho_dx, drho_dy = periodic_gradient_2d(rho, dx, dy, order=4)

    vort = (dvy_dx - dvx_dy).abs()
    v2 = (vx**2 + vy**2).clamp(max=1.0 - 1e-12)
    W = (1.0 - v2).rsqrt()
    grad_rho = (drho_dx**2 + drho_dy**2).sqrt() / rho.clamp(min=1e-12)

    strain, shear, div_v = d["strain"], d["shear"], d["div_v"]
    # The Smagorinsky combination itself, handed to the model so that rungs 3-5
    # can only do better than rung 1, never worse for want of the right variable.
    smag_strain = rho * lam**2 * strain**2
    smag_shear = rho * lam**2 * shear**2

    feats = {
        "log_rho": rho.clamp(min=1e-12).log(),
        "strain": strain,
        "shear": shear,
        "abs_div_v": div_v.abs(),
        "vorticity": vort,
        "speed": v2.sqrt(),
        "lorentz_W": W,
        "grad_rho_over_rho": grad_rho * dx,
        "log_smag_strain": (smag_strain + E_FLOOR).log(),
        "log_smag_shear": (smag_shear + E_FLOOR).log(),
    }
    names = list(feats)
    return torch.stack([feats[k] for k in names], dim=-1), names


def add_stencil(X: torch.Tensor) -> torch.Tensor:
    """Append 5x5 periodic neighbourhood mean and std of every feature."""
    T, Ny, Nx, Fdim = X.shape
    z = X.permute(0, 3, 1, 2).reshape(T * Fdim, 1, Ny, Nx)
    z = F.pad(z, (2, 2, 2, 2), mode="circular")
    k = torch.ones(1, 1, 5, 5, dtype=z.dtype) / 25.0
    mean = F.conv2d(z, k)
    sq = F.conv2d(z**2, k)
    std = (sq - mean**2).clamp(min=0.0).sqrt()
    back = lambda a: a.reshape(T, Fdim, Ny, Nx).permute(0, 2, 3, 1)
    return torch.cat([X, back(mean), back(std)], dim=-1)


def r2(pred: torch.Tensor, y: torch.Tensor) -> float:
    ss_res = (pred - y).pow(2).sum()
    ss_tot = (y - y.mean()).pow(2).sum()
    return float(1.0 - ss_res / ss_tot.clamp(min=1e-300))


def tau_scores(e_pred: torch.Tensor, tau: torch.Tensor, beta: torch.Tensor,
               S: torch.Tensor | None = None) -> dict:
    """Score beta_ij * e_pred against the exact stress, as MINIT.md does.

    With ``S`` also the energy transfer Pi = tau^ij S_ij, which is what governs
    whether a closure can move a trajectory at all: MINIT.md records the global
    coefficients supplying only 2% of the exact transfer (and the wrong sign for
    the `fitted` variant), so a ladder that improved corr(tau) while leaving Pi
    at 2% would not be worth building.
    """
    tau_m = e_pred.unsqueeze(-1) * beta
    out = score_tau(tau_m, tau)
    if S is not None:
        Pi_m, Pi_x = energy_transfer(tau_m, S), energy_transfer(tau, S)
        out["pi_ratio"] = float(Pi_m.mean() / Pi_x.mean())
        out["pi_sign_agree"] = float((Pi_m.sign() == Pi_x.sign()).double().mean())
        out["amplitude"] = float(tau_m.norm() / tau.norm())
    return out


class MLP(torch.nn.Module):
    def __init__(self, n_in: int, width: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(),
            torch.nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(Xtr, ytr, Xte, seed=0, epochs=300, device="cpu"):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-12)
    Xtr_n, Xte_n = (Xtr - mu) / sd, (Xte - mu) / sd
    ym, ys = ytr.mean(), ytr.std().clamp(min=1e-12)

    model = MLP(Xtr.shape[-1]).to(device=device, dtype=Xtr.dtype)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xtr_n, ytr_n = Xtr_n.to(device), ((ytr - ym) / ys).to(device)

    n, bs = Xtr_n.shape[0], 65536
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.mse_loss(model(Xtr_n[idx]), ytr_n[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        out = []
        Xte_n = Xte_n.to(device)
        for i in range(0, Xte_n.shape[0], bs):
            out.append(model(Xte_n[i:i + bs]).cpu())
    return torch.cat(out) * ys + ym


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(CACHE, map_location="cpu")["data"]
    t = d["t"]
    keep = t >= T_MIN
    d = {k: (v[keep] if v.shape[0] == t.shape[0] else v) for k, v in d.items()}
    t = d["t"]

    Ny, Nx = d["rho"].shape[-2:]
    lam = 15.8 * (1.0 / Nx)          # filter scale: the coarse cell is the filter box
    e = d["e_dns"].clamp(min=0.0)
    tau = d["tau"]

    X, names = build_features(d, lam)
    Xs = add_stencil(X)
    y = (e + E_FLOOR).log()

    results = {"n_snapshots": int(t.numel()), "t_min": float(t[0]), "t_max": float(t[-1]),
               "lambda_KH": lam, "features": names, "splits": {}}

    splits = {
        # Interleaved: "is e a function of the local state at all?"
        "interleaved": (torch.arange(t.numel()) % 2 == 0, torch.arange(t.numel()) % 2 == 1),
        # Forward in time: "does that function transfer to a regime never seen?"
        "forward_time": (t <= 10.0, t > 10.0),
    }

    for split_name, (tr_m, te_m) in splits.items():
        e_tr, e_te = e[tr_m], e[te_m]
        tau_tr, tau_te = tau[tr_m], tau[te_m]
        S_te = d["S"][te_m]
        y_tr, y_te = y[tr_m].reshape(-1), y[te_m].reshape(-1)

        # beta from the TRAIN half only, so no test information leaks in.
        bfit = fit_beta_lsq(e_tr, tau_tr)
        beta = torch.tensor([bfit["beta_xx"], bfit["beta_yy"], bfit["beta_xy"]],
                            dtype=tau.dtype)

        rows = {}

        # --- rung 0: oracle -------------------------------------------------
        rows["0_oracle"] = {"r2_log": 1.0, "r2_e": 1.0, **tau_scores(e_te, tau_te, beta, S_te)}

        # --- rung 1: MInIT equilibrium, one global k ------------------------
        base_tr = (d["rho"][tr_m] * lam**2 * d["shear"][tr_m] ** 2)
        base_te = (d["rho"][te_m] * lam**2 * d["shear"][te_m] ** 2)
        k = float((base_tr * e_tr).sum() / base_tr.pow(2).sum().clamp(min=1e-300))
        e_eq = (k * base_te).clamp(min=0.0)
        rows["1_equilibrium_global_k"] = {
            "k": k,
            "r2_log": r2((e_eq + E_FLOOR).log().reshape(-1), y_te),
            "r2_e": r2(e_eq.reshape(-1), e_te.reshape(-1)),
            **tau_scores(e_eq, tau_te, beta, S_te),
        }

        # --- rung 2: per-snapshot k (a dynamic procedure, no network) -------
        num = (base_te * e_te).flatten(1).sum(-1)
        den = base_te.pow(2).flatten(1).sum(-1).clamp(min=1e-300)
        k_t = (num / den).reshape(-1, 1, 1)
        e_dyn = (k_t * base_te).clamp(min=0.0)
        rows["2_equilibrium_per_snapshot_k"] = {
            "k_min": float(k_t.min()), "k_max": float(k_t.max()),
            "r2_log": r2((e_dyn + E_FLOOR).log().reshape(-1), y_te),
            "r2_e": r2(e_dyn.reshape(-1), e_te.reshape(-1)),
            **tau_scores(e_dyn, tau_te, beta, S_te),
        }

        # --- rung 3: linear in the features --------------------------------
        Xtr = X[tr_m].reshape(-1, X.shape[-1])
        Xte = X[te_m].reshape(-1, X.shape[-1])
        A = torch.cat([Xtr, torch.ones_like(Xtr[:, :1])], dim=-1)
        w = torch.linalg.lstsq(A, y_tr.unsqueeze(-1)).solution
        pred = (torch.cat([Xte, torch.ones_like(Xte[:, :1])], dim=-1) @ w).squeeze(-1)
        e_lin = (pred.exp() - E_FLOOR).clamp(min=0.0).reshape(e_te.shape)
        rows["3_linear"] = {
            "r2_log": r2(pred, y_te), "r2_e": r2(e_lin.reshape(-1), e_te.reshape(-1)),
            **tau_scores(e_lin, tau_te, beta, S_te),
        }

        # --- rungs 4 and 5: pointwise MLP, then a 5x5 stencil --------------
        for tag, src in (("4_mlp_pointwise", X), ("5_mlp_stencil5", Xs)):
            Xtr_ = src[tr_m].reshape(-1, src.shape[-1])
            Xte_ = src[te_m].reshape(-1, src.shape[-1])
            pred = fit_mlp(Xtr_, y_tr, Xte_, device=device)
            e_ml = (pred.exp() - E_FLOOR).clamp(min=0.0).reshape(e_te.shape)
            rows[tag] = {
                "n_features": int(src.shape[-1]),
                "r2_log": r2(pred, y_te), "r2_e": r2(e_ml.reshape(-1), e_te.reshape(-1)),
                **tau_scores(e_ml, tau_te, beta, S_te),
            }

        results["splits"][split_name] = {
            "beta": bfit, "n_train_snap": int(tr_m.sum()), "n_test_snap": int(te_m.sum()),
            "rungs": rows,
        }

        print(f"\n=== split: {split_name}  "
              f"({int(tr_m.sum())} train / {int(te_m.sum())} test snapshots)")
        print(f"{'rung':<30} {'R2(log e)':>10} {'R2(e)':>8} {'corr(tau)':>10} "
              f"{'relL2':>8} {'amp':>6} {'Pi/Pi_ex':>9} {'Pi sign':>8}")
        for name, r in rows.items():
            print(f"{name:<30} {r['r2_log']:>10.3f} {r['r2_e']:>8.3f} "
                  f"{r['corr']:>10.3f} {r['rel_l2']:>8.3f} {r['amplitude']:>6.3f} "
                  f"{r['pi_ratio']:>9.3f} {r['pi_sign_agree']*100:>7.1f}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
