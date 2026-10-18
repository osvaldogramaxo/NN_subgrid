r"""Cross-realisation version of the learned-coefficient ladder.

``scratch/minit_learned_coeff_feasibility.py`` splits one DNS run. That answers
"is e_KH a function of the local state", but a closure has to work on a flow it
has never seen. This trains the coefficient predictor on all of ``dns_khi`` and
scores it on every snapshot of ``dns_khi_s1`` (a different seed) -- the same
protocol ``src.evaluate_apriori`` uses for the TBNN, so the two numbers can be
put side by side.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.minit_learned_coeff_feasibility import (  # noqa: E402
    E_FLOOR, MLP, add_stencil, build_features, fit_mlp, r2, tau_scores,
)
from src.evaluate_apriori import highk_fraction  # noqa: E402
from src.minit_calibrate import fit_beta_lsq, score_tau  # noqa: E402


def apriori_style(e_pred, tau_ex, beta, S, stride=10, k_hi=16):
    """Aggregate exactly as ``src.evaluate_apriori`` does, so the rows here and
    the TBNN rows it prints can be put in one table: per-snapshot ratios first,
    then averaged, on the same stride-10 subsample. A global ratio-of-means is
    not the same number and would flatter whichever model is strongest where
    the stress is largest."""
    tau_m = e_pred.unsqueeze(-1) * beta
    amp, corr, pi, hik, hik_ex, sign = [], [], [], [], [], []
    for i in range(0, tau_ex.shape[0], stride):
        m, x, s_ = tau_m[i], tau_ex[i], S[i]
        tr = lambda z: (z[..., 0]*s_[..., 0] + z[..., 1]*s_[..., 1] + 2.0*z[..., 2]*s_[..., 2])
        pm, px = float(tr(m).mean()), float(tr(x).mean())
        a, b = m.flatten() - m.mean(), x.flatten() - x.mean()
        amp.append(float(m.norm() / x.norm().clamp(min=1e-300)))
        corr.append(float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-300)))
        pi.append(pm / px if px != 0 else float("nan"))
        sign.append(pm * px > 0)
        hik.append(highk_fraction(m[..., 0], k_hi))
        hik_ex.append(highk_fraction(x[..., 0], k_hi))
    n = len(amp)
    mean = lambda v: float(sum(v) / len(v))
    return {"ap_amp": mean(amp), "ap_corr": mean(corr), "ap_pi": mean(pi),
            "ap_pi_sign": mean([float(b) for b in sign]),
            "ap_hik": mean(hik), "ap_hik_exact": mean(hik_ex),
            "ap_amp_first": mean(amp[: n // 2]), "ap_amp_second": mean(amp[n // 2:])}

TRAIN_CACHE = Path("results/minit/dns_reduction.pt")
TEST_CACHE = Path("results/minit/dns_reduction_s1.pt")
OUT = Path("results/minit/learned_coeff_crossrealisation.json")
T_MIN = 1.0


def load(path: Path):
    d = torch.load(path, map_location="cpu", weights_only=False)["data"]
    keep = d["t"] >= T_MIN
    d = {k: (v[keep] if v.shape[0] == keep.shape[0] else v) for k, v in d.items()}
    Nx = d["rho"].shape[-1]
    lam = 15.8 * (1.0 / Nx)
    X, names = build_features(d, lam)
    return d, X, add_stencil(X), lam, names


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, Xtr_p, Xtr_s, lam, names = load(TRAIN_CACHE)
    te, Xte_p, Xte_s, lam_te, _ = load(TEST_CACHE)
    assert abs(lam - lam_te) < 1e-12, "filter scales differ between the two runs"

    e_tr, e_te = tr["e_dns"].clamp(min=0.0), te["e_dns"].clamp(min=0.0)
    tau_tr, tau_te = tr["tau"], te["tau"]
    S_te = te["S"]
    y_tr = (e_tr + E_FLOOR).log().reshape(-1)
    y_te = (e_te + E_FLOOR).log().reshape(-1)

    bfit = fit_beta_lsq(e_tr, tau_tr)
    beta = torch.tensor([bfit["beta_xx"], bfit["beta_yy"], bfit["beta_xy"]], dtype=tau_tr.dtype)

    rows = {"0_oracle": {"r2_log": 1.0, "r2_e": 1.0, **tau_scores(e_te, tau_te, beta, S_te),
                         **apriori_style(e_te, tau_te, beta, S_te)}}

    base_tr = tr["rho"] * lam**2 * tr["shear"] ** 2
    base_te = te["rho"] * lam**2 * te["shear"] ** 2
    k = float((base_tr * e_tr).sum() / base_tr.pow(2).sum().clamp(min=1e-300))
    e_eq = (k * base_te).clamp(min=0.0)
    rows["1_equilibrium_global_k"] = {
        "k": k, "r2_log": r2((e_eq + E_FLOOR).log().reshape(-1), y_te),
        "r2_e": r2(e_eq.reshape(-1), e_te.reshape(-1)), **tau_scores(e_eq, tau_te, beta, S_te),
        **apriori_style(e_eq, tau_te, beta, S_te),
    }

    for tag, A, B in (("4_mlp_pointwise", Xtr_p, Xte_p), ("5_mlp_stencil5", Xtr_s, Xte_s)):
        pred = fit_mlp(A.reshape(-1, A.shape[-1]), y_tr, B.reshape(-1, B.shape[-1]),
                       device=device)
        e_ml = (pred.exp() - E_FLOOR).clamp(min=0.0).reshape(e_te.shape)
        rows[tag] = {
            "n_params": sum(p.numel() for p in MLP(A.shape[-1]).parameters()),
            "n_features": int(A.shape[-1]),
            "r2_log": r2(pred, y_te), "r2_e": r2(e_ml.reshape(-1), e_te.reshape(-1)),
            **tau_scores(e_ml, tau_te, beta, S_te),
            **apriori_style(e_ml, tau_te, beta, S_te),
        }
        # amplitude against time, so a late collapse cannot hide in the mean
        per_t = []
        for i in range(e_te.shape[0]):
            m = (e_ml[i].unsqueeze(-1) * beta)
            per_t.append(float(m.norm() / tau_te[i].norm().clamp(min=1e-300)))
        rows[tag]["amp_first_half"] = float(sum(per_t[: len(per_t) // 2]) / (len(per_t) // 2))
        rows[tag]["amp_second_half"] = float(sum(per_t[len(per_t) // 2:]) / (len(per_t) - len(per_t) // 2))

    print(f"train: {TRAIN_CACHE.name} ({e_tr.shape[0]} snaps)  "
          f"test: {TEST_CACHE.name} ({e_te.shape[0]} snaps, unseen seed)")
    print(f"{'rung':<30} {'R2(log e)':>10} {'R2(e)':>8} {'corr(tau)':>10} {'relL2':>8} {'amp':>6} {'Pi/Pi_ex':>9}")
    for name, r in rows.items():
        print(f"{name:<30} {r['r2_log']:>10.3f} {r['r2_e']:>8.3f} "
              f"{r['corr']:>10.3f} {r['rel_l2']:>8.3f} {r['amplitude']:>6.3f} "
              f"{r['pi_ratio']:>9.3f}")
    print("\n--- aggregated exactly as src.evaluate_apriori (stride 10, "
          "per-snapshot then mean) so these rows sit beside its TBNN rows ---")
    print(f"{'rung':<30} {'amp':>6} {'corr':>7} {'Pi/Pi_ex':>9} {'sign':>6} "
          f"{'hi-k':>6} {'hi-k ex':>8} {'amp 1st->2nd half':>20}")
    for name, r in rows.items():
        print(f"{name:<30} {r['ap_amp']:>6.3f} {r['ap_corr']:>+7.3f} {r['ap_pi']:>9.3f} "
              f"{r['ap_pi_sign']*100:>5.0f}% {r['ap_hik']*100:>5.1f}% {r['ap_hik_exact']*100:>7.1f}% "
              f"{r['ap_amp_first']:>9.3f} -> {r['ap_amp_second']:.3f}")
    for name in ("4_mlp_pointwise", "5_mlp_stencil5"):
        print(f"  {name}: {rows[name]['n_params']} parameters")

    OUT.write_text(json.dumps({"beta": bfit, "lambda_KH": lam, "features": names,
                               "rungs": rows}, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
