r"""train_minit_coeff.py

Fit the state-dependent MInIT coefficient network of :mod:`src.minit_learned`.

    python -m src.train_minit_coeff --output-mode residual
    python -m src.train_minit_coeff --output-mode direct --tag direct

Protocol
--------
Train on ``dns_khi`` with a small interleaved validation split for early
stopping, then score on ``dns_khi_s1`` — a second DNS from a different seed that
training never touches. Once the training window covers a whole run there is no
later time to hold out, and a held-out band of the *same* realisation sits
adjacent to training data in a strongly non-stationary flow; the honest test is
the independent realisation (this is the same argument as
``src/evaluate_apriori.py``).

Scoring here deliberately mirrors ``src.evaluate_apriori``: per-snapshot ratios
first, then averaged, on the same stride. A global ratio-of-means is a different
number and flatters whichever model is strongest where the stress is largest, so
the two would not be comparable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.evaluate_apriori import highk_fraction
from src.minit_calibrate import fit_beta_lsq, l2_norm_paper
from src.minit_learned import (
    E_FLOOR,
    CoefficientNet,
    LearnedMInITConfig,
    compute_invariants,
    predict_e_kh,
)


def load_reduction(path: Path, t_min: float) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)["data"]
    keep = d["t"] >= t_min
    return {k: (v[keep] if v.shape[0] == keep.shape[0] else v) for k, v in d.items()}


def prepare(d: dict, lambda_KH: float, dx: float, dy: float):
    """Invariants for every snapshot, through the shared feature path."""
    feats, base = compute_invariants(d["rho"], d["vx"], d["vy"], dx, dy, lambda_KH)
    e = d["e_dns"].clamp(min=0.0)
    return feats, base, e


def score(net, cfg, d, dx, dy, beta, stride=10, k_hi=16) -> dict:
    with torch.no_grad():
        e_pred = predict_e_kh(net, cfg, d["rho"], d["vx"], d["vy"], dx, dy)
    return score_e(e_pred, d, beta, stride, k_hi)


def score_e(e_pred, d, beta, stride=10, k_hi=16) -> dict:
    """Score ``tau = beta_ij e_pred`` against the exact stress.

    ``src.evaluate_apriori``-style aggregation, so rows line up with its output.
    ``l2_eq25`` is the paper's Eq. (25), summed over space and components per
    snapshot, then RMS over time as the paper does for Fig. 13.
    """
    return score_tau(e_pred.unsqueeze(-1) * beta, d, stride, k_hi)


def score_tau(tau_m, d, stride=10, k_hi=16) -> dict:
    """As :func:`score_e`, for any predicted ``[tau^xx, tau^yy, tau^xy]`` series."""
    tau_ex, S = d["tau"], d["S"]
    e_pred = 0.5 * (tau_m[..., 0] + tau_m[..., 1])

    amp, corr, pi, sign, hik, hik_ex, rl2, eq25 = [], [], [], [], [], [], [], []
    for i in range(0, tau_ex.shape[0], stride):
        m, x, s_ = tau_m[i], tau_ex[i], S[i]
        tr = lambda z: (z[..., 0] * s_[..., 0] + z[..., 1] * s_[..., 1]
                        + 2.0 * z[..., 2] * s_[..., 2])
        pm, px = float(tr(m).mean()), float(tr(x).mean())
        a, b = m.flatten() - m.mean(), x.flatten() - x.mean()
        nx = x.norm().clamp(min=1e-300)
        amp.append(float(m.norm() / nx))
        rl2.append(float((m - x).norm() / nx))
        eq25.append(float(l2_norm_paper(x.flatten(), m.flatten())))
        corr.append(float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-300)))
        pi.append(pm / px if px != 0 else float("nan"))
        sign.append(float(pm * px > 0))
        hik.append(highk_fraction(m[..., 0], k_hi))
        hik_ex.append(highk_fraction(x[..., 0], k_hi))

    n = len(amp)
    mean = lambda v: float(sum(v) / len(v))
    ss_res = float(((e_pred - d["e_dns"].clamp(min=0.0)) ** 2).sum())
    y = d["e_dns"].clamp(min=0.0)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "amp": mean(amp), "corr": mean(corr), "rel_l2": mean(rl2),
        "l2_eq25": mean([v * v for v in eq25]) ** 0.5,
        "pi": mean(pi), "pi_sign": mean(sign),
        "hik": mean(hik), "hik_exact": mean(hik_ex),
        "amp_first": mean(amp[: n // 2]), "amp_second": mean(amp[n // 2:]),
        "r2_e": 1.0 - ss_res / max(ss_tot, 1e-300),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--train-cache", default="results/minit/dns_reduction.pt")
    ap.add_argument("--test-cache", default="results/minit/dns_reduction_s1.pt")
    ap.add_argument("--output-mode", choices=["residual", "direct"], default="residual")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--eq25-weight", type=float, default=0.0,
                    help="add w * mean_snapshots(Eq.25^2) on the stress to the log-MSE "
                         "loss. log-MSE weights every cell equally; Eq. 25 is dominated "
                         "by the shear layers and punishes collapse, which is what the "
                         "evaluation measures. 0 = log-MSE only. Measured at r=16 "
                         "(FEASIBILITY §14): better a-priori, neutral on the short-horizon "
                         "gate, WORSE in the t=1->10 rollout (high-k 0.234 vs 0.292). "
                         "Keep 0 unless re-measured.")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--snapshots-per-batch", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=4,
                    help="every Nth snapshot is held out of training for early stopping")
    ap.add_argument("--t-min", type=float, default=1.0,
                    help="t=0 has e_KH ~ 1e-15 and is not a turbulence sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="results/minit")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    tag = args.tag or args.output_mode

    tr = load_reduction(Path(args.train_cache), args.t_min)
    Ny, Nx = tr["rho"].shape[-2:]
    dx, dy = 1.0 / Nx, 1.0 / Ny
    lambda_KH = 15.8 * dx          # the coarse cell is the filter box

    # beta from training data only -- no test information reaches the model.
    e_tr_all = tr["e_dns"].clamp(min=0.0)
    bfit = fit_beta_lsq(e_tr_all, tr["tau"])
    beta = torch.tensor([bfit["beta_xx"], bfit["beta_yy"], bfit["beta_xy"]],
                        dtype=tr["tau"].dtype)
    print(f"beta (lsq, train only): {  {k: round(v, 4) for k, v in bfit.items()} }")

    feats, base, e = prepare(tr, lambda_KH, dx, dy)
    y = (e + E_FLOOR).log()
    if args.output_mode == "residual":
        target = y - (base + E_FLOOR).log()   # the network learns the correction
    else:
        target = y

    T = feats.shape[0]
    idx = torch.arange(T)
    val_m = (idx % args.val_every) == 0
    tr_m = ~val_m
    print(f"{int(tr_m.sum())} train / {int(val_m.sum())} val snapshots "
          f"(t = {float(tr['t'][0]):.1f}..{float(tr['t'][-1]):.1f}), "
          f"lambda_KH = {lambda_KH:.5f}")

    net = CoefficientNet(width=args.width).to(dtype=feats.dtype)
    flat = feats[tr_m].reshape(-1, feats.shape[-1])
    net.set_normalization(flat.mean(0), flat.std(0))
    tgt_m, tgt_s = target[tr_m].mean(), target[tr_m].std().clamp(min=1e-12)
    net = net.to(dev)

    X_tr, Y_tr = feats[tr_m].to(dev), ((target[tr_m] - tgt_m) / tgt_s).to(dev)
    X_va, Y_va = feats[val_m].to(dev), ((target[val_m] - tgt_m) / tgt_s).to(dev)
    logb = (base + E_FLOOR).log() if args.output_mode == "residual" else torch.zeros_like(base)
    B_tr, B_va = logb[tr_m].to(dev), logb[val_m].to(dev)
    T_tr, T_va = tr["tau"][tr_m].to(dev), tr["tau"][val_m].to(dev)
    beta_d = beta.to(dev)

    def loss_fn(g_norm, y_norm, logbase, tau_ex):
        loss = F.mse_loss(g_norm, y_norm)
        if args.eq25_weight > 0:
            e = ((g_norm * tgt_s + tgt_m + logbase).exp() - E_FLOOR).clamp(min=0.0)
            tau_m = (e.unsqueeze(-1) * beta_d).flatten(1)
            loss = loss + args.eq25_weight * l2_norm_paper(tau_ex.flatten(1), tau_m).pow(2).mean()
        return loss

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best, best_state, best_ep, since = float("inf"), None, -1, 0
    n = X_tr.shape[0]
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, args.snapshots_per_batch):
            j = perm[i:i + args.snapshots_per_batch]
            loss = loss_fn(net(X_tr[j]), Y_tr[j], B_tr[j], T_tr[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(j)
        sched.step()

        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(X_va), Y_va, B_va, T_va))
        if vl < best - 1e-6:
            best, best_ep, since = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            since += 1
        if ep % 25 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:4d}/{args.epochs}  train={tot / n:.4e}  val={vl:.4e}  "
                  f"best={best:.4e}@{best_ep}  ({time.time() - t0:.0f}s)")
        if since >= args.patience:
            print(f"early stop at epoch {ep} (best {best:.4e} at {best_ep})")
            break

    net.load_state_dict(best_state)
    net = net.to("cpu").double().eval()

    cfg = LearnedMInITConfig(
        lambda_KH=lambda_KH,
        beta_xx=bfit["beta_xx"], beta_yy=bfit["beta_yy"], beta_xy=bfit["beta_xy"],
        output_mode=args.output_mode, width=args.width,
        log_e_mean=float(tgt_m), log_e_std=float(tgt_s),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"minit_coeff_{tag}.pt"
    torch.save({"state_dict": net.state_dict(), "config": vars(cfg),
                "best_val": best, "best_epoch": best_ep}, ckpt)

    results = {"config": vars(cfg), "beta": bfit, "best_val": best,
               "best_epoch": best_ep, "n_params": sum(p.numel() for p in net.parameters())}

    print(f"\nparameters: {results['n_params']}")
    k_hi = Nx // 4      # same relative band at every r (16 at r=16)

    # Reference rows on the unseen run: the ansatz ceiling (exact e_KH) and
    # MInIT's equilibrium limit with one global coefficient fitted on train.
    base_tr = tr["rho"] * lambda_KH**2 * tr["shear"] ** 2
    k_eq = float((base_tr * e_tr_all).sum() / base_tr.pow(2).sum().clamp(min=1e-300))

    hdr = (f"{'set':<30} {'amp':>6} {'corr':>7} {'Eq25':>7} {'Pi/Pi_ex':>9} "
           f"{'sign':>6} {'hi-k':>6} {'hi-k ex':>8} {'R2(e)':>7}  amp 1st->2nd")
    print(hdr)
    for name, cache in (("dns_khi (train)", args.train_cache),
                        ("dns_khi_s1 (unseen)", args.test_cache)):
        if not Path(cache).exists():
            print(f"{name:<30} -- cache {cache} missing, skipped")
            continue
        d = load_reduction(Path(cache), args.t_min)
        rows = {name: score(net, cfg, d, dx, dy, beta, k_hi=k_hi)}
        if "unseen" in name:
            e_eq = k_eq * d["rho"] * lambda_KH**2 * d["shear"] ** 2
            rows["  oracle e (ansatz ceiling)"] = score_e(
                d["e_dns"].clamp(min=0.0), d, beta, k_hi=k_hi)
            rows["  MInIT equilibrium, global k"] = score_e(e_eq, d, beta, k_hi=k_hi)
        for label, sc in rows.items():
            results[label.strip()] = sc
            print(f"{label:<30} {sc['amp']:>6.3f} {sc['corr']:>+7.3f} {sc['l2_eq25']:>7.3f} "
                  f"{sc['pi']:>9.3f} {sc['pi_sign']*100:>5.0f}% {sc['hik']*100:>5.1f}% "
                  f"{sc['hik_exact']*100:>7.1f}% {sc['r2_e']:>7.3f}  "
                  f"{sc['amp_first']:.3f} -> {sc['amp_second']:.3f}")

    (out_dir / f"minit_coeff_{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {ckpt} and {out_dir / f'minit_coeff_{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
