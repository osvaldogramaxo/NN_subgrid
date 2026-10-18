r"""
evaluate_apriori.py

Score a trained TBNN checkpoint against the exact SGS stress of *any* DNS
directory, including one it never saw.

Why this exists
---------------
``src.train_apriori`` holds out a contiguous time window of its own dataset for
the final test. That is the right thing while the training window is a subset of
one run, but once training covers the whole of ``dns_khi`` there is no later
time left to hold out, and a held-out band of the same realisation is adjacent
to training data in a strongly non-stationary flow. The honest test is a
*different realisation* — a second DNS from another seed — and scoring it needs
an entry point that takes a checkpoint and a data directory.

What it measures
----------------
Beyond the usual relative L2, the three quantities that a long rollout actually
depends on, each reported against time so a collapse is visible rather than
averaged away:

  * **amplitude**  ``||tau_pred|| / ||tau_exact||``. A model regressed toward
    zero scores well on a normalised point-wise loss and supplies no stress.
  * **high-k content** the fraction of ``tau``'s own power above ``k_hi``. A
    stress that is too smooth cannot sustain the small scales whatever its
    amplitude.
  * **transfer** ``Pi = tau^ij S_ij``, and its sign agreement with the exact
    stress. This is what moves resolved kinetic energy between scales.

These were what exposed the previous model's late-time collapse: fine to
t ~ 12, then 40% of the amplitude, 19% of the high-k content and 11% of the
transfer by t = 20, while the training window had ended at t = 6.4.

Usage
-----
    # Score on an unseen realisation
    python -m src.evaluate_apriori --checkpoint checkpoints_fullrange/best_model.pt \
        --data-dir dns_khi_s1

    # Compare two checkpoints on the same data
    python -m src.evaluate_apriori --checkpoint checkpoints/best_model.pt \
        --data-dir dns_khi_s1 --label old
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.dataset import compute_sgs_stress, read_snapshot
from src.features import extract_tbnn_features, periodic_gradient_2d
from src.minit_calibrate import l2_norm_paper

try:
    from utils import CRMHD_TBNN
except Exception as exc:  # pragma: no cover
    CRMHD_TBNN = None
    _IMPORT_ERR = exc

DTYPE = torch.float64


def shell_power(field: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Isotropically binned power spectrum of a 2D field."""
    Ny, Nx = field.shape
    P = (torch.fft.fft2(field).abs() ** 2).numpy()
    ky = np.fft.fftfreq(Ny, d=1.0 / Ny)
    kx = np.fft.fftfreq(Nx, d=1.0 / Nx)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kk = np.sqrt(KX**2 + KY**2).T
    kbin = np.rint(kk).astype(int)
    kmax = min(Ny, Nx) // 2
    return np.arange(kmax + 1), np.array([P[kbin == k].sum() for k in range(kmax + 1)])


def highk_fraction(field: torch.Tensor, k_hi: int) -> float:
    """Share of a field's spectral power at or above ``k_hi``."""
    k, E = shell_power(field)
    total = float(E[k >= 1].sum())
    return float(E[k >= k_hi].sum()) / max(total, 1e-300)


def score_checkpoint(
    checkpoint: str,
    data_dir: str,
    r: int = 16,
    stride: int = 10,
    t_lo: float = 1.0,
    t_hi: float = 1e30,
    k_hi: int | None = None,
    device_str: str = "cpu",
) -> dict:
    """Score one checkpoint against the exact SGS stress of ``data_dir``."""
    if CRMHD_TBNN is None:  # pragma: no cover
        raise ImportError(f"CRMHD_TBNN unavailable: {_IMPORT_ERR}")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model, ck = CRMHD_TBNN.from_checkpoint(checkpoint, device=device, dtype=DTYPE)
    model.eval()
    inv_mean, inv_std = ck["inv_mean"].cpu(), ck["inv_std"].cpu()
    th_mean, th_std = ck["thermo_mean"].cpu(), ck["thermo_std"].cpu()

    split = ck.get("split_times", {})
    train_hi = max(split["train"]) if split.get("train") else None

    files = sorted(glob.glob(os.path.join(data_dir, "h*.h5")))
    if not files:
        raise FileNotFoundError(f"no h*.h5 snapshots under {data_dir!r}")

    rows: list[dict] = []
    pbar = tqdm(files[::stride], desc=f"Scoring on {data_dir}", unit="snap",
                dynamic_ncols=True, file=sys.stdout)
    for path in pbar:
        snap = read_snapshot(path)
        t = float(snap["t"])
        if not (t_lo - 1e-9 <= t <= t_hi + 1e-9):
            continue

        fine = [torch.tensor(snap[k], dtype=DTYPE) for k in ("rho", "vx", "vy", "p")]
        coarse, tau_full = compute_sgs_stress(*fine, r=r, project_to_fluid_frame=True)
        rho, vx = coarse["rho_bar"], coarse["vx_tilde"]
        vy, p = coarse["vy_tilde"], coarse["p_bar"]
        Ny, Nx = rho.shape
        if k_hi is None:
            k_hi = Nx // 4      # same relative band at every r (16 at r=16)
        dx = dy = 1.0 / Nx

        tau_ex = torch.stack(
            [tau_full[..., 1, 1], tau_full[..., 2, 2], tau_full[..., 1, 2]], dim=-1
        )
        dvx_dx, dvx_dy = periodic_gradient_2d(vx, dx, dy, order=4)
        dvy_dx, dvy_dy = periodic_gradient_2d(vy, dx, dy, order=4)
        S = torch.stack([dvx_dx, dvy_dy, 0.5 * (dvx_dy + dvy_dx)], dim=-1)

        feats = extract_tbnn_features(rho, vx, vy, p, dx, dy, gradient_order=4)
        N = Ny * Nx
        inv = ((feats["invariants"].reshape(N, 15) - inv_mean) / inv_std).to(DTYPE)
        th = ((feats["thermo_inputs"].reshape(N, 2) - th_mean) / th_std).to(DTYPE)
        basis = feats["basis_tensors"].reshape(N, 12, 4, 4).to(DTYPE)
        with torch.no_grad():
            dT, _, _ = model(inv.to(device), th.to(device), basis.to(device))
        tp = dT.cpu().reshape(Ny, Nx, 4, 4)
        tau_pr = torch.stack([tp[..., 1, 1], tp[..., 2, 2], tp[..., 1, 2]], dim=-1)

        def transfer(tau):
            return (tau[..., 0] * S[..., 0] + tau[..., 1] * S[..., 1]
                    + 2.0 * tau[..., 2] * S[..., 2])

        pi_pr, pi_ex = float(transfer(tau_pr).mean()), float(transfer(tau_ex).mean())
        amp_ex = float(tau_ex.norm())
        a = tau_pr.flatten() - tau_pr.mean()
        b = tau_ex.flatten() - tau_ex.mean()

        rows.append({
            "t": t,
            "seen_in_training": (train_hi is not None and t <= train_hi + 1e-9),
            "amp_ratio": float(tau_pr.norm()) / max(amp_ex, 1e-300),
            "rel_l2": float((tau_pr - tau_ex).norm() / max(amp_ex, 1e-300)),
            "l2_eq25": float(l2_norm_paper(tau_ex.flatten(), tau_pr.flatten())),
            "corr": float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-300)),
            "hifrac_pred": highk_fraction(tau_pr[..., 0], k_hi),
            "hifrac_exact": highk_fraction(tau_ex[..., 0], k_hi),
            "pi_pred": pi_pr,
            "pi_exact": pi_ex,
            "pi_ratio": pi_pr / pi_ex if abs(pi_ex) > 0 else float("nan"),
            "pi_sign_ok": bool(pi_pr * pi_ex > 0),
        })
        pbar.set_postfix(t=f"{t:.1f}")

    return {
        "checkpoint": checkpoint,
        "data_dir": data_dir,
        "coarsening_factor": r,
        "k_hi": k_hi,
        "train_hi": train_hi,
        "rows": rows,
    }


def report(res: dict, label: str = "") -> None:
    rows = res["rows"]
    if not rows:
        print("No snapshots in range.")
        return

    tag = f" [{label}]" if label else ""
    print(f"\n=== A-priori score{tag}: {Path(res['checkpoint']).parent.name} "
          f"on {res['data_dir']} (r={res['coarsening_factor']}) ===")
    if res["train_hi"] is not None:
        print(f"  checkpoint's training window ends at t = {res['train_hi']}")
    print(f"  high-k measured at k >= {res['k_hi']}\n")

    print(f"{'t':>6} {'seen':>5} {'amp':>7} {'rel_L2':>8} {'corr':>7} "
          f"{'hi-k pred':>10} {'hi-k exact':>11} {'Pi/Pi_ex':>9}")
    print("-" * 74)
    for r_ in rows:
        print(f"{r_['t']:6.1f} {'yes' if r_['seen_in_training'] else 'no':>5} "
              f"{r_['amp_ratio']:7.3f} {r_['rel_l2']:8.3f} {r_['corr']:+7.3f} "
              f"{r_['hifrac_pred']:10.1%} {r_['hifrac_exact']:11.1%} "
              f"{r_['pi_ratio']:9.3f}")

    def agg(key):
        return float(np.mean([r_[key] for r_ in rows]))

    print(f"\n  mean amplitude ratio  : {agg('amp_ratio'):.3f}   (1 = right magnitude)")
    print(f"  mean correlation      : {agg('corr'):+.3f}")
    if all("l2_eq25" in r_ for r_ in rows):   # absent in results saved before it existed
        eq25 = float(np.sqrt(np.mean([r_["l2_eq25"] ** 2 for r_ in rows])))
        print(f"  Eq. 25 L2, RMS in t   : {eq25:.3f}   "
              f"(0 perfect, 1 ~ corr 0.5, sqrt2 uncorrelated)")
    print(f"  mean high-k in tau    : {agg('hifrac_pred'):.1%} "
          f"vs {agg('hifrac_exact'):.1%} exact")
    print(f"  mean Pi ratio         : {agg('pi_ratio'):.3f}")
    print(f"  Pi sign agreement     : {np.mean([r_['pi_sign_ok'] for r_ in rows]):.0%}")

    # The failure that a single average hides: a model can be right early and
    # collapse late, which is exactly what a long rollout exposes.
    half = len(rows) // 2
    if half >= 2:
        early = float(np.mean([r_["amp_ratio"] for r_ in rows[:half]]))
        late = float(np.mean([r_["amp_ratio"] for r_ in rows[half:]]))
        print(f"\n  amplitude, first half {early:.3f} -> second half {late:.3f}")
        if late < 0.7 * early:
            print("  The stress decays over the run. A time-averaged score hides this, "
                  "and it is\n  what lets high-k energy drain away in a long rollout.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score a TBNN checkpoint on any DNS")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    ap.add_argument("--data-dir", default="dns_khi_s1")
    ap.add_argument("--coarsening-factor", "-r", type=int, default=16)
    ap.add_argument("--stride", type=int, default=10, help="snapshot stride")
    ap.add_argument("--t-lo", type=float, default=1.0)
    ap.add_argument("--t-hi", type=float, default=1e30)
    ap.add_argument("--k-hi", type=int, default=None,
                    help="first shell counted as high-k (Nyquist is 32 at r=16)")
    ap.add_argument("--label", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    res = score_checkpoint(
        checkpoint=args.checkpoint, data_dir=args.data_dir,
        r=args.coarsening_factor, stride=args.stride,
        t_lo=args.t_lo, t_hi=args.t_hi, k_hi=args.k_hi, device_str=args.device,
    )
    report(res, args.label)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
