"""Why does the TBNN not restore high-k energy, when the exact stress does?

At t = 20 the oracle holds 33% of the filtered DNS's high-k energy against 8%
for the unclosed solver — so the SGS term *can* sustain the small scales here.
The TBNN reaches 12%, below MInIT's 16%. This measures, on clean filtered-DNS
input (no closed-loop feedback), what its predicted stress is missing:

  * amplitude  -- is tau_pred simply too small?
  * spectrum   -- is it too smooth, i.e. missing high-k content of its own?
  * transfer   -- Pi = tau^ij S_ij. Restoring small-scale energy requires
                  *backscatter*, Pi < 0. A closure that is net dissipative can
                  only ever remove high-k energy, however accurate it looks.

Run: PYTHONPATH=. ./.venv/bin/python -u scratch/tbnn_highk_diagnosis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import compute_sgs_stress, read_snapshot
from src.features import extract_tbnn_features, periodic_gradient_2d
from src.minit import MInITCoefficients, assemble_tau
from utils import CRMHD_TBNN

DTYPE = torch.float64
BANDS = {"low": (1, 5), "mid": (6, 15), "high": (16, 32)}


def shell_spectrum(field: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Isotropically binned power spectrum of a 2D field."""
    Ny, Nx = field.shape
    F = torch.fft.fft2(field)
    P = (F.abs() ** 2).numpy()
    ky = np.fft.fftfreq(Ny, d=1.0 / Ny)
    kx = np.fft.fftfreq(Nx, d=1.0 / Nx)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kk = np.sqrt(KX**2 + KY**2).T
    kbin = np.rint(kk).astype(int)
    kmax = min(Ny, Nx) // 2
    out = np.array([P[kbin == k].sum() for k in range(kmax + 1)])
    return np.arange(kmax + 1), out


def band_energy(field: torch.Tensor) -> dict[str, float]:
    k, E = shell_spectrum(field)
    return {name: float(E[(k >= lo) & (k <= hi)].sum()) for name, (lo, hi) in BANDS.items()}


def main():
    import glob

    # Read the snapshots directly rather than the calibration cache: that cache
    # does not store the coarse pressure, and the TBNN's thermo inputs and basis
    # tensors both depend on it. Substituting a constant silently changes what
    # the network is being asked.
    files = sorted(glob.glob("dns_khi/h*.h5"))
    snaps = [read_snapshot(f) for f in files]
    snaps.sort(key=lambda s: s["t"])
    wanted = [1.0, 3.0, 5.0, 6.0, 6.5, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    picked = [min(snaps, key=lambda s, w=w: abs(s["t"] - w)) for w in wanted]

    r = 16
    model, ck = CRMHD_TBNN.from_checkpoint(
        "checkpoints/best_model.pt", device=torch.device("cpu"), dtype=DTYPE,
    )
    model.eval()
    inv_mean, inv_std = ck["inv_mean"], ck["inv_std"]
    th_mean, th_std = ck["thermo_mean"], ck["thermo_std"]
    train_hi = max(ck["split_times"]["train"])
    print(f"TBNN training window ends at t = {train_hi}\n")

    minit = MInITCoefficients.paper()
    rows = []

    for snap in picked:
        fine = [torch.tensor(snap[k], dtype=DTYPE) for k in ("rho", "vx", "vy", "p")]
        coarse, tau_full = compute_sgs_stress(*fine, r=r, project_to_fluid_frame=True)
        rho = coarse["rho_bar"]; vx = coarse["vx_tilde"]
        vy = coarse["vy_tilde"]; p = coarse["p_bar"]
        Ny, Nx = rho.shape
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
            dT, _, _ = model(inv, th, basis)
        tp = dT.reshape(Ny, Nx, 4, 4)
        tau_tb = torch.stack([tp[..., 1, 1], tp[..., 2, 2], tp[..., 1, 2]], dim=-1)

        e_kh = 0.5 * (tau_ex[..., 0] + tau_ex[..., 1])
        prims = torch.stack([rho, vx, vy, p], dim=-1)
        tm = assemble_tau(e_kh, prims, minit)
        tau_mi = torch.stack([tm[..., 1, 1], tm[..., 2, 2], tm[..., 1, 2]], dim=-1)

        def transfer(tau):
            return (tau[..., 0] * S[..., 0] + tau[..., 1] * S[..., 1]
                    + 2 * tau[..., 2] * S[..., 2])

        row = {"t": float(snap["t"]), "train": float(snap["t"]) <= train_hi + 1e-9}
        for name, tau in (("exact", tau_ex), ("tbnn", tau_tb), ("minit", tau_mi)):
            row[f"{name}_amp"] = float(tau.norm())
            row[f"{name}_pi"] = float(transfer(tau).mean())
            be = band_energy(tau[..., 0])
            row[f"{name}_hifrac"] = be["high"] / max(sum(be.values()), 1e-300)
        rows.append(row)

    print("Pi = tau^ij S_ij.  Pi > 0 feeds the RESOLVED scales (backscatter);")
    print("Pi < 0 drains them (forward transfer). Sign agreement is what matters.\n")
    print(f"{'t':>6} {'split':>6} | {'||tb||/||ex||':>12} {'hi-k tb':>8} {'hi-k ex':>8} "
          f"| {'Pi tbnn':>11} {'Pi exact':>11} {'Pi minit':>11} {'sign':>5}")
    print("-" * 96)
    for r_ in rows:
        ok = "ok" if r_["tbnn_pi"] * r_["exact_pi"] > 0 else "WRONG"
        print(f"{r_['t']:6.1f} {'train' if r_['train'] else 'test':>6} | "
              f"{r_['tbnn_amp']/r_['exact_amp']:12.3f} {r_['tbnn_hifrac']:8.2%} "
              f"{r_['exact_hifrac']:8.2%} | {r_['tbnn_pi']:+11.3e} "
              f"{r_['exact_pi']:+11.3e} {r_['minit_pi']:+11.3e} {ok:>5}")

    tr = [r_ for r_ in rows if r_["train"]]
    te = [r_ for r_ in rows if not r_["train"]]
    for name, grp in (("TRAIN", tr), ("TEST", te)):
        if not grp:
            continue
        amp = np.mean([g["tbnn_amp"] / g["exact_amp"] for g in grp])
        agree = np.mean([g["tbnn_pi"] * g["exact_pi"] > 0 for g in grp])
        print(f"\n{name}: amplitude ratio {amp:.3f}, Pi sign agreement {agree:.0%}, "
              f"hi-k tau {np.mean([g['tbnn_hifrac'] for g in grp]):.1%} "
              f"vs exact {np.mean([g['exact_hifrac'] for g in grp]):.1%}")


if __name__ == "__main__":
    main()
