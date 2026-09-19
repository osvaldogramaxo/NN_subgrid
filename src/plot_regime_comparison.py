r"""
plot_regime_comparison.py

Generates a multi-snapshot regime validation comparison for CRMHD-TBNN.
For 10 snapshots across the entire simulation time of the DNS (from laminar
shear to fully developed turbulence), generates a 5-panel row:
  1. Original FULL DNS (\rho on fine grid 1024x1024)
  2. Filtered DNS (\bar{\rho} on coarse grid 256x256)
  3. Filtered SGS (Exact \tau_SGS on coarse grid)
  4. Predicted SGS (TBNN \Delta T_SGS prediction)
  5. Difference (Pred - Exact) with relative L2 error

Saves:
  - results/regime_comparison_10snapshots.png (10x5 master grid)
  - results/regimes/snapshot_{i:02d}_t{t:.2f}.png (individual per-snapshot rows)
  - results/regime_metrics.json (quantitative error breakdown)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import CRMHD_TBNN
from src.dataset import (
    read_snapshot,
    compute_sgs_stress,
)
from src.features import extract_tbnn_features


REGIME_LABELS = {
    0.0: "Initial Shear Layers (t=0.0)",
    2.0: "Primary KH Roll-up (t=2.0)",
    5.0: "Secondary Vortex Cores (t=5.0)",
    9.0: "Vortex Pairing & Merger (t=9.0)",
    20.0: "Early Turbulent Cascade (t=20.0)",
    35.0: "Turbulent Mixing (t=35.0)",
    50.0: "Fully Developed Turbulence (t=50.0)",
    65.0: "Decaying / Mixed Turbulence (t=65.0)",
    80.0: "Validation Time Window (t=80.0)",
    100.0: "Asymptotic / Out-of-Distribution (t=100.0)",
}


def select_snapshots(
    data_dir: str,
    target_times: list[float] | None = None,
    num_snapshots: int = 10,
) -> list[tuple[str, float, str]]:
    """Select representative snapshot files across simulation timeline."""
    files = sorted(glob.glob(os.path.join(data_dir, "h*.h5")))
    if not files:
        raise FileNotFoundError(f"No HDF5 snapshots found in {data_dir}")

    all_snaps = []
    for f in files:
        snap = read_snapshot(f)
        all_snaps.append((f, float(snap["t"])))

    all_snaps.sort(key=lambda x: x[1])

    if target_times is None:
        min_t, max_t = all_snaps[0][1], all_snaps[-1][1]
        target_times = list(np.linspace(min_t, max_t, num_snapshots))

    selected = []
    used_files = set()
    for tt in target_times:
        candidates = [s for s in all_snaps if s[0] not in used_files]
        if not candidates:
            break
        best = min(candidates, key=lambda s: abs(s[1] - tt))
        used_files.add(best[0])

        # Assign descriptive label
        label = REGIME_LABELS.get(round(best[1], 1), f"Simulation t={best[1]:.2f}")
        selected.append((best[0], best[1], label))

    selected.sort(key=lambda x: x[1])
    return selected


def evaluate_snapshot(
    model: CRMHD_TBNN,
    ckpt: dict,
    snap_path: str,
    coarsening_factor: int = 4,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Evaluate DNS and TBNN on a single snapshot."""
    if device is None:
        device = torch.device("cpu")

    snap = read_snapshot(snap_path)
    t = float(snap["t"])
    r = coarsening_factor

    rho_fine = torch.tensor(snap["rho"], dtype=dtype, device=device)
    vx_fine = torch.tensor(snap["vx"], dtype=dtype, device=device)
    vy_fine = torch.tensor(snap["vy"], dtype=dtype, device=device)
    p_fine = torch.tensor(snap["p"], dtype=dtype, device=device)

    # 1. Exact coarse and SGS stress
    coarse, tau_exact = compute_sgs_stress(
        rho_fine, vx_fine, vy_fine, p_fine, r=r, project_to_fluid_frame=True,
    )

    Ny_c, Nx_c = coarse["rho_bar"].shape
    dx = 1.0 / Nx_c
    dy = 1.0 / Ny_c

    # 2. Extract features
    features = extract_tbnn_features(
        coarse["rho_bar"],
        coarse["vx_tilde"],
        coarse["vy_tilde"],
        coarse["p_bar"],
        dx,
        dy,
        gradient_order=4,
    )

    N = Ny_c * Nx_c
    inv = features["invariants"].reshape(N, 15)
    thermo = features["thermo_inputs"].reshape(N, 2)
    basis = features["basis_tensors"].reshape(N, 12, 4, 4)

    inv_mean = ckpt["inv_mean"].to(device, dtype=dtype)
    inv_std = ckpt["inv_std"].to(device, dtype=dtype)
    thermo_mean = ckpt["thermo_mean"].to(device, dtype=dtype)
    thermo_std = ckpt["thermo_std"].to(device, dtype=dtype)

    inv_norm = (inv - inv_mean) / inv_std
    thermo_norm = (thermo - thermo_mean) / thermo_std

    with torch.no_grad():
        delta_T, G, P = model(inv_norm, thermo_norm, basis)

    tau_pred = delta_T.reshape(Ny_c, Nx_c, 4, 4)

    # Metrics
    diff = tau_pred - tau_exact
    rel_l2_total = (torch.norm(diff) / torch.norm(tau_exact)).item()
    rel_l2_xx = (torch.norm(diff[..., 1, 1]) / torch.norm(tau_exact[..., 1, 1])).item()
    rel_l2_xy = (torch.norm(diff[..., 1, 2]) / torch.norm(tau_exact[..., 1, 2])).item()
    rel_l2_yy = (torch.norm(diff[..., 2, 2]) / torch.norm(tau_exact[..., 2, 2])).item()

    return {
        "t": t,
        "rho_fine": rho_fine.cpu().numpy(),
        "rho_coarse": coarse["rho_bar"].cpu().numpy(),
        "tau_exact": tau_exact.cpu().numpy(),
        "tau_pred": tau_pred.cpu().numpy(),
        "rel_l2_total": rel_l2_total,
        "rel_l2_xx": rel_l2_xx,
        "rel_l2_xy": rel_l2_xy,
        "rel_l2_yy": rel_l2_yy,
    }


def generate_regime_plots(
    checkpoint_path: str = "checkpoints/best_model.pt",
    data_dir: str = "test1",
    coarsening_factor: int = 4,
    output_dir: str = "results",
    device_str: str = "cpu",
    component: str = "xx",
):
    """Generate 10-snapshot 5-column regime comparison plot."""
    device = torch.device(device_str)
    dtype = torch.float64
    os.makedirs(output_dir, exist_ok=True)
    regimes_dir = os.path.join(output_dir, "regimes")
    os.makedirs(regimes_dir, exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_path}...")
    model, ckpt = CRMHD_TBNN.from_checkpoint(checkpoint_path, device=device, dtype=dtype)
    model.eval()

    target_times = [0.0, 2.0, 5.0, 9.0, 20.0, 35.0, 50.0, 65.0, 80.0, 100.0]
    selected_snapshots = select_snapshots(data_dir, target_times=target_times)
    num_snaps = len(selected_snapshots)

    comp_indices = {"xx": (1, 1), "xy": (1, 2), "yy": (2, 2), "0x": (0, 1)}
    mu, nu = comp_indices.get(component, (1, 1))
    comp_title = f"τ^{{{component}}}"

    print(f"\nEvaluating {num_snaps} snapshots across regimes (component: {comp_title})...")

    # Set up master figure: 10 rows x 5 columns
    fig, axes = plt.subplots(num_snaps, 5, figsize=(24, 4.2 * num_snaps))
    if num_snaps == 1:
        axes = np.expand_dims(axes, 0)

    column_titles = [
        "1. Original FULL DNS\n(ρ on 1024×1024)",
        "2. Filtered DNS\n(ρ̄ on 256×256)",
        f"3. Filtered SGS\n(Exact {comp_title})",
        f"4. Predicted SGS\n(TBNN {comp_title})",
        "5. Difference (Pred − Exact)\n& Rel L2 Error",
    ]

    metrics_list = []

    for row, (fpath, t_val, label) in enumerate(selected_snapshots):
        print(f"  [{row+1}/{num_snaps}] Processing {label} (t={t_val:.2f})...", flush=True)
        res = evaluate_snapshot(
            model, ckpt, fpath, coarsening_factor=coarsening_factor, device=device, dtype=dtype,
        )
        metrics_list.append({
            "snapshot": os.path.basename(fpath),
            "t": t_val,
            "label": label,
            "rel_l2_total": res["rel_l2_total"],
            "rel_l2_xx": res["rel_l2_xx"],
            "rel_l2_xy": res["rel_l2_xy"],
            "rel_l2_yy": res["rel_l2_yy"],
        })

        rho_fine = res["rho_fine"]
        rho_coarse = res["rho_coarse"]
        exact_comp = res["tau_exact"][..., mu, nu]
        pred_comp = res["tau_pred"][..., mu, nu]
        diff_comp = pred_comp - exact_comp
        rel_l2 = (res["rel_l2_xx"] if component == "xx" else
                  res["rel_l2_xy"] if component == "xy" else
                  res["rel_l2_total"])

        # Colorscales
        rho_vmin, rho_vmax = np.min(rho_fine), np.max(rho_fine)
        sgs_vmin = min(np.min(exact_comp), np.min(pred_comp))
        sgs_vmax = max(np.max(exact_comp), np.max(pred_comp))

        # Ensure symmetric zero for shear components if applicable
        if sgs_vmin < 0 and sgs_vmax > 0:
            limit = max(abs(sgs_vmin), abs(sgs_vmax))
            sgs_vmin, sgs_vmax = -limit, limit

        diff_lim = max(abs(np.min(diff_comp)), abs(np.max(diff_comp)), 1e-12)

        # Plot 1: Full DNS
        ax = axes[row, 0]
        im = ax.imshow(rho_fine, origin="lower", cmap="viridis", vmin=rho_vmin, vmax=rho_vmax, aspect="equal")
        ax.set_ylabel(f"{label}", fontsize=12, fontweight="bold")
        if row == 0:
            ax.set_title(column_titles[0], fontsize=13, fontweight="bold", pad=12)
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)

        # Plot 2: Filtered DNS
        ax = axes[row, 1]
        im = ax.imshow(rho_coarse, origin="lower", cmap="viridis", vmin=rho_vmin, vmax=rho_vmax, aspect="equal")
        if row == 0:
            ax.set_title(column_titles[1], fontsize=13, fontweight="bold", pad=12)
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)

        # Plot 3: Filtered SGS
        ax = axes[row, 2]
        cmap_sgs = "coolwarm" if (sgs_vmin < 0 and sgs_vmax > 0) else "plasma"
        im = ax.imshow(exact_comp, origin="lower", cmap=cmap_sgs, vmin=sgs_vmin, vmax=sgs_vmax, aspect="equal")
        if row == 0:
            ax.set_title(column_titles[2], fontsize=13, fontweight="bold", pad=12)
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)

        # Plot 4: Predicted SGS
        ax = axes[row, 3]
        im = ax.imshow(pred_comp, origin="lower", cmap=cmap_sgs, vmin=sgs_vmin, vmax=sgs_vmax, aspect="equal")
        if row == 0:
            ax.set_title(column_titles[3], fontsize=13, fontweight="bold", pad=12)
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)

        # Plot 5: Difference
        ax = axes[row, 4]
        im = ax.imshow(diff_comp, origin="lower", cmap="seismic", vmin=-diff_lim, vmax=diff_lim, aspect="equal")
        ax.set_title(f"Rel L2 Err: {rel_l2*100:.2f}%", fontsize=11, fontweight="bold", color="darkred" if rel_l2 > 0.15 else "darkgreen")
        if row == 0:
            ax.text(0.5, 1.25, column_titles[4], transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)

        # Also save an individual 1x5 figure for this snapshot
        fig_single, axes_s = plt.subplots(1, 5, figsize=(24, 4.8))
        # 1
        im = axes_s[0].imshow(rho_fine, origin="lower", cmap="viridis", vmin=rho_vmin, vmax=rho_vmax, aspect="equal")
        axes_s[0].set_title(f"Original Full DNS ρ (1024²)\n[{label}]", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=axes_s[0], shrink=0.8)
        # 2
        im = axes_s[1].imshow(rho_coarse, origin="lower", cmap="viridis", vmin=rho_vmin, vmax=rho_vmax, aspect="equal")
        axes_s[1].set_title("Filtered DNS ρ̄ (256²)", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=axes_s[1], shrink=0.8)
        # 3
        im = axes_s[2].imshow(exact_comp, origin="lower", cmap=cmap_sgs, vmin=sgs_vmin, vmax=sgs_vmax, aspect="equal")
        axes_s[2].set_title(f"Filtered Exact SGS {comp_title}", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=axes_s[2], shrink=0.8)
        # 4
        im = axes_s[3].imshow(pred_comp, origin="lower", cmap=cmap_sgs, vmin=sgs_vmin, vmax=sgs_vmax, aspect="equal")
        axes_s[3].set_title(f"Predicted SGS {comp_title}", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=axes_s[3], shrink=0.8)
        # 5
        im = axes_s[4].imshow(diff_comp, origin="lower", cmap="seismic", vmin=-diff_lim, vmax=diff_lim, aspect="equal")
        axes_s[4].set_title(f"Difference (Pred − Exact)\nRel L2 Err: {rel_l2*100:.2f}%", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=axes_s[4], shrink=0.8)

        plt.tight_layout()
        single_path = os.path.join(regimes_dir, f"snapshot_{row+1:02d}_t{t_val:06.2f}.png")
        fig_single.savefig(single_path, dpi=150, bbox_inches="tight")
        plt.close(fig_single)

    plt.tight_layout()
    master_path = os.path.join(output_dir, f"regime_comparison_10snapshots_{component}.png")
    fig.savefig(master_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Master 10-snapshot grid saved to: {master_path}")
    print(f"✓ Individual snapshot figures saved to: {regimes_dir}/")

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "regime_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")

    # Print summary table
    print("\n" + "=" * 85)
    print(f"{'Time':>6}  {'Regime Description':<38}  {'Rel L2 (Total)':>15}  {'Rel L2 (τ_xx)':>15}")
    print("=" * 85)
    for m in metrics_list:
        print(f"{m['t']:6.2f}  {m['label']:<38}  {m['rel_l2_total']*100:14.2f}%  {m['rel_l2_xx']*100:14.2f}%")
    print("=" * 85)

    return master_path, metrics_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-snapshot regime validation plot")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data-dir", type=str, default="test1")
    parser.add_argument("--coarsening-factor", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--component", type=str, default="xx", choices=["xx", "xy", "yy", "0x"])
    args = parser.parse_args()

    generate_regime_plots(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        coarsening_factor=args.coarsening_factor,
        output_dir=args.output_dir,
        device_str=args.device,
        component=args.component,
    )
