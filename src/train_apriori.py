r"""
train_apriori.py

A priori supervised training of the CRMHD-TBNN model.

Trains the network to predict the exact subgrid-scale stress tensor
τ_SGS^{μν} from Lorentz-invariant features extracted from Favre-filtered
coarse-grained DNS snapshots.

Loss function:
  Physics-based composite objective (see rmhd_sgs_loss_functions.md):
    L_total = L_tensor + α L_P + β L_spectral + γ L_ortho + δ L_WEC
  with covariant tensor regression, dissipation rate matching, spectral
  cascade preservation, orthogonality constraints, and WEC realizability.

Training uses AdamW optimizer with ReduceLROnPlateau schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import CRMHD_TBNN
from src.dataset import SGSDataset, create_dataloaders
from src.features import extract_tbnn_features
from src.losses import RMHDSGSLoss

# Cache format version — bump when adding new stored fields to force regeneration
CACHE_VERSION = 3


# ---------------------------------------------------------------------------
# Extended Dataset: pairs (invariants, thermo, basis_tensors) ↔ τ_SGS
# ---------------------------------------------------------------------------

class TBNNDataset(torch.utils.data.Dataset):
    """Dataset providing (invariants, thermo_inputs, basis_tensors, τ_SGS)
    tuples for training the CRMHD_TBNN model.

    Preprocesses all snapshots: for each coarse grid point, computes the
    full TBNN feature set (invariants + basis tensors) and pairs it with
    the exact SGS stress tensor.

    Parameters
    ----------
    data_dir : str or Path
    coarsening_factor : int
    time_range : tuple[float, float] | None
    dtype : torch.dtype
    """

    def __init__(
        self,
        data_dir: str | Path,
        coarsening_factor: int = 4,
        time_range: tuple[float, float] | None = None,
        stride: int = 1,
        cache_path: str | Path | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        from src.dataset import (
            load_all_snapshots, relativistic_favre_filter, compute_sgs_stress,
        )

        self.dtype = dtype
        r = coarsening_factor

        if cache_path is not None and os.path.exists(cache_path):
            cached = torch.load(cache_path, weights_only=False)
            if cached.get("cache_version", 1) >= CACHE_VERSION:
                print(f"  Loading cached dataset from {cache_path}...")
                self.invariants = cached["invariants"].to(dtype)
                self.thermo = cached["thermo"].to(dtype)
                self.basis = cached["basis"].to(dtype)
                self.tau_sgs = cached["tau_sgs"].to(dtype)
                self.u_con = cached["u_con"].to(dtype)
                self.grad_u_cov = cached["grad_u_cov"].to(dtype)
                self.rho = cached["rho"].to(dtype)
                self.p = cached["p"].to(dtype)
                self.inv_mean = cached["inv_mean"].to(dtype)
                self.inv_std = cached["inv_std"].to(dtype)
                self.thermo_mean = cached["thermo_mean"].to(dtype)
                self.thermo_std = cached["thermo_std"].to(dtype)
                self.tau_scale = cached["tau_scale"]
                self.snapshot_boundaries = cached["snapshot_boundaries"]
                return
            else:
                print(f"  Cache v{cached.get('cache_version', 1)} outdated (need v{CACHE_VERSION}). Regenerating...")

        snapshots = load_all_snapshots(data_dir)
        if time_range is not None:
            t_min, t_max = time_range
            snapshots = [s for s in snapshots if t_min <= s["t"] <= t_max]

        if stride > 1:
            snapshots = snapshots[::stride]

        if not snapshots:
            raise ValueError(f"No snapshots in time range {time_range}")

        print(f"  Processing {len(snapshots)} snapshots (stride={stride})...")

        all_inv = []
        all_thermo = []
        all_basis = []
        all_tau = []
        all_u_con = []
        all_grad_u = []
        all_rho = []
        all_p = []
        snapshot_boundaries: list[tuple[int, int, int]] = []  # (start_idx, Ny, Nx)
        current_offset = 0

        for snap in snapshots:
            rho = torch.tensor(snap["rho"], dtype=dtype)
            vx = torch.tensor(snap["vx"], dtype=dtype)
            vy = torch.tensor(snap["vy"], dtype=dtype)
            p = torch.tensor(snap["p"], dtype=dtype)

            # 1. Compute exact SGS stress on coarse grid
            coarse, tau_sgs = compute_sgs_stress(rho, vx, vy, p, r)

            # 2. Extract TBNN features on coarse grid
            Nc_y, Nc_x = coarse["rho_bar"].shape
            dx = 1.0 / Nc_x
            dy = 1.0 / Nc_y

            features = extract_tbnn_features(
                coarse["rho_bar"], coarse["vx_tilde"],
                coarse["vy_tilde"], coarse["p_bar"],
                dx, dy, gradient_order=4,
            )

            # Flatten spatial dims: (Nc_y * Nc_x, ...)
            N = Nc_y * Nc_x
            all_inv.append(features["invariants"].reshape(N, -1))
            all_thermo.append(features["thermo_inputs"].reshape(N, -1))
            all_basis.append(features["basis_tensors"].reshape(N, 12, 4, 4))
            all_tau.append(tau_sgs.reshape(N, 4, 4))
            all_u_con.append(features["u_con"].reshape(N, 4))
            all_grad_u.append(features["grad_u_cov"].reshape(N, 4, 4))
            all_rho.append(coarse["rho_bar"].reshape(N))
            all_p.append(coarse["p_bar"].reshape(N))
            snapshot_boundaries.append((current_offset, Nc_y, Nc_x))
            current_offset += N

        self.invariants = torch.cat(all_inv, dim=0)      # (Total, 15)
        self.thermo = torch.cat(all_thermo, dim=0)        # (Total, 2)
        self.basis = torch.cat(all_basis, dim=0)           # (Total, 12, 4, 4)
        self.tau_sgs = torch.cat(all_tau, dim=0)           # (Total, 4, 4)
        self.u_con = torch.cat(all_u_con, dim=0)           # (Total, 4)
        self.grad_u_cov = torch.cat(all_grad_u, dim=0)     # (Total, 4, 4)
        self.rho = torch.cat(all_rho, dim=0)               # (Total,)
        self.p = torch.cat(all_p, dim=0)                   # (Total,)
        self.snapshot_boundaries = snapshot_boundaries

        # Normalization
        self.inv_mean = self.invariants.mean(dim=0)
        self.inv_std = self.invariants.std(dim=0).clamp(min=1e-12)
        self.thermo_mean = self.thermo.mean(dim=0)
        self.thermo_std = self.thermo.std(dim=0).clamp(min=1e-12)
        self.tau_scale = self.tau_sgs.norm(dim=(-2, -1)).mean()

        # Save cache if cache_path is provided
        if cache_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            torch.save({
                "cache_version": CACHE_VERSION,
                "invariants": self.invariants, "thermo": self.thermo,
                "basis": self.basis, "tau_sgs": self.tau_sgs,
                "u_con": self.u_con, "grad_u_cov": self.grad_u_cov,
                "rho": self.rho, "p": self.p,
                "inv_mean": self.inv_mean, "inv_std": self.inv_std,
                "thermo_mean": self.thermo_mean, "thermo_std": self.thermo_std,
                "tau_scale": self.tau_scale,
                "snapshot_boundaries": self.snapshot_boundaries,
            }, cache_path)
            print(f"  Saved v{CACHE_VERSION} cache to {cache_path}")

    def __len__(self) -> int:
        return self.invariants.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.invariants[idx],
            self.thermo[idx],
            self.basis[idx],
            self.tau_sgs[idx],
            self.u_con[idx],
            self.grad_u_cov[idx],
            self.rho[idx],
            self.p[idx],
        )


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Early stopping monitor to terminate training when validation loss plateaus.

    Parameters
    ----------
    patience : int | None
        Number of epochs to wait without improvement before stopping.
        If <= 0 or None, early stopping is disabled.
    min_delta : float
        Minimum absolute improvement in monitored metric to qualify as an improvement.
    mode : str
        'min' (lower is better, e.g. MSE) or 'max' (higher is better).
    """

    def __init__(self, patience: int | None = None, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience if (patience is not None and patience > 0) else None
        self.min_delta = abs(min_delta)
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.best_epoch: int = 0
        self.early_stop = False

    def step(self, current_val: float, epoch: int) -> bool:
        """Check if metric improved or increment counter.

        Parameters
        ----------
        current_val : float
            Current value of the monitored metric.
        epoch : int
            Current epoch index (1-based).

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        if self.patience is None:
            return False

        if self.best_score is None:
            self.best_score = float(current_val)
            self.best_epoch = epoch
            self.counter = 0
            return False

        if self.mode == "min":
            improved = current_val < (self.best_score - self.min_delta)
        else:
            improved = current_val > (self.best_score + self.min_delta)

        if improved:
            self.best_score = float(current_val)
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: RMHDSGSLoss,
    inv_mean: torch.Tensor,
    inv_std: torch.Tensor,
    thermo_mean: torch.Tensor,
    thermo_std: torch.Tensor,
    device: torch.device,
    g_cov: torch.Tensor,
    dataset: TBNNDataset | None = None,
    spectral_snapshots: int = 4,
) -> dict[str, float]:
    """Train for one epoch with physics-based composite loss.

    Point-wise losses (L_tensor, L_P, L_ortho, L_WEC) are computed per
    minibatch.  The spectral cascade loss (L_spectral) requires spatially
    structured 2D grids and is computed in a separate pass at the end of
    the epoch on a random subset of training snapshots.

    Returns dict with total loss and individual component values.
    """
    model.train()
    total_loss = 0.0
    count = 0
    comp_accum: dict[str, float] = {}

    pbar = tqdm(dataloader, desc="  Train Batch", leave=False, disable=not sys.stdout.isatty())
    for inv, thermo, basis, tau_target, u_con, grad_u, rho_b, p_b in pbar:
        inv = inv.to(device)
        thermo = thermo.to(device)
        basis = basis.to(device)
        tau_target = tau_target.to(device)
        u_con = u_con.to(device)
        grad_u = grad_u.to(device)
        rho_b = rho_b.to(device)
        p_b = p_b.to(device)

        # Normalize inputs
        inv_norm = (inv - inv_mean) / inv_std
        thermo_norm = (thermo - thermo_mean) / thermo_std

        # Forward
        delta_T, coeffs, p_sgs = model(inv_norm, thermo_norm, basis)

        # Physics composite loss (point-wise terms, excludes spectral)
        loss, components = criterion(delta_T, tau_target, u_con, grad_u, g_cov, rho_b, p_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = inv.shape[0]
        total_loss += loss.item() * bs
        count += bs
        for k, v in components.items():
            comp_accum[k] = comp_accum.get(k, 0.0) + v * bs

    # -----------------------------------------------------------------------
    # Spectral loss pass on full snapshots (requires 2D grid structure)
    # -----------------------------------------------------------------------
    spec_loss_val = 0.0
    if (criterion.beta > 0
            and dataset is not None
            and hasattr(dataset, "snapshot_boundaries")
            and len(dataset.snapshot_boundaries) > 0):
        n_snaps = len(dataset.snapshot_boundaries)
        n_use = min(spectral_snapshots, n_snaps)
        snap_indices = torch.randperm(n_snaps)[:n_use].tolist()

        optimizer.zero_grad()
        spec_sum = torch.tensor(0.0, device=device, dtype=inv_mean.dtype)
        for si in snap_indices:
            start, Ny, Nx = dataset.snapshot_boundaries[si]
            N = Ny * Nx

            s_inv = dataset.invariants[start:start + N].to(device)
            s_thermo = dataset.thermo[start:start + N].to(device)
            s_basis = dataset.basis[start:start + N].to(device)
            s_tau = dataset.tau_sgs[start:start + N].to(device)

            s_inv_n = (s_inv - inv_mean) / inv_std
            s_thermo_n = (s_thermo - thermo_mean) / thermo_std

            s_pred, _, _ = model(s_inv_n, s_thermo_n, s_basis)

            tau_pred_2d = s_pred.reshape(Ny, Nx, 4, 4)
            tau_dns_2d = s_tau.reshape(Ny, Nx, 4, 4)

            s_loss, s_val = criterion.compute_spectral(tau_pred_2d, tau_dns_2d)
            spec_sum = spec_sum + s_loss / n_use
            spec_loss_val += s_val / n_use

        spec_sum.backward()
        optimizer.step()

    # Total loss reported includes spectral loss if enabled
    result = {"train_loss": total_loss / max(count, 1) + criterion.beta * spec_loss_val}
    for k, v in comp_accum.items():
        result[k] = v / max(count, 1)

    result["L_spectral"] = spec_loss_val
    return result


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: RMHDSGSLoss,
    inv_mean: torch.Tensor,
    inv_std: torch.Tensor,
    thermo_mean: torch.Tensor,
    thermo_std: torch.Tensor,
    device: torch.device,
    g_cov: torch.Tensor,
    dataset: TBNNDataset | None = None,
) -> dict[str, float]:
    """Evaluate model on validation data with physics-based composite loss.

    Returns dict with val_loss, mse, rel_l2, and per-component loss values.
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_norm_sq = 0.0
    total_diff_sq = 0.0
    count = 0
    comp_accum: dict[str, float] = {}

    for inv, thermo, basis, tau_target, u_con, grad_u, rho_b, p_b in dataloader:
        inv = inv.to(device)
        thermo = thermo.to(device)
        basis = basis.to(device)
        tau_target = tau_target.to(device)
        u_con = u_con.to(device)
        grad_u = grad_u.to(device)
        rho_b = rho_b.to(device)
        p_b = p_b.to(device)

        inv_norm = (inv - inv_mean) / inv_std
        thermo_norm = (thermo - thermo_mean) / thermo_std

        delta_T, coeffs, p_sgs = model(inv_norm, thermo_norm, basis)

        loss, components = criterion(delta_T, tau_target, u_con, grad_u, g_cov, rho_b, p_b)

        bs = inv.shape[0]
        total_loss += loss.item() * bs
        count += bs

        diff = delta_T - tau_target
        total_mse += diff.pow(2).sum().item()
        total_diff_sq += diff.pow(2).sum().item()
        total_norm_sq += tau_target.pow(2).sum().item()

        for k, v in components.items():
            comp_accum[k] = comp_accum.get(k, 0.0) + v * bs

    n_elem = count * 16  # 4x4 tensor

    # Spectral loss on validation snapshots (all of them)
    spec_loss_val = 0.0
    if (criterion.beta > 0
            and dataset is not None
            and hasattr(dataset, "snapshot_boundaries")
            and len(dataset.snapshot_boundaries) > 0):
        n_snaps = len(dataset.snapshot_boundaries)
        for si in range(n_snaps):
            start, Ny, Nx = dataset.snapshot_boundaries[si]
            N = Ny * Nx

            s_inv = dataset.invariants[start:start + N].to(device)
            s_thermo = dataset.thermo[start:start + N].to(device)
            s_basis = dataset.basis[start:start + N].to(device)
            s_tau = dataset.tau_sgs[start:start + N].to(device)

            s_inv_n = (s_inv - inv_mean) / inv_std
            s_thermo_n = (s_thermo - thermo_mean) / thermo_std

            s_pred, _, _ = model(s_inv_n, s_thermo_n, s_basis)

            tau_pred_2d = s_pred.reshape(Ny, Nx, 4, 4)
            tau_dns_2d = s_tau.reshape(Ny, Nx, 4, 4)

            _, s_val = criterion.compute_spectral(tau_pred_2d, tau_dns_2d)
            spec_loss_val += s_val / n_snaps

    val_composite = total_loss / max(count, 1) + criterion.beta * spec_loss_val
    result: dict[str, float] = {
        "val_loss": val_composite,
        "mse": total_mse / max(n_elem, 1),
        "rel_l2": (total_diff_sq / max(total_norm_sq, 1e-30)) ** 0.5,
    }
    for k, v in comp_accum.items():
        result[f"val_{k}"] = v / max(count, 1)
    result["val_L_spectral"] = spec_loss_val
    return result


# ---------------------------------------------------------------------------
# Plotting & Diagnostics
# ---------------------------------------------------------------------------

def plot_loss_curves(
    history: dict[str, list[float]],
    save_path: str | Path,
    include_mse: bool = True,
) -> None:
    """Plot training/validation loss curves, relative L2 error, adaptive parameters, and learning rate.

    Features:
    - Logarithmic scale for Relative L2 Error to resolve early transients (e.g. >1000%)
      and settled fine-tuning (e.g. 60-80%) across distinct training regimes.
    - Physical Validation MSE panel (when include_mse=True and val_mse exists) spanning
      physical error scales (10^-7 to 10^-3).
    - Settled-regime insets (epochs >= 50) for both Barron loss and Relative L2 error.
    - Vertical dashed demarcations at learning rate transitions across all subplots.
    - Star marker and annotation for the minimum validation L2 error and MSE.
    - Constrained layout padding to prevent twinx axis label collisions.
    """
    n_epochs = len(history.get("train_loss", []))
    if n_epochs == 0:
        return

    epochs = np.arange(1, n_epochs + 1)
    has_components = "L_tensor" in history and len(history["L_tensor"]) > 0
    has_adaptive = "alpha" in history and len(history["alpha"]) > 0
    has_mse = include_mse and "val_mse" in history and len(history["val_mse"]) > 0

    cols = ["loss"]
    if has_mse:
        cols.append("mse")
    cols.append("l2")
    if has_components:
        cols.append("components")
    elif has_adaptive:
        cols.append("adaptive")
    cols.append("lr")

    def _format_pct_log(val: float, _pos: object) -> str:
        return f"{val:g}%"

    def _format_pct_linear(val: float, _pos: object) -> str:
        return f"{val:.0f}%"

    num_cols = len(cols)
    fig, axes = plt.subplots(1, num_cols, figsize=(5.6 * num_cols, 4.8), layout="constrained")
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None and hasattr(layout_engine, "set"):
        getattr(layout_engine, "set")(wspace=0.10)
    if num_cols == 1:
        axes = [axes]
    else:
        axes = list(axes)

    # Detect learning rate change epochs (regime boundaries)
    lr_arr = np.array(history["lr"]) if "lr" in history and len(history["lr"]) == n_epochs else np.array([])
    lr_change_epochs: list[int] = []
    if len(lr_arr) > 1:
        change_idx = np.where(lr_arr[:-1] != lr_arr[1:])[0]
        lr_change_epochs = [int(epochs[i + 1]) for i in change_idx]

    def mark_regimes(ax: Axes) -> None:
        for ep in lr_change_epochs:
            ax.axvline(x=float(ep), color="indigo", linestyle=":", alpha=0.55, lw=1.2)

    train_loss = np.array(history["train_loss"])
    val_key = "val_loss" if "val_loss" in history and len(history["val_loss"]) > 0 else "val_mse"
    val_loss = np.array(history[val_key])

    has_inset = n_epochs >= 40
    settled_start = 50 if n_epochs >= 70 else max(5, int(n_epochs * 0.2))
    settled_mask = epochs >= settled_start

    # 1. Loss curve
    ax_loss = axes[cols.index("loss")]
    ax_loss.plot(epochs, train_loss, "b-", label="Train Loss", lw=1.6)
    ax_loss.plot(epochs, val_loss, "r--", label=f"Val Loss ({val_key})", lw=1.6)
    mark_regimes(ax_loss)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Composite Physics Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="lower left", fontsize=9, framealpha=0.85)

    if has_inset and np.any(settled_mask):
        axins_loss = ax_loss.inset_axes([0.44, 0.44, 0.52, 0.50])
        axins_loss.plot(epochs[settled_mask], train_loss[settled_mask], "b-", lw=1.2)
        axins_loss.plot(epochs[settled_mask], val_loss[settled_mask], "r--", lw=1.2)
        for ep in lr_change_epochs:
            if ep >= settled_start:
                axins_loss.axvline(x=float(ep), color="indigo", linestyle=":", alpha=0.55, lw=1.0)
        axins_loss.set_title(f"Settled (ep ≥ {settled_start})", fontsize=8, pad=2)
        axins_loss.tick_params(labelsize=7)
        axins_loss.grid(True, alpha=0.25)

    # 2. Validation MSE (Physical)
    if has_mse:
        ax_mse = axes[cols.index("mse")]
        val_mse = np.array(history["val_mse"])
        ax_mse.semilogy(epochs, val_mse, color="darkorange", lw=1.8, label="Val MSE")
        mark_regimes(ax_mse)
        min_mse_idx = int(np.argmin(val_mse))
        ax_mse.plot(epochs[min_mse_idx], val_mse[min_mse_idx], "r*", markersize=9,
                    label=f"Min: {val_mse[min_mse_idx]:.2e}")
        ax_mse.set_xlabel("Epoch")
        ax_mse.set_ylabel("Mean Squared Error")
        ax_mse.set_title("Validation MSE (Physical)")
        ax_mse.grid(True, which="both", alpha=0.3)
        ax_mse.legend(loc="upper right", fontsize=9, framealpha=0.85)

    # 3. Relative L2 error (log scale)
    ax_l2 = axes[cols.index("l2")]
    val_rel_l2 = np.array(history["val_rel_l2"]) * 100.0
    ax_l2.semilogy(epochs, val_rel_l2, color="forestgreen", lw=1.8, label="Val Rel L2")
    mark_regimes(ax_l2)
    min_l2_idx = int(np.argmin(val_rel_l2))
    ax_l2.plot(epochs[min_l2_idx], val_rel_l2[min_l2_idx], "r*", markersize=9,
               label=f"Min: {val_rel_l2[min_l2_idx]:.1f}% (ep {epochs[min_l2_idx]})")
    ax_l2.set_xlabel("Epoch")
    ax_l2.set_ylabel("Relative L2 Error (%)")
    ax_l2.set_title("Validation Relative L2 Error")
    ax_l2.grid(True, which="both", alpha=0.3)
    ax_l2.yaxis.set_major_formatter(FuncFormatter(_format_pct_log))
    ax_l2.legend(loc="lower left", fontsize=9, framealpha=0.85)

    if has_inset and np.any(settled_mask):
        axins_l2 = ax_l2.inset_axes([0.44, 0.44, 0.52, 0.50])
        axins_l2.plot(epochs[settled_mask], val_rel_l2[settled_mask], color="forestgreen", lw=1.2)
        for ep in lr_change_epochs:
            if ep >= settled_start:
                axins_l2.axvline(x=float(ep), color="indigo", linestyle=":", alpha=0.55, lw=1.0)
        axins_l2.set_title(f"Settled (ep ≥ {settled_start}, linear %)", fontsize=8, pad=2)
        y_settled = val_rel_l2[settled_mask]
        q_high = float(np.percentile(y_settled, 96))
        axins_l2.set_ylim(float(np.min(y_settled)) - 2.0, max(q_high + 5.0, 80.0))
        axins_l2.tick_params(labelsize=7)
        axins_l2.grid(True, alpha=0.25)
        axins_l2.yaxis.set_major_formatter(FuncFormatter(_format_pct_linear))

    # 4. Physics Loss Components
    if has_components:
        ax_comp = axes[cols.index("components")]
        comp_configs = [
            ("L_tensor", r"$\mathcal{L}_{\mathrm{tensor}}$", "tab:blue"),
            ("L_P", r"$\mathcal{L}_{\mathcal{P}}$", "crimson"),
            ("L_ortho", r"$\mathcal{L}_{\mathrm{ortho}}$", "purple"),
            ("L_WEC", r"$\mathcal{L}_{\mathrm{WEC}}$", "darkgreen"),
            ("L_spectral", r"$\mathcal{L}_{\mathrm{spectral}}$", "darkorange"),
        ]
        for k, lbl, col in comp_configs:
            if k in history and len(history[k]) == n_epochs:
                arr = np.array(history[k])
                ax_comp.semilogy(epochs, np.maximum(arr, 1e-15), color=col, lw=1.4, label=lbl)
        mark_regimes(ax_comp)
        ax_comp.set_xlabel("Epoch")
        ax_comp.set_ylabel("Raw Component Value")
        ax_comp.set_title("Physics Loss Components")
        ax_comp.grid(True, which="both", alpha=0.3)
        ax_comp.legend(loc="upper right", fontsize=8, framealpha=0.85)
    elif has_adaptive:
        ax_adapt = axes[cols.index("adaptive")]
        color_alpha = "tab:blue"
        ax_adapt.plot(epochs, history["alpha"], color=color_alpha, lw=1.8, label=r"Shape $\alpha$")
        ax_adapt.set_xlabel("Epoch")
        ax_adapt.set_ylabel(r"Shape $\alpha$", color=color_alpha)
        ax_adapt.tick_params(axis="y", labelcolor=color_alpha)
        ax_adapt.grid(True, alpha=0.3)
        mark_regimes(ax_adapt)

        if "scale" in history and len(history["scale"]) == n_epochs:
            ax_scale = ax_adapt.twinx()
            color_scale = "tab:orange"
            ax_scale.plot(epochs, history["scale"], color=color_scale, lw=1.8, linestyle="--", label=r"Scale $c$")
            ax_scale.set_ylabel(r"Scale $c$", color=color_scale)
            ax_scale.tick_params(axis="y", labelcolor=color_scale)
            if min(history["scale"]) > 0:
                ax_scale.set_yscale("log")
        ax_adapt.set_title("Adaptive Parameters")

    # 5. Learning rate schedule
    ax_lr = axes[cols.index("lr")]
    if len(lr_arr) == n_epochs:
        ax_lr.plot(epochs, lr_arr, color="darkmagenta", lw=1.8)
        if np.min(lr_arr) > 0:
            ax_lr.set_yscale("log")
        for ep in [epochs[0]] + list(lr_change_epochs):
            ep_idx = int(np.where(epochs == ep)[0][0])
            cur_val = lr_arr[ep_idx]
            ax_lr.annotate(
                f"{cur_val:.1e}",
                xy=(ep, cur_val),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="darkmagenta",
                weight="bold",
            )
    mark_regimes(ax_lr)
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.set_title("ReduceLROnPlateau Schedule")
    ax_lr.grid(True, which="both", alpha=0.3)

    plt.savefig(save_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_pred_vs_truth(
    model: nn.Module,
    val_ds: TBNNDataset,
    inv_mean: torch.Tensor,
    inv_std: torch.Tensor,
    thermo_mean: torch.Tensor,
    thermo_std: torch.Tensor,
    device: torch.device,
    save_path: str | Path,
    epoch: int,
    grid_size: tuple[int, int] = (256, 256),
) -> None:
    """Plot 2D spatial fields of predicted vs. ground truth SGS stresses and error maps."""
    model.eval()
    Ny, Nx = grid_size
    N = Ny * Nx

    # Take the first validation snapshot
    inv = val_ds.invariants[:N].to(device)
    thermo = val_ds.thermo[:N].to(device)
    basis = val_ds.basis[:N].to(device)
    tau_target = val_ds.tau_sgs[:N].to(device)

    inv_norm = (inv - inv_mean) / inv_std
    thermo_norm = (thermo - thermo_mean) / thermo_std

    delta_T, _, _ = model(inv_norm, thermo_norm, basis)

    tau_pred_2d = delta_T.reshape(Ny, Nx, 4, 4).cpu().numpy()
    tau_true_2d = tau_target.reshape(Ny, Nx, 4, 4).cpu().numpy()

    # Components to plot:
    # (1, 1): tau^xx (normal stress x)
    # (1, 2): tau^xy (shear stress xy)
    # (2, 2): tau^yy (normal stress y)
    # (0, 1): tau^0x (relativistic energy/momentum flux)
    components = [
        ((1, 1), r"\tau^{xx} \text{ (Normal Stress X)}"),
        ((1, 2), r"\tau^{xy} \text{ (Shear Stress)}"),
        ((2, 2), r"\tau^{yy} \text{ (Normal Stress Y)}"),
        ((0, 1), r"\tau^{0x} \text{ (Energy Flux X)}"),
    ]

    fig, axes = plt.subplots(len(components), 4, figsize=(20, 3.8 * len(components)))

    for row_idx, ((mu, nu), comp_name) in enumerate(components):
        truth = tau_true_2d[..., mu, nu]
        pred = tau_pred_2d[..., mu, nu]
        diff = pred - truth

        vmin = float(min(truth.min(), pred.min()))
        vmax = float(max(truth.max(), pred.max()))
        err_bound = float(max(abs(diff.min()), abs(diff.max()), 1e-12))

        # Ground truth heatmap
        ax0 = axes[row_idx, 0]
        im0 = ax0.imshow(truth, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax0.set_title(f"Exact ${comp_name}$ (Epoch {epoch})")
        plt.colorbar(im0, ax=ax0, shrink=0.85)

        # Prediction heatmap
        ax1 = axes[row_idx, 1]
        im1 = ax1.imshow(pred, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax1.set_title(f"TBNN Pred ${comp_name}$")
        plt.colorbar(im1, ax=ax1, shrink=0.85)

        # Difference heatmap
        ax2 = axes[row_idx, 2]
        im2 = ax2.imshow(diff, origin="lower", cmap="coolwarm", vmin=-err_bound, vmax=err_bound)
        ax2.set_title(f"Difference (Pred - Exact)")
        plt.colorbar(im2, ax=ax2, shrink=0.85)

        # Parity scatter plot
        ax3 = axes[row_idx, 3]
        sub = slice(None, None, 8)
        t_flat = truth[sub, sub].flatten()
        p_flat = pred[sub, sub].flatten()
        ax3.scatter(t_flat, p_flat, alpha=0.3, s=8, c="royalblue", edgecolors="none")
        line_min = float(min(vmin, t_flat.min(), p_flat.min()))
        line_max = float(max(vmax, t_flat.max(), p_flat.max()))
        ax3.plot([line_min, line_max], [line_min, line_max], "r--", lw=1.5, label="y = x")
        ax3.set_xlabel("Exact")
        ax3.set_ylabel("Predicted")
        ax3.set_title(f"Parity Plot (${comp_name}$)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train(
    data_dir: str = "test1",
    coarsening_factor: int = 4,
    train_time: tuple[float, float] = (1.0, 70.0),
    val_time: tuple[float, float] = (70.0, 85.0),
    stride: int = 5,
    epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 1e-3,
    lr_patience: int = 100,
    patience: int | None = None,
    min_delta: float = 0.0,
    restore_best_weights: bool = True,
    hidden_dim: int = 512,
    num_hidden_layers: int = 3,
    loss_alpha: float = 0.5,
    loss_beta: float = 0.1,
    loss_gamma: float = 5.0,
    loss_delta: float = 2.0,
    lambda_stab: float = 10.0,
    spectral_snapshots: int = 4,
    plot_interval: int = 10,
    checkpoint_dir: str = "checkpoints",
    device_str: str = "cuda",
    dtype_str: str = "float64",
    resume: str | bool | None = None,
    reset_lr: bool = False,
    reset_best: bool = False,
):
    """Main training function with physics-based composite loss and checkpoint resumption."""
    if patience is None:
        patience = 2 * lr_patience
    if device_str == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device(device_str if (torch.cuda.is_available() and device_str == "cuda") else "cpu")
    if device.type == "cuda":
        try:
            _ = torch.zeros(1, device=device)
        except Exception as e:
            print(f"CUDA device unavailable ({e}), falling back to CPU.")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            device = torch.device("cpu")

    dtype = torch.float64 if dtype_str == "float64" else torch.float32

    os.makedirs(checkpoint_dir, exist_ok=True)
    plots_dir = os.path.join(checkpoint_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Resolve resume checkpoint if requested
    resume_path: str | None = None
    ckpt: dict[str, Any] | None = None
    if resume:
        if resume is True or str(resume).lower() in ("auto", "true", "1"):
            # Auto-detect latest or best checkpoint in checkpoint_dir
            candidates = [
                os.path.join(checkpoint_dir, "latest_model.pt"),
                os.path.join(checkpoint_dir, "best_model.pt"),
                os.path.join(checkpoint_dir, "final_model.pt"),
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    resume_path = cand
                    break
            if resume_path is None:
                raise FileNotFoundError(
                    f"Resume requested, but no checkpoint found in '{checkpoint_dir}' (checked: {candidates})"
                )
        else:
            resume_path = str(resume)
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(f"Resume checkpoint file not found: {resume_path}")

        print(f"\n[Resume] Loading checkpoint from: {resume_path}")
        loaded = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(loaded, dict):
            ckpt = loaded
            if "hidden_dim" in ckpt:
                hidden_dim = int(ckpt["hidden_dim"])
            if "num_hidden_layers" in ckpt:
                num_hidden_layers = int(ckpt["num_hidden_layers"])

    train_cache = os.path.join(checkpoint_dir, f"train_cache_r{coarsening_factor}_s{stride}.pt")
    val_cache = os.path.join(checkpoint_dir, f"val_cache_r{coarsening_factor}_s{stride}.pt")

    print(f"Device: {device}")
    print(f"Loading training data from {data_dir}, r={coarsening_factor}, stride={stride}...")

    train_ds = TBNNDataset(
        data_dir, coarsening_factor=coarsening_factor, time_range=train_time,
        stride=stride, cache_path=train_cache, dtype=dtype,
    )

    val_ds = TBNNDataset(
        data_dir, coarsening_factor=coarsening_factor, time_range=val_time,
        stride=max(1, stride // 2), cache_path=val_cache, dtype=dtype,
    )

    coarse_grid = (
        (val_ds.snapshot_boundaries[0][1], val_ds.snapshot_boundaries[0][2])
        if hasattr(val_ds, "snapshot_boundaries") and len(val_ds.snapshot_boundaries) > 0
        else (1024 // coarsening_factor, 1024 // coarsening_factor)
    )

    print(f"  Training samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    print(f"  τ_SGS scale: {train_ds.tau_scale:.6e}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    # Normalization stats on device
    inv_mean = train_ds.inv_mean.to(device)
    inv_std = train_ds.inv_std.to(device)
    thermo_mean = train_ds.thermo_mean.to(device)
    thermo_std = train_ds.thermo_std.to(device)

    # Metric tensor: Minkowski diag(-1, 1, 1, 1)
    g_cov = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=dtype, device=device))

    # Model
    model = CRMHD_TBNN(
        num_invariants=15,
        num_thermo_inputs=2,
        num_basis_tensors=12,
        num_hidden_layers=num_hidden_layers,
        hidden_dim=hidden_dim,
    ).to(dtype=dtype, device=device)

    # Physics-based composite loss criterion
    criterion = RMHDSGSLoss(
        alpha=loss_alpha,
        beta=loss_beta,
        gamma=loss_gamma,
        delta=loss_delta,
        lambda_stab=lambda_stab,
    ).to(dtype=dtype, device=device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Physics Loss weights: α={loss_alpha} (dissipation), β={loss_beta} (spectral), "
        + f"γ={loss_gamma} (ortho), δ={loss_delta} (WEC), λ_stab={lambda_stab}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience, min_lr=1e-7,
    )
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode="min")

    best_val_loss = float("inf")
    best_epoch = 1
    start_epoch = 0

    # Restore state from checkpoint if resuming
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Resume] Restored model weights (hidden_dim={hidden_dim}, num_hidden_layers={num_hidden_layers})")

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if reset_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                print(f"[Resume] Overrode optimizer learning rate with --lr {lr}")
            else:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(f"[Resume] Restored optimizer state (learning rate: {cur_lr:.1e})")

        start_epoch = int(ckpt.get("epoch", 0))

        if reset_best:
            # Treat the checkpoint purely as a weight initialization (e.g. fine-tuning
            # onto a harder task where loss values from the old run aren't comparable):
            # drop the previous best_val_loss/scheduler baseline/early-stopping best
            # instead of carrying them over as the comparison target.
            print("[Resume] --reset-best: discarding checkpoint's best_val_loss / scheduler baseline / early-stopping best")
        else:
            best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))))
            best_epoch = int(ckpt.get("best_epoch", ckpt.get("epoch", 1)))

            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                print("[Resume] Restored ReduceLROnPlateau scheduler state")
            else:
                if best_val_loss < float("inf"):
                    scheduler.best = best_val_loss
                    print(f"[Resume] Initialized scheduler baseline best={scheduler.best:.6e}")

            if "early_stopping" in ckpt and isinstance(ckpt["early_stopping"], dict):
                es_dict: dict[str, Any] = ckpt["early_stopping"]
                best_sc = es_dict.get("best_score", best_val_loss)
                early_stopping.best_score = float(best_sc) if best_sc is not None else None
                early_stopping.best_epoch = int(es_dict.get("best_epoch", best_epoch))
            else:
                early_stopping.best_score = best_val_loss
                early_stopping.best_epoch = best_epoch

        # Reset the patience counter and stop flag on resume: a resumed run
        # gets a fresh window to improve on the (restored or reset) best, rather
        # than inheriting a stale non-improvement streak (or a stuck early_stop=True).
        early_stopping.counter = 0
        early_stopping.early_stop = False
        print(
            f"[Resume] Early stopping state: best_score={early_stopping.best_score} "
            + f"(best_epoch={early_stopping.best_epoch}); patience counter reset to 0/{early_stopping.patience}"
        )

        print(
            f"[Resume] Resuming training from epoch {start_epoch} up to target {epochs} "
            + f"(best epoch {best_epoch} with val loss {best_val_loss:.6e})\n"
        )

        if start_epoch >= epochs:
            print(
                f"[Resume] Checkpoint epoch ({start_epoch}) >= target epochs ({epochs}). Training is already complete!\n"
                + f"To train further, specify a higher --epochs value (e.g. --epochs {start_epoch + 100})."
            )
            return model, {}

    if early_stopping.patience is not None:
        print(f"LR scheduler patience: {lr_patience} epochs | Early stopping patience: {early_stopping.patience} epochs (2x LR patience, min_delta={min_delta})")
    else:
        print(f"LR scheduler patience: {lr_patience} epochs | Early stopping: disabled")

    early_stopped = False
    stopped_epoch = epochs
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    latest_model_path = os.path.join(checkpoint_dir, "latest_model.pt")
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_mse": [],
        "val_rel_l2": [],
        "L_tensor": [],
        "L_P": [],
        "L_ortho": [],
        "L_WEC": [],
        "L_spectral": [],
        "val_L_tensor": [],
        "val_L_P": [],
        "val_L_ortho": [],
        "val_L_WEC": [],
        "val_L_spectral": [],
        "lr": [],
    }

    metrics_csv_path = os.path.join(checkpoint_dir, "metrics.csv")
    history_file = os.path.join(checkpoint_dir, "history.json")

    # Restore history and metrics.csv synchronization
    if ckpt is not None:
        # 1. Try loading history from ckpt or history.json
        if "history" in ckpt and isinstance(ckpt["history"], dict):
            ckpt_hist: dict[str, Any] = ckpt["history"]
            for k in history:
                if k in ckpt_hist:
                    history[k] = [float(x) for x in list(ckpt_hist[k])[:start_epoch]]
            print(f"[Resume] Loaded history for {len(history['train_loss'])} epochs from checkpoint dictionary")
        elif os.path.isfile(history_file):
            try:
                with open(history_file, "r") as f:
                    h_data = json.load(f)
                for k in history:
                    if k in h_data and isinstance(h_data[k], list):
                        history[k] = [float(x) for x in list(h_data[k])[:start_epoch]]
                print(f"[Resume] Loaded history for {len(history['train_loss'])} epochs from {history_file}")
            except Exception as e:
                print(f"[Resume] Warning: could not load {history_file} ({e})")

        # 2. If history is incomplete, reconstruct from metrics.csv
        if len(history["train_loss"]) < start_epoch and os.path.isfile(metrics_csv_path):
            try:
                with open(metrics_csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    csv_rows = [row for row in reader if int(row.get("epoch", 0)) <= start_epoch]
                if len(csv_rows) > len(history["train_loss"]):
                    for k in history:
                        history[k] = []
                    for row in csv_rows:
                        for k in history:
                            if k in row and row[k] != "":
                                history[k].append(float(row[k]))
                    print(f"[Resume] Reconstructed history for {len(history['train_loss'])} epochs from {metrics_csv_path}")
            except Exception as e:
                print(f"[Resume] Warning: could not parse {metrics_csv_path} ({e})")

        # 3. Synchronize metrics.csv: preserve rows up to start_epoch
        if os.path.isfile(metrics_csv_path):
            try:
                with open(metrics_csv_path, "r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    valid_rows = []
                    if header is not None:
                        epoch_idx = header.index("epoch") if "epoch" in header else 0
                        for r in reader:
                            if r and len(r) > epoch_idx:
                                try:
                                    if int(r[epoch_idx]) <= start_epoch:
                                        valid_rows.append(r)
                                except ValueError:
                                    pass
                with open(metrics_csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    if header:
                        writer.writerow(header)
                    writer.writerows(valid_rows)
                print(f"[Resume] Synchronized {metrics_csv_path} (preserved up to epoch {start_epoch})")
            except Exception as e:
                print(f"[Resume] Warning: could not synchronize {metrics_csv_path} ({e})")
        else:
            with open(metrics_csv_path, "w", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([
                    "epoch", "train_loss", "val_loss", "val_mse", "val_rel_l2",
                    "L_tensor", "L_P", "L_ortho", "L_WEC", "L_spectral",
                    "val_L_tensor", "val_L_P", "val_L_ortho", "val_L_WEC", "val_L_spectral",
                    "lr", "elapsed_seconds",
                ])
    else:
        # Fresh training run: initialize new CSV
        with open(metrics_csv_path, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "epoch", "train_loss", "val_loss", "val_mse", "val_rel_l2",
                "L_tensor", "L_P", "L_ortho", "L_WEC", "L_spectral",
                "val_L_tensor", "val_L_P", "val_L_ortho", "val_L_WEC", "val_L_spectral",
                "lr", "elapsed_seconds",
            ])

    epochs_to_run = epochs - start_epoch
    if epochs_to_run <= 0:
        print(f"\nTarget epochs ({epochs}) already reached or exceeded. Nothing to train.")
        return model, {}

    print(f"\nTraining from epoch {start_epoch + 1} to {epochs} (total: {epochs_to_run} epochs, plots saved every {plot_interval} epochs)...\n")

    pbar = tqdm(range(start_epoch + 1, epochs + 1), desc="Epochs", unit="epoch", initial=start_epoch, total=epochs)
    for epoch in pbar:
        t0 = time.time()

        train_res = train_epoch(
            model, train_loader, optimizer, criterion,
            inv_mean, inv_std, thermo_mean, thermo_std, device,
            g_cov=g_cov, dataset=train_ds, spectral_snapshots=spectral_snapshots,
        )
        val_metrics = validate(
            model, val_loader, criterion,
            inv_mean, inv_std, thermo_mean, thermo_std, device,
            g_cov=g_cov, dataset=val_ds,
        )

        train_loss = train_res["train_loss"]
        monitor_loss = val_metrics.get("val_loss", val_metrics["mse"])
        scheduler.step(monitor_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(monitor_loss))
        history["val_mse"].append(float(val_metrics["mse"]))
        history["val_rel_l2"].append(float(val_metrics["rel_l2"]))
        history["L_tensor"].append(float(train_res.get("L_tensor", 0.0)))
        history["L_P"].append(float(train_res.get("L_P", 0.0)))
        history["L_ortho"].append(float(train_res.get("L_ortho", 0.0)))
        history["L_WEC"].append(float(train_res.get("L_WEC", 0.0)))
        history["L_spectral"].append(float(train_res.get("L_spectral", 0.0)))
        history["val_L_tensor"].append(float(val_metrics.get("val_L_tensor", 0.0)))
        history["val_L_P"].append(float(val_metrics.get("val_L_P", 0.0)))
        history["val_L_ortho"].append(float(val_metrics.get("val_L_ortho", 0.0)))
        history["val_L_WEC"].append(float(val_metrics.get("val_L_WEC", 0.0)))
        history["val_L_spectral"].append(float(val_metrics.get("val_L_spectral", 0.0)))
        history["lr"].append(float(current_lr))

        # Update tqdm postfix with live training & validation metrics
        pbar.set_postfix({
            "loss": f"{train_loss:.2e}",
            "val": f"{monitor_loss:.2e}",
            "mse": f"{val_metrics['mse']:.2e}",
            "rel_l2": f"{val_metrics['rel_l2'] * 100:.1f}%",
            "L_P": f"{train_res.get('L_P', 0.0):.2e}",
            "L_spec": f"{train_res.get('L_spectral', 0.0):.2e}",
            "lr": f"{current_lr:.1e}",
        })

        # Log to CSV
        with open(metrics_csv_path, "a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                epoch,
                f"{train_loss:.8e}",
                f"{monitor_loss:.8e}",
                f"{val_metrics['mse']:.8e}",
                f"{val_metrics['rel_l2']:.6f}",
                f"{train_res.get('L_tensor', 0.0):.8e}",
                f"{train_res.get('L_P', 0.0):.8e}",
                f"{train_res.get('L_ortho', 0.0):.8e}",
                f"{train_res.get('L_WEC', 0.0):.8e}",
                f"{train_res.get('L_spectral', 0.0):.8e}",
                f"{val_metrics.get('val_L_tensor', 0.0):.8e}",
                f"{val_metrics.get('val_L_P', 0.0):.8e}",
                f"{val_metrics.get('val_L_ortho', 0.0):.8e}",
                f"{val_metrics.get('val_L_WEC', 0.0):.8e}",
                f"{val_metrics.get('val_L_spectral', 0.0):.8e}",
                f"{current_lr:.8e}",
                f"{elapsed:.2f}",
            ])

        # Current checkpoint snapshot
        current_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "early_stopping": {
                "counter": early_stopping.counter,
                "best_score": early_stopping.best_score,
                "best_epoch": early_stopping.best_epoch,
                "early_stop": early_stopping.early_stop,
            },
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "val_loss": monitor_loss,
            "val_mse": val_metrics["mse"],
            "val_rel_l2": val_metrics["rel_l2"],
            "loss_weights": {
                "alpha": loss_alpha,
                "beta": loss_beta,
                "gamma": loss_gamma,
                "delta": loss_delta,
                "lambda_stab": lambda_stab,
            },
            "inv_mean": inv_mean.cpu(),
            "inv_std": inv_std.cpu(),
            "thermo_mean": thermo_mean.cpu(),
            "thermo_std": thermo_std.cpu(),
            "hidden_dim": hidden_dim,
            "num_hidden_layers": num_hidden_layers,
            "num_invariants": 15,
            "num_thermo_inputs": 2,
            "num_basis_tensors": 12,
            "history": history,
        }

        # Save best checkpoint
        if monitor_loss < best_val_loss:
            best_val_loss = monitor_loss
            best_epoch = epoch
            current_checkpoint["best_val_loss"] = best_val_loss
            current_checkpoint["best_epoch"] = best_epoch
            torch.save(current_checkpoint, best_model_path)

        # Save latest checkpoint atomically every epoch for instant resumption
        latest_tmp_path = os.path.join(checkpoint_dir, "latest_model.pt.tmp")
        torch.save(current_checkpoint, latest_tmp_path)
        os.replace(latest_tmp_path, latest_model_path)

        # Check early stopping (triggers after 2x LR patience without improvement)
        if early_stopping.step(monitor_loss, epoch):
            tqdm.write(
                f"\n[EarlyStopping] Validation loss did not improve for {early_stopping.patience} "
                + f"consecutive epochs. Stopping training early at epoch {epoch} "
                + f"(best epoch was {early_stopping.best_epoch} with val loss {early_stopping.best_score:.6e})."
            )
            early_stopped = True
            stopped_epoch = epoch
            break

        # Periodic visualization of predictions vs ground truth and loss curves
        if epoch % plot_interval == 0 or epoch == start_epoch + 1 or epoch == epochs:
            plot_loss_curves(history, os.path.join(checkpoint_dir, "loss_history.png"))
            plot_pred_vs_truth(
                model, val_ds, inv_mean, inv_std, thermo_mean, thermo_std,
                device, os.path.join(plots_dir, f"epoch_{epoch:04d}_pred_vs_truth.png"),
                epoch=epoch, grid_size=coarse_grid,
            )
            plot_pred_vs_truth(
                model, val_ds, inv_mean, inv_std, thermo_mean, thermo_std,
                device, os.path.join(checkpoint_dir, "latest_pred_vs_truth.png"),
                epoch=epoch, grid_size=coarse_grid,
            )
            tqdm.write(f"  → [Plots updated: {plots_dir}/epoch_{epoch:04d}_pred_vs_truth.png and {checkpoint_dir}/loss_history.png]")

    final_epoch = stopped_epoch if early_stopped else epochs

    # Ensure final plots are updated up to the completed / stopped epoch
    plot_loss_curves(history, os.path.join(checkpoint_dir, "loss_history.png"))
    plot_pred_vs_truth(
        model, val_ds, inv_mean, inv_std, thermo_mean, thermo_std,
        device, os.path.join(plots_dir, f"epoch_{final_epoch:04d}_pred_vs_truth.png"),
        epoch=final_epoch, grid_size=coarse_grid,
    )
    plot_pred_vs_truth(
        model, val_ds, inv_mean, inv_std, thermo_mean, thermo_std,
        device, os.path.join(checkpoint_dir, "latest_pred_vs_truth.png"),
        epoch=final_epoch, grid_size=coarse_grid,
    )
    tqdm.write(f"  → [Final plots updated for epoch {final_epoch}: {plots_dir}/epoch_{final_epoch:04d}_pred_vs_truth.png and {checkpoint_dir}/loss_history.png]")

    # Save final checkpoint and history
    torch.save({
        "epoch": final_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "early_stopping": {
            "counter": early_stopping.counter,
            "best_score": early_stopping.best_score,
            "best_epoch": early_stopping.best_epoch,
            "early_stop": early_stopping.early_stop,
        },
        "early_stopped": early_stopped,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "loss_weights": {
            "alpha": loss_alpha,
            "beta": loss_beta,
            "gamma": loss_gamma,
            "delta": loss_delta,
            "lambda_stab": lambda_stab,
        },
        "inv_mean": inv_mean.cpu(),
        "inv_std": inv_std.cpu(),
        "thermo_mean": thermo_mean.cpu(),
        "thermo_std": thermo_std.cpu(),
        "hidden_dim": hidden_dim,
        "num_hidden_layers": num_hidden_layers,
        "num_invariants": 15,
        "num_thermo_inputs": 2,
        "num_basis_tensors": 12,
        "history": history,
    }, os.path.join(checkpoint_dir, "final_model.pt"))

    history_meta = {
        **history,
        "early_stopped": early_stopped,
        "stopped_epoch": final_epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "loss_alpha": loss_alpha,
        "loss_beta": loss_beta,
        "loss_gamma": loss_gamma,
        "loss_delta": loss_delta,
        "lambda_stab": lambda_stab,
        "patience": early_stopping.patience,
        "lr_patience": lr_patience,
    }
    with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
        json.dump(history_meta, f, indent=2)

    # Restore best checkpoint weights before TorchScript export
    if restore_best_weights and os.path.exists(best_model_path):
        best_ckpt = torch.load(best_model_path, weights_only=False, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        print(f"\nRestored best model weights from epoch {best_ckpt.get('epoch', best_epoch)} "
              + f"(val loss: {best_ckpt.get('val_loss', best_val_loss):.6e}) for TorchScript export.")

    # Export TorchScript
    model.eval()
    sample_inv = torch.randn(1, 15, dtype=dtype, device=device)
    sample_thermo = torch.randn(1, 2, dtype=dtype, device=device)
    sample_basis = torch.randn(1, 12, 4, 4, dtype=dtype, device=device)
    traced = torch.jit.trace(model, (sample_inv, sample_thermo, sample_basis))
    traced.save(os.path.join(checkpoint_dir, "tbnn_traced.pt"))
    print(f"TorchScript model saved to {checkpoint_dir}/tbnn_traced.pt")

    status_str = f"early stopped at epoch {final_epoch}" if early_stopped else f"completed {epochs} epochs"
    print(f"\nTraining {status_str}. Best val loss: {best_val_loss:.6e} (at epoch {best_epoch})")
    return model, history_meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRMHD-TBNN (a priori) with physics-based loss")
    parser.add_argument("--data-dir", type=str, default="test1")
    parser.add_argument("--coarsening-factor", type=int, default=8)
    parser.add_argument("--stride", type=int, default=5, help="Snapshot sampling stride")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--loss-alpha", type=float, default=0.5,
                        help="Weight alpha for dissipation rate loss L_P (default: 0.5)")
    parser.add_argument("--loss-beta", type=float, default=0.1,
                        help="Weight beta for spectral cascade loss L_spectral (default: 0.1)")
    parser.add_argument("--loss-gamma", type=float, default=5.0,
                        help="Weight gamma for orthogonality constraint loss L_ortho (default: 5.0)")
    parser.add_argument("--loss-delta", type=float, default=2.0,
                        help="Weight delta for Weak Energy Condition loss L_WEC (default: 2.0)")
    parser.add_argument("--lambda-stab", type=float, default=10.0,
                        help="Anti-dissipation stability barrier weight in L_P (default: 10.0)")
    parser.add_argument("--spectral-snapshots", type=int, default=4,
                        help="Number of snapshots per epoch for spectral cascade loss (default: 4)")
    parser.add_argument("--lr-patience", type=int, default=25,
                        help="Patience in epochs for ReduceLROnPlateau scheduler (default: 25)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience in epochs (default: 2x lr_patience, or <=0 to disable)")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Minimum change in validation loss to qualify as improvement (default: 0.0)")
    parser.add_argument("--no-restore-best", action="store_true",
                        help="Do not restore best checkpoint weights before exporting TorchScript and final model")
    parser.add_argument("--plot-interval", type=int, default=10, help="Interval in epochs to save diagnostics plots")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float64",
                        choices=["float32", "float64"])
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Resume training. If passed without value (--resume), loads latest_model.pt or best_model.pt "
            + "from --checkpoint-dir. If a path is provided (--resume path/to/model.pt), loads that checkpoint."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Explicit path to checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--reset-lr",
        action="store_true",
        help="When resuming, override the checkpoint's saved optimizer learning rate with the value of --lr.",
    )
    parser.add_argument(
        "--reset-best",
        action="store_true",
        help=(
            "When resuming, discard the checkpoint's best_val_loss / scheduler baseline / early-stopping best "
            + "and start fresh. Use this when resuming onto a harder task where the old run's loss values "
            + "aren't a meaningful comparison target (e.g. using the checkpoint only as a weight initialization)."
        ),
    )
    args = parser.parse_args()

    resume_arg = args.resume_from if args.resume_from is not None else args.resume

    train(
        data_dir=args.data_dir,
        coarsening_factor=args.coarsening_factor,
        stride=args.stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_patience=args.lr_patience,
        patience=args.patience,
        min_delta=args.min_delta,
        restore_best_weights=not args.no_restore_best,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        loss_alpha=args.loss_alpha,
        loss_beta=args.loss_beta,
        loss_gamma=args.loss_gamma,
        loss_delta=args.loss_delta,
        lambda_stab=args.lambda_stab,
        spectral_snapshots=args.spectral_snapshots,
        plot_interval=args.plot_interval,
        checkpoint_dir=args.checkpoint_dir,
        device_str=args.device,
        dtype_str=args.dtype,
        resume=resume_arg,
        reset_lr=args.reset_lr,
        reset_best=args.reset_best,
    )
