r"""
evaluate_rollout.py

Benchmarking and diagnostic evaluation for the CRMHD-TBNN SGS closure.

Compares simulation trajectories on the coarse grid:
  1. TBNN-closed simulation — solver + CRMHD_TBNN closure
  2. Exact DNS-closed simulation — oracle SGS stress tensor from filtered DNS
  3. Filtered high-res benchmark — true Favre-filtered DNS states

Diagnostic metrics:
  - Relative L2 error on density and velocity fields
  - Kinetic energy power spectrum E(k)
  - Time evolution and rollout animations
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
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import CRMHD_TBNN
from src.dataset import (
    read_snapshot, relativistic_favre_filter, GAMMA_EOS, compute_sgs_stress,
)
from src.features import extract_tbnn_features
from src.solver import RHDSolver, compute_transition_sensor, clip_sgs_tensor


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def kinetic_energy_spectrum(
    rho: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 2D isotropic kinetic energy power spectrum E(k).

    Parameters
    ----------
    rho, vx, vy : (Ny, Nx) tensors

    Returns
    -------
    k_bins : (N_bins,) array — wavenumber bin centres
    E_k    : (N_bins,) array — energy per bin
    """
    Ny, Nx = rho.shape

    # Momentum in Fourier space
    sx = (rho * vx).numpy()
    sy = (rho * vy).numpy()

    sx_hat = np.fft.fft2(sx)
    sy_hat = np.fft.fft2(sy)

    # Kinetic energy density in k-space: 0.5 * |ŝ|² / ρ̄
    rho_mean = rho.mean().item()
    E_hat = 0.5 * (np.abs(sx_hat)**2 + np.abs(sy_hat)**2) / (rho_mean * Nx * Ny)

    # Isotropic binning
    kx = np.fft.fftfreq(Nx, d=1.0/Nx)
    ky = np.fft.fftfreq(Ny, d=1.0/Ny)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    k_max = int(min(Nx, Ny) / 2)
    k_bins = np.arange(1, k_max + 1, dtype=float)
    # ponytail: vectorised binning replaces per-bin Python loop
    bin_idx = np.digitize(K.ravel(), k_bins - 0.5) - 1
    valid = (bin_idx >= 0) & (bin_idx < k_max)
    E_k = np.bincount(bin_idx[valid], weights=E_hat.ravel()[valid], minlength=k_max)[:k_max].astype(float)

    return k_bins, E_k


def relative_l2_error(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Compute relative L2 error between two 2D fields."""
    err = torch.norm(pred - target).item()
    ref = torch.norm(target).item()
    return err / max(ref, 1e-15)


from collections.abc import Mapping

def create_rollout_animation(
    history_dns: Mapping[str, object],
    history_tbnn: Mapping[str, object] | None = None,
    history_exact: Mapping[str, object] | None = None,
    output_path: str = "results/rollout_comparison.gif",
    fps: int = 10,
) -> str:
    """Generate a synchronized 4-panel animation comparing DNS, TBNN, and exact models alongside error."""
    times = np.asarray(history_dns["t"])
    N_frames = len(times)
    if N_frames == 0:
        print("[ANIM] No frames to animate.")
        return output_path

    rho_dns_all = history_dns["rho"]
    assert isinstance(rho_dns_all, torch.Tensor)

    rho_tbnn_all: torch.Tensor | None = None
    if history_tbnn is not None and "rho" in history_tbnn:
        tbnn_val = history_tbnn["rho"]
        assert isinstance(tbnn_val, torch.Tensor)
        rho_tbnn_all = tbnn_val

    rho_exact_all: torch.Tensor | None = None
    if history_exact is not None and "rho" in history_exact:
        exact_val = history_exact["rho"]
        assert isinstance(exact_val, torch.Tensor)
        rho_exact_all = exact_val

    # Determine global vmin, vmax across all density models and timestamps
    all_rho: list[torch.Tensor] = [rho_dns_all]
    if rho_tbnn_all is not None:
        all_rho.append(rho_tbnn_all)
    if rho_exact_all is not None:
        all_rho.append(rho_exact_all)

    vmin = min(float(t.min()) for t in all_rho)
    vmax = max(float(t.max()) for t in all_rho)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.subplots_adjust(left=0.06, right=0.88, top=0.92, bottom=0.06, wspace=0.18, hspace=0.22)
    ax_dns, ax_tbnn = axes[0, 0], axes[0, 1]
    ax_exact, ax_err = axes[1, 0], axes[1, 1]

    im_dns = ax_dns.imshow(
        rho_dns_all[0].numpy(), origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal"
    )
    ax_dns.set_title("Filtered DNS Ground Truth", fontsize=11, fontweight="bold")

    im_tbnn = None
    if rho_tbnn_all is not None:
        im_tbnn = ax_tbnn.imshow(
            rho_tbnn_all[0].numpy(), origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal"
        )
        err_t0 = relative_l2_error(rho_tbnn_all[0], rho_dns_all[0])
        ax_tbnn.set_title(f"TBNN-Closed Solver\nrel L2: {err_t0:.3f}", fontsize=11, fontweight="bold")
    else:
        ax_tbnn.text(0.5, 0.5, "TBNN Disabled\n(No Checkpoint)", ha="center", va="center", transform=ax_tbnn.transAxes, fontsize=12, color="gray")
        ax_tbnn.axis("off")

    im_exact = None
    if rho_exact_all is not None:
        im_exact = ax_exact.imshow(
            rho_exact_all[0].numpy(), origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal"
        )
        err_e0 = relative_l2_error(rho_exact_all[0], rho_dns_all[0])
        ax_exact.set_title(f"Exact DNS-Closed Solver\nrel L2: {err_e0:.3f}", fontsize=11, fontweight="bold")
    else:
        ax_exact.text(0.5, 0.5, "Exact Solver Skipped", ha="center", va="center", transform=ax_exact.transAxes, fontsize=12, color="gray")
        ax_exact.axis("off")

    im_err = None
    target_err_all = rho_tbnn_all if rho_tbnn_all is not None else rho_exact_all
    err_title = "|TBNN - DNS| Absolute Error" if rho_tbnn_all is not None else "|Exact - DNS| Absolute Error"
    if target_err_all is not None:
        err0 = (target_err_all[0] - rho_dns_all[0]).abs().numpy()
        max_err_val = max(0.5, float((target_err_all - rho_dns_all).abs().max()))
        im_err = ax_err.imshow(
            err0, origin="lower", cmap="magma", vmin=0.0, vmax=max_err_val, aspect="equal"
        )
        ax_err.set_title(err_title, fontsize=11, fontweight="bold")
    else:
        ax_err.text(0.5, 0.5, "No Model Available", ha="center", va="center", transform=ax_err.transAxes, fontsize=12, color="gray")
        ax_err.axis("off")

    # Shared colorbar for density
    cbar_ax = fig.add_axes((0.90, 0.15, 0.02, 0.70))
    cbar = fig.colorbar(im_dns, cax=cbar_ax)
    cbar.set_label("Rest-frame Density ρ̄", fontsize=11)

    pbar_anim = tqdm(total=N_frames, desc="Rendering animation frames", unit="frame", dynamic_ncols=True, file=sys.stdout)

    def update(frame_idx: int):
        t_val = float(times[frame_idx])
        fig.suptitle(f"Relativistic Hydrodynamics Rollout Comparison  (t = {t_val:.2f})", fontsize=14, fontweight="bold")

        rho_dns_k = rho_dns_all[frame_idx]
        im_dns.set_data(rho_dns_k.numpy())

        if im_tbnn is not None and rho_tbnn_all is not None:
            rho_tbnn_k = rho_tbnn_all[frame_idx]
            err_t = relative_l2_error(rho_tbnn_k, rho_dns_k)
            im_tbnn.set_data(rho_tbnn_k.numpy())
            ax_tbnn.set_title(f"TBNN-Closed Solver\nrel L2: {err_t:.3f}", fontsize=11, fontweight="bold")

        if im_exact is not None and rho_exact_all is not None:
            rho_exact_k = rho_exact_all[frame_idx]
            err_e = relative_l2_error(rho_exact_k, rho_dns_k)
            im_exact.set_data(rho_exact_k.numpy())
            ax_exact.set_title(f"Exact DNS-Closed Solver\nrel L2: {err_e:.3f}", fontsize=11, fontweight="bold")

        if im_err is not None and target_err_all is not None:
            err_map_k = (target_err_all[frame_idx] - rho_dns_k).abs().numpy()
            im_err.set_data(err_map_k)

        pbar_anim.update(1)
        return [im_dns]

    ani = anim.FuncAnimation(fig, update, frames=N_frames, interval=int(1000 / fps), blit=False)
    writer = anim.PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    pbar_anim.close()
    plt.close(fig)
    print(f"\n[ANIM] Animation successfully saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# SGS Model Wrapper for Solver Coupling
# ---------------------------------------------------------------------------

class TBNNSGSModel:
    """Wraps a trained CRMHD_TBNN model for injection into the RHD solver.

    At each solver step, this callable:
      1. Extracts TBNN features from current primitives
      2. Runs the neural network forward pass
      3. Converts the predicted ΔT_SGS into a conservative source term

    Parameters
    ----------
    model : CRMHD_TBNN — trained model
    inv_mean, inv_std : normalization statistics for invariants
    thermo_mean, thermo_std : normalization statistics for thermo inputs
    dx, dy : grid spacings
    """

    def __init__(
        self,
        model: CRMHD_TBNN,
        inv_mean: torch.Tensor,
        inv_std: torch.Tensor,
        thermo_mean: torch.Tensor,
        thermo_std: torch.Tensor,
        dx: float,
        dy: float,
    ):
        self.model = model
        self.model.eval()
        # ponytail: float32 inference — halves bandwidth, enables tensor cores
        self.model_f32 = model.float()
        self.inv_mean = inv_mean.float()
        self.inv_std = inv_std.float()
        self.thermo_mean = thermo_mean.float()
        self.thermo_std = thermo_std.float()
        self.dx = dx
        self.dy = dy
        # ponytail: compile the forward path for kernel fusion (CUDA only)
        # Guard: test actual device access and tensor allocation, not just driver presence
        _cuda_ok = False
        try:
            model_dev = next(model.parameters()).device
            if model_dev.type == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() > 0:
                _test_t = torch.zeros(1, device=model_dev)
                del _test_t
                _cuda_ok = True
        except Exception:
            _cuda_ok = False

        if _cuda_ok:
            try:
                self._compiled_forward = torch.compile(
                    self._forward_impl,
                    mode="default",
                    dynamic=False,
                )
            except Exception:
                self._compiled_forward = self._forward_impl
        else:
            self._compiled_forward = self._forward_impl

    @torch.no_grad()
    def _forward_impl(self, prims: torch.Tensor) -> torch.Tensor:
        """Core forward: features → model → source term. Compiled target."""
        rho = prims[..., 0]
        vx = prims[..., 1]
        vy = prims[..., 2]
        p = prims[..., 3]

        # Extract features (stays in prims dtype = float64)
        features = extract_tbnn_features(
            rho, vx, vy, p, self.dx, self.dy, gradient_order=4,
        )

        Ny, Nx = rho.shape
        N = Ny * Nx

        # Flatten for batch processing + cast to float32
        inv = features["invariants"].reshape(N, 15).float()
        thermo = features["thermo_inputs"].reshape(N, 2).float()
        basis = features["basis_tensors"].reshape(N, 12, 4, 4).float()

        # Normalize
        inv_norm = (inv - self.inv_mean) / self.inv_std
        thermo_norm = (thermo - self.thermo_mean) / self.thermo_std

        # Forward pass in float32
        delta_T, _, _ = self.model_f32(inv_norm, thermo_norm, basis)
        # Cast back to float64 for solver
        delta_T = delta_T.to(prims.dtype).reshape(Ny, Nx, 4, 4)

        # Convert T^{μν}_SGS divergence to conservative source term:
        # S^ν = -∂_μ τ_SGS^{μν}
        # 4th-order central: (-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / 12Δ
        # (matches the gradient order used in feature extraction)

        # ∂_x τ_SGS^{1ν}
        tau_1nu = delta_T[:, :, 1, :]  # (Ny, Nx, 4)
        dtau_x = (-torch.roll(tau_1nu, -2, dims=1)
                  + 8.0 * torch.roll(tau_1nu, -1, dims=1)
                  - 8.0 * torch.roll(tau_1nu, 1, dims=1)
                  + torch.roll(tau_1nu, 2, dims=1)) / (12.0 * self.dx)
        # ∂_y τ_SGS^{2ν}
        tau_2nu = delta_T[:, :, 2, :]  # (Ny, Nx, 4)
        dtau_y = (-torch.roll(tau_2nu, -2, dims=0)
                  + 8.0 * torch.roll(tau_2nu, -1, dims=0)
                  - 8.0 * torch.roll(tau_2nu, 1, dims=0)
                  + torch.roll(tau_2nu, 2, dims=0)) / (12.0 * self.dy)

        div_tau = dtau_x + dtau_y  # (Ny, Nx, 4)

        # Source = -∂_μ τ^{μν}
        # Map: ν=0 → τ (energy), ν=1 → Sx, ν=2 → Sy
        source = torch.zeros(Ny, Nx, 4, dtype=prims.dtype, device=prims.device)
        source[..., 0] = 0.0       # D (mass) — no SGS source for continuity
        source[..., 1] = -div_tau[..., 1]  # Sx source
        source[..., 2] = -div_tau[..., 2]  # Sy source
        source[..., 3] = -div_tau[..., 0]  # τ (energy) source from T^{0ν}

        # Realizability clipping: prevent excessive backscatter
        # Use per-cell local pressure to avoid amputating physics in
        # strong-gradient regions (previous 10%-of-mean was too aggressive).
        max_source = 0.5 * prims[..., 3].abs().unsqueeze(-1)  # 50% of local pressure
        source = source.clamp(-max_source, max_source)

        return source

    @torch.no_grad()
    def __call__(self, prims: torch.Tensor) -> torch.Tensor:
        """Compute SGS source term from primitive variables.

        Parameters
        ----------
        prims : (Ny, Nx, 4) tensor — [ρ, vx, vy, p]

        Returns
        -------
        source : (Ny, Nx, 4) tensor — source term in conservative variables
        """
        try:
            return self._compiled_forward(prims)
        except Exception:
            self._compiled_forward = self._forward_impl
            return self._forward_impl(prims)


class TBNNSGSClosure:
    r"""Frame-consistent TBNN closure for the conservative viscous flux formulation.

    Returns the raw \tau_\perp^{\mu\nu} tensor (clipped for realizability)
    plus a transition sensor \alpha(x,y) for modulation, instead of a
    pre-computed source term.

    This is used with ``RHDSolver.step(sgs_closure=...)`` to enable the
    conservative flux coupling that properly captures \partial_t \tau^{0\nu}.

    Parameters
    ----------
    model : CRMHD_TBNN — trained model
    inv_mean, inv_std : normalization statistics for invariants
    thermo_mean, thermo_std : normalization statistics for thermo inputs
    dx, dy : grid spacings
    sensor_mode : str — "none" (default; α ≡ 1), "enstrophy", or "germano"
    clip_mode : str — "frobenius" (default) or "eigenvalue"
    clip_kappa : float — clipping threshold as fraction of local pressure
    """

    def __init__(
        self,
        model: CRMHD_TBNN,
        inv_mean: torch.Tensor,
        inv_std: torch.Tensor,
        thermo_mean: torch.Tensor,
        thermo_std: torch.Tensor,
        dx: float,
        dy: float,
        sensor_mode: str = "none",
        clip_mode: str = "frobenius",
        clip_kappa: float = 0.5,
    ):
        self.model = model
        self.model.eval()
        self.model_f32 = model.float()
        self.inv_mean = inv_mean.float()
        self.inv_std = inv_std.float()
        self.thermo_mean = thermo_mean.float()
        self.thermo_std = thermo_std.float()
        self.dx = dx
        self.dy = dy
        self.sensor_mode = sensor_mode
        self.clip_mode = clip_mode
        self.clip_kappa = clip_kappa

    @torch.no_grad()
    def __call__(self, prims: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute SGS stress tensor and transition sensor.

        Parameters
        ----------
        prims : (Ny, Nx, 4) tensor — [ρ, vx, vy, p]

        Returns
        -------
        tau_perp : (Ny, Nx, 4, 4) tensor — clipped, sensor-unmodulated SGS stress
        alpha : (Ny, Nx) tensor — transition sensor ∈ [0, 1]
        """
        rho = prims[..., 0]
        vx = prims[..., 1]
        vy = prims[..., 2]
        p = prims[..., 3]

        # Extract features
        features = extract_tbnn_features(
            rho, vx, vy, p, self.dx, self.dy, gradient_order=4,
        )

        Ny, Nx = rho.shape
        N = Ny * Nx

        inv = features["invariants"].reshape(N, 15).float()
        thermo = features["thermo_inputs"].reshape(N, 2).float()
        basis = features["basis_tensors"].reshape(N, 12, 4, 4).float()

        inv_norm = (inv - self.inv_mean) / self.inv_std
        thermo_norm = (thermo - self.thermo_mean) / self.thermo_std

        # Forward pass → τ_⊥^{μν}
        delta_T, _, _ = self.model_f32(inv_norm, thermo_norm, basis)
        tau_perp = delta_T.to(prims.dtype).reshape(Ny, Nx, 4, 4)

        # Realizability clipping
        tau_perp = clip_sgs_tensor(
            tau_perp, prims, mode=self.clip_mode, kappa=self.clip_kappa,
        )

        # Transition sensor
        alpha = compute_transition_sensor(
            prims, self.dx, self.dy, mode=self.sensor_mode,
        )

        return tau_perp, alpha


# ---------------------------------------------------------------------------
# Exact DNS SGS Model Wrapper (Oracle / Ground-Truth Benchmark)
# ---------------------------------------------------------------------------

class ExactDNSSGSModel:
    r"""Exact DNS subgrid-scale closure wrapper for the coarse RHD solver.

    Precomputes the exact Favre-filtered subgrid-scale stress tensor τ_SGS^{μν}(t_k)
    directly from high-resolution DNS snapshots:
        τ_SGS^{μν} = \overline{T^{μν}} - T^{μν}(\bar{U})
    projected into the comoving fluid frame (Option A).

    During coarse simulation, linearly interpolates τ_SGS(t) in time between available
    DNS snapshots, and evaluates the conservative source term:
        S^ν = -∂_μ τ_SGS^{μν}
    """

    def __init__(
        self,
        snapshots: list[dict],
        coarsening_factor: int,
        dx: float,
        dy: float,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
        clip_fraction: float | None = 0.5,
    ):
        self.dx = dx
        self.dy = dy
        self.device = device
        self.dtype = dtype
        self.clip_fraction = clip_fraction

        sorted_snaps = sorted(snapshots, key=lambda s: s["t"])
        self.times = [float(s["t"]) for s in sorted_snaps]
        self.times_np = np.array(self.times, dtype=np.float64)

        self.taus: list[torch.Tensor] = []
        print(f"  Precomputing exact DNS SGS stress tensors for {len(sorted_snaps)} snapshots...")
        for s in sorted_snaps:
            rho = torch.tensor(s["rho"], dtype=dtype, device=device)
            vx = torch.tensor(s["vx"], dtype=dtype, device=device)
            vy = torch.tensor(s["vy"], dtype=dtype, device=device)
            p = torch.tensor(s["p"], dtype=dtype, device=device)

            _, tau = compute_sgs_stress(
                rho, vx, vy, p, r=coarsening_factor, project_to_fluid_frame=True,
            )
            self.taus.append(tau)
            del rho, vx, vy, p

        self.current_t = self.times[0]

    def set_time(self, t: float) -> None:
        """Update the current simulation time."""
        self.current_t = float(t)

    @torch.no_grad()
    def __call__(self, prims: torch.Tensor) -> torch.Tensor:
        """Compute the exact SGS source term interpolated to current simulation time."""
        t = self.current_t
        if t <= self.times[0]:
            tau = self.taus[0]
        elif t >= self.times[-1]:
            tau = self.taus[-1]
        else:
            idx = int(np.searchsorted(self.times_np, t)) - 1
            idx = max(0, min(idx, len(self.times) - 2))
            t0 = self.times[idx]
            t1 = self.times[idx + 1]
            dt = t1 - t0
            alpha = (t - t0) / dt if dt > 1e-14 else 0.0
            tau = (1.0 - alpha) * self.taus[idx] + alpha * self.taus[idx + 1]

        Ny, Nx = prims.shape[:2]

        # 4th-order central: (-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / 12Δ
        tau_1nu = tau[:, :, 1, :]  # (Ny, Nx, 4)
        dtau_x = (-torch.roll(tau_1nu, -2, dims=1)
                  + 8.0 * torch.roll(tau_1nu, -1, dims=1)
                  - 8.0 * torch.roll(tau_1nu, 1, dims=1)
                  + torch.roll(tau_1nu, 2, dims=1)) / (12.0 * self.dx)
        tau_2nu = tau[:, :, 2, :]  # (Ny, Nx, 4)
        dtau_y = (-torch.roll(tau_2nu, -2, dims=0)
                  + 8.0 * torch.roll(tau_2nu, -1, dims=0)
                  - 8.0 * torch.roll(tau_2nu, 1, dims=0)
                  + torch.roll(tau_2nu, 2, dims=0)) / (12.0 * self.dy)

        div_tau = dtau_x + dtau_y

        source = torch.zeros(Ny, Nx, 4, dtype=prims.dtype, device=prims.device)
        source[..., 0] = 0.0
        source[..., 1] = -div_tau[..., 1]
        source[..., 2] = -div_tau[..., 2]
        source[..., 3] = -div_tau[..., 0]

        if self.clip_fraction is not None:
            max_source = self.clip_fraction * prims[..., 3].abs().unsqueeze(-1)  # per-cell local pressure
            source = source.clamp(-max_source, max_source)

        return source


class ExactDNSSGSClosure:
    r"""Frame-consistent exact DNS closure for the conservative viscous flux path.

    Same interface as TBNNSGSClosure: returns (tau_perp, alpha) for use with
    ``RHDSolver.step(sgs_closure=...)``.

    Parameters
    ----------
    snapshots : list of DNS snapshot dicts
    coarsening_factor : int
    dx, dy : grid spacings
    device : torch.device
    dtype : torch.dtype
    sensor_mode : str — "none" (default; α ≡ 1), "enstrophy", or "germano"
    clip_mode : str — "frobenius" (default) or "eigenvalue"
    clip_kappa : float — clipping threshold
    """

    def __init__(
        self,
        snapshots: list[dict],
        coarsening_factor: int,
        dx: float,
        dy: float,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
        sensor_mode: str = "none",
        clip_mode: str = "frobenius",
        clip_kappa: float = 0.5,
    ):
        self.dx = dx
        self.dy = dy
        self.device = device
        self.dtype = dtype
        self.sensor_mode = sensor_mode
        self.clip_mode = clip_mode
        self.clip_kappa = clip_kappa

        sorted_snaps = sorted(snapshots, key=lambda s: s["t"])
        self.times = [float(s["t"]) for s in sorted_snaps]
        self.times_np = np.array(self.times, dtype=np.float64)

        self.taus: list[torch.Tensor] = []
        print(f"  [ExactDNSSGSClosure] Precomputing SGS tensors for {len(sorted_snaps)} snapshots...")
        for s in sorted_snaps:
            rho = torch.tensor(s["rho"], dtype=dtype, device=device)
            vx = torch.tensor(s["vx"], dtype=dtype, device=device)
            vy = torch.tensor(s["vy"], dtype=dtype, device=device)
            p = torch.tensor(s["p"], dtype=dtype, device=device)

            _, tau = compute_sgs_stress(
                rho, vx, vy, p, r=coarsening_factor, project_to_fluid_frame=True,
            )
            self.taus.append(tau)
            del rho, vx, vy, p

        self.current_t = self.times[0]

    def set_time(self, t: float) -> None:
        """Update the current simulation time."""
        self.current_t = float(t)

    @torch.no_grad()
    def __call__(self, prims: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute time-interpolated SGS tensor and transition sensor.

        Returns
        -------
        tau_perp : (Ny, Nx, 4, 4) — clipped SGS stress tensor
        alpha : (Ny, Nx) — transition sensor ∈ [0, 1]
        """
        t = self.current_t
        if t <= self.times[0]:
            tau = self.taus[0]
        elif t >= self.times[-1]:
            tau = self.taus[-1]
        else:
            idx = int(np.searchsorted(self.times_np, t)) - 1
            idx = max(0, min(idx, len(self.times) - 2))
            t0 = self.times[idx]
            t1 = self.times[idx + 1]
            dt_span = t1 - t0
            frac = (t - t0) / dt_span if dt_span > 1e-14 else 0.0
            tau = (1.0 - frac) * self.taus[idx] + frac * self.taus[idx + 1]

        # Realizability clipping
        tau = clip_sgs_tensor(
            tau, prims, mode=self.clip_mode, kappa=self.clip_kappa,
        )

        # Transition sensor
        alpha = compute_transition_sensor(
            prims, self.dx, self.dy, mode=self.sensor_mode,
        )

        return tau, alpha


# ---------------------------------------------------------------------------
# Rollout Evaluation Pipeline
# ---------------------------------------------------------------------------

def evaluate_rollout(
    data_dir: str = "test1",
    coarsening_factor: int = 8,
    checkpoint_path: str = "checkpoints/best_model.pt",
    t_start: float = 0.0,
    t_end: float = 10.0,
    cfl: float = 0.3,
    reconstruction: str = "mp5",
    riemann_solver: str = "hll",
    output_dir: str = "results",
    device_str: str = "cuda",
    skip_exact: bool = False,
    save_history: bool = False,
    make_anim: bool = False,
    anim_interval: float = 0.1,
    anim_fps: int = 10,
    sgs_coupling: str = "source",
):
    """Run a comparative rollout evaluation.

    Compares:
      1. TBNN-closed coarse solver
      2. Exact DNS-closed coarse solver (oracle)
      3. Filtered DNS ground truth

    Parameters
    ----------
    data_dir : directory with HDF5 snapshots
    coarsening_factor : downsampling factor r
    checkpoint_path : path to trained model checkpoint
    t_start, t_end : time range for rollout
    cfl : CFL number
    reconstruction : str — "mp5" (default) or "muscl"
    riemann_solver : str — "hll" (default), "musta_hll", or "hllc"
    output_dir : directory for output plots/data
    device_str : 'cuda' or 'cpu'
    skip_exact : bool — skip running exact DNS SGS solver
    save_history : bool — save full trajectory arrays to rollout_trajectory.pt
    make_anim : bool — generate animated comparison GIF
    anim_interval : float — simulation time interval between frames
    anim_fps : int — animation playback frames per second
    sgs_coupling : str — "source" (legacy, default) or "flux" (conservative
        viscous flux formulation with transition sensor)
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    r = coarsening_factor
    os.makedirs(output_dir, exist_ok=True)
    save_int = anim_interval if (save_history or make_anim) else None

    # Load all snapshots and identify start/end
    files = sorted(glob.glob(os.path.join(data_dir, "h*.h5")))
    snapshots = []
    pbar_snaps = tqdm(files, desc="Scanning DNS snapshots", unit="snap", dynamic_ncols=True, file=sys.stdout)
    for f in pbar_snaps:
        snap = read_snapshot(f)
        if t_start - 0.1 <= snap["t"] <= t_end + 0.1:
            snapshots.append(snap)
            pbar_snaps.set_postfix(matched=len(snapshots), t=f"{snap['t']:.2f}")

    if len(snapshots) < 2:
        raise ValueError(f"Need at least 2 snapshots in [{t_start}, {t_end}]")

    # Get grid dimensions from first snapshot
    Ny_fine, Nx_fine = snapshots[0]["rho"].shape
    Nc_y, Nc_x = Ny_fine // r, Nx_fine // r
    dx = 1.0 / Nc_x
    dy = 1.0 / Nc_y

    print(f"\nGrid: {Ny_fine}x{Nx_fine} -> {Nc_y}x{Nc_x} (r={r})")
    print(f"Time range: [{t_start}, {t_end}] ({len(snapshots)} DNS snapshots)")
    print(f"Solver: CFL={cfl}, recon={reconstruction}, riemann={riemann_solver}, device={device}")
    if save_int is not None:
        print(f"Trajectory history: enabled (interval={save_int:.3f} sim_s)")
    if make_anim:
        print(f"Animation: enabled (fps={anim_fps})")

    # Filter initial condition at t_start
    snap0 = min(snapshots, key=lambda s: abs(s["t"] - t_start))
    rho0 = torch.tensor(snap0["rho"], dtype=dtype)
    vx0 = torch.tensor(snap0["vx"], dtype=dtype)
    vy0 = torch.tensor(snap0["vy"], dtype=dtype)
    p0 = torch.tensor(snap0["p"], dtype=dtype)

    coarse0 = relativistic_favre_filter(rho0, vx0, vy0, p0, r)

    # --- Solver 1: TBNN-closed (if checkpoint exists) ---
    prims_tbnn = None
    history_tbnn = None
    has_checkpoint = os.path.exists(checkpoint_path)

    if has_checkpoint:
        print("\n--- Running TBNN-closed coarse solver ---")
        model, ckpt = CRMHD_TBNN.from_checkpoint(
            checkpoint_path, device=device, dtype=dtype,
        )
        model.eval()

        solver_tbnn = RHDSolver(
            Nc_y, Nc_x, cfl=cfl, reconstruction=reconstruction,
            riemann_solver=riemann_solver, dtype=dtype, device=device,
        )
        solver_tbnn.initialize(
            coarse0["rho_bar"].to(device),
            coarse0["vx_tilde"].to(device),
            coarse0["vy_tilde"].to(device),
            coarse0["p_bar"].to(device),
        )
        solver_tbnn.t = snap0["t"]

        if sgs_coupling == "flux":
            print("  Coupling: conservative viscous flux formulation")
            tbnn_closure = TBNNSGSClosure(
                model,
                ckpt["inv_mean"].to(device),
                ckpt["inv_std"].to(device),
                ckpt["thermo_mean"].to(device),
                ckpt["thermo_std"].to(device),
                dx, dy,
            )
            solver_tbnn.evolve(
                t_end,
                sgs_closure=tbnn_closure,
                verbose=True,
                progress_bar=True,
                desc="TBNN-Closed (Flux)",
                save_interval=save_int,
            )
        else:
            print("  Coupling: legacy source-term")
            sgs_model = TBNNSGSModel(
                model,
                ckpt["inv_mean"].to(device),
                ckpt["inv_std"].to(device),
                ckpt["thermo_mean"].to(device),
                ckpt["thermo_std"].to(device),
                dx, dy,
            )
            solver_tbnn.evolve(
                t_end,
                sgs_model=sgs_model,
                verbose=True,
                progress_bar=True,
                desc="TBNN-Closed (Source)",
                save_interval=save_int,
            )

        prims_tbnn = solver_tbnn.get_primitives()
        if save_int is not None:
            history_tbnn = solver_tbnn.get_history()
    else:
        print(f"\n[SKIP] No checkpoint found at {checkpoint_path}")

    # --- Solver 2: Exact DNS-closed solver ---
    prims_exact = None
    history_exact = None
    if not skip_exact:
        print("\n--- Running Exact DNS-closed coarse solver ---")

        solver_exact = RHDSolver(
            Nc_y, Nc_x, cfl=cfl, reconstruction=reconstruction,
            riemann_solver=riemann_solver, dtype=dtype, device=device,
        )
        solver_exact.initialize(
            coarse0["rho_bar"].to(device),
            coarse0["vx_tilde"].to(device),
            coarse0["vy_tilde"].to(device),
            coarse0["p_bar"].to(device),
        )
        solver_exact.t = snap0["t"]

        if sgs_coupling == "flux":
            print("  Coupling: conservative viscous flux formulation")
            exact_closure = ExactDNSSGSClosure(
                snapshots=snapshots,
                coarsening_factor=r,
                dx=dx,
                dy=dy,
                device=device,
                dtype=dtype,
            )
            solver_exact.evolve(
                t_end,
                sgs_closure=exact_closure,
                verbose=True,
                progress_bar=True,
                desc="Exact DNS-Closed (Flux)",
                save_interval=save_int,
            )
        else:
            print("  Coupling: legacy source-term")
            exact_sgs_model = ExactDNSSGSModel(
                snapshots=snapshots,
                coarsening_factor=r,
                dx=dx,
                dy=dy,
                device=device,
                dtype=dtype,
            )
            solver_exact.evolve(
                t_end,
                sgs_model=exact_sgs_model,
                verbose=True,
                progress_bar=True,
                desc="Exact DNS-Closed (Source)",
                save_interval=save_int,
            )

        prims_exact = solver_exact.get_primitives()
        if save_int is not None:
            history_exact = solver_exact.get_history()

    # --- Ground truth: filtered DNS at t_end ---
    snap_final = min(snapshots, key=lambda s: abs(s["t"] - t_end))
    rho_f = torch.tensor(snap_final["rho"], dtype=dtype)
    vx_f = torch.tensor(snap_final["vx"], dtype=dtype)
    vy_f = torch.tensor(snap_final["vy"], dtype=dtype)
    p_f = torch.tensor(snap_final["p"], dtype=dtype)
    coarse_final = relativistic_favre_filter(rho_f, vx_f, vy_f, p_f, r)

    # --- Compute metrics ---
    print(f"\n=== Metrics at t = {snap_final['t']:.2f} ===")
    results = {}

    if prims_tbnn is not None:
        rho_err_tbnn = relative_l2_error(prims_tbnn["rho"].cpu(), coarse_final["rho_bar"])
        vx_err_tbnn = relative_l2_error(prims_tbnn["vx"].cpu(), coarse_final["vx_tilde"])
        print(f"  TBNN ρ rel L2: {rho_err_tbnn:.6f}")
        print(f"  TBNN vx rel L2: {vx_err_tbnn:.6f}")
        results["tbnn_rho_relL2"] = rho_err_tbnn
        results["tbnn_vx_relL2"] = vx_err_tbnn

    if prims_exact is not None:
        rho_err_exact = relative_l2_error(prims_exact["rho"].cpu(), coarse_final["rho_bar"])
        vx_err_exact = relative_l2_error(prims_exact["vx"].cpu(), coarse_final["vx_tilde"])
        print(f"  Exact DNS ρ rel L2: {rho_err_exact:.6f}")
        print(f"  Exact DNS vx rel L2: {vx_err_exact:.6f}")
        results["exact_rho_relL2"] = rho_err_exact
        results["exact_vx_relL2"] = vx_err_exact

    # --- Energy spectra ---
    k_dns, E_dns = kinetic_energy_spectrum(
        coarse_final["rho_bar"], coarse_final["vx_tilde"], coarse_final["vy_tilde"]
    )

    k_tbnn, E_tbnn = None, None
    if prims_tbnn is not None:
        k_tbnn, E_tbnn = kinetic_energy_spectrum(
            prims_tbnn["rho"].cpu(), prims_tbnn["vx"].cpu(), prims_tbnn["vy"].cpu()
        )

    k_exact, E_exact = None, None
    if prims_exact is not None:
        k_exact, E_exact = kinetic_energy_spectrum(
            prims_exact["rho"].cpu(), prims_exact["vx"].cpu(), prims_exact["vy"].cpu()
        )

    # 2x3 layout:
    # [0, 0] Ground Truth ρ | [0, 1] TBNN ρ | [0, 2] Exact DNS ρ
    # [1, 0] Error Map |Δρ| | [1, 1] Energy Spectrum | [1, 2] Error Bar Chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Color scaling reference from ground truth
    rho_dns_np = coarse_final["rho_bar"].numpy()
    vmin = float(np.min(rho_dns_np))
    vmax = float(np.max(rho_dns_np))

    # [0, 0] Filtered DNS Ground Truth
    im0 = axes[0, 0].imshow(rho_dns_np, origin="lower", cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Filtered DNS ρ (Ground Truth)", fontweight="bold")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    # [0, 1] TBNN-Closed Solver
    if prims_tbnn is not None:
        im1 = axes[0, 1].imshow(prims_tbnn["rho"].cpu().numpy(), origin="lower", cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
        err_t_badge = results.get("tbnn_rho_relL2", 0.0)
        axes[0, 1].set_title(f"TBNN-Closed Solver ρ\nrel L2: {err_t_badge:.4f}", fontweight="bold")
        plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    else:
        axes[0, 1].text(0.5, 0.5, "TBNN Disabled\n(No Checkpoint)", ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[0, 1].axis("off")

    # [0, 2] Exact DNS-Closed Solver
    if prims_exact is not None:
        im2 = axes[0, 2].imshow(prims_exact["rho"].cpu().numpy(), origin="lower", cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
        err_e_badge = results.get("exact_rho_relL2", 0.0)
        axes[0, 2].set_title(f"Exact DNS-Closed Solver ρ\nrel L2: {err_e_badge:.4f}", fontweight="bold")
        plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)
    else:
        axes[0, 2].text(0.5, 0.5, "Exact Solver Skipped", ha="center", va="center", transform=axes[0, 2].transAxes)
        axes[0, 2].axis("off")

    # [1, 0] Absolute Error Map
    target_prims = prims_tbnn if prims_tbnn is not None else prims_exact
    target_name = "TBNN" if prims_tbnn is not None else "Exact"
    if target_prims is not None:
        diff_np = (target_prims["rho"].cpu() - coarse_final["rho_bar"]).abs().numpy()
        im3 = axes[1, 0].imshow(diff_np, origin="lower", cmap="magma", aspect="equal", vmin=0.0, vmax=max(0.5, float(np.max(diff_np))))
        axes[1, 0].set_title(f"|{target_name} - DNS| Absolute Error", fontweight="bold")
        cbar_err = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
        cbar_err.set_label("Absolute Error |Δρ|")
    else:
        axes[1, 0].axis("off")

    # [1, 1] Spectrum plot
    ax_sp = axes[1, 1]
    ax_sp.loglog(k_dns, E_dns, "k-", lw=2, label="Filtered DNS")
    if prims_tbnn is not None and k_tbnn is not None and E_tbnn is not None:
        ax_sp.loglog(k_tbnn, E_tbnn, "r-.", lw=1.5, label="TBNN")
    if prims_exact is not None and k_exact is not None and E_exact is not None:
        ax_sp.loglog(k_exact, E_exact, "g:", lw=1.8, label="Exact DNS")
    ax_sp.set_xlabel("k")
    ax_sp.set_ylabel("E(k)")
    ax_sp.set_title(f"Kinetic Energy Spectrum (t={snap_final['t']:.1f})", fontweight="bold")
    ax_sp.legend()
    ax_sp.grid(True, alpha=0.3)

    # [1, 2] Relative L2 Error Comparison Bar Chart
    ax_bar = axes[1, 2]
    models = []
    rho_errs = []
    vx_errs = []

    if prims_tbnn is not None and "tbnn_rho_relL2" in results:
        models.append("TBNN")
        rho_errs.append(results["tbnn_rho_relL2"])
        vx_errs.append(results["tbnn_vx_relL2"])

    if prims_exact is not None and "exact_rho_relL2" in results:
        models.append("Exact DNS")
        rho_errs.append(results["exact_rho_relL2"])
        vx_errs.append(results["exact_vx_relL2"])

    if models:
        x = np.arange(len(models))
        bar_width = 0.35
        b1 = ax_bar.bar(x - bar_width / 2, rho_errs, bar_width, label="ρ rel L2", color="#4C72B0")
        b2 = ax_bar.bar(x + bar_width / 2, vx_errs, bar_width, label="vx rel L2", color="#DD8452")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(models, fontweight="bold")
        ax_bar.set_ylabel("Relative L2 Error")
        ax_bar.set_title(f"State Error Comparison (t={snap_final['t']:.1f})", fontweight="bold")
        ax_bar.legend(loc="upper right")
        ax_bar.grid(True, axis="y", alpha=0.3)
        for rect in list(b1) + list(b2):
            h = rect.get_height()
            ax_bar.text(rect.get_x() + rect.get_width() / 2.0, h, f"{h:.4f}",
                        ha="center", va="bottom", fontsize=8)
    else:
        ax_bar.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "rollout_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {fig_path}")

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    ref_history = history_tbnn if history_tbnn is not None else history_exact
    history_dns = None
    if (save_history or make_anim) and ref_history is not None and "t" in ref_history:
        times = np.asarray(ref_history["t"], dtype=np.float64)
        sorted_snaps = sorted(snapshots, key=lambda s: float(s["t"]))
        snap_times = np.array([float(s["t"]) for s in sorted_snaps], dtype=np.float64)

        # Precompute filtered snapshots
        dns_cache = []
        for s in sorted_snaps:
            c_k = relativistic_favre_filter(
                torch.tensor(s["rho"], dtype=dtype),
                torch.tensor(s["vx"], dtype=dtype),
                torch.tensor(s["vy"], dtype=dtype),
                torch.tensor(s["p"], dtype=dtype),
                r,
            )
            dns_cache.append({k: v.cpu() for k, v in c_k.items()})

        rho_list, vx_list, vy_list, p_list = [], [], [], []
        for t_val in times:
            if t_val <= snap_times[0]:
                c = dns_cache[0]
                rho_list.append(c["rho_bar"])
                vx_list.append(c["vx_tilde"])
                vy_list.append(c["vy_tilde"])
                p_list.append(c["p_bar"])
            elif t_val >= snap_times[-1]:
                c = dns_cache[-1]
                rho_list.append(c["rho_bar"])
                vx_list.append(c["vx_tilde"])
                vy_list.append(c["vy_tilde"])
                p_list.append(c["p_bar"])
            else:
                idx = int(np.searchsorted(snap_times, t_val)) - 1
                idx = max(0, min(idx, len(snap_times) - 2))
                t0 = snap_times[idx]
                t1 = snap_times[idx + 1]
                dt_span = t1 - t0
                alpha = float((t_val - t0) / dt_span) if dt_span > 1e-14 else 0.0
                c0 = dns_cache[idx]
                c1 = dns_cache[idx + 1]
                rho_list.append((1.0 - alpha) * c0["rho_bar"] + alpha * c1["rho_bar"])
                vx_list.append((1.0 - alpha) * c0["vx_tilde"] + alpha * c1["vx_tilde"])
                vy_list.append((1.0 - alpha) * c0["vy_tilde"] + alpha * c1["vy_tilde"])
                p_list.append((1.0 - alpha) * c0["p_bar"] + alpha * c1["p_bar"])

        history_dns = {
            "t": torch.tensor(times, dtype=dtype),
            "rho": torch.stack(rho_list, dim=0),
            "vx": torch.stack(vx_list, dim=0),
            "vy": torch.stack(vy_list, dim=0),
            "p": torch.stack(p_list, dim=0),
        }

    if save_history and history_dns is not None and ref_history is not None:
        traj_path = os.path.join(output_dir, "rollout_trajectory.pt")
        traj_data = {
            "t": ref_history["t"] if isinstance(ref_history["t"], torch.Tensor) else torch.tensor(ref_history["t"], dtype=dtype),
            "dns": history_dns,
            "tbnn": history_tbnn,
            "exact": history_exact,
            "grid": {"Ny": Nc_y, "Nx": Nc_x, "dx": dx, "dy": dy, "r": r},
        }
        torch.save(traj_data, traj_path)
        print(f"\nTrajectory history saved to {traj_path}")

    if make_anim and history_dns is not None:
        anim_path = os.path.join(output_dir, "rollout_comparison.gif")
        print("\n--- Generating rollout comparison animation ---")
        create_rollout_animation(
            history_dns=history_dns,
            history_tbnn=history_tbnn,
            history_exact=history_exact,
            output_path=anim_path,
            fps=anim_fps,
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TBNN rollout")
    parser.add_argument("--data-dir", type=str, default="test1")
    parser.add_argument("--coarsening-factor", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--t-start", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=10.0)
    parser.add_argument("--cfl", type=float, default=0.3)
    parser.add_argument("--reconstruction", type=str, default="mp5", choices=["mp5", "muscl"])
    parser.add_argument("--riemann-solver", type=str, default="hll", choices=["hll", "musta_hll", "hllc"])
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-exact", action="store_true", help="Skip running the exact DNS SGS solver")
    parser.add_argument("--sgs-coupling", type=str, default="flux", choices=["source", "flux"],
                        help="SGS coupling mode: 'source' (legacy) or 'flux' (conservative viscous flux)")
    parser.add_argument("--save-history", action="store_true", help="Save intermediate snapshot trajectories to rollout_trajectory.pt")
    parser.add_argument("--make-anim", action="store_true", help="Generate an animated comparison GIF across the rollout")
    parser.add_argument("--anim-interval", type=float, default=0.1, help="Simulation time interval between animation frames")
    parser.add_argument("--anim-fps", type=int, default=10, help="Frames per second for generated animation")
    args = parser.parse_args()

    evaluate_rollout(
        data_dir=args.data_dir,
        coarsening_factor=args.coarsening_factor,
        checkpoint_path=args.checkpoint,
        t_start=args.t_start,
        t_end=args.t_end,
        cfl=args.cfl,
        reconstruction=args.reconstruction,
        riemann_solver=args.riemann_solver,
        output_dir=args.output_dir,
        device_str=args.device,
        skip_exact=args.skip_exact,
        save_history=args.save_history,
        make_anim=args.make_anim,
        anim_interval=args.anim_interval,
        anim_fps=args.anim_fps,
        sgs_coupling=args.sgs_coupling,
    )
