r"""
evaluate_rollout.py

Benchmarking and diagnostic evaluation for the CRMHD-TBNN SGS closure.

Compares simulation trajectories on the coarse grid:
  1. TBNN-closed simulation — solver + CRMHD_TBNN closure
  2. Exact DNS-closed simulation — oracle SGS stress tensor from filtered DNS
  3. Filtered high-res benchmark — true Favre-filtered DNS states

Diagnostic metrics:
  - Kinetic energy power spectrum E(k) and scale-resolved band energy ratios
  - Log-spectral distance against filtered DNS, with a persistence baseline

Pointwise relative L2 is deliberately NOT reported here. On a chaotic trajectory it
measures phase decorrelation rather than closure quality, and it has been observed to
rank a visibly noise-contaminated field above a nearly exact one.
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
    read_snapshot, snapshot_files, relativistic_favre_filter, GAMMA_EOS, compute_sgs_stress,
    compute_sgs_stress_on_faces,
)
from src.config import add_config_arguments, apply_config_defaults, configure
from src.features import extract_tbnn_features
from src.minit import MInITClosure, MInITCoefficients, subgrid_energy_from_stress
from src.minit_learned import HybridClosure, LearnedMInITClosure
from src.solver import RHDSolver, compute_transition_sensor, clip_sgs_tensor, clip_backscatter, smooth_sgs_tensor


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


def spectral_metrics(
    rho: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    rho_ref: torch.Tensor,
    vx_ref: torch.Tensor,
    vy_ref: torch.Tensor,
) -> dict[str, float]:
    r"""Scale-resolved comparison of a rollout against filtered DNS.

    Pointwise norms cannot rank closures on a chaotic trajectory: two runs decorrelate
    in phase regardless of how well either reproduces the turbulence, and a run that is
    visibly wrong can score better than a nearly perfect one. These metrics compare the
    *distribution* of energy across scales instead.

    ``log_spectral_distance`` is the headline number: mean |log10(E/E_ref)| over resolved
    wavenumbers, so an order-of-magnitude excess anywhere costs 1.0 regardless of where it
    sits. ``band_ratio_high`` is the most diagnostic single value, since a closure that
    injects grid-scale noise shows up there first.

    Returns
    -------
    dict with e_ratio_total, band_ratio_{low,mid,high}, log_spectral_distance
    """
    k, E = kinetic_energy_spectrum(rho, vx, vy)
    _, E_ref = kinetic_energy_spectrum(rho_ref, vx_ref, vy_ref)

    valid = (E_ref > 0) & (E > 0) & (k > 0)
    lsd = float(np.abs(np.log10(E[valid] / E_ref[valid])).mean()) if valid.any() else float("nan")

    k_max = k.max()
    bands = {"low": (0.0, k_max / 8), "mid": (k_max / 8, k_max / 3), "high": (k_max / 3, k_max)}
    out = {
        "e_ratio_total": float(E.sum() / max(E_ref.sum(), 1e-30)),
        "log_spectral_distance": lsd,
    }
    for name, (lo, hi) in bands.items():
        m = (k > lo) & (k <= hi)
        out[f"band_ratio_{name}"] = (
            float(E[m].sum() / max(E_ref[m].sum(), 1e-30)) if m.any() else float("nan")
        )
    return out


def perturb_initial_state(coarse: dict, seed: int, amplitude: float = 1e-3) -> dict:
    """Add a small large-scale velocity perturbation to the filtered initial state.

    One rollout is one draw of a chaotic trajectory, so a ranking between closures needs
    several. Every run in an evaluation starts from the same perturbed state, so the
    comparison between closures stays paired within a member. The noise is band-limited
    to k <= N/8 (grid-scale noise would just be dissipated), scaled to ``amplitude``
    times the rms velocity, and leaves density and pressure untouched.
    """
    g = torch.Generator().manual_seed(int(seed))
    vx, vy = coarse["vx_tilde"], coarse["vy_tilde"]
    Ny, Nx = vx.shape
    ky = torch.fft.fftfreq(Ny, d=1.0 / Ny).abs()[:, None]
    kx = torch.fft.fftfreq(Nx, d=1.0 / Nx).abs()[None, :]
    keep = ((kx**2 + ky**2).sqrt() <= min(Ny, Nx) / 8).to(vx.dtype)
    rms = float(torch.sqrt((vx**2 + vy**2).mean()))
    out = dict(coarse)
    for key, v in (("vx_tilde", vx), ("vy_tilde", vy)):
        noise = torch.randn(Ny, Nx, generator=g, dtype=torch.float64).to(v)
        noise = torch.fft.ifft2(torch.fft.fft2(noise) * keep).real
        out[key] = v + amplitude * rms * noise / noise.std().clamp(min=1e-300)
    return out


def windowed_band_ratios(
    history_ref: dict,
    runs: dict[str, dict],
    valid: np.ndarray,
    windows: list[tuple[float, float]],
    baseline: str | None = None,
    t_tol: float = np.inf,
    block_time: float = 0.5,
    n_boot: int = 1000,
) -> dict[str, dict[str, dict[str, float]]]:
    r"""Band ratios averaged over time windows, instead of read off one frame.

    A single-frame band ratio is noise on a chaotic rollout: on ``dns_khi`` at r=16 the
    high band swings 0.08-0.34 frame to frame, and three conclusions drawn from final-time
    values had to be withdrawn (FEASIBILITY §16-17).

    * **Per-frame geometric mean**, not a ratio of time averages. A ratio of averages
      weights each frame by the DNS's own band energy, which roughly doubles over a KHI
      run. The geometric mean weights frames equally and is scale-free.
    * **Separate windows per regime.** Retention itself falls over a run (high band
      ~0.44 -> ~0.30 on dns_khi) and closures tie early but separate late, so one long
      average mixes regimes. Quote per-window numbers.
    * **Reference = nearest real DNS snapshot**, never a blend of two (``history_ref``);
      ``valid`` marks frames with a snapshot close enough. Run frames whose own time is
      more than ``t_tol`` from the reference frame's are dropped, so a solver that skips
      a save cannot silently shift its index alignment.
    * **Paired comparison against ``baseline``** (normally the exact-stress oracle, the
      best any closure can do on this solver): ``vs_baseline_high`` =
      exp(mean(log r_run - log r_base)), where the DNS reference cancels exactly. Raw
      retention is *not* a quality score -- the no-model run and the TBNN retain more
      high-k than the oracle -- closeness to the oracle is.
    * **Circular moving-block bootstrap**, one set of resampled frames per window shared
      by every run, ``ceil(n/L)`` blocks truncated to ``n`` frames, ``block_time`` 0.5
      (the measured decorrelation time of the log ratios). The CI covers frame sampling
      along this one trajectory only -- not initial-condition or training-seed spread.

    Returns ``{window: {run: {band_ratio_low/mid/high, e_ratio_total,
    log_spectral_distance, high_err, vs_baseline_high, vs_baseline_lo, vs_baseline_hi,
    n_frames, partial}}}``.
    """
    t = np.asarray(history_ref["t"], dtype=np.float64)
    keys = ("band_ratio_low", "band_ratio_mid", "band_ratio_high",
            "e_ratio_total", "log_spectral_distance")
    rng = np.random.default_rng(0)
    out: dict[str, dict[str, dict[str, float]]] = {}

    def usable(h):
        n = min(len(h["rho"]), len(t))
        ok = np.zeros(len(t), dtype=bool)
        th = np.asarray(h["t"], dtype=np.float64)[:n] if "t" in h else t[:n]
        ok[:n] = np.abs(th - t[:n]) <= t_tol
        return ok

    for lo, hi in windows:
        win = valid & (t >= lo - 1e-9) & (t <= hi + 1e-9)
        frames = np.where(win)[0]
        wlabel = f"{lo:g}-{hi:g}"
        out[wlabel] = {}
        if len(frames) == 0:
            continue
        n = len(frames)
        dt_frame = np.median(np.diff(t[frames])) if n > 1 else block_time
        L = min(n, max(1, int(round(block_time / dt_frame))))
        starts = rng.integers(0, n, size=(n_boot, -(-n // L)))
        boot = ((starts[:, :, None] + np.arange(L)) % n).reshape(n_boot, -1)[:, :n]

        logs = {}                                   # run -> (n, 4) log ratios, NaN off-frame
        for label, h in runs.items():
            ok = usable(h)[frames]
            if not ok.any():
                continue
            per = np.full((n, len(keys)), np.nan)
            for j, i in enumerate(frames):
                if ok[j]:
                    m = spectral_metrics(
                        h["rho"][i], h["vx"][i], h["vy"][i],
                        history_ref["rho"][i], history_ref["vx"][i], history_ref["vy"][i],
                    )
                    per[j] = [m[k] for k in keys]
            lg = np.log(np.where(per[:, :4] > 0, per[:, :4], np.nan))
            logs[label] = lg
            row = {k: float(np.exp(np.nanmean(lg[:, c]))) for c, k in enumerate(keys[:4])}
            row["log_spectral_distance"] = float(np.nanmean(per[:, 4]))
            row["n_frames"] = int(np.isfinite(lg[:, 2]).sum())
            row["partial"] = row["n_frames"] < n    # diverged / misaligned: not comparable
            row["high_err"] = (float(np.exp(lg[boot, 2].mean(axis=1)).std())
                               if L < n and not row["partial"] else float("nan"))
            out[wlabel][label] = row

        if baseline in logs:
            base = logs[baseline][:, 2]
            for label, lg in logs.items():
                d = lg[:, 2] - base
                if not np.isfinite(d).all():        # partial run or baseline: no CI
                    out[wlabel][label].update(vs_baseline_high=float(np.exp(np.nanmean(d))),
                                              vs_baseline_lo=float("nan"),
                                              vs_baseline_hi=float("nan"))
                    continue
                b = np.exp(d[boot].mean(axis=1))
                out[wlabel][label].update(
                    vs_baseline_high=float(np.exp(d.mean())),
                    vs_baseline_lo=float(np.percentile(b, 2.5)) if L < n else float("nan"),
                    vs_baseline_hi=float(np.percentile(b, 97.5)) if L < n else float("nan"),
                )
    return out


from collections.abc import Mapping

# One style per MInIT variant, so a variant can never be dropped from a plot by
# running past the end of a style list, and so colours stay stable across figures.
_MINIT_STYLE = {
    "fitted": {"color": "tab:cyan", "ls": "-"},
    "paper": {"color": "tab:olive", "ls": "-."},
    "equilibrium": {"color": "tab:pink", "ls": "-"},
    "learned-residual": {"color": "tab:red", "ls": "-"},
    "learned-direct": {"color": "tab:brown", "ls": "--"},
    "hybrid": {"color": "tab:purple", "ls": "-"},
    "ablation-exactE": {"color": "tab:gray", "ls": ":"},
    "ablation-exactS": {"color": "tab:olive", "ls": ":"},
}

# Learned-coefficient runs share the MInIT run/history dicts, so that the
# spectra, bar chart, density panels, animation and saved history pick them up
# without a parallel set of plumbing that a later edit could forget to update.
# They are a different model, though, so they must not be *labelled* MInIT.
_LEARNED_PREFIX = "learned-"


def _variant_label(variant: str) -> str:
    """Display name for a run stored in the MInIT dicts."""
    if variant == "hybrid":
        return "hybrid (TBNN shape x MiNNit amp)"
    if variant == "ablation-exactE":
        return "ablation (TBNN shape x exact energy)"
    if variant == "ablation-exactS":
        return "ablation (exact shape x MiNNit energy)"
    if variant.startswith(_LEARNED_PREFIX):
        return f"learned-coeff ({variant[len(_LEARNED_PREFIX):]})"
    return f"MInIT ({variant})"


def _variant_key(variant: str) -> str:
    """Results-dict key for a run stored in the MInIT dicts."""
    if variant == "hybrid":
        return "hybrid"
    if variant.startswith("ablation-"):
        return variant.replace("-", "_")
    if variant.startswith(_LEARNED_PREFIX):
        return f"learned_{variant[len(_LEARNED_PREFIX):]}"
    return f"minit_{variant}"


def _collect_runs(
    history_tbnn: Mapping[str, object] | None,
    history_exact: Mapping[str, object] | None,
    history_unclosed: Mapping[str, object] | None,
    history_minit: Mapping[str, Mapping[str, object]] | None,
) -> list[tuple[str, torch.Tensor]]:
    """Gather every available run's density history, in a fixed display order.

    Ordered control first, then the competing closures, then the oracle, so the
    panels read left to right as "does anything change?". Built from whatever is
    present rather than from a fixed layout, so adding a closure does not
    silently drop it from the figures — which is what happened when MInIT was
    first wired in.
    """
    candidates: list[tuple[str, Mapping[str, object] | None]] = [
        ("Unclosed (no SGS)", history_unclosed),
    ]
    for variant, hist in sorted((history_minit or {}).items()):
        candidates.append((_variant_label(variant), hist))
    candidates += [
        ("TBNN-Closed", history_tbnn),
        ("Exact DNS-Closed", history_exact),
    ]

    runs: list[tuple[str, torch.Tensor]] = []
    for label, hist in candidates:
        if hist is None or "rho" not in hist:
            continue
        rho = hist["rho"]
        if isinstance(rho, torch.Tensor) and rho.numel():
            runs.append((label, rho))
    return runs


def create_rollout_animation(
    history_dns: Mapping[str, object],
    history_tbnn: Mapping[str, object] | None = None,
    history_exact: Mapping[str, object] | None = None,
    history_unclosed: Mapping[str, object] | None = None,
    history_minit: Mapping[str, Mapping[str, object]] | None = None,
    output_path: str = "results/rollout_comparison.gif",
    fps: int = 10,
) -> str:
    """Synchronised panel animation comparing DNS against every available solver.

    The layout is built from the runs that are actually present: the top row
    shows the filtered DNS and each solver's density, the bottom row the
    corresponding ``|run - DNS|`` error map under each one. A fixed grid would
    quietly omit any closure it was not written for.
    """
    times = np.asarray(history_dns["t"])
    N_frames = len(times)
    if N_frames == 0:
        print("[ANIM] No frames to animate.")
        return output_path

    rho_dns_all = history_dns["rho"]
    assert isinstance(rho_dns_all, torch.Tensor)

    runs = _collect_runs(history_tbnn, history_exact, history_unclosed, history_minit)
    if not runs:
        print("[ANIM] No solver runs to compare against; skipping animation.")
        return output_path

    # Shared density scale across every panel and frame, so panels are comparable.
    all_rho = [rho_dns_all] + [r for _, r in runs]
    vmin = min(float(t.min()) for t in all_rho)
    vmax = max(float(t.max()) for t in all_rho)

    # One error scale as well, so the maps rank against each other rather than
    # each being normalised to its own worst cell.
    max_err = max(
        float((r[:N_frames] - rho_dns_all[:N_frames]).abs().max()) for _, r in runs
    )
    max_err = max(max_err, 1e-12)

    n_cols = 1 + len(runs)
    panel_w = 6.5 if n_cols <= 3 else max(3.2, 20.0 / n_cols)
    fig, axes = plt.subplots(
        2, n_cols, figsize=(panel_w * n_cols, 2.05 * panel_w), squeeze=False,
    )
    fig.subplots_adjust(
        left=0.04, right=0.88, top=0.90, bottom=0.05, wspace=0.16, hspace=0.18,
    )

    im_dns = axes[0, 0].imshow(
        rho_dns_all[0].numpy(), origin="lower", cmap="viridis",
        vmin=vmin, vmax=vmax, aspect="equal",
    )
    axes[0, 0].set_title("Filtered DNS Ground Truth", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    ims_rho: list[tuple[torch.Tensor, object]] = []
    ims_err: list[tuple[torch.Tensor, object]] = []
    for col, (label, rho_all) in enumerate(runs, start=1):
        im = axes[0, col].imshow(
            rho_all[0].numpy(), origin="lower", cmap="viridis",
            vmin=vmin, vmax=vmax, aspect="equal",
        )
        axes[0, col].set_title(label, fontsize=11, fontweight="bold")
        ims_rho.append((rho_all, im))

        im_e = axes[1, col].imshow(
            (rho_all[0] - rho_dns_all[0]).abs().numpy(), origin="lower",
            cmap="magma", vmin=0.0, vmax=max_err, aspect="equal",
        )
        axes[1, col].set_title(f"|{label} - DNS|", fontsize=10)
        ims_err.append((rho_all, im_e))

    cbar_rho = fig.colorbar(im_dns, cax=fig.add_axes((0.90, 0.54, 0.015, 0.34)))
    cbar_rho.set_label("Rest-frame Density ρ̄", fontsize=10)
    if ims_err:
        cbar_err = fig.colorbar(ims_err[0][1], cax=fig.add_axes((0.90, 0.10, 0.015, 0.34)))
        cbar_err.set_label("|Δρ̄|", fontsize=10)

    pbar_anim = tqdm(
        total=N_frames, desc="Rendering animation frames", unit="frame",
        dynamic_ncols=True, file=sys.stdout,
    )

    def update(frame_idx: int):
        t_val = float(times[frame_idx])
        fig.suptitle(
            f"Relativistic Hydrodynamics Rollout Comparison  (t = {t_val:.2f})",
            fontsize=14, fontweight="bold",
        )

        rho_dns_k = rho_dns_all[frame_idx]
        im_dns.set_data(rho_dns_k.numpy())

        for rho_all, im in ims_rho:
            im.set_data(rho_all[frame_idx].numpy())
        for rho_all, im_e in ims_err:
            im_e.set_data((rho_all[frame_idx] - rho_dns_k).abs().numpy())

        pbar_anim.update(1)
        return [im_dns]

    ani = anim.FuncAnimation(
        fig, update, frames=N_frames, interval=int(1000 / fps), blit=False,
    )
    ani.save(output_path, writer=anim.PillowWriter(fps=fps))
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
    backscatter_clip : str or None — None (allow backscatter), "zero", or "project"
    backscatter_fraction : float — fraction of backscatter to retain (0 = strict clip)
    tau_smooth_passes : int — (1,2,1)/4 low-pass applications on tau before use
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
        backscatter_clip: str | None = None,
        backscatter_fraction: float = 0.0,
        tau_smooth_passes: int = 0,
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
        self.backscatter_clip = backscatter_clip
        self.backscatter_fraction = backscatter_fraction
        self.tau_smooth_passes = tau_smooth_passes

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

        if self.tau_smooth_passes:
            tau_perp = smooth_sgs_tensor(tau_perp, self.tau_smooth_passes)

        if self.backscatter_clip is not None:
            tau_perp = clip_backscatter(
                tau_perp, prims, self.dx, self.dy, mode=self.backscatter_clip,
                fraction=self.backscatter_fraction,
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
        backscatter_clip: str | None = None,
        backscatter_fraction: float = 0.0,
        tau_smooth_passes: int = 0,
        face_closure: bool = False,
    ):
        self.dx = dx
        self.dy = dy
        self.device = device
        self.dtype = dtype
        self.sensor_mode = sensor_mode
        self.clip_mode = clip_mode
        self.clip_kappa = clip_kappa
        self.backscatter_clip = backscatter_clip
        self.backscatter_fraction = backscatter_fraction
        self.tau_smooth_passes = tau_smooth_passes
        self.provides_face_stress = face_closure

        sorted_snaps = sorted(snapshots, key=lambda s: s["t"])
        self.times = [float(s["t"]) for s in sorted_snaps]
        self.times_np = np.array(self.times, dtype=np.float64)

        self.taus: list[torch.Tensor] = []
        self.taus_x: list[torch.Tensor] = []
        self.taus_y: list[torch.Tensor] = []
        extra = " (+ face-staggered)" if face_closure else ""
        print(f"  [ExactDNSSGSClosure] Precomputing SGS tensors{extra} "
              f"for {len(sorted_snaps)} snapshots...")
        for s in sorted_snaps:
            rho = torch.tensor(s["rho"], dtype=dtype, device=device)
            vx = torch.tensor(s["vx"], dtype=dtype, device=device)
            vy = torch.tensor(s["vy"], dtype=dtype, device=device)
            p = torch.tensor(s["p"], dtype=dtype, device=device)

            _, tau = compute_sgs_stress(
                rho, vx, vy, p, r=coarsening_factor, project_to_fluid_frame=True,
            )
            self.taus.append(tau)
            if face_closure:
                tau_x, tau_y = compute_sgs_stress_on_faces(
                    rho, vx, vy, p, r=coarsening_factor, project_to_fluid_frame=True,
                )
                self.taus_x.append(tau_x)
                self.taus_y.append(tau_y)
            del rho, vx, vy, p

        self.current_t = self.times[0]

    def _interp(self, series: list[torch.Tensor]) -> torch.Tensor:
        """Linearly interpolate a precomputed per-snapshot series to the current time."""
        t = self.current_t
        if t <= self.times[0]:
            return series[0]
        if t >= self.times[-1]:
            return series[-1]
        idx = int(np.searchsorted(self.times_np, t)) - 1
        idx = max(0, min(idx, len(self.times) - 2))
        t0, t1 = self.times[idx], self.times[idx + 1]
        span = t1 - t0
        frac = (t - t0) / span if span > 1e-14 else 0.0
        return (1.0 - frac) * series[idx] + frac * series[idx + 1]

    @torch.no_grad()
    def faces(self, prims: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Time-interpolated SGS stress at x- and y-faces.

        Only the realizability clip and tau smoothing are applied here; the sensor and
        backscatter clip act on the cell-centred tensor in __call__, which is what
        gates delta_U.
        """
        if not self.provides_face_stress:
            raise RuntimeError("this closure was not built with face_closure=True")
        out = []
        for series in (self.taus_x, self.taus_y):
            tau = clip_sgs_tensor(
                self._interp(series), prims, mode=self.clip_mode, kappa=self.clip_kappa,
            )
            if self.tau_smooth_passes:
                tau = smooth_sgs_tensor(tau, self.tau_smooth_passes)
            out.append(tau)
        return out[0], out[1]

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

        if self.tau_smooth_passes:
            tau = smooth_sgs_tensor(tau, self.tau_smooth_passes)

        if self.backscatter_clip is not None:
            tau = clip_backscatter(
                tau, prims, self.dx, self.dy, mode=self.backscatter_clip,
                fraction=self.backscatter_fraction,
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
    backscatter_clip: str | None = None,
    backscatter_fraction: float = 0.0,
    tau_smooth_passes: int = 0,
    face_closure: bool = False,
    with_unclosed: bool = False,
    minit_variants: tuple[str, ...] = (),
    minit_coefficients: str | None = None,
    minit_seed: str = "dns",
    minit_e_seed: float = 0.0,
    minit_floor_fraction: float = 0.0,
    minit_advect: bool = False,
    minit_shear_mode: str = "dvxdy",
    minit_average: str = "cell",
    learned_coeff: tuple[str, ...] = (),
    learned_gain: float = 1.0,
    hybrid_coeff: str | None = None,
    hybrid_clip: bool = True,
    hybrid_ablations: bool = False,
    band_windows: list[tuple[float, float]] | None = None,
    band_skip: float = 0.5,
    perturb_seed: int = 0,
    perturb_amplitude: float = 1e-3,
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
    backscatter_clip : str or None — None, "zero" or "project" (see clip_backscatter)
    face_closure : bool — evaluate the exact SGS stress on cell faces (staggered
        filter boxes) so its divergence is compact, instead of interpolating a
        cell-centred stress. Exact-DNS closure only.
    with_unclosed : bool — also run a solver with no SGS term, as a control
    minit_variants : any of "fitted", "paper", "equilibrium" — MInIT runs to add.
        "fitted" uses coefficients refit on this DNS by ``src.minit_calibrate``;
        "paper" transfers the published 3D Newtonian MHD calibration verbatim,
        and the gap between the two separates the model from its calibration;
        "equilibrium" replaces the turbulent-energy ODE with its fixed point,
        which is what the free calibration converges towards and which scores
        best a-priori.
    minit_coefficients : path to the refit written by ``src.minit_calibrate``
    minit_seed : "dns" (seed e_KH from the exact filtered-DNS subgrid energy at
        t_start, mirroring the paper and giving MInIT the same single snapshot of
        ground truth the exact-DNS oracle gets), or "uniform" (seed from
        ``minit_e_seed``, fully self-contained). These are different experiments.
    minit_e_seed : uniform seed value used when minit_seed == "uniform"
    minit_floor_fraction : hold e_KH at this fraction of its local saturation
        value, so a decayed region can restart
    minit_advect : include the advective term the paper drops
    minit_shear_mode : "dvxdy" (Eq. 23 literally) or "strain" (invariant)
    learned_coeff : paths to checkpoints from ``src.train_minit_coeff`` — MInIT
        with a state-dependent coefficient (``src.minit_learned``). These runs
        are stateless and take **no seed**: unlike the ``minit_variants`` rows,
        which are handed the exact filtered-DNS e_KH at t_start, they read only
        resolved state. Read any comparison between them with that in mind.
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

    # Solvers that blow up raise FloatingPointError from evolve(); catching it here
    # (instead of letting it crash the process) preserves whatever states were
    # already recorded in the solver's history buffer up to that point.
    diverged_at: dict[str, float] = {}

    def _safe_evolve(solver: "RHDSolver", key: str, **kwargs) -> bool:
        try:
            solver.evolve(t_end, **kwargs)
            return True
        except FloatingPointError as e:
            print(f"\n[DIVERGED] {key} solver diverged: {e}")
            diverged_at[key] = float(solver.t)
            return False

    # Load the snapshots in the window. Times are read from the file headers first,
    # so files outside it are never loaded (a 2048^2 run is ~57 GB on disk).
    files = snapshot_files(data_dir, (t_start - 0.1, t_end + 0.1))
    snapshots = []
    pbar_snaps = tqdm(files, desc="Loading DNS snapshots", unit="snap", dynamic_ncols=True, file=sys.stdout)
    for f in pbar_snaps:
        snapshots.append(read_snapshot(f))
        pbar_snaps.set_postfix(t=f"{snapshots[-1]['t']:.2f}")

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
    if perturb_seed:
        coarse0 = perturb_initial_state(coarse0, perturb_seed, perturb_amplitude)
        print(f"Initial state perturbed: seed {perturb_seed}, amplitude {perturb_amplitude:g} "
              f"x rms velocity at k <= N/8 (same for every run)")

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
                backscatter_clip=backscatter_clip,
                backscatter_fraction=backscatter_fraction,
                tau_smooth_passes=tau_smooth_passes,
            )
            ok_tbnn = _safe_evolve(
                solver_tbnn, "tbnn",
                sgs_closure=tbnn_closure,
                verbose=True,
                progress_bar=True,
                desc="TBNN-Closed (Flux)",
                save_interval=save_int,
            )
        else:
            print("  Coupling: legacy source-term")
            if backscatter_clip is not None:
                raise ValueError(
                    "--backscatter-clip requires --sgs-coupling flux; the legacy "
                    "source-term path does not apply it."
                )
            sgs_model = TBNNSGSModel(
                model,
                ckpt["inv_mean"].to(device),
                ckpt["inv_std"].to(device),
                ckpt["thermo_mean"].to(device),
                ckpt["thermo_std"].to(device),
                dx, dy,
            )
            ok_tbnn = _safe_evolve(
                solver_tbnn, "tbnn",
                sgs_model=sgs_model,
                verbose=True,
                progress_bar=True,
                desc="TBNN-Closed (Source)",
                save_interval=save_int,
            )

        prims_tbnn = solver_tbnn.get_primitives() if ok_tbnn else None
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
                backscatter_clip=backscatter_clip,
                backscatter_fraction=backscatter_fraction,
                tau_smooth_passes=tau_smooth_passes,
                face_closure=face_closure,
            )
            ok_exact = _safe_evolve(
                solver_exact, "exact",
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
            ok_exact = _safe_evolve(
                solver_exact, "exact",
                sgs_model=exact_sgs_model,
                verbose=True,
                progress_bar=True,
                desc="Exact DNS-Closed (Source)",
                save_interval=save_int,
            )

        prims_exact = solver_exact.get_primitives() if ok_exact else None
        if save_int is not None:
            history_exact = solver_exact.get_history()

    # --- Solver 3: unclosed (no SGS term at all) ---
    # The control. Any closure that does not separate from this has not demonstrably
    # done anything, whatever its error metric says.
    prims_unclosed = None
    history_unclosed = None
    if with_unclosed:
        print("\n--- Running unclosed coarse solver (no SGS) ---")

        solver_unclosed = RHDSolver(
            Nc_y, Nc_x, cfl=cfl, reconstruction=reconstruction,
            riemann_solver=riemann_solver, dtype=dtype, device=device,
        )
        solver_unclosed.initialize(
            coarse0["rho_bar"].to(device),
            coarse0["vx_tilde"].to(device),
            coarse0["vy_tilde"].to(device),
            coarse0["p_bar"].to(device),
        )
        solver_unclosed.t = snap0["t"]
        ok_unclosed = _safe_evolve(
            solver_unclosed, "unclosed",
            verbose=True,
            progress_bar=True,
            desc="Unclosed (no SGS)",
            save_interval=save_int,
        )

        prims_unclosed = solver_unclosed.get_primitives() if ok_unclosed else None
        if save_int is not None:
            history_unclosed = solver_unclosed.get_history()

    # --- Solver 4: MInIT-closed (competing algebraic closure) ---
    # MInIT carries its own evolved turbulent energy density, so unlike the TBNN
    # and exact-DNS closures it is stateful; RHDSolver.evolve advances it once
    # per completed step via the closure's advance() hook.
    prims_minit: dict[str, dict] = {}
    history_minit: dict[str, dict] = {}
    minit_closures: dict[str, MInITClosure] = {}

    if minit_variants:
        # "equilibrium" drops the ODE for its fixed point; see MInITClosure.
        coeff_sets: dict[str, tuple[MInITCoefficients, str]] = {}
        for variant in minit_variants:
            if variant == "paper":
                coeff_sets["paper"] = (MInITCoefficients.paper(), "evolve")
                continue

            mode = "equilibrium" if variant == "equilibrium" else "evolve"
            path = minit_coefficients
            if variant == "equilibrium" and path:
                eq_path = Path(path).with_name(Path(path).stem + "_equilibrium.json")
                path = str(eq_path)
            if not path or not os.path.exists(path):
                raise FileNotFoundError(
                    f"MInIT variant {variant!r} needs refit coefficients, but "
                    f"{path!r} does not exist. Run `python -m src.minit_calibrate` "
                    f"first, or ask for --with-minit paper."
                )
            coeff_sets[variant] = (MInITCoefficients.load(path), mode)

        # e_KH = 0 is a fixed point of the model, so without a seed MInIT is
        # inert and the run would silently reproduce the unclosed control.
        e_seed_field = None
        if minit_seed == "dns":
            _, tau0 = compute_sgs_stress(rho0, vx0, vy0, p0, r=r, project_to_fluid_frame=True)
            e_seed_field = subgrid_energy_from_stress(tau0).to(device)
            print(f"\nMInIT seed: exact filtered-DNS e_KH at t={snap0['t']:.2f} "
                  f"(mean {float(e_seed_field.mean()):.3e}, max {float(e_seed_field.max()):.3e})")
        elif minit_seed == "uniform":
            print(f"\nMInIT seed: uniform e_KH = {minit_e_seed:.3e} (no DNS information)")
        else:
            raise ValueError(f"unknown MInIT seed mode {minit_seed!r}")

        for variant, (coeffs, mode) in coeff_sets.items():
            key = f"minit_{variant}"
            print(f"\n--- Running MInIT-closed coarse solver [{variant}, {mode}] ---")
            print(f"  beta_xx={coeffs.beta_xx:.4f} beta_yy={coeffs.beta_yy:.4f} "
                  f"beta_xy={coeffs.beta_xy:.4f} A_gamma={coeffs.A_gamma:.4f} C={coeffs.C:.4f}")

            solver_minit = RHDSolver(
                Nc_y, Nc_x, cfl=cfl, reconstruction=reconstruction,
                riemann_solver=riemann_solver, dtype=dtype, device=device,
            )
            solver_minit.initialize(
                coarse0["rho_bar"].to(device),
                coarse0["vx_tilde"].to(device),
                coarse0["vy_tilde"].to(device),
                coarse0["p_bar"].to(device),
            )
            solver_minit.t = snap0["t"]

            # delta_f is the filter scale, which for a coarse cell that *is* the
            # filter box equals the coarse spacing.
            closure = MInITClosure(
                dx=dx, dy=dy, delta_f=dx, coeffs=coeffs,
                shear_mode=minit_shear_mode,
                mode=mode,
                average=minit_average,
                advect=minit_advect,
                e_seed=minit_e_seed,
                floor_fraction=minit_floor_fraction,
                backscatter_clip=backscatter_clip,
                backscatter_fraction=backscatter_fraction,
                tau_smooth_passes=tau_smooth_passes,
            )
            closure.initialize(solver_minit.prims, e_KH=e_seed_field)

            ok_minit = _safe_evolve(
                solver_minit, key,
                sgs_closure=closure,
                verbose=True,
                progress_bar=True,
                desc=f"MInIT-Closed ({variant})",
                save_interval=save_int,
            )

            if ok_minit:
                prims_minit[variant] = solver_minit.get_primitives()
            minit_closures[variant] = closure
            if save_int is not None:
                history_minit[variant] = solver_minit.get_history()

    learned_jobs = [(p_, "learned") for p_ in learned_coeff]
    if hybrid_coeff:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("--with-hybrid needs the TBNN --checkpoint for its shape")
        learned_jobs.append((hybrid_coeff, "hybrid"))
        if hybrid_ablations:
            # Shape/energy ablations: swap one factor of the hybrid for the exact DNS's.
            learned_jobs += [(hybrid_coeff, "exactE"), (hybrid_coeff, "exactS")]

    for ckpt_path, kind in learned_jobs:
        is_hybrid = kind != "learned"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"learned coefficients {ckpt_path!r} does not exist. Run "
                f"`python -m src.train_minit_coeff` first."
            )
        name = Path(ckpt_path).stem.replace("minit_coeff_", "")
        if learned_gain != 1.0:
            name = f"{name}-g{learned_gain:g}"
        variant = {"learned": f"{_LEARNED_PREFIX}{name}", "hybrid": "hybrid",
                   "exactE": "ablation-exactE", "exactS": "ablation-exactS"}[kind]
        key = _variant_key(variant)
        print(f"\n--- Running {_variant_label(variant)} [{name}] ---")

        closure = LearnedMInITClosure.load(
            ckpt_path, dx, dy, device=device,
            backscatter_clip=backscatter_clip,
            backscatter_fraction=backscatter_fraction,
            tau_smooth_passes=tau_smooth_passes,
        )
        closure.cfg.gain = learned_gain
        print(f"  mode={closure.cfg.output_mode} gain={learned_gain:g} "
              f"beta_xx={closure.cfg.beta_xx:.4f} beta_yy={closure.cfg.beta_yy:.4f} "
              f"beta_xy={closure.cfg.beta_xy:.4f} "
              f"params={sum(p_.numel() for p_ in closure.net.parameters())}")

        diag_closure = closure          # carries e_KH for the diagnostics below
        if is_hybrid:
            # Same TBNN closure as the TBNN row; its energy is replaced by MiNNit's.
            tbnn_c = TBNNSGSClosure(
                model,
                ckpt["inv_mean"].to(device), ckpt["inv_std"].to(device),
                ckpt["thermo_mean"].to(device), ckpt["thermo_std"].to(device),
                dx, dy,
                backscatter_clip=backscatter_clip,
                backscatter_fraction=backscatter_fraction,
                tau_smooth_passes=tau_smooth_passes,
            )
            exact_c = None
            if kind in ("exactE", "exactS"):
                exact_c = ExactDNSSGSClosure(
                    snapshots=snapshots, coarsening_factor=r, dx=dx, dy=dy,
                    device=device, dtype=dtype,
                )
            clip = "frobenius" if hybrid_clip else None
            closure = {
                "hybrid": lambda: HybridClosure(tbnn_c, closure, clip_mode=clip),
                "exactE": lambda: HybridClosure(tbnn_c, closure, clip_mode=clip, energy_closure=exact_c),
                "exactS": lambda: HybridClosure(exact_c, closure, clip_mode=clip),
            }[kind]()

        solver_l = RHDSolver(
            Nc_y, Nc_x, cfl=cfl, reconstruction=reconstruction,
            riemann_solver=riemann_solver, dtype=dtype, device=device,
        )
        solver_l.initialize(
            coarse0["rho_bar"].to(device),
            coarse0["vx_tilde"].to(device),
            coarse0["vy_tilde"].to(device),
            coarse0["p_bar"].to(device),
        )
        solver_l.t = snap0["t"]

        ok_l = _safe_evolve(
            solver_l, key,
            sgs_closure=closure,
            verbose=True,
            progress_bar=True,
            desc=_variant_label(variant),
            save_interval=save_int,
        )
        if ok_l:
            prims_minit[variant] = solver_l.get_primitives()
        minit_closures[variant] = diag_closure
        if save_int is not None:
            history_minit[variant] = solver_l.get_history()

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

    ref = (coarse_final["rho_bar"], coarse_final["vx_tilde"], coarse_final["vy_tilde"])

    def _report(label: str, key: str, fields) -> None:
        m = spectral_metrics(*fields, *ref)
        print(f"  {label}")
        print(f"    log-spectral distance : {m['log_spectral_distance']:.4f}   (0 = perfect)")
        print(f"    band energy ratio      low {m['band_ratio_low']:.3f}  "
              f"mid {m['band_ratio_mid']:.3f}  high {m['band_ratio_high']:.3f}   (1 = perfect)")
        print(f"    total KE ratio        : {m['e_ratio_total']:.4f}")
        for name, val in m.items():
            results[f"{key}_{name}"] = val

    # Persistence: freeze the initial field. Any run that cannot beat this has not
    # earned its compute, and on a chaotic trajectory pointwise norms often cannot.
    _report("persistence (frozen initial field)", "persistence",
            (coarse0["rho_bar"], coarse0["vx_tilde"], coarse0["vy_tilde"]))

    if prims_tbnn is not None:
        _report("TBNN-closed", "tbnn",
                (prims_tbnn["rho"].cpu(), prims_tbnn["vx"].cpu(), prims_tbnn["vy"].cpu()))

    if prims_unclosed is not None:
        _report("unclosed (no SGS)", "unclosed",
                (prims_unclosed["rho"].cpu(), prims_unclosed["vx"].cpu(), prims_unclosed["vy"].cpu()))

    # Sorted, to match the order the density panels and bar chart use.
    for variant in sorted(prims_minit):
        pm = prims_minit[variant]
        _report(_variant_label(variant), _variant_key(variant),
                (pm["rho"].cpu(), pm["vx"].cpu(), pm["vy"].cpu()))

    if prims_exact is not None:
        _report("exact DNS-closed", "exact",
                (prims_exact["rho"].cpu(), prims_exact["vx"].cpu(), prims_exact["vy"].cpu()))

    # MInIT's own state is a diagnostic in its own right: an e_KH that collapsed
    # or ran away explains a result that the field metrics only describe.
    for variant in sorted(minit_closures):
        closure = minit_closures[variant]
        if closure.e_KH is not None:
            results[f"{_variant_key(variant)}_e_kh_final_mean"] = float(closure.e_KH.mean())
            results[f"{_variant_key(variant)}_e_kh_final_max"] = float(closure.e_KH.max())
            print(f"  {_variant_label(variant)} final e_KH: mean "
                  f"{float(closure.e_KH.mean()):.4e}, max {float(closure.e_KH.max()):.4e}")

    # --- Energy spectra ---
    k_dns, E_dns = kinetic_energy_spectrum(
        coarse_final["rho_bar"], coarse_final["vx_tilde"], coarse_final["vy_tilde"]
    )

    k_tbnn, E_tbnn = None, None
    if prims_tbnn is not None:
        k_tbnn, E_tbnn = kinetic_energy_spectrum(
            prims_tbnn["rho"].cpu(), prims_tbnn["vx"].cpu(), prims_tbnn["vy"].cpu()
        )

    k_unclosed, E_unclosed = None, None
    if prims_unclosed is not None:
        k_unclosed, E_unclosed = kinetic_energy_spectrum(
            prims_unclosed["rho"].cpu(), prims_unclosed["vx"].cpu(), prims_unclosed["vy"].cpu()
        )

    k_exact, E_exact = None, None
    if prims_exact is not None:
        k_exact, E_exact = kinetic_energy_spectrum(
            prims_exact["rho"].cpu(), prims_exact["vx"].cpu(), prims_exact["vy"].cpu()
        )

    spectra_minit = {
        variant: kinetic_energy_spectrum(
            pm["rho"].cpu(), pm["vx"].cpu(), pm["vy"].cpu()
        )
        for variant, pm in prims_minit.items()
    }

    # The density panels are built from the runs that are present, so a newly
    # added closure appears automatically instead of being dropped by a layout
    # written before it existed. The diagnostic row underneath stays fixed:
    # error map | spectrum | band-energy ratios.
    panel_runs: list[tuple[str, torch.Tensor, str]] = []
    if prims_unclosed is not None:
        panel_runs.append(("Unclosed (no SGS)", prims_unclosed["rho"].cpu(), "unclosed"))
    for variant in sorted(prims_minit):
        panel_runs.append((
            _variant_label(variant), prims_minit[variant]["rho"].cpu(),
            _variant_key(variant),
        ))
    if prims_tbnn is not None:
        panel_runs.append(("TBNN-Closed", prims_tbnn["rho"].cpu(), "tbnn"))
    if prims_exact is not None:
        panel_runs.append(("Exact DNS-Closed", prims_exact["rho"].cpu(), "exact"))

    n_top = 1 + len(panel_runs)
    panel_w = max(3.4, min(6.0, 26.0 / n_top))
    title_fs = 11 if n_top <= 4 else 9
    fig = plt.figure(figsize=(max(18.0, panel_w * n_top), 9.5))
    # The density row needs a wider gutter than the diagnostic row: each panel
    # carries its own colorbar, which a title will otherwise run into.
    outer = fig.add_gridspec(2, 1, hspace=0.32, height_ratios=(1.0, 1.15))
    gs_top = outer[0].subgridspec(1, n_top, wspace=0.55)
    gs_bot = outer[1].subgridspec(1, 3, wspace=0.28)

    rho_dns_np = coarse_final["rho_bar"].numpy()
    vmin = float(np.min(rho_dns_np))
    vmax = float(np.max(rho_dns_np))

    ax_dns = fig.add_subplot(gs_top[0])
    im0 = ax_dns.imshow(rho_dns_np, origin="lower", cmap="viridis",
                        aspect="equal", vmin=vmin, vmax=vmax)
    ax_dns.set_title("Filtered DNS ρ\n(Ground Truth)", fontweight="bold", fontsize=title_fs)
    plt.colorbar(im0, ax=ax_dns, shrink=0.75)

    for col, (label, rho_run, key) in enumerate(panel_runs, start=1):
        ax = fig.add_subplot(gs_top[col])
        im = ax.imshow(rho_run.numpy(), origin="lower", cmap="viridis",
                       aspect="equal", vmin=vmin, vmax=vmax)
        hi = results.get(f"{key}_band_ratio_high", float("nan"))
        ax.set_title(f"{label}\nhigh-k E/E$_{{DNS}}$: {hi:.2f}",
                     fontweight="bold", fontsize=title_fs)
        plt.colorbar(im, ax=ax, shrink=0.75)

    # [1, 0] Absolute error map for the last-listed run (the oracle when present,
    # which is the one worth eyeballing against the others' metrics).
    ax_err = fig.add_subplot(gs_bot[0])
    if panel_runs:
        err_label, err_rho, _ = panel_runs[-1]
        diff_np = (err_rho - coarse_final["rho_bar"]).abs().numpy()
        im3 = ax_err.imshow(diff_np, origin="lower", cmap="magma", aspect="equal",
                            vmin=0.0, vmax=max(0.5, float(np.max(diff_np))))
        ax_err.set_title(f"|{err_label} - DNS| Absolute Error", fontweight="bold")
        plt.colorbar(im3, ax=ax_err, shrink=0.8).set_label("Absolute Error |Δρ|")
    else:
        ax_err.axis("off")

    # [1, 1] Spectrum plot
    ax_sp = fig.add_subplot(gs_bot[1])
    ax_sp.loglog(k_dns, E_dns, "k-", lw=2, label="Filtered DNS")
    if prims_tbnn is not None and k_tbnn is not None and E_tbnn is not None:
        ax_sp.loglog(k_tbnn, E_tbnn, "r-.", lw=1.5, label="TBNN")
    if prims_unclosed is not None and k_unclosed is not None and E_unclosed is not None:
        ax_sp.loglog(k_unclosed, E_unclosed, color="tab:blue", ls="--", lw=1.5, label="Unclosed")
    # Fixed per-variant styles rather than zipping against a short list, which
    # silently drops any variant past the end of it.
    for variant, (k_m, E_m) in sorted(spectra_minit.items()):
        style = _MINIT_STYLE.get(variant, {"color": "tab:purple", "ls": "-"})
        ax_sp.loglog(k_m, E_m, lw=1.4, alpha=0.9, label=_variant_label(variant),
                     color=style["color"], ls=style["ls"])
    if prims_exact is not None and k_exact is not None and E_exact is not None:
        ax_sp.loglog(k_exact, E_exact, "g:", lw=1.8, label="Exact DNS")
    ax_sp.set_xlabel("k")
    ax_sp.set_ylabel("E(k)")
    ax_sp.set_title(f"Kinetic Energy Spectrum (t={snap_final['t']:.1f})", fontweight="bold")
    ax_sp.legend()
    ax_sp.grid(True, alpha=0.3)

    # [1, 2] Band-resolved energy ratio (1.0 = matches filtered DNS at that scale)
    ax_bar = fig.add_subplot(gs_bot[2])
    models = []
    band_mid = []
    band_high = []

    bar_entries = [("persistence", "Persistence"), ("unclosed", "Unclosed")]
    bar_entries += [(_variant_key(v), _variant_label(v).replace(" (", "\n("))
                    for v in sorted(prims_minit)]
    bar_entries += [("tbnn", "TBNN"), ("exact", "Exact DNS")]

    for key, label in bar_entries:
        if f"{key}_band_ratio_high" in results:
            models.append(label)
            band_mid.append(results[f"{key}_band_ratio_mid"])
            band_high.append(results[f"{key}_band_ratio_high"])

    if models:
        x = np.arange(len(models))
        bar_width = 0.35
        b1 = ax_bar.bar(x - bar_width / 2, band_mid, bar_width, label="mid-k", color="#4C72B0")
        b2 = ax_bar.bar(x + bar_width / 2, band_high, bar_width, label="high-k", color="#DD8452")
        ax_bar.axhline(1.0, color="k", ls="--", lw=1.2, label="filtered DNS")
        ax_bar.set_yscale("log")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(
            models, fontweight="bold",
            fontsize=8 if len(models) > 4 else 10,
            rotation=30 if len(models) > 4 else 0,
            ha="right" if len(models) > 4 else "center",
        )
        ax_bar.set_ylabel("E / E$_{DNS}$ in band")
        ax_bar.set_title(f"Band Energy Ratio (t={snap_final['t']:.1f})", fontweight="bold")
        ax_bar.legend(loc="best", fontsize=8)
        ax_bar.grid(True, axis="y", alpha=0.3)
        for rect in list(b1) + list(b2):
            h = rect.get_height()
            ax_bar.text(rect.get_x() + rect.get_width() / 2.0, h, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=8)
    else:
        ax_bar.axis("off")

    # No tight_layout: the gridspec above already sets its own spacing, and
    # letting tight_layout re-pack a nested gridspec squashes the panel titles.
    fig_path = os.path.join(output_dir, "rollout_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {fig_path}")

    if diverged_at:
        results["diverged_at"] = diverged_at
        print(f"\n[DIVERGED] Solver(s) diverged before t_end={t_end}: {diverged_at}")

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # All runs save on the same save_interval; take the LONGEST history as the time
    # axis, so a run that diverged early cannot truncate everyone else's frames.
    _cands = [h for h in (history_tbnn, history_exact, history_unclosed,
                          *[history_minit[v] for v in sorted(history_minit)])
              if h is not None and "t" in h]
    ref_history = max(_cands, key=lambda h: len(h["t"])) if _cands else None
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

    if history_dns is not None:
        t_lo = float(snap0["t"]) + band_skip
        windows = band_windows or [
            (t_lo + j * (t_end - t_lo) / 3, t_lo + (j + 1) * (t_end - t_lo) / 3) for j in range(3)
        ]
        windows = list(windows) + [(t_lo, t_end)]
        runs_w = {n: h for n, h in (("unclosed", history_unclosed), ("TBNN", history_tbnn),
                                    ("exact DNS", history_exact)) if h is not None}
        runs_w.update({_variant_label(v): history_minit[v] for v in sorted(history_minit)})
        # Nearest real snapshot per frame -- not history_dns, which blends two.
        nearest = np.abs(times[:, None] - snap_times[None, :]).argmin(axis=1)
        spacing = float(np.median(np.diff(snap_times))) if len(snap_times) > 1 else 1.0
        # Frames are saved at the first solver step past each save mark, so they sit up
        # to one coarse dt after it -- 0.007 at r=16 but 0.035 at r=64. Half the snapshot
        # spacing keeps the nearest snapshot unique and still rejects a skipped save
        # (~1 spacing off); a quarter silently dropped most r=64 frames.
        tol = 0.5 * spacing
        valid = np.abs(times - snap_times[nearest]) <= tol
        history_ref = {
            "t": times,
            **{q: torch.stack([dns_cache[j][src] for j in nearest])
               for q, src in (("rho", "rho_bar"), ("vx", "vx_tilde"), ("vy", "vy_tilde"))},
        }
        base = "exact DNS" if history_exact is not None else None
        wres = windowed_band_ratios(history_ref, runs_w, valid, windows,
                                    baseline=base, t_tol=tol)
        results["windowed_band_ratios"] = wres
        print(f"\n=== Band ratios per window: geometric mean over frames vs the nearest DNS "
              f"snapshot (first {band_skip:g} skipped) ===")
        if base:
            print("  'high/oracle' is the paired ratio to the exact-stress run with a 95% block-"
                  "bootstrap CI;\n  it, not raw retention, is the quality score (more high-k "
                  "than the oracle is not better).\n  CIs cover frame sampling along this one "
                  "trajectory only.")
        for wl, rows in wres.items():
            print(f"  t = {wl}")
            for n, rw in rows.items():
                vs = (f"  high/oracle {rw['vs_baseline_high']:.3f} "
                      f"[{rw['vs_baseline_lo']:.3f}, {rw['vs_baseline_hi']:.3f}]"
                      if "vs_baseline_high" in rw else "")
                print(f"    {n:<36} low {rw['band_ratio_low']:.3f}  mid {rw['band_ratio_mid']:.3f}"
                      f"  high {rw['band_ratio_high']:.3f} +- {rw['high_err']:.3f}{vs}"
                      f"  LSD {rw['log_spectral_distance']:.3f}  KE {rw['e_ratio_total']:.3f}"
                      f"  ({rw['n_frames']} frames{', PARTIAL' if rw['partial'] else ''})")
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(results, f, indent=2)

    if save_history and history_dns is not None and ref_history is not None:
        traj_path = os.path.join(output_dir, "rollout_trajectory.pt")
        traj_data = {
            "t": ref_history["t"] if isinstance(ref_history["t"], torch.Tensor) else torch.tensor(ref_history["t"], dtype=dtype),
            "dns": history_dns,
            "tbnn": history_tbnn,
            "exact": history_exact,
            "unclosed": history_unclosed,
            "minit": history_minit,
            "minit_e_kh": {
                v: getattr(c, "history", None)
                for v, c in minit_closures.items()
                if getattr(c, "history", None)
            },
            "grid": {"Ny": Nc_y, "Nx": Nc_x, "dx": dx, "dy": dy, "r": r},
            "diverged_at": diverged_at,
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
            history_unclosed=history_unclosed,
            history_minit=history_minit,
            output_path=anim_path,
            fps=anim_fps,
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_minit_variants(raw: list[str] | None) -> tuple[str, ...]:
    """Expand the repeatable --with-minit flag, with 'both' as a shorthand."""
    if not raw:
        return ()
    out: list[str] = []
    expand = {"both": ("fitted", "paper"), "all": ("fitted", "paper", "equilibrium")}
    for item in raw:
        for variant in expand.get(item, (item,)):
            if variant not in out:
                out.append(variant)
    return tuple(out)


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
    parser.add_argument("--with-unclosed", action="store_true",
                        help="Also run an unclosed solver (no SGS term) as a control baseline")
    parser.add_argument("--backscatter-clip", type=str, default=None, choices=["zero", "project"],
                        help="Suppress SGS backscatter (tau:S > 0). 'zero' nulls the tensor in "
                             "those cells, 'project' removes only the strain-aligned part. "
                             "Default: allow backscatter.")
    parser.add_argument("--backscatter-fraction", type=float, default=0.0,
                        help="Fraction of backscatter to RETAIN when --backscatter-clip is set. "
                             "0.0 = strict clip (default), 1.0 = no clipping. Use to trade "
                             "stability against 2D inverse-cascade physics.")
    parser.add_argument("--tau-smooth-passes", type=int, default=0,
                        help="Low-pass the SGS stress with an (1,2,1)/4 filter this many times "
                             "before applying it. Removes near-grid-scale tau content that the "
                             "flux divergence can inject but not damp. Try 1 or 2.")
    parser.add_argument("--face-closure", action="store_true",
                        help="Evaluate the exact SGS stress on cell faces via staggered filter "
                             "boxes, making its divergence compact. Without this the stress is "
                             "cell-centred and interpolated, which has zero response at 2dx. "
                             "Exact-DNS closure only; requires an even coarsening factor.")
    parser.add_argument("--with-minit", dest="minit_variants", action="append",
                        choices=["fitted", "paper", "equilibrium", "both", "all"],
                        default=None,
                        help="Add a MInIT-closed run. 'fitted' uses coefficients refit "
                             "on this DNS by src.minit_calibrate; 'paper' transfers the "
                             "published 3D Newtonian MHD calibration verbatim (their "
                             "difference separates the model from its calibration); "
                             "'equilibrium' drops the turbulent-energy ODE for its fixed "
                             "point, which scores best a-priori. 'both' = fitted+paper, "
                             "'all' = all three. Repeatable.")
    parser.add_argument("--with-learned", dest="learned_coeff", action="append",
                        default=None, metavar="CKPT",
                        help="Add a learned-coefficient MInIT run from a "
                             "src.train_minit_coeff checkpoint (e.g. "
                             "results/minit/minit_coeff_residual.pt). Unlike the "
                             "--with-minit rows this closure is stateless and takes "
                             "no e_KH seed -- it reads only resolved state. "
                             "Repeatable.")
    parser.add_argument("--band-windows", nargs="+", default=None, metavar="T0:T1",
                        help="Time windows for the windowed band ratios (needs --save-history "
                             "or --make-anim). Default: thirds of [t_start + band_skip, t_end], "
                             "plus the whole span.")
    parser.add_argument("--perturb-seed", type=int, default=0,
                        help="Perturb the filtered initial state (0 = off). Each seed is one "
                             "ensemble member; all runs in one invocation share it.")
    parser.add_argument("--perturb-amplitude", type=float, default=1e-3,
                        help="Perturbation size as a fraction of the rms velocity")
    parser.add_argument("--band-skip", type=float, default=0.5,
                        help="Skip this long after t_start in the windowed band ratios: every "
                             "run, closed or not, loses ~70%% of its high-k energy in the "
                             "first ~0.7 time units as the coarse solver adjusts.")
    parser.add_argument("--with-hybrid", dest="hybrid_coeff", default=None, metavar="CKPT",
                        help="Add a hybrid run: the TBNN's (--checkpoint) tensor shape "
                             "scaled cell by cell to the energy of this MiNNit checkpoint.")
    parser.add_argument("--hybrid-ablations", action="store_true",
                        help="With --with-hybrid, also run TBNN shape x exact energy and "
                             "exact shape x MiNNit energy (FEASIBILITY §23).")
    parser.add_argument("--hybrid-no-clip", dest="hybrid_clip", action="store_false",
                        help="Disable the realizability clip after the hybrid's rescale "
                             "(the pre-fix behaviour; it diverged in 1 of 14 rollouts).")
    parser.add_argument("--learned-gain", type=float, default=1.0,
                        help="Multiply the learned e_KH by this factor. The a-priori "
                             "optimum is ~0.96, so this is not calibration: it "
                             "deliberately over-drives the closure to offset the coarse "
                             "solver's own numerical dissipation, which even the exact "
                             "stress cannot undo (the oracle retains only 34%% of the "
                             "DNS high-k energy at t=10).")
    parser.add_argument("--minit-average", choices=["cell", "x"], default="cell",
                        help="'cell': Eq. 12 cell by cell. 'x': the paper's mean-field form, "
                             "growth rate from the x-averaged shear and e_KH uniform along x "
                             "(use coefficients calibrated with minit_calibrate --average x).")
    parser.add_argument("--minit-coefficients", type=str,
                        default="results/minit/minit_coefficients.json",
                        help="Refit coefficients written by src.minit_calibrate")
    parser.add_argument("--minit-seed", type=str, default="dns", choices=["dns", "uniform"],
                        help="e_KH = 0 is a fixed point, so MInIT needs a seed. 'dns' "
                             "takes the exact filtered-DNS subgrid energy at t_start "
                             "(one snapshot of ground truth, as the oracle also gets); "
                             "'uniform' is self-contained. Different experiments.")
    parser.add_argument("--minit-e-seed", type=float, default=1e-6,
                        help="Uniform seed value for --minit-seed uniform")
    parser.add_argument("--minit-floor-fraction", type=float, default=0.0,
                        help="Hold e_KH at this fraction of its local saturation value "
                             "so a decayed region can restart (0 disables)")
    parser.add_argument("--minit-advect", action="store_true",
                        help="Include the advective term the paper drops. Its periodicity "
                             "argument holds along x here but not across the shear layer.")
    parser.add_argument("--minit-shear-mode", type=str, default="dvxdy",
                        choices=["dvxdy", "strain"],
                        help="Growth-rate estimator: Eq. 23 literally, or its "
                             "rotationally-invariant equivalent")
    parser.add_argument("--sgs-coupling", type=str, default="flux", choices=["source", "flux"],
                        help="SGS coupling mode: 'source' (legacy) or 'flux' (conservative viscous flux)")
    parser.add_argument("--save-history", action="store_true", help="Save intermediate snapshot trajectories to rollout_trajectory.pt")
    parser.add_argument("--make-anim", action="store_true", help="Generate an animated comparison GIF across the rollout")
    parser.add_argument("--anim-interval", type=float, default=0.1, help="Simulation time interval between animation frames")
    parser.add_argument("--anim-fps", type=int, default=10, help="Frames per second for generated animation")
    add_config_arguments(parser)
    cfg = apply_config_defaults(parser, {
        "data_dir": "data.data_dir",
        "coarsening_factor": "data.coarsening_factor",
        "cfl": "solver.cfl",
        "reconstruction": "solver.reconstruction",
        "riemann_solver": "solver.riemann_solver",
    })
    args = parser.parse_args()
    configure(args, cfg)

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
        backscatter_clip=args.backscatter_clip,
        backscatter_fraction=args.backscatter_fraction,
        tau_smooth_passes=args.tau_smooth_passes,
        face_closure=args.face_closure,
        with_unclosed=args.with_unclosed,
        minit_variants=_resolve_minit_variants(args.minit_variants),
        minit_coefficients=args.minit_coefficients,
        minit_seed=args.minit_seed,
        minit_e_seed=args.minit_e_seed,
        minit_floor_fraction=args.minit_floor_fraction,
        minit_advect=args.minit_advect,
        minit_shear_mode=args.minit_shear_mode,
        minit_average=args.minit_average,
        learned_coeff=tuple(args.learned_coeff or ()),
        learned_gain=args.learned_gain,
        hybrid_coeff=args.hybrid_coeff,
        hybrid_clip=args.hybrid_clip,
        hybrid_ablations=args.hybrid_ablations,
        band_windows=[tuple(float(x) for x in w.split(":")) for w in args.band_windows]
        if args.band_windows else None,
        band_skip=args.band_skip,
        perturb_seed=args.perturb_seed,
        perturb_amplitude=args.perturb_amplitude,
        save_history=args.save_history,
        make_anim=args.make_anim,
        anim_interval=args.anim_interval,
        anim_fps=args.anim_fps,
        sgs_coupling=args.sgs_coupling,
    )
