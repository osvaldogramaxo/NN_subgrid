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

Batching
--------
Batches are whole **snapshots**, not shuffled grid points. On a 1024² DNS at
r = 16 a snapshot is a 64×64 coarse field — 4096 points — so a handful of them
fit in one step, and the spectral cascade term can be evaluated on the exact
periodic domain inside the same forward pass as the point-wise terms. Every
term then contributes to one ``backward()`` and one optimizer step.

The previous point-shuffled batching made that impossible: locality was
destroyed, so L_spectral had to run as a separate pass with its own
``zero_grad``/``backward``/``step`` after the epoch's point-wise steps. That
made the final update of each epoch a spectral-only gradient from four random
snapshots, and it was the dominant source of validation noise (train
L_spectral CV 44%, val_rel_l2 CV 8.7%, LR reduced four times on what was
mostly sampling noise).

Splitting
---------
Train/val come from *interleaved* snapshots of the same time window (every
``val_every``-th snapshot is held out), because the flow is strongly
non-stationary: on `dns_khi` the subgrid fraction of kinetic energy rises from
0.02% at t = 0 to 0.96% at t = 8. A contiguous late-time validation window
measures a different regime, and combined with self-normalizing losses made
train and val numbers incomparable. A contiguous tail (``test_time``) is held
out separately and never used for selection.

Training uses AdamW with gradient clipping and, by default, a cosine LR
schedule with warmup. ReduceLROnPlateau remains available but is only sound
once the monitored metric is smooth; see ``--lr-schedule`` and ``--monitor-ema``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import deque
import os
import sys
import time
from pathlib import Path
from collections.abc import Sequence
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
from src.config import add_config_arguments, apply_config_defaults, configure  # noqa: E402
from utils import CRMHD_TBNN
from src.dataset import SGSDataset, create_dataloaders
from src.features import extract_tbnn_features
from src.losses import RMHDSGSLoss

# Cache format version — bump when adding new stored fields to force regeneration
CACHE_VERSION = 4


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
            read_snapshot, snapshot_files, relativistic_favre_filter, compute_sgs_stress,
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
                self.p_scale = cached["p_scale"]
                self.snapshot_boundaries = cached["snapshot_boundaries"]
                self.times = cached["times"]
                return
            else:
                print(f"  Cache v{cached.get('cache_version', 1)} outdated (need v{CACHE_VERSION}). Regenerating...")

        # One snapshot in memory at a time: a 2048^2 run is ~27 GB of primitives.
        files = snapshot_files(data_dir, time_range, stride)
        if not files:
            raise ValueError(f"No snapshots in time range {time_range}")
        snapshots = (read_snapshot(f) for f in files)

        print(f"  Processing {len(files)} snapshots (stride={stride})...")

        all_inv = []
        all_thermo = []
        all_basis = []
        all_tau = []
        all_u_con = []
        all_grad_u = []
        all_rho = []
        all_p = []
        snapshot_boundaries: list[tuple[int, int, int]] = []  # (start_idx, Ny, Nx)
        times: list[float] = []
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
            times.append(float(snap["t"]))
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
        self.times = times

        # Normalization
        self.inv_mean = self.invariants.mean(dim=0)
        self.inv_std = self.invariants.std(dim=0).clamp(min=1e-12)
        self.thermo_mean = self.thermo.mean(dim=0)
        self.thermo_std = self.thermo.std(dim=0).clamp(min=1e-12)

        # Dataset-level physical scales. These make the loss components
        # dimensionless; see RMHDSGSLoss.set_scales. tau_scale is the mean
        # Frobenius norm of the exact SGS stress -- note src.dataset.SGSDataset
        # used to define its own tau_scale as abs().mean(), a ~7x smaller
        # number, and neither was ever used for anything. One definition now.
        self.tau_scale = self.tau_sgs.norm(dim=(-2, -1)).mean()
        P_dns = -torch.einsum("nmk,nmk->n", self.tau_sgs, self.grad_u_cov)
        self.p_scale = P_dns.abs().mean()

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
                "tau_scale": self.tau_scale, "p_scale": self.p_scale,
                "snapshot_boundaries": self.snapshot_boundaries,
                "times": self.times,
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

    @property
    def num_snapshots(self) -> int:
        return len(self.snapshot_boundaries)

    def snapshot_view(self, indices: Sequence[int]) -> "SnapshotView":
        """A non-copying view over a subset of snapshots.

        Used for the train/val/test split, so all splits share one
        preprocessing pass, one cache and one set of normalization statistics.
        """
        return SnapshotView(self, indices)


class SnapshotView:
    """Snapshot-granular, non-copying view over a :class:`TBNNDataset`.

    Iterating yields *groups of whole snapshots*. Each group is one optimizer
    step: the point-wise loss terms are evaluated on all of the group's points
    and the spectral term on each of its snapshots as a full periodic 2D field,
    all inside a single forward/backward.

    ``ordered_points`` gives the group's flat feature tensors plus the
    per-snapshot ``(offset, Ny, Nx)`` needed to reshape predictions back to 2D.
    """

    def __init__(self, base: TBNNDataset, indices: Sequence[int]):
        self.base = base
        self.indices = [int(i) for i in indices]
        if not self.indices:
            raise ValueError("SnapshotView requires at least one snapshot index")
        n = base.num_snapshots
        bad = [i for i in self.indices if not (0 <= i < n)]
        if bad:
            raise IndexError(f"snapshot indices out of range for {n} snapshots: {bad}")
        self.times = [base.times[i] for i in self.indices]
        self.snapshot_boundaries = [base.snapshot_boundaries[i] for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def num_points(self) -> int:
        return sum(Ny * Nx for _, Ny, Nx in self.snapshot_boundaries)

    def point_index(self) -> torch.Tensor:
        """Flat indices into the base dataset for every point in this view.

        Used to compute split-local statistics (the τ and P scales) without
        materialising a copy of the fields.
        """
        chunks = [
            torch.arange(start, start + Ny * Nx, dtype=torch.long)
            for start, Ny, Nx in self.snapshot_boundaries
        ]
        return torch.cat(chunks) if len(chunks) > 1 else chunks[0]

    def groups(
        self,
        snapshots_per_group: int,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
    ) -> list[list[int]]:
        """Partition the view's snapshots into groups of at most n.

        No ``drop_last``: every snapshot is seen every epoch. With 60-odd
        training snapshots and a group of 2 that is ~30 steps per epoch, each
        from ~8k points -- far fewer, far less noisy steps than the previous
        162 steps of 2048 shuffled points.
        """
        k = max(1, int(snapshots_per_group))
        order = list(range(len(self.indices)))
        if shuffle:
            perm = torch.randperm(len(order), generator=generator).tolist()
            order = [order[i] for i in perm]
        return [order[i:i + k] for i in range(0, len(order), k)]

    def gather(
        self,
        local_indices: Sequence[int],
        device: torch.device | None = None,
    ) -> dict[str, Any]:
        """Concatenated fields for a group of snapshots (by *local* index).

        Returns flat tensors of shape (P, ...) where P is the group's total
        point count, plus ``shapes`` = [(offset, Ny, Nx), ...] into that flat
        axis so predictions can be reshaped to 2D for the spectral term.
        """
        b = self.base
        parts: dict[str, list[torch.Tensor]] = {
            k: [] for k in
            ("invariants", "thermo", "basis", "tau_sgs", "u_con", "grad_u_cov", "rho", "p")
        }
        shapes: list[tuple[int, int, int]] = []
        offset = 0
        for li in local_indices:
            gi = self.indices[li]
            start, Ny, Nx = b.snapshot_boundaries[gi]
            N = Ny * Nx
            sl = slice(start, start + N)
            for k in parts:
                parts[k].append(getattr(b, k)[sl])
            shapes.append((offset, Ny, Nx))
            offset += N

        out: dict[str, Any] = {}
        for k, chunks in parts.items():
            t = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
            out[k] = t.to(device) if device is not None else t
        out["shapes"] = shapes
        out["times"] = [self.times[li] for li in local_indices]

        # Per-snapshot L_tensor floor, expanded to one value per point. A single
        # global floor is only sound while the subgrid intensity is roughly
        # stationary across the training window; over dns_khi's full t = 0..20
        # it varies ~40x, and every cell of the weak snapshots then falls below
        # it, where the term degenerates to an absolute MSE and predicting zero
        # is near-optimal. See CovariantTensorLoss.
        floors = getattr(b, "snapshot_tau_floor", None)
        if floors is not None:
            per_point = torch.cat([
                floors[self.indices[li]].expand(
                    b.snapshot_boundaries[self.indices[li]][1]
                    * b.snapshot_boundaries[self.indices[li]][2]
                )
                for li in local_indices
            ])
            out["tau_floor"] = (
                per_point.to(device) if device is not None else per_point
            )
        return out


# ---------------------------------------------------------------------------
# Monitored-metric smoothing
# ---------------------------------------------------------------------------

class EMAMonitor:
    r"""Exponential moving average of the monitored validation metric.

    Everything that *decides* something -- best-checkpoint selection, LR
    reduction, early stopping -- reads the smoothed value, while the raw value
    is still logged.

    Without this, selection is dominated by noise. Over epochs 100-150 of the
    previous run ``val_rel_l2`` had mean 19.4% and sd 1.7%; the checkpoint kept
    as "best" was the 16.3% at epoch 107, a -1.8σ draw rather than a better
    model. The same noise drove all four ReduceLROnPlateau reductions.

    .. math:: s_n = \lambda s_{n-1} + (1 - \lambda) x_n

    starting from :math:`s_0 = 0` with bias correction
    :math:`\hat{s}_n = s_n / (1 - \lambda^n)`, so the very first epochs are not
    dragged toward the cold start.

    Parameters
    ----------
    beta : float
        Smoothing factor λ in [0, 1). 0 disables smoothing (returns the raw
        value); 0.8 averages over roughly the last 5 epochs.
    """

    def __init__(self, beta: float = 0.8):
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"EMA beta must be in [0, 1) (got {beta})")
        self.beta = beta
        self.state = 0.0
        self.n = 0

    def update(self, value: float) -> float:
        """Push a raw observation, return the smoothed value."""
        v = float(value)
        self.n += 1
        if self.beta == 0.0:
            self.state = v
            return v
        self.state = self.beta * self.state + (1.0 - self.beta) * v
        return self.state / (1.0 - self.beta ** self.n)

    @property
    def value(self) -> float | None:
        """Current smoothed value, or None before the first observation."""
        if self.n == 0:
            return None
        if self.beta == 0.0:
            return self.state
        return self.state / (1.0 - self.beta ** self.n)

    def state_dict(self) -> dict[str, Any]:
        return {"beta": self.beta, "state": self.state, "n": self.n}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.beta = float(d.get("beta", self.beta))
        self.state = float(d.get("state", 0.0) or 0.0)
        self.n = int(d.get("n", 0))


class AdaptiveGradClip:
    r"""Gradient-norm clip set from a running median of observed norms.

    A fixed threshold is the wrong tool for this objective. ``L_tensor`` is a
    ratio whose denominator is floored at the median :math:`\lVert\tau\rVert_F`
    (~1.3e-3 on `dns_khi`), so :math:`\partial L / \partial \tau` is
    intrinsically O(10³) and the gradient norm runs 10¹–10³ through a healthy
    run. A clip of 1.0 therefore fires on *every* step: the update keeps the
    gradient's direction and throws its magnitude away, which silently turns
    AdamW into a normalized-step method and hides genuine spikes among the
    ordinary steps.

    Clipping at ``mult`` times the running median of recent norms keeps the
    intended behaviour -- catch the outlier, leave the typical step alone --
    without needing to know the objective's gradient scale in advance.

    Parameters
    ----------
    mult : float
        Threshold as a multiple of the running median. 5 leaves ordinary steps
        untouched and truncates genuine spikes.
    window : int
        Number of recent norms the median is taken over.
    warmup_steps : int
        Steps observed before clipping starts, so the median is meaningful.
        During warmup a generous fallback of ``mult × the largest norm seen``
        still guards against a blow-up.
    """

    def __init__(self, mult: float = 5.0, window: int = 100, warmup_steps: int = 20):
        self.mult = float(mult)
        self.window = int(window)
        self.warmup_steps = int(warmup_steps)
        self._norms: deque[float] = deque(maxlen=self.window)
        self._max_seen = 0.0

    @property
    def threshold(self) -> float | None:
        """Current clip threshold, or None while still warming up."""
        if len(self._norms) < self.warmup_steps:
            return self.mult * self._max_seen if self._max_seen > 0 else None
        return self.mult * float(statistics.median(self._norms))

    def __call__(self, model: nn.Module) -> tuple[float, bool]:
        """Clip ``model``'s gradients in place. Returns (pre-clip norm, clipped)."""
        thr = self.threshold
        if thr is None or thr <= 0:
            # Nothing to clip against yet: measure only.
            total = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            norm = float(total)
        else:
            total = torch.nn.utils.clip_grad_norm_(model.parameters(), thr)
            norm = float(total)
        self._norms.append(norm)
        self._max_seen = max(self._max_seen, norm)
        return norm, bool(thr is not None and norm > thr)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mult": self.mult, "window": self.window,
            "warmup_steps": self.warmup_steps,
            "norms": list(self._norms), "max_seen": self._max_seen,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.mult = float(d.get("mult", self.mult))
        self.window = int(d.get("window", self.window))
        self.warmup_steps = int(d.get("warmup_steps", self.warmup_steps))
        self._norms = deque(
            [float(x) for x in d.get("norms", [])], maxlen=self.window
        )
        self._max_seen = float(d.get("max_seen", 0.0))


def build_lr_schedule(
    optimizer: torch.optim.Optimizer,
    kind: str,
    epochs: int,
    warmup_epochs: int = 5,
    min_lr_frac: float = 0.01,
    lr_patience: int = 25,
) -> tuple[Any, bool]:
    """Create the LR schedule.

    Returns ``(scheduler, is_plateau)``; a plateau scheduler needs the
    monitored metric passed to ``step``, a cosine one does not.

    Cosine-with-warmup is the default. ReduceLROnPlateau is only sound when
    the monitored metric is smooth enough that a plateau is distinguishable
    from a fluctuation; with the previous noise level it fired on sampling
    noise. Warmup matters here because the first steps see a randomly
    initialised prediction whose spectrum sits decades below the DNS one.
    """
    if kind == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=lr_patience, min_lr=1e-7,
            ),
            True,
        )
    if kind != "cosine":
        raise ValueError(f"unknown lr schedule {kind!r} (expected 'cosine' or 'plateau')")

    warm = max(0, int(warmup_epochs))
    total = max(1, int(epochs))

    def lr_lambda(epoch: int) -> float:
        if warm > 0 and epoch < warm:
            return (epoch + 1) / warm
        progress = (epoch - warm) / max(1, total - warm)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), False


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

def _forward_group(
    model: nn.Module,
    group: dict[str, Any],
    inv_mean: torch.Tensor,
    inv_std: torch.Tensor,
    thermo_mean: torch.Tensor,
    thermo_std: torch.Tensor,
) -> torch.Tensor:
    """Predict τ^{μν} for every point of a snapshot group. Returns (P, 4, 4)."""
    inv_norm = (group["invariants"] - inv_mean) / inv_std
    thermo_norm = (group["thermo"] - thermo_mean) / thermo_std
    delta_T, _coeffs, _p_sgs = model(inv_norm, thermo_norm, group["basis"])
    return delta_T


def _group_loss(
    criterion: RMHDSGSLoss,
    tau_pred: torch.Tensor,
    group: dict[str, Any],
    g_cov: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Full composite loss for one snapshot group, spectral term included.

    This is the whole point of snapshot batching: L_spectral is evaluated on
    each snapshot's exact periodic 64×64 field inside the same graph as the
    point-wise terms, so one ``backward()`` carries every component and the
    optimizer takes exactly one step per group.
    """
    total, comps = criterion(
        tau_pred, group["tau_sgs"], group["u_con"], group["grad_u_cov"],
        g_cov, group["rho"], group["p"],
        tau_floor=group.get("tau_floor"),
    )

    spec_val = 0.0
    if criterion.beta > 0:
        shapes = group["shapes"]
        spec_terms = []
        for offset, Ny, Nx in shapes:
            N = Ny * Nx
            pred_2d = tau_pred[offset:offset + N].reshape(Ny, Nx, 4, 4)
            dns_2d = group["tau_sgs"][offset:offset + N].reshape(Ny, Nx, 4, 4)
            spec_terms.append(criterion.spectral_term(pred_2d, dns_2d))
        spec_weighted = torch.stack(spec_terms).mean()
        total = total + spec_weighted
        spec_val = float(spec_weighted.detach()) / criterion.beta

    comps["L_spectral"] = spec_val
    return total, comps


def train_epoch(
    model: nn.Module,
    view: "SnapshotView",
    optimizer: torch.optim.Optimizer,
    criterion: RMHDSGSLoss,
    inv_mean: torch.Tensor,
    inv_std: torch.Tensor,
    thermo_mean: torch.Tensor,
    thermo_std: torch.Tensor,
    device: torch.device,
    g_cov: torch.Tensor,
    snapshots_per_batch: int = 2,
    grad_clip: float | AdaptiveGradClip | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Train for one epoch over snapshot groups.

    One group → one forward → one composite loss (point-wise *and* spectral)
    → one ``backward()`` → one clipped optimizer step. No term gets its own
    pass, and nothing is applied after the last group, so the model state that
    validation sees is the state the epoch's optimization actually produced.

    Returns dict with the point-weighted mean total loss and components.
    """
    model.train()
    total_loss = 0.0
    count = 0
    comp_accum: dict[str, float] = {}
    grad_norm_sum = 0.0
    n_clipped = 0
    n_steps = 0

    groups = view.groups(snapshots_per_batch, shuffle=True, generator=generator)
    pbar = tqdm(groups, desc="  Train Group", leave=False, disable=not sys.stdout.isatty())
    for local_idx in pbar:
        group = view.gather(local_idx, device=device)
        tau_pred = _forward_group(
            model, group, inv_mean, inv_std, thermo_mean, thermo_std
        )
        loss, components = _group_loss(criterion, tau_pred, group, g_cov)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if isinstance(grad_clip, AdaptiveGradClip):
            gn, clipped = grad_clip(model)
            grad_norm_sum += gn
            n_clipped += int(clipped)
        elif grad_clip is not None and grad_clip > 0:
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
            grad_norm_sum += gn
            n_clipped += int(gn > grad_clip)
        n_steps += 1
        optimizer.step()

        bs = tau_pred.shape[0]
        total_loss += loss.item() * bs
        count += bs
        for k, v in components.items():
            comp_accum[k] = comp_accum.get(k, 0.0) + v * bs

    result = {"train_loss": total_loss / max(count, 1)}
    for k, v in comp_accum.items():
        result[k] = v / max(count, 1)
    result["grad_norm"] = grad_norm_sum / max(n_steps, 1)
    # Reported because a clip that fires on every step is not a safety net: it
    # replaces the gradient magnitude with a constant. Watch this stay low.
    result["clip_frac"] = n_clipped / max(n_steps, 1)
    result["steps"] = float(n_steps)
    return result


@torch.no_grad()
def validate(
    model: nn.Module,
    view: "SnapshotView",
    criterion: RMHDSGSLoss,
    inv_mean: torch.Tensor,
    inv_std: torch.Tensor,
    thermo_mean: torch.Tensor,
    thermo_std: torch.Tensor,
    device: torch.device,
    g_cov: torch.Tensor,
    snapshots_per_batch: int = 2,
) -> dict[str, float]:
    """Evaluate on a snapshot view, using the same objective as training.

    Deterministic: groups are taken in order and every snapshot is visited
    once, so the only epoch-to-epoch variation in the reported numbers comes
    from the model changing.

    Returns val_loss, rel_l2, raw mse, and per-component values.
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_norm_sq = 0.0
    total_diff_sq = 0.0
    count = 0
    comp_accum: dict[str, float] = {}

    for local_idx in view.groups(snapshots_per_batch, shuffle=False):
        group = view.gather(local_idx, device=device)
        tau_pred = _forward_group(
            model, group, inv_mean, inv_std, thermo_mean, thermo_std
        )
        loss, components = _group_loss(criterion, tau_pred, group, g_cov)

        bs = tau_pred.shape[0]
        total_loss += loss.item() * bs
        count += bs

        diff = tau_pred - group["tau_sgs"]
        total_diff_sq += diff.pow(2).sum().item()
        total_mse += diff.pow(2).sum().item()
        total_norm_sq += group["tau_sgs"].pow(2).sum().item()

        for k, v in components.items():
            comp_accum[k] = comp_accum.get(k, 0.0) + v * bs

    n_elem = count * 16  # 4x4 tensor

    result: dict[str, float] = {
        "val_loss": total_loss / max(count, 1),
        # Relative L2 is the headline accuracy number. Raw MSE is kept only
        # for continuity: it reads ~1e-7 purely because ‖τ‖ ~ 7e-3, and says
        # nothing about accuracy on its own.
        "mse": total_mse / max(n_elem, 1),
        "rel_l2": (total_diff_sq / max(total_norm_sq, 1e-30)) ** 0.5,
    }
    for k, v in comp_accum.items():
        result[f"val_{k}"] = v / max(count, 1)
    return result


def split_snapshots(
    times: Sequence[float],
    time_range: tuple[float, float] | None = None,
    val_every: int = 5,
    test_time: tuple[float, float] | None = None,
) -> dict[str, list[int]]:
    """Interleaved train/val split plus a contiguous held-out test window.

    Snapshots inside ``test_time`` are removed first and never used for
    training or model selection. Of what remains inside ``time_range``, every
    ``val_every``-th snapshot goes to validation and the rest to training.

    Interleaving is not a stylistic choice here. The flow is strongly
    non-stationary -- the subgrid fraction of kinetic energy on `dns_khi`
    climbs from 0.02% at t = 0 to 0.96% at t = 8 -- so a contiguous validation
    window is a different physical regime from the training data. The earlier
    split (train t ∈ [1, 70], val t ∈ [70, 85]) also produced train and val τ
    scales differing by 2.2x, which alone accounted for the apparent train/val
    loss gap.

    Returns
    -------
    dict with keys "train", "val", "test" -> lists of snapshot indices.
    """
    if val_every < 2:
        raise ValueError(f"val_every must be >= 2 (got {val_every})")

    idx_all = list(range(len(times)))
    test: list[int] = []
    if test_time is not None:
        t0, t1 = test_time
        test = [i for i in idx_all if t0 <= times[i] <= t1]

    pool = [i for i in idx_all if i not in set(test)]
    if time_range is not None:
        t0, t1 = time_range
        pool = [i for i in pool if t0 <= times[i] <= t1]

    val = [g for k, g in enumerate(pool) if k % val_every == val_every // 2]
    train = [g for g in pool if g not in set(val)]

    if not train:
        raise ValueError(
            f"Empty training split: time_range={time_range}, test_time={test_time}, "
            f"available times {min(times) if times else None}..{max(times) if times else None}"
        )
    if not val:
        raise ValueError(
            f"Empty validation split: {len(pool)} snapshots in range but val_every={val_every} "
            "selected none. Lower --val-every or widen --time-range."
        )
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Plotting & Diagnostics
# ---------------------------------------------------------------------------

def plot_loss_curves(
    history: dict[str, list[float]],
    save_path: str | Path,
    include_mse: bool = False,
) -> None:
    """Plot training/validation loss curves, relative L2 error, components, gradient norm and LR.

    Features:
    - The smoothed validation metric (``val_loss_ema``) is drawn over the raw
      one wherever both exist. The smoothed curve is what selects the best
      checkpoint and drives the LR schedule, so a plot showing only the raw
      trace invites reading noise as signal.
    - Logarithmic Relative L2 Error to resolve early transients and the
      settled regime, with a linear-% inset.
    - Gradient-norm panel (pre-clipping), so a run pinned at the clip
      threshold is visible rather than inferred.
    - Vertical demarcations at LR transitions, but only for step-like
      schedules: a cosine schedule changes every epoch, and marking all of
      them would paint the whole figure.
    - Raw MSE is off by default (``include_mse=False``). It reads ~1e-7 only
      because ‖τ‖ ~ 7e-3 and carries no accuracy information on its own;
      relative L2 is the headline number.
    """
    n_epochs = len(history.get("train_loss", []))
    if n_epochs == 0:
        return

    epochs = np.arange(1, n_epochs + 1)
    has_components = "L_tensor" in history and len(history["L_tensor"]) > 0
    has_adaptive = "alpha" in history and len(history["alpha"]) > 0
    has_mse = include_mse and "val_mse" in history and len(history["val_mse"]) > 0
    has_grad = "grad_norm" in history and len(history["grad_norm"]) == n_epochs \
        and any(v > 0 for v in history["grad_norm"])

    cols = ["loss"]
    if has_mse:
        cols.append("mse")
    cols.append("l2")
    if has_components:
        cols.append("components")
    elif has_adaptive:
        cols.append("adaptive")
    if has_grad:
        cols.append("grad")
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
    # A continuous schedule changes the LR every epoch; marking each change
    # would obliterate the figure. Only step-like schedules get regime lines.
    if len(lr_change_epochs) > max(8, 0.2 * n_epochs):
        lr_change_epochs = []

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
    has_ema = "val_loss_ema" in history and len(history["val_loss_ema"]) == n_epochs
    val_ema = np.array(history["val_loss_ema"]) if has_ema else None
    ax_loss.plot(epochs, train_loss, "b-", label="Train Loss", lw=1.6)
    ax_loss.plot(epochs, val_loss, "r--", label=f"Val Loss ({val_key}, raw)", lw=1.1, alpha=0.45)
    if val_ema is not None:
        ax_loss.plot(epochs, val_ema, color="darkred", lw=2.0,
                     label="Val Loss (EMA — drives selection)")
    mark_regimes(ax_loss)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Composite Physics Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="lower left", fontsize=9, framealpha=0.85)

    if has_inset and np.any(settled_mask):
        axins_loss = ax_loss.inset_axes([0.44, 0.44, 0.52, 0.50])
        axins_loss.plot(epochs[settled_mask], train_loss[settled_mask], "b-", lw=1.2)
        axins_loss.plot(epochs[settled_mask], val_loss[settled_mask], "r--", lw=1.0, alpha=0.45)
        if val_ema is not None:
            axins_loss.plot(epochs[settled_mask], val_ema[settled_mask], color="darkred", lw=1.5)
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
        ax_mse.set_title("Validation MSE (raw, scale-dependent)")
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
        # A component that is identically zero for the whole run is named in
        # the subtitle instead of plotted: floored onto a log axis it is a flat
        # line at the bottom that stretches the range over 15 dead decades and
        # compresses the terms that are actually doing something. L_ortho at
        # gamma=0 and an inactive L_WEC hinge are both legitimately zero.
        inactive: list[str] = []
        for k, lbl, col in comp_configs:
            if k in history and len(history[k]) == n_epochs:
                arr = np.array(history[k])
                if not np.any(arr > 0):
                    inactive.append(lbl)
                    continue
                ax_comp.semilogy(epochs, np.maximum(arr, 1e-30), color=col, lw=1.4, label=lbl)
        mark_regimes(ax_comp)
        ax_comp.set_xlabel("Epoch")
        ax_comp.set_ylabel("Raw Component Value")
        title = "Physics Loss Components"
        if inactive:
            title += "\n" + ", ".join(inactive) + " ≡ 0 (not shown)"
        ax_comp.set_title(title, fontsize=10 if inactive else None)
        ax_comp.grid(True, which="both", alpha=0.3)
        if ax_comp.get_legend_handles_labels()[0]:
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

    # 5. Gradient norm (pre-clipping) and the fraction of steps clipped
    if has_grad:
        ax_g = axes[cols.index("grad")]
        gn = np.array(history["grad_norm"])
        ax_g.semilogy(epochs, np.maximum(gn, 1e-30), color="teal", lw=1.5,
                      label="mean |∇| per step")
        mark_regimes(ax_g)
        ax_g.set_xlabel("Epoch")
        ax_g.set_ylabel("Gradient norm (pre-clip)")
        ax_g.set_title("Gradient Norm")
        ax_g.grid(True, which="both", alpha=0.3)

        # A clip firing on most steps means the update magnitude is the
        # threshold, not the gradient. Overlaid so it cannot go unnoticed.
        if "clip_frac" in history and len(history["clip_frac"]) == n_epochs:
            ax_clip = ax_g.twinx()
            cf = np.array(history["clip_frac"]) * 100.0
            ax_clip.plot(epochs, cf, color="crimson", lw=1.2, linestyle="--",
                         label="% steps clipped")
            ax_clip.set_ylabel("Steps clipped (%)", color="crimson")
            ax_clip.tick_params(axis="y", labelcolor="crimson")
            ax_clip.set_ylim(-2, 102)
            h1, l1 = ax_g.get_legend_handles_labels()
            h2, l2 = ax_clip.get_legend_handles_labels()
            ax_g.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, framealpha=0.85)
        else:
            ax_g.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # 6. Learning rate schedule
    ax_lr = axes[cols.index("lr")]
    if len(lr_arr) == n_epochs:
        ax_lr.plot(epochs, lr_arr, color="darkmagenta", lw=1.8)
        if np.min(lr_arr) > 0:
            ax_lr.set_yscale("log")
        # Annotate only step transitions; a continuous schedule has none
        # flagged, so this stays readable.
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
    ax_lr.set_title("LR Schedule")
    ax_lr.grid(True, which="both", alpha=0.3)

    plt.savefig(save_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_pred_vs_truth(
    model: nn.Module,
    val_view: "SnapshotView",
    inv_mean: torch.Tensor,
    inv_std: torch.Tensor,
    thermo_mean: torch.Tensor,
    thermo_std: torch.Tensor,
    device: torch.device,
    save_path: str | Path,
    epoch: int,
    grid_size: tuple[int, int] | None = None,
    snapshot: int = 0,
) -> None:
    """Plot 2D spatial fields of predicted vs. ground truth SGS stresses and error maps.

    Uses one snapshot of the validation view, taken by its own grid shape
    rather than by slicing the first ``Ny*Nx`` rows -- the flat layout is
    snapshot-concatenated, so a mismatched ``grid_size`` used to silently plot
    a wrapped mixture of snapshots.
    """
    model.eval()
    group = val_view.gather([snapshot], device=device)
    _offset, Ny, Nx = group["shapes"][0]
    if grid_size is not None and tuple(grid_size) != (Ny, Nx):
        print(f"  [plot] ignoring grid_size={tuple(grid_size)}; "
              f"snapshot {snapshot} is {Ny}x{Nx}")

    inv_norm = (group["invariants"] - inv_mean) / inv_std
    thermo_norm = (group["thermo"] - thermo_mean) / thermo_std
    basis = group["basis"]
    tau_target = group["tau_sgs"]

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
    data_dir: str = "dns_khi",
    coarsening_factor: int = 16,
    time_range: tuple[float, float] | None = (1.0, 6.5),
    test_time: tuple[float, float] | None = (6.5, 1e30),
    val_every: int = 5,
    stride: int = 1,
    epochs: int = 50,
    snapshots_per_batch: int = 2,
    eval_snapshots_per_batch: int = 4,
    lr: float = 1e-3,
    lr_schedule: str = "plateau",
    warmup_epochs: int = 5,
    min_lr_frac: float = 0.01,
    grad_clip_mode: str = "adaptive",
    grad_clip: float = 0.0,
    grad_clip_mult: float = 5.0,
    monitor_ema: float = 0.8,
    lr_patience: int = 100,
    patience: int | None = None,
    min_delta: float = 0.0,
    restore_best_weights: bool = True,
    hidden_dim: int = 512,
    num_hidden_layers: int = 3,
    loss_alpha: float = 0.5,
    loss_beta: float = 0.1,
    loss_gamma: float = 0.0,
    loss_delta: float = 2.0,
    lambda_stab: float = 10.0,
    wec_margin: float = 0.5,
    tensor_floor_quantile: float = 0.5,
    tensor_floor_scope: str = "snapshot",
    tensor_floor_floor_frac: float = 0.05,
    spectral_highk_tilt: float = 0.0,
    calibrate_output: bool = True,
    gamma_eos: float | None = None,
    seed: int = 0,
    plot_interval: int = 10,
    checkpoint_dir: str = "checkpoints",
    device_str: str = "cuda",
    dtype_str: str = "float64",
    resume: str | bool | None = None,
    reset_lr: bool = False,
    reset_best: bool = False,
):
    """Main training function with physics-based composite loss and checkpoint resumption.

    Parameters of note
    ------------------
    time_range : the window used for training and validation, interleaved.
        For `dns_khi` the default starts at t = 1.0 because before that the
        subgrid fraction is ~0.03% -- there is essentially no subgrid signal to
        learn, and with a per-point normalized tensor loss those cells are
        amplified noise rather than data.
    test_time : contiguous window held out entirely. Never used for training
        or for model selection.
    snapshots_per_batch : snapshots per optimizer step. One step sees
        ``snapshots_per_batch × Ny × Nx`` points and computes every loss term,
        spectral included, in a single backward pass.
    gamma_eos : adiabatic index for L_P and L_WEC. Leave None to take the value
        resolved by ``src.config`` from the dataset metadata -- do not pass a
        literal.
    """
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

    torch.manual_seed(seed)
    epoch_gen = torch.Generator().manual_seed(seed)

    # One preprocessing pass, one cache, one set of normalization statistics
    # for every split. The cache key covers everything that changes its
    # contents, so a different coarsening or stride cannot silently reuse it.
    cache_path = os.path.join(
        checkpoint_dir, f"dataset_cache_r{coarsening_factor}_s{stride}.pt"
    )

    print(f"Device: {device}")
    print(f"Loading data from {data_dir}, r={coarsening_factor}, stride={stride}...")

    dataset = TBNNDataset(
        data_dir, coarsening_factor=coarsening_factor, time_range=None,
        stride=stride, cache_path=cache_path, dtype=dtype,
    )

    splits = split_snapshots(
        dataset.times, time_range=time_range, val_every=val_every, test_time=test_time,
    )
    train_view = dataset.snapshot_view(splits["train"])
    val_view = dataset.snapshot_view(splits["val"])

    coarse_grid = (
        (dataset.snapshot_boundaries[0][1], dataset.snapshot_boundaries[0][2])
        if len(dataset.snapshot_boundaries) > 0
        else (1024 // coarsening_factor, 1024 // coarsening_factor)
    )

    def _span(ts: Sequence[float]) -> str:
        return f"t=[{min(ts):.3g}, {max(ts):.3g}]" if len(ts) else "empty"

    print(f"  Snapshots available: {dataset.num_snapshots}  {_span(dataset.times)}")
    print(f"  Train: {len(train_view)} snapshots, {train_view.num_points:,} points  "
          f"{_span(train_view.times)}")
    print(f"  Val:   {len(val_view)} snapshots, {val_view.num_points:,} points  "
          f"{_span(val_view.times)}  (every {val_every}th snapshot, interleaved)")
    if splits["test"]:
        test_times = [dataset.times[i] for i in splits["test"]]
        print(f"  Test:  {len(splits['test'])} snapshots {_span(test_times)} "
              "(held out, never used for training or selection)")
    else:
        print("  Test:  none held out")
    print(f"  Coarse grid: {coarse_grid[0]}x{coarse_grid[1]}  "
          f"({coarse_grid[0] * coarse_grid[1]:,} points/snapshot)")
    print(f"  Scales (from the full set): τ={float(dataset.tau_scale):.6e}  "
          f"P={float(dataset.p_scale):.6e}")

    # Scales come from the TRAINING split only, so val/test are measured on the
    # training yardstick rather than each normalizing itself.
    train_pts = train_view.point_index()
    train_tau = dataset.tau_sgs[train_pts]
    train_grad = dataset.grad_u_cov[train_pts]
    train_tau_norm = train_tau.norm(dim=(-2, -1))
    tau_scale = float(train_tau_norm.mean())
    p_scale = float((-torch.einsum("nmk,nmk->n", train_tau, train_grad)).abs().mean())

    # The L_tensor denominator floor is a quantile of the training ‖τ‖_F
    # distribution, not a fraction of its mean: the distribution is
    # heavy-tailed (mean / median ≈ 5 on dns_khi), so a fraction of the mean
    # is not portable. The median damps the below-median half of the domain.
    tau_floor = float(torch.quantile(train_tau_norm.double(), tensor_floor_quantile))
    print(f"  Scales (train split, used in the loss): τ={tau_scale:.6e}  P={p_scale:.6e}")
    print(f"  ‖τ‖_F train distribution: median={float(train_tau_norm.median()):.3e}  "
          f"mean={tau_scale:.3e}  max={float(train_tau_norm.max()):.3e}  "
          f"(mean/median={tau_scale / max(float(train_tau_norm.median()), 1e-300):.2f})")
    print(f"  L_tensor floor = q{tensor_floor_quantile:g}(‖τ‖_F) = {tau_floor:.3e}")

    # Per-snapshot floors. Computed for every snapshot in the dataset (not just
    # the training split) because validation and test batches need one too, and
    # each is a quantile of that snapshot's own ‖τ‖_F — which is the point: the
    # weak-cell damping then scales with the snapshot's own subgrid intensity
    # instead of with whatever the training window happened to average.
    if tensor_floor_scope == "snapshot":
        floors = []
        for start, Ny, Nx in dataset.snapshot_boundaries:
            n = Ny * Nx
            norms = dataset.tau_sgs[start:start + n].norm(dim=(-2, -1)).double()
            floors.append(torch.quantile(norms, tensor_floor_quantile))
        raw = torch.stack(floors)

        # Guard the bottom. A snapshot with no subgrid signal at all — t = 0 on
        # dns_khi has a median ‖τ‖_F of ~3e-15 — would otherwise get a
        # vanishing denominator and dominate the gradient with pure noise. The
        # unguarded spread across this dataset is ~6e11. Clamping from below at
        # a fraction of the global training floor keeps the per-snapshot
        # adaptation where it matters (weak *but real* late-time snapshots)
        # without letting an empty one take over.
        floor_min = tensor_floor_floor_frac * tau_floor
        clamped = raw.clamp(min=floor_min)
        n_clamped = int((raw < floor_min).sum())
        dataset.snapshot_tau_floor = clamped.to(dtype)
        f = dataset.snapshot_tau_floor
        print(f"  L_tensor floor scope = per-snapshot: {len(f)} floors, "
              f"min={float(f.min()):.3e} max={float(f.max()):.3e} "
              f"(spread {float(f.max() / f.min().clamp(min=1e-300)):.1f}x); "
              f"{n_clamped} clamped up to {floor_min:.3e} "
              f"= {tensor_floor_floor_frac:g} x the global floor")
    else:
        dataset.snapshot_tau_floor = None
        print(f"  L_tensor floor scope = global (one value for every snapshot)")

    # Which cells actually drive the gradient. Reported because this is the
    # single easiest thing to get wrong in a normalized regression loss, and
    # it is invisible in the loss value itself: a batch-level normalizer gives
    # the weakest decile ~0% and lets a handful of strong cells dominate,
    # while too low a floor hands the weakest decile 40%+.
    with torch.no_grad():
        w = 1.0 / (train_tau_norm.double() ** 2 + tau_floor ** 2)
        w = w / w.sum()
        order = torch.argsort(train_tau_norm)
        k = max(1, len(order) // 10)
        print(f"  Gradient share (uniform-error regime): weakest 10% of cells "
              f"{100 * float(w[order[:k]].sum()):.1f}%, strongest 10% "
              f"{100 * float(w[order[-k:]].sum()):.2f}%")

    steps_per_epoch = len(train_view.groups(snapshots_per_batch))
    print(f"  Optimizer steps/epoch: {steps_per_epoch} "
          f"({snapshots_per_batch} snapshot(s) = "
          f"{snapshots_per_batch * coarse_grid[0] * coarse_grid[1]:,} points per step)")

    # Normalization stats on device
    inv_mean = dataset.inv_mean.to(device)
    inv_std = dataset.inv_std.to(device)
    thermo_mean = dataset.thermo_mean.to(device)
    thermo_std = dataset.thermo_std.to(device)

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

    # Calibrate the output magnitude before the first step.
    #
    # coeff_head gets PyTorch's default Linear init, which has no knowledge of
    # the target scale. Contracted against the basis (mean ‖E‖_F ≈ 0.17) it
    # produces ‖τ_pred‖_F ≈ 5e-2 while the DNS labels sit at ≈ 7e-3 -- and
    # against the per-point floor the weak cells see a relative error of
    # O(10³). Measured on dns_khi, that is a first-epoch loss of 4.6e3 and a
    # gradient norm of 1.7e5, i.e. ~10⁵x the clip threshold: every step in the
    # opening phase is a pure clipped direction with its magnitude discarded,
    # and the run spends that phase doing nothing but shrinking its output.
    #
    # Scaling the (bias-free) output head by the measured ratio puts the
    # initial prediction at the target scale, which is a property of the head's
    # linearity, not a heuristic. This does not change the architecture or the
    # export contract.
    if calibrate_output:
        with torch.no_grad():
            probe = train_view.gather(train_view.groups(1)[0], device=device)
            probe_inv = (probe["invariants"] - inv_mean) / inv_std
            probe_thermo = (probe["thermo"] - thermo_mean) / thermo_std
            tau0, _c, _p = model(probe_inv, probe_thermo, probe["basis"])
            pred_scale = float(tau0.norm(dim=(-2, -1)).mean())
            target_scale = float(probe["tau_sgs"].norm(dim=(-2, -1)).mean())
            if pred_scale > 0 and target_scale > 0:
                factor = target_scale / pred_scale
                model.coeff_head.weight.mul_(factor)
                print(f"  Output calibration: ‖τ_pred‖_F {pred_scale:.3e} → "
                      f"{target_scale:.3e} (coeff_head × {factor:.4f})")
            else:
                print(f"  Output calibration skipped: pred_scale={pred_scale:.3e}, "
                      f"target_scale={target_scale:.3e}")

    # Physics-based composite loss criterion. Γ must match the data: take the
    # value src.config resolved from the dataset's own metadata.
    if gamma_eos is None:
        from src.dataset import GAMMA_EOS as _resolved_gamma
        gamma_eos = float(_resolved_gamma)
    criterion = RMHDSGSLoss(
        alpha=loss_alpha,
        beta=loss_beta,
        gamma=loss_gamma,
        delta=loss_delta,
        lambda_stab=lambda_stab,
        wec_margin=wec_margin,
        gamma_eos=gamma_eos,
        spectral_highk_tilt=spectral_highk_tilt,
    ).to(dtype=dtype, device=device)
    criterion.set_scales(tau_scale=tau_scale, p_scale=p_scale, tau_floor=tau_floor)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Physics Loss weights: α={loss_alpha} (dissipation), β={loss_beta} (spectral), "
        + f"γ={loss_gamma} (ortho), δ={loss_delta} (WEC), λ_stab={lambda_stab}"
    )
    if spectral_highk_tilt != 0.0:
        print(f"  L_spectral shell weight: (k/k_min)^{spectral_highk_tilt:g}, "
              f"normalised to mean 1")
    print(f"  Γ_EOS in L_P / L_WEC: {gamma_eos:.10g}")
    print(f"  L_WEC margin: {wec_margin} of resolved comoving energy density")
    if loss_gamma == 0.0:
        print("  L_ortho: disabled (γ=0). u_μ E^{μν} = 0 holds to ~3e-16 by "
              "construction, so the term can only ever return 0.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler, sched_is_plateau = build_lr_schedule(
        optimizer, lr_schedule, epochs=epochs, warmup_epochs=warmup_epochs,
        min_lr_frac=min_lr_frac, lr_patience=lr_patience,
    )
    monitor_smoother = EMAMonitor(beta=monitor_ema)

    # Gradient clipping. Adaptive by default: this objective's gradient norm
    # is O(10)-O(10^3) because L_tensor is a ratio floored at the median
    # ||tau||_F, so any fixed threshold small enough to sound safe fires on
    # every step and discards the gradient magnitude entirely.
    clipper: float | AdaptiveGradClip | None
    if grad_clip_mode == "adaptive":
        clipper = AdaptiveGradClip(mult=grad_clip_mult)
        clip_desc = f"adaptive ({grad_clip_mult}x running median)"
    elif grad_clip_mode == "absolute":
        if grad_clip <= 0:
            raise ValueError(
                "--grad-clip-mode absolute requires --grad-clip > 0"
            )
        clipper = grad_clip
        clip_desc = f"absolute {grad_clip:g}"
    elif grad_clip_mode == "off":
        clipper = None
        clip_desc = "off"
    else:
        raise ValueError(
            f"unknown --grad-clip-mode {grad_clip_mode!r} "
            "(expected 'adaptive', 'absolute' or 'off')"
        )

    print(f"LR schedule: {lr_schedule}"
          + (f" (warmup {warmup_epochs} ep, floor {min_lr_frac:g}·lr)" if not sched_is_plateau
             else f" (factor 0.5, patience {lr_patience})")
          + f" | grad clip: {clip_desc} | monitor EMA β: {monitor_ema}")
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

            # A checkpoint's scheduler state belongs to the schedule it was
            # saved under. Loading a ReduceLROnPlateau state into a cosine
            # LambdaLR (or the reverse) is a mismatch, not a resume, so it is
            # reported and skipped rather than raising or silently misapplying.
            if "scheduler_state_dict" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    print(f"[Resume] Restored {lr_schedule} scheduler state")
                except (KeyError, ValueError, TypeError) as e:
                    print(f"[Resume] Scheduler state does not match --lr-schedule "
                          f"{lr_schedule} ({e}); starting the schedule fresh")
            elif sched_is_plateau and best_val_loss < float("inf"):
                scheduler.best = best_val_loss
                print(f"[Resume] Initialized scheduler baseline best={scheduler.best:.6e}")

            if isinstance(clipper, AdaptiveGradClip) and isinstance(
                ckpt.get("grad_clip"), dict
            ):
                clipper.load_state_dict(ckpt["grad_clip"])
                thr = clipper.threshold
                print("[Resume] Restored adaptive grad-clip history "
                      f"(threshold={thr if thr is None else f'{thr:.3e}'})")

            if "monitor_ema" in ckpt and isinstance(ckpt["monitor_ema"], dict):
                saved_beta = float(ckpt["monitor_ema"].get("beta", monitor_ema))
                if abs(saved_beta - monitor_ema) < 1e-12:
                    monitor_smoother.load_state_dict(ckpt["monitor_ema"])
                    print(f"[Resume] Restored monitor EMA (β={saved_beta}, "
                          f"n={monitor_smoother.n}, value={monitor_smoother.value})")
                else:
                    print(f"[Resume] Checkpoint monitor EMA β={saved_beta} differs from "
                          f"--monitor-ema {monitor_ema}; restarting the EMA. The first "
                          "few epochs' smoothed values will be noisier than the rest.")

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
        "val_loss_ema": [],
        "grad_norm": [],
        "clip_frac": [],
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
                    "epoch", "train_loss", "val_loss", "val_loss_ema", "grad_norm",
                    "clip_frac", "val_mse", "val_rel_l2",
                    "L_tensor", "L_P", "L_ortho", "L_WEC", "L_spectral",
                    "val_L_tensor", "val_L_P", "val_L_ortho", "val_L_WEC", "val_L_spectral",
                    "lr", "elapsed_seconds",
                ])
    else:
        # Fresh training run: initialize new CSV
        with open(metrics_csv_path, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "epoch", "train_loss", "val_loss", "val_loss_ema", "grad_norm",
                "clip_frac", "val_mse", "val_rel_l2",
                "L_tensor", "L_P", "L_ortho", "L_WEC", "L_spectral",
                "val_L_tensor", "val_L_P", "val_L_ortho", "val_L_WEC", "val_L_spectral",
                "lr", "elapsed_seconds",
            ])

    epochs_to_run = epochs - start_epoch
    if epochs_to_run <= 0:
        print(f"\nTarget epochs ({epochs}) already reached or exceeded. Nothing to train.")
        return model, {}

    print(f"\nTraining from epoch {start_epoch + 1} to {epochs} (total: {epochs_to_run} epochs, plots saved every {plot_interval} epochs)...\n")

    # Under Condor stdout is not a tty, and a redirected tqdm bar rewrites the
    # same line with carriage returns -- the whole run arrives as one
    # unreadable line in the .out file. Fall back to one line per epoch there.
    interactive = sys.stdout.isatty()
    pbar = tqdm(
        range(start_epoch + 1, epochs + 1), desc="Epochs", unit="epoch",
        initial=start_epoch, total=epochs, disable=not interactive,
    )
    for epoch in pbar:
        t0 = time.time()

        train_res = train_epoch(
            model, train_view, optimizer, criterion,
            inv_mean, inv_std, thermo_mean, thermo_std, device,
            g_cov=g_cov, snapshots_per_batch=snapshots_per_batch,
            grad_clip=clipper, generator=epoch_gen,
        )
        val_metrics = validate(
            model, val_view, criterion,
            inv_mean, inv_std, thermo_mean, thermo_std, device,
            g_cov=g_cov, snapshots_per_batch=eval_snapshots_per_batch,
        )

        train_loss = train_res["train_loss"]
        raw_val = float(val_metrics.get("val_loss", val_metrics["mse"]))
        # Every decision below reads the smoothed value; the raw one is logged.
        monitor_loss = monitor_smoother.update(raw_val)

        # The LR step comes *after* the monitor is smoothed, and before the
        # next epoch's optimizer steps.
        if sched_is_plateau:
            scheduler.step(monitor_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(raw_val))
        history["val_loss_ema"].append(float(monitor_loss))
        history["grad_norm"].append(float(train_res.get("grad_norm", 0.0)))
        history["clip_frac"].append(float(train_res.get("clip_frac", 0.0)))
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
        postfix = {
            "loss": f"{train_loss:.2e}",
            "val": f"{raw_val:.2e}",
            "val_ema": f"{monitor_loss:.2e}",
            "rel_l2": f"{val_metrics['rel_l2'] * 100:.1f}%",
            "L_P": f"{train_res.get('L_P', 0.0):.2e}",
            "L_spec": f"{train_res.get('L_spectral', 0.0):.2e}",
            "|g|": f"{train_res.get('grad_norm', 0.0):.1e}",
            "clip": f"{train_res.get('clip_frac', 0.0):.0%}",
            "lr": f"{current_lr:.1e}",
        }
        if interactive:
            pbar.set_postfix(postfix)
        else:
            print(
                f"ep {epoch:5d}/{epochs} "
                + "  ".join(f"{k}={v}" for k, v in postfix.items())
                + f"  ({elapsed:.1f}s)",
                flush=True,
            )

        # Log to CSV
        with open(metrics_csv_path, "a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                epoch,
                f"{train_loss:.8e}",
                f"{raw_val:.8e}",
                f"{monitor_loss:.8e}",
                f"{train_res.get('grad_norm', 0.0):.8e}",
                f"{train_res.get('clip_frac', 0.0):.4f}",
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
                "wec_margin": wec_margin,
                "tensor_floor_quantile": tensor_floor_quantile,
                # Which floor regime produced these weights. Without it a
                # checkpoint cannot be told apart from one trained under a
                # single global floor, and the two are not comparable.
                "tensor_floor_scope": tensor_floor_scope,
                "tensor_floor_floor_frac": tensor_floor_floor_frac,
                "spectral_highk_tilt": spectral_highk_tilt,
            },
            "loss_scales": {
                "tau_scale": tau_scale, "p_scale": p_scale, "tau_floor": tau_floor,
                "snapshot_tau_floor": getattr(dataset, "snapshot_tau_floor", None),
            },
            "gamma_eos": gamma_eos,
            "splits": splits,
            "split_times": {
                k: [dataset.times[i] for i in v] for k, v in splits.items()
            },
            "monitor_ema": monitor_smoother.state_dict(),
            "grad_clip": (clipper.state_dict()
                          if isinstance(clipper, AdaptiveGradClip) else clipper),
            "val_loss_raw": raw_val,
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
                model, val_view, inv_mean, inv_std, thermo_mean, thermo_std,
                device, os.path.join(plots_dir, f"epoch_{epoch:04d}_pred_vs_truth.png"),
                epoch=epoch, grid_size=coarse_grid,
            )
            plot_pred_vs_truth(
                model, val_view, inv_mean, inv_std, thermo_mean, thermo_std,
                device, os.path.join(checkpoint_dir, "latest_pred_vs_truth.png"),
                epoch=epoch, grid_size=coarse_grid,
            )
            tqdm.write(f"  → [Plots updated: {plots_dir}/epoch_{epoch:04d}_pred_vs_truth.png and {checkpoint_dir}/loss_history.png]")

    final_epoch = stopped_epoch if early_stopped else epochs

    # Ensure final plots are updated up to the completed / stopped epoch
    plot_loss_curves(history, os.path.join(checkpoint_dir, "loss_history.png"))
    plot_pred_vs_truth(
        model, val_view, inv_mean, inv_std, thermo_mean, thermo_std,
        device, os.path.join(plots_dir, f"epoch_{final_epoch:04d}_pred_vs_truth.png"),
        epoch=final_epoch, grid_size=coarse_grid,
    )
    plot_pred_vs_truth(
        model, val_view, inv_mean, inv_std, thermo_mean, thermo_std,
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
            "wec_margin": wec_margin,
            "tensor_floor_quantile": tensor_floor_quantile,
            "tensor_floor_scope": tensor_floor_scope,
            "tensor_floor_floor_frac": tensor_floor_floor_frac,
            "spectral_highk_tilt": spectral_highk_tilt,
        },
        "loss_scales": {
            "tau_scale": tau_scale, "p_scale": p_scale, "tau_floor": tau_floor,
            "snapshot_tau_floor": getattr(dataset, "snapshot_tau_floor", None),
        },
        "gamma_eos": gamma_eos,
        "splits": splits,
        "split_times": {k: [dataset.times[i] for i in v] for k, v in splits.items()},
        "monitor_ema": monitor_smoother.state_dict(),
        "grad_clip": (clipper.state_dict()
                      if isinstance(clipper, AdaptiveGradClip) else clipper),
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
        "lr_schedule": lr_schedule,
        "monitor_ema_beta": monitor_ema,
        "grad_clip_mode": grad_clip_mode,
        "grad_clip_mult": grad_clip_mult,
        "snapshots_per_batch": snapshots_per_batch,
        "tau_scale": tau_scale,
        "p_scale": p_scale,
        "tau_floor": tau_floor,
        "gamma_eos": gamma_eos,
        "splits": splits,
        "split_times": {k: [dataset.times[i] for i in v] for k, v in splits.items()},
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
    print(f"\nTraining {status_str}. Best smoothed val loss: {best_val_loss:.6e} "
          f"(at epoch {best_epoch})")

    # Final measurement on the held-out test window, with the restored best
    # weights. This never influenced training or selection, so it is the only
    # number here that speaks to generalization across the regime.
    if splits["test"]:
        test_view = dataset.snapshot_view(splits["test"])
        test_metrics = validate(
            model, test_view, criterion,
            inv_mean, inv_std, thermo_mean, thermo_std, device,
            g_cov=g_cov, snapshots_per_batch=eval_snapshots_per_batch,
        )
        t_times = test_view.times
        print(f"Held-out test ({len(test_view)} snapshots, "
              f"t=[{min(t_times):.3g}, {max(t_times):.3g}]): "
              f"rel_L2={test_metrics['rel_l2'] * 100:.2f}%  "
              f"loss={test_metrics['val_loss']:.6e}  "
              f"L_tensor={test_metrics.get('val_L_tensor', 0.0):.4e}  "
              f"L_spectral={test_metrics.get('val_L_spectral', 0.0):.4e}")
        history_meta["test_metrics"] = test_metrics
        with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
            json.dump(history_meta, f, indent=2)

    return model, history_meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRMHD-TBNN (a priori) with physics-based loss")
    parser.add_argument("--data-dir", type=str, default="dns_khi")
    parser.add_argument("--coarsening-factor", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1, help="Snapshot sampling stride")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--time-range", type=float, nargs=2, default=(1.0, 6.5), metavar=("T0", "T1"),
        help="Time window used for training+validation, interleaved (default: 1.0 6.5). "
             "dns_khi spans t=0..8; before t~1 the subgrid fraction is ~0.03%%, i.e. "
             "no signal to learn.",
    )
    parser.add_argument(
        "--test-time", type=float, nargs=2, default=(6.5, 1e30), metavar=("T0", "T1"),
        help="Contiguous window held out for the final test measurement only "
             "(default: 6.5 onwards). Pass --no-test-split to disable.",
    )
    parser.add_argument("--no-test-split", action="store_true",
                        help="Do not hold out a test window; use the whole --time-range.")
    parser.add_argument(
        "--val-every", type=int, default=5,
        help="Hold out every Nth snapshot inside --time-range for validation "
             "(default: 5). Interleaved, not contiguous: the flow is strongly "
             "non-stationary, so a contiguous window is a different regime.",
    )
    parser.add_argument(
        "--snapshots-per-batch", type=int, default=2,
        help="Snapshots per optimizer step (default: 2). One step evaluates every "
             "loss term, spectral included, in a single backward pass.",
    )
    parser.add_argument("--eval-snapshots-per-batch", type=int, default=4,
                        help="Snapshots per forward pass during validation (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", type=str, default="plateau",
                        choices=["cosine", "plateau"],
                        help="LR schedule (default: plateau, on the EMA-smoothed metric). "
                             "Measured against cosine on dns_khi: plateau matched or beat it "
                             "(test rel_L2 85.1%% vs 86.6%%) and adapts its reductions to the "
                             "actual run rather than committing to a total epoch count in "
                             "advance -- the earlier default was cosine only because plateau's "
                             "reductions were firing on sampling noise from the old two-phase "
                             "spectral pass; that noise source is gone.")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear LR warmup epochs for the cosine schedule (default: 5)")
    parser.add_argument("--min-lr-frac", type=float, default=0.01,
                        help="Cosine schedule floor as a fraction of --lr (default: 0.01)")
    parser.add_argument("--grad-clip-mode", type=str, default="adaptive",
                        choices=["adaptive", "absolute", "off"],
                        help="Gradient clipping mode (default: adaptive). This objective's "
                             "gradient norm runs 1e1-1e3 because L_tensor is a ratio floored at "
                             "the median ||tau||_F, so a fixed small threshold fires on every "
                             "step and replaces the gradient magnitude with a constant.")
    parser.add_argument("--grad-clip-mult", type=float, default=5.0,
                        help="Adaptive clip threshold, as a multiple of the running median "
                             "gradient norm (default: 5.0)")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Absolute max gradient norm, used only with "
                             "--grad-clip-mode absolute")
    parser.add_argument(
        "--monitor-ema", type=float, default=0.8,
        help="EMA smoothing of the validation metric that drives best-checkpoint "
             "selection, LR reduction and early stopping (default: 0.8, ~5 epochs). "
             "0 disables smoothing and selects on the raw noisy value.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for init and batch order")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--loss-alpha", type=float, default=0.5,
                        help="Weight alpha for dissipation rate loss L_P (default: 0.5)")
    parser.add_argument("--loss-beta", type=float, default=0.1,
                        help="Weight beta for spectral cascade loss L_spectral (default: 0.1)")
    parser.add_argument("--loss-gamma", type=float, default=0.0,
                        help="Weight gamma for L_ortho (default: 0). The term is identically "
                             "zero for a basis-contracted output -- u_mu E^{mu nu} = 0 to ~3e-16 "
                             "by construction -- so a nonzero weight only costs compute. "
                             "Set > 0 to re-enable it as a diagnostic.")
    parser.add_argument("--loss-delta", type=float, default=2.0,
                        help="Weight delta for Weak Energy Condition loss L_WEC (default: 2.0)")
    parser.add_argument("--wec-margin", type=float, default=0.5,
                        help="Fraction of resolved comoving energy density the prediction must "
                             "retain before L_WEC activates (default: 0.5). 0 recovers the plain "
                             "WEC, which cannot activate on this data: resolved energy density is "
                             ">= 2.1 while max ||tau||_F is 0.47.")
    parser.add_argument("--tensor-floor-scope", type=str, default="snapshot",
                        choices=["snapshot", "global"],
                        help="Whether the L_tensor denominator floor is a quantile of each "
                             "snapshot's own ||tau||_F distribution (default) or one value for "
                             "the whole training split. Per-snapshot matters whenever the "
                             "training window spans a large change in subgrid intensity: "
                             "||tau|| varies ~40x across dns_khi t=0..20, and a global floor "
                             "puts every cell of the weak snapshots below it, where the term "
                             "becomes an absolute MSE and predicting zero is near-optimal. "
                             "Use 'global' to reproduce runs made before this option existed.")
    parser.add_argument("--spectral-highk-tilt", type=float, default=0.0,
                        help="Per-shell weight (k/k_min)**tilt on L_spectral, normalised to "
                             "mean 1. Default 0 weights all shells equally, as this term "
                             "always has. A positive tilt trades large-scale spectral accuracy "
                             "for small-scale: the trained model's stress carries ~38%% of its "
                             "power above k=16 against ~51%% for the exact stress, and that "
                             "deficit is what lets high-k energy decay in a long rollout.")
    parser.add_argument("--tensor-floor-floor-frac", type=float, default=0.05,
                        help="Lower bound on each per-snapshot floor, as a fraction of the "
                             "global training floor (default 0.05). Stops a snapshot with no "
                             "subgrid signal from acquiring a near-zero denominator and "
                             "dominating the gradient with noise; unguarded, the spread across "
                             "dns_khi t=0..20 is ~6e11. Only used with "
                             "--tensor-floor-scope snapshot.")
    parser.add_argument("--tensor-floor-quantile", type=float, default=0.5,
                        help="Per-point L_tensor denominator floor, as a quantile of the training "
                             "||tau||_F distribution (default: 0.5, the median). The distribution is "
                             "heavy-tailed (mean/median ~5 on dns_khi), so a quantile is portable "
                             "where a fraction of the mean is not. Lower values hand more of the "
                             "gradient to quasi-laminar cells: at 0.05*mean the weakest decile "
                             "carries 43%% of the loss, at the median 20%%.")
    parser.add_argument("--no-calibrate-output", action="store_true",
                        help="Skip rescaling coeff_head at init to match the target tau scale. "
                             "Without calibration the first epochs only shrink the output: "
                             "measured first-epoch gradient norm 1.7e5 against a clip of 1.0.")
    parser.add_argument("--lambda-stab", type=float, default=10.0,
                        help="Anti-dissipation stability barrier weight in L_P (default: 10.0)")
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
    add_config_arguments(parser)
    cfg = apply_config_defaults(parser, {
        "data_dir": "data.data_dir",
        "coarsening_factor": "data.coarsening_factor",
        "stride": "data.stride",
        "dtype": "solver.dtype",
        "val_every": "training.val_every",
        "snapshots_per_batch": "training.snapshots_per_batch",
        "lr": "training.lr",
        "lr_schedule": "training.lr_schedule",
        "warmup_epochs": "training.warmup_epochs",
        "grad_clip_mode": "training.grad_clip_mode",
        "grad_clip_mult": "training.grad_clip_mult",
        "monitor_ema": "training.monitor_ema",
        "loss_alpha": "training.loss_alpha",
        "loss_beta": "training.loss_beta",
        "loss_gamma": "training.loss_gamma",
        "loss_delta": "training.loss_delta",
        "wec_margin": "training.wec_margin",
        "tensor_floor_quantile": "training.tensor_floor_quantile",
        "lambda_stab": "training.lambda_stab",
    })
    # The time windows are a pair per option, so they are seeded by hand
    # rather than through the dest->key mapping.
    if "training.time_lo" in cfg.explicit or "training.time_hi" in cfg.explicit:
        parser.set_defaults(time_range=(cfg.training.time_lo, cfg.training.time_hi))
    if "training.test_lo" in cfg.explicit or "training.test_hi" in cfg.explicit:
        parser.set_defaults(test_time=(cfg.training.test_lo, cfg.training.test_hi))
    args = parser.parse_args()
    configure(args, cfg)

    # An empty or inverted test window means "no test split".
    if tuple(args.test_time)[0] >= tuple(args.test_time)[1]:
        args.no_test_split = True

    resume_arg = args.resume_from if args.resume_from is not None else args.resume

    train(
        data_dir=args.data_dir,
        coarsening_factor=args.coarsening_factor,
        time_range=tuple(args.time_range),
        test_time=None if args.no_test_split else tuple(args.test_time),
        val_every=args.val_every,
        stride=args.stride,
        epochs=args.epochs,
        snapshots_per_batch=args.snapshots_per_batch,
        eval_snapshots_per_batch=args.eval_snapshots_per_batch,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        min_lr_frac=args.min_lr_frac,
        grad_clip_mode=args.grad_clip_mode,
        grad_clip=args.grad_clip,
        grad_clip_mult=args.grad_clip_mult,
        monitor_ema=args.monitor_ema,
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
        wec_margin=args.wec_margin,
        tensor_floor_quantile=args.tensor_floor_quantile,
        tensor_floor_scope=args.tensor_floor_scope,
        tensor_floor_floor_frac=args.tensor_floor_floor_frac,
        spectral_highk_tilt=args.spectral_highk_tilt,
        calibrate_output=not args.no_calibrate_output,
        gamma_eos=None,   # resolved from dataset metadata by configure()
        seed=args.seed,
        plot_interval=args.plot_interval,
        checkpoint_dir=args.checkpoint_dir,
        device_str=args.device,
        dtype_str=args.dtype,
        resume=resume_arg,
        reset_lr=args.reset_lr,
        reset_best=args.reset_best,
    )
