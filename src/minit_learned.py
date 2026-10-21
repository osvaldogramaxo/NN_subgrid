r"""minit_learned.py

MInIT with a *state-dependent* closure coefficient: Phase 1 of
``minit/FEASIBILITY_learned_coefficients.md``.

The idea, and why it is this and not something else
---------------------------------------------------
The a-priori decomposition in ``minit/MINIT.md`` §4 shows MInIT's proportionality
ansatz ``tau^ij = beta_ij e_KH`` is sound (corr 0.93 against the DNS ``e_KH``)
and that the *evolution equation* for ``e_KH`` is what fails, within a single
snapshot interval. So the coefficient is the thing to replace, and the measured
ceiling for replacing it is the ``oracle_e`` row — that row *is* this model with
a perfect coefficient.

This module keeps MInIT's stress ansatz and its equilibrium (Smagorinsky) scaling
and replaces the single global constant with a small convolutional network of
**5x5 receptive field** over resolved-state invariants:

.. math::
    e_\text{KH}(x) = \exp\big[g_\theta(\mathcal{I}(x))\big]\cdot
                     \rho\,\lambda_\text{KH}^2\,|S|^2,
    \qquad \tau_\perp^{ij} = \beta_{ij}\,e_\text{KH}

``g_theta`` supplies a dimensionless correction only; positivity of ``e_KH`` and
the Smagorinsky scaling are structural, not learned. The model is **memoryless**,
so unlike :class:`src.minit.MInITClosure` it carries no state, needs no
``advance()`` hook, and cannot drift.

Two output modes
----------------
``"residual"`` (default) is the factored form above. ``"direct"`` predicts
``log e_KH`` outright, with the Smagorinsky combinations supplied as input
features instead. ``direct`` is strictly more expressive — it can place
sub-grid energy where the resolved strain vanishes, which is the transported
turbulence the ODE was meant to carry — while ``residual`` has the better
inductive bias and forces ``e_KH -> 0`` with the strain. The feasibility study
measured ``direct``; both are kept so the choice is made on evidence rather than
taste. See ``scripts`` note in :func:`main`.

Why 5x5
-------
The ladder in the feasibility report found a pointwise predictor recovers ~69% of
the gap between the global-coefficient model and the ceiling, and a 5x5
neighbourhood ~81%: the missing information is local but not *pointwise*. Two
3x3 convolutions give exactly that receptive field at ~12k parameters, against
the TBNN's 139,521.

Periodicity
-----------
All padding is circular, matching the DNS boundary conditions. Getting this wrong
shows up as a rim of bad coefficients one cell wide, which is easy to miss in a
mean score and fatal in a rollout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features import periodic_gradient_2d
from src.minit import MInITCoefficients, assemble_tau

# Small compared with the median e_KH of ~7.5e-4 on dns_khi at r=16, large
# enough that log() of an empty cell is finite rather than -inf.
E_FLOOR = 1e-8

INVARIANT_NAMES = (
    "log_rho",
    "strain",
    "shear",
    "abs_div_v",
    "vorticity",
    "speed",
    "lorentz_W",
    "grad_rho_over_rho",
    "log_smag_strain",
    "log_smag_shear",
)
N_INVARIANTS = len(INVARIANT_NAMES)


def compute_invariants(
    rho: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    dx: float,
    dy: float,
    lambda_KH: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Resolved-state invariants and the Smagorinsky base, from coarse fields.

    **This is the only place features are defined.** Both the training path (which
    reads a cached DNS reduction) and the solver path (which has ``prims``) call
    it, because a feature computed one way at fit time and another way at
    inference is a silent, untestable error.

    Note that pressure is deliberately absent. It is not needed by any of these
    invariants, and ``results/minit/dns_reduction.pt`` does not store the coarse
    pressure — a previous diagnostic that quietly substituted a constant for it
    produced a spurious result (``minit/MINIT.md`` §6).

    Parameters
    ----------
    rho, vx, vy : (..., Ny, Nx) coarse Favre-filtered fields.
    dx, dy : coarse grid spacings.
    lambda_KH : filter scale, ``15.8 * delta_f``.

    Returns
    -------
    invariants : (..., Ny, Nx, 10) tensor, ordered as :data:`INVARIANT_NAMES`.
    base : (..., Ny, Nx) tensor — ``rho * lambda_KH^2 * |S|^2``, the equilibrium
        (Smagorinsky) energy up to the coefficient. Returned alongside so the
        residual output mode does not recompute it.
    """
    dvx_dx, dvx_dy = periodic_gradient_2d(vx, dx, dy, order=4)
    dvy_dx, dvy_dy = periodic_gradient_2d(vy, dx, dy, order=4)
    drho_dx, drho_dy = periodic_gradient_2d(rho, dx, dy, order=4)

    S_xx, S_yy = dvx_dx, dvy_dy
    S_xy = 0.5 * (dvx_dy + dvy_dx)
    strain = (2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2)).clamp(min=0.0).sqrt()
    shear = dvx_dy.abs()
    div_v = dvx_dx + dvy_dy
    vort = (dvy_dx - dvx_dy).abs()

    v2 = (vx**2 + vy**2).clamp(max=1.0 - 1e-12)
    W = (1.0 - v2).rsqrt()
    rho_safe = rho.clamp(min=1e-12)
    grad_rho = (drho_dx**2 + drho_dy**2).sqrt() / rho_safe

    base = rho_safe * lambda_KH**2 * strain**2
    smag_shear = rho_safe * lambda_KH**2 * shear**2

    feats = torch.stack(
        [
            rho_safe.log(),
            strain,
            shear,
            div_v.abs(),
            vort,
            v2.sqrt(),
            W,
            grad_rho * dx,
            (base + E_FLOOR).log(),
            (smag_shear + E_FLOOR).log(),
        ],
        dim=-1,
    )
    return feats, base


def _conv2d_gemm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    r"""``F.conv2d`` written as im2col + GEMM, avoiding cuDNN entirely.

    Why this exists
    ---------------
    On this cluster ``F.conv2d`` **cannot run on the GPU at all**. The installed
    PyTorch ships cuDNN 9.24, which refuses any device with SM < 7.5, and the
    nodes have V100s (sm_70)::

        RuntimeError: cuDNN version 92400 is not compatible with devices with
        SM < 7.5

    so every ``conv2d`` call dies with the unhelpful "GET was unable to find an
    engine to execute this computation" — in float32 and float64 alike. It is
    not a precision problem, which is worth stating because forcing float32 is
    the obvious first guess and it does not help.

    ``unfold`` + ``matmul`` routes through cuBLAS instead and works in both
    precisions on sm_70. It is kept as the last-resort path in :func:`_conv2d`;
    the scoped cuDNN disable there is 1.9x faster and bit-identical, so this is
    a fallback rather than the default (all three measured,
    ``scratch/gpu_conv_probe.py`` and ``scratch/gpu_closure_breakdown.py``).

    Input must already be padded; this is the ``padding=0`` case, which is all
    :class:`CoefficientNet` needs since it pads circularly itself.
    """
    out_ch, in_ch, kh, kw = weight.shape
    B, _, H, W = x.shape
    cols = F.unfold(x, kernel_size=(kh, kw))              # (B, in_ch*kh*kw, L)
    out = weight.reshape(out_ch, -1) @ cols               # (B, out_ch, L)
    if bias is not None:
        out = out + bias[None, :, None]
    return out.reshape(B, out_ch, H - kh + 1, W - kw + 1)


_CUDNN_CONV_OK: dict[tuple[str, int | None], bool] = {}


def _cudnn_conv_usable(device: torch.device) -> bool:
    """Can ``F.conv2d`` use cuDNN on this device? Probed once, then cached.

    On the cluster's V100s it cannot: the installed cuDNN 9.24 refuses SM < 7.5,
    so conv2d raises "GET was unable to find an engine to execute this
    computation". Probing rather than hard-coding means a newer GPU still gets
    the fast native path instead of paying for a workaround it does not need.
    """
    key = (device.type, device.index)
    if key not in _CUDNN_CONV_OK:
        if device.type != "cuda":
            _CUDNN_CONV_OK[key] = True          # CPU never goes through cuDNN
        else:
            try:
                probe = torch.zeros(1, 1, 3, 3, device=device)
                F.conv2d(probe, torch.zeros(1, 1, 3, 3, device=device))
                _CUDNN_CONV_OK[key] = True
            except RuntimeError:
                _CUDNN_CONV_OK[key] = False
    return _CUDNN_CONV_OK[key]


def _conv2d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """``F.conv2d`` by the fastest route that actually runs on this device.

    Three tiers, in measured order of preference at 64x64 float64:

    1. plain ``F.conv2d`` (cuDNN) where cuDNN works at all;
    2. ``F.conv2d`` under ``torch.backends.cudnn.flags(enabled=False)`` -- a
       *context manager*, so the flag is restored rather than left flipped for
       the rest of the process. 0.047 ms, and bit-identical to (1);
    3. :func:`_conv2d_gemm`, 0.089 ms, if even that path is unavailable.

    Convolution is ~24% of a closure call, so tier 2 over tier 3 is worth about
    11% of the closure's cost -- small, but free and exactly equal numerically.
    """
    if _cudnn_conv_usable(x.device):
        return F.conv2d(x, weight, bias)
    try:
        with torch.backends.cudnn.flags(enabled=False):
            return F.conv2d(x, weight, bias)
    except RuntimeError:
        return _conv2d_gemm(x, weight, bias)


class CoefficientNet(nn.Module):
    r"""``g_theta``: resolved invariants -> one scalar per cell, 5x5 receptive field.

    Two 3x3 circular convolutions give the 5x5 support; a final 1x1 collapses to
    the scalar. Input standardisation lives in buffers rather than in the training
    script, so a loaded checkpoint normalises exactly as it was fitted — the
    alternative is a model that scores well in the harness and silently wrong in
    the solver.
    """

    def __init__(self, n_in: int = N_INVARIANTS, width: int = 32,
                 out_clamp: float = 12.0):
        super().__init__()
        self.conv1 = nn.Conv2d(n_in, width, 3, padding=0)
        self.conv2 = nn.Conv2d(width, width, 3, padding=0)
        self.head = nn.Conv2d(width, 1, 1)
        self.out_clamp = out_clamp
        self.register_buffer("feat_mean", torch.zeros(n_in))
        self.register_buffer("feat_std", torch.ones(n_in))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feat_mean.copy_(mean.to(self.feat_mean))
        self.feat_std.copy_(std.clamp(min=1e-12).to(self.feat_std))

    def forward(self, invariants: torch.Tensor) -> torch.Tensor:
        """(B, Ny, Nx, F) or (Ny, Nx, F) -> matching (B, Ny, Nx) or (Ny, Nx).

        Convolutions go through :func:`_conv2d`, which picks a route that runs
        on the current device -- cuDNN is unusable on this cluster's sm_70 GPUs. The input is also cast to the module's own
        dtype and the result cast back, so a network stored in one precision can
        serve a solver running in another.
        """
        squeeze = invariants.dim() == 3
        if squeeze:
            invariants = invariants.unsqueeze(0)

        in_dtype = invariants.dtype
        invariants = invariants.to(self.feat_mean.dtype)
        z = (invariants - self.feat_mean) / self.feat_std
        z = z.permute(0, 3, 1, 2)
        z = F.pad(z, (1, 1, 1, 1), mode="circular")
        z = torch.tanh(_conv2d(z, self.conv1.weight, self.conv1.bias))
        z = F.pad(z, (1, 1, 1, 1), mode="circular")
        z = torch.tanh(_conv2d(z, self.conv2.weight, self.conv2.bias))
        out = _conv2d(z, self.head.weight, self.head.bias).squeeze(1)

        # Bounded so a bad step cannot produce exp(large) and a stress that
        # destroys the rollout before the loss ever sees it.
        out = self.out_clamp * torch.tanh(out / self.out_clamp)
        out = out.to(in_dtype)
        return out.squeeze(0) if squeeze else out


@dataclass
class LearnedMInITConfig:
    """Everything needed to reproduce a fitted coefficient field."""

    lambda_KH: float
    beta_xx: float
    beta_yy: float
    beta_xy: float
    output_mode: str = "residual"
    width: int = 32
    gain: float = 1.0
    log_e_mean: float = 0.0     # target standardisation, applied at predict time
    log_e_std: float = 1.0
    # ``gain`` multiplies e_KH after prediction. It is NOT a calibration: the
    # a-priori optimum is 0.958, i.e. the fitted amplitude is already right.
    # It exists because the *true* stress is not the optimal stress for a
    # dissipative discretisation -- the oracle itself only retains 34% of the
    # DNS high-k energy at t=10, so a closure that merely reproduces the true
    # stress cannot recover what the solver's own dissipation removes.
    # Over-driving trades a-priori accuracy for resolved small-scale energy.

    def coefficients(self) -> MInITCoefficients:
        return MInITCoefficients(
            beta_xx=self.beta_xx, beta_yy=self.beta_yy, beta_xy=self.beta_xy
        )


def predict_e_kh(
    net: CoefficientNet,
    cfg: LearnedMInITConfig,
    rho: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """``e_KH`` from coarse fields — the model's whole forward pass."""
    feats, base = compute_invariants(rho, vx, vy, dx, dy, cfg.lambda_KH)
    g = net(feats) * cfg.log_e_std + cfg.log_e_mean
    if cfg.output_mode == "residual":
        log_e = g + (base + E_FLOOR).log()
    elif cfg.output_mode == "direct":
        log_e = g
    else:
        raise ValueError(f"unknown output_mode {cfg.output_mode!r}")
    return (cfg.gain * (log_e.exp() - E_FLOOR)).clamp(min=0.0)


class LearnedMInITClosure:
    r"""MInIT with a learned coefficient, as a solver closure.

    Conforms to the same ``f(prims) -> (tau_perp, alpha)`` protocol as
    :class:`src.minit.MInITClosure` and the TBNN, so it drops into
    ``RHDSolver.step(sgs_closure=...)`` and into ``src.evaluate_rollout``
    unchanged. Unlike ``MInITClosure`` it is **stateless** — the equilibrium
    limit is memoryless — so it exposes no ``advance()`` and there is no
    ``e_KH`` field to seed, drift, or synchronise.

    ``alpha`` is identically one: like MInIT, this model carries no transition
    sensor. Its own ``e_KH`` is the activation.
    """

    def __init__(
        self,
        net: CoefficientNet,
        cfg: LearnedMInITConfig,
        dx: float,
        dy: float,
        clip_mode: str | None = None,
        clip_kappa: float = 0.5,
        tau_smooth_passes: int = 0,
        backscatter_clip: str | None = None,
        backscatter_fraction: float = 1.0,
    ):
        self.net = net.eval()
        self.cfg = cfg
        self.coeffs = cfg.coefficients()
        self.dx, self.dy = dx, dy
        self.clip_mode = clip_mode
        self.clip_kappa = clip_kappa
        self.tau_smooth_passes = tau_smooth_passes
        self.backscatter_clip = backscatter_clip
        self.backscatter_fraction = backscatter_fraction
        self.e_KH: torch.Tensor | None = None   # last prediction, for diagnostics

    @classmethod
    def load(
        cls,
        path: str | Path,
        dx: float,
        dy: float,
        device: torch.device | str | None = None,
        **kw,
    ) -> "LearnedMInITClosure":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = LearnedMInITConfig(**blob["config"])
        net = CoefficientNet(width=cfg.width)
        net.load_state_dict(blob["state_dict"])
        closure = cls(net, cfg, dx, dy, **kw)
        closure.to(device or "cpu")
        return closure

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "LearnedMInITClosure":
        """Move the network, defaulting to the solver's float64.

        Double precision is kept on GPU as well as CPU. That is only possible
        because :func:`_conv2d_gemm` bypasses cuDNN — through ``F.conv2d`` no
        precision works on these sm_70 nodes at all. Keeping float64 means a
        CUDA rollout and a CPU one agree to round-off, so the a-priori scores
        measured on CPU describe the model that actually runs on the cluster.
        """
        device = torch.device(device)
        self.net = self.net.to(device=device, dtype=dtype or torch.float64)
        return self

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.net.state_dict(), "config": vars(self.cfg)}, path
        )

    def __call__(self, prims: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from src.solver import clip_backscatter, clip_sgs_tensor, smooth_sgs_tensor

        with torch.no_grad():
            e_KH = predict_e_kh(
                self.net, self.cfg,
                prims[..., 0], prims[..., 1], prims[..., 2],
                self.dx, self.dy,
            )
        self.e_KH = e_KH

        tau = assemble_tau(e_KH, prims, self.coeffs)

        if self.clip_mode is not None:
            tau = clip_sgs_tensor(tau, prims, mode=self.clip_mode, kappa=self.clip_kappa)
        if self.tau_smooth_passes:
            tau = smooth_sgs_tensor(tau, self.tau_smooth_passes)
        if self.backscatter_clip is not None:
            tau = clip_backscatter(
                tau, prims, self.dx, self.dy, mode=self.backscatter_clip,
                fraction=self.backscatter_fraction,
            )

        alpha = torch.ones(prims.shape[:-1], dtype=prims.dtype, device=prims.device)
        return tau, alpha


class HybridClosure:
    r"""TBNN tensor *shape* x MiNNit *amplitude* (FEASIBILITY §5, §13).

    The filter scan found a clean split: the TBNN has the better shape (its
    correlation beats the fixed-beta ceiling at every filter size) and MiNNit the
    better amplitude (no late-time decay, more high-k). So keep the TBNN's
    tensor and replace its energy with MiNNit's, cell by cell:

    .. math::
        \tau^{\mu\nu} = e_\text{MiNNit}\,\frac{\tau^{\mu\nu}_\text{TBNN}}{e_\text{TBNN}},
        \qquad e = \tfrac12(\tau^{xx} + \tau^{yy})

    Scaling the whole 4x4 tensor by a scalar preserves transversality. Where the
    TBNN's energy is too small for its shape to mean anything (below ``floor``
    times the snapshot's mean MiNNit energy, or negative), the cell falls back to
    MiNNit's own ``beta_ij e`` -- the same normalisation, since
    ``(beta_xx + beta_yy)/2 = 1``. No new training: both parts are the fitted models.

    The result is clipped for realizability *after* the rescale (``clip_mode``,
    ``‖τ‖_F ≤ κ p`` by default, the same bound the TBNN closure applies). The TBNN
    closure clips its own output, but the rescale ``e_MiNNit / e_TBNN`` exceeds 700x
    where ``e_TBNN`` sits just above the floor, so without a second clip the hybrid
    stress is unbounded -- it diverged in 1 of 14 rollouts that way (FEASIBILITY
    §19-20). ``clip_mode=None`` restores the unclipped behaviour.

    Shape/energy ablations (FEASIBILITY §23): the shape comes from ``tbnn_closure``,
    which may be any closure (the exact-DNS one included). ``energy_closure``, if
    given, replaces MiNNit as the energy source, ``e = (tau^xx + tau^yy)/2`` of its
    stress; the fallback is then MInIT's ``beta_ij e`` with that energy, so only the
    factor under test changes.
    """

    def __init__(self, tbnn_closure, learned_closure: LearnedMInITClosure,
                 floor: float = 0.05, clip_mode: str | None = "frobenius",
                 clip_kappa: float = 0.5, energy_closure=None):
        self.tbnn, self.learned, self.floor = tbnn_closure, learned_closure, floor
        self.energy = energy_closure
        self.clip_mode, self.clip_kappa = clip_mode, clip_kappa
        self.fallback_fraction: float | None = None    # diagnostic, last call
        self.max_scale: float | None = None             # diagnostic, last call

    def set_time(self, t: float) -> None:
        # The solver syncs time-dependent closures (the exact-DNS one) through set_time;
        # a wrapper that swallowed it would freeze them at their first snapshot.
        for c in (self.tbnn, self.learned, self.energy):
            if c is not None and hasattr(c, "set_time"):
                c.set_time(t)

    def advance(self, prims: torch.Tensor, dt: float, t: float | None = None) -> None:
        # Stateful parts (MInIT's evolved e_KH as the energy source) step once per
        # completed solver step; without this they would stay frozen at their seed.
        for c in (self.tbnn, self.learned, self.energy):
            if c is not None and hasattr(c, "advance"):
                c.advance(prims, dt, t=t)

    def __call__(self, prims: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tau_t, alpha = self.tbnn(prims)
        if self.energy is None:
            tau_l, _ = self.learned(prims)
            e_l = self.learned.e_KH
        else:
            tau_e, _ = self.energy(prims)
            e_l = (0.5 * (tau_e[..., 1, 1] + tau_e[..., 2, 2])).clamp(min=0.0)
            tau_l = assemble_tau(e_l, prims, self.learned.coeffs)
        e_t = 0.5 * (tau_t[..., 1, 1] + tau_t[..., 2, 2])
        ok = e_t > self.floor * e_l.mean()
        scale = torch.where(ok, e_l / e_t.clamp(min=1e-300), torch.zeros_like(e_t))
        tau = torch.where(ok[..., None, None], tau_t * scale[..., None, None], tau_l)
        self.fallback_fraction = float((~ok).double().mean())
        self.max_scale = float(scale.max())
        if self.clip_mode is not None:
            from src.solver import clip_sgs_tensor
            tau = clip_sgs_tensor(tau, prims, mode=self.clip_mode, kappa=self.clip_kappa)
        return tau, alpha
