r"""
losses.py

Physics-based composite loss function for integrity-basis SGS models
in relativistic MHD (RMHD).

Implements the formulations from rmhd_sgs_loss_functions.md:

    L_total = L_tensor + α L_P + β L_spectral + γ L_ortho + δ L_WEC

Components:
    1. CovariantTensorLoss   — Metric-contracted normalized Frobenius regression (§2.1)
    2. DissipationRateLoss   — Subgrid production matching + anti-dissipation barrier (§2.2)
    3. SpectralCascadeLoss   — Log-space power spectrum matching (§2.3)
    4. OrthogonalityLoss     — Fluid rest-frame projection constraints (§2.4)
    5. WECLoss               — Weak Energy Condition realizability (§2.5)
    6. RMHDSGSLoss           — Composite weighted objective (§3)

Conventions:
    - Metric signature: (-, +, +, +)
    - Greek indices: spacetime {0, 1, 2, 3}
    - τ^{μν} in contravariant form throughout
    - grad_u_cov[..., ν, μ] = ∇_ν u_μ  (covariant four-velocity gradient)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# §2.1 — Covariant Normalized Tensor Regression Loss
# ---------------------------------------------------------------------------

class CovariantTensorLoss(nn.Module):
    r"""Covariant normalized tensor regression loss (§2.1).

    Computes the metric-contracted Frobenius norm of the residual, normalized
    by local subgrid intensity:

    .. math::
        \mathcal{L}_{\text{tensor}} = \frac{1}{N} \sum_{i=1}^{N}
            \frac{g_{\mu\alpha} g_{\nu\beta}
                  (\tau_{\text{pred}}^{\mu\nu} - \tau_{\text{DNS}}^{\mu\nu})
                  (\tau_{\text{pred}}^{\alpha\beta} - \tau_{\text{DNS}}^{\alpha\beta})}
                 {\lVert \tau_{\text{DNS},i} \rVert_F^2 + \epsilon}

    where :math:`\lVert A \rVert_F^2 = g_{\mu\alpha} g_{\nu\beta} A^{\mu\nu} A^{\alpha\beta}`.

    The normalization is **per point**, which is what makes this loss differ
    from plain MSE: every cell contributes an O(1) relative error regardless of
    its subgrid intensity, so the gradient is not monopolised by the handful of
    cells with the largest :math:`\lVert \tau \rVert_F`. On `dns_khi` at t = 4
    the per-cell Frobenius norm spans mean 7.3e-3 to max 4.7e-1 — a factor of
    65 — so a batch-level normalizer (the earlier behaviour) reduced this to
    MSE times a constant and starved the low-intensity 99% of the domain.

    The price of per-point normalization is the opposite bias: quasi-laminar
    cells, where :math:`\lVert \tau_{\text{DNS}} \rVert_F \to 0`, would be
    amplified without bound. The denominator therefore carries a floor taken
    from the training distribution of :math:`\lVert \tau \rVert_F`, not an
    absolute constant.

    The floor is worth setting from a *quantile*, because the distribution is
    heavy-tailed: over the `dns_khi` training window the mean is 5.2x the
    median. Measured share of the loss carried by the weakest and strongest
    deciles (uniform absolute error, i.e. the regime at initialization):

    ==================  =============  ===============
    floor               weakest 10%    strongest 10%
    ==================  =============  ===============
    0.05 x mean            43.0%           0.01%
    median (0.2 x mean)    20.0%           0.05%
    0.5 x mean             14.5%           0.24%
    batch-level (old)      ~0%            dominant
    ==================  =============  ===============

    The median is the default: it damps the below-median half of the domain
    while leaving the resolved-turbulence cells in charge of the gradient.

    Parameters
    ----------
    epsilon : float
        Last-resort floor, used only when neither ``tau_floor`` nor
        ``tau_scale`` is supplied.
    floor_frac : float
        Fallback floor as a fraction of ``tau_scale``, used when no explicit
        ``tau_floor`` is given.
    """

    def __init__(self, epsilon: float = 1e-8, floor_frac: float = 0.2):
        super().__init__()
        self.epsilon = epsilon
        self.floor_frac = floor_frac

    def forward(
        self,
        tau_pred: torch.Tensor,
        tau_dns: torch.Tensor,
        g_cov: torch.Tensor,
        tau_scale: torch.Tensor | float | None = None,
        tau_floor: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tau_pred : (..., 4, 4) — predicted SGS stress τ^{μν}
        tau_dns  : (..., 4, 4) — ground truth SGS stress τ^{μν}_DNS
        g_cov    : (4, 4) or (..., 4, 4) — metric tensor g_{μν}
        tau_floor : scalar **or tensor broadcastable over the leading dims** —
            ‖τ‖_F below which a cell stops being normalized, in physical units.
            Preferred: a quantile of the training distribution (the median by
            default).

            A *per-snapshot* floor is strongly preferred over a global one
            whenever the training window spans a range over which the subgrid
            intensity changes a lot. ‖τ‖_F varies ~40x across ``dns_khi``
            (t = 0..20), so one global floor puts nearly every cell of the weak
            snapshots below it, where this term degenerates into an absolute
            MSE and predicting zero becomes near-optimal. That is precisely how
            a model ends up with the right stress in mid-run and a collapsed one
            at late times.
        tau_scale : scalar — dataset-level mean ‖τ_DNS‖_F. Used with
            ``floor_frac`` when ``tau_floor`` is not given.
        """
        diff = tau_pred - tau_dns

        # ‖Δτ‖²_F = g_{μα} g_{νβ} Δτ^{μν} Δτ^{αβ}
        diff_low = torch.einsum("...ma,...nb,...ab->...mn", g_cov, g_cov, diff)
        frob_diff = torch.clamp(torch.einsum("...mn,...mn->...", diff, diff_low), min=0.0)

        # ‖τ_DNS‖²_F
        dns_low = torch.einsum("...ma,...nb,...ab->...mn", g_cov, g_cov, tau_dns)
        frob_dns = torch.clamp(torch.einsum("...mn,...mn->...", tau_dns, dns_low), min=0.0)

        if tau_floor is not None:
            if isinstance(tau_floor, torch.Tensor):
                # Broadcast against the per-point frob_* fields, which have the
                # leading dims of tau with the two tensor axes reduced away.
                floor = tau_floor.to(frob_dns.device, frob_dns.dtype) ** 2
                while floor.dim() > frob_dns.dim():
                    floor = floor.squeeze(-1)
            else:
                floor = float(tau_floor) ** 2
        elif tau_scale is not None:
            floor = (self.floor_frac * float(tau_scale)) ** 2
        else:
            floor = self.epsilon

        # Per-point relative Frobenius error, then averaged.
        return (frob_diff / (frob_dns + floor)).mean()


# ---------------------------------------------------------------------------
# §2.2 — Kinetic & Magnetic Dissipation Rate Penalty
# ---------------------------------------------------------------------------

class DissipationRateLoss(nn.Module):
    r"""Subgrid dissipation rate penalty (§2.2).

    Matches the local subgrid production rate and enforces a global stability
    barrier against net anti-dissipation:

    .. math::
        \mathcal{L}_{\mathcal{P}} = \frac{1}{N} \sum_{i=1}^N
            \left| \mathcal{P}_{\text{pred},i} - \mathcal{P}_{\text{DNS},i} \right|
            + \lambda_{\text{stab}} \operatorname{ReLU}\!\left(
                -\int_\Omega \mathcal{P}_{\text{pred}} \, dV \right)

    where :math:`\mathcal{P}_{\text{SGS}} = -\tau^{\mu\nu} \nabla_\mu \bar{u}_\nu`.

    Both terms are divided by a production scale so that the component is
    dimensionless and comparable across datasets and coarsening factors.
    :math:`\mathcal{P}` carries units of :math:`\tau \nabla u`, so an
    unnormalized L1 term silently rescales α whenever the subgrid intensity
    changes — which it does by a factor of ~40 over the course of this KHI run.

    Parameters
    ----------
    lambda_stab : float
        Weight of the global anti-dissipation stability barrier.
    """

    def __init__(self, lambda_stab: float = 10.0):
        super().__init__()
        self.lambda_stab = lambda_stab

    def forward(
        self,
        tau_pred: torch.Tensor,
        tau_dns: torch.Tensor,
        grad_u_cov: torch.Tensor,
        p_scale: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tau_pred    : (..., 4, 4) — predicted SGS stress τ^{μν}
        tau_dns     : (..., 4, 4) — ground truth SGS stress τ^{μν}_DNS
        grad_u_cov  : (..., 4, 4) — velocity gradient ∇_ν u_μ
                      Convention: grad_u_cov[..., ν, μ] = ∂_ν u_μ
                      So ∂_μ u_ν = grad_u_cov[..., μ, ν]
        p_scale     : scalar — dataset-level |P_DNS| scale used to make the
                      component dimensionless. When None, falls back to the
                      current batch's |P_DNS| mean (detached), which keeps the
                      term O(1) but makes it batch-dependent.
        """
        # P_SGS = -τ^{μν} ∇_μ u_ν = -τ^{μν} grad_u_cov[μ, ν]
        P_pred = -torch.einsum("...mn,...mn->...", tau_pred, grad_u_cov)
        P_dns = -torch.einsum("...mn,...mn->...", tau_dns, grad_u_cov)

        if p_scale is None:
            scale = P_dns.abs().mean().detach().clamp(min=1e-30)
        else:
            scale = torch.as_tensor(
                float(p_scale), dtype=P_dns.dtype, device=P_dns.device
            ).clamp(min=1e-30)

        # L1 matching of local production rates, in units of the P scale
        l1_match = (torch.abs(P_pred - P_dns) / scale).mean()

        # Global stability barrier: penalize net anti-dissipation
        # (negative mean production → energy flowing from subgrid to resolved)
        stability = self.lambda_stab * torch.relu(-P_pred.mean() / scale)

        return l1_match + stability


# ---------------------------------------------------------------------------
# §2.3 — Multi-Scale Spectral Cascade Loss
# ---------------------------------------------------------------------------

class SpectralCascadeLoss(nn.Module):
    r"""Multi-scale spectral cascade loss (§2.3).

    Matches the radially averaged power spectrum of the SGS stress tensor
    in log-space to preserve inertial-range power-law scalings:

    .. math::
        \mathcal{L}_{\text{spectral}} = \int_{k_{\min}}^{k_{\max}}
            \left| \log_{10} E_{\text{pred}}(k) - \log_{10} E_{\text{DNS}}(k)
            \right|^2 dk

    Requires spatially structured 2D grid data (Ny, Nx, 4, 4).

    The log floor is **relative** to the DNS spectrum, not an absolute
    constant. An absolute floor is a scale trap here: the unnormalized FFT
    power of τ ~ 1e-3 on a 64² grid lands around 1e-8·N⁴, so a fixed 1e-12
    floor is either irrelevant or dominant depending only on grid size and
    subgrid intensity. Worse, at initialization the predicted spectrum can sit
    many decades below the DNS one in the high-k shells, and log10 of a
    near-zero spectrum produces a gradient spike that is pure noise. Flooring
    at ``epsilon_rel`` times the mean DNS shell power bounds the log residual
    at roughly ``|log10 epsilon_rel|`` per shell.

    Parameters
    ----------
    epsilon_rel : float
        Floor as a fraction of the mean DNS shell power (detached). 1e-8 caps
        a single shell's contribution at (8 decades)² regardless of units.
    epsilon : float
        Absolute floor, still added for exact-zero safety.
    k_min : int
        First shell included. 1 keeps everything above the DC bin.
    highk_tilt : float
        Exponent of a per-shell weight ``(k / k_min) ** highk_tilt`` applied to
        the log residual before averaging. ``0.0`` (default) weights every shell
        equally, which is the behaviour this term has always had.

        Why it exists: the term as written treats a decade of error at k = 2 and
        at k = 30 identically, but the resolved field's small scales are what a
        closure is for, and the trained model's stress carries ~38% of its power
        above k = 16 against ~51% for the exact stress. A positive tilt buys
        that deficit at the cost of the large scales, which the point-wise terms
        already constrain well. Weights are normalised to mean 1 so the tilt
        does not silently rescale the term against ``beta``.
    """

    def __init__(
        self,
        epsilon: float = 1e-30,
        epsilon_rel: float = 1e-8,
        k_min: int = 1,
        highk_tilt: float = 0.0,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.epsilon_rel = epsilon_rel
        self.k_min = max(1, int(k_min))
        self.highk_tilt = float(highk_tilt)

    def _shell_spectrum(self, field_2d: torch.Tensor) -> torch.Tensor:
        """Compute radially (shell) averaged 2D power spectrum.

        Parameters
        ----------
        field_2d : (C, Ny, Nx) — multi-channel 2D field

        Returns
        -------
        spectrum : (k_max - 1,) — radially averaged power per wavenumber
                   shell, from k=1 to k=k_max-1 (DC bin excluded).
        """
        C, Ny, Nx = field_2d.shape

        # 2D real FFT → power spectrum |F(k)|²
        fft_c = torch.fft.rfft2(field_2d, dim=(-2, -1))       # (C, Ny, Nx//2+1)
        power = (fft_c.real.pow(2) + fft_c.imag.pow(2)).sum(dim=0)  # (Ny, Nx//2+1)

        # Integer wavenumber magnitudes
        ky = torch.fft.fftfreq(Ny, device=field_2d.device, dtype=field_2d.dtype) * Ny
        kx = torch.fft.rfftfreq(Nx, device=field_2d.device, dtype=field_2d.dtype) * Nx
        ky_g, kx_g = torch.meshgrid(ky, kx, indexing="ij")
        k_mag = torch.sqrt(ky_g ** 2 + kx_g ** 2)
        k_int = k_mag.round().long()

        k_max = min(Ny, Nx) // 2

        k_flat = k_int.reshape(-1)
        p_flat = power.reshape(-1)

        # Select valid modes (skip DC, cap at Nyquist)
        valid = (k_flat > 0) & (k_flat < k_max)
        k_v = k_flat[valid]
        p_v = p_flat[valid]

        # Differentiable shell averaging via out-of-place scatter_add
        spectrum = torch.zeros(k_max, device=field_2d.device, dtype=field_2d.dtype)
        spectrum = spectrum.scatter_add(0, k_v, p_v)

        counts = torch.zeros(k_max, device=field_2d.device, dtype=field_2d.dtype)
        counts.scatter_add_(0, k_v, torch.ones_like(p_v))

        spectrum = spectrum / counts.clamp(min=1)

        # Return k=k_min..k_max-1 (DC bin excluded by construction)
        return spectrum[self.k_min:] + self.epsilon

    def forward(
        self,
        tau_pred_2d: torch.Tensor,
        tau_dns_2d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tau_pred_2d : (Ny, Nx, 4, 4) — predicted SGS stress on 2D grid
        tau_dns_2d  : (Ny, Nx, 4, 4) — ground truth SGS stress on 2D grid
        """
        Ny, Nx = tau_pred_2d.shape[:2]

        # Reshape tensor components to channels: (16, Ny, Nx)
        pred_c = tau_pred_2d.reshape(Ny, Nx, 16).permute(2, 0, 1)
        dns_c = tau_dns_2d.reshape(Ny, Nx, 16).permute(2, 0, 1)

        E_pred = self._shell_spectrum(pred_c)
        E_dns = self._shell_spectrum(dns_c)

        # Relative log floor: bounds the residual of any single shell, and
        # keeps the component invariant under a rescaling of τ.
        floor = self.epsilon_rel * E_dns.mean().detach()

        # Log-space L2 matching
        log_diff = torch.log10(E_pred + floor) - torch.log10(E_dns + floor)
        sq = log_diff ** 2

        if self.highk_tilt != 0.0:
            k = torch.arange(
                self.k_min, self.k_min + sq.shape[-1],
                device=sq.device, dtype=sq.dtype,
            )
            w = (k / self.k_min) ** self.highk_tilt
            w = w / w.mean()          # mean 1, so beta keeps its meaning
            sq = sq * w

        return sq.mean()


# ---------------------------------------------------------------------------
# §2.4 — Physical Invariant & Orthogonality Constraints
# ---------------------------------------------------------------------------

class OrthogonalityLoss(nn.Module):
    r"""Fluid rest-frame orthogonality constraints (§2.4).

    For SGS stress defined as spatial-only in the fluid rest frame, the
    projections onto the four-velocity must vanish identically:

    .. math::
        \mathcal{L}_{\text{ortho}} =
            \frac{1}{N} \sum_{i=1}^N
                \left( \bar{u}_\mu \tau_{\text{pred}}^{\mu\nu} \bar{u}_\nu \right)^2
            + \frac{1}{N} \sum_{i=1}^N
                g_{\nu\lambda}
                \left( \bar{u}_\mu \tau_{\text{pred}}^{\mu\nu} \right)
                \left( \bar{u}_\alpha \tau_{\text{pred}}^{\alpha\lambda} \right)

    .. warning::
       For the current model this term is **identically zero and contributes no
       gradient**. ``CRMHD_TBNN`` forms its output as
       :math:`\tau^{\mu\nu} = \sum_n G^{(n)} E^{(n)\mu\nu}`, and the basis
       tensors from ``extract_tbnn_features`` satisfy
       :math:`\bar{u}_\mu E^{(n)\mu\nu} = 0` to machine precision (measured:
       3e-16 relative on `dns_khi` at t = 4). Both projections therefore vanish
       by construction for *any* coefficients. The default weight γ is 0 for
       that reason — the constraint is enforced by the architecture, not by the
       loss. Keep the class: it is the right diagnostic to run if the basis or
       the output head ever stops being rest-frame spatial (e.g. if the unused
       ``p_sgs`` head is wired into the stress).
    """

    def forward(
        self,
        tau_pred: torch.Tensor,
        u_con: torch.Tensor,
        g_cov: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tau_pred : (..., 4, 4) — predicted SGS stress τ^{μν}
        u_con    : (..., 4)    — four-velocity u^μ
        g_cov    : (4, 4) or (..., 4, 4) — metric tensor g_{μν}
        """
        # u_μ = g_{μα} u^α
        u_cov = torch.einsum("...ij,...j->...i", g_cov, u_con)

        # Scalar projection: u_μ τ^{μν} u_ν  (should vanish)
        scalar_proj = torch.einsum("...m,...mn,...n->...", u_cov, tau_pred, u_cov)
        L_scalar = (scalar_proj ** 2).mean()

        # Vector projection: v^ν = u_μ τ^{μν}  (should vanish)
        vec_proj = torch.einsum("...m,...mn->...n", u_cov, tau_pred)

        # In Lorentzian signature (-, +, +, +), g_{νλ} v^ν v^λ = -(u·v)² + ‖v_⊥‖².
        # Using spatial projector h_{νλ} = g_{νλ} + u_ν u_λ ensures positive-definiteness:
        #   ‖v_⊥‖² = h_{νλ} v^ν v^λ = g_{νλ} v^ν v^λ + (u_ν v^ν)² ≥ 0
        # Total fluid-frame loss = L_scalar + L_spatial = (v⁰)² + (v¹)² + (v²)² + (v³)² ≥ 0
        v_cov = torch.einsum("...ij,...j->...i", g_cov, vec_proj)
        u_dot_v = torch.einsum("...i,...i->...", u_cov, vec_proj)
        L_spatial = (torch.einsum("...i,...i->...", vec_proj, v_cov) + u_dot_v ** 2).mean()

        return L_scalar + L_spatial


# ---------------------------------------------------------------------------
# §2.5 — Realizability and Energy Conditions (WEC)
# ---------------------------------------------------------------------------

class WECLoss(nn.Module):
    r"""Weak Energy Condition (WEC) realizability loss (§2.5).

    The effective stress-energy tensor
    :math:`T_{\text{eff}}^{\mu\nu} = T_{\text{resolved}}^{\mu\nu} + \tau_{\text{pred}}^{\mu\nu}`
    must have non-negative energy density as measured by the comoving fluid
    observer:

    .. math::
        \varepsilon_{\text{eff}} = T_{\text{eff}}^{\mu\nu} u_\mu u_\nu \ge 0

    This is a necessary condition for the WEC and prevents unphysical negative
    pressure or imaginary characteristic speeds that cause primitive variable
    recovery failure.

    Parameters
    ----------
    The hinge is on the **ratio** to the resolved energy density, not on the
    absolute value. An absolute ``relu(-ε_eff)`` cannot activate on this
    problem: the resolved comoving energy density :math:`\rho h - p` is ≥ 2.1
    (mean 9.3) on `dns_khi` at t = 4, while the largest exact
    :math:`\lVert\tau\rVert_F` in the domain is 0.47 — a 5× margin — so the
    term sat at exactly 0.0 for every epoch of the previous run and δ bought
    nothing. Penalising :math:`\varepsilon_{\text{eff}} < \text{margin} \cdot
    \varepsilon_{\text{resolved}}` instead gives a constraint that is inactive
    on the DNS labels but bites on a prediction that starts eating the resolved
    energy density, which is the failure mode that actually ends a rollout.

    Parameters
    ----------
    gamma_eos : float
        Adiabatic index Γ for the ideal gas EOS. Must match the data; pass the
        value resolved by ``src.config``, never a literal.
    margin : float
        Fraction of the resolved comoving energy density that the effective
        energy density must retain. 0.0 recovers the plain (unreachable) WEC;
        0.5 penalises an SGS stress that removes more than half of it.
    """

    def __init__(self, gamma_eos: float = 4.0 / 3.0, margin: float = 0.5):
        super().__init__()
        self.gamma = gamma_eos
        self.margin = margin

    def forward(
        self,
        tau_pred: torch.Tensor,
        u_con: torch.Tensor,
        rho: torch.Tensor,
        p: torch.Tensor,
        g_cov: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tau_pred : (..., 4, 4) — predicted SGS stress τ^{μν}
        u_con    : (..., 4)    — four-velocity u^μ
        rho      : (...)       — rest-mass density ρ̄
        p        : (...)       — thermal pressure p̄
        g_cov    : (4, 4) or (..., 4, 4) — metric tensor g_{μν}
                   (In Minkowski, η_{μν} = η^{μν}, so this doubles as g^{μν}.)
        """
        # Compute resolved stress-energy from primitive fields:
        # T^{μν} = ρh u^μ u^ν + p η^{μν}
        rho_safe = rho.clamp(min=1e-15)
        eps_thermo = p / ((self.gamma - 1.0) * rho_safe)
        h = 1.0 + eps_thermo + p / rho_safe              # specific enthalpy
        rho_h = rho * h

        uu = torch.einsum("...i,...j->...ij", u_con, u_con)      # u^μ u^ν
        # T_resolved^{μν} = ρh u^μ u^ν + p g^{μν}
        # For Minkowski: g^{μν} ≡ g_{μν} = diag(-1,1,1,1)
        T_resolved = (rho_h.unsqueeze(-1).unsqueeze(-1) * uu
                      + p.unsqueeze(-1).unsqueeze(-1) * g_cov)

        # Energy density observed by comoving fluid: ε = T^{μν} u_μ u_ν
        u_cov = torch.einsum("...ij,...j->...i", g_cov, u_con)
        e_resolved = torch.einsum("...mn,...m,...n->...", T_resolved, u_cov, u_cov)
        e_sgs = torch.einsum("...mn,...m,...n->...", tau_pred, u_cov, u_cov)
        e_eff = e_resolved + e_sgs

        # Relative hinge: require ε_eff >= margin * ε_resolved.
        # Normalising by ε_resolved makes the term dimensionless and, unlike
        # relu(-ε_eff), reachable — see the class docstring.
        denom = e_resolved.detach().abs().clamp(min=1e-30)
        return torch.relu(self.margin - e_eff / denom).mean()


# ---------------------------------------------------------------------------
# §3 — Composite Objective Function
# ---------------------------------------------------------------------------

class RMHDSGSLoss(nn.Module):
    r"""Composite physics-based loss for RMHD SGS models (§3).

    .. math::
        \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{tensor}}
            + \alpha \, \mathcal{L}_{\mathcal{P}}
            + \beta  \, \mathcal{L}_{\text{spectral}}
            + \gamma \, \mathcal{L}_{\text{ortho}}
            + \delta \, \mathcal{L}_{\text{WEC}}

    Point-wise losses (tensor, dissipation, ortho, WEC) are computed via
    :meth:`forward`.  The spectral loss requires spatially structured 2D
    grid data and is computed separately via :meth:`compute_spectral`.

    Every component is dimensionless, so the weights mean the same thing
    across datasets, coarsening factors and flow regimes. That requires the
    dataset-level scales ``tau_scale`` and ``p_scale``: set them once with
    :meth:`set_scales` (the training entry point does this) and they are
    threaded into the components that need them.

    Parameters
    ----------
    alpha : float
        Weight for subgrid dissipation rate loss.
    beta : float
        Weight for spectral cascade loss.
    gamma : float
        Weight for orthogonality constraint loss. **Default 0**: the term is
        identically zero for this architecture — see :class:`OrthogonalityLoss`.
    delta : float
        Weight for WEC realizability loss.
    lambda_stab : float
        Global anti-dissipation barrier strength in L_P.
    tensor_epsilon : float
        Fallback denominator floor for the tensor loss, used only when no
        ``tau_scale`` has been set.
    tensor_floor_frac : float
        Per-point denominator floor as a fraction of ``tau_scale``.
    spectral_epsilon_rel : float
        Spectral log floor as a fraction of the mean DNS shell power.
    wec_margin : float
        Fraction of resolved comoving energy density the prediction must
        retain before L_WEC activates.
    gamma_eos : float
        Adiabatic index Γ for the ideal gas EOS. Pass the value resolved by
        ``src.config`` from the dataset metadata; the default here is only a
        fallback and a wrong Γ corrupts L_P and L_WEC silently.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.1,
        gamma: float = 0.0,
        delta: float = 2.0,
        lambda_stab: float = 10.0,
        tensor_epsilon: float = 1e-8,
        tensor_floor_frac: float = 0.05,
        spectral_epsilon_rel: float = 1e-8,
        wec_margin: float = 0.5,
        gamma_eos: float = 4.0 / 3.0,
        spectral_highk_tilt: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma_w = gamma      # 'gamma' conflicts with EOS gamma
        self.delta = delta

        self.tensor_loss = CovariantTensorLoss(
            epsilon=tensor_epsilon, floor_frac=tensor_floor_frac
        )
        self.dissipation_loss = DissipationRateLoss(lambda_stab=lambda_stab)
        self.spectral_loss = SpectralCascadeLoss(
            epsilon_rel=spectral_epsilon_rel, highk_tilt=spectral_highk_tilt,
        )
        self.ortho_loss = OrthogonalityLoss()
        self.wec_loss = WECLoss(gamma_eos=gamma_eos, margin=wec_margin)

        #: dataset-level scales, set by :meth:`set_scales`
        self.tau_scale: float | None = None
        self.p_scale: float | None = None
        #: scalar, or a per-point tensor when a per-snapshot floor is in use
        self.tau_floor: float | torch.Tensor | None = None

    def set_scales(
        self,
        tau_scale: torch.Tensor | float | None = None,
        p_scale: torch.Tensor | float | None = None,
        tau_floor: torch.Tensor | float | None = None,
    ) -> None:
        """Bind the dataset-level τ and P scales used to keep terms dimensionless.

        Both must come from the *training* split only, so that validation and
        test numbers are measured on the same yardstick rather than each split
        normalizing itself (which is what made the previous run's train and
        val ``L_tensor`` incomparable — 0.199 vs 0.020 with no overfitting
        involved, just different denominators).
        """
        if tau_scale is not None:
            self.tau_scale = float(tau_scale)
        if p_scale is not None:
            self.p_scale = float(p_scale)
        if tau_floor is not None:
            # Kept as a tensor when one is given, so a per-snapshot floor
            # survives instead of being collapsed to a single number.
            self.tau_floor = (
                tau_floor if isinstance(tau_floor, torch.Tensor) else float(tau_floor)
            )

    def forward(
        self,
        tau_pred: torch.Tensor,
        tau_dns: torch.Tensor,
        u_con: torch.Tensor,
        grad_u_cov: torch.Tensor,
        g_cov: torch.Tensor,
        rho: torch.Tensor,
        p: torch.Tensor,
        tau_floor: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the point-wise composite loss (excludes spectral term).

        Parameters
        ----------
        tau_pred    : (..., 4, 4) — predicted SGS stress
        tau_dns     : (..., 4, 4) — ground truth SGS stress
        u_con       : (..., 4)    — four-velocity u^μ
        grad_u_cov  : (..., 4, 4) — velocity gradient ∇_ν u_μ
        g_cov       : (4, 4)      — metric tensor g_{μν}
        rho         : (...)       — rest-mass density
        p           : (...)       — thermal pressure
        tau_floor   : optional per-point floor for this batch, overriding the
            one bound by :meth:`set_scales`. This is how a per-snapshot floor
            is supplied, since it differs from batch to batch.

        Returns
        -------
        total : torch.Tensor
            Scalar composite loss (without spectral term).
        components : dict[str, float]
            Individual loss component values for logging.
        """
        L_tensor = self.tensor_loss(
            tau_pred, tau_dns, g_cov,
            tau_scale=self.tau_scale,
            tau_floor=self.tau_floor if tau_floor is None else tau_floor,
        )
        L_P = self.dissipation_loss(tau_pred, tau_dns, grad_u_cov, p_scale=self.p_scale)
        L_wec = self.wec_loss(tau_pred, u_con, rho, p, g_cov)

        total = L_tensor + self.alpha * L_P + self.delta * L_wec

        # L_ortho is identically zero for a basis-contracted output, so it is
        # skipped entirely unless γ > 0 — computing it costs two einsums per
        # batch to always produce 0.0. Set --loss-gamma > 0 to re-enable it as
        # a diagnostic.
        if self.gamma_w != 0.0:
            L_ortho = self.ortho_loss(tau_pred, u_con, g_cov)
            total = total + self.gamma_w * L_ortho
            ortho_val = L_ortho.item()
        else:
            ortho_val = 0.0

        components = {
            "L_tensor": L_tensor.item(),
            "L_P": L_P.item(),
            "L_ortho": ortho_val,
            "L_WEC": L_wec.item(),
        }

        return total, components

    def compute_spectral(
        self,
        tau_pred_2d: torch.Tensor,
        tau_dns_2d: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Compute spectral loss on spatially structured 2D grid data.

        Parameters
        ----------
        tau_pred_2d : (Ny, Nx, 4, 4) — predicted SGS stress on 2D grid
        tau_dns_2d  : (Ny, Nx, 4, 4) — ground truth SGS stress on 2D grid

        Returns
        -------
        loss : torch.Tensor
            β * L_spectral (weighted).
        value : float
            Raw (unweighted) L_spectral value for logging.
        """
        L_spec = self.spectral_loss(tau_pred_2d, tau_dns_2d)
        return self.beta * L_spec, L_spec.item()

    def spectral_term(
        self,
        tau_pred_2d: torch.Tensor,
        tau_dns_2d: torch.Tensor,
    ) -> torch.Tensor:
        """β · L_spectral as a differentiable scalar, for summing into the
        point-wise objective before a single ``backward()``.

        Prefer this over :meth:`compute_spectral` inside a training step. The
        earlier code path ran ``compute_spectral`` in its own pass with its own
        ``zero_grad``/``backward``/``step`` at the end of each epoch, which made
        the last update of every epoch a spectral-only gradient estimated from
        four random snapshots — measured CV of 44% on the reported value, and
        the dominant source of epoch-to-epoch validation noise.
        """
        return self.beta * self.spectral_loss(tau_pred_2d, tau_dns_2d)
