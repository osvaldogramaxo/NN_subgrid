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

    Parameters
    ----------
    epsilon : float
        Regularizer preventing singularity in quasi-laminar regions.
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        tau_pred: torch.Tensor,
        tau_dns: torch.Tensor,
        g_cov: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tau_pred : (..., 4, 4) — predicted SGS stress τ^{μν}
        tau_dns  : (..., 4, 4) — ground truth SGS stress τ^{μν}_DNS
        g_cov    : (4, 4) or (..., 4, 4) — metric tensor g_{μν}
        """
        diff = tau_pred - tau_dns

        # ‖Δτ‖²_F = g_{μα} g_{νβ} Δτ^{μν} Δτ^{αβ}
        diff_low = torch.einsum("...ma,...nb,...ab->...mn", g_cov, g_cov, diff)
        frob_diff = torch.clamp(torch.einsum("...mn,...mn->...", diff, diff_low), min=0.0)

        # ‖τ_DNS‖²_F
        dns_low = torch.einsum("...ma,...nb,...ab->...mn", g_cov, g_cov, tau_dns)
        frob_dns = torch.clamp(torch.einsum("...mn,...mn->...", tau_dns, dns_low), min=0.0)

        # Batch-averaged Frobenius regression normalized by batch DNS intensity.
        # This prevents catastrophic division-by-zero spikes in laminar / quiescent regions.
        return frob_diff.mean() / (frob_dns.mean() + self.epsilon)


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
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tau_pred    : (..., 4, 4) — predicted SGS stress τ^{μν}
        tau_dns     : (..., 4, 4) — ground truth SGS stress τ^{μν}_DNS
        grad_u_cov  : (..., 4, 4) — velocity gradient ∇_ν u_μ
                      Convention: grad_u_cov[..., ν, μ] = ∂_ν u_μ
                      So ∂_μ u_ν = grad_u_cov[..., μ, ν]
        """
        # P_SGS = -τ^{μν} ∇_μ u_ν = -τ^{μν} grad_u_cov[μ, ν]
        P_pred = -torch.einsum("...mn,...mn->...", tau_pred, grad_u_cov)
        P_dns = -torch.einsum("...mn,...mn->...", tau_dns, grad_u_cov)

        # L1 matching of local production rates
        l1_match = torch.abs(P_pred - P_dns).mean()

        # Global stability barrier: penalize net anti-dissipation
        # (negative mean production → energy flowing from subgrid to resolved)
        stability = self.lambda_stab * torch.relu(-P_pred.mean())

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

    Parameters
    ----------
    epsilon : float
        Floor added to spectrum before taking log to prevent -inf.
    """

    def __init__(self, epsilon: float = 1e-12):
        super().__init__()
        self.epsilon = epsilon

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

        # Return k=1..k_max-1, add epsilon for log safety
        return spectrum[1:] + self.epsilon

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

        # Log-space L2 matching
        log_diff = torch.log10(E_pred) - torch.log10(E_dns)
        return (log_diff ** 2).mean()


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
    gamma_eos : float
        Adiabatic index Γ for the ideal gas EOS.
    """

    def __init__(self, gamma_eos: float = 4.0 / 3.0):
        super().__init__()
        self.gamma = gamma_eos

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

        T_eff = T_resolved + tau_pred

        # Energy density observed by comoving fluid: ε = T_eff^{μν} u_μ u_ν
        u_cov = torch.einsum("...ij,...j->...i", g_cov, u_con)
        energy_density = torch.einsum("...mn,...m,...n->...", T_eff, u_cov, u_cov)

        # Penalize negative energy density
        return torch.relu(-energy_density).mean()


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

    Parameters
    ----------
    alpha : float
        Weight for subgrid dissipation rate loss.
    beta : float
        Weight for spectral cascade loss.
    gamma : float
        Weight for orthogonality constraint loss.
    delta : float
        Weight for WEC realizability loss.
    lambda_stab : float
        Global anti-dissipation barrier strength in L_P.
    tensor_epsilon : float
        Regularizer ε for the covariant tensor loss normalization.
    spectral_epsilon : float
        Floor ε for power spectrum before log.
    gamma_eos : float
        Adiabatic index Γ for the ideal gas EOS.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.1,
        gamma: float = 5.0,
        delta: float = 2.0,
        lambda_stab: float = 10.0,
        tensor_epsilon: float = 1e-8,
        spectral_epsilon: float = 1e-12,
        gamma_eos: float = 4.0 / 3.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma_w = gamma      # 'gamma' conflicts with EOS gamma
        self.delta = delta

        self.tensor_loss = CovariantTensorLoss(epsilon=tensor_epsilon)
        self.dissipation_loss = DissipationRateLoss(lambda_stab=lambda_stab)
        self.spectral_loss = SpectralCascadeLoss(epsilon=spectral_epsilon)
        self.ortho_loss = OrthogonalityLoss()
        self.wec_loss = WECLoss(gamma_eos=gamma_eos)

    def forward(
        self,
        tau_pred: torch.Tensor,
        tau_dns: torch.Tensor,
        u_con: torch.Tensor,
        grad_u_cov: torch.Tensor,
        g_cov: torch.Tensor,
        rho: torch.Tensor,
        p: torch.Tensor,
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

        Returns
        -------
        total : torch.Tensor
            Scalar composite loss (without spectral term).
        components : dict[str, float]
            Individual loss component values for logging.
        """
        L_tensor = self.tensor_loss(tau_pred, tau_dns, g_cov)
        L_P = self.dissipation_loss(tau_pred, tau_dns, grad_u_cov)
        L_ortho = self.ortho_loss(tau_pred, u_con, g_cov)
        L_wec = self.wec_loss(tau_pred, u_con, rho, p, g_cov)

        total = (L_tensor
                 + self.alpha * L_P
                 + self.gamma_w * L_ortho
                 + self.delta * L_wec)

        components = {
            "L_tensor": L_tensor.item(),
            "L_P": L_P.item(),
            "L_ortho": L_ortho.item(),
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
