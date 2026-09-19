r"""
crmhd_tbnn.py

Tensor Basis Neural Network (TBNN) formulation for Compressible Relativistic 
Magnetohydrodynamics (CRMHD) Subgrid-Scale Closures.

Conventions:
- Metric signature: (-, +, +, +)
- Greek indices: spacetime coordinates {0, 1, 2, 3}
- Covariant derivative tensor: grad_u[..., \nu, \mu] = \nabla_\nu u_\mu
- All basis tensors E^(n)\mu\nu are output in contravariant form. (https://www.youtube.com/watch?v=6L3iF_h4s9E)
"""

from __future__ import annotations

import os
from pathlib import Path
import torch
import torch.nn as nn
from typing_extensions import override


def get_minkowski_metric(
    batch_shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Returns the Minkowski metric eta_{\mu\nu} and its inverse eta^{\mu\nu}."""
    eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=dtype, device=device))
    if len(batch_shape) > 0:
        eta = eta.expand(*batch_shape, 4, 4).clone()
    eta_inv = eta.clone()
    return eta, eta_inv


def decompose_relativistic_kinematics(
    u_con: torch.Tensor,
    grad_u_cov: torch.Tensor,
    g_cov: torch.Tensor,
    g_con: torch.Tensor,
) -> dict[str, torch.Tensor]:
    r"""Decomposes the four-velocity gradient into irreducible kinematic quantities.

    Parameters
    ----------
    u_con : torch.Tensor
        Four-velocity vector $u^\mu$ with shape (..., 4).
    grad_u_cov : torch.Tensor
        Four-velocity gradient $\nabla_\nu u_\mu$ with shape (..., 4, 4),
        where index -2 is $\nu$ and index -1 is $\mu$.
    g_cov : torch.Tensor
        Spacetime metric $g_{\mu\nu}$ with shape (..., 4, 4).
    g_con : torch.Tensor
        Inverse spacetime metric $g^{\mu\nu}$ with shape (..., 4, 4).

    Returns
    -------
    dict[str, torch.Tensor] containing h_cov, h_con, a_cov, a_con, theta,
    sigma_cov, sigma_con, omega_cov, omega_con.
    """
    # Lower velocity index: u_\mu = g_{\mu\alpha} u^\alpha
    u_cov = torch.einsum("...ij,...j->...i", g_cov, u_con)

    # Spatial projection tensor: h_{\mu\nu} = g_{\mu\nu} + u_\mu u_\nu
    h_cov = g_cov + torch.einsum("...i,...j->...ij", u_cov, u_cov)
    # Inverse projector: h^{\mu\nu} = g^{\mu\nu} + u^\mu u^\nu
    h_con = g_con + torch.einsum("...i,...j->...ij", u_con, u_con)
    # Mixed projector: h^\mu_\alpha = \delta^\mu_\alpha + u^\mu u_\alpha
    delta = torch.eye(4, dtype=u_con.dtype, device=u_con.device).expand_as(g_cov)
    h_mixed = delta + torch.einsum("...i,...j->...ij", u_con, u_cov)

    # Four-acceleration: a_\mu = u^\nu \nabla_\nu u_\mu
    a_cov = torch.einsum("...j,...ji->...i", u_con, grad_u_cov)
    a_con = torch.einsum("...ij,...j->...i", g_con, a_cov)

    # Expansion scalar: \theta = g^{\mu\nu} \nabla_\nu u_\mu
    theta = torch.einsum("...ij,...ji->...", g_con, grad_u_cov)

    # Fully projected spatial gradient: \bar{\nabla}_\mu u_\nu = h^\alpha_\mu h^\beta_\nu \nabla_\alpha u_\beta
    # h^\alpha_\mu = \delta^\alpha_\mu + u^\alpha u_\mu, where index 0 is \alpha (upper) and index 1 is \mu (lower).
    projected_grad = torch.einsum("...ai,...bj,...ab->...ij", h_mixed, h_mixed, grad_u_cov)

    # Shear tensor: \sigma_{\mu\nu} = 0.5 * (\bar{\nabla}_\mu u_\nu + \bar{\nabla}_\nu u_\mu) - (1/3) \theta h_{\mu\nu}
    sym_grad = 0.5 * (projected_grad + projected_grad.transpose(-1, -2))
    sigma_cov = sym_grad - (1.0 / 3.0) * theta.unsqueeze(-1).unsqueeze(-1) * h_cov

    # Vorticity tensor: \omega_{\mu\nu} = 0.5 * (\bar{\nabla}_\mu u_\nu - \bar{\nabla}_\nu u_\mu)
    omega_cov = 0.5 * (projected_grad - projected_grad.transpose(-1, -2))

    # Raise indices to contravariant form
    sigma_con = torch.einsum("...ia,...jb,...ab->...ij", g_con, g_con, sigma_cov)
    omega_con = torch.einsum("...ia,...jb,...ab->...ij", g_con, g_con, omega_cov)

    return {
        "h_cov": h_cov,
        "h_con": h_con,
        "a_cov": a_cov,
        "a_con": a_con,
        "theta": theta,
        "sigma_cov": sigma_cov,
        "sigma_con": sigma_con,
        "omega_cov": omega_cov,
        "omega_con": omega_con,
    }


def compute_crmhd_invariants(
    sigma_con: torch.Tensor,
    sigma_cov: torch.Tensor,
    omega_con: torch.Tensor,
    omega_cov: torch.Tensor,
    b_con: torch.Tensor,
    b_cov: torch.Tensor,
    a_con: torch.Tensor,
    a_cov: torch.Tensor,
    g_cov: torch.Tensor,
    g_con: torch.Tensor,
) -> torch.Tensor:
    r"""Computes the 15 irreducible Lorentz scalar invariants $\Lambda_1 \dots \Lambda_{15}$.

    Returns
    -------
    torch.Tensor with shape (..., 15).
    """
    # Mixed tensors: T^\mu_\nu = T^{\mu\alpha} g_{\alpha\nu}
    sigma_mixed = torch.einsum("...ia,...aj->...ij", sigma_con, g_cov)
    omega_mixed = torch.einsum("...ia,...aj->...ij", omega_con, g_cov)

    # Pure hydrodynamic invariants
    lam1 = torch.einsum("...ij,...ij->...", sigma_con, sigma_cov)
    lam2 = torch.einsum("...ij,...ij->...", omega_con, omega_cov)
    
    # \Lambda_3 = \text{Tr}(\sigma^3)
    sigma_sq = torch.einsum("...ia,...aj->...ij", sigma_mixed, sigma_mixed)
    lam3 = torch.einsum("...ia,...ai->...", sigma_sq, sigma_mixed)

    # \Lambda_4 = \text{Tr}(\omega^2 \sigma)
    omega_sq = torch.einsum("...ia,...aj->...ij", omega_mixed, omega_mixed)
    lam4 = torch.einsum("...ia,...ai->...", omega_sq, sigma_mixed)

    # \Lambda_5 = \text{Tr}(\omega^2 \sigma^2)
    lam5 = torch.einsum("...ia,...ai->...", omega_sq, sigma_sq)

    # Magnetic coupling invariants
    lam6 = torch.einsum("...i,...i->...", b_con, b_cov)
    lam7 = torch.einsum("...i,...ij,...j->...", b_con, sigma_cov, b_con)
    
    # \Lambda_8 = b_\mu \sigma^{\mu\alpha} \sigma_{\alpha\nu} b^\nu
    sigma_b_cov = torch.einsum("...ij,...j->...i", sigma_cov, b_con)
    sigma_b_con = torch.einsum("...ij,...j->...i", g_con, sigma_b_cov)
    lam8 = torch.einsum("...i,...i->...", sigma_b_con, sigma_b_cov)

    # \Lambda_9 = b_\mu \omega^{\mu\alpha} \omega_{\alpha\nu} b^\nu
    omega_b_cov = torch.einsum("...ij,...j->...i", omega_cov, b_con)
    omega_b_con = torch.einsum("...ij,...j->...i", g_con, omega_b_cov)
    lam9 = torch.einsum("...i,...i->...", omega_b_con, omega_b_cov)

    # \Lambda_{10} = b_\mu \omega^{\mu\alpha} \sigma_{\alpha\nu} b^\nu
    lam10 = torch.einsum("...i,...i->...", omega_b_con, sigma_b_cov)

    # Acceleration & cross-coupling invariants
    lam11 = torch.einsum("...i,...i->...", a_con, a_cov)
    lam12 = torch.einsum("...i,...ij,...j->...", a_con, sigma_cov, a_con)

    omega_a_cov = torch.einsum("...ij,...j->...i", omega_cov, a_con)
    omega_a_con = torch.einsum("...ij,...j->...i", g_con, omega_a_cov)
    lam13 = torch.einsum("...i,...i->...", omega_a_con, omega_a_cov)

    lam14 = torch.einsum("...i,...i->...", a_con, b_cov)
    lam15 = torch.einsum("...i,...ij,...j->...", a_con, sigma_cov, b_con)

    return torch.stack(
        [
            lam1, lam2, lam3, lam4, lam5,
            lam6, lam7, lam8, lam9, lam10,
            lam11, lam12, lam13, lam14, lam15,
        ],
        dim=-1,
    )


def compute_crmhd_basis_tensors(
    sigma_con: torch.Tensor,
    omega_con: torch.Tensor,
    b_con: torch.Tensor,
    a_con: torch.Tensor,
    h_con: torch.Tensor,
    g_cov: torch.Tensor,
) -> torch.Tensor:
    r"""Constructs the 12 isotropic symmetric basis tensors $E^{(0)\mu\nu} \dots E^{(11)\mu\nu}$.

    Returns
    -------
    torch.Tensor with shape (..., 12, 4, 4).
    """
    # Lower secondary index for matrix multiplications: T^{\mu\alpha} g_{\alpha\beta}
    sigma_low = torch.einsum("...ia,...ab->...ib", sigma_con, g_cov)
    omega_low = torch.einsum("...ia,...ab->...ib", omega_con, g_cov)

    # Contraction products: (T_1 T_2)^{\mu\nu} = T_1^{\mu\alpha} g_{\alpha\beta} T_2^{\beta\nu}
    sig_om = torch.einsum("...ia,...ab->...ib", sigma_low, omega_con)
    om_sig = torch.einsum("...ia,...ab->...ib", omega_low, sigma_con)

    sig_sig = torch.einsum("...ia,...ab->...ib", sigma_low, sigma_con)
    om_om = torch.einsum("...ia,...ab->...ib", omega_low, omega_con)

    om_sig_sig = torch.einsum("...ia,...ab->...ib", omega_low, sig_sig)
    sig_sig_om = torch.einsum("...ia,...ab->...ib", torch.einsum("...ia,...ab->...ib", sig_sig, g_cov), omega_con)

    # Traces for traceless projectors
    tr_sig_sq = torch.einsum("...ia,...ab,...bi->...", sigma_low, sigma_con, g_cov)
    tr_om_sq = torch.einsum("...ia,...ab,...bi->...", omega_low, omega_con, g_cov)
    b_sq = torch.einsum("...i,...ia,...a->...", b_con, g_cov, b_con)
    a_sq = torch.einsum("...i,...ia,...a->...", a_con, g_cov, a_con)
    a_dot_b = torch.einsum("...i,...ia,...a->...", a_con, g_cov, b_con)
    b_sig_b = torch.einsum("...i,...ij,...j->...", b_con, sigma_low, b_con)
    a_sig_a = torch.einsum("...i,...ij,...j->...", a_con, sigma_low, a_con)

    # Outer products
    bb = torch.einsum("...i,...j->...ij", b_con, b_con)
    aa = torch.einsum("...i,...j->...ij", a_con, a_con)
    ab_sym = torch.einsum("...i,...j->...ij", a_con, b_con) + torch.einsum("...i,...j->...ij", b_con, a_con)

    # Tensor projections
    sig_b = torch.einsum("...ia,...a->...i", sigma_con, torch.einsum("...ij,...j->...i", g_cov, b_con))
    sig_bb = torch.einsum("...i,...j->...ij", sig_b, b_con) + torch.einsum("...i,...j->...ij", b_con, sig_b)

    om_b = torch.einsum("...ia,...a->...i", omega_con, torch.einsum("...ij,...j->...i", g_cov, b_con))
    om_bb = torch.einsum("...i,...j->...ij", om_b, b_con) - torch.einsum("...i,...j->...ij", b_con, om_b)

    sig_a = torch.einsum("...ia,...a->...i", sigma_con, torch.einsum("...ij,...j->...i", g_cov, a_con))
    sig_aa = torch.einsum("...i,...j->...ij", sig_a, a_con) + torch.einsum("...i,...j->...ij", a_con, sig_a)

    # Assemble 12 generators
    e0 = h_con
    e1 = sigma_con
    e2 = sig_om - om_sig
    e3 = sig_sig - (1.0 / 3.0) * tr_sig_sq.unsqueeze(-1).unsqueeze(-1) * h_con
    e4 = om_om - (1.0 / 3.0) * tr_om_sq.unsqueeze(-1).unsqueeze(-1) * h_con
    e5 = om_sig_sig - sig_sig_om
    e6 = bb - (1.0 / 3.0) * b_sq.unsqueeze(-1).unsqueeze(-1) * h_con
    e7 = sig_bb - (2.0 / 3.0) * b_sig_b.unsqueeze(-1).unsqueeze(-1) * h_con
    e8 = om_bb
    e9 = aa - (1.0 / 3.0) * a_sq.unsqueeze(-1).unsqueeze(-1) * h_con
    e10 = ab_sym - (2.0 / 3.0) * a_dot_b.unsqueeze(-1).unsqueeze(-1) * h_con
    e11 = sig_aa - (2.0 / 3.0) * a_sig_a.unsqueeze(-1).unsqueeze(-1) * h_con

    return torch.stack([e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11], dim=-3)


class CRMHD_TBNN(nn.Module):
    r"""Tensor Basis Neural Network for Compressible Relativistic MHD SGS Closures[cite: 1].

    Ingests the coordinate-free scalar invariant space $\Lambda_k$ plus
    thermodynamic features (theta, M_t) through deep dense layers and contracts
    the predicted scalar coefficients against the analytical isotropic basis $E^{(n)\mu\nu}$[cite: 1].
    """

    def __init__(
        self,
        num_invariants: int = 15,
        num_thermo_inputs: int = 2,
        num_basis_tensors: int = 12,
        num_hidden_layers: int = 3,
        hidden_dim: int = 512,
        act_fn: nn.Module = nn.GELU(),
    ):
        super().__init__()
        in_dim = num_invariants + num_thermo_inputs
        self.num_basis = num_basis_tensors

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act_fn)

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn)

        self.hidden_mlp = nn.Sequential(*layers)

        # Output head for basis tensor coefficients G^(n)
        self.coeff_head = nn.Linear(hidden_dim, num_basis_tensors, bias=False)
        # Separate scalar head for subgrid isotropic pressure / bulk dissipation
        self.p_sgs_head = nn.Linear(hidden_dim, 1)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        num_thermo_inputs: int = 2,
    ) -> CRMHD_TBNN:
        r"""Instantiate CRMHD_TBNN by inspecting layer weights to adjust shapes dynamically."""
        # 1. Inspect first linear layer for hidden_dim and total input dimension
        w0 = state_dict["hidden_mlp.0.weight"]
        hidden_dim = w0.shape[0]
        in_dim = w0.shape[1]

        # 2. Count hidden linear layers
        num_hidden_layers = sum(
            1 for k in state_dict.keys() if k.startswith("hidden_mlp.") and k.endswith(".weight")
        )

        # 3. Inspect basis tensor dimension from coeff_head
        if "coeff_head.weight" in state_dict:
            num_basis_tensors = state_dict["coeff_head.weight"].shape[0]
        else:
            num_basis_tensors = 12

        # 4. Compute number of invariants
        num_invariants = in_dim - num_thermo_inputs

        model = cls(
            num_invariants=num_invariants,
            num_thermo_inputs=num_thermo_inputs,
            num_basis_tensors=num_basis_tensors,
            num_hidden_layers=num_hidden_layers,
            hidden_dim=hidden_dim,
        )
        _ = model.load_state_dict(state_dict)
        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | os.PathLike[str] | dict[str, object],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        num_thermo_inputs: int = 2,
    ) -> tuple[CRMHD_TBNN, dict[str, object]]:
        r"""Load checkpoint weights first, adjust network creation shapes to match, and load state dict."""
        ckpt_dict: dict[str, object]
        if isinstance(checkpoint, (str, os.PathLike)):
            loaded = torch.load(checkpoint, map_location=device, weights_only=False)
            if isinstance(loaded, dict):
                ckpt_dict = loaded
            else:
                raise TypeError(f"Expected dict from checkpoint, got {type(loaded)}")
        else:
            ckpt_dict = checkpoint

        raw_state = ckpt_dict.get("model_state_dict", ckpt_dict)
        if not isinstance(raw_state, dict):
            raise TypeError(f"Expected dict for state_dict, got {type(raw_state)}")
        state_dict: dict[str, torch.Tensor] = raw_state

        thermo_inputs_raw = ckpt_dict.get("num_thermo_inputs", num_thermo_inputs)
        thermo_inputs = thermo_inputs_raw if isinstance(thermo_inputs_raw, int) else num_thermo_inputs

        model = cls.from_state_dict(
            state_dict=state_dict,
            num_thermo_inputs=thermo_inputs,
        ).to(dtype=dtype, device=device)

        return model, ckpt_dict

    @override
    def forward(
        self,
        invariants: torch.Tensor,
        thermo_inputs: torch.Tensor,
        basis_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward evaluation of the covariant closure.

        Parameters
        ----------
        invariants : torch.Tensor
            Scalar invariants $\Lambda_1 \dots \Lambda_{15}$ with shape (..., 15).
        thermo_inputs : torch.Tensor
            Thermodynamic/dilatational inputs $(\theta, M_t)$ with shape (..., 2).
        basis_tensors : torch.Tensor
            Isotropic tensor basis $E^{(n)\mu\nu}$ with shape (..., 12, 4, 4).

        Returns
        -------
        delta_T_sgs : torch.Tensor
            Reconstructed subgrid stress tensor $\Delta T^{\mu\nu}_{\text{SGS}}$ with shape (..., 4, 4).
        coeffs : torch.Tensor
            Predicted scalar coefficients $G^{(n)}$ with shape (..., 12).
        p_sgs : torch.Tensor
            Predicted scalar isotropic SGS pressure $P_{\text{SGS}}$ with shape (..., 1).
        """
        features = torch.cat([invariants, thermo_inputs], dim=-1)
        latent = self.hidden_mlp(features)

        coeffs = self.coeff_head(latent)  # Shape: (..., 12)
        p_sgs = self.p_sgs_head(latent)   # Shape: (..., 1)

        # Multiplicative layer: \Delta T^{\mu\nu} = \sum_{n=0}^{11} G^{(n)} E^{(n)\mu\nu}
        delta_T_sgs = torch.einsum("...n,...nij->...ij", coeffs, basis_tensors)

        return delta_T_sgs, coeffs, p_sgs


if __name__ == "__main__":
    # Verification test on synthetic batched data
    batch_size = 32
    device = torch.device("cpu")

    # Metrics
    g_cov, g_con = get_minkowski_metric((batch_size,), device=device)

    # Unit timelike four-velocity: u^\mu = (1, 0, 0, 0)
    u_con = torch.zeros(batch_size, 4, dtype=torch.float64, device=device)
    u_con[:, 0] = 1.0

    # Random spatial velocity gradient
    grad_u = torch.randn(batch_size, 4, 4, dtype=torch.float64, device=device) * 0.05
    # Enforce u^\mu \nabla_\nu u_\mu = 0
    grad_u[:, :, 0] = 0.0

    # Comoving spatial vectors
    b_con = torch.randn(batch_size, 4, dtype=torch.float64, device=device) * 0.1
    b_con[:, 0] = 0.0  # b^\mu u_\mu = 0
    b_cov = torch.einsum("...ij,...j->...i", g_cov, b_con)

    # Kinematics
    kinematics = decompose_relativistic_kinematics(u_con, grad_u, g_cov, g_con)

    # Invariants and Bases
    invariants = compute_crmhd_invariants(
        kinematics["sigma_con"],
        kinematics["sigma_cov"],
        kinematics["omega_con"],
        kinematics["omega_cov"],
        b_con,
        b_cov,
        kinematics["a_con"],
        kinematics["a_cov"],
        g_cov,
        g_con,
    )

    basis_tensors = compute_crmhd_basis_tensors(
        kinematics["sigma_con"],
        kinematics["omega_con"],
        b_con,
        kinematics["a_con"],
        kinematics["h_con"],
        g_cov,
    )

    # Thermodynamic features: (expansion scalar \theta, turbulent Mach number M_t)
    m_t = torch.rand(batch_size, 1, dtype=torch.float64, device=device) * 0.5
    thermo_inputs = torch.cat([kinematics["theta"].unsqueeze(-1), m_t], dim=-1)

    # Model evaluation
    model = CRMHD_TBNN(num_invariants=15, num_thermo_inputs=2, num_basis_tensors=12).to(torch.float64)
    delta_T, g_coeffs, p_sgs = model(invariants, thermo_inputs, basis_tensors)

    print("Execution verification:")
    print("Invariants tensor shape:    ", invariants.shape)
    print("Basis tensors shape:        ", basis_tensors.shape)
    print("Output stress tensor shape: ", delta_T.shape)
    print("Predicted coefficients shape:", g_coeffs.shape)
    assert not torch.isnan(delta_T).any(), "NaN values detected in forward pass output."