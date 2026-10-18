r"""
minit.py

The MInIT sub-grid closure (Miravet-Tenés, Cerdá-Durán, Obergaulinger & Font
2023, arXiv:2308.06041) adapted to the 2D special-relativistic hydrodynamics
solver in :mod:`src.solver`, for use as a competing closure alongside the TBNN.

What survives the port
----------------------
MInIT is an *MHD* model: it closes the turbulent Maxwell (``M_ij``), Reynolds
(``R_ij``) and Faraday (``F_ij``) stresses, all three proportional to a single
evolved scalar, the sub-grid turbulent kinetic energy density ``e_KH``. This
code base solves pure SRHD — there is no magnetic field for the Maxwell and
Faraday closures to act on — so only the Reynolds part is transferable. That
part is exactly the object the TBNN predicts, so the comparison is well posed:
a one-parameter algebraic closure against a 15-invariant tensor-basis network,
scored on the same target.

The paper's coefficients are kept in :data:`BETA_KH_PAPER` for the
"as-published transfer" baseline, but they were calibrated on *Newtonian, 3D*
MHD box simulations at filter scales Δ_f/L = 0.08–0.11. This port runs 2D SRHD
at Δ_f/L = 0.016, so ``src/minit_calibrate.py`` refits (β, A_γ, C) against
the local DNS; that refit is the "serious competitor" configuration.

Mapping onto the solver's stress
--------------------------------
The solver's SGS variable is the fluid-frame stress τ_⊥^{μν} (see
``dataset.compute_sgs_stress``), which carries units of energy density. MInIT's
Reynolds closure is R_ij = β_ij e_KH / ρ, and τ^{ij} = ρ R_ij, so the density
cancels exactly:

.. math::
    \tau_\perp^{ij} = \beta^{\rm KH}_{ij}\, e_{\rm KH}

The time components follow from transversality u_μ τ_⊥^{μν} = 0, i.e.
τ^{0ν} = v_i τ^{iν}, and are filled in here so the tensor handed to the solver
is consistent with the one the exact-DNS and TBNN closures produce.

``e_KH`` is defined as e_KH = ½ ρ Σ_i R_ii = ½ Σ_i τ^{ii}. In 2D the sum runs
over x and y only — τ^{zz} of a 2D SRHD state is identically zero, since
T^{zz} = p g^{zz} and the box filter of p equals the coarse p̄ — so the
normalisation constraint on the coefficients is β_xx + β_yy = 2, against the
paper's 3D β_xx + β_yy + β_zz = 1.99.

The stiff ODE has a closed form
-------------------------------
The turbulent-energy equation (paper Eq. 12, advective term separated out)

.. math::
    \partial_t e = 2\gamma_{\rm KH} e - \frac{C}{\sqrt{\rho}\,\lambda_{\rm KH}} e^{3/2}

is a Bernoulli equation. Substituting w = e^{-1/2} linearises it exactly:

.. math::
    \dot w = -\gamma w + K/2, \qquad K = C / (\sqrt{\rho}\,\lambda_{\rm KH})

so for γ, ρ frozen over a step the update is exact, unconditionally stable and
positivity preserving:

.. math::
    w^{n+1} = w^n e^{-\gamma\Delta t} + \frac{K}{2}\,\frac{1 - e^{-\gamma\Delta t}}{\gamma}

This replaces the paper's Strang splitting with implicit sub-stepping: there is
no stiffness to fight, no scipy dependency, and the whole update is a handful
of vectorised torch ops on the coarse grid. The saturation value
e_sat = (2γ√ρ λ_KH / C)² is the fixed point of the same equation, and
reproduces the paper's Eq. 15 scaling analytically.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.features import periodic_gradient_2d
from src.solver import clip_backscatter, clip_sgs_tensor, smooth_sgs_tensor


# ---------------------------------------------------------------------------
# Calibrated coefficients from the paper (Table 2, Sec. 4.3.3)
# ---------------------------------------------------------------------------

# Reynolds closure, R_ij = beta_ij / rho * e_KH. The only tensor of the three
# that has anything to act on in pure hydrodynamics.
BETA_KH_PAPER = {
    "xx": 0.74, "yy": 0.64, "zz": 0.61,
    "xy": -0.01, "xz": -0.041, "yz": -0.02,
}

# Maxwell (M_ij = alpha_ij e_KH) and Faraday (F_ij = gamma_ij / sqrt(rho) e_KH)
# closures. Recorded for completeness and for anyone extending this to RMHD;
# both are inert in an SRHD solver.
ALPHA_KH_PAPER = {
    "xx": 0.70, "yy": 0.64, "zz": 0.64,
    "xy": -0.01, "xz": -0.02, "yz": -0.026,
}
GAMMA_KH_PAPER = {"xy": -0.010, "xz": -0.01, "yz": -0.008}

A_GAMMA_PAPER = 0.14    # growth-rate calibration constant (Eq. 8, Eq. 23)
C_PAPER = 10.6          # Kolmogorov dissipation constant (Eq. 13, Eq. 26)

# lambda_KH = 15.8 * Delta_f — wavelength of the fastest growing KHI mode with
# the shear length a_l assimilated to the filter scale (Eq. 9 with Eq. 24).
LAMBDA_OVER_DELTA = 15.8


@dataclass
class MInITCoefficients:
    r"""The full free-parameter set of the 2D SRHD MInIT closure.

    Two instances matter for the comparison: :meth:`paper`, which transfers the
    published 3D Newtonian MHD calibration verbatim, and whatever
    ``src/minit_calibrate.py`` fits against the local DNS. The difference
    between the two is itself a result — it measures how much of MInIT's
    performance is the model and how much is the calibration.

    Parameters
    ----------
    beta_xx, beta_yy, beta_xy : Reynolds closure coefficients (Eq. 16b).
    A_gamma : constant relating resolved shear to growth rate (Eq. 23).
    C : Kolmogorov dissipation constant (Eq. 13).
    """

    beta_xx: float = BETA_KH_PAPER["xx"]
    beta_yy: float = BETA_KH_PAPER["yy"]
    beta_xy: float = BETA_KH_PAPER["xy"]
    A_gamma: float = A_GAMMA_PAPER
    C: float = C_PAPER

    @classmethod
    def paper(cls) -> "MInITCoefficients":
        """The published calibration, transferred as-is."""
        return cls()

    @classmethod
    def load(cls, path: str | Path) -> "MInITCoefficients":
        """Read a refit produced by ``src/minit_calibrate.py``."""
        with open(path) as fh:
            payload = json.load(fh)
        fitted = payload.get("coefficients", payload)
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: float(v) for k, v in fitted.items() if k in fields})

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"coefficients": asdict(self)}, fh, indent=2)

    @property
    def trace_residual(self) -> float:
        r"""``beta_xx + beta_yy - 2``, which the definition of e_KH forces to zero.

        e_KH = ½ Σ_i τ^{ii} and τ^{ij} = β_ij e_KH together imply
        ½(β_xx + β_yy) = 1 in 2D. A fit that averages per-component ratios does
        not enforce this, so the residual is a genuine consistency diagnostic:
        the paper's 3D coefficients leave 1.99 - 2 = -0.01.
        """
        return self.beta_xx + self.beta_yy - 2.0

    def normalized(self) -> "MInITCoefficients":
        """Rescale the diagonal so ``trace_residual`` is exactly zero."""
        total = self.beta_xx + self.beta_yy
        if abs(total) < 1e-12:
            raise ValueError("cannot normalize coefficients with vanishing diagonal trace")
        scale = 2.0 / total
        return MInITCoefficients(
            beta_xx=self.beta_xx * scale,
            beta_yy=self.beta_yy * scale,
            beta_xy=self.beta_xy * scale,
            A_gamma=self.A_gamma,
            C=self.C,
        )


# ---------------------------------------------------------------------------
# Model pieces, as free functions so the a-priori fit can reuse them
# ---------------------------------------------------------------------------

def growth_rate(
    vx: torch.Tensor,
    vy: torch.Tensor,
    dx: float,
    dy: float,
    A_gamma: float = A_GAMMA_PAPER,
    mode: str = "dvxdy",
    order: int = 4,
) -> torch.Tensor:
    r"""KHI growth rate estimated from the resolved velocity field (Eq. 23).

    The paper evaluates ``gamma = A_gamma * d(vx)/dy`` with a centred difference
    across the shear layer, then averages over sample points *and over the two
    shear layers*, which have opposite sign. Pointwise on a grid that averaging
    is unavailable, so the magnitude is taken instead — a negative growth rate
    would otherwise drive e_KH to zero on one of the two layers.

    Parameters
    ----------
    vx, vy : (Ny, Nx) tensors — resolved (coarse-grid) velocities.
    dx, dy : grid spacings.
    A_gamma : calibration constant.
    mode : str
        ``"dvxdy"`` (default) — ``A_gamma * |d(vx)/dy|``, the literal Eq. 23.
        ``"strain"`` — ``A_gamma * sqrt(2 S_ij S_ij)`` over the resolved rate of
        strain, which is rotationally invariant and agrees with ``"dvxdy"``
        exactly for a pure shear, so the calibration of ``A_gamma`` carries over.
    order : finite-difference accuracy, 2 or 4.

    Returns
    -------
    gamma_KH : (Ny, Nx) tensor, non-negative.
    """
    dvx_dx, dvx_dy = periodic_gradient_2d(vx, dx, dy, order=order)

    if mode == "dvxdy":
        return A_gamma * dvx_dy.abs()

    if mode == "strain":
        dvy_dx, dvy_dy = periodic_gradient_2d(vy, dx, dy, order=order)
        S_xx = dvx_dx
        S_yy = dvy_dy
        S_xy = 0.5 * (dvx_dy + dvy_dx)
        # 2 S_ij S_ij = 2(S_xx^2 + S_yy^2 + 2 S_xy^2); reduces to |dvx/dy| for
        # a pure shear, which is what fixes the normalisation.
        mag_sq = 2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2)
        return A_gamma * mag_sq.clamp(min=0.0).sqrt()

    raise ValueError(f"unknown growth-rate mode {mode!r}; use 'dvxdy' or 'strain'")


def saturation_energy(
    gamma_KH: torch.Tensor,
    rho: torch.Tensor,
    lambda_KH: float,
    C: float = C_PAPER,
) -> torch.Tensor:
    r"""Fixed point of the turbulent-energy equation, ``e_sat = (2 gamma sqrt(rho) lambda / C)^2``.

    Setting the right-hand side of Eq. (12) to zero gives the energy density the
    cascade can sustain against the instability drive. This is the analytic form
    of the paper's Eq. (15) scaling, and is what the ``floor_fraction`` seeding
    mode is measured against.
    """
    return (2.0 * gamma_KH * rho.clamp(min=1e-15).sqrt() * lambda_KH / C) ** 2


def advance_e_kh(
    e_KH: torch.Tensor,
    gamma_KH: torch.Tensor,
    rho: torch.Tensor,
    dt: float,
    lambda_KH: float,
    C: float = C_PAPER,
) -> torch.Tensor:
    r"""Exact update of ``e_KH`` over ``dt`` for frozen ``gamma_KH`` and ``rho``.

    Integrates ``de/dt = 2 gamma e - C e^{3/2} / (sqrt(rho) lambda)`` in closed
    form through ``w = e^{-1/2}`` (see the module docstring). Exact for constant
    coefficients, unconditionally stable, and positivity preserving for any
    ``dt``, so the closure never constrains the hydrodynamic time step.

    ``e_KH = 0`` is a fixed point of the equation and is preserved exactly; the
    closure therefore needs an explicit seed (see :class:`MInITClosure`).
    """
    K = C / (rho.clamp(min=1e-15).sqrt() * lambda_KH)

    decay = torch.exp(-gamma_KH * dt)
    # (1 - exp(-g dt)) / g, continued to its limit dt as g -> 0.
    g_dt = gamma_KH * dt
    phi = torch.where(
        g_dt > 1e-8,
        -torch.expm1(-g_dt) / gamma_KH.clamp(min=1e-30),
        dt * (1.0 - 0.5 * g_dt),
    )

    positive = e_KH > 0
    w = torch.where(positive, e_KH.clamp(min=1e-300).rsqrt(), torch.zeros_like(e_KH))
    w_new = w * decay + 0.5 * K * phi

    return torch.where(positive, w_new.clamp(min=1e-300) ** -2, torch.zeros_like(e_KH))


def advect_e_kh(
    e_KH: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    dx: float,
    dy: float,
    dt: float,
) -> torch.Tensor:
    r"""Apply the advective term ``-d_i(v_i e_KH)`` with first-order donor-cell upwinding.

    The paper drops this term: its box simulations are periodic along the only
    direction with a non-zero mean velocity, so any average along that direction
    is blind to advection. That argument holds along x here too, but the KHI
    roll-up transports turbulent energy across the layer in y, where it does not
    hold — hence the term is available, and both settings are worth testing.

    Donor-cell rather than a higher-order scheme because e_KH must stay
    non-negative for the ``w = e^{-1/2}`` substitution to be defined. Stable
    under the hydrodynamic CFL condition, which is set by the fast speed and so
    is always at least as restrictive as the advective one; the final clamp
    guards the corner where the 2D sum of Courant numbers still exceeds one.
    """
    def _flux_divergence(v: torch.Tensor, dim: int, d: float) -> torch.Tensor:
        e_plus = torch.roll(e_KH, -1, dims=dim)
        v_face = 0.5 * (v + torch.roll(v, -1, dims=dim))
        e_face = torch.where(v_face > 0, e_KH, e_plus)
        flux = v_face * e_face                      # at i+1/2
        return (flux - torch.roll(flux, 1, dims=dim)) / d

    div = _flux_divergence(vx, -1, dx) + _flux_divergence(vy, -2, dy)
    return (e_KH - dt * div).clamp(min=0.0)


def assemble_tau(
    e_KH: torch.Tensor,
    prims: torch.Tensor,
    coeffs: MInITCoefficients,
) -> torch.Tensor:
    r"""Build the fluid-frame stress ``tau_perp^{mu nu} = beta_ij e_KH``.

    The spatial block is the MInIT Reynolds closure with the density cancelled
    (module docstring). The time components are then fixed by transversality,
    ``u_mu tau_perp^{mu nu} = 0``, which for ``u_mu = W(-1, v_x, v_y, 0)`` reads
    ``tau^{0 nu} = v_i tau^{i nu}``. Filling them here — rather than leaving
    zeros for ``compute_sgs_viscous_fluxes`` to derive — keeps the tensor's norm
    comparable to the TBNN's and the exact DNS stress, which matters as soon as
    realizability clipping is switched on.

    The zz component is left at zero: for a 2D SRHD state ``T^{zz} = p g^{zz}``,
    whose box filter equals the coarse ``p_bar``, so the exact SGS stress has
    ``tau^{zz} = 0`` identically. It is also dynamically inert, there being no
    z-momentum equation.

    Parameters
    ----------
    e_KH : (Ny, Nx) tensor — turbulent kinetic energy density.
    prims : (Ny, Nx, 4) tensor — [rho, vx, vy, p].
    coeffs : the closure coefficients.

    Returns
    -------
    tau_perp : (Ny, Nx, 4, 4) tensor.
    """
    vx = prims[..., 1]
    vy = prims[..., 2]

    tau = torch.zeros(*e_KH.shape, 4, 4, dtype=prims.dtype, device=prims.device)
    tau[..., 1, 1] = coeffs.beta_xx * e_KH
    tau[..., 2, 2] = coeffs.beta_yy * e_KH
    tau[..., 1, 2] = coeffs.beta_xy * e_KH
    tau[..., 2, 1] = coeffs.beta_xy * e_KH

    # Transversality: tau^{0 nu} = v_i tau^{i nu}, and tau^{00} = v_i v_j tau^{ij}.
    for nu in range(1, 4):
        tau[..., 0, nu] = vx * tau[..., 1, nu] + vy * tau[..., 2, nu]
        tau[..., nu, 0] = tau[..., 0, nu]
    tau[..., 0, 0] = vx * tau[..., 0, 1] + vy * tau[..., 0, 2]

    return tau


def subgrid_energy_from_stress(tau_sgs: torch.Tensor) -> torch.Tensor:
    r"""Extract ``e_KH = 1/2 sum_i tau^{ii}`` from an exact SGS stress tensor.

    This is the measurement that the DNS seeding mode and the whole a-priori
    calibration are built on. The sum runs over x and y only, matching the 2D
    definition used throughout this module.
    """
    return 0.5 * (tau_sgs[..., 1, 1] + tau_sgs[..., 2, 2])


# ---------------------------------------------------------------------------
# Closure object for the solver
# ---------------------------------------------------------------------------

class MInITClosure:
    r"""MInIT as a stateful closure for ``RHDSolver.step(sgs_closure=...)``.

    Unlike the TBNN and exact-DNS closures, MInIT carries state: ``e_KH`` is a
    field evolved alongside the hydrodynamics. The solver's closure protocol is
    stateless — ``f(prims) -> (tau_perp, alpha)``, called once per RK stage and
    again per predictor-corrector recovery — so ``e_KH`` is held frozen within a
    step and advanced once per completed step through :meth:`advance`, which
    ``RHDSolver.evolve`` calls when a closure exposes it. That is operator
    splitting between the turbulence model and the hydrodynamics, which is what
    the paper does too (it uses Strang splitting for the same reason).

    Parameters
    ----------
    dx, dy : coarse grid spacings.
    delta_f : filter scale, i.e. ``coarsening_factor * fine_grid_spacing``. Sets
        ``lambda_KH = 15.8 * delta_f``. Defaults to ``dx`` when not given, which
        is the right choice whenever the coarse cell *is* the filter box.
    coeffs : :class:`MInITCoefficients` — defaults to the published calibration.
    shear_mode : ``"dvxdy"`` or ``"strain"``, see :func:`growth_rate`.
    advect : bool — include the advective term (see :func:`advect_e_kh`).
    e_seed : float — uniform initial ``e_KH`` used when :meth:`initialize` is
        called without a field. Self-contained seeding, no DNS information.
    floor_fraction : float — if positive, ``e_KH`` is held at or above this
        fraction of the local saturation energy after every advance, so the
        field cannot die out in a region and fail to restart. Zero disables it.
    mode : ``"evolve"`` (default) — integrate Eq. (12) as the paper does.
        ``"equilibrium"`` — set ``e_KH`` to its fixed point ``e_sat`` at every
        step instead, dropping the ODE. This is the ``A_gamma -> infinity`` limit
        at fixed ``A_gamma / C``, in which the turbulent energy relaxes to
        saturation faster than the flow evolves. It is worth carrying as a
        variant because the free calibration on this DNS *asks* for it: the
        optimiser pushes ``A_gamma`` an order of magnitude above the Miura &
        Pritchett value, which does nothing except drive the model towards this
        limit. Note what the limit is —
        ``tau^ij = beta_ij (2 A_gamma / C)^2 rho lambda^2 |S|^2`` — an algebraic
        stress proportional to ``rho lambda^2 |S|^2``, which is Smagorinsky with
        MInIT's anisotropy. Only the ratio ``A_gamma / C`` is identifiable here.
    clip_mode, clip_kappa : realizability clipping, off by default.
    backscatter_clip, backscatter_fraction : backscatter control, off by default.
        MInIT's diagonal-dominant stress is not automatically dissipative — with
        an incompressible resolved field the transfer is
        ``e_KH[(beta_xx - beta_yy) S_xx + 2 beta_xy S_xy]``, whose sign varies —
        so the option is meaningful here.
    tau_smooth_passes : int — low-pass applications on tau before use.
    """

    def __init__(
        self,
        dx: float,
        dy: float,
        delta_f: float | None = None,
        coeffs: MInITCoefficients | None = None,
        shear_mode: str = "dvxdy",
        mode: str = "evolve",
        advect: bool = False,
        e_seed: float = 0.0,
        floor_fraction: float = 0.0,
        clip_mode: str | None = None,
        clip_kappa: float = 0.5,
        backscatter_clip: str | None = None,
        backscatter_fraction: float = 0.0,
        tau_smooth_passes: int = 0,
    ):
        self.dx = dx
        self.dy = dy
        self.delta_f = float(delta_f if delta_f is not None else dx)
        self.lambda_KH = LAMBDA_OVER_DELTA * self.delta_f
        self.coeffs = coeffs if coeffs is not None else MInITCoefficients.paper()
        self.shear_mode = shear_mode
        if mode not in ("evolve", "equilibrium"):
            raise ValueError(f"unknown MInIT mode {mode!r}; use 'evolve' or 'equilibrium'")
        self.mode = mode
        self.advect = advect
        self.e_seed = e_seed
        self.floor_fraction = floor_fraction
        self.clip_mode = clip_mode
        self.clip_kappa = clip_kappa
        self.backscatter_clip = backscatter_clip
        self.backscatter_fraction = backscatter_fraction
        self.tau_smooth_passes = tau_smooth_passes

        self.e_KH: torch.Tensor | None = None
        # Diagnostics: (t, mean e_KH, max e_KH) recorded at each advance.
        self.history: list[tuple[float, float, float]] = []

    # ------------------------------------------------------------------
    def initialize(self, prims: torch.Tensor, e_KH: torch.Tensor | None = None) -> None:
        """Set the initial ``e_KH`` field.

        ``e_KH`` is a fixed point at zero, so the seed determines whether the
        model ever activates. Passing a field seeds from a measurement — in
        practice ``subgrid_energy_from_stress`` applied to the exact filtered
        DNS stress at the start time, which mirrors the paper's use of its own
        simulation's e_KH and hands MInIT the same one snapshot of ground truth
        the exact-DNS oracle gets. Omitting it falls back to the uniform
        ``e_seed``, which is fully self-contained. The two are different
        experiments and belong in separate rows of any results table.
        """
        if e_KH is not None:
            self.e_KH = e_KH.to(dtype=prims.dtype, device=prims.device).clamp(min=0.0)
        else:
            self.e_KH = torch.full(
                prims.shape[:-1], float(self.e_seed),
                dtype=prims.dtype, device=prims.device,
            )

        if self.mode == "equilibrium":
            # The fixed point is a function of the state alone, so there is
            # nothing to seed and any passed field would be overwritten on the
            # first advance anyway. Set it now so the first step is consistent.
            gamma = growth_rate(
                prims[..., 1], prims[..., 2], self.dx, self.dy,
                self.coeffs.A_gamma, mode=self.shear_mode,
            )
            self.e_KH = saturation_energy(
                gamma, prims[..., 0], self.lambda_KH, self.coeffs.C,
            )

        self._apply_floor(prims)

    def _apply_floor(self, prims: torch.Tensor) -> None:
        if self.floor_fraction <= 0.0:
            return
        gamma = growth_rate(
            prims[..., 1], prims[..., 2], self.dx, self.dy,
            self.coeffs.A_gamma, mode=self.shear_mode,
        )
        e_sat = saturation_energy(gamma, prims[..., 0], self.lambda_KH, self.coeffs.C)
        self.e_KH = torch.maximum(self.e_KH, self.floor_fraction * e_sat)

    # ------------------------------------------------------------------
    def advance(self, prims: torch.Tensor, dt: float, t: float | None = None) -> None:
        """Advance ``e_KH`` over one completed hydrodynamic step.

        Called by ``RHDSolver.evolve`` after ``step`` returns, so ``prims`` is
        the state at the end of the step and ``dt`` the step actually taken.
        """
        if self.e_KH is None:
            self.initialize(prims)

        rho = prims[..., 0]
        gamma = growth_rate(
            prims[..., 1], prims[..., 2], self.dx, self.dy,
            self.coeffs.A_gamma, mode=self.shear_mode,
        )

        if self.mode == "equilibrium":
            self.e_KH = saturation_energy(gamma, rho, self.lambda_KH, self.coeffs.C)
            self._apply_floor(prims)
            if t is not None:
                self.history.append(
                    (float(t), float(self.e_KH.mean()), float(self.e_KH.max()))
                )
            return

        if self.advect:
            self.e_KH = advect_e_kh(
                self.e_KH, prims[..., 1], prims[..., 2], self.dx, self.dy, dt,
            )

        self.e_KH = advance_e_kh(
            self.e_KH, gamma, rho, dt, self.lambda_KH, self.coeffs.C,
        )
        self._apply_floor(prims)

        if t is not None:
            self.history.append(
                (float(t), float(self.e_KH.mean()), float(self.e_KH.max()))
            )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(self, prims: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(tau_perp, alpha)`` from the frozen ``e_KH`` field.

        Parameters
        ----------
        prims : (Ny, Nx, 4) tensor — [rho, vx, vy, p].

        Returns
        -------
        tau_perp : (Ny, Nx, 4, 4) tensor.
        alpha : (Ny, Nx) tensor — identically one. MInIT carries no transition
            sensor; its own ``e_KH`` is the activation, growing only where there
            is resolved shear to drive it.
        """
        if self.e_KH is None:
            self.initialize(prims)

        tau = assemble_tau(self.e_KH, prims, self.coeffs)

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
