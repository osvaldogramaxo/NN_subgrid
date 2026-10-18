"""
minit_khi_closure.py
=====================

Implementation of the MInIT (MHD-Instability-induced-Turbulence) sub-grid
closure model for Kelvin-Helmholtz-instability (KHI) driven turbulence, as
described in Miravet-Tenés, Cerdá-Durán, Obergaulinger & Font (2023),
"Assessment of a new sub-grid model for magnetohydrodynamical turbulence.
II. Kelvin-Helmholtz instability", MNRAS.

The model has three ingredients:

1. An estimate of the KHI growth rate gamma_KH in terms of resolved
   (large-scale) quantities (Sec. 3.1, Eq. 8).
2. An evolution equation for the sub-grid turbulent kinetic energy density
   e_KH, sourced by the instability and drained by a Kolmogorov-type
   dissipation term (Sec. 3.2, Eq. 12-13).
3. A closure relation that expresses the turbulent Maxwell (M_ij),
   Reynolds (R_ij) and Faraday (F_ij) stress tensors as proportional to
   e_KH (Sec. 3.3, Eq. 16), with coefficients calibrated against box
   simulations of the KHI (Table 2).

Units follow the paper's simulation setup (code units), where rho, v, B
are all O(1) quantities in a periodic box of size L = 1.

Author: (generated for user, based on arXiv:2308.06041v2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Calibrated coefficients (Table 2 of the paper)
# ---------------------------------------------------------------------------

# alpha^KH_ij : Maxwell stress closure coefficients  (M_ij = alpha_ij * e_KH)
ALPHA_KH = {
    "xx": 0.70, "yy": 0.64, "zz": 0.64,
    "xy": -0.01, "xz": -0.02, "yz": -0.026,
}

# beta^KH_ij : Reynolds stress closure coefficients  (R_ij = beta_ij/rho * e_KH)
BETA_KH = {
    "xx": 0.74, "yy": 0.64, "zz": 0.61,
    "xy": -0.01, "xz": -0.041, "yz": -0.02,
}

# gamma^KH_ij : Faraday stress closure coefficients  (F_ij = gamma_ij/sqrt(rho) * e_KH)
# Only off-diagonal components are defined/non-zero for the Faraday tensor.
GAMMA_KH = {
    "xy": -0.010, "xz": -0.01, "yz": -0.008,
}

# Optimal free parameters of the model (Sec. 4.3.3)
A_GAMMA_DEFAULT = 0.14     # calibration constant in the growth-rate estimate (Eq. 23)
C_OPT_DEFAULT = 10.6       # Kolmogorov dissipation constant (Eq. 13, Eq. 26)


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

@dataclass
class MInITKHIModel:
    """
    MInIT closure model for KHI-induced MHD turbulence.

    Parameters
    ----------
    C : float
        Dimensionless Kolmogorov-dissipation constant in Eq. (13).
        Calibrated value: C = 10.6 +/- 1.5 (Eq. 26).
    A_gamma : float
        Calibration constant relating the resolved shear to the KHI growth
        rate (Eq. 23). Theoretical value from Miura & Pritchett (1982) is
        0.14; the paper finds the results are only weakly sensitive to it.
    alpha : dict
        Maxwell-stress closure coefficients alpha^KH_ij (Eq. 16a).
    beta : dict
        Reynolds-stress closure coefficients beta^KH_ij (Eq. 16b).
    gamma : dict
        Faraday-stress closure coefficients gamma^KH_ij (Eq. 16c).
    """

    C: float = C_OPT_DEFAULT
    A_gamma: float = A_GAMMA_DEFAULT
    alpha: dict = field(default_factory=lambda: dict(ALPHA_KH))
    beta: dict = field(default_factory=lambda: dict(BETA_KH))
    gamma: dict = field(default_factory=lambda: dict(GAMMA_KH))

    # ------------------------------------------------------------------
    # 1. KHI growth rate (Sec. 3.1 / Sec. 4.3.1, Eqs. 8 and 23)
    # ------------------------------------------------------------------
    def growth_rate(self, dvx_dy: float) -> float:
        """
        Estimate of the KHI growth rate gamma_KH from a resolved shear.

        In the analytic (initial-data) limit this is gamma_KH = A_gamma * v/a_l
        (Eq. 8, with the theoretical value A_gamma = 0.14). In a running
        simulation v/a_l is replaced by the numerically-evaluated resolved
        shear d(vx)/dy (Eq. 23), since that is what is actually available
        on a coarse/sub-grid mesh.

        Parameters
        ----------
        dvx_dy : float
            Resolved shear rate ∂_y v_x (centered finite difference across
            the shear layer on the coarse grid), in code units [1/time].

        Returns
        -------
        gamma_KH : float
            KHI growth rate, same units as dvx_dy.
        """
        return self.A_gamma * dvx_dy

    @staticmethod
    def wavelength(delta_f: float) -> float:
        """
        Wavelength of the fastest growing KHI mode (Eq. 9), assimilating the
        shear scale a_l to the filter/box size Delta_f (Eq. 24):

            lambda_KH ≈ 15.8 * Delta_f
        """
        return 15.8 * delta_f

    # ------------------------------------------------------------------
    # 2. Kolmogorov dissipation term (Eq. 13)
    # ------------------------------------------------------------------
    def dissipation(self, e_KH: float, rho: float, lambda_KH: float) -> float:
        """
        Turbulent-energy dissipation term S_TD (Eq. 13):

            S_TD = C * e_KH^(3/2) / (sqrt(rho) * lambda_KH)

        e_KH is clipped at 0 to avoid nan/negative fractional powers should
        the ODE integrator ever produce a tiny negative overshoot.
        """
        e = max(e_KH, 0.0)
        return self.C * e ** 1.5 / (np.sqrt(rho) * lambda_KH)

    # ------------------------------------------------------------------
    # 3. Evolution equation for e_KH (Eq. 12, without the advective term,
    #    consistent with the box-simulation tests in the paper - Sec 4.3.1)
    # ------------------------------------------------------------------
    def rhs(self, e_KH: float, rho: float, gamma_KH: float, lambda_KH: float) -> float:
        """
        Right-hand side of d(e_KH)/dt = 2*gamma_KH*e_KH - S_TD.
        """
        return 2.0 * gamma_KH * e_KH - self.dissipation(e_KH, rho, lambda_KH)

    def evolve(
        self,
        e_KH0: float,
        t_span: tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
        gamma_KH: float | Callable[[float], float] = 0.14,
        rho: float | Callable[[float], float] = 1.5,
        delta_f: float = 1.0,
        method: str = "Radau",
        **solve_ivp_kwargs,
    ):
        """
        Integrate the turbulent-energy evolution equation (Eq. 12) in time.

        Because the source term grows exponentially while the sink term
        grows as e_KH^1.5, the ODE is generally stiff (as noted in the
        paper, Sec. 4.3.1); an implicit method (default: 'Radau') is used
        instead of the paper's Strang-splitting scheme, but is numerically
        equivalent for this purpose.

        Parameters
        ----------
        e_KH0 : float
            Initial turbulent energy density at t_span[0].
        t_span : (t0, tf)
            Integration interval.
        t_eval : array, optional
            Times at which to store the solution.
        gamma_KH : float or callable(t) -> float
            KHI growth rate, constant or time-dependent.
        rho : float or callable(t) -> float
            Mass density, constant or time-dependent.
        delta_f : float
            Filter/box size used to set lambda_KH = 15.8 * delta_f (Eq. 24).
        method : str
            Integrator passed to scipy.integrate.solve_ivp. 'Radau' or
            'BDF' are recommended for stiffness.

        Returns
        -------
        sol : scipy.integrate.OdeResult
            Standard solve_ivp result; sol.t and sol.y[0] give the time
            series of e_KH(t).
        """
        lambda_KH = self.wavelength(delta_f)

        gamma_fn = gamma_KH if callable(gamma_KH) else (lambda t: gamma_KH)
        rho_fn = rho if callable(rho) else (lambda t: rho)

        def ode(t, y):
            e = y[0]
            return [self.rhs(e, rho_fn(t), gamma_fn(t), lambda_KH)]

        sol = solve_ivp(
            ode, t_span, [e_KH0], t_eval=t_eval, method=method,
            **solve_ivp_kwargs,
        )
        return sol

    # ------------------------------------------------------------------
    # 4. Closure relations: stresses from e_KH (Eq. 16)
    # ------------------------------------------------------------------
    def maxwell_stress(self, e_KH: float, component: str = "xx") -> float:
        """M_ij(t, r) = alpha^KH_ij * e_KH(t, r)   (Eq. 16a)."""
        return self.alpha[component] * e_KH

    def reynolds_stress(self, e_KH: float, rho: float, component: str = "xx") -> float:
        """R_ij(t, r) = (1/rho) * beta^KH_ij * e_KH(t, r)   (Eq. 16b)."""
        return self.beta[component] * e_KH / rho

    def faraday_stress(self, e_KH: float, rho: float, component: str = "xy") -> float:
        """F_ij(t, r) = gamma^KH_ij / sqrt(rho) * e_KH(t, r)   (Eq. 16c)."""
        if component not in self.gamma:
            raise KeyError(
                f"Faraday coefficient not calibrated for component '{component}'. "
                f"Available: {list(self.gamma)}"
            )
        return self.gamma[component] * e_KH / np.sqrt(rho)

    def all_stresses(self, e_KH: float, rho: float) -> dict:
        """
        Convenience method: compute the full set of stress-tensor
        components for a given e_KH and rho.

        Returns a dict with keys 'M', 'R', 'F', each mapping component
        labels ('xx', 'xy', ...) to values.
        """
        M = {c: self.maxwell_stress(e_KH, c) for c in self.alpha}
        R = {c: self.reynolds_stress(e_KH, rho, c) for c in self.beta}
        F = {c: self.faraday_stress(e_KH, rho, c) for c in self.gamma}
        return {"M": M, "R": R, "F": F}


# ---------------------------------------------------------------------------
# Demo / sanity check reproducing the qualitative behaviour of Fig. 11
# ---------------------------------------------------------------------------

def _demo():
    """
    Reproduce the qualitative shape of Fig. 11: exponential growth of e_KH
    from a small seed value, followed by saturation/decay once the
    Kolmogorov sink term balances the KHI source term, for the
    paper's optimal C = 10.6, A_gamma = 0.14.
    """
    import matplotlib.pyplot as plt

    model = MInITKHIModel(C=10.6, A_gamma=0.14)

    # Representative values inspired by the KH-H1 simulation setup:
    # rho ~ 1.5-2.0 near the shear layer, an initial shear-driven growth
    # rate consistent with gamma_KH ~ O(0.1-1), and box filter size
    # Delta_f corresponding to the wavelength of the fastest growing mode.
    rho0 = 1.5
    a_l = 0.01
    v0 = 0.5
    dvx_dy0 = v0 / a_l          # initial resolved shear, per Eq. 8/23
    gamma0 = model.growth_rate(dvx_dy0)

    delta_f = a_l               # filter size ~ shear width at the start
    e_KH0 = 1e-6                # small seed turbulent energy density

    t_span = (0.0, 40.0)
    t_eval = np.linspace(*t_span, 400)

    sol = model.evolve(
        e_KH0=e_KH0,
        t_span=t_span,
        t_eval=t_eval,
        gamma_KH=gamma0,
        rho=rho0,
        delta_f=delta_f,
    )

    e_KH_t = sol.y[0]

    # Evaluate stress tensors along the trajectory
    stresses_final = model.all_stresses(e_KH_t[-1], rho0)

    print("Growth rate gamma_KH  :", gamma0)
    print("lambda_KH             :", model.wavelength(delta_f))
    print("Final e_KH            :", e_KH_t[-1])
    print("Final stresses        :", stresses_final)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(sol.t, e_KH_t)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$e_{\rm KH}$")
    ax.set_title("MInIT KHI turbulent energy density evolution")
    fig.tight_layout()
    fig.savefig("minit_khi_demo.png", dpi=150)
    print("Saved plot to minit_khi_demo.png")


if __name__ == "__main__":
    _demo()
