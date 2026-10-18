# The MInIT Closure Model for KHI-Driven MHD Turbulence

A plain-language guide to the physics behind `minit_khi_closure.py`, and how
the code maps onto the equations in Miravet-Tenés et al. (2023), *"Assessment
of a new sub-grid model for magnetohydrodynamical turbulence. II.
Kelvin-Helmholtz instability"* (arXiv:2308.06041).

---

## 1. The problem it solves

Simulating turbulence directly requires resolving every scale down to where
kinetic and magnetic energy dissipate into heat. For astrophysical systems
like binary neutron star mergers, the smallest relevant scales are on the
order of centimeters, while the whole simulated object is tens of
kilometers across. No feasible simulation resolves that whole range at
once — the Kelvin-Helmholtz instability (KHI), the main mechanism that
amplifies magnetic fields during the merger, ends up under-resolved, and
the simulation under-predicts how strong the magnetic field actually gets.

A **sub-grid model** is a shortcut: instead of resolving the small-scale
turbulence directly, you carry a small number of extra variables that
represent *how much* unresolved turbulence there is, and use them to
correct the large-scale equations for the effect that missing turbulence
would have had. **MInIT** ("MHD-Instability-induced-Turbulence") is one
such model. This paper is Part II — it adapts a model originally built for
the magnetorotational instability (MRI) to the KHI instead.

## 2. Splitting the flow into "resolved" and "turbulent"

Any field (velocity, magnetic field) is split into an averaged,
large-scale part and a small-scale fluctuating part:

$$\boldsymbol{A} = \overline{\boldsymbol{A}} + \boldsymbol{A}'$$

The average can be over a volume or a time window. When you plug this
split into the MHD equations and average them, you get the same equations
back for the large-scale fields, plus extra source terms built from
products of the fluctuating parts. Three such products matter here — the
**Maxwell**, **Reynolds**, and **Faraday stress tensors**:

$$M_{ij} = \overline{B_i' B_j'}, \qquad
R_{ij} = \overline{v_i' v_j'}, \qquad
F_{ij} = \overline{v_i' B_j' - v_j' B_i'}$$

These three tensors are exactly what a sub-grid model needs to supply,
since a simulation that doesn't resolve the turbulence has no direct way
to compute them.

## 3. The three ingredients of MInIT

Rather than modeling each stress tensor component separately, MInIT
tracks a *single* scalar — the turbulent kinetic energy density
$e_{\rm KH}$ — and derives all the stresses from it. Three pieces make
this work.

### (a) How fast does turbulence grow? — the KHI growth rate

The KHI is triggered by velocity shear. Miura & Pritchett (1982) showed
that, for a shear of half-width $a_l$ and velocity jump $2v$, the
fastest-growing unstable mode has

$$\gamma_{\rm KH} \approx 0.14\, \frac{v}{a_l}$$

In a running simulation you don't have a clean "$v/a_l$" — you only have
the resolved velocity field — so the code instead evaluates the *local
resolved shear* $\partial_y v_x$ on the coarse grid and uses that as a
stand-in, with a free calibration constant $A_\gamma$ in place of the
fixed 0.14:

$$\gamma_{\rm KH}^{\rm num} \approx A_\gamma\, \partial_y v_x$$

`MInITKHIModel.growth_rate()` implements exactly this.

### (b) How does the turbulent energy evolve in time?

The turbulent kinetic energy density is defined as

$$e_{\rm KH} = \tfrac{1}{2}\rho \sum_i R_{ii}$$

and its evolution is modeled as a competition between two effects:
the instability **injecting** energy from the large-scale shear, and a
Kolmogorov-type turbulent cascade **draining** it away as it cascades
down to the dissipation scale and turns into heat:

$$\partial_t e_{\rm KH} + \partial_i(v_i e_{\rm KH})
= \underbrace{2\gamma_{\rm KH} e_{\rm KH}}_{\text{growth}}
- \underbrace{S_{\rm TD}}_{\text{dissipation}}$$

The dissipation term is modeled (following Landau & Lifshitz's treatment
of the Kolmogorov cascade) as

$$S_{\rm TD} = C\, \frac{e_{\rm KH}^{3/2}}{\sqrt{\rho}\,\lambda_{\rm KH}}$$

where $\lambda_{\rm KH} \approx 15.8\,\Delta_f$ is the wavelength of the
fastest-growing mode, tied to whatever filter/grid scale $\Delta_f$ you're
resolving down to, and $C$ is a second free calibration constant.

Two things make this equation interesting numerically:
- It's **exponentially growing** at first (when $e_{\rm KH}$ is small,
  the $e_{\rm KH}^{3/2}$ sink term is negligible next to the linear
  $2\gamma_{\rm KH}e_{\rm KH}$ growth term), then **saturates** once the
  sink term catches up — this is exactly the curve produced by
  `minit_khi_demo.png` in the earlier response.
- It's **stiff**: the growth and dissipation terms can differ by many
  orders of magnitude in size during the transient, which is why the
  paper uses Strang operator splitting, and why the code uses an
  implicit ODE method (`Radau`) instead of a plain explicit integrator —
  an explicit method would need very small time steps to stay stable.

`MInITKHIModel.evolve()` implements this equation and integrates it in
time given a growth rate, density, and filter scale.

### (c) Turning energy back into stresses — the closure relation

The final assumption of MInIT is that each stress tensor component is
just *proportional* to $e_{\rm KH}$, with time- and position-independent
proportionality coefficients:

$$M_{ij} = \alpha^{\rm KH}_{ij}\, e_{\rm KH}, \qquad
R_{ij} = \frac{\beta^{\rm KH}_{ij}}{\rho}\, e_{\rm KH}, \qquad
F_{ij} = \frac{\gamma^{\rm KH}_{ij}}{\sqrt{\rho}}\, e_{\rm KH}$$

The $1/\rho$ and $1/\sqrt{\rho}$ factors are just there to make the units
work out — $M_{ij}$ has units of magnetic energy density, while $R_{ij}$
and $F_{ij}$ need a density factor to match.

The $\alpha$, $\beta$, $\gamma$ coefficients aren't derived analytically —
they're **measured** from well-resolved box simulations of the KHI (the
paper's Table 2) by directly computing the stress tensors and $e_{\rm KH}$
from simulation data and taking their ratio, averaged over space and over
the late-time, fully-turbulent phase of the simulation. Diagonal
components come out largest (0.6–0.7); off-diagonal components are
small and mostly consistent with zero.

`MInITKHIModel.maxwell_stress()`, `.reynolds_stress()`, and
`.faraday_stress()` implement these three relations directly, pre-loaded
with the paper's calibrated values.

## 4. The two free constants, and their calibrated values

| Constant | Meaning | Calibrated value |
|---|---|---|
| $A_\gamma$ | scales resolved shear → growth rate estimate | $0.14$ (matches theory; results are only weakly sensitive to it) |
| $C$ | strength of Kolmogorov dissipation | $10.6 \pm 1.5$ (fit by minimizing the mismatch between modeled and simulated stresses) |

Both are exposed as fields on `MInITKHIModel` so you can override them —
e.g. to reproduce the paper's Fig. 11 sensitivity sweep over $A_\gamma$
and $C$, or to recalibrate against your own simulation data.

## 5. Code-to-paper map

| Code | Paper |
|---|---|
| `MInITKHIModel.growth_rate()` | Eq. 8, Eq. 23 |
| `MInITKHIModel.wavelength()` | Eq. 9, Eq. 24 |
| `MInITKHIModel.dissipation()` | Eq. 13 |
| `MInITKHIModel.rhs()` / `.evolve()` | Eq. 12 |
| `MInITKHIModel.maxwell_stress()` | Eq. 16a |
| `MInITKHIModel.reynolds_stress()` | Eq. 16b |
| `MInITKHIModel.faraday_stress()` | Eq. 16c |
| `ALPHA_KH`, `BETA_KH`, `GAMMA_KH` | Table 2 |
| `A_GAMMA_DEFAULT`, `C_OPT_DEFAULT` | Sec. 4.3.3, Eq. 26 |

## 6. What the implementation deliberately simplifies

- **No advection term.** The full evolution equation includes
  $\partial_i(v_i e_{\rm KH})$, but the paper itself drops this for its
  box-simulation tests (periodic boundaries make the average of the
  advective term vanish along the flow direction). The code follows suit
  — it treats $e_{\rm KH}$ as a spatially local ODE, not a PDE on a grid.
  A spatially resolved version (coupled to an actual fluid solver) would
  need to reinstate this term and solve it on a mesh.
- **Faraday tensor has no diagonal components.** Physically $F_{ii} = 0$
  by construction (it's antisymmetric), so only off-diagonal
  coefficients are calibrated; the code raises an error if you ask for
  `faraday_stress(..., component="xx")`.
- **A-priori, not a-posteriori.** Like the paper, this is meant for
  testing the closure against known simulation data (or exploring its
  qualitative behavior), not for coupling live into a dynamical MHD
  solver — that "a-posteriori" step is explicitly left to future work in
  the paper itself.

## 7. Suggested next steps

- Feed in real shear-rate and density time series from your own
  simulation data to `evolve()` and compare against directly measured
  stresses (recreating the paper's Fig. 12/13 $L_2$-norm tests).
- Wire `A_gamma` and `C` into a small optimizer (e.g.
  `scipy.optimize.minimize_scalar`) to recalibrate them against your own
  data, mirroring the paper's Eq. 25 procedure.
- Extend `evolve()` to a spatial grid with the advective term restored,
  for a genuine a-posteriori (dynamically coupled) test.
