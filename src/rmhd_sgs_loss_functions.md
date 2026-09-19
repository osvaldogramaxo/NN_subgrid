# Loss Function Design for Integrity-Basis Subgrid-Scale (SGS) Models in Relativistic MHD (RMHD)

Subgrid-scale (SGS) closures in relativistic magnetohydrodynamics (RMHD) using polynomial integrity bases ensure frame covariance (Lorentz and spatial rotation invariance) and parity preservation by parameterizing subgrid tensors as linear combinations of invariant basis tensors. When training such models—whether on high-resolution shock-capturing (HRSC) direct numerical simulations (DNS) or via a posteriori filtering—the design of the loss function is critical. A naive Cartesian $L_2$ regression yields spectral bias, violates physical energy bounds, and introduces anti-dissipative numerical instabilities.

Below is an academic and systematic formulation of loss objectives for integrity-basis RMHD subgrid-scale models.

---

## 1. Problem Formulation and Tensor Representation

Let the filtered energy-momentum tensor be decomposed as:

$$\overline{T^{\mu\nu}} = T^{\mu\nu}(\overline{W}) + \tau^{\mu\nu}$$

where $\overline{W}$ denotes the resolved primitive state vector $(\bar{\rho}, \bar{u}^\mu, \bar{p}, \bar{b}^\mu)$ and $\tau^{\mu\nu}$ is the subgrid stress-energy tensor. Using an integrity basis expansion, $\tau^{\mu\nu}$ is represented as:

$$\tau^{\mu\nu} = \sum_{n=1}^{K} g^{(n)}(\mathcal{I}_1, \dots, \mathcal{I}_M) \, T_{(n)}^{\mu\nu}$$

where:
- $\{T_{(n)}^{\mu\nu}\}$ is the complete, irreducible set of basis tensors constructed from local kinematic and magnetic gradients (e.g., filtered velocity gradient $\nabla_\nu \bar{u}^\mu$, projected 4-magnetic field $\bar{b}^\mu$, 4-acceleration $\bar{a}^\mu$, and fluid-frame strain rate $\sigma^{\mu\nu}$).
- $\{\mathcal{I}_m\}$ is the set of scalar invariants forming the functional basis.
- $\{g^{(n)}\}$ are scalar coefficient functions learned by the neural network.

---

## 2. Component Loss Formulations

### 2.1 Covariant Normalized Tensor Regression Loss ($\mathcal{L}_{\text{tensor}}$)

Standard element-wise mean squared error (MSE) is metric-dependent and heavily dominated by intermittent shock fronts and thin current sheets at the expense of distributed inertial-range turbulent structures. 

The primary structural loss is formulated using the metric-contracted Frobenius norm, normalized by batch DNS intensity:

$$\mathcal{L}_{\text{tensor}} = \frac{\frac{1}{N} \sum_{i=1}^{N} g_{\mu\alpha} g_{\nu\beta} \left(\tau_{\text{pred}}^{\mu\nu} - \tau_{\text{DNS}}^{\mu\nu}\right)_i \left(\tau_{\text{pred}}^{\alpha\beta} - \tau_{\text{DNS}}^{\alpha\beta}\right)_i}{\frac{1}{N} \sum_{i=1}^{N} \lVert \tau_{\text{DNS}, i} \rVert_F^2 + \epsilon}$$

where:
- $g_{\mu\nu}$ is the background or spacetime metric (reducing to the Minkowski metric $\eta_{\mu\nu} = \text{diag}(-1, 1, 1, 1)$ in special RMHD).
- $\lVert A \rVert_F^2 = g_{\mu\alpha} g_{\nu\beta} A^{\mu\nu} A^{\alpha\beta}$.
- $\epsilon > 0$ is a small regularizer (default: $10^{-8}$) preventing division by zero while avoiding point-wise singularity in quasi-laminar zones.

---

### 2.2 Kinetic & Magnetic Dissipation Rate Penalty ($\mathcal{L}_{\mathcal{P}}$)

A central failure mode of offline-trained neural closures is numerical instability (blow-up) during time integration due to uncontrolled spurious anti-dissipation (net negative turbulence production).

The subgrid dissipation rate density $\mathcal{P}_{\text{SGS}}$ is governed by contraction with the resolved fluid gradients:

$$\mathcal{P}_{\text{SGS}} = -\tau^{\mu\nu} \nabla_\mu \bar{u}_\nu$$

While physical local backscatter ($\mathcal{P}_{\text{SGS}} < 0$) occurs in turbulent dynamo regimes, net spatial anti-dissipation destabilizes standard Runge-Kutta or IMEX schemes. The dissipation loss is enforced via:

$$\mathcal{L}_{\mathcal{P}} = \frac{1}{N} \sum_{i=1}^N \left| \mathcal{P}_{\text{pred}, i} - \mathcal{P}_{\text{DNS}, i} \right| + \lambda_{\text{stab}} \operatorname{ReLU}\left( -\int_\Omega \mathcal{P}_{\text{pred}} \, dV \right)$$

The first term matches local transfer rates, while the second acts as a hard global stability barrier against unbounded energy transfer from subgrid scales back into resolved modes.

---

### 2.3 Physical Invariant & Orthogonality Constraints ($\mathcal{L}_{\text{ortho}}$)

In conservative 3+1 formulations of RMHD, the primitive variable recovery step (inversion of conserved variables $\mathbf{U} \to \mathbf{W}$) strictly relies on consistency relations:
- Transversality of the 4-magnetic field: $\bar{u}_\mu \bar{b}^\mu = 0$.
- Fluid-frame stress projection: if $\tau^{\mu\nu}$ is defined purely as spatial stress in the fluid rest frame, the condition $\bar{u}_\mu \tau^{\mu\nu} = 0$ must hold identically.

In a Lorentzian signature $(-, +, +, +)$, the naive contraction $g_{\nu\lambda} v^\nu v^\lambda = -(v^0)^2 + \mathbf{v}^2$ is indefinite and cancels out timelike projection errors ($v^0 = \bar{u}_\mu \tau^{\mu\nu} \bar{u}_\nu$). To ensure a strictly positive-definite penalty across all spacetime directions, we decompose the rest-frame projection into a scalar timelike component and a spatial projector $h_{\nu\lambda} = g_{\nu\lambda} + \bar{u}_\nu \bar{u}_\lambda$:

$$\mathcal{L}_{\text{ortho}} = \frac{1}{N} \sum_{i=1}^N \left[ \left( \bar{u}_\mu \tau_{\text{pred}}^{\mu\nu} \bar{u}_\nu \right)^2 + h_{\nu\lambda} \left( \bar{u}_\mu \tau_{\text{pred}}^{\mu\nu} \right) \left( \bar{u}_\alpha \tau_{\text{pred}}^{\alpha\lambda} \right) \right]$$

Since $h_{\nu\lambda}$ is positive-semidefinite and orthogonal to $\bar{u}^\nu$, $\mathcal{L}_{\text{ortho}} = (v^0)^2 + \mathbf{v}_{\perp}^2 \ge 0$, vanishing if and only if $\bar{u}_\mu \tau_{\text{pred}}^{\mu\nu} = 0$.

---

### 2.4 Multi-Scale Spectral Cascade Loss ($\mathcal{L}_{\text{spectral}}$)

Neural networks optimized solely under point-wise $L_2$ objectives suffer from spectral bias, heavily penalizing high frequencies and behaving as overly dissipative eddy viscosities. To sustain correct inertial-range power-law scalings (such as Goldreich-Sridhar $k_\perp^{-5/3}$ or Iroshnikov-Kraichnan $k^{-3/2}$ cascades):

$$\mathcal{L}_{\text{spectral}} = \int_{k_{\text{min}}}^{k_{\text{max}}} \left| \log_{10} E_{\text{pred}}(k) - \log_{10} E_{\text{DNS}}(k) \right|^2 dk$$

where $E(k)$ is the radially integrated 2D shell energy spectrum density of the subgrid stress tensor components computed via differentiable fast Fourier transforms (FFT) on full spatial snapshots.

---

### 2.5 Realizability and Energy Conditions ($\mathcal{L}_{\text{WEC}}$)

Physical stress-energy tensors must respect the Weak Energy Condition (WEC):

$$T^{\mu\nu} v_\mu v_\nu \ge 0 \quad \forall \text{ future-directed timelike vectors } v^\mu$$

Unconstrained ML closures can predict subgrid corrections that drive the effective local energy density negative, causing catastrophic failures during primitive variable recovery (e.g., negative enthalpy or imaginary sound speed).

Tested against the fluid's own 4-velocity $\bar{u}_\mu$, the effective resolved plus subgrid comoving energy density must remain positive:

$$\varepsilon_{\text{eff}} = \left( T_{\text{resolved}}^{\mu\nu} + \tau_{\text{pred}}^{\mu\nu} \right) \bar{u}_\mu \bar{u}_\nu \ge 0$$

The penalty is enforced as:

$$\mathcal{L}_{\text{WEC}} = \frac{1}{N} \sum_{i=1}^N \operatorname{ReLU}\left( -\varepsilon_{\text{eff}, i} \right)$$

---

## 3. Composite Objective Function

The full a priori optimization objective is constructed as:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{tensor}} + \alpha \mathcal{L}_{\mathcal{P}} + \beta \mathcal{L}_{\text{spectral}} + \gamma \mathcal{L}_{\text{ortho}} + \delta \mathcal{L}_{\text{WEC}}$$

### Summary Matrix of Loss Terms & Default Hyperparameters

| Term | Target Objective | Weight | Default Value | Failure Mode if Omitted |
| :--- | :--- | :--- | :--- | :--- |
| $\mathcal{L}_{\text{tensor}}$ | Covariant projection alignment | Base | $1.0$ | Inaccurate subgrid anisotropy and spatial correlation. |
| $\mathcal{L}_{\mathcal{P}}$ | Subgrid production & net dissipation | $\alpha$ | $0.5$ ($\lambda_{\text{stab}}=10.0$) | Spurious backscatter causing Runge-Kutta numerical explosion. |
| $\mathcal{L}_{\text{spectral}}$ | Maintenance of turbulent inertial cascade | $\beta$ | $0.1$ | Excessive numerical diffusion and premature high-$k$ roll-off. |
| $\mathcal{L}_{\text{ortho}}$ | Adherence to fluid rest-frame constraints | $\gamma$ | $5.0$ | Metric contamination and primitive variable recovery failure. |
| $\mathcal{L}_{\text{WEC}}$ | Weak Energy Condition compliance | $\delta$ | $2.0$ | Unphysical negative pressure or imaginary characteristic speeds. |

---

## 4. PyTorch Implementation Reference (`src/losses.py`)

All loss components are implemented in [`src/losses.py`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/losses.py):

- `CovariantTensorLoss`: Evaluates metric-contracted Frobenius error normalized by local Frobenius norm $\|\tau_{\text{DNS}}\|_F^2 + \epsilon$, with non-negative clamping against floating-point roundoff.
- `DissipationRateLoss`: Evaluates $L_1$ error of subgrid production $\mathcal{P} = -\tau^{\mu\nu} \nabla_\mu \bar{u}_\nu$ and penalizes domain-wide net anti-dissipation via $\lambda_{\text{stab}} \operatorname{ReLU}(-\int \mathcal{P}_{\text{pred}} dV)$.
- `SpectralCascadeLoss`: Performs 2D `torch.fft.rfft2`, computes isotropic radial shell energies $E(k)$ via `scatter_add`, and evaluates mean squared logarithmic difference $\operatorname{MSE}(\log_{10} E_{\text{pred}}, \log_{10} E_{\text{DNS}})$.
- `OrthogonalityLoss`: Enforces $\bar{u}_\mu \tau^{\mu\nu} = 0$ via the positive-definite fluid rest-frame spatial projector $h_{\nu\lambda} = g_{\nu\lambda} + \bar{u}_\nu \bar{u}_\lambda$.
- `WECLoss`: Evaluates comoving energy density $\varepsilon_{\text{eff}} = (T_{\text{resolved}} + \tau_{\text{pred}})^{\mu\nu} \bar{u}_\mu \bar{u}_\nu$ and penalizes violations via $\operatorname{ReLU}(-\varepsilon_{\text{eff}})$.
- `RMHDSGSLoss`: Master composite class encapsulating minibatch loss terms and snapshot-level spectral evaluation.

---

## 5. A Posteriori Rollout / Adjoint Solver Loss

If offline a priori training exhibits significant distribution shift when coupled to the PDE solver, transition to an unrolled differentiable solver or adjoint-state framework over $m$ simulation steps:

$$\mathcal{L}_{\text{rollout}} = \sum_{t=1}^{m} \left\lVert \mathbf{U}_{\text{sim}}^t [\tau_{\text{pred}}] - \mathbf{U}_{\text{DNS}}^t \right\rVert_{\mathbf{M}}^2$$

where $\mathbf{U} = (\sqrt{\gamma} D, \sqrt{\gamma} S_j, \sqrt{\gamma} B^j, \sqrt{\gamma} \tau)$ represents the conserved variables in the 3+1 ADM decomposition of RMHD, and $\mathbf{M}$ is a diagonal normalization matrix scaling each conserved field by its domain-wide variance.

