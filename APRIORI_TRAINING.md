# *A Priori* Training Process in CRMHD-TBNN

This document provides a comprehensive, step-by-step description of the ***a priori* supervised training pipeline** implemented in [`src/train_apriori.py`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/train_apriori.py). It covers data generation, tensor algebra, neural closure modeling, robust adaptive loss formulation, training dynamics, diagnostics, and deployment export.

---

## Table of Contents
1. [Overview & Physical Motivation](#1-overview--physical-motivation)
2. [High-Level Architecture & Workflow](#2-high-level-architecture--workflow)
3. [Stage 1: Initialization & Environment Setup](#3-stage-1-initialization--environment-setup)
4. [Stage 2: Data Pipeline & Preprocessing (`TBNNDataset`)](#4-stage-2-data-pipeline--preprocessing-tbnndataset)
   - [Favre Filtering & Subgrid Stress Calculation](#favre-filtering--subgrid-stress-calculation)
   - [Covariant Feature Extraction on the Coarse Grid](#covariant-feature-extraction-on-the-coarse-grid)
   - [Standardization & Dataset Caching](#standardization--dataset-caching)
5. [Stage 3: Neural Closure Model (`CRMHD_TBNN`)](#5-stage-3-neural-closure-model-crmhd_tbnn)
   - [Lorentz Equivariance by Construction](#lorentz-equivariance-by-construction)
   - [Network Forward Architecture](#network-forward-architecture)
6. [Stage 4: Physics-Informed Composite Loss (`RMHDSGSLoss`)](#6-stage-4-physics-informed-composite-loss-rmhdsgsloss)
   - [Covariant Tensor Loss](#covariant-tensor-loss)
   - [Kinetic & Magnetic Dissipation Rate Penalty](#kinetic--magnetic-dissipation-rate-penalty)
   - [Multi-Scale Spectral Cascade Loss](#multi-scale-spectral-cascade-loss)
   - [Physical Invariant & Orthogonality Constraints](#physical-invariant--orthogonality-constraints)
   - [Realizability & Weak Energy Condition](#realizability--weak-energy-condition)
   - [Composite Objective & Default Weights](#composite-objective--default-weights)
   - [Two-Phase Spectral Integration Dynamics](#two-phase-spectral-integration-dynamics)
7. [Stage 5: Optimization & Early Stopping](#7-stage-5-optimization--early-stopping)
   - [AdamW, plateau schedule, adaptive clipping](#adamw-plateau-schedule-adaptive-clipping)
   - [Early Stopping Monitor](#early-stopping-monitor)
8. [Stage 6: Epoch Execution & Checkpointing](#8-stage-6-epoch-execution--checkpointing)
   - [Training & Validation Loops](#training--validation-loops)
   - [Best Model Restoration](#best-model-restoration)
9. [Stage 7: Diagnostics, Logging & Visualization](#9-stage-7-diagnostics-logging--visualization)
   - [Multi-Panel Convergence Curves](#multi-panel-convergence-curves)
   - [2D Spatial Stress & Parity Maps](#2d-spatial-stress--parity-maps)
   - [Metrics CSV & JSON History](#metrics-csv--json-history)
10. [Stage 8: TorchScript JIT Export for Solver Deployment](#10-stage-8-torchscript-jit-export-for-solver-deployment)
11. [Command-Line Interface (CLI) Reference](#11-command-line-interface-cli-reference)

---

## 1. Overview & Physical Motivation

In Large Eddy Simulations (LES) of compressible relativistic hydrodynamics (RHD), spatial filtering of the non-linear conservation equations yields unresolved terms known as the **Subgrid-Scale (SGS) stress tensor** $\tau_{\text{SGS}}^{\mu\nu}$. 

$$\nabla_\mu \overline{T^{\mu\nu}} = 0 \iff \nabla_\mu \widetilde{T}^{\mu\nu} = -\nabla_\mu \tau_{\text{SGS}}^{\mu\nu}$$

where:
- $\overline{T^{\mu\nu}}$ is the spatially filtered energy-momentum tensor.
- $\widetilde{T}^{\mu\nu} = T^{\mu\nu}(\widetilde{u}, \overline{\rho}, \overline{p})$ is the energy-momentum tensor evaluated from the Favre-filtered coarse fields.
- $\tau_{\text{SGS}}^{\mu\nu} = \overline{T^{\mu\nu}} - \widetilde{T}^{\mu\nu}$ is the residual subgrid stress.

***A priori* training** performs supervised offline optimization where the neural network closure is trained to reproduce the exact subgrid stress tensor $\tau_{\text{SGS}}^{\mu\nu}$ extracted from high-resolution Direct Numerical Simulation (DNS) snapshots.

Unlike unconstrained neural networks (e.g., standard CNNs or MLPs), [`src/train_apriori.py`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/train_apriori.py) employs a **Tensor Basis Neural Network (TBNN)**. The network predicts scalar coefficients of an integrity basis of isotropic symmetric 4-tensors, guaranteeing **manifest Lorentz covariance and frame-invariance by construction**.

---

## 2. High-Level Architecture & Workflow

The entire lifecycle inside [`src/train_apriori.py`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/train_apriori.py) executes according to the following flowchart:

```
[DNS HDF5 Snapshots (dns_khi, 81 snapshots, t = 0..8)]
         │
         ▼
[Relativistic Favre Filtering (r=16)] ─────► Compute Exact SGS Stress: τ_SGS = T_bar - T_coarse
         │
         ▼
[4th-Order Finite Differences] ────────────► Kinematics: σ^μν (shear), ω^μν (vorticity), θ (expansion)
                                             Euler Eq: a^μ (acceleration, avoiding ∂_t)
         │
         ▼
[Covariant Tensor Basis Algebra] ──────────► 15 Invariants Λ_k  +  2 Thermo (θ, M_t)
                                             12 Basis Tensors E^(n)μν
         │
         ▼
[Dataset Caching & Normalization] ─────────► dataset_cache_r{r}_s{stride}.pt (v4)
                                             ONE cache and ONE set of (μ, σ) for all splits
                                             Cached: u^μ, ∇_μ u_ν, ρ, p, snapshot bounds, times
         │
         ▼
[Interleaved Snapshot Split] ──────────────► train / val: every Nth snapshot of t ∈ [1.0, 6.5]
                                             test: contiguous tail, never used for selection
         │
         ▼
[Snapshot-Granular Batching]  ─────────────► groups of whole 64×64 coarse fields
         │                                   (NOT shuffled points — the spectral term
         ▼                                    needs the periodic 2D grid intact)
┌────────────────────────────────────────────────────────────────────────┐
│                        EPOCH TRAINING LOOP                             │
│                                                                        │
│   Normalized Inputs [Λ_k, θ, M_t] ──► CRMHD_TBNN (MLP Backbone)       │
│                                              │                         │
│                                              ├──► Coeffs G^(n) (12)    │
│                                              └──► Pressure P_SGS (1)   │
│                                              │                         │
│   Tensor Contraction: ΔT_SGS^μν = Σ G^(n) E^(n)μν                      │
│                                              │                         │
│   Point-wise terms on the group's points:    ▼                         │
│   L_pt = L_tensor + α L_P + δ L_WEC   (+ γ L_ortho only if γ > 0)      │
│                                              │                         │
│   Spectral term on the SAME snapshots,       ▼                         │
│   in the SAME graph (exact periodic FFT):                              │
│   L_spectral = MSE(log10 E_pred(k), log10 E_DNS(k))                    │
│                                              │                         │
│   Composite: L_total = L_pt + β L_spectral   ▼                         │
│   ONE backward() ──► adaptive grad clip ──► ONE optimizer step         │
└────────────────────────────────────────────────────────────────────────┘
         │
         ├──► Per-epoch Validation (same objective, deterministic order)
         ├──► EMA-smooth the monitored metric (everything below reads it)
         ├──► Learning Rate Update (plateau on the smoothed metric, or cosine w/ warmup)
         ├──► EarlyStopping Check (patience = 2x lr_patience)
         ├──► Best Checkpoint Save (best_model.pt)
         ├──► Periodic Diagnostics (loss_history.png, pred_vs_truth.png)
         ▼
[Restore Best Model Weights]
         │
         ▼
[TorchScript JIT Tracing] ─────────────────► tbnn_traced.pt (Ready for RHD Solver)
```

---

## 3. Stage 1: Initialization & Environment Setup

When [`train()`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/train_apriori.py#L724-L1075) is invoked:

1. **Hardware & Precision Selection**:
   - Device detection checks for CUDA availability. If CUDA initialization fails or is unavailable, it falls back seamlessly to CPU.
   - Precision defaults to `torch.float64` (double precision), essential in relativistic hydrodynamics where small velocity differences near the speed of light ($v \to 1$) cause extreme Lorentz factor variations ($W = (1 - v^2)^{-1/2}$).
2. **Directory & Cache Configuration**:
   - Prepares `--checkpoint-dir` (default: `checkpoints/`) and subfolder `checkpoints/plots/`.
   - Derives cache filenames deterministically based on coarsening factor $r$ and sampling stride $s$:
     - Training cache: `checkpoints/train_cache_r{r}_s{stride}.pt`
     - Validation cache: `checkpoints/val_cache_r{r}_s{stride}.pt`

---

## 4. Stage 2: Data Pipeline & Preprocessing (`TBNNDataset`)

The dataset class [`TBNNDataset`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/train_apriori.py#L50-L164) manages snapshot loading, Favre filtering, feature extraction, and normalization.

### Favre Filtering & Subgrid Stress Calculation
For each simulation snapshot loaded from HDF5 via [`load_all_snapshots()`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/dataset.py):
1. **Fine-Grid Stress**: The full energy-momentum tensor is evaluated on the fine $1024 \times 1024$ grid:
   $$T^{\mu\nu} = (\rho + \rho\epsilon + p) u^\mu u^\nu + p\,\eta^{\mu\nu} = \rho h\, u^\mu u^\nu + p\,\eta^{\mu\nu}$$
2. **Box Spatial Filtering**: $T^{\mu\nu}$ is spatially averaged over $r \times r$ cell blocks to produce $\overline{T^{\mu\nu}}$ on the coarse $N_c \times N_c$ grid ($N_c = 1024 / r$).
3. **Relativistic Favre Filtering**: Coarse primitive fields are computed:
   - Relativistic density: $D = \rho W$
   - Filtered density: $\overline{\rho} = \overline{D} / \widetilde{W}$
   - Favre velocity: $\widetilde{v}^i = \overline{D v^i} / \overline{D}$
   - Coarse Lorentz factor: $\widetilde{W} = (1 - \widetilde{v}_x^2 - \widetilde{v}_y^2)^{-1/2}$
   - Filtered pressure: $\overline{p}$
4. **Exact Subgrid Stress Residual & Fluid Rest-Frame Projection**:
   $$\tau_{\text{SGS}}^{\mu\nu} = \overline{T^{\mu\nu}} - T^{\mu\nu}(\overline{\rho}, \widetilde{v}_x, \widetilde{v}_y, \overline{p})$$
   Because the TBNN integrity basis $E^{(n)\mu\nu}$ is constructed strictly in the fluid rest frame ($u_\mu E^{(n)\mu\nu} \equiv 0$), the raw residual is projected into the fluid frame:
   $$\tau_{\perp}^{\mu\nu} = h^\mu_\alpha h^\nu_\beta \tau_{\text{SGS}}^{\alpha\beta}$$
   where $h^\mu_\alpha = \delta^\mu_\alpha + \widetilde{u}^\mu \widetilde{u}_\alpha$ is the comoving spatial projector. This isolates the spatial subgrid stress tensor, ensuring exact mathematical consistency with the tensor basis and eliminating spurious non-orthogonal subgrid energy density and heat flux components.

### Covariant Feature Extraction on the Coarse Grid
Feature extraction is performed via [`extract_tbnn_features()`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/features.py#L214-L305):
1. **Spatial Gradients**: 4th-order central finite differences compute spatial derivatives with periodic boundary conditions:
   $$\partial_x f_{i, j} = \frac{-f_{i, j+2} + 8 f_{i, j+1} - 8 f_{i, j-1} + f_{i, j-2}}{12 \Delta x}$$
2. **Four-Acceleration from Euler Equation**: To compute $a^\mu = u^\nu \nabla_\nu u^\mu$ on single static snapshots without noisy or unavailable time derivatives $\partial_t u^\mu$, the code utilizes the relativistic Euler equation:
   $$a_\mu = -\frac{1}{\rho h} \left( \partial_\mu p + u_\mu u^\nu \partial_\nu p \right)$$
3. **Kinematic Decomposition**:
   - Projection tensor orthogonal to four-velocity: $P^{\mu\nu} = \eta^{\mu\nu} + u^\mu u^\nu$
   - Shear tensor: $\sigma^{\mu\nu} = P^{\mu\alpha} P^{\nu\beta} \nabla_{(\alpha} u_{\beta)} - \frac{1}{3} \theta P^{\mu\nu}$
   - Vorticity tensor: $\omega^{\mu\nu} = P^{\mu\alpha} P^{\nu\beta} \nabla_{[\alpha} u_{\beta]}$
   - Expansion scalar: $\theta = \nabla_\mu u^\mu$
4. **15 Lorentz Invariants $\Lambda_1 \dots \Lambda_{15}$**: Scalar traces formed from contractions of $\sigma$, $\omega$, and $a$ (e.g., $\text{Tr}(\sigma^2)$, $\text{Tr}(\omega^2)$, $a_\mu a^\mu$, $\sigma^{\mu\nu} a_\mu a_\nu$, etc.).
5. **2 Thermodynamic Inputs**: Expansion scalar $\theta$ and turbulent Mach number $M_t = \sqrt{v_x^2 + v_y^2} / c_s$, where $c_s = \sqrt{\Gamma p / (\rho h)}$.
6. **12 Isotropic Basis Tensors $E^{(n)\mu\nu}$**: Complete relativistic integrity basis of symmetric rank-2 tensors (spanning terms like $P^{\mu\nu}$, $\sigma^{\mu\nu}$, $\sigma^2$, $\omega^2$, $a^\mu a^\nu$, $(\sigma a)^{(\mu} a^{\nu)}$, etc.).

### Standardization & Dataset Caching
All spatial points across snapshots are flattened into a 1D sample dimension:
- Invariants: $(N_{\text{samples}}, 15)$
- Thermodynamic features: $(N_{\text{samples}}, 2)$
- Basis tensors: $(N_{\text{samples}}, 12, 4, 4)$
- Exact targets $\tau_{\text{SGS}}$: $(N_{\text{samples}}, 4, 4)$

Feature statistics are calculated across the training set:
$$\mu_{\text{inv}} = \mathbb{E}[\Lambda], \quad \sigma_{\text{inv}} = \text{std}[\Lambda], \quad \mu_{\text{thermo}} = \mathbb{E}[(\theta, M_t)], \quad \sigma_{\text{thermo}} = \text{std}[(\theta, M_t)]$$
Characteristic stress magnitude:
$$\tau_{\text{scale}} = \frac{1}{N} \sum_{i=1}^N \|\tau_{\text{SGS}, i}\|_F$$
The processed arrays and statistics are serialized into PyTorch `.pt` cache files, reducing startup time on subsequent runs from minutes to under a second.

---

## 5. Stage 3: Neural Closure Model (`CRMHD_TBNN`)

The model [`CRMHD_TBNN`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/utils.py#L251-L394) is an invariant-input, covariant-output architecture.

```
Inputs: 15 Normalized Invariants ⊕ 2 Normalized Thermo Features
                           │
                           ▼
               ┌───────────────────────┐
               │  Linear(17, H)        │
               │  LeakyReLU            │
               │  ×(L-1) Hidden Layers │
               │  Linear(H, H)         │
               │  LeakyReLU            │
               └───────────┬───────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
   coeff_head: Linear(H, 12)     p_sgs_head: Linear(H, 1)
            │                             │
            ▼                             ▼
   Coefficients G^(n)              Isotropic Pressure P_SGS
            │
            ▼
   Contraction with Basis Tensors:
   ΔT_SGS^μν = Σ_{n=0}^{11} G^(n) E^(n)μν
```

### Lorentz Equivariance by Construction
Standard neural networks operating on spatial grids violate Lorentz boost and rotational symmetries. In CRMHD-TBNN:
- Inputs to the MLP are strictly scalar Lorentz invariants $\Lambda_k$ and scalar thermodynamic quantities $\theta, M_t$.
- Output coefficients $G^{(n)}$ are scalar functions of invariants.
- The predicted tensor $\Delta T_{\text{SGS}}^{\mu\nu} = \sum_{n=0}^{11} G^{(n)} E^{(n)\mu\nu}$ is a linear combination of covariant basis tensors $E^{(n)\mu\nu}$.
Therefore, when the input state transforms under a Lorentz transformation $\Lambda^\mu{}_\alpha$, the predicted subgrid stress transforms identically:
$$\Delta T_{\text{SGS}}'^{\mu\nu} = \Lambda^\mu{}_\alpha \Lambda^\nu{}_\beta \Delta T_{\text{SGS}}^{\alpha\beta}$$

---

## 6. Stage 4: Physics-Informed Composite Loss (`RMHDSGSLoss`)

Standard element-wise mean squared error ($L_2$) is metric-dependent, heavily dominated by localized shock fronts at the expense of distributed inertial-range turbulent structures, and agnostic to relativistic thermodynamics. Unconstrained closures frequently cause numerical blow-ups during simulation rollouts due to spurious anti-dissipation (negative turbulence production), frame-invariance leakage, or violations of energy conditions.

[`RMHDSGSLoss`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/losses.py) implements the multi-objective physics-informed loss framework formulated in [`src/rmhd_sgs_loss_functions.md`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/rmhd_sgs_loss_functions.md).

### Covariant Tensor Loss ($\mathcal{L}_{\text{tensor}}$)
The foundational structural regression loss contracts tensor prediction errors using the spacetime metric $g_{\mu\nu} = \text{diag}(-1, 1, 1, 1)$, normalized **per point** by the local subgrid intensity:

$$\mathcal{L}_{\text{tensor}} = \frac{1}{N} \sum_{i=1}^{N} \frac{g_{\mu\alpha} g_{\nu\beta} \left(\tau_{\text{pred}}^{\mu\nu} - \tau_{\text{DNS}}^{\mu\nu}\right)_i \left(\tau_{\text{pred}}^{\alpha\beta} - \tau_{\text{DNS}}^{\alpha\beta}\right)_i}{\lVert \tau_{\text{DNS}, i} \rVert_F^2 + \tau_{\text{floor}}^2}$$

where $\lVert A \rVert_F^2 = g_{\mu\alpha} g_{\nu\beta} A^{\mu\nu} A^{\alpha\beta}$. Because fluid rest-frame tensors satisfy $\bar{u}_\mu \tau^{\mu\nu} = 0$, the metric contraction is non-negative and is clamped at $\ge 0$ against numerical roundoff.

**Why per point, and why a floor.** Per-point normalization is what distinguishes this loss from plain MSE: each cell contributes an $O(1)$ relative error regardless of its subgrid intensity. That matters because the intensity distribution is wide — on `dns_khi` the per-cell $\lVert \tau \rVert_F$ spans a mean of $7.3 \times 10^{-3}$ to a maximum of $4.7 \times 10^{-1}$.

An earlier version normalized by the *batch* mean instead. That reduces the loss to MSE times a constant, so the gradient is monopolized by the few strongest cells and the low-intensity majority of the domain receives almost none. It also made train and validation values incomparable, because each split normalized itself: the two differed by 2.2× in $\tau$ scale, which alone produced an apparent train/val gap (train $\mathcal{L}_{\text{tensor}} = 0.199$ against val $0.020$, with no overfitting involved).

The cost of per-point normalization is the opposite bias — quasi-laminar cells, where $\lVert \tau_{\text{DNS}} \rVert_F \to 0$, would be amplified without bound. $\tau_{\text{floor}}$ is therefore set to a **quantile** of the training $\lVert \tau \rVert_F$ distribution (`--tensor-floor-quantile`, default the median), not a fraction of its mean: the distribution is heavy-tailed, with mean/median $\approx 5.2$ over the `dns_khi` training window, so a fraction of the mean is not portable between datasets. Measured share of the loss carried by the weakest and strongest deciles in the uniform-absolute-error regime (which is the regime at initialization):

| Floor | Weakest 10% of cells | Strongest 10% |
| :--- | :--- | :--- |
| $0.05 \times$ mean | 43.0% | 0.01% |
| median ($\approx 0.2 \times$ mean) | 20.0% | 0.05% |
| $0.5 \times$ mean | 14.5% | 0.24% |
| batch-level (earlier behaviour) | $\approx 0$% | dominant |

The run prints its own measured decile shares at startup, so this is observable rather than assumed.

### Kinetic & Magnetic Dissipation Rate Penalty ($\mathcal{L}_{\mathcal{P}}$)
Subgrid dissipation rate density measures energy transfer between resolved and unresolved scales:

$$\mathcal{P}_{\text{SGS}} = -\tau^{\mu\nu} \nabla_\mu \bar{u}_\nu$$

Net anti-dissipation ($\int_\Omega \mathcal{P}_{\text{pred}} dV < 0$) pumps spurious energy back into resolved modes, rapidly destabilizing standard Runge-Kutta time integration. The dissipation loss enforces:

$$\mathcal{L}_{\mathcal{P}} = \frac{1}{\mathcal{P}_{\text{scale}}} \left[ \frac{1}{N} \sum_{i=1}^N \left| \mathcal{P}_{\text{pred}, i} - \mathcal{P}_{\text{DNS}, i} \right| + \lambda_{\text{stab}} \operatorname{ReLU}\left( -\overline{\mathcal{P}_{\text{pred}}} \right) \right]$$

where $\lambda_{\text{stab}} = 10.0$ establishes a hard numerical stability barrier and $\mathcal{P}_{\text{scale}} = \langle |\mathcal{P}_{\text{DNS}}| \rangle$ over the training split. $\mathcal{P}$ carries units of $\tau \nabla u$, so without that division the term silently rescales $\alpha$ whenever the subgrid intensity changes — and it changes by a factor of ~40 over the course of this KHI run.

### Multi-Scale Spectral Cascade Loss ($\mathcal{L}_{\text{spectral}}$)
Neural networks trained on point-wise losses suffer from spectral bias, damping high-wavenumber modes and acting as overly diffusive eddy viscosities. To preserve the turbulent power spectrum across spatial scales:

$$\mathcal{L}_{\text{spectral}} = \frac{1}{K_{\text{shells}}} \sum_{k} \left| \log_{10} E_{\text{pred}}(k) - \log_{10} E_{\text{DNS}}(k) \right|^2$$

where $E(k)$ is the radially integrated 2D shell energy spectrum density of the subgrid stress tensor components computed using differentiable 2D Fast Fourier Transforms (`torch.fft.rfft2`) and radial binning via `scatter_add`.

Both spectra are floored **relative** to the DNS spectrum, at `epsilon_rel` $\times \langle E_{\text{DNS}} \rangle$ (default $10^{-8}$), rather than at an absolute constant. An absolute floor is a scale trap: the unnormalized FFT power of $\tau \sim 10^{-3}$ on a $64^2$ grid lands near $10^{-8} N^4$, so a fixed $10^{-12}$ is either irrelevant or dominant depending only on grid size and subgrid intensity. It also matters at initialization, where the predicted spectrum can sit many decades below the DNS one in the high-$k$ shells and $\log_{10}$ of a near-zero spectrum produces a gradient spike that is pure noise. The relative floor bounds any single shell's residual at roughly $|\log_{10} \epsilon_{\text{rel}}|^2$ and makes the term invariant under a rescaling of $\tau$.

### Physical Invariant & Orthogonality Constraints ($\mathcal{L}_{\text{ortho}}$)
In conservative 3+1 relativistic hydrodynamics, subgrid stresses must reside purely in the fluid rest frame ($\bar{u}_\mu \tau^{\mu\nu} = 0$). In signature $(-, +, +, +)$, the naive contraction $g_{\nu\lambda} v^\nu v^\lambda = -(v^0)^2 + \mathbf{v}^2$ is indefinite and cancels timelike projection errors. To guarantee a strictly positive-definite metric penalty across all spacetime directions, we use the fluid-frame spatial projector $h_{\nu\lambda} = g_{\nu\lambda} + \bar{u}_\nu \bar{u}_\lambda$:

$$\mathcal{L}_{\text{ortho}} = \frac{1}{N} \sum_{i=1}^N \left[ \left( \bar{u}_\mu \tau_{\text{pred}}^{\mu\nu} \bar{u}_\nu \right)^2 + h_{\nu\lambda} \left( \bar{u}_\mu \tau_{\text{pred}}^{\mu\nu} \right) \left( \bar{u}_\alpha \tau_{\text{pred}}^{\alpha\lambda} \right) \right] = \frac{1}{N} \sum_{i=1}^N \left( (v^0)^2 + \mathbf{v}_{\perp}^2 \right) \ge 0$$

> **$\gamma$ defaults to 0: this term is identically zero for the current architecture.**
> `CRMHD_TBNN` forms its output as $\tau^{\mu\nu} = \sum_n G^{(n)} E^{(n)\mu\nu}$, and the basis tensors produced by `extract_tbnn_features` satisfy $\bar{u}_\mu E^{(n)\mu\nu} = 0$ to machine precision — measured at $3 \times 10^{-16}$ relative on `dns_khi` at $t = 4$. Both projections therefore vanish for *any* coefficients the network can emit. Over 151 epochs of an earlier run $\mathcal{L}_{\text{ortho}}$ sat at $\sim 10^{-33}$ throughout, so $\gamma = 5$ weighted a term that could only return zero, at the cost of two einsums per batch. The constraint is enforced by the architecture, not by the loss.
>
> The class is kept, and `--loss-gamma > 0` re-enables it, because it is the right diagnostic if the basis or the output head ever stops being rest-frame spatial — for instance if the currently unused `p_sgs` head is wired into the stress.

### Realizability & Weak Energy Condition ($\mathcal{L}_{\text{WEC}}$)
The effective stress-energy tensor $\overline{T}^{\mu\nu}_{\text{eff}} = T_{\text{resolved}}^{\mu\nu} + \tau_{\text{pred}}^{\mu\nu}$ must satisfy the Weak Energy Condition (WEC):

$$\varepsilon_{\text{eff}} = \overline{T}^{\mu\nu}_{\text{eff}} \bar{u}_\mu \bar{u}_\nu = \left( \rho h \bar{u}^\mu \bar{u}^\nu + p g^{\mu\nu} + \tau_{\text{pred}}^{\mu\nu} \right) \bar{u}_\mu \bar{u}_\nu = \rho(1 + \epsilon_{\text{int}}) + \tau_{\text{pred}}^{\mu\nu} \bar{u}_\mu \bar{u}_\nu \ge 0$$

Violations drive effective local energy density negative, causing catastrophic failure during primitive variable recovery ($D, S_j, \tau \to \rho, v^i, p$). The loss penalizes states approaching that failure:

$$\mathcal{L}_{\text{WEC}} = \frac{1}{N} \sum_{i=1}^N \operatorname{ReLU}\left( m - \frac{\varepsilon_{\text{eff}, i}}{\varepsilon_{\text{resolved}, i}} \right)$$

The hinge is on the **ratio** to the resolved comoving energy density, with margin $m$ (`--wec-margin`, default 0.5), not on the absolute value. The absolute form $\operatorname{ReLU}(-\varepsilon_{\text{eff}})$ cannot activate on this problem: $\varepsilon_{\text{resolved}} = \rho h - p \ge 2.1$ (mean 9.3) on `dns_khi` at $t = 4$, while the largest exact $\lVert \tau \rVert_F$ anywhere in the domain is 0.47 — a 5× margin. Measured over 151 epochs of an earlier run, the term was *exactly* 0.0 at every epoch, so $\delta$ bought nothing and no realizability constraint was in force at all.

Requiring $\varepsilon_{\text{eff}} \ge m \, \varepsilon_{\text{resolved}}$ instead gives a constraint that is inactive on the DNS labels but bites on a prediction that starts consuming the resolved energy density — which is the failure mode that actually ends a rollout. The term is dimensionless, so $\delta$ means the same thing at every subgrid intensity.

### Composite Objective & Default Weights
The total loss objective combines all five terms:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{tensor}} + \alpha \mathcal{L}_{\mathcal{P}} + \beta \mathcal{L}_{\text{spectral}} + \gamma \mathcal{L}_{\text{ortho}} + \delta \mathcal{L}_{\text{WEC}}$$

| Hyperparameter | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| $\alpha$ | `--loss-alpha` | `0.5` | Subgrid kinetic dissipation rate weight |
| $\beta$ | `--loss-beta` | `0.1` | Multi-scale 2D spectral cascade matching weight |
| $\gamma$ | `--loss-gamma` | `0.0` | Fluid rest-frame orthogonality weight. **Zero by default — the term is identically zero for this architecture** (see above). |
| $\delta$ | `--loss-delta` | `2.0` | Weak Energy Condition realizability penalty weight |
| $m$ | `--wec-margin` | `0.5` | Fraction of resolved comoving energy density the prediction must retain before $\mathcal{L}_{\text{WEC}}$ activates |
| $\lambda_{\text{stab}}$ | `--lambda-stab` | `10.0` | Global anti-dissipation stability barrier multiplier |
| $\tau_{\text{floor}}$ | `--tensor-floor-quantile` | `0.5` | $\mathcal{L}_{\text{tensor}}$ denominator floor, as a quantile of the $\lVert\tau\rVert_F$ distribution |
| — | `--tensor-floor-scope` | `snapshot` | Whether that quantile is taken **per snapshot** (default) or once over the whole training split. See below. |
| — | `--tensor-floor-floor-frac` | `0.05` | Lower bound on each per-snapshot floor, as a fraction of the global one |
| — | `--spectral-highk-tilt` | `0.0` | Per-shell weight $(k/k_{\min})^{\text{tilt}}$ on $\mathcal{L}_{\text{spectral}}$, normalised to mean 1 |

Every component is dimensionless, so these weights mean the same thing across datasets, coarsening factors and flow regimes. That relies on the dataset-level scales $\tau_{\text{scale}}$, $\tau_{\text{floor}}$ and $\mathcal{P}_{\text{scale}}$, which are computed from the **training split only** (via `RMHDSGSLoss.set_scales`) so that validation and test are measured against the training yardstick rather than each normalizing itself.

#### Why $\tau_{\text{floor}}$ is per-snapshot by default

$\mathcal{L}_{\text{tensor}}$ normalises each cell by $\lVert\tau_{\text{DNS}}\rVert^2 + \tau_{\text{floor}}^2$. A single global floor is only sound while the subgrid intensity is roughly stationary across the training window. It is not: $\lVert\tau\rVert_F$ varies by **~40×** across `dns_khi` over $t = 0 \ldots 20$. With one global floor, nearly every cell of the weak snapshots sits below it, the term degenerates into an absolute MSE there, and **predicting zero becomes near-optimal**.

That is not hypothetical. The model trained before this option existed (`checkpoints/best_model.pt`, window $t \in [1, 6.4]$, one global floor) holds its stress to $t \approx 12$ and then collapses — by $t = 20$ it has 40% of the right amplitude, 19% of the right high-$k$ content against 49% for the exact stress, and delivers 11% of the true energy transfer. Measured directly: a per-snapshot floor makes collapsing a 100× weaker snapshot **43× more expensive**.

**Extending the training window and switching the floor scope must be done together.** A longer window with a global floor makes the collapse worse, not better, because it adds exactly the weak snapshots the global floor silences.

Per-snapshot floors need a lower guard. At $t = 0$ there is no subgrid signal at all — the median $\lVert\tau\rVert_F$ is $\sim 3\times10^{-15}$ — and an unguarded per-snapshot floor spans $\sim 6\times10^{11}$, which would hand a signal-free snapshot a vanishing denominator and let it dominate the gradient with noise. `--tensor-floor-floor-frac 0.05` clamps each floor at 5% of the global one, bringing the spread to ~28×.

Use `--tensor-floor-scope global` to reproduce runs made before this option existed.

#### Tilting $\mathcal{L}_{\text{spectral}}$ toward small scales

As written, the spectral term weights a decade of error at $k = 2$ and at $k = 30$ identically. But the resolved field's small scales are what a closure exists to sustain, and the trained stress carries ~38% of its power above $k = 16$ against ~51% for the exact stress. `--spectral-highk-tilt` applies $(k/k_{\min})^{\text{tilt}}$ to the per-shell log residual, normalised to mean 1 so the tilt does not silently rescale the term against $\beta$. It buys small-scale accuracy at the cost of the large scales, which the point-wise terms already constrain well. Default `0.0` keeps the original behaviour.

### Single-Step Spectral Integration
Fast Fourier Transforms need structured 2D spatial grids, so the batching has to supply them. Batches are therefore whole **snapshots**: at $r = 16$ a snapshot is a $64 \times 64$ coarse field (4096 points), so a handful fit in one step and $\mathcal{L}_{\text{spectral}}$ is evaluated on the exact periodic domain inside the same forward pass as the point-wise terms. One group $\to$ one forward $\to$ one composite loss $\to$ one `backward()` $\to$ one clipped optimizer step.

> **What this replaced, and why it mattered.** With point-shuffled minibatches, locality was destroyed and $\mathcal{L}_{\text{spectral}}$ could only be computed in a separate pass, with its own `zero_grad` / `backward` / `step` *after* the epoch's point-wise steps. Three consequences, all measured on a 151-epoch run:
> - The final update of every epoch was a spectral-only gradient estimated from four random snapshots, reusing AdamW momentum contaminated by the point-wise pass and with no clipping.
> - That estimate was extremely noisy — the reported train $\mathcal{L}_{\text{spectral}}$ had a coefficient of variation of **44%**, against 4.5% for the same quantity measured over all validation snapshots.
> - Validation metrics therefore swung by ~2× between adjacent epochs and never settled, even at a learning rate of $5 \times 10^{-5}$: `val_rel_l2` had mean 19.4% and sd 1.7% over epochs 100–150, while the point-wise $\mathcal{L}_{\text{tensor}}$ was flat to 1.5%. All four `ReduceLROnPlateau` reductions fired on that noise.

### Output Calibration
`coeff_head` receives PyTorch's default `Linear` initialization, which knows nothing about the target scale. Contracted against the basis (mean $\lVert E \rVert_F \approx 0.17$) it emits $\lVert \tau_{\text{pred}} \rVert_F \approx 5 \times 10^{-2}$ while the DNS labels sit at $\approx 7 \times 10^{-3}$, so against the per-point floor the weak cells see a relative error of $O(10^3)$. Measured on `dns_khi`, that is a first-epoch loss of $4.6 \times 10^3$ and a gradient norm of $1.7 \times 10^5$ — and the opening phase of training does nothing but shrink the output.

Because the head is linear and bias-free, scaling its weight by the measured ratio $\langle\lVert \tau_{\text{DNS}} \rVert_F\rangle / \langle\lVert \tau_{\text{pred}} \rVert_F\rangle$ puts the initial prediction exactly at the target scale. This is a property of the head, not a heuristic, and changes neither the architecture nor the export contract. Measured effect: first-epoch loss $4.6 \times 10^3 \to 7.6$, first-epoch gradient norm $1.7 \times 10^5 \to 1.7 \times 10^3$, initial `rel_l2` 256% $\to$ 99%. Disable with `--no-calibrate-output`.

---

## 7. Stage 5: Optimization & Early Stopping

### AdamW, plateau schedule, adaptive clipping
- **Optimizer**: `AdamW` with learning rate $\eta = 10^{-3}$ and weight decay $\lambda = 10^{-5}$ applied directly to `model.parameters()`.
- **Scheduler** (`--lr-schedule`, default `plateau`): `ReduceLROnPlateau(factor=0.5, patience=lr_patience, min_lr=1e-7)` on the **EMA-smoothed** metric (see below), not the raw one. Plateau was unusable in the earlier run because its reductions fired on sampling noise from the two-phase spectral pass — the noise, not plateau scheduling itself, was the problem. With that pass gone (spectral is now a single-step term, see above) and the monitor smoothed, plateau is sound again, and it has two real advantages over a fixed decay here: it adapts reduction timing to the run actually happening instead of a schedule fixed at the start, and it needs no commitment to a total epoch count, which matters for `--resume` where the final epoch is not known in advance.

  Measured head-to-head on `dns_khi` (60 epochs, identical seed/config otherwise): plateau reached a better smoothed val loss (0.766 vs 0.778) and a better held-out test relative $L_2$ (85.1% vs 86.6%). At `--lr-patience 10` it never reduced the LR across the whole 60-epoch run — the smoothed metric kept inching down within every 10-epoch window — while cosine, whose decay schedule is anchored to the total `--epochs`, had decayed to $10^{-5}$ by epoch 60 regardless of whether it was still improving.

  `--lr-schedule cosine` is still available: linear warmup over `--warmup-epochs` (default 5) to $\eta$, then cosine decay to `--min-lr-frac` $\times \eta$. It needs no patience/factor tuning and decays monotonically, which can be preferable for a short, fixed-length run where a correct total epoch count is known upfront.
- **Gradient clipping** (`--grad-clip-mode`, default `adaptive`): clips at `--grad-clip-mult` (default 5) times the running median of recent gradient norms. A fixed threshold is the wrong tool here — $\mathcal{L}_{\text{tensor}}$ is a ratio whose denominator is floored at the median $\lVert\tau\rVert_F$ ($\approx 1.3 \times 10^{-3}$), so $\partial \mathcal{L} / \partial \tau$ is intrinsically $O(10^3)$ and the gradient norm runs $10^1$–$10^3$ through a healthy run. A clip of 1.0 fires on *every* step, which keeps the direction and discards the magnitude, silently turning AdamW into a normalized-step method and hiding genuine spikes among ordinary ones. The `clip_frac` column in `metrics.csv` reports the fraction of steps clipped; watch it stay low. `--grad-clip-mode absolute --grad-clip X` and `off` are also available.

### Monitored-metric smoothing
Everything that *decides* something — best-checkpoint selection, LR reduction, early stopping — reads an EMA of the validation metric (`--monitor-ema`, default 0.8, ≈ 5 epochs), bias-corrected so early epochs are not dragged toward the cold start. The raw value is still logged, in the `val_loss` column beside `val_loss_ema`.

Selecting on the raw value means selecting noise. In the earlier run, `val_rel_l2` had mean 19.4% and sd 1.7% over epochs 100–150; the checkpoint kept as "best" was the 16.3% at epoch 107, a $-1.8\sigma$ draw rather than a better model. `--monitor-ema 0.0` restores raw selection.

### Early Stopping Monitor
The `EarlyStopping` utility tracks the **smoothed** validation composite loss:
- Defaults to `patience = 2 * lr_patience`. This guarantees that after a learning rate reduction, the optimizer has at least one full plateau cycle to explore the finer objective landscape before training terminates.
- Tracks `best_score` and `best_epoch`. If no improvement greater than `min_delta` occurs for `patience` consecutive epochs, training terminates gracefully.

---

## 8. Stage 6: Epoch Execution & Checkpointing

### Training & Validation Loops
At each epoch:
1. `train_epoch()`:
   - The training view's snapshots are partitioned into shuffled **groups** of `--snapshots-per-batch`. No `drop_last`: every snapshot is seen every epoch.
   - Each group is gathered as flat tensors (invariants, thermodynamic features, basis tensors, exact stress $\tau_{\text{SGS}}$, 4-velocity $u^\mu$, velocity gradients $\nabla_\mu u_\nu$, density $\rho$, pressure $p$) plus the per-snapshot `(offset, Ny, Nx)` needed to reshape predictions back to 2D.
   - Features are normalized using the training statistics $(\mu, \sigma)$.
   - One forward pass produces $\Delta T_{\text{SGS}}$ for every point in the group. The point-wise terms are evaluated on all of them and the spectral term on each snapshot's exact periodic field, summed into a single scalar.
   - One `backward()`, one adaptive clip, one optimizer step. Nothing is applied after the last group, so the model state validation sees is the state the epoch's optimization actually produced.
2. `validate()`:
   - Same objective, no gradients, groups taken **in order** — every snapshot visited exactly once, so the only epoch-to-epoch variation in the reported numbers comes from the model changing.
     - **Val Loss**: Full composite physics loss $\mathcal{L}_{\text{total}}$.
     - **Relative $L_2$ Error** (the headline accuracy number):
       $$\text{Rel } L_2 = \frac{\|\Delta T_{\text{SGS}} - \tau_{\text{SGS}}\|_F}{\|\tau_{\text{SGS}}\|_F}$$
     - **Raw MSE**: $\frac{1}{16} \sum_{\mu, \nu} (\Delta T^{\mu\nu} - \tau^{\mu\nu})^2$, kept for continuity only — it reads $\sim 10^{-7}$ because $\lVert\tau\rVert \sim 7 \times 10^{-3}$ and says nothing about accuracy on its own.
     - Individual loss components: $\mathcal{L}_{\text{tensor}}, \mathcal{L}_{\mathcal{P}}, \mathcal{L}_{\text{spectral}}, \mathcal{L}_{\text{ortho}}, \mathcal{L}_{\text{WEC}}$.
3. **Progress Feedback**:
   On a tty, `tqdm` shows a live bar. When stdout is redirected — which is every Condor run — it prints one line per epoch instead, because a redirected tqdm bar rewrites the same line with carriage returns and the whole run arrives as one unreadable line in the `.out` file:
   ```
   ep    41/2500 loss=6.69e-01  val=7.79e-01  val_ema=8.03e-01  rel_l2=84.0%  L_P=7.43e-01  L_spec=4.15e-01  |g|=3.7e+01  clip=0%  lr=2.7e-04  (0.3s)
   ```

### Checkpointing & Resumption
During training, two complementary checkpoints are maintained in `--checkpoint-dir`:
1. **`latest_model.pt` (Saved every epoch)**:
   - Serialized atomically after every completed epoch.
   - Contains current epoch, model weights, optimizer state, LR scheduler state, `EarlyStopping` counters, the monitor EMA and adaptive-clip histories, the resolved $\Gamma$, the loss scales ($\tau_{\text{scale}}$, $\tau_{\text{floor}}$, $\mathcal{P}_{\text{scale}}$), the snapshot index split and its times, normalization statistics, and the complete metrics history.
   - Enables seamless resumption from the exact last completed epoch if training is interrupted.
2. **`best_model.pt` (Saved on validation loss improvement)**:
   - Updated whenever the **smoothed** validation loss achieves a new global minimum.
   - Contains all state attributes present in `latest_model.pt`.

> [!IMPORTANT]
> At the end of training (or upon early stopping), [`train_apriori.py`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/train_apriori.py) automatically reloads the weights from `best_model.pt` before exporting TorchScript or final artifacts, ensuring the deployed model never suffers from late-epoch overfitting.

### Resuming Training
Training can be resumed using `--resume`:
- `python -m src.train_apriori --resume`: Automatically detects and loads `latest_model.pt` (or falls back to `best_model.pt` or `final_model.pt`) from `--checkpoint-dir`.
- `python -m src.train_apriori --resume path/to/checkpoint.pt` or `--resume-from path/to/checkpoint.pt`: Resumes from an explicit checkpoint file.
- `--reset-lr`: Resets the optimizer learning rate to `--lr` instead of restoring the checkpoint's saved learning rate.
- **Log Continuity**: When resuming, `metrics.csv` preserves all rows up to the resume epoch and appends subsequent epochs without duplicate entries; `loss_history.png` continues plotting seamlessly across all epochs.

---

## 9. Stage 7: Diagnostics, Logging & Visualization

Diagnostics are generated every `--plot-interval` epochs (default: 10):

### Multi-Panel Convergence Curves
Saved to `checkpoints/loss_history.png` by [`plot_loss_curves()`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/train_apriori.py#L457-L638):
1. **Total Composite Loss**: Train, raw validation, and the **EMA-smoothed validation curve drawn over it**. The smoothed curve is what selects the best checkpoint and drives the LR schedule, so plotting only the raw trace invites reading noise as signal. Inset zoom for settled epochs.
2. **Relative $L_2$ Error (%)**: Logarithmic scale spanning initial transients down to fine convergence, with a linear settled-regime inset.
3. **Physics Loss Components**: Semilog breakdown tracking $\mathcal{L}_{\text{tensor}}$, $\mathcal{L}_{\mathcal{P}}$, $\mathcal{L}_{\text{spectral}}$, $\mathcal{L}_{\text{ortho}}$, and $\mathcal{L}_{\text{WEC}}$.
4. **Gradient Norm** with the **fraction of steps clipped** overlaid on a second axis. A clip firing on most steps is not a safety net — it replaces the gradient magnitude with a constant — so it is plotted where it cannot go unnoticed.
5. **Learning Rate Schedule**. Regime demarcation lines are drawn only for step-like schedules; a cosine schedule changes the LR every epoch, and marking each change would paint over the whole figure.

Raw MSE is **off by default** (`include_mse=False`). It reads $\sim 10^{-7}$ only because $\lVert \tau \rVert \sim 7 \times 10^{-3}$ and carries no accuracy information on its own; relative $L_2$ is the headline number.

### 2D Spatial Stress & Parity Maps
Saved to `checkpoints/plots/epoch_XXXX_pred_vs_truth.png` and `checkpoints/latest_pred_vs_truth.png` via [`plot_pred_vs_truth()`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/train_apriori.py#L641-L731):
Visualizes 4 representative tensor components on a $256 \times 256$ slice:
- $\tau^{xx}$: Normal stress in $x$
- $\tau^{xy}$: Shear stress
- $\tau^{yy}$: Normal stress in $y$
- $\tau^{0x}$: Relativistic energy/momentum flux

For each component, 4 columns are displayed:
1. **Exact Ground Truth** (Viridis colormap)
2. **TBNN Prediction** (Viridis colormap, identical bounds)
3. **Difference Map** ($\Delta T - \tau$, Coolwarm colormap centered at 0)
4. **Parity Scatter Plot** (Exact vs. Predicted with dashed $y = x$ reference line)

### Metrics CSV & JSON History
- `checkpoints/metrics.csv`: Real-time append of per-epoch tabular data:
  `epoch, train_loss, val_loss, val_loss_ema, grad_norm, clip_frac, val_mse, val_rel_l2, L_tensor, L_P, L_ortho, L_WEC, L_spectral, val_L_*, lr, elapsed_seconds`
  `val_loss` is the raw value and `val_loss_ema` the smoothed one that drives every decision. `clip_frac` is the fraction of the epoch's optimizer steps whose gradient was clipped — watch it stay low.
- `checkpoints/history.json`: Machine-readable summary of entire loss histories, early stopping status, and final parameter values.

---

## 10. Stage 8: TorchScript JIT Export for Solver Deployment

Once training is complete and best weights are restored, [`train_apriori.py`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/train_apriori.py#L975-L985) compiles the model into a standalone TorchScript binary:

```python
sample_inv = torch.randn(1, 15, dtype=dtype, device=device)
sample_thermo = torch.randn(1, 2, dtype=dtype, device=device)
sample_basis = torch.randn(1, 12, 4, 4, dtype=dtype, device=device)
traced = torch.jit.trace(model, (sample_inv, sample_thermo, sample_basis))
traced.save(os.path.join(checkpoint_dir, "tbnn_traced.pt"))
```

This traced model [`tbnn_traced.pt`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/checkpoints/tbnn_traced.pt) is optimized for zero-Python-overhead C++ and PyTorch JIT execution inside the hydrodynamic solver during *a posteriori* rollouts ([`src/solver.py`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/solver.py) and [`src/evaluate_rollout.py`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/evaluate_rollout.py)).

---

## 11. Command-Line Interface (CLI) Reference

### Full Arguments List

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data-dir` | `str` | `dns_khi` | Path to directory containing HDF5 simulation snapshots |
| `--coarsening-factor` | `int` | `16` | Spatial filtering coarsening ratio $r$. At $r = 16$ a $1024^2$ DNS gives a $64^2$ coarse field. |
| `--stride` | `int` | `1` | Temporal sampling stride across simulation snapshots |
| `--epochs` | `int` | `50` | Maximum number of training epochs |
| `--time-range` | `float float` | `1.0 6.5` | Window used for training **and** validation, interleaved inside it |
| `--test-time` | `float float` | `6.5 1e30` | Contiguous window held out for the final measurement only |
| `--no-test-split` | `flag` | `False` | Do not hold out a test window |
| `--val-every` | `int` | `5` | Hold out every $N$th snapshot of the window for validation |
| `--snapshots-per-batch` | `int` | `2` | Snapshots per optimizer step (4096 points each at $r = 16$) |
| `--eval-snapshots-per-batch` | `int` | `4` | Snapshots per forward pass during validation |
| `--lr` | `float` | `1e-3` | Peak AdamW learning rate |
| `--lr-schedule` | `str` | `plateau` | `plateau` (on the EMA-smoothed metric) or `cosine` (with warmup) |
| `--warmup-epochs` | `int` | `5` | Linear LR warmup epochs for the cosine schedule |
| `--min-lr-frac` | `float` | `0.01` | Cosine floor as a fraction of `--lr` |
| `--grad-clip-mode` | `str` | `adaptive` | `adaptive` (× running median), `absolute`, or `off` |
| `--grad-clip-mult` | `float` | `5.0` | Adaptive clip threshold as a multiple of the running median |
| `--grad-clip` | `float` | `0.0` | Absolute clip value, used only with `--grad-clip-mode absolute` |
| `--monitor-ema` | `float` | `0.8` | EMA of the validation metric that drives selection, LR and early stopping. `0` selects on the raw value. |
| `--seed` | `int` | `0` | Seed for initialization and batch order |
| `--lr-patience` | `int` | `100` | Plateau epochs before halving the LR (`--lr-schedule plateau` only) |
| `--patience` | `int` | `2 * lr_patience` | Early stopping patience epochs (set $\le 0$ to disable) |
| `--min-delta` | `float` | `0.0` | Minimum validation loss improvement threshold |
| `--hidden-dim` | `int` | `256` | Width of MLP hidden layers |
| `--num-hidden-layers` | `int` | `3` | Number of dense hidden layers in MLP backbone |
| `--loss-alpha` | `float` | `0.5` | Weight $\alpha$ for kinetic dissipation rate loss $\mathcal{L}_{\mathcal{P}}$ |
| `--loss-beta` | `float` | `0.1` | Weight $\beta$ for 2D shell spectral cascade loss $\mathcal{L}_{\text{spectral}}$ |
| `--loss-gamma` | `float` | `0.0` | Weight $\gamma$ for $\mathcal{L}_{\text{ortho}}$. **Zero: the term is identically zero for this architecture.** |
| `--loss-delta` | `float` | `2.0` | Weight $\delta$ for Weak Energy Condition loss $\mathcal{L}_{\text{WEC}}$ |
| `--wec-margin` | `float` | `0.5` | Fraction of resolved comoving energy density retained before $\mathcal{L}_{\text{WEC}}$ activates |
| `--tensor-floor-quantile` | `float` | `0.5` | $\mathcal{L}_{\text{tensor}}$ denominator floor, as a quantile of the training $\lVert\tau\rVert_F$ |
| `--lambda-stab` | `float` | `10.0` | Multiplier for global anti-dissipation stability barrier |
| `--no-calibrate-output` | `flag` | `False` | Skip rescaling `coeff_head` at init to match the target $\tau$ scale |
| `--no-restore-best` | `flag` | `False` | Skip restoring best validation weights before TorchScript export |
| `--plot-interval` | `int` | `10` | Frequency in epochs to save diagnostics and spatial plots |
| `--checkpoint-dir` | `str` | `checkpoints` | Target folder for checkpoints, plots, and logs |
| `--device` | `str` | `cpu` | Device specification (`cuda` or `cpu`) |
| `--dtype` | `str` | `float64` | Numerical precision (`float64` or `float32`) |
| `--resume` | `str` (optional) | `None` | Resume training (`--resume` for auto-discovery in checkpoint dir, or `--resume <path>`) |
| `--resume-from` | `str` | `None` | Explicit path to checkpoint file to resume training from |
| `--reset-lr` | `flag` | `False` | Override saved learning rate in checkpoint with value of `--lr` |

### Example Invocations

#### Interactive Training (GPU, Double Precision, Production Run):
```bash
source .venv/bin/activate
python -m src.train_apriori \
    --data-dir dns_khi \
    --coarsening-factor 16 \
    --stride 1 \
    --epochs 2500 \
    --snapshots-per-batch 2 \
    --lr 1e-3 \
    --lr-schedule plateau \
    --warmup-epochs 5 \
    --loss-alpha 0.5 \
    --loss-beta 0.1 \
    --loss-delta 2.0 \
    --hidden-dim 256 \
    --num-hidden-layers 3 \
    --checkpoint-dir checkpoints \
    --device cuda \
    --dtype float64
```

Most of those are already the defaults, and `crmhd.toml` carries them in its `[training]` section, so the short form is equivalent:

```bash
python -m src.train_apriori --epochs 2500 --device cuda
```

$\Gamma$ is **not** a flag here: it is resolved from the dataset's own metadata, which outranks the config file (`dns_khi` records `eos_gamma = 1.3333...`). The run prints what it resolved and from where.

#### Resuming an Existing or Interrupted Run:
```bash
source .venv/bin/activate
# Automatically resumes from checkpoints/latest_model.pt (or best_model.pt):
python -m src.train_apriori \
    --data-dir dns_khi \
    --coarsening-factor 16 \
    --stride 1 \
    --epochs 3000 \
    --checkpoint-dir checkpoints \
    --device cuda \
    --dtype float64 \
    --resume

# Or resume from a specific checkpoint with a reset learning rate:
python -m src.train_apriori \
    --resume-from checkpoints/best_model.pt \
    --reset-lr \
    --lr 5e-5 \
    --epochs 3000 \
    --device cuda
```

#### Fast Development / Debug Run:
```bash
source .venv/bin/activate
python -m src.train_apriori \
    --data-dir dns_khi \
    --coarsening-factor 16 \
    --stride 8 \
    --epochs 5 \
    --val-every 3 \
    --snapshots-per-batch 2 \
    --plot-interval 1 \
    --device cpu \
    --dtype float32
```

A coarse `--stride` leaves few snapshots, so `--val-every 3` is needed to keep the
validation split non-empty. An empty split is a hard error, not a silent skip.

#### HTCondor Batch Submission:
```bash
condor_submit condor_train.sub
# Monitor live output
tail -f condor_logs/train_<ClusterId>_0.out
```
