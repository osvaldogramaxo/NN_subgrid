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
   - [AdamW & ReduceLROnPlateau](#adamw--reducelronplateau)
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
[DNS HDF5 Snapshots (test1)]
         │
         ▼
[Relativistic Favre Filtering (r=4/8)] ────► Compute Exact SGS Stress: τ_SGS = T_bar - T_coarse
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
[Dataset Caching & Normalization] ─────────► train_cache.pt / val_cache.pt (v2)
                                             Cached: u^μ, ∇_μ u_ν, ρ, p, snapshot boundaries
         │
         ▼
[Minibatch PyTorch DataLoader]
         │
         ▼
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
│   Minibatch Physics Losses:                  ▼                         │
│   L_mb = L_tensor + α L_P + γ L_ortho + δ L_WEC                        │
│                                              │                         │
│   Full Snapshot Pass (End of Epoch):         ▼                         │
│   L_spectral = MSE(log10 E_pred(k), log10 E_DNS(k))                    │
│                                              │                         │
│   Composite Loss: L_total = L_mb + β L_spectral                        │
│   Backprop (AdamW) & Step ◄──────────────────┘                         │
└────────────────────────────────────────────────────────────────────────┘
         │
         ├──► Per-epoch Validation (Composite loss, Physical MSE, Rel L2 error)
         ├──► ReduceLROnPlateau Learning Rate Update
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
The foundational structural regression loss contracts tensor prediction errors using the spacetime metric $g_{\mu\nu} = \text{diag}(-1, 1, 1, 1)$, normalized by batch DNS intensity:

$$\mathcal{L}_{\text{tensor}} = \frac{\frac{1}{N} \sum_{i=1}^{N} g_{\mu\alpha} g_{\nu\beta} \left(\tau_{\text{pred}}^{\mu\nu} - \tau_{\text{DNS}}^{\mu\nu}\right)_i \left(\tau_{\text{pred}}^{\alpha\beta} - \tau_{\text{DNS}}^{\alpha\beta}\right)_i}{\frac{1}{N} \sum_{i=1}^{N} \lVert \tau_{\text{DNS}, i} \rVert_F^2 + \epsilon}$$

where $\lVert A \rVert_F^2 = g_{\mu\alpha} g_{\nu\beta} A^{\mu\nu} A^{\alpha\beta}$ and $\epsilon = 10^{-8}$. Batch-level normalization eliminates point-wise division-by-zero singularities in quasi-laminar/low-shear zones while preserving scale-invariant Frobenius regression. Because fluid rest-frame tensors satisfy $\bar{u}_\mu \tau^{\mu\nu} = 0$, the metric contraction is non-negative and is clamped at $\ge 0$ against numerical roundoff.

### Kinetic & Magnetic Dissipation Rate Penalty ($\mathcal{L}_{\mathcal{P}}$)
Subgrid dissipation rate density measures energy transfer between resolved and unresolved scales:

$$\mathcal{P}_{\text{SGS}} = -\tau^{\mu\nu} \nabla_\mu \bar{u}_\nu$$

Net anti-dissipation ($\int_\Omega \mathcal{P}_{\text{pred}} dV < 0$) pumps spurious energy back into resolved modes, rapidly destabilizing standard Runge-Kutta time integration. The dissipation loss enforces:

$$\mathcal{L}_{\mathcal{P}} = \frac{1}{N} \sum_{i=1}^N \left| \mathcal{P}_{\text{pred}, i} - \mathcal{P}_{\text{DNS}, i} \right| + \lambda_{\text{stab}} \operatorname{ReLU}\left( -\int_\Omega \mathcal{P}_{\text{pred}} \, dV \right)$$

where $\lambda_{\text{stab}} = 10.0$ establishes a hard numerical stability barrier.

### Multi-Scale Spectral Cascade Loss ($\mathcal{L}_{\text{spectral}}$)
Neural networks trained on point-wise losses suffer from spectral bias, damping high-wavenumber modes and acting as overly diffusive eddy viscosities. To preserve the turbulent power spectrum across spatial scales:

$$\mathcal{L}_{\text{spectral}} = \frac{1}{K_{\text{shells}}} \sum_{k} \left| \log_{10} E_{\text{pred}}(k) - \log_{10} E_{\text{DNS}}(k) \right|^2$$

where $E(k)$ is the radially integrated 2D shell energy spectrum density of the subgrid stress tensor components computed using differentiable 2D Fast Fourier Transforms (`torch.fft.rfft2`) and radial binning via `scatter_add`.

### Physical Invariant & Orthogonality Constraints ($\mathcal{L}_{\text{ortho}}$)
In conservative 3+1 relativistic hydrodynamics, subgrid stresses must reside purely in the fluid rest frame ($\bar{u}_\mu \tau^{\mu\nu} = 0$). In signature $(-, +, +, +)$, the naive contraction $g_{\nu\lambda} v^\nu v^\lambda = -(v^0)^2 + \mathbf{v}^2$ is indefinite and cancels timelike projection errors. To guarantee a strictly positive-definite metric penalty across all spacetime directions, we use the fluid-frame spatial projector $h_{\nu\lambda} = g_{\nu\lambda} + \bar{u}_\nu \bar{u}_\lambda$:

$$\mathcal{L}_{\text{ortho}} = \frac{1}{N} \sum_{i=1}^N \left[ \left( \bar{u}_\mu \tau_{\text{pred}}^{\mu\nu} \bar{u}_\nu \right)^2 + h_{\nu\lambda} \left( \bar{u}_\mu \tau_{\text{pred}}^{\mu\nu} \right) \left( \bar{u}_\alpha \tau_{\text{pred}}^{\alpha\lambda} \right) \right] = \frac{1}{N} \sum_{i=1}^N \left( (v^0)^2 + \mathbf{v}_{\perp}^2 \right) \ge 0$$

### Realizability & Weak Energy Condition ($\mathcal{L}_{\text{WEC}}$)
The effective stress-energy tensor $\overline{T}^{\mu\nu}_{\text{eff}} = T_{\text{resolved}}^{\mu\nu} + \tau_{\text{pred}}^{\mu\nu}$ must satisfy the Weak Energy Condition (WEC):

$$\varepsilon_{\text{eff}} = \overline{T}^{\mu\nu}_{\text{eff}} \bar{u}_\mu \bar{u}_\nu = \left( \rho h \bar{u}^\mu \bar{u}^\nu + p g^{\mu\nu} + \tau_{\text{pred}}^{\mu\nu} \right) \bar{u}_\mu \bar{u}_\nu = \rho(1 + \epsilon_{\text{int}}) + \tau_{\text{pred}}^{\mu\nu} \bar{u}_\mu \bar{u}_\nu \ge 0$$

Violations drive effective local energy density negative, causing catastrophic failure during primitive variable recovery ($D, S_j, \tau \to \rho, v^i, p$). The loss penalizes non-physical states:

$$\mathcal{L}_{\text{WEC}} = \frac{1}{N} \sum_{i=1}^N \operatorname{ReLU}\left( -\varepsilon_{\text{eff}, i} \right)$$

### Composite Objective & Default Weights
The total loss objective combines all five terms:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{tensor}} + \alpha \mathcal{L}_{\mathcal{P}} + \beta \mathcal{L}_{\text{spectral}} + \gamma \mathcal{L}_{\text{ortho}} + \delta \mathcal{L}_{\text{WEC}}$$

| Hyperparameter | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| $\alpha$ | `--loss-alpha` | `0.5` | Subgrid kinetic dissipation rate weight |
| $\beta$ | `--loss-beta` | `0.1` | Multi-scale 2D spectral cascade matching weight |
| $\gamma$ | `--loss-gamma` | `5.0` | Fluid rest-frame orthogonality constraint weight |
| $\delta$ | `--loss-delta` | `2.0` | Weak Energy Condition realizability penalty weight |
| $\lambda_{\text{stab}}$ | `--lambda-stab` | `10.0` | Global anti-dissipation stability barrier multiplier |

### Two-Phase Spectral Integration Dynamics
Because Fast Fourier Transforms require structured 2D spatial grids ($H \times W$), point-wise shuffled minibatches cannot directly evaluate $\mathcal{L}_{\text{spectral}}$. [`train_apriori.py`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/train_apriori.py) implements a clean two-phase architecture:
1. **Minibatch Phase**: Shuffled batches accumulate gradients from $\mathcal{L}_{\text{tensor}} + \alpha \mathcal{L}_{\mathcal{P}} + \gamma \mathcal{L}_{\text{ortho}} + \delta \mathcal{L}_{\text{WEC}}$.
2. **Snapshot Spectral Phase**: At the conclusion of each epoch, the model evaluates complete 2D spatial snapshots (`--spectral-snapshots`, default: 2), computes $\beta \mathcal{L}_{\text{spectral}}$, and backpropagates through the network parameters before the optimizer step.

---

## 7. Stage 5: Optimization & Early Stopping

### AdamW & ReduceLROnPlateau
- **Optimizer**: `AdamW` with learning rate $\eta = 10^{-3}$ and weight decay $\lambda = 10^{-5}$ applied directly to neural network weights `model.parameters()`.
- **Scheduler**: `ReduceLROnPlateau(mode="min", factor=0.5, patience=lr_patience, min_lr=1e-7)`.
  When validation loss ceases to decrease for `lr_patience` epochs (default: 100), the learning rate is halved.

### Early Stopping Monitor
The [`EarlyStopping`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/train_apriori.py#L248-L310) utility tracks validation composite loss:
- Defaults to `patience = 2 * lr_patience` (e.g., 200 epochs). This guarantees that after a learning rate reduction, the optimizer has at least one full plateau cycle to explore the finer objective landscape before training terminates.
- Tracks `best_score` and `best_epoch`. If no improvement greater than `min_delta` occurs for `patience` consecutive epochs, training terminates gracefully.

---

## 8. Stage 6: Epoch Execution & Checkpointing

### Training & Validation Loops
At each epoch:
1. [`train_epoch()`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/train_apriori.py#L313-L377):
   - Minibatches (batch size 2048) are fetched with shuffling, supplying invariants, thermodynamic features, basis tensors, exact stress $\tau_{\text{SGS}}$, 4-velocity $u^\mu$, velocity gradients $\nabla_\mu u_\nu$, density $\rho$, and pressure $p$.
   - Features are normalized using saved training statistics $(\mu, \sigma)$.
   - Predictions $\Delta T_{\text{SGS}}$ are generated and evaluated with `criterion(..., compute_spectral=False)`.
   - Full 2D snapshots evaluate `criterion.compute_spectral(...)` and accumulate gradients.
   - Optimizer steps and updates model parameters.
2. [`validate()`](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/src/train_apriori.py#L380-L451):
   - Computes validation metrics without gradient computation:
     - **Val Loss**: Full composite physics loss $\mathcal{L}_{\text{total}}$.
     - **Physical MSE**: Mean squared difference $\frac{1}{16} \sum_{\mu, \nu} (\Delta T^{\mu\nu} - \tau^{\mu\nu})^2$.
     - **Relative $L_2$ Error**: Normalized Frobenius norm per sample:
       $$\text{Rel } L_2 = \frac{\|\Delta T_{\text{SGS}} - \tau_{\text{SGS}}\|_F}{\|\tau_{\text{SGS}}\|_F}$$
     - Individual loss components: $\mathcal{L}_{\text{tensor}}, \mathcal{L}_{\mathcal{P}}, \mathcal{L}_{\text{spectral}}, \mathcal{L}_{\text{ortho}}, \mathcal{L}_{\text{WEC}}$.
3. **Live Progress Feedback**:
   `tqdm` displays a live status bar showing current loss, validation loss, MSE, individual loss components, and current learning rate:
   ```
   Epochs:  42%|████▏     | 1050/2500 [1:12:03<1:39:20,  4.11s/epoch, loss=1.24e-01, val=1.31e-01, mse=4.12e-06, L_t=0.08, L_P=0.02, L_k=0.01, L_o=0.01, lr=2.5e-04]
   ```

### Checkpointing & Resumption
During training, two complementary checkpoints are maintained in `--checkpoint-dir`:
1. **`latest_model.pt` (Saved every epoch)**:
   - Serialized atomically after every completed epoch.
   - Contains current epoch, model weights, optimizer state, `ReduceLROnPlateau` scheduler state, `EarlyStopping` counters, best score/epoch trackers, normalization statistics, and complete metrics history.
   - Enables seamless resumption from the exact last completed epoch if training is interrupted.
2. **`best_model.pt` (Saved on validation loss improvement)**:
   - Updated whenever validation loss achieves a new global minimum.
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
1. **Total Composite Loss**: Train vs. Validation loss curves with an inset zoom for settled epochs ($\ge 50$).
2. **Physical Validation MSE**: Semilog plot displaying true dimensional squared error ($10^{-7}$ to $10^{-3}$) with a red star marker at the minimum.
3. **Relative $L_2$ Error (%)**: Logarithmic scale spanning initial transients down to fine convergence ($60\% - 80\%$), accompanied by a linear settled-regime inset.
4. **Physics Loss Components**: Semilog breakdown tracking individual trajectories: $\mathcal{L}_{\text{tensor}}$, $\alpha \mathcal{L}_{\mathcal{P}}$, $\beta \mathcal{L}_{\text{spectral}}$, $\gamma \mathcal{L}_{\text{ortho}}$, and $\delta \mathcal{L}_{\text{WEC}}$.
5. **Learning Rate Schedule**: Log scale showing step drops at plateaus. Vertical dotted lines mark LR transitions across all subplots.

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
  `epoch, train_loss, val_loss, val_mse, val_rel_l2, L_tensor, L_diss, L_spectral, L_ortho, L_wec, lr, elapsed_seconds`
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
| `--data-dir` | `str` | `test1` | Path to directory containing Aenus HDF5 simulation snapshots |
| `--coarsening-factor` | `int` | `8` | Spatial filtering coarsening ratio $r$ (e.g. 4 or 8) |
| `--stride` | `int` | `5` | Temporal sampling stride across simulation snapshots |
| `--epochs` | `int` | `50` | Maximum number of training epochs |
| `--batch-size` | `int` | `2048` | Minibatch size in grid points |
| `--lr` | `float` | `1e-3` | Initial AdamW learning rate |
| `--lr-patience` | `int` | `100` | Number of plateau epochs before halving learning rate |
| `--patience` | `int` | `2 * lr_patience` | Early stopping patience epochs (set $\le 0$ to disable) |
| `--min-delta` | `float` | `0.0` | Minimum validation loss improvement threshold |
| `--hidden-dim` | `int` | `64` | Width of MLP hidden layers |
| `--num-hidden-layers` | `int` | `6` | Number of dense hidden layers in MLP backbone |
| `--loss-alpha` | `float` | `0.5` | Weight $\alpha$ for kinetic dissipation rate loss $\mathcal{L}_{\mathcal{P}}$ |
| `--loss-beta` | `float` | `0.1` | Weight $\beta$ for 2D shell spectral cascade loss $\mathcal{L}_{\text{spectral}}$ |
| `--loss-gamma` | `float` | `5.0` | Weight $\gamma$ for fluid rest-frame orthogonality loss $\mathcal{L}_{\text{ortho}}$ |
| `--loss-delta` | `float` | `2.0` | Weight $\delta$ for Weak Energy Condition loss $\mathcal{L}_{\text{WEC}}$ |
| `--lambda-stab` | `float` | `10.0` | Multiplier for global anti-dissipation stability barrier |
| `--spectral-snapshots`| `int` | `2` | Number of validation snapshots evaluated for spectral loss |
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
    --data-dir test1 \
    --coarsening-factor 4 \
    --stride 5 \
    --epochs 2500 \
    --batch-size 2048 \
    --lr 1e-3 \
    --lr-patience 100 \
    --loss-alpha 0.5 \
    --loss-beta 0.1 \
    --loss-gamma 5.0 \
    --loss-delta 2.0 \
    --hidden-dim 64 \
    --num-hidden-layers 8 \
    --checkpoint-dir checkpoints \
    --device cuda \
    --dtype float64
```

#### Resuming an Existing or Interrupted Run:
```bash
source .venv/bin/activate
# Automatically resumes from checkpoints/latest_model.pt (or best_model.pt):
python -m src.train_apriori \
    --data-dir test1 \
    --coarsening-factor 4 \
    --stride 5 \
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
    --data-dir test1 \
    --coarsening-factor 8 \
    --stride 10 \
    --epochs 5 \
    --batch-size 512 \
    --plot-interval 1 \
    --device cpu \
    --dtype float32
```

#### HTCondor Batch Submission:
```bash
condor_submit condor_train.sub
# Monitor live output
tail -f condor_logs/train_<ClusterId>_0.out
```
