# CRMHD-TBNN

**Tensor Basis Neural Network for Compressible Relativistic Hydrodynamic Subgrid-Scale Closures**

A PyTorch implementation of a physics-informed neural network that learns subgrid-scale (SGS) stress tensor closures for Large Eddy Simulations (LES) of compressible relativistic hydrodynamics. The network architecture is built on a covariant tensor basis expansion, guaranteeing that predictions respect the Lorentz symmetries of special relativity by construction.

---

## Overview

When simulating turbulent relativistic flows at tractable resolutions, the effect of unresolved small-scale dynamics on the resolved (coarse) fields must be modelled — this is the *closure problem*. Classical algebraic models (e.g., Smagorinsky-type) lack the expressiveness needed for relativistic regimes, while unconstrained neural networks offer no symmetry guarantees.

**CRMHD-TBNN** bridges this gap with a three-stage pipeline:

1. **Data generation** — High-resolution DNS snapshots (from the Aenus solver) are Favre-filtered onto a coarse grid, and the exact SGS stress tensor τ<sub>SGS</sub><sup>μν</sup> is computed as the filtering residual.

2. **Feature extraction** — At each coarse grid point, the code computes 15 Lorentz-invariant scalar quantities Λ<sub>k</sub> (from the shear tensor σ<sup>μν</sup>, vorticity ω<sup>μν</sup>, four-acceleration a<sup>μ</sup>, and magnetic field b<sup>μ</sup>) together with 12 isotropic symmetric basis tensors E<sup>(n)μν</sup>.

3. **Neural closure** — A deep MLP maps the invariants + thermodynamic features (expansion scalar θ, turbulent Mach number M<sub>t</sub>) to 12 scalar coefficients G<sup>(n)</sup>, which are contracted against the basis tensors to reconstruct the full SGS stress:

$$\Delta T_{\text{SGS}}^{\mu\nu} = \sum_{n=0}^{11} G^{(n)}(\Lambda_k,\,\theta,\,M_t)\; E^{(n)\mu\nu}$$

Because both the invariants and the basis tensors transform covariantly, the output tensor automatically satisfies Lorentz equivariance regardless of the learned coefficients.

## Project Structure

```
miquel/
├── pyproject.toml              # Project metadata & dependencies (managed by uv)
├── utils.py                    # Core tensor algebra: kinematic decomposition,
│                               #   invariants, basis tensors, and CRMHD_TBNN model
├── src/
│   ├── __init__.py
│   ├── dataset.py              # HDF5 I/O, Favre filtering, SGS stress computation,
│   │                           #   PyTorch Dataset / DataLoader creation
│   ├── features.py             # Grid-level feature extraction pipeline:
│   │                           #   4th-order gradients, acceleration from Euler eq.,
│   │                           #   full TBNN feature set per snapshot
│   ├── train_apriori.py        # A priori supervised training loop with AdamW,
│   │                           #   ReduceLROnPlateau, diagnostics, and TorchScript export
│   ├── solver.py               # GPU-accelerated 2D RHD finite-volume solver:
│   │                           #   5th-order MP5 (or MUSCL) + HLL (or MUSTA/HLLC) + SSP-RK3
│   ├── evaluate_rollout.py     # A posteriori rollout benchmark: unclosed vs.
│   │                           #   TBNN-closed vs. MInIT-closed vs. filtered DNS
│   ├── minit.py                # MInIT competing SGS closure (Miravet-Tenes et al. 2023),
│   │                           #   ported to 2D SRHD -- see MINIT.md
│   └── minit_calibrate.py      # A priori calibration of MInIT against this project's DNS
├── tests/
│   ├── test_barron_loss.py     # Adaptive robust Barron loss behavior and gradients
│   ├── test_early_stopping.py  # LR plateau and early stopping state mechanics
│   ├── test_kinematics.py      # Kinematic decomposition constraint verification
│   ├── test_minit.py           # MInIT model identities, closure behavior, solver hook
│   ├── test_minit_calibrate.py # MInIT estimator recovery and a priori scoring
│   └── test_solver.py          # MP5 convergence (5th order), monotonicity, Riemann solvers, conservation
├── run_train.sh                # Shell wrapper for HTCondor training jobs
├── run_evaluate.sh             # Shell wrapper for HTCondor evaluation jobs
├── condor_train.sub            # HTCondor submit file for training
├── condor_evaluate.sub         # HTCondor submit file for evaluation
├── checkpoints/                # Saved model weights, training history, and plots
└── results/                    # Rollout evaluation outputs (spectra, field comparisons)
```

## Physics & Conventions

| Item | Convention |
|---|---|
| Metric signature | (−, +, +, +) |
| Spacetime indices | Greek μ, ν ∈ {0, 1, 2, 3} |
| Equation of state | Ideal gas, Γ = 4/3 |
| Spatial domain | 2D periodic box, v<sub>z</sub> = 0, ∂<sub>z</sub> = 0 |
| Filtering | Relativistic Favre (density-weighted by D = ρW) |
| Coarsening factor | r = 4 (1024² → 256²) |
| Gradient stencil | 4th-order central finite differences |

### Key quantities

- **Shear tensor** σ<sup>μν</sup> — traceless symmetric part of the projected velocity gradient
- **Vorticity tensor** ω<sup>μν</sup> — antisymmetric part of the projected velocity gradient
- **Four-acceleration** a<sup>μ</sup> — computed from the relativistic Euler equation (avoids temporal derivatives)
- **Expansion scalar** θ = ∇<sub>μ</sub>u<sup>μ</sup>
- **Turbulent Mach number** M<sub>t</sub> = v<sub>rms</sub> / c<sub>s</sub>

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
git clone <repository-url>
cd miquel

# Create the virtual environment and install all dependencies
uv sync
```

### Dependencies

| Package | Purpose |
|---|---|
| `torch ≥ 2.14` | Neural network, tensor operations, GPU acceleration |
| `torchvision ≥ 0.29` | PyTorch ecosystem utilities |
| `h5py ≥ 3.16` | Reading Aenus HDF5 simulation snapshots |
| `matplotlib ≥ 3.10` | Training diagnostics and evaluation plots |
| `tqdm ≥ 4.70` | Progress bars |

PyTorch is installed with CUDA 12.6 support by default (configurable in `pyproject.toml`).

## Usage

### Data Format

Input data consists of HDF5 snapshots produced by the Aenus relativistic hydrodynamics solver. Each file contains:

```
hydro/data : (1, Ny+6, Nx+6, 5)  — conserved variables [ρ, E, Sx, Sy, Sz]
thd/data   : (1, Ny+6, Nx+6, 2)  — thermodynamic variables [p, cs]
Parameters/t                       — simulation time
```

The 3-cell ghost zone on each boundary is stripped automatically on load.

### A Priori Training

Train the TBNN to predict τ<sub>SGS</sub><sup>μν</sup> from Favre-filtered DNS snapshots (see [APRIORI_TRAINING.md](file:///lustre/ific.uv.es/ml/uv098/osvaldogramaxo/miquel/APRIORI_TRAINING.md) for a comprehensive description of the training pipeline, mathematics, and diagnostics):

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run training
python -m src.train_apriori \
    --data-dir test1 \
    --coarsening-factor 4 \
    --stride 5 \
    --epochs 2500 \
    --batch-size 2048 \
    --lr 1e-3 \
    --plot-interval 10 \
    --checkpoint-dir checkpoints \
    --device cuda \
    --dtype float64
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `test1` | Directory containing HDF5 snapshots |
| `--coarsening-factor` | `4` | Spatial downsampling ratio |
| `--stride` | `5` | Temporal stride over snapshots |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `2048` | Samples per batch |
| `--lr` | `1e-3` | Initial learning rate (AdamW) |
| `--hidden-dim` | `64` | Hidden layer width |
| `--num-hidden-layers` | `8` | Number of hidden layers |
| `--plot-interval` | `10` | Epoch interval for saving diagnostic plots |
| `--dtype` | `float64` | Tensor precision (`float32` or `float64`) |

**Outputs** (saved to `checkpoints/`):
- `best_model.pt` — Best validation checkpoint (with normalization statistics)
- `final_model.pt` — Last-epoch checkpoint
- `tbnn_traced.pt` — TorchScript-traced model for deployment
- `loss_history.png` — Training/validation loss curves
- `metrics.csv` — Per-epoch metrics log
- `plots/epoch_XXXX_pred_vs_truth.png` — Spatial prediction diagnostics

### A Posteriori Rollout Evaluation

Compare the trained closure against an unclosed baseline and filtered DNS ground truth using the Aenus-matching solver:

```bash
python -m src.evaluate_rollout \
    --data-dir test1 \
    --coarsening-factor 4 \
    --checkpoint checkpoints/best_model.pt \
    --t-start 0.0 \
    --t-end 10.0 \
    --cfl 0.3 \
    --reconstruction mp5 \
    --riemann-solver hll \
    --output-dir results \
    --device cuda
```

**Key evaluation options:**

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `test1` | Directory containing HDF5 snapshots |
| `--coarsening-factor` | `4` | Spatial downsampling ratio $r$ |
| `--checkpoint` | `checkpoints/best_model.pt` | Path to trained model checkpoint |
| `--t-start` | `0.0` | Rollout start simulation time |
| `--t-end` | `10.0` | Rollout end simulation time |
| `--cfl` | `0.3` | CFL number |
| `--reconstruction` | `mp5` | Spatial reconstruction (`mp5` or `muscl`) |
| `--riemann-solver` | `hll` | Riemann solver (`hll`, `musta_hll`, or `hllc`) |
| `--output-dir` | `results` | Output directory for metrics and spectra plots |
| `--device` | `cuda` | Target device (`cuda` or `cpu`) |

This runs three simulations on the coarse (256²) grid:
1. **Unclosed** — RHD solver without any SGS model
2. **TBNN-closed** — RHD solver with the learned neural closure
3. **Filtered DNS** — Favre-filtered high-resolution reference

**Outputs** (saved to `results/`):
- `rollout_comparison.png` — Energy spectra and density field comparisons
- `metrics.json` — Relative L2 errors for density and velocity

### Running on HTCondor (Artemisa)

Submit jobs to the HTCondor batch system:

```bash
# Training job (8 CPUs, 32 GB RAM, 1 GPU)
condor_submit condor_train.sub

# Evaluation job
condor_submit condor_evaluate.sub

# Monitor logs
tail -f condor_logs/train_<ClusterId>_0.out
```

## Architecture

### CRMHD_TBNN Model

```
Input: Λ_k (15 invariants) ⊕ θ, M_t (2 thermo features)
         │
         ▼
   ┌─────────────┐
   │  Linear(17, H)  │
   │  LeakyReLU       │
   │  ×(L−1) layers   │
   │  Linear(H, H)    │
   │  LeakyReLU       │
   └────────┬────────┘
            │
     ┌──────┴──────┐
     ▼              ▼
  coeff_head     p_sgs_head
  Linear(H, 12)  Linear(H, 1)
     │              │
     ▼              ▼
  G^(n)          P_SGS
     │
     ▼
  Σ G^(n) E^(n)^{μν}  →  ΔT_SGS^{μν}
```

Default configuration: H = 64, L = 8 (~40k parameters).

### Numerical Solver

The built-in 2D RHD solver (`src/solver.py`) is formulated in conservative finite-volume form to match the numerical method of the **Aenus** code (Obergaulinger 2008):

$$\partial_t U + \partial_x F^x + \partial_y F^y = S_{\text{SGS}}$$

- **Spatial reconstruction**: 5th-order Monotonicity-Preserving (MP5) scheme (Suresh & Huynh 1997) evaluated directly on primitive variables $(\rho, v_x, v_y, p)$ with physical bounds, preserving high-order accuracy at smooth extrema while enforcing strict monotonicity bounds near discontinuities. (Optional: 2nd-order MUSCL with minmod/MC limiter; optional: conserved variable reconstruction).
- **Riemann solver**: HLL approximate Riemann solver (Harten, Lax, van Leer 1983) by default. (Optional: MUSTA-HLL multi-stage predictor-corrector, or HLLC restoring the contact discontinuity).
- **Time integration**: 3rd-order Strong Stability Preserving Runge-Kutta (SSP-RK3 / Shu & Osher 1988).
- **Wave speed estimates**: Exact multi-dimensional relativistic eigenvalues (Font et al. 1994).
- **Primitive recovery**: Robust 1D Newton-Raphson pressure solver with analytical bracket bounds and EOS consistency.
- **Boundary conditions**: Periodic in $x$ and $y$.
- **CFL number**: 0.4 (or 0.3).

## Tests

The test suite covers tensor kinematics, adaptive loss mechanics, training stability, and solver convergence:

```bash
# Run all tests
python -m pytest tests/ -v

# Individual test suites
python -m pytest tests/test_solver.py -v         # MP5 5th-order convergence, monotonicity, Riemann solvers, conservation
python -m pytest tests/test_kinematics.py -v     # σ·u=0, ω·u=0, Tr(σ)=0, a·u=0
python -m pytest tests/test_barron_loss.py -v   # Adaptive robust Barron loss & scale parameter behavior
python -m pytest tests/test_early_stopping.py -v # LR plateau and early stopping state transitions
```

## References

- Obergaulinger, M. (2008). *Astrophysical Applications of Core Collapse Supernovae and Relativistic Magnetohydrodynamics*. PhD Thesis, Technische Universität München / Max-Planck-Institut für Astrophysik.
- Suresh, A. & Huynh, H. T. (1997). *Accurate Monotonicity-Preserving Schemes with Runge-Kutta Time Stepping*. Journal of Computational Physics, 136(1), 83–110.
- Harten, A., Lax, P. D., & van Leer, B. (1983). *On Upstream Differencing and Godunov-Type Schemes for Hyperbolic Conservation Laws*. SIAM Review, 25(1), 35–61.
- Shu, C.-W. & Osher, S. (1988). *Efficient Implementation of Essentially Non-oscillatory Shock-Capturing Schemes*. Journal of Computational Physics, 77(2), 439–471.
- Ling, J., Kurzawski, A., & Templeton, J. (2016). *Reynolds averaged turbulence modelling using deep neural networks with embedded invariance*. Journal of Fluid Mechanics, 807, 155–166.
- Mignone, A. & McKinney, J. C. (2007). *Equation of state in relativistic magnetohydrodynamics: variable versus constant adiabatic index*. MNRAS, 378(3), 1118–1130.

## License

This project is developed at IFIC (Instituto de Física Corpuscular), Universitat de València.
