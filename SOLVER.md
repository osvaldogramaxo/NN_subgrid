# The RHD Solver and How to Drive It

Reference for [`src/solver.py`](src/solver.py) — the 2D relativistic hydrodynamics
finite-volume solver — and for every command-line entry point in this project.

---

## 1. What the solver is

A GPU-capable 2D special-relativistic hydrodynamics solver written entirely in PyTorch,
built to match the numerical method of the **Aenus** code (Obergaulinger 2008) so that a
coarse run can be compared meaningfully against filtered Aenus DNS.

It solves

$$\partial_t U + \partial_x F^x + \partial_y F^y = S_\text{SGS}$$

with the conserved vector

| symbol | definition | meaning |
|---|---|---|
| $D$ | $\rho W$ | relativistic mass density |
| $S_i$ | $\rho h W^2 v_i$ | momentum density |
| $\tau$ | $\rho h W^2 - p - D$ | energy density **minus** rest mass |

$W = 1/\sqrt{1-v^2}$ is the Lorentz factor, $h = 1 + \varepsilon + p/\rho$ the specific
enthalpy, and the equation of state is an ideal gas with $\Gamma = 4/3$.

> **The $\tau$ convention matters.** Several published schemes (notably the Mignone & Bodo
> HLLC derivation) are written in the *total* energy $E = \tau + D$. `hllc_flux` converts
> internally and converts back; if you add a scheme, check which convention it assumes.

### Numerical method

| Stage | Method |
|---|---|
| Spatial reconstruction | 5th-order MP5 (Suresh & Huynh 1997), optional 2nd-order MUSCL with MC limiter |
| Riemann solver | HLL (default), HLLC, or MUSTA-HLL |
| Time integration | 3rd-order SSP Runge-Kutta (Shu & Osher 1988) |
| Wave speeds | Exact multi-dimensional relativistic eigenvalues (Font et al. 1994) |
| Primitive recovery | 1D Newton-Raphson on pressure with an analytic derivative |
| Boundaries | **Periodic only**, in both directions |

Reconstruction acts on primitive variables $(\rho, v_x, v_y, p)$ by default, matching
Aenus; pass `reconstruct_primitives=False` to reconstruct conserved variables instead.

### Boundaries are periodic, always

Every stencil uses `torch.roll`. There is no ghost-cell or boundary-condition machinery.
This is correct for the Kelvin-Helmholtz problem the project targets, but it means the
`__main__` block's Sod-like initial data is **not** an analytic verification — the
wrap-around creates a second Riemann problem that has interacted with the first well
before the final time. Use `pytest tests/test_solver.py` for actual verification.

---

## 2. Python API

### Typical use

```python
from src.solver import RHDSolver

solver = RHDSolver(
    Ny=256, Nx=256,
    x_range=(0.0, 1.0), y_range=(0.0, 1.0),
    cfl=0.3,
    reconstruction="mp5",        # or "muscl"
    riemann_solver="hll",        # or "hllc", "musta_hll"
    dtype=torch.float64,
    device=torch.device("cuda"),
)
solver.initialize(rho, vx, vy, p)     # each (Ny, Nx)
solver.t = t_start
solver.evolve(t_final, save_interval=0.1)
prims = solver.get_primitives()       # {"rho", "vx", "vy", "p"}
```

### `RHDSolver` methods

| Method | Purpose |
|---|---|
| `initialize(rho, vx, vy, p)` | Set the state from primitives. **Warns and rescales** if any cell has $v^2 > 0.98$ — the evolved state then no longer matches your input. |
| `step(dt=None, sgs_source=…, sgs_model=…, sgs_closure=…)` | One SSP-RK3 step. `dt=None` uses the CFL condition. |
| `evolve(t_final, …)` | Step to `t_final` with a progress bar; raises `FloatingPointError` on non-finite state. |
| `get_primitives()` | Current state as a dict. |
| `get_history()` / `clear_history()` | Recorded trajectory (requires `save_interval`). |

### The three SGS coupling paths

`step()` accepts exactly one of these:

- **`sgs_source`** — a static $(N_y, N_x, 4)$ source tensor, reused across all RK stages.
  Legacy; does not preserve the SSP property.
- **`sgs_model`** — a callable `f(prims) -> source`, re-evaluated every sub-stage. Legacy
  but SSP-correct.
- **`sgs_closure`** — a callable `f(prims) -> (tau_perp, alpha)` returning a fluid-frame
  $(N_y, N_x, 4, 4)$ stress and a gating field. **Preferred.** Uses the conservative
  viscous-flux formulation, which evolves $\tilde U = U + \delta U$ so that the
  $\partial_t \tau_\perp^{0\nu}$ term is accounted for rather than dropped.

Recovering $U$ from $\tilde U$ is implicit, since $\delta U$ depends on the state being
solved for. It is done by a fixed-point iteration (`_SGS_RECOVERY_ITERS = 2`), which
contracts by a factor ≈0.006 per iteration; one iteration is not enough and drags the
scheme to ~1st order.

### Useful standalone functions

| Function | Purpose |
|---|---|
| `primitives_to_conserved` / `conserved_to_primitives` | State conversion. C2P failures are **not** laundered — non-finite values propagate so `evolve()` raises. |
| `compute_dt(prims, dx, dy, cfl)` | CFL-limited timestep. |
| `compute_wave_speeds` | Relativistic characteristic speeds. |
| `mp5_reconstruct` / `muscl_reconstruct` | Interface reconstruction. |
| `hll_flux` / `hllc_flux` / `musta_hll_flux` | Riemann solvers. |
| `compute_sgs_viscous_fluxes` | Maps $\tau_\perp^{\mu\nu}$ to $(\delta U, \delta F^x, \delta F^y)$. |
| `clip_sgs_tensor` | Realizability clipping, $\|\tau\| \le \kappa p$. |
| `clip_backscatter` | Suppresses local $\tau\!:\!S > 0$ (see below). |
| `compute_transition_sensor` | SGS gating field $\alpha \in [0,1]$. |

### Two things that are easy to get wrong

**The transition sensor does not detect transition.** `compute_transition_sensor` defaults
to `mode="none"` ($\alpha \equiv 1$). The `"enstrophy"` mode computes
$|\omega|^2/(|\omega|^2+|S|^2)$, which separates rotation- from strain-dominated regions —
it is *not* a laminar/turbulent detector. For any pure shear $|\omega|^2 = |S|^2$, so
$\alpha \equiv 1/2$ throughout a laminar shear layer. On real filtered DNS at $t=5$ it
averages 0.174, i.e. it silently damps the closure ~5×. Ground-truth (exact-DNS) runs must
use `"none"`: rescaling the true stress makes the reference meaningless.

**The SGS flux divergence cannot see the grid-scale mode.** The viscous flux uses a
4th-order face interpolation, whose response relative to the true divergence is:

| wavelength | 16Δ | 8Δ | 4Δ | 3Δ | 2Δ |
|---|---|---|---|---|---|
| response | 0.99 | 0.97 | 0.80 | 0.57 | **0.00** |

This is structural: *any* conservative flux difference built from a symmetric interpolation
of cell-centred values annihilates the 2Δ mode. The ideal flux escapes it only because the
HLL dissipation term is a difference across the face. Consequence: at the very smallest
scales your numerics, not your closure, set the dissipation.

---

## 3. Commands

### 3.1 A-posteriori rollout — `src.evaluate_rollout`

Runs coarse-grid solvers and compares them against filtered DNS.

```bash
python -m src.evaluate_rollout \
    --data-dir test1 --coarsening-factor 4 \
    --checkpoint checkpoints/best_model.pt \
    --t-start 5.0 --t-end 10.0 \
    --riemann-solver hllc --with-unclosed \
    --output-dir results --device cuda
```

| Argument | Default | Meaning |
|---|---|---|
| `--data-dir` | `test1` | Directory of Aenus HDF5 snapshots |
| `--coarsening-factor` | `4` | Spatial downsampling ratio $r$ (1024² → 256²) |
| `--checkpoint` | `checkpoints/best_model.pt` | Trained TBNN. If absent, the TBNN run is skipped |
| `--t-start`, `--t-end` | `0.0`, `10.0` | Rollout window. **Both must land on real snapshot times** (they are on integers here); ≥2 snapshots must fall in range |
| `--cfl` | `0.3` | CFL number |
| `--reconstruction` | `mp5` | `mp5` or `muscl` |
| `--riemann-solver` | `hll` | `hll`, `hllc`, or `musta_hll` |
| `--sgs-coupling` | `flux` | `flux` (conservative, preferred) or `source` (legacy) |
| `--skip-exact` | off | Skip the exact-DNS oracle run |
| `--with-unclosed` | off | **Also run a no-SGS control.** Strongly recommended |
| `--backscatter-clip` | none | `zero` or `project` — see below. Requires `--sgs-coupling flux` |
| `--save-history` | off | Write `rollout_trajectory.pt` (can be GB-scale) |
| `--make-anim` | off | Write `rollout_comparison.gif` |
| `--anim-interval` | `0.1` | Simulation time between animation frames |
| `--anim-fps` | `10` | Animation playback rate |
| `--output-dir` | `results` | Destination for figures and `metrics.json` |
| `--device` | `cuda` | `cuda` or `cpu` |

**Outputs:** `rollout_comparison.png` (fields, spectrum, band-ratio chart),
`metrics.json`, plus the optional GIF and trajectory.

#### Metrics: why there is no L2

Pointwise relative L2 is **deliberately not reported**. On a chaotic trajectory it measures
phase decorrelation, not closure quality. Measured on this project's data, a field with 5%
grid-scale noise added scores *better* by rel L2 than a frozen initial field (0.032 vs
0.176) while being visibly ruined. What is reported instead:

| Metric | Perfect | Reads as |
|---|---|---|
| `log_spectral_distance` | 0 | Mean $\lvert\log_{10}(E/E_\text{ref})\rvert$ over resolved $k$. Headline number |
| `band_ratio_low/mid/high` | 1 | Energy ratio per band. `high` is the most diagnostic — grid-scale noise shows here first |
| `e_ratio_total` | 1 | Total resolved kinetic energy ratio |

A **persistence baseline** (freeze the initial field, compare to the reference) is always
computed. A run that cannot beat persistence has not earned its compute.

#### `--backscatter-clip`

The resolved kinetic energy gains energy at rate $\Pi = \tau^{ij}S_{ij}$, so cells with
$\Pi > 0$ inject energy — backscatter. This is physical, and in 2D turbulence the inverse
cascade arguably makes it the dominant direction. But it is unbounded here, and an
unconstrained learned closure can use it to drive a grid-scale feedback instability.

- `zero` — null $\tau$ entirely wherever $\Pi > 0$. Strictest; clearest diagnostic.
- `project` — subtract only the strain-aligned component responsible for the backscatter,
  leaving everything orthogonal intact.

Use it as a diagnostic first: if the high-$k$ pile-up disappears, the instability is
backscatter-driven.

### 3.2 A-priori training — `src.train_apriori`

```bash
python -m src.train_apriori \
    --data-dir test1 --coarsening-factor 4 --stride 5 \
    --epochs 2500 --batch-size 2048 --lr 1e-3 \
    --device cuda --dtype float64
```

| Argument | Default | Meaning |
|---|---|---|
| `--data-dir` | `test1` | Snapshot directory |
| `--coarsening-factor` | `8` | Downsampling ratio (note: rollout defaults to 4) |
| `--stride` | `5` | Temporal stride over snapshots |
| `--epochs` | `50` | Training epochs |
| `--batch-size` | `2048` | Samples per batch |
| `--lr` | `1e-4` | Initial AdamW learning rate |
| `--hidden-dim` | `64` | Hidden layer width |
| `--num-hidden-layers` | `6` | Hidden layer count |
| `--loss-alpha` | `0.5` | Weight on dissipation-rate loss $L_P$ |
| `--loss-beta` | `0.1` | Weight on spectral cascade loss |
| `--loss-gamma` | `5.0` | Weight on orthogonality constraint |
| `--loss-delta` | `2.0` | Weight on Weak Energy Condition loss |
| `--lambda-stab` | `10.0` | Anti-dissipation stability barrier inside $L_P$ |
| `--spectral-snapshots` | `4` | Snapshots per epoch for the spectral loss |
| `--lr-patience` | `25` | `ReduceLROnPlateau` patience (epochs) |
| `--patience` | `2 × lr-patience` | Early-stopping patience; ≤0 disables |
| `--min-delta` | `0.0` | Minimum improvement counted as progress |
| `--no-restore-best` | off | Skip restoring best weights before export |
| `--plot-interval` | `10` | Epochs between diagnostic plots |
| `--checkpoint-dir` | `checkpoints` | Output directory |
| `--device` | `cpu` | `cuda` or `cpu` |
| `--dtype` | `float64` | `float32` or `float64` |
| `--resume [PATH]` | none | Bare flag resumes from `latest_model.pt`/`best_model.pt`; with a path, from that file |
| `--resume-from PATH` | none | Explicit checkpoint path |
| `--reset-lr` | off | On resume, use `--lr` instead of the checkpoint's saved LR |

**Outputs** in `--checkpoint-dir`: `best_model.pt`, `final_model.pt`, `tbnn_traced.pt`
(TorchScript), `loss_history.png`, `metrics.csv`, `plots/epoch_*.png`.

A-priori relative L2 **is** reported during training, and that is correct: it compares
$\tau_\text{pred}$ against $\tau_\text{exact}$ at the same instant, which is a sound
regression metric. Only the a-posteriori trajectory ranking was removed.

### 3.3 Divergence diagnostics — `scratch/diagnose_divergence.py`

For when a rollout blows up. Records per-step scalars for the whole run and keeps a rolling
buffer of full states so the approach to failure can be replayed.

```bash
python scratch/diagnose_divergence.py --t-start 5.0 --t-end 10.0 --sensor none
```

Key arguments: `--sensor`, `--clip-kappa`, `--cfl`, `--reconstruction`,
`--riemann-solver`, `--ring` (full states retained, default 120), `--every` (print
interval), `--out`. On failure it names the offending cells, their state on the previous
step, and local minima of $\rho$ and $p$, then dumps everything to a `.pt`.

### 3.4 Shell wrappers and HTCondor

Both wrappers take **positional** arguments; extra flags pass through after the fixed ones.

```bash
# run_evaluate.sh <data_dir> <coarsening> <checkpoint> <t_start> <t_end> <device> <cfl> <recon> <riemann> [extra flags...]
./run_evaluate.sh test1 4 checkpoints/best_model.pt 5.0 10.0 cuda 0.3 mp5 hllc --with-unclosed --make-anim

# run_train.sh <data_dir> <coarsening> <stride> <epochs> <batch_size> <lr> <device> <plot_interval> <patience>
./run_train.sh test1 4 5 2500 2048 1e-3 cuda 10
```

Submit to the batch system with:

```bash
condor_submit condor_train.sub       # 8 CPUs, 32 GB, 1 GPU
condor_submit condor_evaluate.sub
tail -f condor_logs/eval_<ClusterId>_0.err
```

Edit the `arguments = "..."` line in the `.sub` file to change the run.

---

## 4. Running on this cluster

- GPUs are reached with `gpurun <command>`, **capped at 5 minutes**, in Exclusive Process
  mode (one job at a time). Anything longer must go through HTCondor.
- Always run Python **unbuffered** (`python -u`) under `gpurun`. If the cap kills the job,
  buffered stdout is lost and you get an empty log with exit code 0.
- PyTorch lives only in `./.venv`, not the system Python, and `gpurun` does not reliably
  forward `PYTHONPATH`. Either use `PYTHONPATH=. ./.venv/bin/python …` or insert the repo
  root on `sys.path` inside the script.

A full `t=5→10` TBNN rollout is roughly 1300 steps at ~7 network evaluations per step —
comfortably past the 5-minute cap. Use Condor.

---

## 5. Tests

```bash
python -m pytest tests/ -v                    # 46 tests
python -m pytest tests/test_solver.py -v      # solver only
```

`tests/test_solver.py` covers MP5 5th-order convergence and monotonicity, all three Riemann
solvers, conservation, the C2P Newton derivative against finite differences, C2P round-trip
over Lorentz factors 1–50, non-laundering of failures, exact HLLC contact and shear
resolution, SGS-coupled conservation and temporal order, the SGS face interpolation, and
backscatter clipping.

---

## 6. Known limitations

| Limitation | Impact |
|---|---|
| Grid-scale (2Δ) blindness of the SGS divergence | Numerics, not the closure, set dissipation at the smallest scales. Fixing it requires evaluating the closure at faces, which the precomputed exact-DNS closure cannot do |
| Periodic boundaries only | No shock-tube or inflow/outflow problems without new boundary machinery |
| Two corrupt DNS snapshots | `t=10.0` (321 cells) and `t=13.0` (71 cells) contain superluminal velocities. `read_snapshot` detects and repairs them with a warning |
| Heavy-tailed SGS targets | $\tau \sim W^4$, so a few hundred high-Lorentz-factor cells carry targets ~4 orders above the median. Decide explicitly whether to train on, exclude, or reweight them |
| SGS energy transfer sign | $\langle\tau\!:\!S\rangle$ measures positive on this data. Physically admissible in 2D, but not independently confirmed against the filtered equations |

---

## References

- Obergaulinger, M. (2008). *Astrophysical Applications of Core Collapse Supernovae and Relativistic Magnetohydrodynamics*. PhD thesis, TU München / MPA.
- Suresh, A. & Huynh, H. T. (1997). *Accurate Monotonicity-Preserving Schemes with Runge-Kutta Time Stepping*. JCP 136(1), 83–110.
- Harten, A., Lax, P. D. & van Leer, B. (1983). *On Upstream Differencing and Godunov-Type Schemes for Hyperbolic Conservation Laws*. SIAM Review 25(1), 35–61.
- Mignone, A. & Bodo, G. (2005). *An HLLC Riemann solver for relativistic flows*. MNRAS 364(1), 126–136.
- Shu, C.-W. & Osher, S. (1988). *Efficient Implementation of Essentially Non-oscillatory Shock-Capturing Schemes*. JCP 77(2), 439–471.
- Font, J. A. et al. (1994). *Multidimensional relativistic hydrodynamics: characteristic fields and modern high-resolution shock-capturing schemes*. A&A 282, 304.
