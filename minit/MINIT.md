# MInIT as a Competing SGS Closure

Reference for [`src/minit.py`](../src/minit.py) and [`src/minit_calibrate.py`](../src/minit_calibrate.py) —
the MInIT sub-grid model of Miravet-Tenés, Cerdá-Durán, Obergaulinger & Font (2023,
[arXiv:2308.06041](minit/2308.06041v2.pdf)) ported into this project's 2D SRHD solver and
calibrated against its own DNS, so it can be scored beside the TBNN closure.

---

## 1. What MInIT is, and what survives the port

MInIT ("MHD-Instability-induced-Turbulence") closes the turbulent **Maxwell**, **Reynolds**
and **Faraday** stresses of KHI-driven MHD turbulence, all three taken proportional to a
single evolved scalar — the sub-grid turbulent kinetic energy density $e_\text{KH}$:

$$M_{ij} = \alpha^\text{KH}_{ij} e_\text{KH}, \qquad
R_{ij} = \frac{\beta^\text{KH}_{ij}}{\rho} e_\text{KH}, \qquad
F_{ij} = \frac{\gamma^\text{KH}_{ij}}{\sqrt{\rho}} e_\text{KH}$$

$e_\text{KH}$ is grown by the resolved shear and drained by a Kolmogorov cascade:

$$\partial_t e_\text{KH} + \partial_i(v_i e_\text{KH}) = 2\gamma_\text{KH} e_\text{KH} - \frac{C\, e_\text{KH}^{3/2}}{\sqrt{\rho}\,\lambda_\text{KH}},
\qquad \gamma_\text{KH} = A_\gamma \partial_y v_x, \qquad \lambda_\text{KH} = 15.8\,\Delta_f$$

**This solver is pure SRHD.** There is no magnetic field, so the Maxwell and Faraday
closures have nothing to act on and only the Reynolds part transfers. That part is exactly
the object the TBNN predicts, so the comparison is well posed — a one-parameter algebraic
closure against a 15-invariant tensor-basis network, scored on the same target.

### 1.1 The other two closures — $\alpha^{\rm KH}_{ij}$ and $\gamma^{\rm KH}_{ij}$ (Eqs. 16a, 16c)

Eq. (16) closes *three* tensors, all proportional to the same scalar $e_\text{KH}$, with the
$\rho$ factors present only to make the coefficients dimensionless:

| Eq. | tensor | definition (Eq. 6) | coefficient |
|---|---|---|---|
| 16a | Maxwell $M_{ij}$ | $\overline{B'_i B'_j}$ | $\alpha^{\rm KH}_{ij}$ |
| 16b | Reynolds $R_{ij}$ | $\overline{v'_i v'_j}$ | $\beta^{\rm KH}_{ij}/\rho$ |
| 16c | Faraday $F_{ij}$ | $\overline{v'_i B'_j - v'_j B'_i}$ | $\gamma^{\rm KH}_{ij}/\sqrt{\rho}$ |

$\alpha$ and $\gamma$ multiply *magnetic* fluctuations. This code base solves pure SRHD, so
$B' = 0$ and $M_{ij}$, $F_{ij}$ are identically zero — not dropped for convenience, absent.
Only 16b has a target here, and it is exactly the object the TBNN predicts.

Two things in Table 2 are worth knowing before anyone refits them:

* $\sum_i\beta_{ii} = 0.74+0.64+0.61 = 1.99$ is **not a result** — it is forced. Since
  $e_\text{KH} \equiv \tfrac12\rho\sum_i R_{ii}$, the definition gives $\sum_i\beta_{ii} = 2$
  identically. `fit_beta_lsq` satisfies it exactly and for free (§3); the paper's 1.99 is a
  consistency check on its calibration procedure, not a measurement.
* $\sum_i\alpha_{ii} = 0.70+0.64+0.64 = 1.98$ is **not** forced, and says the turbulent
  magnetic energy $\tfrac12\sum_i M_{ii} \approx e_\text{KH}$ — equipartition in their boxes.
  With $\pm0.3/\pm0.18/\pm0.19$ on the diagonal the sum carries roughly $\pm0.4$, so this is
  *consistent with* equipartition rather than established by it.

**$\gamma$ is the paper's own worst result, and the reason matters here.** Its $L_2$ is
$\approx 10$ against $\approx 1$ for Maxwell and Reynolds, and the stated cause (their Sec. 4,
p. 10) is that "the time and spatial averages of the proportionality coefficients render
difficult to capture the variability and the change of sign of this tensor components."

That is precisely the premise of §8. Eq. 16c is in one respect a *better* target for a
state-dependent coefficient than 16b: §8 finds the Reynolds case capped because
$\operatorname{sign}\Pi = \operatorname{sign}(\beta_{ij}S^{ij})$ is fixed by the resolved
strain whatever the network predicts, whereas $F_{ij}$ is antisymmetric and what fails is the
sign of the coefficient itself — representable by a signed coefficient field, not by a
constant. Every published $\gamma$ is also consistent with zero within its error bars
($-0.010\pm0.017$, $-0.01\pm0.03$, $-0.008\pm0.015$), so there is little signal to transfer.

None of this is actionable without a magnetic field. If this project moves to RMHD, the
structural point is that **all three tensors share one $e_\text{KH}$**, so the coefficient
network of §8 would drive all three and only the tensor coefficients would need recalibrating.

### Mapping onto the solver's stress

The solver's SGS variable is the fluid-frame $\tau_\perp^{\mu\nu}$ (see
`dataset.compute_sgs_stress`), which carries units of energy density. Since
$\tau^{ij} = \rho R_{ij}$, the density cancels exactly:

$$\boxed{\ \tau_\perp^{ij} = \beta^\text{KH}_{ij}\, e_\text{KH}\ }$$

Time components follow from transversality $u_\mu \tau_\perp^{\mu\nu} = 0$, i.e.
$\tau^{0\nu} = v_i \tau^{i\nu}$, and are filled in so the tensor's norm is comparable to
the TBNN's and the exact DNS stress.

$e_\text{KH} \equiv \tfrac12 \sum_i \tau^{ii}$. **In 2D the sum runs over $x,y$ only** —
$T^{zz} = p\,g^{zz}$ and the box filter of $p$ is the coarse $\bar p$, so $\tau^{zz} = 0$
identically — which forces $\beta_{xx} + \beta_{yy} = 2$ against the paper's 3D
$\beta_{xx}+\beta_{yy}+\beta_{zz} = 1.99$.

### The stiff ODE has a closed form

The paper calls the turbulent-energy equation stiff and uses Strang splitting. It is a
**Bernoulli equation**: substituting $w = e^{-1/2}$ linearises it exactly to
$\dot w = -\gamma w + K/2$ with $K = C/(\sqrt{\rho}\lambda_\text{KH})$, so for $\gamma,\rho$
frozen over a step the update

$$w^{n+1} = w^n e^{-\gamma\Delta t} + \frac{K}{2}\cdot\frac{1 - e^{-\gamma\Delta t}}{\gamma}$$

is **exact, unconditionally stable and positivity preserving**. No sub-stepping, no scipy,
no stiffness, and the closure never constrains the hydrodynamic time step. The fixed point
$e_\text{sat} = (2\gamma\sqrt{\rho}\lambda_\text{KH}/C)^2$ is the paper's Eq. (15) scaling
in closed form.

---

## 2. Four transfer gaps

| Gap | Paper | Here | Consequence |
|---|---|---|---|
| Physics | Newtonian MHD (Aenus) | SRHD | $\beta$ is a Newtonian calibration |
| Dimension | 3D | 2D | **Severe** — $C e^{3/2}/\lambda$ encodes a *3D forward* cascade |
| Filter scale | $\Delta_f/L = 0.078$–$0.11$ | $0.0156$ | Table 3 already shows $C_\text{opt}$ drifting 8.8→11.0 over a narrower range |
| Components | 6 | 3 ($xx$, $yy$, $xy$) | only $\tfrac12(\beta_{xx}+\beta_{yy})$ of $e_\text{KH}$ is dynamically active |

The 2D gap means $C = 10.6$ carries **no physical authority** here, which is why the
refit is mandatory rather than optional.

---

## 3. Calibrating on our own DNS

```bash
# Full calibration. Caches the DNS reduction, so refits afterwards are instant.
python -m src.minit_calibrate --data-dir dns_khi --coarsening-factor 16 --t-lo 3.0 --t-hi 6.4
```

The window **must match the TBNN's training split** (`t <= 6.4` for
`checkpoints/best_model.pt`). Calibrating on `t in [6, 20]` lets MInIT see the TBNN's
held-out test window and makes the a-posteriori comparison meaningless.

### $\beta_{ij}$: two estimators

`fit_beta_lsq` (default) is the least-squares slope through the origin,
$\beta_{ij} = \langle\tau^{ij} e\rangle / \langle e^2\rangle$ — the minimiser of
$\sum(\tau^{ij} - \beta_{ij}e)^2$, robust where $e_\text{KH}$ is near zero, which on a
$64^2$ coarse grid is most cells outside the mixing layer. It satisfies
$\beta_{xx}+\beta_{yy} = 2$ **exactly and for free**, since $e_\text{KH}$ is itself defined
as $\tfrac12(\tau^{xx}+\tau^{yy})$.

`fit_beta_block_ratio` reproduces the paper's Sec. 4.3.2 procedure (block-average over
boxes of size $\sim\lambda_\text{KH}$, then take the ratio) for like-for-like comparison
with Table 2. It carries no trace guarantee, so its residual measures how well the
proportionality ansatz actually holds.

**Result on `dns_khi`, $t\in[3.0, 6.4]$, $r = 16$:**

| | $\beta_{xx}$ | $\beta_{yy}$ | $\beta_{xy}$ |
|---|---|---|---|
| least squares | **1.718** | **0.282** | **0.131** |
| block ratio | 1.429 | 0.571 | 0.004 |
| published (Table 2) | 0.74 | 0.64 | −0.01 |

The 2D SRHD sub-grid stress is **6:1 anisotropic**, against the paper's near-isotropic
1.16:1. Streamwise fluctuations dominate, as they must without vortex stretching.

### $(A_\gamma, C)$: a degeneracy, not an optimum

$e_\text{sat} \propto (A_\gamma/C)^2$, so in the saturated regime only the *ratio* is
identifiable. This is the paper's own Fig. 11 observation ("for a wide range of $A_\gamma$
it is always possible to find a $C$"), and it is why the paper fixes $A_\gamma = 0.14$.
The scan reports both fits every run so the choice is auditable:

```
free 2-parameter : A_gamma=2.2388  C=63.9534   L2=1.0730
A_gamma fixed    : A_gamma=0.1400  C=2.9079    L2=1.2052
published        : A_gamma=0.14    C=10.6
```

The free fit buys 12% of $L_2$ by pushing $A_\gamma$ **16× above** the Miura & Pritchett
value. It is not finding a better growth rate — it is shortening the relaxation time until
the ODE stops doing anything. Fitting that limit directly is the honest version.

### The equilibrium limit: MInIT is asking to become Smagorinsky

Setting $e_\text{KH} = e_\text{sat}$ and dropping the ODE gives an **algebraic** closure:

$$\tau^{ij} = \beta_{ij}\left(\frac{2A_\gamma}{C}\right)^2 \rho\,\lambda_\text{KH}^2\,|S|^2$$

a Smagorinsky-type stress carrying MInIT's calibrated anisotropy, with one identifiable
parameter $A_\gamma/C$. On our DNS it fits at $A_\gamma/C = 0.0333$ and **scores best**:

```
equilibrium limit : A_gamma/C = 0.03333  (C=4.2006)   L2=1.0234
```

`MInITClosure(mode="equilibrium")` implements it; it is memoryless, so it cannot drift.

---

## 4. The a-priori error decomposition

The paper reports only the free-running comparison. Splitting it into three levels is what
locates the failure:

| stage | what it adds |
|---|---|
| `oracle_e` | $\beta_{ij} e_\text{KH}^\text{DNS}$ — the **proportionality ansatz** alone |
| `one_step` | + one snapshot interval of the ODE, re-seeded from DNS each snapshot |
| `full` | + the ODE run **free** across the whole window |

**Correlation with the exact SGS stress, $t\in[3.0, 6.4]$:**

| variant | `oracle_e` | `one_step` | `full` | $\langle\Pi\rangle$ / exact |
|---|---|---|---|---|
| fitted | **0.928** | 0.451 | 0.280 | −0.15 |
| paper | 0.782 | 0.383 | 0.231 | +0.001 |
| **equilibrium** | **0.928** | **0.495** | **0.498** | +0.021 |

### What this says

1. **The proportionality ansatz is sound** — 0.93 with refit coefficients, 0.78 with the
   published ones. MInIT's core assumption survives the transfer to 2D SRHD.
2. **The evolution equation is what fails, and it fails immediately.** Correlation halves
   in a *single* $\Delta t = 0.1$ interval (0.93 → 0.45), so this is not accumulated drift
   — the rate law is wrong. $\gamma_\text{KH} \propto |\partial_y v_x|$ grows $e_\text{KH}$
   wherever there is instantaneous local shear, while the true sub-grid energy is
   transported with the vortices.
3. **Removing the ODE improves the model.** Equilibrium beats the evolving version at every
   stage, and being memoryless its `one_step` and `full` scores coincide.
4. **The exact $\Pi$ is negative** — the true SGS stress in this 2D flow *backscatters* on
   average, consistent with a 2D inverse cascade. MInIT reproduces at most a few percent of
   its magnitude, and the `fitted` variant gets the sign wrong.

Point 4 is the one that governs a-posteriori relevance: a closure supplying ~2% of the true
energy transfer cannot move a trajectory much, whatever its correlation.

---

## 5. Running it a-posteriori

```bash
python -m src.evaluate_rollout \
    --data-dir dns_khi --coarsening-factor 16 \
    --t-start 6.5 --t-end 8.0 \
    --checkpoint checkpoints/best_model.pt \
    --with-minit all --with-unclosed --sgs-coupling flux
```

`--with-minit` takes `fitted`, `paper`, `equilibrium`, `both` (= fitted+paper) or `all`,
and is repeatable. The gap between `fitted` and `paper` separates the model from its
calibration.

| flag | default | why |
|---|---|---|
| `--minit-seed` | `dns` | $e_\text{KH} = 0$ is a **fixed point**, so MInIT needs a seed or it is inert and silently reproduces the unclosed control. `dns` takes the exact filtered-DNS sub-grid energy at `t_start` — one snapshot of ground truth, the same as the oracle gets. `uniform` is self-contained. **These are different experiments and belong in separate rows.** |
| `--minit-floor-fraction` | `0` | holds $e_\text{KH}$ at a fraction of its local $e_\text{sat}$, so a decayed region can restart |
| `--minit-advect` | off | the paper drops $\partial_i(v_i e_\text{KH})$ on periodicity grounds. That holds along $x$ here but **not** across the shear layer, where roll-up transports turbulent energy in $y$ |
| `--minit-shear-mode` | `dvxdy` | Eq. (23) literally; `strain` is the rotationally-invariant $\sqrt{2S_{ij}S_{ij}}$, normalised to agree for pure shear so $A_\gamma$ keeps its meaning |

### How it couples to the solver

MInIT is the project's first **stateful** closure. The solver's protocol is stateless —
`f(prims) -> (tau_perp, alpha)`, called once per RK stage and again per predictor-corrector
recovery, ~9–12 times a step — so $e_\text{KH}$ is frozen within a step and advanced once
per completed step through an `advance(prims, dt, t)` hook that `RHDSolver.evolve` calls on
any closure exposing one. That is operator splitting between turbulence model and
hydrodynamics, which is what the paper does for the same reason. Stateless closures are
untouched; the hook is opt-in.

MInIT supplies no transition sensor — $\alpha \equiv 1$. Its own $e_\text{KH}$ is the
activation, growing only where there is resolved shear to drive it.

### Result on `dns_khi`, $t: 6.5 \to 8.0$, $r = 16$, all runs DNS-seeded

```bash
python -m src.evaluate_rollout --data-dir dns_khi --coarsening-factor 16 \
    --t-start 6.5 --t-end 8.0 --checkpoint checkpoints/best_model.pt \
    --with-minit all --with-unclosed --sgs-coupling flux
```

| run | log-spectral dist. | high-$k$ $E/E_\text{DNS}$ | mid-$k$ | total KE |
|---|---|---|---|---|
| Persistence (frozen field) | **0.137** | 1.182 | 1.941 | 1.257 |
| Unclosed (no SGS) | 0.522 | 0.340 | 0.872 | 0.992 |
| MInIT (paper) | 0.525 | 0.338 | 0.874 | 0.992 |
| MInIT (fitted) | 0.530 | 0.335 | 0.882 | 0.992 |
| MInIT (equilibrium) | 0.518 | 0.350 | 0.877 | 0.991 |
| TBNN | 0.556 | 0.327 | 0.865 | 0.991 |
| **Exact DNS oracle** | **0.566** | 0.321 | 0.883 | 0.992 |

No run diverged.

Both figures build their layout from the runs that actually executed, so every MInIT
variant appears in `rollout_comparison.png` (density panel, spectrum curve, band-ratio bar)
and in `rollout_comparison.gif` (density panel plus its own `|run - DNS|` error map, on a
shared error scale). Add `--make-anim` for the GIF. The six error maps in the animation
being visually indistinguishable is the same result the table states, seen directly.

### Read this table carefully before ranking anything

**The exact-DNS oracle is the worst run in the table** — 0.566 against 0.522 for doing
nothing at all. Supplying the *true* SGS stress at every step makes the long rollout worse
than leaving the term out. (Before the §5.1 fix the oracle scored 0.516 and looked best;
that was an artefact of its stress being frozen at `t_start`, which made it a mild, smooth,
time-independent perturbation. Driving it correctly makes it worse, not better.)

That is the finding, and it is about the *coupling*, not the models. Over ~400 steps the
cell-centred stress and its flux divergence inject grid-scale content the discretisation
handles badly — a symmetric interpolation of a cell-centred quantity has exactly zero
response at the 2Δ mode, so the closure cannot act where it most needs to and accumulates
error where it should not.

**Nothing beats persistence.** A frozen initial field scores 0.137 against 0.518 for the
best dynamic run, because all of them shed high-$k$ energy over the window while the frozen
field keeps it. None of these runs has earned its compute on this metric.

What this table does **not** license: any claim that MInIT beats or loses to the TBNN, or
that refitting helped a-posteriori. The non-oracle runs sit within ~2% of each other while
the oracle — the ceiling — is *below* them all, so there is no scale against which to rank.
Use the short-horizon test in §5.2 instead, where the oracle does separate.

---

## 5.1 A bug that invalidated every oracle row

`RHDSolver.evolve` synchronised time-dependent SGS objects by calling `set_time` — but
only on the legacy `sgs_model` argument, never on `sgs_closure`. `ExactDNSSGSClosure`
interpolates a precomputed stress between DNS snapshots and depends on that call, so on
the conservative-flux path **the exact-DNS oracle ran an entire rollout with its stress
frozen at the first snapshot**. Verified directly: after evolving 6.5 → 7.0 the closure's
`current_t` was still 6.5.

Nothing in the output revealed it. The oracle produced a plausible trajectory and
plausible metrics, and it quietly crippled the one baseline every other result is judged
against — including the conclusion that "the oracle does not separate from unclosed, so
nothing can be ranked". Fixed in `src/solver.py` (both coupling paths are now
synchronised) with regression tests in `tests/test_minit.py::TestClosureTimeSync`.

**Any result produced with `--sgs-coupling flux` and the exact-DNS oracle before this fix
should be rerun.**

## 5.2 The short-horizon test — `src.evaluate_shorthorizon`

The long rollout conflates two failures: over ~400 steps a chaotic trajectory decorrelates
and pointwise norms stop measuring closure quality, *and* the SGS term is a ~6%
perturbation to the right-hand side. The first is avoidable. Starting every closure from
the same filtered-DNS state and integrating one snapshot interval (~27 steps) keeps the
runs correlated, so a 6% difference in the RHS is directly visible.

The metric is the error in the *increment*, since over a short interval the state barely
moves:

$$\varepsilon = \frac{\lVert \Delta U_\text{model} - \Delta U_\text{DNS}\rVert}{\lVert \Delta U_\text{DNS}\rVert},
\qquad \Delta U = U(t+\Delta t) - U_\text{DNS}(t)$$

so 0 is a perfect increment and 1.0 is the score for predicting no change. The unclosed
solver scores ~0.16, i.e. it already gets 84% of the increment right — that 16% is all any
closure can compete for.

```bash
python -m src.evaluate_shorthorizon --t-lo 6.5 --t-hi 8.0
python -m src.evaluate_shorthorizon --t-lo 14.0 --t-hi 20.0 --skip-tbnn
```

**Early window, t = 6.5–8.0, 15 starts** (subgrid = 11–18% of fluctuation energy):

| run | mean ε | skill vs unclosed | beats unclosed |
|---|---|---|---|
| frozen (no change) | 1.0000 | −506% | 0/15 |
| unclosed | 0.1656 | — | — |
| MInIT (fitted) | 0.1691 | −2.17% | 2/15 |
| MInIT (equilibrium) | 0.1656 | −0.02% | 10/15 |
| MInIT (paper) | 0.1635 | +1.19% | 9/15 |
| **Exact DNS oracle** | 0.1630 | **+1.52%** | **14/15** |
| TBNN | 0.1629 | +1.59% | 12/15 |

**Late window, t = 14–20, 20 starts** (subgrid = 29–35% of fluctuation energy, 3× more):

| run | mean ε | skill vs unclosed | beats unclosed |
|---|---|---|---|
| unclosed | 0.1568 | — | — |
| MInIT (equilibrium) | 0.1569 | −0.04% | 1/20 |
| MInIT (fitted) | 0.1568 | +0.01% | 12/20 |
| MInIT (paper) | 0.1564 | +0.23% | 20/20 |
| **Exact DNS oracle** | 0.1575 | **−0.43%** | **3/20** |

### What these say

1. **At this horizon the oracle does separate — in the early window.** +1.52% on 14/15
   starts is small but consistent and paired. So the metric works and the long rollout's
   failure to rank was decorrelation, not a dead configuration.
2. **The ceiling is ~1.5%.** Perfect SGS stress buys 1.5% of the increment error. Read
   every other row as a fraction of that, not as an absolute gain.
3. **More turbulence does not help — it hurts.** The late window has 3× the subgrid
   energy fraction and the oracle goes *negative* (−0.43%, 3/20). This matches the direct
   force measurement in `scratch/sgs_magnitude_scan.py`: at t=20 the SGS contribution to
   the RHS is 2.6%, its *lowest* value, while at t=8 it is 12%. The subgrid *energy*
   fraction and the SGS *force* peak at opposite ends of the run, and it is the force that
   moves a trajectory. **Do not chase a more turbulent DNS on the strength of the energy
   fraction.**
4. **Refitting made MInIT worse a-posteriori.** `fitted` scores best a-priori (corr 0.93
   vs 0.78) and worst here (−2.17%). Its refit `C = 2.9` against 10.6 makes `e_KH` ~4.6×
   larger, and the stronger stress overshoots. A-priori correlation does not imply
   a-posteriori skill — the two rankings are inverted here.
5. **Treat MInIT (paper)'s consistency with suspicion.** It beats the control on 20/20
   late starts while the *true* stress loses on 17/20. A closure that helps where ground
   truth hurts is most likely acting as artificial viscosity — its small, near-isotropic,
   smooth stress damping grid-scale noise — rather than supplying the physics. That is a
   numerical benefit, not a validated closure, and it should not be reported as one
   without separating the two.

---

## 6. Gotchas

- **A zero seed makes MInIT a silent no-op.** `tests/test_minit.py` pins both directions:
  a zero-seeded run must reproduce the unclosed solver bit for bit, and a seeded one must
  not. If a "MInIT" row ever matches `unclosed` exactly, the seed was lost.
- **Calibrate on the training split.** See §3.
- **Eq. (25) diverges when the model collapses.** `l2_norm_paper` floors the model
  denominator at $10^{-12}$ of the measured one, capping the penalty at a finite,
  scale-invariant $\sim 7\times10^5$ instead of letting it land wherever the float epsilon
  puts it ($\sim 10^{150}$).
- **Backscatter clipping is meaningful here.** MInIT's diagonal-dominant stress is not
  automatically dissipative: with an incompressible resolved field the transfer is
  $e_\text{KH}[(\beta_{xx}-\beta_{yy})S_{xx} + 2\beta_{xy}S_{xy}]$, whose sign varies.
- **Time-dependent closures need `set_time` on both coupling paths.** See §5.1. If you
  add another closure that interpolates precomputed data, check it is being advanced.
- **Free-fit boundary pinning is reported.** An optimum on the edge of `--A-range` or
  `--C-range` records where the scan was cut, not where the objective turns over.

---

## 7. Files

| path | role |
|---|---|
| [`src/minit.py`](../src/minit.py) | the closure: coefficients, growth rate, exact $e_\text{KH}$ update, advection, stress assembly, `MInITClosure` |
| [`src/minit_calibrate.py`](../src/minit_calibrate.py) | DNS reduction (cached), $\beta$ / $(A_\gamma, C)$ / equilibrium fits, a-priori scoring, figures |
| [`tests/test_minit.py`](../tests/test_minit.py) | model identities, closure behaviour, solver integration |
| [`tests/test_minit_calibrate.py`](../tests/test_minit_calibrate.py) | estimator recovery, Eq. (25) behaviour, the equilibrium identity, figure coverage |
| [`src/evaluate_shorthorizon.py`](../src/evaluate_shorthorizon.py) | short-horizon increment test (§5.2) |
| [`scratch/sgs_magnitude_scan.py`](../scratch/sgs_magnitude_scan.py) | how big the SGS term is, vs coarsening factor and time |
| [`minit/FEASIBILITY_learned_coefficients.md`](FEASIBILITY_learned_coefficients.md) | **§8**: can the coefficients be functions of the local state? Measured ceiling, ladder, a-posteriori gate |
| [`scratch/minit_learned_coeff_feasibility.py`](../scratch/minit_learned_coeff_feasibility.py) | that ladder, split within `dns_khi` |
| [`scratch/minit_crossrealisation.py`](../scratch/minit_crossrealisation.py) | the same ladder, `dns_khi` → `dns_khi_s1` |
| `results/minit/` | `minit_coefficients.json`, `minit_coefficients_equilibrium.json`, `minit_apriori.json`, `learned_coeff_*.json`, figures |

---

## 8. State-dependent coefficients

Full report: [`minit/FEASIBILITY_learned_coefficients.md`](FEASIBILITY_learned_coefficients.md).

The §4 decomposition says the ansatz is sound and the coefficient is what fails, so the
coefficient is the thing to make state-dependent. Measured on `dns_khi` → `dns_khi_s1`
(unseen seed), a ~6 k-parameter network predicting a single scalar field recovers **75–80 %
of the gap** between the global-coefficient model and the `oracle_e` ceiling, on amplitude,
correlation, $\Pi$ and high-$k$ content alike. It lifts $\Pi$ from 0.014 to 0.548 within
`dns_khi`, which removes the "2 % of the true transfer" objection in §4.

Two results constrain the design:

* A **per-snapshot global** coefficient — a Germano-style dynamic procedure — buys almost
  nothing (corr 0.522 → 0.538). The missing structure is spatial, so the classical cheap
  alternative is not adequate here.
* The **TBNN's correlation (0.832) already exceeds the `oracle_e` ceiling (0.770)**. With
  $\beta_{ij}$ fixed and $e\ge0$, $\operatorname{sign}\Pi = \operatorname{sign}(\beta_{ij}S^{ij})$,
  capping pointwise $\Pi$-sign agreement at 76.8 % whatever the network does. A
  state-dependent-coefficient MInIT is a much better MInIT, **not** a better TBNN.

**Built and measured (2026-09-23).** [`src/minit_learned.py`](../src/minit_learned.py)
implements it as a stateless solver closure (12,193 parameters, 5x5 receptive field);
[`src/train_minit_coeff.py`](../src/train_minit_coeff.py) fits it. The §5 precedent —
refitting improved MInIT a-priori and made it *worse* a-posteriori — **did not repeat**:

| run (short horizon, $t=6.5$–$8.0$, 15 starts) | mean $\varepsilon$ | skill | wins |
|---|---|---|---|
| MInIT (fitted / paper), DNS-seeded | 0.1635 | +1.16 / +1.19% | 10 / 9 of 15 |
| **learned coefficient, no seed** | **0.1623** | **+1.92%** | **14/15** |
| TBNN | 0.1620 | +2.06% | 13/15 |
| Exact DNS oracle | 0.1630 | +1.52% | 14/15 |

The learned rows read *only resolved state*, while every `minit_*` row is seeded from the
exact filtered-DNS $e_\text{KH}$ at $t_0$. Exceeding the oracle is not beating physics: the
oracle supplies the true stress, which is not the optimal correction for a discretised
solver (§5.2 point 3). Full results in the feasibility report §10.
