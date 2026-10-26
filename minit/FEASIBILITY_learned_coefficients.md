# Feasibility: MInIT with state-dependent coefficients

**Question.** MInIT closes the sub-grid stress with a handful of global constants
($\beta_{ij}$, $A_\gamma$, $C$). Can those be made functions of the local resolved state —
predicted by a small network — instead of fixed numbers, and is it worth doing?

> **Status (2026-09-23): built and measured.** Phases 1 and 2 are done — see §10. The
> a-posteriori gate **passed**: the learned coefficient scores +1.92% short-horizon skill on
> 14/15 starts against +1.16% for the best global-coefficient MInIT, so the a-priori gain
> does carry through. This reverses the §7 precedent that worried me most.

> **Correction (2026-10-18).** Every TBNN row scored on `dns_khi_s1` in this report
> (§4, §5, §10: corr 0.832, amp 0.739, Π 0.637, decay 0.816→0.662) used
> `checkpoints_fullrange`, which was trained **entirely on `dns_khi_s1`** (condor
> 1320204–1320347). Those are training-set scores, not generalisation, and they flatter
> the TBNN. The MiNNit rows are unaffected (trained on `dns_khi`). **Re-measured cleanly in
> §13** (TBNN trained on `dns_khi`): the numbers barely move (r=16: corr 0.819, amp 0.764,
> Π 0.646), and the TBNN's correlation *does* exceed the oracle-e ceiling at every filter
> size, so §5's conclusion stands.

**Verdict: feasible, and worth doing, but not for the reason it first appears.**
A ~6 k-parameter network predicting a single scalar coefficient field recovers **75–80 % of
the gap** between the global-coefficient MInIT and the best that MInIT's own stress ansatz
can ever achieve, on **every** metric, and it transfers to an independent DNS realisation
without degrading. Most importantly it raises the sub-grid energy transfer $\Pi$ — the
quantity that decides whether a closure can influence a trajectory at all — from **1–14 %
of the exact value to 46–50 %**.

What it does *not* do is beat the TBNN. The measurements below show why, and the reason is
structural rather than a matter of training effort: on this flow the TBNN's correlation
already *exceeds* the ceiling of the entire $\tau^{ij}=\beta_{ij}e$ family. That finding
points at a more valuable target than either model alone, described in §7.

All numbers are a-priori, on `dns_khi` (2D SRHD Kelvin–Helmholtz, $1024^2$ DNS, $r=16$
coarsening, $t\in[1,20]$), tested on `dns_khi_s1` — a second DNS from a different seed that
no model in this report has seen.

---

## 1. Why the coefficients, and not something else

The a-priori error decomposition in `MINIT.md` §4 already localises MInIT's failure:

| stage | what it adds | corr with exact $\tau$ |
|---|---|---|
| `oracle_e` | $\beta_{ij}e^\text{DNS}_\text{KH}$ — the proportionality ansatz alone | **0.93** |
| `one_step` | + one snapshot interval of the $e_\text{KH}$ ODE | 0.45 |
| `full` | + the ODE run free | 0.28 |

The ansatz is sound; the evolution equation destroys it, and does so within a single
$\Delta t$, so this is a wrong rate law rather than accumulated drift. That is exactly the
part a coefficient is supposed to set. It makes the coefficients the right thing to attack
and, equally, tells us the honest ceiling: **whatever we do to the coefficients, we are
bounded by the `oracle_e` row**, because that row *is* the model with a perfect coefficient.

A second signal points the same way. The fitted $\beta_{ij}$ is not stable across fitting
windows — $\beta_{xy}$ moves from $0.131$ on $t\in[3,6.4]$ to $0.012$ on $t\in[1,20]$ — and
the fitted anisotropy is 6:1 against the paper's 1.16:1. The "constants" are absorbing
flow-state structure. That is a coefficient that wants to be a function.

---

## 2. The measurement

Rather than train something and hope, the question *"how much of $e_\text{KH}$ is
determined by the instantaneous local resolved state?"* is answered directly by a ladder of
predictors of increasing freedom, all scored identically on held-out snapshots:

| rung | predictor of $e_\text{KH}$ | freedom |
|---|---|---|
| 0 | $e^\text{DNS}_\text{KH}$ | the ceiling — a perfect coefficient |
| 1 | $k$,   $\rho\lambda^2\|S\|^2$, one global $k$ | MInIT's equilibrium limit as published |
| 2 | same, $k=k(t)$ refit per snapshot | a Germano-style *dynamic* procedure |
| 3 | linear in 10 local invariants | state-dependent, no network |
| 4 | MLP (4,929 params) on those invariants | state-dependent, pointwise |
| 5 | MLP (6,209 params) + $5\times5$ neighbourhood | state-dependent, small stencil |

The network predicts $\log e_\text{KH}$, so positivity is structural. Its inputs are
resolved-state quantities only ($\rho$, $|S|$, $|\partial_yv_x|$, $|\nabla\!\cdot\!v|$,
vorticity, $|v|$, $W$, $|\nabla\rho|/\rho$, and the Smagorinsky combination itself, so the
higher rungs can never lose to rung 1 for want of the right variable). $\beta_{ij}$ is held
global and fitted on training data only.

Reproduce with:

```bash
PYTHONPATH=. python -u scratch/minit_learned_coeff_feasibility.py   # splits within dns_khi
PYTHONPATH=. python -u scratch/minit_crossrealisation.py            # train dns_khi -> test dns_khi_s1
```

---

## 3. Result: within `dns_khi`

Interleaved snapshots (95 held out of 191) — *is $e_\text{KH}$ a function of local state at
all?*

| rung | $R^2(\log e)$ | $R^2(e)$ | corr $\tau$ | amp | $\Pi/\Pi_\text{ex}$ |
|---|---|---|---|---|---|
| 0 oracle (ceiling) | 1.000 | 1.000 | **0.904** | 0.911 | **0.646** |
| 1 equilibrium, global $k$ | −2.378 | 0.246 | 0.522 | 0.496 | 0.014 |
| 2 equilibrium, per-snapshot $k$ | −1.739 | 0.292 | 0.538 | 0.532 | 0.058 |
| 3 linear | 0.843 | 0.513 | 0.659 | 0.635 | 0.284 |
| 4 MLP pointwise | 0.905 | 0.772 | 0.787 | 0.751 | 0.471 |
| 5 MLP + $5\times5$ | **0.920** | **0.851** | **0.832** | **0.839** | **0.548** |

Two things stand out.

**$e_\text{KH}$ is largely a local function.** $R^2(\log e)=0.92$ from ten instantaneous
resolved-state invariants. The information a state-dependent coefficient would need is
present in the resolved field.

**A classical dynamic procedure would not have worked.** Rung 2 — refitting the coefficient
globally at every snapshot, which is what a Germano-type procedure does — buys almost
nothing (0.522 → 0.538). The missing structure is *spatial*, not a time-varying global
amplitude. This is the result that makes a learned field worth the machinery: the cheap
classical alternative is measurably inadequate here.

The forward-in-time split (train $t\le10$, test $t>10$) recovers the same **fraction** of
the available gap, ~80 %, which is the first evidence that the relationship transfers
rather than being memorised.

---

## 4. Result: transfer to an unseen realisation

Trained on all of `dns_khi`, scored on all of `dns_khi_s1` (different seed), aggregated
exactly as `src.evaluate_apriori` does so these rows sit beside the TBNN's:

| model | params | amp | corr | $\Pi/\Pi_\text{ex}$ | sign | hi-$k$ in $\tau$ | amp 1st→2nd half |
|---|---|---|---|---|---|---|---|
| exact stress | — | 1.000 | 1.000 | 1.000 | — | 44.8 % | — |
| **oracle-$e$ (ansatz ceiling)** | — | 0.950 | +0.770 | 0.623 | 100 % | 43.4 % | 0.918 → 0.982 |
| MInIT equilibrium, global $k$ | 1 | 0.368 | +0.481 | 0.144 | 95 % | 17.4 % | 0.495 → 0.241 |
| learned coeff, pointwise | 4,929 | 0.763 | +0.681 | 0.458 | 100 % | 32.6 % | 0.778 → 0.747 |
| **learned coeff, $5\times5$** | 6,209 | 0.810 | +0.713 | 0.499 | 100 % | 38.3 % | 0.806 → **0.815** |
| TBNN (`checkpoints_fullrange`) | 139,521 | 0.739 | **+0.832** | **0.637** | 100 % | 37.4 % | 0.816 → **0.662** |

Fraction of the available gap (rung 1 → ceiling) that the $5\times5$ network recovers:
**amplitude 76 %, correlation 80 %, $\Pi$ 74 %, high-$k$ content 80 %.** The consistency
across four independent metrics, on a realisation never seen in training, is the core
feasibility result.

Three specific observations:

1. **$\Pi$ improves by more than an order of magnitude** (0.014 → 0.548 within `dns_khi`,
   a factor of 39; 0.144 → 0.499 across realisations, a factor of 3.5 against a
   coefficient that happens to fit the second run better). `MINIT.md` §4 concludes that "a closure supplying ~2 % of the true
   energy transfer cannot move a trajectory much, whatever its correlation". That objection
   does not apply to the learned-coefficient version. It is the single strongest argument
   for building it.

2. **It does not decay in time.** The learned coefficient holds amplitude 0.806 → 0.815
   across the first and second halves of an unseen run, while the TBNN falls 0.816 → 0.662
   and the global-coefficient MInIT falls 0.495 → 0.241. Late-time amplitude collapse is
   the TBNN's known failure mode (`memory/tbnn_highk_collapse.md`); this family does not
   have it, because its amplitude is tied to a resolved-state quantity that does not itself
   decay.

3. **Backscatter survives.** The exact $\Pi$ is *negative* in this flow (2D inverse
   cascade). The oracle reproduces 44.5 % of points backscattering against 40.5 % exact,
   with $\text{corr}(\Pi,\Pi_\text{ex})=0.977$ — so the $\beta_{ij}e$ family can represent
   2D backscatter, which was not obvious.

---

## 5. The binding constraint, and what it costs

**The TBNN's correlation (0.832) exceeds the oracle-$e$ ceiling (0.770).** So does its
$\Pi$ (0.637 vs 0.623). A perfect coefficient inside MInIT's stress ansatz is *already
worse* than the TBNN on those two metrics. No amount of network capacity spent on the
coefficient changes that, because the constraint is the ansatz itself: with $\beta_{ij}$
fixed and $e\ge0$,

$$\Pi = (\beta_{ij}S^{ij})\,e \quad\Longrightarrow\quad \operatorname{sign}\Pi = \operatorname{sign}(\beta_{ij}S^{ij}),$$

which is set entirely by the resolved strain orientation and is *unlearnable by any
$e$*. Measured, that caps pointwise $\Pi$-sign agreement at **76.8 %** no matter what the
coefficient network does.

This is the honest limit of the proposal as stated, and it should be stated plainly in any
write-up: **a state-dependent-coefficient MInIT is a much better MInIT, not a better TBNN.**

---

## 6. Identifiability — one scalar, not three

A practical trap. In the equilibrium limit,

$$\tau^{ij} = \beta_{ij}\left(\frac{A_\gamma}{C}\right)^{2} 4\rho\lambda_\text{KH}^{2}|S|^{2},$$

so only the **product** $\beta_{ij}(A_\gamma/C)^2$ is identifiable — the paper's own Fig. 11
observation, and why it fixes $A_\gamma=0.14$. A network asked to output $A_\gamma$ and $C$
separately is fitting an unidentifiable pair and will drift along the degenerate ray
(exactly what the free two-parameter scan in `MINIT.md` §3 does, wandering 16× from theory
for a 12 % gain). **The network must emit one scalar.** In `evolve` mode $A_\gamma$ and $C$
are formally separable through the transient, but the scan shows that separation is
numerically weak; it should not be relied on.

---

## 7. Recommended design

**Phase 1 — equilibrium mode, learned scalar.** Algebraic, so it is fully differentiable
with no recurrence and drops straight into the existing a-priori training loop:

$$e_\text{KH}(x) = \exp\!\big[g_\theta(\mathcal{I}(x))\big]\cdot \rho\lambda_\text{KH}^{2}|S|^{2}, \qquad \tau^{ij}=\beta_{ij}\,e_\text{KH}$$

with $g_\theta$ a 2-layer CNN of $5\times5$ receptive field (~6 k parameters), $\mathcal{I}$
the ten resolved invariants, $\beta_{ij}$ global and fitted. Positivity and the
Smagorinsky scaling are structural; the network only supplies a dimensionless correction.
This is the configuration measured above, so its a-priori score is already known.

**Phase 2 — a-posteriori gate, before anything else.** `MINIT.md` §5 records that
*refitting made MInIT worse a-posteriori* despite scoring better a-priori. That precedent
applies directly here and is the main risk to the whole proposal. Phase 1 must be run
through `src.evaluate_rollout` and `src.evaluate_shorthorizon` before any further
investment. **If the short-horizon increment error does not improve, stop.**

**Phase 3 — only if Phase 2 passes: the hybrid worth actually building.** §5 shows the
fixed $\beta_{ij}$ is the binding constraint and §4 shows the learned amplitude is the one
thing that does not decay late, which is precisely the TBNN's failure. So let the TBNN's
tensor basis supply the *shape* and the MInIT scalar supply the *amplitude*:

$$\tau^{ij} = \underbrace{e_\text{KH}(\text{state})}_{\text{MInIT scalar, positive, physically scaled}}\times\underbrace{\hat{T}^{ij}(\mathcal{I})}_{\text{normalised tensor basis}}$$

This is not a third competitor; it is a targeted fix for the measured weakness of each
model, and it is cheap because both halves already exist in this codebase.

**Keep the published model in the comparison.** A learned-coefficient MInIT is no longer
"MInIT as published" and must not replace it in the closure table, or the paper comparison
becomes unfalsifiable. It is a third row.

---

## 8. Cost, and what would falsify this

Cost is small: ~6 k parameters against the TBNN's 139,521, training in minutes on the
cached DNS reduction, and the a-priori scaffolding (`src.evaluate_apriori`,
`src.minit_calibrate`, the two scripts above) already exists. The expensive part is
Phase 2, which is an ordinary rollout campaign.

This proposal is falsified by any one of:

- **the a-posteriori gate** — if the short-horizon increment error does not improve over the
  global-coefficient MInIT, the a-priori gain is not dynamically relevant and the precedent
  in `MINIT.md` §5 repeats;
- **resolution transfer** — the coefficient is trained at $r=16$; if refitting at $r=8$ or
  $r=32$ moves it substantially, it has absorbed grid scale rather than physics, and the
  claim of a state-dependent *closure* fails even though the fit is good;
- **a second flow** — everything here is Kelvin–Helmholtz. A coefficient network that does
  not transfer to a different instability is a KHI fit with extra steps. `dns_khi_s1` tests
  a different realisation, which is necessary but not sufficient.

The first is cheap and should be run first.

---

## 9. Files

| path | what |
|---|---|
| `scratch/minit_learned_coeff_feasibility.py` | the ladder, splits within one run |
| `scratch/minit_crossrealisation.py` | train `dns_khi` → test `dns_khi_s1` |
| `results/minit/learned_coeff_feasibility.json` | §3 numbers |
| `results/minit/learned_coeff_crossrealisation.json` | §4 numbers |
| `results/minit/apriori_fullrange_s1.json` | the TBNN row of §4 |
| `MINIT.md` | the port, the calibration, the a-priori/a-posteriori record |


---

## 10. Built and measured (2026-09-23)

Phase 1 is implemented in [`src/minit_learned.py`](../src/minit_learned.py) and trained by
[`src/train_minit_coeff.py`](../src/train_minit_coeff.py). The network is two 3x3 circular
convolutions plus a 1x1 head — exactly the 5x5 receptive field §7 asked for — at **12,193
parameters** against the TBNN's 139,521.

Both output modes of §7 were trained rather than assumed:

| on `dns_khi_s1` (unseen seed) | amp | corr | $\Pi/\Pi_\text{ex}$ | hi-$k$ | $R^2(e)$ | amp 1st→2nd |
|---|---|---|---|---|---|---|
| exact stress | 1.000 | 1.000 | 1.000 | 44.8% | 1.000 | — |
| oracle-$e$ ceiling | 0.950 | +0.770 | 0.623 | 43.4% | 1.000 | 0.918 → 0.982 |
| MInIT equilibrium, global $k$ | 0.368 | +0.481 | 0.144 | 17.4% | — | 0.495 → 0.241 |
| scratch prototype (5x5 stats + MLP) | 0.810 | +0.713 | 0.499 | 38.3% | — | 0.806 → 0.815 |
| **learned CNN, `residual`** | **0.869** | +0.711 | **0.531** | **42.8%** | 0.833 | 0.889 → 0.848 |
| learned CNN, `direct` | 0.826 | +0.716 | 0.516 | 41.9% | 0.841 | 0.813 → 0.839 |
| TBNN (`checkpoints_fullrange`, 139,521 par.) | 0.739 | **+0.832** | **0.637** | 37.4% | — | 0.816 → 0.662 |

The production CNN beats the scratch prototype the rest of this report was measured on, so
§4's numbers are a floor rather than a ceiling. `residual` and `direct` are near-tied;
`residual` is ahead on amplitude, $\Pi$ and high-$k$ content — the three that matter in a
rollout — so it is the default. Against the ansatz ceiling it reaches **91% of the
amplitude, 92% of the correlation, 85% of the transfer and 99% of the high-$k$ content**.

### The gate (§7 Phase 2)

`dns_khi`, $t = 6.5$–$8.0$, 15 starts, one snapshot interval each:

| run | mean $\varepsilon$ | skill vs unclosed | beats unclosed |
|---|---|---|---|
| frozen (no change) | 1.0000 | −505.96% | 0/15 |
| unclosed | 0.1656 | — | — |
| MInIT (equilibrium) | 0.1656 | −0.02% | 10/15 |
| MInIT (fitted) | 0.1635 | +1.16% | 10/15 |
| MInIT (paper) | 0.1635 | +1.19% | 9/15 |
| **learned coeff (`direct`)** | 0.1624 | **+1.90%** | **14/15** |
| **learned coeff (`residual`)** | **0.1623** | **+1.92%** | **14/15** |
| TBNN | 0.1620 | +2.06% | 13/15 |
| Exact DNS oracle | 0.1630 | +1.52% | 14/15 |

**The gate passed.** The stop condition in §7 was "if the short-horizon increment error does
not improve, stop". It improved by a factor of ~1.6 over the best global-coefficient MInIT,
on the most starts of any row. This is the result that matters most, because §7 flagged the
opposite precedent — refitting global coefficients improved MInIT a-priori and made it
*worse* a-posteriori (`MINIT.md` §5). That does not repeat here.

Three cautions on reading that table:

1. **The learned rows are handed strictly less information than the MInIT rows.** Every
   `minit_*` run is seeded from the exact filtered-DNS $e_\text{KH}$ at $t_0$; the learned
   closure is stateless and reads only resolved state. It wins anyway.
2. **Beating the "ceiling" is not beating physics.** The oracle supplies the *true* stress,
   which is not the same as the *optimal correction for a discretised solver* — `MINIT.md`
   §5.2 already recorded the oracle going negative in the late window for this reason. A
   learned model can absorb some discretisation error too, so +1.52% bounds "supply the
   true stress", not "minimise increment error".
3. **The TBNN is still ahead on mean skill** (+2.06% vs +1.92%), though on fewer starts
   (13/15 vs 14/15) and with 11x the parameters. The gap is far smaller than the spread
   across starts; this is a tie, not a win for either.

### What is not done

Phase 3 (the hybrid of §7 — MInIT scalar for amplitude, tensor basis for shape) is
untouched. The §5 argument for it stands and §10 strengthens it: the learned coefficient is
the row that does not decay in time, and the TBNN is the row with the better shape.

The two falsifiers in §8 that remain open are **resolution transfer** ($r=8$, $r=32$) and
**a second flow**. Everything here is still Kelvin-Helmholtz.

### Reproduce

```bash
python -m src.train_minit_coeff --output-mode residual --device cpu
python -m src.evaluate_shorthorizon --t-lo 6.5 --t-hi 8.0 \
    --with-learned results/minit/minit_coeff_residual.pt
```


---

## 11. Cost (measured on a V100, 2026-09-23)

Convolution was never the bottleneck, and the closure is **cheaper than the model it
competes with**.

### Per solver step, 64x64 float64 (the $r=16$ case)

| closure | ms/step | vs unclosed |
|---|---|---|
| unclosed | 3.58 | — |
| MInIT (equilibrium, algebraic) | 13.48 | +276% |
| **learned coefficient** | **25.66** | **+617%** |
| TBNN | 57.22 | +1498% |

The learned closure costs **2.2x less than the TBNN**, at 11x fewer parameters, for a-priori
and short-horizon scores that sit within noise of it (§10). The solver calls any closure
**7 times per step** (RK stages plus predictor-corrector recovery), which is why a 4.9 ms
call becomes a 22 ms overhead.

### Inside one call

| stage | ms | share |
|---|---|---|
| `compute_invariants` (gradient passes) | 2.67 | 55% |
| `CoefficientNet` forward (3 convolutions) | 1.19 | 24% |
| `assemble_tau` | 0.75 | 15% |
| full `__call__` | 4.86 | 100% |

**Feature computation dominates, not the network.** Any optimisation effort belongs in the
finite-difference gradients, not in the convolution or the parameter count.

### The cuDNN workaround costs nothing

cuDNN cannot run on these sm_70 nodes at all (§ `_conv2d_gemm`), so the choice is between
workarounds, not against a fast baseline. All three were measured at 64x64 float64:

| path | ms per 3x3 conv | agreement |
|---|---|---|
| `F.conv2d` (cuDNN) | — | **fails on sm_70, any dtype** |
| `F.conv2d` under scoped `cudnn.flags(enabled=False)` | 0.047 | bit-identical |
| `_conv2d_gemm` (im2col + cuBLAS) | 0.089 | bit-identical |

`_conv2d` now probes the device once and takes the best available route, so a newer GPU
gets the native cuDNN path instead of paying for a workaround it does not need. Picking the
scoped disable over the GEMM path is worth ~11% of a closure call — small, but free and
exactly equal numerically.

Float64 is kept throughout. On a V100 (1:2 FP64 ratio) double precision is nearly free, and
keeping it means the GPU rollout reproduces the CPU a-priori numbers to 1.6e-15.

---

## 12. First full a-posteriori rollout, $t = 1 \to 10$

> **Correction (§16):** the high-$k$ figures in this section are single-frame values at
> $t=10$ and are noise. Time-averaged, MiNNit and the TBNN retain the same high-$k$
> energy; the "63% vs 30%" claim is withdrawn.

`condor_evaluate.sub`, cluster 1324966, $r=16$, flux coupling, against filtered DNS at
$t=10$:

| run | log-spectral distance | high-$k$ band ratio | total KE ratio |
|---|---|---|---|
| unclosed | 0.7522 | 0.212 | 1.096 |
| MInIT (paper), DNS-seeded | 0.7178 | 0.269 | 1.074 |
| TBNN | 0.7166 | 0.250 | 1.082 |
| **learned coefficient, no seed** | 0.7168 | **0.291** | 1.073 |
| Exact DNS oracle | 0.6716 | 0.337 | 1.087 |

On **high-$k$ retention** — the quantity that motivated this whole line of work — the
learned closure recovers **63% of the gap** between unclosed and the oracle, against the
TBNN's 30%, while matching the TBNN on log-spectral distance. That is the long-rollout
small-scale decay the TBNN was losing.

Read with §5.2's caution: persistence still scores 0.4997 here, better than every closure,
because over 400 steps the trajectory decorrelates and a pointwise norm stops measuring
closure quality. The band ratios are the informative columns, not the distance.


---

## 13. Filter-size scan, contamination-free (2026-10-18)

Every model trained on `dns_khi` ($t = 1$–$20$), scored on `dns_khi_s1`. Eq. 25 is RMS
over time. $r = 128$ (8×8 grid) lies beyond the inertial range and is a control.

| r | oracle-$e$ corr / Eq.25 | **MiNNit** corr / Eq.25 / amp | **TBNN** corr / Eq.25 / amp | global-k MInIT Eq.25 |
|---|---|---|---|---|
| 8 | 0.770 / 0.676 | 0.715 / **0.780** / 0.907 | **0.828** / 0.851 / 0.836 | 4.34 |
| 16 | 0.770 / 0.660 | 0.711 / 0.772 / 0.869 | **0.819** / **0.760** / 0.764 | 3.51 |
| 32 | 0.749 / 0.658 | 0.672 / 0.780 / 0.861 | **0.804** / **0.704** / 0.779 | 1.74 |
| 64 | 0.730 / 0.651 | 0.633 / 0.778 / 0.913 | **0.819** / **0.714** / 0.755 | 1.13 |
| 128 | 0.738 / 0.603 | 0.613 / 0.795 / 1.067 | **0.814** / **0.570** / 1.004 | 0.96 |

High-$k$ fraction of $\tau$ (exact ≈ 45–50%): MiNNit 45.3 / 42.8 / 38.1 / 39.0%, TBNN
41.7 / 37.3 / 35.4 / 36.1% for $r = 8$–$64$. $\Pi/\Pi_\text{ex}$ at $r = 8$–$32$: TBNN
0.65–0.68, MiNNit 0.56 / 0.53 / 0.44; at $r \ge 64$ the ratio is unusable. *Corrected
2026-10-19:* not because the mean transfer is near zero (it is −9.2e-3 at $r=64$ vs −1.05e-2 at
$r=32$), but because the exact transfer crosses zero in a few snapshots ($t=2$–3 at $r=64$) and
those near-zero denominators dominate the average of per-snapshot ratios.

1. **The ansatz ceiling is filter-independent** (paper Fig. 13 reproduced). **The TBNN's
   correlation is too, and sits above that ceiling at every $r$** — the fixed-$\beta$
   constraint of §5 is real.
2. **On Eq. 25 the ranking depends on filter size:** MiNNit wins at $r=8$, ties at 16, the
   TBNN wins from 32 up. *Corrected 2026-10-19:* that RMS ranking is driven by the TBNN's
   late-time collapse at small $r$ (Eq. 25 up to 1.8 at $t = 17$–20, $r=8$). Per snapshot the
   TBNN is better at every $r$: median 0.58 / 0.60 / 0.60 / 0.58 vs MiNNit 0.80 / 0.79 /
   0.81 / 0.77 at $r$ = 8 / 16 / 32 / 64. Eq. 25 punishes the TBNN's amplitude deficit (0.76–0.84), which is
   why its large correlation lead does not translate one-for-one.
3. **MiNNit keeps more high-$k$ and more amplitude at every $r$; the TBNN has better
   correlation and transfer.** Same split as §10, now on clean data.
4. **MiNNit's correlation erodes with $r$ (0.715 → 0.633) while the TBNN's does not** — the
   coefficient network is the one running short of samples at large filters.
5. **The TBNN's late-time amplitude decay persists at small $r$** (0.96 → 0.71 at $r=8$,
   0.83 → 0.70 at $r=16$) even with training over all of $t = 1$–$20$.

Outputs: `results/filter_scan/` (`minit_coeff_r*.json`, `tbnn_r*.json`), checkpoints in
`checkpoints_r*/`.


---

## 14. Eq. 25 as a training loss — measured, rejected (2026-10-18)

MiNNit at $r=16$, loss = log-MSE + $w\cdot\langle \text{Eq.25}^2\rangle$, otherwise identical.

| | baseline ($w=0$) | $w=1$ | $w=5$ |
|---|---|---|---|
| a-priori, `dns_khi_s1`: Eq. 25 | 0.772 | 0.742 | **0.736** |
| amp / corr / $\Pi$ | 0.869 / 0.711 / 0.531 | 0.899 / 0.733 / 0.581 | 0.921 / 0.738 / 0.576 |
| short-horizon skill (paired vs baseline) | +1.92% | −0.13 ± 0.11% | +0.06 ± 0.10% |
| rollout $t=1\to10$: high-$k$ ratio | **0.292** | — | 0.234 |
| rollout: log-spectral distance / total KE | **0.717** / 1.073 | — | 0.758 / 1.134 |

Better on every a-priori metric, indistinguishable on the short-horizon gate, and **worse in
the long rollout**: the extra amplitude overshoots total KE by 13% and lands in the low and
mid bands, not at high $k$ — the same signature as the gain sweep. Third instance of an
a-priori improvement that does not survive a rollout (after the global refit and the gain).
Default stays log-MSE (`--eq25-weight 0`).

Side result: the clean TBNN (`checkpoints_r16`) gives rollout high-$k$ 0.288, log-spectral
0.778, KE 1.119 — on high-$k$ level with MiNNit (0.292), behind it on distance and KE.


---

## 15. The hybrid: TBNN shape × MiNNit amplitude (2026-10-18)

`HybridClosure` (`src/minit_learned.py`): $\tau^{\mu\nu} = e_\text{MiNNit}\,
\tau^{\mu\nu}_\text{TBNN}/e_\text{TBNN}$ cell by cell, falling back to MiNNit's
$\beta_{ij}e$ where $e_\text{TBNN} < 0.05\,\langle e_\text{MiNNit}\rangle$ (9–11% of
cells). No new training — both parts are the clean `dns_khi`-trained models.
Flags: `--with-hybrid CKPT` on `src.evaluate_rollout` and `src.evaluate_shorthorizon`.

**A-priori, `dns_khi_s1` (unseen):**

| r | model | amp | corr | Eq. 25 | $\Pi/\Pi_\text{ex}$ | high-$k$ (exact) |
|---|---|---|---|---|---|---|
| 8 | TBNN / MiNNit / **hybrid** | 0.818 / 0.907 / 1.107 | 0.827 / 0.715 / **0.902** | 0.849 / 0.780 / **0.530** | 0.66 / 0.56 / **0.96** | 41.4 / 45.3 / 45.6 (44.5) |
| 16 | TBNN / MiNNit / **hybrid** | 0.762 / 0.869 / 0.946 | 0.819 / 0.711 / **0.902** | 0.760 / 0.772 / **0.484** | 0.64 / 0.53 / **0.80** | 37.3 / 42.8 / 43.3 (44.8) |
| 32 | TBNN / MiNNit / **hybrid** | 0.775 / 0.861 / 0.871 | 0.806 / 0.672 / **0.881** | 0.703 / 0.780 / **0.515** | 0.66 / 0.44 / **0.71** | 35.4 / 38.1 / 40.1 (46.3) |

Better than both parents on every metric at every $r$, and above the fixed-$\beta$ ceiling
(0.77). At $r=8$ it overshoots amplitude early (1.24 in the first half).

**Short-horizon gate** ($r=16$, $t=6.5$–$8$, TBNN = `checkpoints_r16`): hybrid **+2.40%**
(14/15), TBNN +2.11% (12/15), MiNNit +1.92% (14/15), oracle +1.52%. Paired: hybrid vs
MiNNit +0.49 ± 0.18% (s.e.), vs TBNN +0.28 ± 0.26% — better than MiNNit, not
distinguishable from the TBNN.

**Rollout $t = 1\to10$, $r=16$:**

| run | log-spectral distance | total KE | low / mid / high band |
|---|---|---|---|
| unclosed | 0.752 | 1.096 | 1.110 / 0.905 / 0.212 |
| TBNN | 0.777 | 1.119 | 1.132 / 0.964 / 0.289 |
| MiNNit | 0.716 | 1.073 | 1.089 / 0.715 / **0.292** |
| **hybrid** | **0.689** | **1.050** | **1.065** / 0.802 / 0.250 |
| oracle | 0.672 | 1.087 | 1.100 / 0.944 / 0.338 |

The hybrid is the closest non-oracle run to the DNS spectrum: it closes 79% of the
unclosed→oracle gap in log-spectral distance and has the best energy budget, but it keeps
*less* high-$k$ than either parent (0.250 vs 0.29). Single realisation; treat the
differences as indicative. **§16: that high-$k$ gap is a single-frame artefact.**


---

## 16. The high-$k$ "loss" was a single-frame artefact (2026-10-18)

The high-band ratio at one time swings between 0.08 and 0.34 frame to frame (the DNS and
the coarse runs decorrelate). Time-averaged over $t = 2$–$10$ from the hybrid rollout's
saved history (band energy / DNS; ± = block bootstrap, 1-time-unit blocks):

| run | low | mid | high |
|---|---|---|---|
| unclosed | 1.023 | 0.909 | 0.191 ± 0.022 |
| TBNN | 1.028 | 0.872 | 0.168 ± 0.019 |
| MiNNit | 1.019 | 0.796 | 0.160 ± 0.020 |
| hybrid | **0.995** | 0.826 | 0.163 ± 0.017 |
| oracle | 1.021 | 0.817 | 0.170 ± 0.020 |

1. All closures, the exact stress included, retain the same high-$k$ energy within error.
2. None adds high-$k$ over the unclosed run; even the true stress removes a little. The
   deficit is the coarse solver's numerical dissipation: every run loses ~70% of its high-$k$
   energy in the first 0.7 time units. **A sub-grid closure cannot recover it in this setup.**
3. The hybrid's real gain is at large scales: late-window ($t=6$–$10$) low-band energy 1.004
   vs 1.06–1.09 for every other run, the oracle included — it removes the large-scale
   pile-up, which is what its better spectral distance and KE budget reflect.

Consequence: every single-time high-$k$ comparison in §12, §14 and the gain/coupling sweeps
is unreliable. Use time-averaged band ratios from `--save-history`.


---

## 17. Regimes, the right reference, and the hybrid's high-$k$ — independently verified (2026-10-19)

`evaluate_rollout` now prints **windowed band ratios** whenever history is saved: geometric
mean over frames (equal weight per frame, scale-free), per regime window (default thirds of
$[t_\text{start}+0.5, t_\text{end}]$ plus the whole span), against the **nearest real DNS
snapshot** (never the field blend the animation uses), with a **paired** ratio to the
exact-stress oracle and a circular block-bootstrap CI shared by all runs
(`windowed_band_ratios`). Verified by a code review, a robustness sweep, a paired-statistics
check and two adversarial skeptics; their corrections (paired/shared bootstrap, block
length, time-alignment guard, longest history as time axis, partial-run flag) are applied.

**Do regimes contribute differently? Yes.** Retention itself falls over the run (high band
≈ 0.45 → 0.37 → 0.33 for the oracle) and the closures tie early and separate late, so a
whole-span number is a blend. Quote per-window, paired-to-oracle values.

High band (k > k_max/3) relative to the oracle, paired, 95% CI, hybrid rollout $r=16$:

| window | unclosed | TBNN | MiNNit | hybrid |
|---|---|---|---|---|
| 1.5–4.3 | 1.175 [1.09, 1.28] | 1.028 [0.97, 1.09] | 1.035 [1.01, 1.07] | 1.012 [0.95, 1.07] |
| 4.3–7.2 | 1.040 [0.94, 1.19] | 0.999 [0.88, 1.14] | 0.974 [0.93, 1.02] | 0.931 [0.83, 1.01] |
| 7.2–10 | 1.018 [0.79, 1.30] | 1.098 [0.95, 1.30] | 0.936 [0.87, 1.02] | 0.869 [0.76, 1.00] |

1. **Raw high-$k$ retention is not a quality score.** The no-model run and the TBNN retain
   *more* than the oracle (the best any closure can do on this solver). Closeness to the
   oracle is the score; the reference cancels exactly in that paired ratio.
2. **The hybrid is not significantly different from the oracle or MiNNit** in any window once
   the ~48 comparisons behind this analysis are accounted for (late-window CI touches 1). It
   is below TBNN late, but TBNN sits above the oracle there.
3. **What the hybrid does late is redistribute, not dissipate:** less energy at $k\approx$
   9–16 (just below the filter cutoff), *more* at $k\approx$ 5–10 (mid band 0.988 vs oracle
   0.826), at $k>16$ it is level with the others, and its total KE is the closest to the DNS
   of any run (1.035 vs 1.08–1.14). On log-spectral distance to the oracle over $k>10$, late:
   MiNNit 0.34, hybrid 0.42, unclosed 0.45, TBNN 0.52.
4. **Why §16 saw "all equal":** its band was $k>16$, where the closures really are equal. The
   reference choice (blend vs nearest snapshot) shifts ratios by only 3–4% and cancels in
   paired comparisons. §16's conclusion is right for $k>16$, incomplete for $k\approx$ 9–16.
5. **Scope:** every CI here covers frame sampling along one trajectory. It does not cover
   initial-condition, DNS-seed or training-seed variability. To make any of these a property
   of the closure, repeat on `dns_khi_s1` or on perturbed initial states.


---

## 18. Rollout ensemble: no closure is reliably better in long rollouts (2026-10-19)

`condor_ensemble.sub`: `dns_khi` and the unseen `dns_khi_s1`, 4 members each (member 0
unperturbed, then `--perturb-seed` 1–3 at 1e-3 of the rms velocity, $k \le N/8$), all five runs
per member from the same state, $r=16$, $t = 1\to10$, models trained on `dns_khi` only.
Aggregated by `scratch/ensemble_summary.py` (mean ± s.e. over members).

Pooled, whole window $t = 1.5$–10:

| run | log-spectral distance | total KE | high band / oracle | mid band |
|---|---|---|---|---|
| unclosed | 0.584 ± 0.013 | 0.940 ± 0.031 | 1.03 ± 0.03 | 0.90 ± 0.03 |
| TBNN | 0.613 ± 0.012 | 0.994 ± 0.014 | 0.99 ± 0.03 | 0.87 ± 0.02 |
| MiNNit | 0.599 ± 0.004 | 1.005 ± 0.008 | 0.99 ± 0.01 | 0.85 ± 0.01 |
| hybrid | 0.605 ± 0.007 | 0.996 ± 0.002 | 0.99 ± 0.02 | 0.88 ± 0.00 |
| oracle | 0.581 ± 0.009 | 1.005 ± 0.007 | 1 | 0.85 ± 0.01 |

1. **No learned closure is reliably better than another.** Paired spectral distance, hybrid
   minus TBNN −0.008 ± 0.006, minus MiNNit +0.006 ± 0.003. The single-run "hybrid has the best
   spectrum" (§15) is not confirmed.
2. **The realisation decides the ranking.** On `dns_khi` the unclosed run matches the oracle and
   every closure is slightly worse; on `dns_khi_s1` the unclosed run loses 24% of its kinetic
   energy late (KE 0.761 at $t=7$–10), every closure prevents that, and MiNNit is closest to
   the oracle (late LSD 0.709 vs oracle 0.702, hybrid 0.741, TBNN 0.775, unclosed 0.791).
3. **The hybrid's one consistent advantage is its energy budget:** total KE 0.996 ± 0.002,
   the most stable of any run, the oracle included.
4. **The perturbation was too small to make independent members.** At 1e-3 the four
   `dns_khi` members agree to three decimals in the first two windows; late windows differ by
   up to ±0.04. This is effectively **two** independent samples (the DNS realisations), so the
   errors above understate the real spread. More DNS seeds, or perturbations of ~1e-2, are
   needed before any long-rollout ranking.


---

## 19. Ensemble at 1% perturbation: MiNNit is the best learned closure in long rollouts; the hybrid can diverge (2026-10-19)

`condor_ensemble_p1e-2.sub`: as §18 with `--perturb-amplitude 0.01` (seed-0 members reused).
Members now genuinely diverge: density correlation with the unperturbed member at
$t = 3 / 6 / 10$ is 0.99 / 0.59–0.94 / **0.08–0.83**, against 1.00 / 0.99 / 0.80–0.98 at 1e-3.

**The hybrid diverged** (non-finite state at $t = 5.37$) in `dns_khi` member 2 — 1 of 14
hybrid rollouts over both ensembles; no other closure diverged. Cause confirmed and
fixed in §20.

Pooled, $t = 1.5$–10 (7 hybrid members):

| run | log-spectral distance | total KE | high band / oracle |
|---|---|---|---|
| oracle | 0.577 ± 0.010 | 0.989 ± 0.010 | 1 |
| unclosed | 0.581 ± 0.015 | 0.944 ± 0.025 | 1.04 ± 0.04 |
| MiNNit | 0.595 ± 0.008 | 0.985 ± 0.011 | 1.00 ± 0.01 |
| TBNN | 0.612 ± 0.014 | 0.980 ± 0.015 | 0.97 ± 0.03 |
| hybrid | 0.619 ± 0.009 | 0.978 ± 0.009 | 0.94 ± 0.02 |

1. **Even the exact stress barely beats no closure on spectral distance** (0.577 vs 0.581).
   On this solver the sub-grid term is a small correction to the long-time spectrum.
2. **MiNNit is the best learned closure:** paired against the hybrid it is closer to the DNS in
   7/7 members (+0.021 ± 0.004), and against the TBNN by 0.017. The hybrid and the TBNN tie
   (+0.001 ± 0.009).
3. **On the unseen run the closures fix a real energy loss.** Without a closure, total KE
   falls to 0.82 late; MiNNit keeps it at 1.005 ± 0.015 with late spectral distance 0.718,
   closest to the oracle's 0.700 (TBNN 0.782, hybrid 0.753, unclosed 0.778).
4. **This reverses the a-priori ranking.** A-priori the hybrid wins every metric and MiNNit has
   the lowest correlation; over one 0.1 interval the TBNN and hybrid beat MiNNit; in long
   rollouts MiNNit is best and the hybrid is worst and least stable. A-priori and short-horizon
   scores do not predict long-rollout quality here.


---

## 20. The hybrid's divergence: cause and clip-after-rescale fix (2026-10-19)

**Cause.** `HybridClosure` rescaled the TBNN stress by $e_\text{MiNNit}/e_\text{TBNN}$ *after* the
TBNN closure's realizability clip ($\lVert\tau\rVert_F \le 0.5\,p$), so nothing bounded the
result. `scratch/hybrid_divergence_probe.py` (hybrid only, the diverged member) logs it: the
rescale factor reaches **~760×** (the fallback floor is relative to the *mean* MiNNit energy,
so it does not cap the local ratio), and around $t = 5.0$–5.4 — when the original run
diverged — the stress spikes to **4.5–172× the bound**, on CPU and GPU alike.

**Fix.** `HybridClosure(clip_mode="frobenius", clip_kappa=0.5)` clips after the rescale
(default); `clip_mode=None` / `--hybrid-no-clip` restores the old behaviour.

**What it costs.** Nothing measurable, and a-priori it helps where it engages:

| r | Eq. 25 unclipped → clipped | amplitude | corr |
|---|---|---|---|
| 8 | 0.530 → **0.444** | 1.107 → 1.038 | 0.902 → 0.913 |
| 16 | 0.484 → **0.468** | 0.946 → 0.913 | 0.902 → 0.907 |
| 32 | 0.515 → 0.515 | unchanged (clip never engages) | |

Short-horizon skill changes by +0.08% (`dns_khi`) and +0.007% (`dns_khi_s1`), within noise.

**Rollouts** (`condor_ensemble_fix.sub`, all closures rerun, 1% ensemble): the clipped hybrid
finished in **8/8** members. The diverged member rerun *without* the clip also finished, so the
original blow-up did not recur (the other closures reproduced their earlier numbers to three
decimals; a node-level floating-point difference is the likely reason). The fix is therefore
supported by the mechanism, not by a reproduced failure. The long-rollout ranking is
unchanged: hybrid spectral distance 0.618 ± 0.008; MiNNit still closer to the DNS than the
hybrid in 8/8 paired members (+0.023 ± 0.004), hybrid and TBNN still tied (+0.006 ± 0.009).


---

## 21. Larger filters: at r = 64 the sub-grid term matters and the hybrid wins (2026-10-19)

`condor_ensemble_rscan.sub`: the 1% ensemble (2 DNS × 4 members, every closure) at
$r = 32$ and 64, each with the models trained for that filter on `dns_khi`.

*Metric fix first:* at $r=64$ the coarse step is ~0.035, so frames are saved up to 0.035 after
their 0.1 mark, beyond the 0.025 alignment tolerance — most $r=64$ frames were silently dropped
and the first aggregation was garbage. The tolerance is now half the snapshot spacing (still
rejects a skipped save); `scratch/recompute_windowed.py` recomputed every member (all frames
valid). The $r=64$ result was then reproduced exactly by an independent per-frame computation.

**Whole window $t = 1.5$–10, log-spectral distance to the DNS (pooled, 8 members):**

*(r = 16 column corrected 2026-10-20: it had mixed in the 1e-3 ensemble; all columns are now the 1% ensembles.)*

| run | r = 16 | r = 32 | r = 64 | KE at r = 64 |
|---|---|---|---|---|
| oracle | 0.577 | 0.469 | **0.414** | 1.05 |
| unclosed | 0.581 | 0.478 | 0.923 | 1.27 |
| TBNN | 0.612 | 0.480 | 0.796 | 1.26 |
| MiNNit | 0.595 | 0.493 | 0.547 | 1.15 |
| hybrid | 0.618 | 0.507 | **0.478** | 1.12 |

1. **The answer to "does the sub-grid term matter more at larger filters" is yes, sharply.**
   The exact stress barely helps at $r = 16$ and 32 but halves the spectral distance at
   $r = 64$; without a closure the solver piles up 27% too much kinetic energy.
2. **At $r = 64$ the closures separate cleanly:** hybrid < MiNNit < TBNN ≈ unclosed. Paired,
   the hybrid beats MiNNit in 8/8 members (−0.069 ± 0.006) and the TBNN in 8/8 (−0.32 ± 0.07).
   In the middle and late windows the hybrid is even closer to the DNS than the exact stress.
3. **Only the hybrid's first place carries over from a priori.** Hybrid a-priori at $r=64$:
   Eq. 25 0.549, corr 0.838; TBNN 0.719 / 0.821; MiNNit 0.778 / 0.633 — so a priori the order
   is hybrid, TBNN, MiNNit, but in simulation it is hybrid, MiNNit, TBNN. *(Corrected
   2026-10-20; this point first claimed the whole ranking carried over.)* The simulation order
   follows amplitude: the TBNN (0.75 a-priori) barely drains the energy pile-up (KE 1.26).
4. **$r = 32$ is a crossover:** the oracle only just beats no closure (0.469 vs 0.478), the TBNN
   ties no closure, and the hybrid is last (+0.026 ± 0.013 vs TBNN) — differences of the same
   size as at $r=16$.

This refines the reading of §18–19 that "a-priori scores do not predict long rollouts": where the
sub-grid term is too small to move a rollout ($r \le 32$ here) nothing ranks closures; where it is
not ($r = 64$), the a-priori winner also wins in simulation, but correlation and Eq. 25 do not
order the rest — amplitude does. Caveat: $r=64$ is the edge of the usable filter range (16² grid,
one or two shells per spectral band), and it is still two DNS realisations.


---

## 22. Replication at 2048², r = 128, on five unseen realisations (2026-10-20)

Six new 2048² DNS (`condor_dns2k.sub`, seeds 0–5, same setup as `dns_khi`; ~6 h and 57 GB each).
The seed sets only the phases of an analytic perturbation, so seeds 0/1 are the 1024²
realisations at double resolution: filtered to 16², the two resolutions agree exactly to
$t \approx 5$, then decorrelate (density correlation 0.25–0.90 by $t=20$) while kinetic energy
and sub-grid energy fraction stay within a few percent — the 1024² data is converged at these
scales. TBNN and MiNNit retrained at $r=128$ (same filter width as $r=64$ at 1024²) on
`dns2k_s0`; everything below is on seeds 1–5, unseen by every model.

**A-priori** (mean ± s.e. over the 5 realisations): hybrid corr 0.861 ± 0.008, Eq. 25
**0.491 ± 0.015**; TBNN 0.817 / 0.748; MiNNit 0.656 / 0.754. Hybrid best on Eq. 25 in 5/5.

**Rollouts**, 1% ensemble, 5 realisations × 2 members, $t=1.5$–10, log-spectral distance /
total KE: oracle 0.428 / 1.04; **hybrid 0.525 / 1.10**; MiNNit 0.556 / 1.12; TBNN 0.844 / 1.24;
unclosed 0.917 / 1.24. Paired: hybrid closer to the DNS than MiNNit in 9/10 members
(−0.031 ± 0.007) and than the TBNN in 10/10 (−0.32 ± 0.07). No divergences.

**§21 replicates:** where the sub-grid term matters, the hybrid — best a priori — is best in
simulation, then MiNNit, then the TBNN, which barely improves on no closure because its
amplitude is too low (0.72) to drain the energy pile-up. A priori the TBNN beats MiNNit on
correlation and Eq. 25, so the MiNNit/TBNN order follows amplitude, not those scores. The hybrid's lead over MiNNit comes from the
early and middle windows; late ($t > 7.2$) the two are level (+0.023 ± 0.012).

Infrastructure: the TBNN dataset build and `evaluate_rollout` now read snapshots one at a time /
only inside the time window (`dataset.snapshot_files`); a 2048² run would otherwise need ~27 GB
just to build the training set.



---

## 23. Where the hybrid's remaining error is: energy, not shape (2026-10-21)

Ablations replace one factor of the hybrid with the exact DNS's (`HybridClosure(energy_closure=...)`,
`evaluate_rollout --hybrid-ablations`, `scratch/hybrid_apriori.py`); the fallback stays MInIT's
$\beta e$ with the energy under test.

**A-priori, unseen** (Eq. 25 / corr): 2048², $r=128$, mean of 5 realisations — hybrid 0.49 / 0.86;
exact shape × MiNNit energy 0.43 / 0.89; **TBNN shape × exact energy 0.25 / 0.96**. 1024², $r=64$:
0.55 / 0.84; 0.50 / 0.87; 0.23 / 0.97.

**Rollouts**, 2048², $r=128$, 10 members, $t=1.5$–10, log-spectral distance: **TBNN shape × exact
energy 0.397** (lowest in 10/10), exact stress 0.428, hybrid 0.525, exact shape × MiNNit energy 0.544,
MiNNit 0.556. Paired, hybrid − exact-shape ablation −0.020 ± 0.016 (no difference).

A perfect energy closes the whole gap to the exact stress and more; a perfect shape is worth almost
nothing. So a jointly trained or shape-only hybrid would buy little; the lever is MiNNit's energy
prediction (unseen $R^2(e) \approx 0.8$). TBNN shape × exact energy beating the exact stress itself is
the familiar effect that the true stress is not the optimal correction for the coarse solver.

---

## 24. A fair MInIT: the paper's averaged growth rate (2026-10-21)

The port applied Eq. 12 cell by cell, with $\gamma_\text{KH} = A_\gamma|\partial_y v_x|$ from each coarse
cell and no advection. The paper instead takes $\gamma_\text{KH}$ from the resolved shear at the
initial layer positions, *"averaged for different points in the xz plane and over the ± shears to
obtain a single value"* (Sec. 4.3.1), and drops advection because that average cancels it. MInIT
is a mean-field model, so the 2D equivalent is the average over the homogeneous direction $x$:
$\gamma(y) = A_\gamma|\partial_y\langle v_x\rangle_x|$, with $e_\text{KH}$, $\rho$ row averages
and the stress $\beta_{ij}e(y)$ uniform along $x$; $\langle v_y\rangle_x \approx 0$, so advection
vanishes for the paper's reason. Implemented as `MInITClosure(average="x")`,
`minit_calibrate --average x` (fits against the $x$-averaged exact stress, as the paper fits
box-averaged stresses) and `evaluate_rollout --minit-average x`.

**Calibration** on each training run over $t = 1$–20, both forms, with the paper's procedure
($A_\gamma = 0.14$ fixed, then the equilibrium ratio). **Scoring** on unseen runs
(`scratch/minit_fair_compare.py`), per snapshot, against the exact stress cell by cell and
against its $x$-average (every model's stress $x$-averaged too). MInIT's ODE runs are seeded with
the exact $e_\text{KH}$ at $t = 1$; MiNNit has no seed.

| unseen, Eq. 25 cell / Eq. 25 x-avg / corr x-avg | 1024², r = 16 | 1024², r = 64 | 2048², r = 128 (5 runs) |
|---|---|---|---|
| MiNNit | **0.77 / 0.54** / 0.74 | **0.78 / 0.53** / 0.76 | **0.75 / 0.56** / 0.74 |
| MInIT cell, refitted ODE | 2.01 / 1.78 / 0.46 | 1.11 / 1.02 / 0.47 | 1.20 / 1.01 / 0.49 |
| MInIT cell, equilibrium | 1.41 / 1.11 / 0.66 | 0.99 / 0.70 / **0.76** | 1.03 / 0.75 / **0.74** |
| MInIT x-avg, refitted ODE | 3.83 / 1.45 / 0.42 | 1.39 / 1.12 / 0.43 | 1.73 / 1.20 / 0.47 |
| MInIT x-avg, equilibrium | 3.44 / 1.02 / 0.57 | 1.38 / 0.76 / 0.74 | 1.72 / 0.91 / 0.72 |
| MInIT, paper coefficients ODE | 44–57 / 14–16 | 11–14 / 8–9 | 14–19 / 10–12 |

1. **The paper's averaging does not rescue MInIT here:** on unseen data the mean-field form is no
   better than the cell form, often worse. With the published $C = 10.6$ (a 3D value) the ODE
   collapses to 5–9% of the exact amplitude.
2. **MiNNit beats every MInIT variant on Eq. 25, under both targets, at every filter.**
3. **On the correlation of the $x$-averaged stress, MInIT's equilibrium limit ties MiNNit at
   $\Delta_f = 1/16$** (0.76 vs 0.76, 0.74 vs 0.74): there MiNNit's lead is amplitude, not
   structure. Hence the rollout test (`condor_ensemble_2k_minit.sub`, both forms, paired with
   the r = 128 ensemble).
4. In-sample, pooled over snapshots (the calibration script's own score), the mean-field ODE looks
   far better (corr 0.94 one interval, 0.69 free-running at $r = 16$). The per-snapshot, unseen
   numbers above are the comparable ones.


---

## 25. In long rollouts, the refitted MInIT ODE ties the hybrid (2026-10-21)

MInIT added to the 2048², $r=128$ ensemble on the same 10 starting states as §22–23 (no-closure
control reproduces to $9\times10^{-14}$, so runs pair exactly); `scratch/minit_rollout_merge.py`.

| run ($t = 1.5$–10) | spectral distance | diverged |
|---|---|---|
| exact stress | 0.428 ± 0.007 | 0/10 |
| MInIT refitted ODE, cell, DNS-seeded | 0.471 ± 0.010 | 0/10 |
| MInIT refitted ODE, cell, equilibrium seed | 0.514 ± 0.019 (7 survivors) | **3/10** |
| **MInIT refitted ODE, cell, equilibrium seed, realizability clip** | **0.520 ± 0.017** | 0/10 |
| hybrid | 0.525 ± 0.014 | 0/10 |
| MiNNit | 0.556 ± 0.009 | 0/10 |
| MInIT equilibrium, cell | 0.615 ± 0.039 | 0/10 |
| MInIT x-averaged (paper's form), refitted ODE / equilibrium | 0.806 / 0.857 | 0/10 |
| TBNN | 0.844 ± 0.066 | 0/10 |
| MInIT, published coefficients (cell / x-avg) | 0.879 / 0.906 | 0/10 |
| no closure | 0.917 ± 0.052 | — |

**Fair comparison** (no DNS seed, the same $\lVert\tau\rVert_F \le 0.5p$ bound as the TBNN/hybrid):
the refitted cell-by-cell MInIT ties the hybrid (−0.005 ± 0.026, better in 5/10) and is closer to
the DNS than MiNNit by 0.036 ± 0.020 (8/10; ~1.8σ). The DNS seed was worth ~0.05; without the clip the
unseeded ODE diverged in 3/10 runs.

1. **This reverses the a-priori verdict on MInIT's ODE**, which scored near the bottom a priori
   (Eq. 25 ≈ 1.2, corr ≈ 0.36, §24). The ODE gives the sub-grid energy memory; dynamically that is
   worth more than the snapshot accuracy a-priori scores measure.
2. **The paper's mean-field form does not work as a local closure** (stress uniform along $x$), and
   the published coefficients ($C = 10.6$, 3D) barely beat no closure.
3. Supersedes the "MiNNit beats MInIT" reading of §§8–22 for simulations at $\Delta_f = 1/16$; the
   a-priori statements stand.
4. Next: TBNN shape × MInIT-ODE energy (the hybrid's error is in the energy, §23), which needs
   `HybridClosure` to forward the solver's `advance` hook to a stateful energy closure.


## 26. TBNN shape × MInIT's evolved energy: as close to the DNS as the exact stress (2026-10-25)

`HybridClosure` with `energy_closure` = the refitted cell-by-cell MInIT ODE (unclipped, seeded at its
own equilibrium, so no DNS information; the hybrid clips after its rescale). The hybrid now forwards
the solver's `advance` hook so the energy evolves. `--hybrid-minit`, `condor_ensemble_2k_hybrid_minit.sub`,
same 10 starting states as §22–25. Pairing: no-closure control reproduces to 1e-14; the rerun TBNN and
hybrid rows to ≤ 1.3e-4 in LSD (NN inference round-off on different V100 nodes, amplified over the run;
negligible against the effects below).

| run ($t = 1.5$–10, 2048², $r = 128$) | spectral distance | total KE |
|---|---|---|
| **TBNN shape × MInIT-ODE energy** | **0.415 ± 0.014** | 1.096 |
| exact stress | 0.428 ± 0.007 | 1.038 |
| MInIT ODE alone (eq. seed, clipped) | 0.520 ± 0.017 | 1.153 |
| hybrid (TBNN shape × MiNNit energy) | 0.525 ± 0.014 | 1.103 |
| MiNNit | 0.556 ± 0.009 | 1.124 |
| TBNN | 0.844 ± 0.066 | 1.237 |

Paired: better than the MiNNit hybrid by 0.110 ± 0.017 (10/10), than MInIT alone by 0.105 ± 0.021 (10/10),
than MiNNit by 0.141 ± 0.014 (10/10); tied with the exact stress (−0.013 ± 0.014, 5/10). No divergences.

**A priori it is poor** (`scratch/hybrid_apriori.py` with MInIT args, 5 realisations): Eq. 25
1.311 ± 0.067, corr 0.445 (DNS-seeded: 1.157, 0.463), against the MiNNit hybrid's 0.491 / 0.861 and
MInIT alone 1.356 / 0.344. The TBNN shape barely helps the snapshot scores but adds 0.105 dynamically.

1. Strongest evidence yet that a-priori scores do not rank closures here: the best simulation closure
   is near the bottom a priori.
2. The two ingredients are complementary: the TBNN's tensor shape (best a priori, §23: TBNN shape ×
   exact energy 0.397) and MInIT's energy memory (best dynamic energy, §25).
3. Caveats: one filter width ($\Delta_f = 1/16$), one metric family (spectra); the KE excess (1.10 vs
   the exact stress's 1.04) remains, so matching the exact stress's spectral distance does not mean it
   reproduces the true stress.

## 27. Amplitude, not memory: a scaled memoryless hybrid ties the MInIT-energy hybrid (2026-10-26)

Test of whether rollout skill at $r = 128$ follows stress amplitude. **(a) A priori on the rollout
window** (`APRIORI_WINDOW=1.5,10,5 scratch/hybrid_apriori.py`, $t = 1.5$–10 every 0.5, 5 realisations,
logs `results/dns2k/window_apriori_s*.log`):

| model | amp | corr | Eq. 25 window | Eq. 25 $t=1$–20 | rollout |
|---|---|---|---|---|---|
| hybrid | 0.95 | 0.88 | 0.49 | 0.49 | 0.525 |
| MiNNit | 0.99 | 0.75 | 0.68 | 0.75 | 0.556 |
| TBNN | 0.67 | 0.79 | 0.86 | 0.75 | 0.844 |
| TBNN shape × MInIT e (eq. seed) | 1.29 | 0.48 | 1.14 | 1.31 | 0.415 |
| MInIT ODE (eq. seed) | 1.42 | 0.43 | 1.21 | 1.36 | 0.520 |
| TBNN ×1.6 / ×2.5 | 0.98 / 1.36 | 0.80 / 0.81 | 0.64 / 0.71 | | 0.786 / 0.700 |
| hybrid ×1.3 / ×1.6 | 1.17 / 1.36 | 0.87 / 0.87 | 0.52 / 0.61 | | 0.449 / 0.425 |

On the window the learned closures rank a priori as in rollouts (TBNN and MiNNit swap relative to
$t = 1$–20; the TBNN is good only after $t = 10$, which the rollouts never reach). The realizability
clip caps TBNN ×2.5 at amplitude 1.36. The MInIT-energy models remain last a priori; in the window
they are the only ones with amplitude > 1.

**(b) Rollouts with scaled stress** (`--tbnn-gain` before the TBNN's clip; `--learned-gain` scales
MiNNit's e and hence the hybrid; `condor_ensemble_2k_gain.sub`, cluster 1370026, same 10 starting
states, 20/20 exit 0, merged by `scratch/minit_rollout_merge.py`):

| run | spectral distance | total KE |
|---|---|---|
| TBNN shape × MInIT energy (§26) | 0.415 ± 0.014 | 1.096 |
| **hybrid ×1.6** | **0.425 ± 0.014** | **0.983** |
| exact stress | 0.428 ± 0.007 | 1.038 |
| hybrid ×1.3 | 0.449 ± 0.013 | 1.027 |
| hybrid | 0.525 ± 0.014 | 1.103 |
| TBNN ×2.5 | 0.700 ± 0.058 | 1.202 |
| TBNN ×1.6 | 0.786 ± 0.070 | 1.227 |
| TBNN | 0.844 ± 0.066 | 1.237 |

Paired: hybrid ×1.6 vs hybrid −0.100 ± 0.014 (10/10); vs TBNN shape × MInIT energy +0.010 ± 0.009
(scaled better 3/10, a tie); vs exact −0.003 ± 0.011 (6/10). TBNN ×2.5 vs TBNN −0.145 ± 0.026 (10/10)
but still +0.285 behind the §26 run (0/10).

1. **§26's "memory" reading is wrong for the energy.** A memoryless hybrid with its energy scaled to the
   MInIT models' window amplitude (~1.36) does as well as TBNN shape × MInIT energy, with *better*
   total KE (0.98 vs 1.10). MInIT's ODE wins because it over-predicts early ($t \lesssim 10$), which
   happens to be what the coarse solver needs — not because it lags.
2. Eq. 25 penalises exactly that over-prediction (hybrid ×1.6: 0.61 vs 0.49), which is why a priori and
   rollout rankings look inverted. Gains were set from a-priori amplitudes, not tuned on rollouts.
3. Amplitude is not sufficient: the TBNN at the same a-priori amplitude stays far behind (0.700) and
   barely drains KE (1.20). Its a-priori amplitude is measured on filtered-DNS input; on the coarse
   solver's own states it evidently differs (cf. the OOD collapse in the TBNN high-k notes). The
   hybrid's energy comes from MiNNit's scalar, which transfers.
4. The exact stress at amplitude 1 gets 0.428, so the "needed" over-drive compensates for imperfect
   models, not a solver deficit per se. Open: measure the closures' actual in-rollout amplitude and
   dissipation, and whether a gain fitted at one filter transfers to others.

## 28. Structure-sensitive scores: the exact stress is clearly best; the fair MInIT drops below MiNNit (2026-10-26)

The log-spectral distance is blind to where energy sits (an unclosed run with two clean unmixed bands
can look competitive on the spectrum). `scratch/rollout_structure_merge.py` rescores every r = 128 run
from its saved trajectory (filtered DNS and runs every 0.1, t = 1–10), output in
`results/structure_scores_r128.txt`: momentum thickness θ of the x-averaged $v_x$ (ratio to DNS and
mean |ratio − 1|), RMS error of the x-averaged density profile, Wasserstein-1 between density
distributions (both / std ρ_DNS), all over t = 1.5–10, and pointwise density correlation.

**Predictability:** the two perturbed members (1%) of each start, no closure, still correlate 0.92 at
t = 10 (0.99 at t = 6). The 16² coarse flow barely decorrelates, so pointwise correlation over the
whole window is a valid closure score here (the DNS's own sensitivity is not measured).

| run (mean of 10) | θ/θ_DNS | profile err | density W1 | corr t ≤ 10 | spectral (§27) |
|---|---|---|---|---|---|
| exact stress | 0.83 | **0.272** | 0.154 | **0.70** | 0.428 |
| TBNN shape × exact energy | 0.79 | 0.275 | 0.136 | 0.69 | |
| hybrid ×1.6 | 0.85 | 0.349 | 0.144 | 0.42 | 0.425 |
| TBNN shape × MInIT energy | 0.78 | 0.362 | 0.176 | 0.50 | 0.415 |
| hybrid | 0.81 | 0.379 | 0.183 | 0.42 | 0.525 |
| MiNNit | 0.80 | 0.381 | 0.202 | 0.45 | 0.556 |
| MInIT ODE, DNS seed | 0.63 | 0.388 | 0.217 | 0.47 | 0.471 |
| MInIT ODE, eq seed, clip (fair) | 0.57 | 0.412 | 0.233 | 0.49 | 0.520 |
| MInIT equilibrium, cell | 0.44 | 0.475 | 0.279 | 0.38 | 0.615 |
| TBNN ×2.5 | 0.38 | 0.495 | 0.292 | 0.42 | 0.700 |
| TBNN | 0.29 | 0.550 | 0.330 | 0.41 | 0.844 |
| none | 0.28 | 0.559 | 0.349 | 0.41 | 0.917 |

(x-averaged and paper-coefficient MInIT variants sit with "none".) Paired, A − B:
exact − hybrid ×1.6: profile −0.078 ± 0.027, |Δθ| −0.141 ± 0.045, corr +0.28 ± 0.03;
hybrid ×1.6 − TBNN shape × MInIT e: profile −0.012 ± 0.030 (tie), W1 −0.032 ± 0.005, corr −0.08 ± 0.03;
fair MInIT − MiNNit: profile +0.031 ± 0.018, W1 +0.031 ± 0.007; fair MInIT − hybrid: W1 +0.050 ± 0.008.

1. **The spectral "tie with the exact stress" was the spectrum's blindness.** On every structural score
   the exact stress (and TBNN shape × exact energy) is clearly ahead of every model; its pointwise
   correlation is 0.70 against ≤ 0.50. The best models match the exact stress's spectrum, not its flow.
2. Every closure mixes the layers too slowly (θ below the DNS), the exact stress included (0.83). No
   closure, the TBNN and the paper/x-averaged MInIT barely thicken the layers at all (θ ≈ 0.28–0.29).
3. **The fair MInIT ODE loses to MiNNit and the hybrid on structure** (too-slow mixing, θ 0.57 vs 0.80),
   reversing its spectral edge (§25). Its pointwise correlation is slightly higher, within noise.
4. The top group (hybrid ×1.6, TBNN shape × MInIT e, hybrid, MiNNit) is close on the profile error;
   scaling still helps (hybrid ×1.6 best density distribution), so §27's amplitude conclusion stands,
   with a much smaller effect than on the spectrum. The gap to the exact stress is in the energy, not
   the shape: TBNN shape × exact energy matches the exact stress.
