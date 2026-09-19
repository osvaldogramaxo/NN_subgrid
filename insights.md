# Rollout Evaluation & Diagnostic Insights

**Simulation Case:** `test1` (Relativistic Kelvin-Helmholtz Instability)  
**Resolution:** Fine $1024 \times 1024 \to$ Coarse $256 \times 256$ ($r = 4$)  
**Numerical Setup:** MP5 reconstruction + HLL Riemann solver, CFL = $0.4$, $\mathrm{SSP}$-$\mathrm{RK}3$  
**Evaluation Job:** Condor `1314275.0` (evaluated on NVIDIA A100 GPU)

---

## 1. Key Observations from Rollout & Animation

### Visual Trajectory Comparison

The time-evolution snapshots across the three models reveal markedly distinct dynamics:

| Time | Filtered DNS Ground Truth | TBNN-Closed Coarse Solver | Exact DNS-Closed Solver (Oracle) |
| :--- | :--- | :--- | :--- |
| **$t = 0.0$** | Sharp horizontal shear layer | Same filtered initial state | Same filtered initial state |
| **$t = 2.5$** | **Vortex Roll-up**: Prominent mode-2 cat's-eye billows with fine filaments | **Suppressed**: Interface remains flat; negligible curl | **Suppressed**: Virtually identical to TBNN; flat interface |
| **$t = 5.0$** | **Vortex Pairing & Mixing**: Secondary instabilities break layer into turbulence | **Damped**: Tiny, squashed laminar curls at interface | **Damped**: Tiny, squashed laminar curls; identical to TBNN |
| **$t = 7.5$** | **Turbulent Layer**: Broad turbulent mixing band across midplane | **Smooth Waves**: Billows fail to break down; acoustic oscillations dominate | **Smooth Waves**: Smeared wave pattern; nearly identical to TBNN |
| **$t = 10.0$** | **Full Domain Turbulence**: Central slab disrupted into fine filamentary vortices | **Frozen Standing Wave**: Smooth, unmixed laminar standing wave | **Global Shear Tear**: Sharp diagonal discontinuity & phase mismatch |

### Quantitative Error Growth

- **$t = 0 \to 1.0$:** Errors remain small ($\rho$ rel $L_2 < 3\%$).
- **$t = 1.0 \to 2.5$:** Errors surge from $3\%$ to **$\sim 20\%$** in both solvers simultaneously. This jump occurs because the DNS rolls up into non-linear billows while both coarse solvers remain flat.
- **$t = 2.5 \to 7.5$:** Both TBNN and Exact DNS track each other with nearly identical errors ($\sim 18\text{--}21\%$).
- **$t = 8.0 \to 10.0$:** 
  - **TBNN:** Final error reaches **$38.1\%$** for $\rho$ and **$110.9\%$** for $v_x$ as DNS enters domain-wide turbulent breakdown while TBNN remains laminar.
  - **Exact DNS:** Final error climbs to **$54.6\%$** for $\rho$ and **$86.3\%$** for $v_x$ due to the coordinate-phase tear.

### Kinetic Energy Spectrum $E(k)$ Analysis

The kinetic energy spectrum at $t = 10.0$ pinpoints the underlying issue:

1. **Filtered DNS (Ground Truth):** Exhibits a continuous, broad-band turbulent power-law cascade extending all the way to $k \sim 100$.
2. **Exact DNS (Oracle):** Retains kinetic energy matching DNS down to $k \sim 40$, with moderate falloff at high $k$ purely due to numerical diffusion on the $256 \times 256$ mesh.
3. **TBNN Closure:** Collapses precipitously for $k > 10$, plunging **$4$ to $5$ orders of magnitude below DNS** at $k \sim 100$ ($E(k) \approx 10^{-6}$).

The TBNN model acts as an **excessively strong dissipative sink** (effective turbulent viscosity is far too high), extinguishing high- and intermediate-wavenumber modes and freezing the flow into a laminar state.

---

## 2. Root Causes of the Observed Behavior

### A. A Priori vs. A Posteriori Distribution Shift

- **A Priori Training Distribution:** The network was trained on Favre-filtered snapshots from fully active and developing turbulence, where gradients $S_{\mu\nu}, \Omega_{\mu\nu}$ and invariants $\Lambda_1 \dots \Lambda_{15}$ have rich spatial distributions.
- **A Posteriori Rollout Reality:** At $t = 0$, the flow is laminar with a smooth shear layer and a weak initial perturbation ($v_y \sim 10^{-2}$). Because the learned closure is predominantly dissipative, it dampens the initial perturbation before the linear Kelvin-Helmholtz instability can reach the non-linear roll-up amplitude.
- In transitional shear flows, classical LES closures (e.g., standard Smagorinsky) suffer from this exact same failure mode unless modulated by a dynamic Germano procedure or transition sensor.

### B. Fluid-Frame Projection vs. Lab-Frame Conservation Laws

- **Comoving Rest-Frame Projection:** The network is trained on:
  $$\tau_\perp^{\mu\nu} = h^\mu_\alpha h^\nu_\beta \tau_{\text{SGS}}^{\alpha\beta}$$
  which is purely spatial in the comoving fluid frame ($u_\mu \tau_\perp^{\mu\nu} \equiv 0$).
- **Lab-Frame Integration:** The RHD solver updates conserved variables $(D, S_x, S_y, \tau)$ in the lab frame governed by $\partial_\mu T^{\mu\nu} = 0$.
- In the lab frame, the full subgrid stress $\tau_{\text{SGS}}^{\mu\nu} = \overline{T^{\mu\nu}} - T^{\mu\nu}(\bar{U})$ possesses non-trivial time-space components $u^{(\mu} q^{\nu)}$ (subgrid energy fluxes and relativistic cross-terms). Injecting $-\partial_j \tau_\perp^{j\nu}$ directly as conservative source terms creates a structural mismatch between momentum diffusion and kinetic energy dissipation.

### C. Suppression of Backscatter

- Real turbulent cascades involve bi-directional energy transfer (forward cascade and inverse backscatter). In shear layer roll-up, Reynolds stresses feed energy back into the primary vortex mode.
- Localized realizability clamping and model conservatism constrain the predicted tensor to be positive-definite (purely dissipative), preventing energy injection into resolved scales.

### D. Why the Exact DNS Solver Failed at Late Times

- The Exact DNS model samples $\tau_{\text{SGS}}(t, x, y)$ from DNS snapshots at fixed coordinates $(x, y)$ and time $t$.
- Because the coarse grid delayed the initial roll-up, by $t \approx 8\text{--}10$ the physical vortices in the coarse simulation are in a completely different location than in the DNS.
- Imposing the true DNS subgrid stresses at mismatched physical locations creates destructive shearing forces that tear the coarse field along coordinate planes.

---

## 3. Recommended Next Steps

1. **Unclosed Coarse Baseline Benchmark:**
   - Execute a rollout with `sgs_model=None` under the exact same grid ($256 \times 256$) and numerical scheme (MP5 + HLL) to isolate the baseline numerical dissipation from the closure dissipation.
2. **Dynamic Scaling / Transition Sensor:**
   - Introduce a localized shear/vorticity sensor or dynamic Germano-style coefficient $\alpha(x, t) \in [0, 1]$ to scale down SGS dissipation during the laminar/transitional phase ($t < 3$) and activate closure only when local subgrid turbulence is established.
3. **Frame-Consistent Closure Formulation:**
   - Either formulate the conservative source term directly from the full lab-frame residual $\tau_{\text{lab}}^{\mu\nu} = \overline{T^{\mu\nu}} - T^{\mu\nu}(\bar{U})$, or apply the full Lorentz back-transformation when converting comoving stress $\tau_\perp^{\mu\nu}$ into conservative flux divergences.
