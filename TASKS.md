Next steps:

NOTE: Filter should not exceed inertial range

1- different filter sizes (up to 128) -> check correlation vs filter size (a priori)
2- A priori w/ NN and MiNNit 
3- Try to measure rel error (eq 25)


28/10/2026 meeting:

Conclusions:
1- Using MInIT to evolve the energy and an NN to get the shape of the stress looks the most promising
2- TBNN's energy issues are puzzling, worth an extra look

Next steps:

1- Set up a git, especially for the solver so Miquel can check
   ~1 h. No remote yet (only local `master`). Tracked files are 111 / 1.8 MB; data,
   checkpoints and results are already gitignored. src/solver.py depends only on
   src/dataset.GAMMA_EOS, and has tests/test_solver.py + SOLVER.md, so a solver-only repo is
   easy too. To decide: whole repo or solver only, and the host (GitHub or IFIC GitLab).
   `gh` is not installed: create the empty remote by hand, then push.

2- Figure out the Energy issue with the TBNN (change the loss function? Solver in the loop?)
   Two separate effects:
   - A priori the shrinkage is expected: an MSE-trained regressor predicts E[tau|inputs],
     for which amplitude = correlation exactly. 2048^2 s1: TBNN amp 0.72, corr 0.81.
     MiNNit trains on log e and gets amp 0.96.
   - In simulation it is not a global scale: TBNN x2.5 still scores 0.70 (no closure
     0.92, hybrids 0.42-0.53), while TBNN shape x another energy works. So the TBNN's
     energy is wrong on the simulation's own states, and differently in different cells.
   Plan, in order:
   a) Diagnostic (<1 day, no training): on the saved rollout trajectories, compare the
      TBNN's energy with MiNNit's, MInIT's and the exact one, per cell, over time, binned
      by |S|. scratch/tbnn_energy_diagnosis.py -> results/tbnn_energy_diag/summary.txt.
      DONE 28/10. The TBNN switches itself off on simulation states:
      - <e_TBNN>/<e_exact>: 0.66-0.74 on filtered-DNS states, 0.30-0.52 on the states of
        EVERY simulation, the exact-stress run included. Not the clip (unclipped: same).
      - On no-closure / TBNN-run states, e_TBNN ~ 0 (<= 5% of MiNNit's mean) in 70-75% of
        cells, vs 1-4% on DNS states after t = 4. Hence TBNN run ~ no closure, and why a
        global gain (x2.5) cannot fix it.
      - MiNNit instead follows the resolved strain (exact run: <rho lam^2 |S|^2> 0.64-0.72
        of DNS, e_MiNNit 0.54-0.62 of exact).
      - Cause: inputs. The acceleration invariants (Lambda11-13) and theta spread widely on
        filtered DNS (p10-p90 ~ 2-3 std) but are almost constant on simulation states; the
        exact-stress run narrows them 3-5x too. The TBNN uses them as a cue for sub-grid
        energy. Setting them (and Mach) to their simulation value on DNS states cuts e_TBNN
        to 0.64x -> 0.45 of exact, which matches what we see in simulation.
      - Reading: the filtered DNS carries pressure-gradient / compression signatures of the
        unresolved turbulence that the 16x16 solver does not produce. A train/test shift
        in the inputs, not a loss problem, so b) alone would not fix it.
      - Why the acceleration (28/10): a comes from the Euler equation, so it is grad p / rho h.
        Filtered-DNS pressure fluctuates 35% (vortex cores, corr(p, Q) -0.46); |a| tracks
        e_exact (corr +0.48). The no-closure 16x16 run never rolls up (rms vy 0.003 vs DNS
        0.186) and its pressure flattens from 5% to 1%, so |a| is 56x smaller. The exact-
        stress run rolls up (vy 0.145, p 28%), but its |a| is still half the DNS value. The
        cue is circular: the TBNN only predicts stress where roll-up has already happened,
        but in the coarse run the roll-up only happens if the closure drives it.
        MiNNit keys on strain, which the laminar shear already has.
      Next (replaces b as the first try, 1 condor retrain): TBNN without the acceleration
      invariants, theta and Mach (sigma/omega invariants only, like MiNNit); score a priori,
      rerun this diagnostic, then rollouts. c) is now better motivated: it trains on the
      solver's own states.
      SUBMITTED 28/10: condor 1371450, --drop-inputs 10-16 --drop-basis 9 10 11 ->
      checkpoints_2k_r128_noaccel. Dropping E9-E11 is safe here: in 2D the fluid-frame
      stress has 4 components (xx, yy, xy, zz), and the best per-cell fit of the exact tau
      (s1, t = 2..20) captures 100.000% with all 12 tensors, with E0-E8, and even with E0-E3
      (85% with E0-E2). The 12 tensors span only rank 4. NOT true in 3D or with B: there
      the a-free set is not complete (Pope needs 10 tensors for sigma, omega alone), so
      revisit before the MHD step.
      RESULT (28/10, trained 23 min, best epoch 343): DOES NOT FIX THE COLLAPSE.
      - A priori, mean over s1-s5 (results/dns2k/noaccel_apriori_s*.log):
        TBNN amp 0.72 -> 0.63, corr 0.82 -> 0.77, Eq25 0.75 -> 0.92.
        The shape is unchanged: hybrid corr 0.860 / Eq25 0.492 (was 0.861 / 0.491),
        TBNN shape x exact e Eq25 0.258 (was ~0.25).
      - Energy on simulation states (results/tbnn_energy_diag_noaccel/summary.txt): DNS
        0.56-0.65, no-closure 0.18-0.28 of exact. As bad as before, slightly worse.
      - The second cause is strain extrapolation. The no-closure run keeps its strain in
        thin, sharp shear sheets: 62% of its rho lam^2 |S|^2 sits in cells above the DNS's
        97th percentile (13% of cells, vs 3% in DNS). There, e/(rho lam^2 |S|^2) relative to
        the DNS-wide exact value is MiNNit 0.86, TBNN old 0.35, new 0.20. Even inside the DNS
        range, the TBNN's coefficient falls with strain (0.9 -> 0.5-0.65), while exact and
        MiNNit stay ~1. MiNNit has e = exp(g) rho lam^2 |S|^2 built in, so it scales with
        strain out of distribution; the TBNN must learn the scaling and does not extrapolate.
      Next idea: build the same scaling into the TBNN (tau = rho lam^2 |S|^2 x sum g_n E_n,
      with normalised E_n). That is close to the hybrid, which conclusion 1 already favours.
      DECIDED 28/10: skip the scaled TBNN. It would land near the MiNNit hybrid (0.525),
      while TBNN shape x MInIT energy (0.415) is within ~1 SE of the TBNN shape x exact e
      ceiling (0.397). Rollouts 1371455/6 cancelled. Next: solver in the loop (c).
      SHAPE NETWORK (28/10, condor 1371460, checkpoints_2k_r128_shape, --shape-only): predicts
      tau / tr(tau) (trace 1 via E0 = h), trained on shape x exact trace; no accel/theta/Mach.
      A priori, mean s1-s5 (results/dns2k/shape_apriori_s*.log), Eq25 vs the current TBNN shape:
        x exact e 0.246 -> 0.228 (5/5 better), x MiNNit e (hybrid) 0.491 -> 0.482 (5/5),
        x MInIT e (eq. seed) 1.311 -> 1.318 (+0.007, 5/5 worse, negligible).
      Guarded cells (rescale denominator < 0.25) 0.0-0.1%, vs 7.6-9% fallback before.
      As good or slightly better, and no fallback machinery. Use it as the frozen shape for (c).
      Before (c): redefine e = 1/2 tr tau (covariant; the hybrid becomes tau = 2 e shape, no
      division) and refit MInIT / MiNNit on it. MInIT is Newtonian; the lab-frame e was our port's
      literal transcription, ~6% above 1/2 tr tau on average.
   b) Loss (1-2 days, one condor retrain): add a log-energy term
      (log tr tau_pred - log tr tau_exact)^2 to the TBNN loss (src/losses.py). Accept only
      if rollouts improve: refit, gain and the Eq.25 loss all helped a priori and hurt
      in simulation.
   c) Solver in the loop (1-2 weeks): the solver is pure PyTorch (only dt goes through
      .item()), and at 16x16 backprop through it is cheap. Fine-tune from the current
      checkpoint on short windows (10-50 steps) against the filtered DNS. Risks: gradients
      through the MP5 limiters and the C2P Newton loop, blow-up over long windows. Do a
      one-step gradient check first. Train on s0 only; s1-s5 stay held out.
   Meeting conclusion 1 already takes the energy from elsewhere, so a)-b) are for
   understanding; c) only if the TBNN's own energy should end up in the paper.

3- Add magnetic fields, compare with Cook
   The largest item (weeks). Which Cook paper/dataset? (not referenced in the repo yet)
   - Solver: SRMHD needs B in the conserved variables, div B control (constrained transport
     or GLM cleaning), a new C2P (the hardest part), an HLL/HLLD flux. Re-verify it.
   - New DNS: 2048^2 is ~57 GB per realisation.
   - Redo the go/no-go first: does the exact stress clearly beat no closure?
   - Closures: the TBNN already has B invariants (Lambda6-10) and basis tensors
     (src/features.py takes b_con; zero so far); MInIT/MiNNit need the magnetic
     energy and the EMF term. MInIT's home ground is MHD.
   Do after 2, since it re-runs the whole pipeline.