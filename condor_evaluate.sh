# ==============================================================================
# condor_evaluate.sub
# HTCondor submit file for CRMHD-TBNN Rollout Evaluation Benchmark
# ==============================================================================

universe       = vanilla
executable     = ./run_evaluate.sh

# Arguments: <data_dir> <coarsening_factor> <checkpoint> <t_start> <t_end> <device> [<cfl>] [<reconstruction>] [<riemann_solver>] --backscatter-clip project --tau_smooth_passes
#
# --with-learned adds MInIT with a state-dependent coefficient (src.minit_learned,
# trained by src.train_minit_coeff). It is stateless and takes NO e_KH seed, unlike
# --with-minit, which is handed the exact filtered-DNS e_KH at t_start -- so the
# learned row competes on strictly less information. Repeat the flag to add the
# 'direct' output mode alongside 'residual'.
#
# NOTE on the checkpoint: checkpoints/best_model.pt was trained on t <= 6.4, so over
# t = 1..10 the TBNN extrapolates for most of the run. checkpoints_fullrange/best_model.pt
# covers the whole range and scores better on an unseen realisation (mean corr 0.832
# vs 0.741 amplitude, and a far gentler late decay). Swap it in if you want the TBNN
# row to be its best rather than its original self.
arguments = "dns_khi 16 checkpoints/best_model.pt 0.0 5.0 cuda 0.75 mp5 hllc --make-anim --save-history --anim-interval 0.1 \
--anim-fps 10 --with-unclosed --with-minit paper \
--with-learned results/minit/minit_coeff_residual.pt"

# Output log files
output         = ./condor_logs/eval_$(Cluster)_$(Process).out
error          = ./condor_logs/eval_$(Cluster)_$(Process).err
log            = ./condor_logs/eval_$(Cluster)_$(Process).log

# Environment
getenv         = True

# Resources
request_cpus   = 8
request_memory = 32768
request_gpus   = 1

queue 1
