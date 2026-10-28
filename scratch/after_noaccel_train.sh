#!/bin/bash
# After the no-acceleration TBNN retrain (TASKS 28/10, condor 1371450) finishes: submit the
# rollouts, then score it a priori on s1-s5 and rerun the energy diagnostic on the login node.
# Launched with: nohup setsid scratch/after_noaccel_train.sh > results/noaccel_pipeline.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
JOB=1371450
CKPT=checkpoints_2k_r128_noaccel/best_model.pt
PY="./.venv/bin/python -u"
export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONPATH=.

echo "$(date) waiting for condor job $JOB"
condor_wait condor_logs/train2k_noaccel_$JOB.log $JOB
echo "$(date) training finished"
tail -5 condor_logs/train2k_noaccel_$JOB.out
if [ ! -f "$CKPT" ] || ! grep -Eq "Training (completed|early stopped)" condor_logs/train2k_noaccel_$JOB.out; then
    echo "training did not complete; nothing submitted" >&2
    exit 1
fi

condor_submit condor_ensemble_2k_noaccel.sub
condor_submit condor_ensemble_2k_noaccel_hybrid_minit.sub

for s in 1 2 3 4 5; do
    $PY scratch/hybrid_apriori.py 128 dns2k_s$s $CKPT results/dns2k/minit_coeff_2k_r128.pt \
        results/minit_fair/coeffs_2k_r128_cell.json results/dns2k/reduction_r128_s$s.pt \
        > results/dns2k/noaccel_apriori_s$s.log 2>&1
    echo "$(date) a priori s$s done"
done

mkdir -p results/tbnn_energy_diag_noaccel
$PY scratch/tbnn_energy_diagnosis.py $CKPT results/tbnn_energy_diag_noaccel \
    2>&1 | grep -v Warn > results/tbnn_energy_diag_noaccel/summary.txt
echo "$(date) energy diagnostic done; rollouts still running (merge: scratch/minit_rollout_merge.py)"
touch results/noaccel_pipeline.done
