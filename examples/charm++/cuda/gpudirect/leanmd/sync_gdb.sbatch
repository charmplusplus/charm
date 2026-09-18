#!/bin/bash
#SBATCH --job-name=lmdgdb
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:16:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# The leanmd N=2 weak SYNC arm exactly as scale.sh runs it (job 22163665
# segfaulted at the first LB step), every rank under gdb batch so the faulting
# rank prints its own stack. Two arms in one allocation: the default (leanmd
# posts its receive buffers as free) and LEANMD_ORDERED_RECV=1 (the runtime
# also orders each copy after earlier work on the posting stream).
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192 CHARM_LB_ASYNC_NO_PEER_DECISION=1
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH
ulimit -c 0
ROOT=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect
LMD=$ROOT/leanmd; RL=$LMD/ranklogs
NODES=$SLURM_JOB_NUM_NODES
SRUN="srun --jobid=$SLURM_JOB_ID --mpi=cray_shasta -N $NODES -n $((4*NODES)) --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --cpu-bind=none --exact --kill-on-bad-exit=1"
COST=$ROOT/sph2d/lbcost.delta-a40.2node.conf
C="16 8 8 100 20 20 -computemap local -density gradient +pe 64 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool +gpupoolsize 2048"
LB="+balancer MetisLB +balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim +LBCostConfig $COST +LBDebug 1 +LBMetisMaxCutRise 1.0"
echo "=== job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST  leanmd $(stat -c %y $LMD/leanmd | cut -c1-16)"
run() { local tag=$1; shift; local dir=$RL/gdb_N2_sync_$tag; rm -rf $dir; mkdir -p $dir
  echo "=== arm $tag ($*) start $(date +%T)"
  ( cd $LMD && env "$@" RANKLOG_DIR=$dir timeout 420 $SRUN stdbuf -oL -eL $RL/rankwrap_gdb.sh $LMD/leanmd $C $LB > $dir/srun.err 2>&1 )
  local rc=$?
  echo "=== arm $tag rc=$rc $(date +%T)  $(awk '/Step [0-9]+ Benchmark/{n=$2;t=$5;if(n>=42){s+=t;c++}} END{if(c)printf "mean42=%.1f",s/c; else printf "mean42=NA"}' $dir/rank_0.log)  migrating=[$(grep -ah 'Objects migrating' $dir/rank_0.log | awk '{printf "%s ",$7}' | tr -d ,)]"
  grep -ah -A 45 'FAULT rank' $dir/rank_*.log | head -80
  grep -ah 'Segmentation\|Aborted\|CANCELLED' $dir/srun.err | head -3; }
run default
run ordered LEANMD_ORDERED_RECV=1
echo "=== DONE $(date +%T)"
