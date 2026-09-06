#!/bin/bash
cd /u/bhosale/charm-reconverse/examples/charm++/barnes
ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export IPATH_NO_BACKTRACE=1
export FI_MR_CACHE_MONITOR=disabled
export BARNES_PHASE_REPORT=1
JOB=${JOB:?}
B="-in=big.bin -killat=30 -b=128 -p=700 -blockmap=1 -qd=0 -let=1 -devwalk=1 -devexch=1"
for v in 1 0; do
  timeout 200 srun --jobid=$JOB -n 4 --cpus-per-task=8 --exact \
    ./numa_wrap.sh ./barnes $B -devlet=$v +p 4 > check/Q5$v.log 2>&1
  echo "devlet=$v exit=$? avg=$(grep -oP 'avg time \K[0-9.]+' check/Q5$v.log | tail -1) splicewarn=$(grep -c '\[LET\]' check/Q5$v.log)"
done
