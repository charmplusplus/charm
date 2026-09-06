#!/bin/bash
# lbfix.sh <jobid> [tags...]
#   tags: <input>_<lb>[_dbg]   input in uni|clu, lb in noLB|sync|async
#   _dbg adds +LBDebug 1 (balancer decisions in the log; a slightly slower
#   step because the debug path takes the PE-0 barrier), the plain tag is the
#   timing run. CHARM_LB_LOADDUMP is always on: it costs nothing and shows
#   per-PE object counts and GPU sums at every step.
J=$1; shift
B=/u/bhosale/charm-reconverse/examples/charm++/barnes; cd $B || exit 1; ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH" IPATH_NO_BACKTRACE=1 FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=1024 CHARM_LB_LOADDUMP=1
NODES=${NODES:-1}; NT=$((NODES * 4))
UNI="-in=big.bin -killat=30 -b=128 -p=700 -blockmap=1 -qd=0 -let=1 -devwalk=1 -devexch=1 -devlet=1"
CLU="-in=clustered.bin -killat=15 -b=128 -p=2048 -blockmap=1 -qd=0 -let=1 -devwalk=1 -devexch=1 -devlet=1"
D="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim ${EXTRA:-}"
IPC="+gpushm +gpupool +gpuipceventpool 256"
OUT=$B/check/lbfix_results.txt; : > $OUT
for tag in "$@"; do
  IFS=_ read -r inp lb dbg <<< "$tag"
  case $inp in uni) BASE="$UNI";; clu) BASE="$CLU";; *) echo "bad input $inp"; exit 2;; esac
  case $lb in noLB) LB="";; sync) LB="-lbperiod=5 $D";; async) LB="-lbperiod=5 -lbasync=1 $D +LBAsync";; *) echo "bad lb $lb"; exit 2;; esac
  [ "$dbg" = dbg ] && LB="$LB +LBDebug 1"
  L=$B/check/lbfix_${tag}.log
  PES=4 timeout ${TMO:-200} srun --unbuffered --jobid=$J --mpi=cray_shasta -N $NODES -n $NT --ntasks-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact stdbuf -oL -eL ./numa_wrap.sh ./barnes $BASE +p 4 $LB $IPC > $L 2>&1; rc=$?; sleep 1
  avg=$(grep -oP 'finished all [0-9]+ iterations with avg time \K[0-9.]+' $L | tail -1)
  en=$(grep -oP 'energy \K[-0-9.]+' $L | tail -1)
  migs=$(grep -oP 'cross node migrations AFTER LB: \K[0-9]+' $L | paste -sd+ | bc)
  zero=$(grep -c 'gpu_sum=0.000000' $L)
  dumps=$(grep -c 'LBLOAD pe=0\]' $L)
  verdicts=$(grep -oP '\-> \K(REVERT|balance)' $L | sort | uniq -c | awk '{printf "%s=%s ", $2, $1}')
  spread=$(grep 'LBLOAD' $L | tail -$((NT*4)) | grep -oP 'objs=\K[0-9]+' | sort -n | awk 'NR==1{mn=$1} {mx=$1} END{printf "%s..%s", mn, mx}')
  ok=1; [ "$rc" = 0 ] && [ -n "$avg" ] || ok=0
  echo "$tag ok=$ok rc=$rc avg_iter=${avg:-NONE} energy=${en:-NONE} xnode_migs=${migs:-0} zeroGpuPEs=$zero/$((dumps*NT*4)) objs_last=$spread verdicts=[${verdicts}]" | tee -a $OUT
  if [ $ok = 0 ]; then echo "STOP at $tag"; grep -anE 'Reason:|Fatal|Abort|outstanding for|WARNING|Need more|Segmentation|ERROR' $L | head -6 | cut -c1-200; echo "== tail: $(tail -2 $L | cut -c1-120 | tr '\n' '|')"; exit 1; fi
done
echo "ALL PASSED"
