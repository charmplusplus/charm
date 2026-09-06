#!/bin/bash
# run_ref.sh <jobid> <tag> [moe_ref.py arguments...]
# One A40 node, four ranks, one per GPU; logs in ranklogs/ref_<tag>/; prints
# the rank-0 summary. NODES=, TMO= (seconds, 600), PY= override the defaults.
J=$1; TAG=$2; shift 2
P=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/moe; R=$P/ref; cd $P || exit 1; ulimit -c 0
PY=${PY:-/sw/rh9.4/user/python/conda-env/pytorch-2.12.1-cu130/bin/python}
export RANKLOG_DIR=$P/ranklogs/ref_$TAG; mkdir -p $RANKLOG_DIR; rm -f $RANKLOG_DIR/*
NODES=${NODES:-1}; NT=$((NODES * 4))
timeout ${TMO:-600} srun --jobid=$J --mpi=cray_shasta -N $NODES -n $NT --ntasks-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact --unbuffered $R/refwrap.sh $PY $R/moe_ref.py "$@"; rc=$?
echo "rc=$rc  logs: $RANKLOG_DIR"
grep -h "^\[PyTorch\|^Experts\|^Tokens\|^Routing\|^Per expert\|^Steps\|^Step \|^Phases\|^Per-rank\|^Dispatch\|^Host\|^Placement\|^Total\|^Average" $RANKLOG_DIR/rank_0.log
if [ $rc != 0 ]; then for f in $RANKLOG_DIR/rank_*.log; do echo "== $f"; grep -n "Error\|error\|Traceback\|NCCL WARN\|Killed" $f | head -5; tail -3 $f | cut -c1-200; done; fi
