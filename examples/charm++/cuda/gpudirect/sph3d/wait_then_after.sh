#!/bin/bash
# wait_then_after.sh <jobid>: once the head1 chain has printed its done/stopped line, run the leftover-minutes
# script after_head1.sh on the SAME allocation (no salloc here, ever). Launched detached by the session.
JID=$1
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d/logs/head1_chain.out
until grep -q '=== head1 done\|=== head1 stopped\|cannot read the job' $L 2>/dev/null; do
  sleep 20
  # the job expired or was cancelled before the chain finished: nothing to run on
  squeue -j $JID -h -o %T 2>/dev/null | grep -q . || { echo "job $JID is gone; nothing to run"; exit 0; }
done
if grep -q '=== head1 done' $L && squeue -j $JID -h -o %T 2>/dev/null | grep -q RUNNING; then
  bash /u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d/after_head1.sh $JID
else
  echo "chain stopped or job not running; after_head1 not started"
fi
