#!/bin/bash
# submit_scale.sh ["1 2 4 8"] [APPS] [KINDS] [ARMS]
#
# Submits ONE job per node count, each sized exactly to that count, so no node
# is ever idle: a 1-node point runs in a 1-node job, not in a corner of an
# 8-node one. Every job runs the FULL matrix for its node count -- weak and
# strong, leanmd and sph2d, all three arms -- in its own allocation.
#
# Walltime comes from scale.sh DRY=1, which reports two sums: EXPECTED_SECONDS
# (what the arms take, from the 2026-09-15 N=1 measurements) and PLAN_SECONDS
# (the sum of their TIMEOUTS, 2-3.7x a healthy run). --time is sized from the
# expectation plus 35%, because sizing it from the timeouts asked for 148 min
# to do 73 min of work at N=1 and then CLAMPED below the plan, which silently
# dropped the last arms at DEADLINE_MIN. A run that overruns is still caught
# by its own timeout.
#
#   bash submit_scale.sh                      # 1 2 4 8, both apps, weak+strong
#   bash submit_scale.sh "1 2 4" leanmd weak  # leanmd weak only
#   CHAIN=0 bash submit_scale.sh              # let the node counts run at once
#   DRY=1  bash submit_scale.sh               # print the sbatch lines, submit nothing
#
# CHAIN=1 (default) gives every job --dependency=singleton under one job name,
# so exactly one scaling job of yours runs at a time: the sweep is one series
# of measurements on a quiet machine, not four jobs racing for the fabric.
ROOT=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect
COUNTS=${1:-1 2 4 8}; APPS=${2:-leanmd sph2d}; KINDS=${3:-weak strong}; ARMS=${4:-nolb sync async}
MAXTIME_MIN=${MAXTIME_MIN:-120}
[ "${PART:-}" = "gpuA40x4-interactive" ] && MAXTIME_MIN=$(( MAXTIME_MIN > 58 ? 58 : MAXTIME_MIN )); CHAIN=${CHAIN:-1}
printf "%-6s %-24s %-14s %-9s %s\n" NODES PLAN WALLTIME JOBID NOTE
for n in $COUNTS; do
  dry=$(DRY=1 bash $ROOT/scale.sh 0 $n "$APPS" "$KINDS" "$ARMS")
  plan=$(echo "$dry" | grep -oE 'PLAN_SECONDS=[0-9]+' | cut -d= -f2)
  exp=$(echo "$dry"  | grep -oE 'EXPECTED_SECONDS=[0-9]+' | cut -d= -f2)
  [ -z "$plan" ] && { echo "  N=$n: could not plan"; continue; }
  # size from the expectation (+25% and 5 min for launch and the summary), not
  # from the timeout sum; fall back to the old basis if unavailable. The flat
  # term was 10 min and the margin 35% until the expectations were corrected
  # against measurement -- together they asked 31 min for 12 min of runs.
  if [ -n "$exp" ] && [ "$exp" -gt 0 ]; then want=$(( exp*125/100/60 + 5 ))
  else want=$(( plan*115/100/60 + 5 )); fi
  note=""
  if [ $want -gt $MAXTIME_MIN ]; then note="CLAMPED from ${want}min -- arms past the deadline will be SKIPPED; split KINDS across jobs or raise MAXTIME_MIN"; want=$MAXTIME_MIN; fi
  [ -n "$WALLTIME_MIN" ] && want=$WALLTIME_MIN   # explicit override (minutes)
  wt=$(printf "%02d:%02d:00" $((want/60)) $((want%60)))
  # The values carry spaces ("nolb sync async"), and Slurm splits --export on
  # commas -- so export them here and let --export=ALL carry the environment.
  export APPS KINDS ARMS STEPS=${STEPS:-100} PERIOD=${PERIOD:-20} LAG=${LAG:-16}
  [ -n "$LMD_GRID" ] && export LMD_GRID
  [ -n "$POOL_ALLOC" ] && export POOL_ALLOC
  [ -n "$COSTCFG" ] && export COSTCFG
  # PART overrides the script's gpuA40x4; gpuA40x4-interactive starts sooner
  # but caps at 1 h, so a plan longer than that must be split across jobs.
  args=(-N $n --time=$wt --export=ALL ${PART:+--partition=$PART})
  if [ "$CHAIN" = 1 ]; then args+=(--job-name=scale --dependency=singleton); else args+=(--job-name=scale-N$n); fi
  if [ -n "$DRY" ]; then printf "%-6s %-24s %-14s %-9s %s\n" "$n" "${exp:-?}s exp / ${plan}s max" "$wt" "-" "sbatch ${args[*]} scale.sbatch  ${note}"; continue; fi
  jid=$(sbatch "${args[@]}" $ROOT/scale.sbatch | grep -oE '[0-9]+$')
  printf "%-6s %-24s %-14s %-9s %s\n" "$n" "${exp:-?}s exp / ${plan}s max" "$wt" "${jid:-FAILED}" "$note"
done
echo "watch: squeue -u $USER -o '%.10i %.9P %.12j %.2D %.10M %.10l %.20R'"
