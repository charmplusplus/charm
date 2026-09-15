#!/bin/bash
# submit_scale.sh ["1 2 4 8"] [APPS] [KINDS] [ARMS]
#
# Submits ONE job per node count, each sized exactly to that count, so no node
# is ever idle: a 1-node point runs in a 1-node job, not in a corner of an
# 8-node one. Walltime comes from the plan itself (scale.sh DRY=1 sums the
# arms' timeouts), so a job is not held longer than its runs need.
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
  plan=$(DRY=1 bash $ROOT/scale.sh 0 $n "$APPS" "$KINDS" "$ARMS" | grep -oE 'PLAN_SECONDS=[0-9]+' | cut -d= -f2)
  [ -z "$plan" ] && { echo "  N=$n: could not plan"; continue; }
  # the plan is a sum of timeouts (worst case); +5 min for launch, startup and I/O
  want=$(( plan*115/100/60 + 5 )); note=""
  if [ $want -gt $MAXTIME_MIN ]; then note="CLAMPED from ${want}min -- split KINDS across jobs"; want=$MAXTIME_MIN; fi
  wt=$(printf "%02d:%02d:00" $((want/60)) $((want%60)))
  # The values carry spaces ("nolb sync async"), and Slurm splits --export on
  # commas -- so export them here and let --export=ALL carry the environment.
  export APPS KINDS ARMS STEPS=${STEPS:-100} PERIOD=${PERIOD:-20} LAG=${LAG:-16}
  [ -n "$LMD_GRID" ] && export LMD_GRID
  [ -n "$COSTCFG" ] && export COSTCFG
  # PART overrides the script's gpuA40x4; gpuA40x4-interactive starts sooner
  # but caps at 1 h, so a plan longer than that must be split across jobs.
  args=(-N $n --time=$wt --export=ALL ${PART:+--partition=$PART})
  if [ "$CHAIN" = 1 ]; then args+=(--job-name=scale --dependency=singleton); else args+=(--job-name=scale-N$n); fi
  if [ -n "$DRY" ]; then printf "%-6s %-24s %-14s %-9s %s\n" "$n" "${plan}s worst case" "$wt" "-" "sbatch ${args[*]} scale.sbatch  ${note}"; continue; fi
  jid=$(sbatch "${args[@]}" $ROOT/scale.sbatch | grep -oE '[0-9]+$')
  printf "%-6s %-24s %-14s %-9s %s\n" "$n" "${plan}s worst case" "$wt" "${jid:-FAILED}" "$note"
done
echo "watch: squeue -u $USER -o '%.10i %.9P %.12j %.2D %.10M %.10l %.20R'"
