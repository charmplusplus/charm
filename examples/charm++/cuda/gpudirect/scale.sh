#!/bin/bash
# scale.sh <JOBID> <NODES> [APPS] [KINDS] [ARMS]   -- weak and strong scaling for
# BOTH sph2d and leanmd in ONE allocation. Quote multi-word arguments:
#   bash scale.sh $JID 1                                  # everything, ~50 min at N=1
#   bash scale.sh $JID 1 leanmd weak                      # one app, one kind
#   bash scale.sh $JID 2 "leanmd sph2d" strong "nolb sync"
#   DRY=1 bash scale.sh $JID 4                            # print the plan, run nothing
# Under sbatch, scale.sbatch calls it with the job's own node count, so every
# srun spans the whole allocation. It REFUSES to run on an allocation bigger
# than NODES (that would idle the rest); FORCE=1 overrides.
# Env: STEPS/PERIOD/LAG (leanmd 100/20/16), LMD_GRID (strong grid, "32 8 8"),
#      DEADLINE_MIN (55) stop launching new arms this long after the start,
#      COSTCFG (sph2d cost table), LMD_COSTCFG (empty: leanmd runs uncalibrated).
#
# ---------------------------------------------------------------- sizing ----
# leanmd WEAK: 128 cells/GPU held constant, grown in x ONLY (processes are
#   x-slabs, so the process interface is a y-z plane; growing z doubled the
#   cross-group cut 2.3e8 -> 4.8e8 bytes). Size sweep 2026-09-14 (job 22072896):
#   the LB win is 23% and flat at 64 and 128 cells/GPU, 9% at 192, gone at 256.
#     N=1 8 8 8 (512)  N=2 16 8 8 (1024)  N=4 32 8 8 (2048)  N=8 64 8 8 (4096)
#   The density gradient always spans the full x range, so the across-device
#   imbalance (~3.4:1 in atoms) is scale-invariant. Ideal = flat ms/step.
# leanmd STRONG: fixed 32 8 8 (2048 cells), 512 -> 64 cells/GPU over N=1..8, so
#   the balancer's win appears as the problem thins out: its cost scales with
#   the problem, its gain does not.
# sph2d WEAK: per-node unit = the 4-GPU 177k problem (12x10 = 120 patches of
#   0.125x0.25 m, 8x8 = 64 fluid), replicated along the flow direction x:
#   dom_lx 1.5*N, n_chares_x 12*N, col_w 1.0*N; y, spacing and col_h fixed, so
#   patch size, fluid fraction and 7.5 patches/PE are all held.
# sph2d STRONG: fixed 64x34 = 2176 patches / 1024 fluid (~181M particles), sized
#   for the 16-node end; N=1 is the heaviest point (~35 GB/GPU) -- run it as the
#   canary and drop -r to 1.5 if it OOMs.
#
# -------------------------------------------------------------- gotchas ----
# * leanmd's +pe is the TOTAL PE count (8 per process), sph2d's +ppn is PER
#   process (4, with PES widening the pemap to match). Both scale below; a
#   copied "+pe 32" would silently run every node past the first with 8 PEs.
# * MetisLB's cut gate (5a90850ac, default: no rise allowed) REFUSES every
#   mapping on leanmd -- computes are tied to immovable cells and the initial
#   placement is already local -- so the "sync"/"async" arms there are
#   Metis-attempts-then-DiffusionLB, which is what the 23% was measured with.
#   sph2d's initial layout is the bad one, so its Metis step applies. To let
#   Metis through on leanmd for a comparison: +LBMetisMaxCutRise 2.
# * The device pool is PRE-SIZED with +gpupoolsize (MB per arena, overriding
#   CK_GPU_ARENA_MB's 256). At the default the pool grows INSIDE the measured
#   window -- leanmd took a 7th 256 MB arena after step 40/43, right at the
#   first Diffusion migration, and sph2d 177k grew to 69 arenas (17 GB) still
#   growing at the last step. Every growth is a cudaMalloc in the timed region.
#   Sizes below cover the observed demand with headroom, so the arena is taken
#   once at startup; each run reports pool=<total>(+<created after step 1>) and
#   a mid-run arena means the number is suspect and the size wants raising.
# * sph2d gets the CALIBRATED cost table; migrate_* in it was measured
#   2026-09-14 (lbcost.delta-a40.migrate.conf), not estimated as in the older
#   lbcost.delta-a40.conf, and MetisLB's stay edge prices moves from it.
JID=$1; NODES=$2; APPS=${3:-leanmd sph2d}; KINDS=${4:-weak strong}; ARMS=${5:-nolb sync async}
[ -z "$NODES" ] && { echo "usage: scale.sh <JOBID> <NODES> [APPS] [KINDS] [ARMS]"; exit 1; }
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192
ulimit -c 0
ROOT=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect
LMD=$ROOT/leanmd; SPH=$ROOT/sph2d; RL=$LMD/ranklogs
STEPS=${STEPS:-100}; PERIOD=${PERIOD:-20}; LAG=${LAG:-16}
COSTCFG=${COSTCFG:-$SPH/lbcost.delta-a40.migrate.conf}
# Arena MB, per app and kind: observed peak demand plus headroom, in ONE arena.
# pool=<arenas>(<created after the first timed step>): leanmd counts one
# process (rank_0.log), sph2d all four (one log), so leanmd 1(+0) and sph2d
# 4(+0) both mean "one arena per process, none mid-run". The sph2d pimb column
# is the log's PARTICLE imbalance (max/avg per patch) -- dam-break physics that
# a balancer cannot change, useful only as an arm-to-arm identity check.
#   leanmd weak   1.75 GB observed at 128 cells/GPU      -> 4 GB
#   leanmd strong 4x the cells per GPU                   -> 8 GB
#   sph2d  weak   17.25 GB observed at 177k/patch        -> 20 GB
#   sph2d  strong ~35 GB/GPU by the config's own estimate -> 40 GB, close to the
#                 48 GB card: expect growth and check the pool= column.
# The buddy allocator behind the pool takes a POWER-OF-TWO region: 20480 MB
# (20 GiB) aborts every rank at startup with "Buddy allocator communication
# region must be a power of two". pow2_mb below rounds up, so a size given here
# or in the environment cannot reintroduce that.
#   sph2d strong wants ~35 GB/GPU, which no single power-of-two arena can hold
#   under 48 GB: it takes 8 GiB arenas and grows, and the pool= column will say
#   so. Nothing else grows.
LMD_POOL_WEAK=${LMD_POOL_WEAK:-2048};  LMD_POOL_STRONG=${LMD_POOL_STRONG:-8192}
SPH_POOL_WEAK=${SPH_POOL_WEAK:-32768}; SPH_POOL_STRONG=${SPH_POOL_STRONG:-8192}
pow2_mb() { awk -v v="$1" 'BEGIN{p=1; while(p<v) p*=2; print p}'; }
# THE FABRIC REFUSES A 4 GB DEVICE MR. Measured 2026-09-14 at 2 nodes (jobs
# 22079951, 22080298), leanmd: 256 / 1024 / 2048 MB all run; 4096 MB aborts
# every rank in poll_comp_impl:65 "Err 5: Input/output error" immediately after
# "device pool: arena (4096 MB) registered once for RDMA" -- the registration
# that only happens when a peer is off-node, which is why 1 node took 4096 and
# 32768 without complaint. So the arena is capped once there is a second node;
# at 1 node the size stands as asked. 2048 is also the smallest that keeps
# leanmd weak to a SINGLE arena through its migrations (1024 -> 2 arenas).
POOL_MULTINODE_MAX=${POOL_MULTINODE_MAX:-2048}
pool_for() { local mb; mb=$(pow2_mb "$1")
  if [ "$NODES" -gt 1 ] && [ "$mb" -gt "$POOL_MULTINODE_MAX" ]; then mb=$POOL_MULTINODE_MAX; fi
  echo "$mb"; }
# arenas created in total, and those created after the first timed step
pool_growth() { local log=$1 mark=$2 t m
  t=$(grep -acE "device pool: arena [0-9]+ of" "$log" 2>/dev/null || echo 0)
  m=$(awk -v mk="$mark" '$0 ~ mk {s=1} /device pool: arena [0-9]+ of/{if(s)m++} END{print m+0}' "$log" 2>/dev/null)
  echo "${t}(+${m:-0})"; }
START=$(date +%s); DEADLINE_MIN=${DEADLINE_MIN:-55}
# Cores per rank: whatever the job holds (16 = a whole NUMA domain under
# scale.sbatch), so the PEs keep their 8 pinned cores and the runtime's helper
# threads get the rest. Pinning is +pemap's job; --cpu-bind=none stays.
CPT=${CPT:-${SLURM_CPUS_PER_TASK:-16}}
SRUN="srun --jobid=$JID --mpi=cray_shasta -N $NODES -n $((4*NODES)) --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=$CPT --cpu-bind=none --exact --kill-on-bad-exit=1"
SUM=$SPH/../scale_N${NODES}_$(date +%m%d_%H%M).tsv
budget_left() { echo $(( DEADLINE_MIN*60 - ($(date +%s) - START) )); }
PLAN_S=0   # sum of the arms' timeouts, printed by DRY=1 so the submitter can size --time

# Never underutilize: the allocation must be exactly the node count being run.
ALLOC=${SLURM_JOB_NUM_NODES:-$(scontrol show job "$JID" -o 2>/dev/null | grep -oE ' NumNodes=[0-9]+' | head -1 | cut -d= -f2)}
if [ -z "$DRY" ] && [ -n "$ALLOC" ] && [ "$ALLOC" != "$NODES" ]; then
  echo "REFUSING: job $JID has $ALLOC node(s), this run uses $NODES -- $((ALLOC-NODES)) would sit idle."
  echo "  submit one job per node count (submit_scale.sh), or FORCE=1 to run anyway."
  [ -z "$FORCE" ] && exit 2
fi

# ---------------------------------------------------------------- leanmd ----
LMD_DIFF="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim"
LMD_MD="+balancer MetisLB $LMD_DIFF ${LMD_COSTCFG:+ +LBCostConfig $LMD_COSTCFG}"
lmd_mean() { awk -v lo=$2 -v hi=$STEPS '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
run_leanmd() { local kind=$1 arm=$2
  local grid pes cells cpg tmo dir rc m21 m42 extra=""
  case $kind in weak) grid="$((8*NODES)) 8 8";; strong) grid="${LMD_GRID:-32 8 8}";; esac
  pes=$((32*NODES)); cells=$(echo $grid | awk '{print $1*$2*$3}'); cpg=$((cells/(4*NODES)))
  case $arm in
    sync)  extra="$LMD_MD +LBDebug 1";;
    async) extra="$LMD_MD -lbasync -lblag $LAG +LBAsync +LBDebug 1";;
  esac
  # ~0.011 s per cell per GPU at the 2744-atom granularity, x2 for noLB+startup
  tmo=$(awk "BEGIN{t=int(2*$STEPS*0.011*$cells/(4*$NODES))+180; print (t<300)?300:t}")
  local pool; case $kind in weak) pool=$LMD_POOL_WEAK;; strong) pool=$LMD_POOL_STRONG;; esac
  pool=$(pool_for $pool)
  dir=$RL/${kind}_N${NODES}_$arm
  local C="$grid $STEPS $PERIOD $PERIOD -computemap local -density gradient +pe $pes +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool +gpupoolsize $pool"
  if [ -n "$DRY" ]; then PLAN_S=$((PLAN_S+tmo)); printf "  [dry] leanmd %-6s %-6s grid=[%s] %d cells %d/GPU %d PEs pool=%dMB timeout=%ds\n" "$kind" "$arm" "$grid" $cells $cpg $pes $pool $tmo; return; fi
  [ $(budget_left) -lt $tmo ] && { printf "  leanmd %-6s %-6s SKIPPED (%ds left, needs %ds)\n" "$kind" "$arm" "$(budget_left)" "$tmo"; return; }
  rm -rf $dir; mkdir -p $dir
  ( cd $LMD && RANKLOG_DIR=$dir env RANKLOG_DIR=$dir timeout $tmo $SRUN stdbuf -oL -eL $RL/rankwrap.sh $LMD/leanmd $C $extra >$dir/srun.err 2>&1 )
  rc=$?; m21=$(lmd_mean $dir/rank_0.log 21); m42=$(lmd_mean $dir/rank_0.log 42)
  printf "  leanmd %-6s %-6s [%s] %4d cells/GPU %4d PEs  mean21=%-8s mean42=%-8s ms/cell=%-6s pool=%-8s rc=%s migr=[%s] %s\n" \
    "$kind" "$arm" "$grid" $cpg $pes "$m21" "$m42" \
    "$(awk -v m="$m42" -v c=$cpg 'BEGIN{if(m+0>0)printf "%.3f",m/c; else print "NA"}')" \
    "$(pool_growth $dir/rank_0.log "Step [0-9]+ Benchmark")" "$rc" \
    "$(grep -ah 'cross node migrations' $dir/rank_0.log 2>/dev/null | awk '{printf "%s ",$NF}')" \
    "$(cat $dir/rank_*.log 2>/dev/null | grep -aoE 'no free CUDA IPC event slot|out of memory|Fatal CUDA|Aborting' | sort | uniq -c | head -1 | tr '\n' ';')"
  printf "leanmd\t%s\t%s\t%d\t%s\t%d\t%s\t%s\t%s\n" "$kind" "$arm" $NODES "$grid" $cpg "$m21" "$m42" "$rc" >> $SUM
  [ $rc -ne 0 ] && head -3 $dir/srun.err 2>/dev/null | sed 's/^/      srun: /' | cut -c1-170
  grep -ah "measured loads explain" $dir/rank_0.log 2>/dev/null | head -2 | sed 's/^/      /' | cut -c1-230
  grep -ah "MetisLB gate" $dir/rank_0.log 2>/dev/null | head -1 | sed 's/^/      /' | cut -c1-250; }

# ----------------------------------------------------------------- sph2d ----
SPH_MD="+balancer MetisLB +balancer DiffusionLB +LBDiffusionCommOn +LBCostConfig $COSTCFG"
run_sph2d() { local kind=$1 arm=$2
  local cfg lbargs tmo log rc ms imb patches fluid pool
  if [ "$kind" = weak ]; then
    local X XC CW; X=$(awk "BEGIN{printf \"%.4f\",1.5*$NODES}"); XC=$((12*NODES)); CW=$(awk "BEGIN{printf \"%.4f\",1.0*$NODES}")
    cfg="-X $X -Y 2.5 -x $XC -y 10 -w $CW -t 2 -s 0.00042 -r 2 -e 0.1 -V 10 -u 200 -i 6000 -S 2000"
    patches=$((120*NODES)); fluid=$((64*NODES)); tmo=300; pool=$(pool_for $SPH_POOL_WEAK)
    case $arm in nolb) lbargs="-f 99999";; sync) lbargs="-f 1000 -b 2000 $SPH_MD";; async) lbargs="-f 1000 -b 2000 -a -l 1800 $SPH_MD +LBAsync";; esac
  else
    cfg="-X 8 -Y 8.5 -x 64 -y 34 -w 4 -t 8 -s 0.00042 -r 2 -e 0.1 -V 10 -u 200 -i 1500 -S 500"
    patches=2176; fluid=1024; tmo=900; pool=$(pool_for $SPH_POOL_STRONG)
    case $arm in nolb) lbargs="-f 99999";; sync) lbargs="-f 500 -b 750 $SPH_MD";; async) lbargs="-f 500 -b 750 -a -l 700 $SPH_MD +LBAsync";; esac
  fi
  log=$SPH/${kind^^}_N${NODES}_$arm.log
  if [ -n "$DRY" ]; then PLAN_S=$((PLAN_S+tmo)); printf "  [dry] sph2d  %-6s %-6s %d patches %d fluid %d PEs pool=%dMB timeout=%ds\n" "$kind" "$arm" $patches $fluid $((16*NODES)) $pool $tmo; return; fi
  [ $(budget_left) -lt $tmo ] && { printf "  sph2d  %-6s %-6s SKIPPED (%ds left, needs %ds)\n" "$kind" "$arm" "$(budget_left)" "$tmo"; return; }
  ( cd $SPH && env X=1 PES=4 timeout $tmo $SRUN --chdir="$SPH" stdbuf -oL -eL $SPH/numa_wrap_sph.sh ./sph2d $cfg $lbargs +gpushm +gpupool +gpupoolsize $pool +gpuipceventpool 256 +ppn 4 > $log 2>&1 )
  rc=$?; ms=$(grep -a 'Average iteration' $log | awk '{print $4}')
  imb=$(grep -a '^  step' $log | tail -1 | sed 's/.*imbalance (max\/avg) //;s/ .*//')
  printf "  sph2d  %-6s %-6s %5d patches %4d fluid %4d PEs  %-9s ms/step  patch/PE=%-5s pimb=%-5s pool=%-8s rc=%s\n" \
    "$kind" "$arm" $patches $fluid $((16*NODES)) "${ms:-HUNG/FAIL}" \
    "$(awk "BEGIN{printf \"%.1f\",$patches/(16.0*$NODES)}")" "${imb:-?}" "$(pool_growth $log '^  step')" $rc
  printf "sph2d\t%s\t%s\t%d\t%dpatch\t%d\t%s\t%s\t%s\n" "$kind" "$arm" $NODES $patches $((patches/(16*NODES))) "${ms:-NA}" "" "$rc" >> $SUM
  grep -aE "Abort|Fatal|Out Of|out of memory|ran more than" $log 2>/dev/null | sed 's/^\[[0-9]*\] //' | sort -u | head -1 | cut -c1-150; }

# ------------------------------------------------------------------ main ----
echo "=== SCALING N=$NODES  apps=[$APPS] kinds=[$KINDS] arms=[$ARMS]  deadline ${DEADLINE_MIN}min"
echo "    leanmd $(date -r $LMD/leanmd +%m-%d_%H:%M)  sph2d $(date -r $SPH/sph2d +%m-%d_%H:%M)  cost $(basename $COSTCFG)  ${CPT} cores/rank  $(date +%T)"
[ -z "$DRY" ] && printf "app\tkind\tarm\tnodes\tsize\tper_gpu\tmetric1\tmetric2\trc\n" > $SUM
for kind in $KINDS; do
  echo "--- $kind"
  for app in $APPS; do for arm in $ARMS; do
    case $app in leanmd) run_leanmd $kind $arm;; sph2d|sph) run_sph2d $kind $arm;; esac
  done; done
done
[ -n "$DRY" ] && echo "PLAN_SECONDS=$PLAN_S"
[ -z "$DRY" ] && echo "=== summary $SUM"
echo "=== DONE $(date +%T), $(( ($(date +%s)-START)/60 )) min used"
