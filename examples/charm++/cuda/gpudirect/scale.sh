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
#      LBDEBUG (+LBDebug for both apps), SPH_LBDEBUG / LMD_LBDEBUG (per app),
#      LMD_EXTRA (extra +LB flags on leanmd's balanced arms, e.g. a cut ceiling),
#      DEADLINE_MIN (55) stop launching new arms this long after the start,
#      COSTCFG (cost table, both apps), LMD_COSTCFG (leanmd override).
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
# sph2d STRONG: fixed SPH_STRONG_X x 34 patches at 177k particles each (default
#   32x34 = 1088 patches / 512 fluid, ~90M particles, 22 GB/GPU at N=1). The
#   per-patch size is what keeps it GPU-bound at every N; the count sets the
#   run time. SPH_STRONG_X=64 SPH_STRONG_W=4 restores the 16-node sizing (2176
#   patches / 1024 fluid, ~35 GB/GPU at N=1 -- drop -r to 1.5 if it OOMs).
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
# * THERE ARE NO OPT-IN LB KNOBS, by design as of 2026-09-15. The migration
#   kick is a plain FIFO push and the reconverse scheduler serves its self
#   queue before its incoming queue; both were ablated at n=5 on leanmd and
#   both alternatives lost. Do NOT reintroduce an urgent/queue-jumping kick:
#   it lands the kick sooner but the element then migrates ahead of the
#   messages already queued for it, and every one has to be parked and
#   forwarded (async 123.6 s urgent vs 119.1 s FIFO, sync 121.4 s). Fair
#   polling (RECONVERSE_SELF_POLL_LIMIT) never cleared its own spread and was
#   worse on top of FIFO. Nothing below sets an LB environment variable and
#   nothing should.
# * RUN-TO-RUN SPREAD on leanmd with LB is 5-10 s of a ~120 s run (noLB is
#   1.0 s), so a single arm at a single node count cannot separate two
#   configurations that differ by less than ~5%. Repeat an arm before
#   believing a difference; ARMS may be given more than once.
# * sph2d gets the CALIBRATED cost table; migrate_* in it was measured
#   2026-09-14 (lbcost.delta-a40.migrate.conf), not estimated as in the older
#   lbcost.delta-a40.conf, and MetisLB's stay edge prices moves from it.
# ------------------------------------------------------- N=1 reference ----
# Measured 2026-09-15 on gpub047 (job 22106974) with the three LB-step fixes
# (METIS idx_t scaling, CUPTI stop = flag flip, HAPI flag ring 4096) and the
# FIFO kick. A weak N=1 point that misses these badly is a regression, not a
# new machine:
#   leanmd weak  mean42 ms/step   noLB 1417   sync 1182   async 1129
#                total s          noLB 134.6  sync 121.4  async 119.1
#   sph2d  weak  ms/step          noLB 21.89  sync 14.61  async 14.55
# The LB step itself is no longer a spike: the leanmd LB window (41-60) sums
# to 22-24 s against a quiet-window equivalent of 23.7 s.
JID=$1; NODES=$2; APPS=${3:-leanmd sph2d}; KINDS=${4:-weak strong}; ARMS=${5:-nolb sync async}
[ -z "$NODES" ] && { echo "usage: scale.sh <JOBID> <NODES> [APPS] [KINDS] [ARMS]"; exit 1; }
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192
ulimit -c 0
ROOT=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect
LMD=$ROOT/leanmd; SPH=$ROOT/sph2d; RL=$LMD/ranklogs
STEPS=${STEPS:-100}; PERIOD=${PERIOD:-20}; LAG=${LAG:-16}
# The 2-node table: the only one whose inter_node tier was measured rather than
# copied from ipc_cross_gpu. leanmd used to run with NO table at all, which left
# every move unpriced -- at 1 node that cost little (all moves local), at 2 it
# let async migrate off-node for free. Both apps now get it.
COSTCFG=${COSTCFG:-$SPH/lbcost.delta-a40.2node.conf}
LMD_COSTCFG=${LMD_COSTCFG:-$COSTCFG}
# Arena MB, per app and kind: observed peak demand plus headroom, in ONE arena.
# pool=<arenas>(<created after the first timed step>): leanmd counts one
# process (rank_0.log), sph2d all four (one log), so leanmd 1(+0) and sph2d
# 4(+0) both mean "one arena per process, none mid-run". The sph2d pimb column
# is the log's PARTICLE imbalance (max/avg per patch) -- dam-break physics that
# a balancer cannot change, useful only as an arm-to-arm identity check.
#   leanmd weak   >2 GB with LB at 128 cells/GPU         -> 4 GB
#   leanmd strong 4x the cells per GPU                   -> 16 GB
#   sph2d  weak   17.25 GB observed at 177k/patch        -> 20 GB (32768 after
#                 pow2; VERIFIED 2026-09-15: exactly one arena per rank in all
#                 three arms, no mid-run growth)
#   sph2d  strong ~35 GB/GPU by the config's own estimate -> 40 GB, close to the
#                 48 GB card: expect growth and check the pool= column.
# The buddy allocator behind the pool takes a POWER-OF-TWO region: 20480 MB
# (20 GiB) aborts every rank at startup with "Buddy allocator communication
# region must be a power of two". pow2_mb below rounds up, so a size given here
# or in the environment cannot reintroduce that.
#   sph2d strong wants ~35 GB/GPU, which no single power-of-two arena can hold
#   under 48 GB: it takes 8 GiB arenas and grows, and the pool= column will say
#   so. Nothing else grows.
# leanmd weak was 2048 until 2026-09-15 (job 22106974): at that size ONE rank
# per run takes a second arena ~50 s in, at the first Diffusion migration, in
# every LB arm (sync and async) and never in noLB -- a cudaMalloc inside the
# timed window. 128 cells/GPU therefore peaks between 2 and 4 GB with LB, so
# weak is 4096. Strong is 512 cells/GPU at N=1, 4x the weak per-GPU load, so
# 16384 by the same ratio -- EXTRAPOLATED, not measured; the pool= column says
# whether it held. Both are capped to 2048 at N>1 by pool_for (fabric limit
# below), so a multi-node leanmd LB arm still grows one arena and there is no
# size that avoids it.
LMD_POOL_WEAK=${LMD_POOL_WEAK:-4096};  LMD_POOL_STRONG=${LMD_POOL_STRONG:-16384}
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
# Under the vmm backend (POOL_ALLOC=vmm) the count is heaps (one per process)
# and growth is every in-place chunk mapping the pool reported after the mark.
pool_growth() { local log=$1 mark=$2 t m
  t=$(grep -acE "device pool: arena [0-9]+ of|device pool \(vmm\): heap at" "$log" 2>/dev/null || echo 0)
  m=$(awk -v mk="$mark" '$0 ~ mk {s=1} /device pool: arena [0-9]+ of|device pool.*: growing for/{if(s)m++} END{print m+0}' "$log" 2>/dev/null)
  echo "${t}(+${m:-0})"; }
START=$(date +%s); DEADLINE_MIN=${DEADLINE_MIN:-55}
# DRY=1 reports two sums. PLAN_SECONDS is the sum of the arms' TIMEOUTS -- the
# hang guard, deliberately 2-3.7x a healthy run -- and sizing a job from it
# asks for two hours to do 75 minutes of work. EXPECTED_SECONDS is the sum of
# what the arms actually take, from the 2026-09-15 N=1 measurements (leanmd
# 1.2 s/step at 128 cells/GPU including startup; sph2d weak ~110 s), and is
# what submit_scale.sh sizes --time from. A run that overruns its expectation
# is still caught by its own timeout, and DEADLINE_MIN still stops the script
# launching an arm it cannot finish.
# Cores per rank: whatever the job holds (16 = a whole NUMA domain under
# scale.sbatch), so the PEs keep their 8 pinned cores and the runtime's helper
# threads get the rest. Pinning is +pemap's job; --cpu-bind=none stays.
CPT=${CPT:-${SLURM_CPUS_PER_TASK:-16}}
SRUN="srun --jobid=$JID --mpi=cray_shasta -N $NODES -n $((4*NODES)) --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=$CPT --cpu-bind=none --exact --kill-on-bad-exit=1"
SUM=$SPH/../scale_N${NODES}_$(date +%m%d_%H%M).tsv
budget_left() { echo $(( DEADLINE_MIN*60 - ($(date +%s) - START) )); }
# An arm's TIMEOUT is 2-3.7x its expectation -- it exists to catch a hang, not
# to reserve wall. Refusing to start an arm that has less than its full timeout
# left skipped a 120 s sph2d run with 298 s in hand, two seconds under the 300 s
# timeout. Start whenever the EXPECTATION plus a margin fits, and clamp the
# timeout to what is actually left so an overrun still cannot outlive the job.
fits() { local exp=$1; [ $(budget_left) -ge $(( exp*130/100 + 45 )) ]; }
clamp_tmo() { local tmo=$1 left; left=$(( $(budget_left) - 20 )); [ $left -lt $tmo ] && tmo=$left; echo $tmo; }
PLAN_S=0; EXP_S=0   # sums of timeouts / expected runtimes, printed by DRY=1

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
    sync)  extra="$LMD_MD +LBDebug ${LMD_LBDEBUG:-${LBDEBUG:-1}} ${LMD_EXTRA:-}";;
    async) extra="$LMD_MD -lbasync -lblag $LAG +LBAsync +LBDebug ${LMD_LBDEBUG:-${LBDEBUG:-1}} ${LMD_EXTRA:-}";;
  esac
  # ~0.011 s per cell per GPU at the 2744-atom granularity, x2 for noLB+startup
  tmo=$(awk "BEGIN{t=int(2*$STEPS*0.011*$cells/(4*$NODES))+180; print (t<300)?300:t}")
  # Expected, not guarded: 0.0098 s per cell per GPU per step is the measured
  # 1.2 s/step at 128 cells/GPU, plus 25 s of srun launch, startup and teardown
  # -- the weak N=1 arm measured 133 s wall against 125 s of computed steps
  # (22106974, 2026-09-15).
  local exp; exp=$(awk "BEGIN{print int($STEPS*0.0098*$cells/(4*$NODES))+25}")
  local pool; case $kind in weak) pool=$LMD_POOL_WEAK;; strong) pool=$LMD_POOL_STRONG;; esac
  pool=$(pool_for $pool)
  dir=$RL/${kind}_N${NODES}_$arm
  local C="$grid $STEPS $PERIOD $PERIOD -computemap local -density gradient +pe $pes +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool +gpupoolsize $pool"
  [ "${POOL_ALLOC:-buddy}" = vmm ] && C="$C +gpupoolalloc vmm"
  if [ -n "$DRY" ]; then PLAN_S=$((PLAN_S+tmo)); EXP_S=$((EXP_S+exp)); printf "  [dry] leanmd %-6s %-6s grid=[%s] %d cells %d/GPU %d PEs pool=%dMB expect=%ds timeout=%ds\n" "$kind" "$arm" "$grid" $cells $cpg $pes $pool $exp $tmo; return; fi
  fits $exp || { printf "  leanmd %-6s %-6s SKIPPED (%ds left, needs %ds for a %ds run)\n" "$kind" "$arm" "$(budget_left)" "$(( exp*130/100 + 45 ))" "$exp"; return; }
  tmo=$(clamp_tmo $tmo)
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
# The async lag (-l) must be SHORTER than the gap to the next LB trigger, not
# just shorter than the period. -f is the FIRST LB step and -b the period, so
# -f 1000 -b 2000 triggers at 1000, 2000, 4000, 6000 and the shortest gap is
# 1000. sph2d drops any trigger that lands inside an open lag window, so -l 1800
# made the async arm balance TWICE where sync balanced four times -- it hid
# 1.38 s of stall and still lost 0.5 s to the stale placement. Weak: gap 1000,
# lag 800. Strong (-f 500 -b 750): gap 250, lag 200. sph2d.C now aborts on a lag
# that straddles a trigger.
# Per-app LB debug levels. sph2d moves 2-4 objects across nodes in a whole run,
# so its +LBDebug output is nearly all of what the LB step costs it -- and it
# lands inside application execution on the async arm, where sync pays it in a
# barrier. SPH_LBDEBUG/LMD_LBDEBUG set each app independently so a timing arm
# can run quiet while the other still records its decisions.
SPH_MD="+balancer MetisLB +balancer DiffusionLB +LBDiffusionCommOn +LBCostConfig $COSTCFG +LBDebug ${SPH_LBDEBUG:-${LBDEBUG:-1}}"
run_sph2d() { local kind=$1 arm=$2
  local cfg lbargs tmo exp log rc ms imb patches fluid pool arenas=1
  if [ "$kind" = weak ]; then
    local X XC CW; X=$(awk "BEGIN{printf \"%.4f\",1.5*$NODES}"); XC=$((12*NODES)); CW=$(awk "BEGIN{printf \"%.4f\",1.0*$NODES}")
    cfg="-X $X -Y 2.5 -x $XC -y 10 -w $CW -t 2 -s 0.00042 -r 2 -e 0.1 -V 10 -u 200 -i 6000 -S 2000"
    # 93-114 s wall measured at N=1 (22106974); 120 covers the slowest arm.
    patches=$((120*NODES)); fluid=$((64*NODES)); tmo=300; exp=120; pool=$(pool_for $SPH_POOL_WEAK)
    case $arm in nolb) lbargs="-f 99999";; sync) lbargs="-f 1000 -b 2000 $SPH_MD";; async) lbargs="-f 1000 -b 2000 -a -l 800 $SPH_MD +LBAsync";; esac
  else
    # A fixed problem whose per-patch size -- 177k particles at spacing 0.00042
    # on 0.125 x 0.25 m patches -- keeps every node count GPU-bound (the size
    # study: host-bound below 60k/patch, GPU-bound at 586k). Its COUNT is the
    # knob: SPH_STRONG_X patches across (default 32: half the 16-node sizing
    # of 64, so N=1 holds 22 GB/GPU and ~15 min of arms instead of 35 GB and
    # 30) with a fluid column SPH_STRONG_W m wide (default 2 = 16 patches),
    # 8 m tall (32 patches) in a domain 34 patches high. LB every
    # SPH_STRONG_B iterations from SPH_STRONG_F, lag SPH_STRONG_LAG: the
    # defaults give four LB steps in 1500 iterations (the lag must stay
    # shorter than the gap; see the weak note above).
    local SX=${SPH_STRONG_X:-32} SW=${SPH_STRONG_W:-2} SI=${SPH_STRONG_ITERS:-1500}
    local SF=${SPH_STRONG_F:-300} SB=${SPH_STRONG_B:-300} SL=${SPH_STRONG_LAG:-200}
    local SDX; SDX=$(awk "BEGIN{printf \"%.4f\", 0.125*$SX}")
    cfg="-X $SDX -Y 8.5 -x $SX -y 34 -w $SW -t 8 -s 0.00042 -r 2 -e 0.1 -V 10 -u 200 -i $SI -S 500"
    patches=$((SX*34)); fluid=$((SW*8*32))
    # Expected time: a GUESS scaled from the original 600 s guess for 256
    # fluid patches/GPU over 1500 iterations (1.56 ms per fluid patch per
    # iteration per GPU) -- sph2d strong had never been run at any node count.
    exp=$(awk "BEGIN{print int(25 + 0.00156*$fluid/(4*$NODES)*$SI)}"); tmo=$(( exp*3/2 + 120 ))
    pool=$(pool_for $SPH_POOL_STRONG)
    # The balancer plans only within the arenas the pool already has (no
    # growth credit, 2026-09-18), so the LB's headroom must be opened at
    # startup: strong needs ~32 GB/GPU of patches at N=1 (4 x 8 GB arenas, the
    # last one nearly full, ~50 MB slack) and a 5th arena for migrations.
    # Multi-node caps the pool at 4096 MB per process (fabric registration).
    arenas=${SPH_ARENAS_STRONG:-5}
    if [ "$NODES" -gt 1 ]; then arenas=$(( 4096 / pool )); [ "$arenas" -lt 1 ] && arenas=1; fi
    case $arm in nolb) lbargs="-f 99999";; sync) lbargs="-f $SF -b $SB $SPH_MD";; async) lbargs="-f $SF -b $SB -a -l $SL $SPH_MD +LBAsync";; esac
  fi
  local poolargs="+gpupool +gpupoolsize $pool"; [ "${arenas:-1}" -gt 1 ] && poolargs="$poolargs +gpupoolarenas $arenas"
  # POOL_ALLOC=vmm selects the virtual-memory pool backend (+gpupoolalloc vmm:
  # one reserved range per device, 256-byte blocks, chunks mapped in place);
  # default buddy. Same +gpupoolsize x +gpupoolarenas budget either way.
  [ "${POOL_ALLOC:-buddy}" = vmm ] && poolargs="$poolargs +gpupoolalloc vmm"
  log=$SPH/${kind^^}_N${NODES}_$arm.log
  if [ -n "$DRY" ]; then PLAN_S=$((PLAN_S+tmo)); EXP_S=$((EXP_S+exp)); printf "  [dry] sph2d  %-6s %-6s %d patches %d fluid %d PEs pool=%dMBx%d expect=%ds timeout=%ds\n" "$kind" "$arm" $patches $fluid $((16*NODES)) $pool $arenas $exp $tmo; return; fi
  fits $exp || { printf "  sph2d  %-6s %-6s SKIPPED (%ds left, needs %ds for a %ds run)\n" "$kind" "$arm" "$(budget_left)" "$(( exp*130/100 + 45 ))" "$exp"; return; }
  tmo=$(clamp_tmo $tmo)
  ( cd $SPH && env X=1 PES=4 timeout $tmo $SRUN --chdir="$SPH" stdbuf -oL -eL $SPH/numa_wrap_sph.sh ./sph2d $cfg $lbargs +gpushm $poolargs +gpuipceventpool 256 +ppn 4 > $log 2>&1 )
  rc=$?; ms=$(grep -a 'Average iteration' $log | awk '{print $4}')
  imb=$(grep -a '^  step' $log | tail -1 | sed 's/.*imbalance (max\/avg) //;s/ .*//')
  printf "  sph2d  %-6s %-6s %5d patches %4d fluid %4d PEs  %-9s ms/step  patch/PE=%-5s pimb=%-5s pool=%-8s rc=%s\n" \
    "$kind" "$arm" $patches $fluid $((16*NODES)) "${ms:-HUNG/FAIL}" \
    "$(awk "BEGIN{printf \"%.1f\",$patches/(16.0*$NODES)}")" "${imb:-?}" "$(pool_growth $log '^Init:')" $rc
  printf "sph2d\t%s\t%s\t%d\t%dpatch\t%d\t%s\t%s\t%s\n" "$kind" "$arm" $NODES $patches $((patches/(16*NODES))) "${ms:-NA}" "" "$rc" >> $SUM
  grep -aE "Abort|Fatal|Out Of|out of memory|ran more than" $log 2>/dev/null | sed 's/^\[[0-9]*\] //' | sort -u | head -1 | cut -c1-150; }

# ------------------------------------------------------------------ main ----
echo "=== SCALING N=$NODES  apps=[$APPS] kinds=[$KINDS] arms=[$ARMS]  pool=${POOL_ALLOC:-buddy}  deadline ${DEADLINE_MIN}min"
echo "    leanmd $(date -r $LMD/leanmd +%m-%d_%H:%M)  sph2d $(date -r $SPH/sph2d +%m-%d_%H:%M)  cost $(basename $COSTCFG)  ${CPT} cores/rank  $(date +%T)"
[ -z "$DRY" ] && printf "app\tkind\tarm\tnodes\tsize\tper_gpu\tmetric1\tmetric2\trc\n" > $SUM
for kind in $KINDS; do
  echo "--- $kind"
  for app in $APPS; do for arm in $ARMS; do
    case $app in leanmd) run_leanmd $kind $arm;; sph2d|sph) run_sph2d $kind $arm;; esac
  done; done
done
[ -n "$DRY" ] && echo "PLAN_SECONDS=$PLAN_S EXPECTED_SECONDS=$EXP_S"
[ -z "$DRY" ] && echo "=== summary $SUM"
echo "=== DONE $(date +%T), $(( ($(date +%s)-START)/60 )) min used"
