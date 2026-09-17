#!/bin/bash
# Run Charm++'s reconverse test tier on an HPC site, inside a batch
# allocation, the way .github/workflows/reconverse-ci.yaml runs it on hosted
# runners: build each directory, then `make test` once as a single process
# and once as PROCS processes, through the site's launcher.
#
# Usage (from a Slurm allocation, e.g. `salloc -N 2 -n 4 -c 8` or an sbatch
# script), with CHARM_BUILD pointing at the reconverse build directory:
#
#   SITE=anvil CHARM_BUILD=$PWD/reconverse-linux-x86_64 tests/reconverse-site-run.sh
#   SITE=delta NODES=2 CHARM_BUILD=... tests/reconverse-site-run.sh tests/charm++/megatest
#
# Environment:
#   SITE        anvil | delta | frontier | generic  (module loads and launcher flags)
#   CHARM_BUILD the build directory (has bin/charmc, bin/testrun)
#   NODES       nodes for the multi-process pass (default 2; 1 = two
#               processes on one node)
#   PROCS       processes for the multi-process pass (default 2)
#   PES         PEs per single-process run (default: what each Makefile asks)
#   LAUNCHER    override the launcher (default "srun --mpi=pmi2"; delta:
#               "srun --mpi=pmix"; frontier: plain "srun")
#   LAUNCHER_ARGS_SINGLE / LAUNCHER_ARGS_MULTI
#               launcher flags for the two passes (defaults are srun's
#               node/cpu flags; set both empty to use lcrun on a workstation)
# Arguments: test directories; default is the TEST_DIRS list of
# .github/workflows/reconverse-ci.yaml, so the site run and CI stay in step.
#
# Output: one "RESULT <dir> <shape> exit=<code>" line per run and a summary;
# paste them on the pull request being validated. When a run aborts, the
# runtime's "Reason:" line is appended, so a launcher or cpuset problem
# (e.g. "Multiple PEs assigned to same core") is not read as a test failure.
set -u
cd "$(dirname "$0")/.." || exit 1
CHARM_BUILD=${CHARM_BUILD:?set CHARM_BUILD to the reconverse build directory}
SITE=${SITE:-generic}
NODES=${NODES:-2}
PROCS=${PROCS:-2}
# The launcher default is per site: Delta's srun does not bootstrap a
# multi-process reconverse job with --mpi=pmi2 (each rank starts as its own
# one-process job and every multi-process test passes vacuously), so the delta
# case below picks pmix; Frontier fails the same way with pmi2 and uses plain
# srun. Anvil works with pmi2. LAUNCHER in the environment overrides any of
# them.
SITE_LAUNCHER="srun --mpi=pmi2"
LAUNCHER_ARGS_SINGLE=${LAUNCHER_ARGS_SINGLE-"-N1 -c8"}
if [[ $NODES -ge 2 ]]; then
  LAUNCHER_ARGS_MULTI=${LAUNCHER_ARGS_MULTI-"-N$NODES --ntasks-per-node=$((PROCS / NODES)) -c8"}
  MULTI_SHAPE="${PROCS}proc-${NODES}nodes"
else
  LAUNCHER_ARGS_MULTI=${LAUNCHER_ARGS_MULTI-"-N1 -c4"}
  MULTI_SHAPE="${PROCS}proc-1node"
fi

case "$SITE" in
  anvil)
    # Purdue Anvil: both modules are required; hwloc's variable is
    # RCAC_HWLOC_ROOT, and batch jobs need its lib on the path or every
    # binary fails to load libhwloc.so.5. Default network: LCI ibv.
    module load python/3.9.5 hwloc
    export LD_LIBRARY_PATH=${RCAC_HWLOC_ROOT:?}/lib:${LD_LIBRARY_PATH:-}
    ;;
  delta)
    # NCSA Delta: Slingshot-11 through libfabric's cxi provider.
    # Load the site's cmake, gcc and libfabric modules before running this;
    # they are not pinned here because the module names change with the
    # software stack.
    export FI_PROVIDER=cxi
    export LCI_NETWORK_BACKENDS=ofi
    # Verified 2026-09-15 on gpuA100x4-interactive: --mpi=pmi2 gives N isolated
    # one-process jobs ("Starting Reconverse with 1 process" on every rank) and
    # LCT_PMI_BACKEND=pmi2 does not help; --mpi=pmix bootstraps correctly.
    SITE_LAUNCHER="srun --mpi=pmix"
    ;;
  frontier)
    # OLCF Frontier: Slingshot-11 through libfabric's cxi provider, as on
    # delta. Load the modules before running this (PrgEnv-gnu cmake hwloc
    # python; a batch script must `source /opt/cray/pe/lmod/lmod/init/bash`
    # first, because `module` is not defined in a non-login shell); they
    # are not pinned here for the same reason as delta. Cray PMI's key-value
    # store holds 30 entries by default, too few for a multi-process launch
    # (_pmi2_add_kvs: "The KVS data segment ... is not large enough").
    # Every step gets --network=single_node_vni: a step confined to one node
    # has no VNI otherwise (job_vni only covers multi-node steps) and the cxi
    # provider aborts in LCI with "Function not implemented". The multi-node
    # pass also needs it, because a Makefile's +p1 cases run as one process
    # and Slurm shrinks that step to one node; the flag is harmless on a step
    # that does span nodes (verified 2026-09-13, 2 nodes x 1 and 2 x 2 tasks).
    export FI_PROVIDER=cxi
    export LCI_NETWORK_BACKENDS=ofi
    export PMI_MAX_KVS_ENTRIES=${PMI_MAX_KVS_ENTRIES:-1000}
    # Launcher: plain srun, NOT the default --mpi=pmi2. With --mpi=pmi2 each
    # srun task starts as its own one-process reconverse job (every rank
    # prints "Starting Reconverse with 1 process"), so every multi-process
    # test passes vacuously; verified 2026-09-17 by in-job probes in jobs
    # 5495903/5495904. Plain srun uses the cray_shasta plugin (the site
    # default; `srun --mpi=list` offers none, cray_shasta, pmi2) and
    # bootstraps the multi-process job correctly.
    SITE_LAUNCHER="srun"
    LAUNCHER_ARGS_SINGLE="$LAUNCHER_ARGS_SINGLE --network=single_node_vni"
    LAUNCHER_ARGS_MULTI="$LAUNCHER_ARGS_MULTI --network=single_node_vni"
    ;;
  generic) ;;
  *) echo "unknown SITE=$SITE" >&2; exit 2 ;;
esac
export LD_LIBRARY_PATH=$CHARM_BUILD/lib:${LD_LIBRARY_PATH:-}
# GNU timeout is `timeout` on Linux, `gtimeout` from coreutils on macOS.
TIMEOUT=$(command -v timeout || command -v gtimeout || true)
# reason <exit> <out-file>: why a failed run aborted, with a leading space so
# the RESULT line can carry it: the runtime's "Reason:" line (CmiAbort), or the
# C++ exception text when the network layer threw instead.
reason() { [[ $1 -ne 0 ]] && grep -m1 -oE 'Reason:.*|what\(\):.*' "$2" 2>/dev/null | cut -c1-200 | sed 's/^/ /'; return 0; }

if [[ $# -gt 0 ]]; then
  DIRS=("$@")
else
  mapfile -t DIRS < <(awk '/TEST_DIRS: >-/{f=1;next} f&&/^    tests\//{print $1;next} f{exit}' .github/workflows/reconverse-ci.yaml)
fi
[[ ${#DIRS[@]} -gt 0 ]] || { echo "no test directories" >&2; exit 2; }

LAUNCHER=${LAUNCHER:-$SITE_LAUNCHER}
echo "SITE=$SITE NODES=$NODES PROCS=$PROCS LAUNCHER=\"$LAUNCHER\" CHARM_BUILD=$CHARM_BUILD"
echo "JOB=${SLURM_JOB_ID:-none} NODELIST=${SLURM_JOB_NODELIST:-none} DATE=$(date)"
fail=0
for d in "${DIRS[@]}"; do
  if ! make -C "$d" CHARMC="$CHARM_BUILD/bin/charmc" > "$d/site-build.out" 2>&1; then
    echo "RESULT $d build exit=BUILD-FAIL"; fail=1; continue
  fi
  # single process: testrun appends -n 1 to the launcher
  TESTRUN_LAUNCHER="$LAUNCHER $LAUNCHER_ARGS_SINGLE" TESTRUN_PROCS=1 \
    ${TIMEOUT:+$TIMEOUT 600} make -C "$d" test > "$d/site-1proc.out" 2>&1; e1=$?
  echo "RESULT $d 1proc exit=$e1$(reason "$e1" "$d/site-1proc.out")"
  TESTRUN_LAUNCHER="$LAUNCHER $LAUNCHER_ARGS_MULTI" TESTRUN_PROCS=$PROCS \
    ${TIMEOUT:+$TIMEOUT 600} make -C "$d" test > "$d/site-$MULTI_SHAPE.out" 2>&1; e2=$?
  echo "RESULT $d $MULTI_SHAPE exit=$e2$(reason "$e2" "$d/site-$MULTI_SHAPE.out")"
  [[ $e1 -eq 0 && $e2 -eq 0 ]] || fail=1
done
echo "SUMMARY: $([[ $fail -eq 0 ]] && echo ALL PASSED || echo FAILURES ABOVE) $(date)"
exit $fail
