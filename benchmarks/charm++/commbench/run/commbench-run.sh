#!/bin/bash
# Launch commbench for one site and placement, and write a sidecar file with
# what the program cannot see itself: the launch line, environment, modules,
# job id, host names and the build's CMake options. collect.py merges the
# sidecar into the run record.
#
# Usage: commbench-run.sh SITE PLACEMENT [ARGS...]
#   SITE       mac | anvil | delta | frontier | generic
#   PLACEMENT  P0  1 process, 2 PEs
#              P1  2 hosts x 1 process
#              P2  1 host x 2 processes
#              P2i P2 with +ipc
#              P3  2 hosts x 2 processes
#   ARGS       passed to commbench as given: runtime options (+ipc,
#              ++ipccutoff 65536, +ipcmode shm, ...) and commbench options
#              (--methods get,put --sizes 8:1m:4 ...).
#
# Run inside a Slurm allocation on the clusters. Environment:
#   CHARM_BUILD  build directory (default: where the tree's bin link points)
#   OUT_DIR      results directory (default: ./commbench-results)
#   COMMBENCH    the binary (default: ../commbench next to this script)
#   LAUNCHER     replace the site's launcher (e.g. "srun --mpi=pmix")
#
# Writes OUT_DIR/<site>-<YYYYmmddTHHMMSS>-<charm commit>-<placement>.jsonl and
# the same name with .sidecar.json; prints the launch line first.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
CHARM_DIR=$(cd "$HERE/../../../.." && pwd)
SITE=${1:?usage: commbench-run.sh SITE PLACEMENT [ARGS...]}
PLACEMENT=${2:?usage: commbench-run.sh SITE PLACEMENT [ARGS...]}
shift 2
EXTRA=("$@")

CHARM_BUILD=${CHARM_BUILD:-$(cd "$(cd "$CHARM_DIR/bin" 2>/dev/null && pwd -P)/.." && pwd)}
COMMBENCH=${COMMBENCH:-$HERE/../commbench}
OUT_DIR=${OUT_DIR:-$PWD/commbench-results}
[[ -x $COMMBENCH ]] || { echo "no commbench binary at $COMMBENCH; run make first" >&2; exit 2; }
mkdir -p "$OUT_DIR" || exit 2

# Site settings, as in tests/reconverse-site-run.sh (see its comments for why).
SRUN="srun --mpi=pmi2"
SINGLE_NODE_FLAGS=()
case "$SITE" in
  mac) ;;
  anvil)
    module load python/3.9.5 hwloc
    export LD_LIBRARY_PATH=${RCAC_HWLOC_ROOT:?}/lib:${LD_LIBRARY_PATH:-}
    ;;
  delta)
    export FI_PROVIDER=cxi LCI_NETWORK_BACKENDS=ofi
    SRUN="srun --mpi=pmix"
    ;;
  frontier)
    export FI_PROVIDER=cxi LCI_NETWORK_BACKENDS=ofi
    export PMI_MAX_KVS_ENTRIES=${PMI_MAX_KVS_ENTRIES:-1000}
    SRUN="srun"
    SINGLE_NODE_FLAGS=(--network=single_node_vni)
    ;;
  generic) ;;
  *) echo "unknown SITE=$SITE" >&2; exit 2 ;;
esac
export LD_LIBRARY_PATH=$CHARM_BUILD/lib:${LD_LIBRARY_PATH:-}
read -r -a LAUNCH <<< "${LAUNCHER:-$SRUN}"

IPC=()
[[ $PLACEMENT == P2i ]] && IPC=(+ipc)
if [[ $SITE == mac ]]; then
  # No affinity on macOS. Multi-process runs go through lcrun, one at a time
  # per machine.
  LCRUN=$CHARM_BUILD/_deps/lci-src/lcrun
  case "$PLACEMENT" in
    P0) CMD=("$COMMBENCH" +pe 2) ;;
    P2|P2i) CMD=("$LCRUN" -n 2 "$COMMBENCH" +pe 2 ${IPC[@]+"${IPC[@]}"}) ;;
    *) echo "placement $PLACEMENT needs two hosts; the mac has one" >&2; exit 2 ;;
  esac
else
  case "$PLACEMENT" in
    P0) CMD=("${LAUNCH[@]}" -N1 -n1 -c2 ${SINGLE_NODE_FLAGS[@]+"${SINGLE_NODE_FLAGS[@]}"} "$COMMBENCH" +pe 2 +pemap 0,1) ;;
    P1) CMD=("${LAUNCH[@]}" -N2 -n2 --ntasks-per-node=1 "$COMMBENCH" +pe 2 +pemap 0) ;;
    P2|P2i) CMD=("${LAUNCH[@]}" -N1 -n2 --cpu-bind=none ${SINGLE_NODE_FLAGS[@]+"${SINGLE_NODE_FLAGS[@]}"} "$COMMBENCH" +pe 2 +pemap 0,1 ${IPC[@]+"${IPC[@]}"}) ;;
    P3) CMD=("${LAUNCH[@]}" -N2 -n4 --ntasks-per-node=2 --cpu-bind=none "$COMMBENCH" +pe 4 +pemap 0,1) ;;
    *) echo "unknown PLACEMENT=$PLACEMENT" >&2; exit 2 ;;
  esac
fi

CHARM7=$(git -C "$CHARM_DIR" rev-parse --short=9 HEAD 2>/dev/null || echo unknown)
STAMP=$(date +%Y%m%dT%H%M%S)
PL=$(echo "$PLACEMENT" | tr '[:upper:]' '[:lower:]')
RUN_ID="$SITE-$STAMP-$CHARM7-$PL"
OUT="$OUT_DIR/$RUN_ID.jsonl"
CMD+=(+showcpuaffinity ${EXTRA[@]+"${EXTRA[@]}"} --run-id "$RUN_ID" --machine "$SITE" --out "$OUT")

LAUNCH_LINE=$(printf '%q ' "${CMD[@]}")
echo "$LAUNCH_LINE"
STARTED=$(date +%Y-%m-%dT%H:%M:%S%z)
"${CMD[@]}"
STATUS=$?

MODULES=""
if type module >/dev/null 2>&1; then MODULES=$(module -t list 2>&1); fi
HOSTS=""
if [[ -n ${SLURM_JOB_NODELIST:-} ]] && command -v scontrol >/dev/null; then
  HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
else
  HOSTS=$(hostname)
fi

LAUNCH_LINE="$LAUNCH_LINE" STARTED="$STARTED" STATUS="$STATUS" MODULES="$MODULES" HOSTS="$HOSTS" \
RUN_ID="$RUN_ID" SITE="$SITE" PLACEMENT="$PLACEMENT" CACHE="$CHARM_BUILD/CMakeCache.txt" \
python3 - "$OUT.sidecar.json" <<'EOF'
import json, os, re, sys
env = os.environ
cmake = {}
keys = ["LCI_USE_REG_CACHE", "LCI_NETWORK_BACKENDS", "LCI_PACKET_SIZE_DEFAULT",
        "RECONVERSE_ENABLE_CPU_AFFINITY", "CMK_USE_SHMEM", "CMAKE_BUILD_TYPE"]
try:
    for line in open(env["CACHE"]):
        m = re.match(r"([A-Za-z0-9_]+):[A-Z]+=(.*)$", line.strip())
        if m and m.group(1) in keys:
            cmake[m.group(1)] = m.group(2)
except OSError:
    pass
prefixes = ("FI_", "LCI_", "LCT_", "CmiIpc", "SLURM_", "PMI_")
side = {
    "type": "sidecar",
    "run_id": env["RUN_ID"],
    "site": env["SITE"],
    "placement": env["PLACEMENT"],
    "launch": env["LAUNCH_LINE"].strip(),
    "started": env["STARTED"],
    "exit_status": int(env["STATUS"]),
    "env": {k: v for k, v in sorted(env.items()) if k.startswith(prefixes)},
    "modules": [m for m in env["MODULES"].split("\n") if m and not m.endswith(":")],
    "job_id": env.get("SLURM_JOB_ID"),
    "hosts": [h for h in env["HOSTS"].split("\n") if h],
    "cmake": cmake,
}
with open(sys.argv[1], "w") as f:
    json.dump(side, f, indent=1)
    f.write("\n")
EOF
echo "exit $STATUS; results in $OUT"
exit $STATUS
