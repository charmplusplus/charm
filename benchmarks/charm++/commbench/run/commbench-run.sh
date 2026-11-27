#!/bin/bash
# Launch commbench for one site and placement, and write a sidecar file with
# what the program cannot see itself: the launch line, environment, modules,
# job id, host names and the build's CMake options. collect.py merges the
# sidecar into the run record.
#
# Usage: commbench-run.sh SITE PLACEMENT [ARGS...]
#   SITE       an entry of sites.json next to this script (mac, anvil, delta,
#              frontier, generic), which gives the launcher, environment and
#              modules
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
#   SITES_FILE   the per-site settings (default: sites.json next to this script)
#   CHARM_BUILD  build directory (default: where the tree's bin link points)
#   OUT_DIR      results directory (default: ./commbench-results)
#   COMMBENCH    the binary (default: ../commbench next to this script)
#   LAUNCHER     replace the site's launcher (e.g. "srun --mpi=pmix")
#   PEMAP        the +pemap list (default 0 for P1, 0,1 for the others)
#   CPU_BIND     the srun --cpu-bind value for every placement (default:
#                none for P2, P2i and P3; Slurm's own binding for P0 and P1)
#   P0_CPUS      CPUs of the P0 step (default 2). srun -c implies --exact,
#                so the step's cpuset holds only those CPUs and a +pemap
#                outside them aborts with "CmiSetCPUAffinity failed"; with
#                PEMAP on a whole node, pass the node's core count.
#   TIMEOUT      bound the run with GNU timeout (e.g. 45m); the sidecar is
#                still written, with exit_status 124 when the limit hit
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

# Site settings from SITES_FILE (see the "_about" entry there). python3 turns
# the site's entry into shell commands; the module loads and exports must run
# in this shell.
SITES_FILE=${SITES_FILE:-$HERE/sites.json}
SITE_SH=$(python3 - "$SITES_FILE" "$SITE" <<'EOF'
import json, shlex, sys
path, site = sys.argv[1], sys.argv[2]
try:
    sites = json.load(open(path))
except (OSError, ValueError) as e:
    sys.exit("cannot read %s: %s" % (path, e))
if site.startswith("_") or site not in sites:
    names = ", ".join(k for k in sites if not k.startswith("_"))
    sys.exit("unknown SITE=%s; %s has: %s" % (site, path, names))
s = sites[site]
q = shlex.quote
out = ["SITE_LAUNCHER=" + q(s.get("launcher", "srun --mpi=pmi2"))]
for m in s.get("modules", []):
    out.append("module load " + q(m))
for c in s.get("modules_if_linked", []):
    body = ["module load " + q(m) for m in c.get("modules", [])]
    if c.get("lib_dir_var"):
        v = c["lib_dir_var"]
        body.append('export LD_LIBRARY_PATH="${%s:?}/lib:${LD_LIBRARY_PATH:-}"' % v)
    out.append('if ldd "$CHARM_BUILD/lib/libreconverse.so" 2>/dev/null | grep -q %s; then %s; fi'
               % (q(c["pattern"]), "; ".join(body) or ":"))
for k, v in s.get("env", {}).items():
    out.append("export %s=%s" % (k, q(str(v))))
for k, v in s.get("env_default", {}).items():
    out.append('if [ -z "${%s:-}" ]; then export %s=%s; else export %s; fi' % (k, k, q(str(v)), k))
out.append("SINGLE_NODE_FLAGS=(" + " ".join(q(f) for f in s.get("single_node_flags", [])) + ")")
print("\n".join(out))
EOF
) || exit 2
eval "$SITE_SH"
export LD_LIBRARY_PATH=$CHARM_BUILD/lib:${LD_LIBRARY_PATH:-}
read -r -a LAUNCH <<< "${LAUNCHER:-$SITE_LAUNCHER}"

IPC=()
[[ $PLACEMENT == P2i ]] && IPC=(+ipc)
if [[ $SITE_LAUNCHER == lcrun ]]; then
  # One host, no Slurm (the Mac: no affinity either). Multi-process runs go
  # through the build's lcrun, one at a time per machine.
  LCRUN=$CHARM_BUILD/_deps/lci-src/lcrun
  case "$PLACEMENT" in
    P0) CMD=("$COMMBENCH" +pe 2) ;;
    P2|P2i) CMD=("$LCRUN" -n 2 "$COMMBENCH" +pe 2 ${IPC[@]+"${IPC[@]}"}) ;;
    *) echo "placement $PLACEMENT needs two hosts; SITE=$SITE has one" >&2; exit 2 ;;
  esac
else
  case "$PLACEMENT" in
    P1) PM=${PEMAP:-0} ;;
    *)  PM=${PEMAP:-0,1} ;;
  esac
  BIND=()
  [[ -n ${CPU_BIND:-} ]] && BIND=(--cpu-bind="$CPU_BIND")
  case "$PLACEMENT" in
    P2|P2i|P3) [[ -z ${CPU_BIND:-} ]] && BIND=(--cpu-bind=none) ;;
  esac
  case "$PLACEMENT" in
    P0) CMD=("${LAUNCH[@]}" -N1 -n1 -c"${P0_CPUS:-2}" ${BIND[@]+"${BIND[@]}"} ${SINGLE_NODE_FLAGS[@]+"${SINGLE_NODE_FLAGS[@]}"} "$COMMBENCH" +pe 2 +pemap "$PM") ;;
    P1) CMD=("${LAUNCH[@]}" -N2 -n2 --ntasks-per-node=1 ${BIND[@]+"${BIND[@]}"} "$COMMBENCH" +pe 2 +pemap "$PM") ;;
    P2|P2i) CMD=("${LAUNCH[@]}" -N1 -n2 ${BIND[@]+"${BIND[@]}"} ${SINGLE_NODE_FLAGS[@]+"${SINGLE_NODE_FLAGS[@]}"} "$COMMBENCH" +pe 2 +pemap "$PM" ${IPC[@]+"${IPC[@]}"}) ;;
    P3) CMD=("${LAUNCH[@]}" -N2 -n4 --ntasks-per-node=2 ${BIND[@]+"${BIND[@]}"} "$COMMBENCH" +pe 4 +pemap "$PM") ;;
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
if [[ -n ${TIMEOUT:-} ]]; then
  timeout -k 30 "$TIMEOUT" "${CMD[@]}"
else
  "${CMD[@]}"
fi
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
    "timeout": env.get("TIMEOUT") or None,
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
