#!/usr/bin/env bash
# Build and run Charm++'s tests against a --coverage build, collecting clang
# profiles per tier. See coverage/README.md.
#
# usage: coverage/run.sh <build-dir> [--tier ci|all|ex|all-tiers]
#                        [--shapes "1 2"] [--out <dir>] [--list]
#
# Tiers:
#   ci   the TEST_DIRS list of .github/workflows/reconverse-ci.yaml
#   all  every directory under tests/ that `make test` reaches through the
#        aggregator Makefiles (TESTDIRS, else DIRS) and has a test target,
#        minus the ci tier
#   ex   the same under examples/
# Directories listed in coverage/exclude.txt are skipped. --list writes and
# prints the directory lists without building or running anything.
#
# Output in <out> (default coverage/out):
#   profraw/<tier>/      raw profiles (LLVM_PROFILE_FILE=%p-%m)
#   results-<tier>.tsv   dir, step, outcome, seconds, note
#   vacuous-<tier>.tsv   passing runs that wrote no profile (ran nothing)
#   dirs-<tier>.txt      the directories run; unreached-<tier>.txt lists
#                        directories with a test target that no aggregator
#                        reaches
#   logs/<tier>/         build and run logs
#   tiers.tsv, meta.txt  wall time per tier; commits and compiler
#
# Environment: COV_BUILD_TIMEOUT (default 900 s), COV_RUN_TIMEOUT (300 s),
# COV_LCRUN_WAIT (1800 s: how long to wait for another lcrun job to finish).

usage() {
  sed -n '5,6p' "$0" | sed 's/^# //' >&2
  exit 2
}
die() { echo "run.sh: $*" >&2; exit 2; }

[[ $# -ge 1 ]] || usage
B=$(cd "$1" 2>/dev/null && pwd -P) || die "no build directory '$1'"
shift
tier_arg=all-tiers
shapes="1 2"
out=""
list_only=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) tier_arg=$2; shift 2 ;;
    --tier=*) tier_arg=${1#*=}; shift ;;
    --shapes) shapes=$2; shift 2 ;;
    --shapes=*) shapes=${1#*=}; shift ;;
    --out) out=$2; shift 2 ;;
    --out=*) out=${1#*=}; shift ;;
    --list) list_only=1; shift ;;
    -h|--help) usage ;;
    *) die "unknown argument '$1'" ;;
  esac
done
case "$tier_arg" in
  ci|all|ex) tiers=$tier_arg ;;
  all-tiers) tiers="ci all ex" ;;
  *) die "--tier must be ci, all, ex or all-tiers" ;;
esac
for p in $shapes; do
  [[ $p =~ ^[1-9][0-9]*$ ]] || die "--shapes takes process counts, e.g. \"1 2\""
done

SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ -f $B/CMakeCache.txt ]] || die "$B is not a Charm++ build directory"
grep -q '^CMK_COVERAGE=1' "$B/include/conv-mach-opt.sh" 2>/dev/null ||
  die "$B was not built with --coverage"
out=${out:-$SRC/coverage/out}
mkdir -p "$out" || die "cannot create $out"
out=$(cd "$out" && pwd -P)

build_tmo=${COV_BUILD_TIMEOUT:-900}
run_tmo=${COV_RUN_TIMEOUT:-300}
lcrun_wait=${COV_LCRUN_WAIT:-1800}

# Runs "$@" with a time limit, killing its whole process group on expiry;
# exit status 124 means it timed out.
if command -v timeout >/dev/null 2>&1; then
  bounded() { local s=$1; shift; timeout -k 10 "$s" "$@"; }
elif command -v gtimeout >/dev/null 2>&1; then
  bounded() { local s=$1; shift; gtimeout -k 10 "$s" "$@"; }
else
  bounded() { local s=$1; shift; perl "$SRC/coverage/tmo.pl" "$s" "$@"; }
fi

# Exclusions: one directory per line, '#' starts a comment; a directory
# excludes everything below it.
excludes=$(sed -e 's/#.*//' -e 's/[[:space:]]*$//' -e '/^$/d' "$SRC/coverage/exclude.txt" 2>/dev/null)
is_excluded() {
  local e
  for e in $excludes; do
    [[ $1 == "$e" || $1 == "$e"/* ]] && return 0
  done
  return 1
}

# The CI tier, read from the workflow so it cannot drift from it.
ci_dirs() {
  awk '/^[ \t]*TEST_DIRS:[ \t]*>-?[ \t]*$/ { on = 1; next }
       on && /^[ \t]+[A-Za-z0-9_.+\/-]+[ \t]*$/ { print $1; next }
       on { exit }' "$SRC/.github/workflows/reconverse-ci.yaml"
}

helper=$out/.subdirs.mk
# shellcheck disable=SC2016 # make syntax, not shell
printf '__cov_subdirs:\n\t@echo $(if $(filter undefined,$(origin DIRS)),,AGG) $(if $(filter undefined,$(origin TESTDIRS)),$(DIRS),$(TESTDIRS))\n' > "$helper"

has_test_target() {
  awk '/^[^#\t][^=]*:/ { split($0, h, ":"); n = split(h[1], t, " ");
         for (i = 1; i <= n; i++) if (t[i] == "test") f = 1 }
       END { exit !f }' "$B/$1/Makefile"
}
# An aggregator whose test target also runs a program of its own.
runs_own_program() {
  local n
  n=$(make -n -C "$B/$1" test DIRS= TESTDIRS= 2>/dev/null | grep -cv -e '^make' -e '^[[:space:]]*$')
  [[ $n -gt 0 ]]
}

# Prints "leaf <dir>", "agg <dir>" (a Makefile that defines DIRS) and
# "mixed <dir>" (an aggregator that also runs a program of its own; it is run
# with DIRS and TESTDIRS emptied) for every directory `make test` in <dir>
# reaches.
walk() {
  local d=$1 subs s
  [[ -f $B/$d/Makefile ]] || return 0
  is_excluded "$d" && return 0
  subs=$(make -s -C "$B/$d" -f Makefile -f "$helper" __cov_subdirs 2>/dev/null)
  if [[ $subs == AGG* ]]; then
    # An aggregator, possibly with an empty list.
    echo "agg $d"
    for s in ${subs#AGG}; do walk "$d/$s"; done
    if has_test_target "$d" && runs_own_program "$d"; then echo "mixed $d"; fi
  elif has_test_target "$d"; then
    echo "leaf $d"
  fi
}

under_ci() {
  local c
  for c in $cilist; do
    [[ $1 == "$c" || $1 == "$c"/* ]] && return 0
  done
  return 1
}

# Writes dirs-<tier>.txt ("<dir> [mixed]") and unreached-<tier>.txt.
list_tier() {
  local t=$1 root kind d walked
  : > "$out/dirs-$t.txt"
  if [[ $t == ci ]]; then
    for d in $cilist; do
      if is_excluded "$d"; then continue; fi
      echo "$d" >> "$out/dirs-$t.txt"
    done
    : > "$out/unreached-$t.txt"
    return
  fi
  if [[ $t == all ]]; then root=tests; else root=examples; fi
  walked=$(walk "$root")
  while read -r kind d; do
    [[ -z $d || $kind == agg ]] && continue
    under_ci "$d" && continue
    if [[ $kind == mixed ]]; then echo "$d mixed"; else echo "$d"; fi
  done <<< "$walked" >> "$out/dirs-$t.txt"
  # Directories with a test target that no aggregator reaches.
  (cd "$B" && find "$root" -name Makefile) | sed 's,/Makefile$,,' | sort |
    while read -r d; do
      has_test_target "$d" || continue
      is_excluded "$d" && continue
      under_ci "$d" && continue
      grep -qxF -e "leaf $d" -e "mixed $d" -e "agg $d" <<< "$walked" && continue
      echo "$d"
    done > "$out/unreached-$t.txt"
}

wait_for_lcrun() {
  local waited=0
  while pgrep -f '(^|/)lcrun( |$)' >/dev/null 2>&1; do
    [[ $waited -eq 0 ]] && echo "  another lcrun job is running; waiting (up to ${lcrun_wait} s)"
    sleep 10
    waited=$((waited + 10))
    if [[ $waited -ge $lcrun_wait ]]; then
      echo "  still running after ${lcrun_wait} s; going ahead"
      return 1
    fi
  done
  return 0
}

# Kills leftovers of a failed run: programs built in the directory.
kill_strays() {
  local x
  find "$B/$1" -maxdepth 1 -type f -perm -u+x -newer "$B/CMakeCache.txt" 2>/dev/null |
    while read -r x; do
      pkill -9 -f "^\./$(basename "$x")( |$)" 2>/dev/null
    done
}

first_error() {
  grep -v '^make' "$1" | grep -m1 -iE "$2" | cut -c1-200 | tr '\t' ' '
}

run_dir() {
  local t=$1 d=$2 mode=$3 tag log s rc e p o note nprof marker
  local -a extra=()
  [[ $mode == mixed ]] && extra=(DIRS= TESTDIRS=)
  tag=$(echo "$d" | tr / _)
  log=$out/logs/$t/$tag
  if [[ ! -f $B/$d/Makefile ]]; then
    printf '%s\tbuild\tFAIL(missing)\t0\tno Makefile in the build tree\n' "$d" >> "$results"
    echo "  $d: missing"
    return
  fi
  bounded 120 make -C "$B/$d" "${extra[@]}" clean > /dev/null 2>&1
  s=$(date +%s)
  bounded "$build_tmo" make -C "$B/$d" "${extra[@]}" > "$log.build.log" 2>&1
  rc=$?
  e=$(( $(date +%s) - s ))
  if [[ $rc -ne 0 ]]; then
    if [[ $rc -eq 124 ]]; then o=TIMEOUT; else o="FAIL(rc=$rc)"; fi
    note=$(first_error "$log.build.log" 'error|fatal')
    printf '%s\tbuild\t%s\t%s\t%s\n' "$d" "$o" "$e" "$note" >> "$results"
    echo "  $d: build $o (${e}s)"
    return
  fi
  printf '%s\tbuild\tok\t%s\t\n' "$d" "$e" >> "$results"
  local line="  $d: build ok (${e}s)"
  for p in $shapes; do
    if [[ $p -gt 1 ]] && wait_for_lcrun; then
      # lcrun's file PMI backend keeps one rendezvous directory per host; a
      # run that died mid-bootstrap leaves it behind and the next launch hangs.
      rm -rf "$HOME/.tmp/lct_pmi_file-"*
    fi
    marker=$out/.marker
    touch "$marker"
    s=$(date +%s)
    TESTRUN_PROCS=$p LLVM_PROFILE_FILE="$out/profraw/$t/%p-%m.profraw" \
      bounded "$run_tmo" make -C "$B/$d" "${extra[@]}" test > "$log.run$p.log" 2>&1
    rc=$?
    e=$(( $(date +%s) - s ))
    nprof=$(find "$out/profraw/$t" -name '*.profraw' -newer "$marker" | wc -l | tr -d ' ')
    note=""
    case $rc in
      0) o=pass ;;
      124) o=TIMEOUT ;;
      *) o="FAIL(rc=$rc)" ;;
    esac
    if [[ $rc -ne 0 ]]; then
      note=$(first_error "$log.run$p.log" 'error|abort|fatal|assert|fail|signal|segment')
      kill_strays "$d"
    elif [[ $nprof -eq 0 ]]; then
      if grep -q 'Skipping the test' "$log.run$p.log"; then
        note="ran nothing: testrun skipped every run (PEs not divisible by $p)"
      elif ! grep -q 'testrun' "$log.run$p.log"; then
        note="ran nothing: the test target started no program"
      else
        note="ran nothing: no profile written"
      fi
      printf '%s\trun%sp\t%s\n' "$d" "$p" "$note" >> "$vacuous"
    elif grep -q 'Skipping the test' "$log.run$p.log"; then
      note="testrun skipped some runs (PEs not divisible by $p)"
    fi
    printf '%s\trun%sp\t%s\t%s\t%s\n' "$d" "$p" "$o" "$e" "$note" >> "$results"
    line="$line, run${p}p $o (${e}s)"
  done
  echo "$line"
}

cilist=$(ci_dirs)
[[ -n $cilist ]] || die "no TEST_DIRS found in .github/workflows/reconverse-ci.yaml"

{
  echo "date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "charm: $(git -C "$SRC" rev-parse --short=10 HEAD 2>/dev/null)"
  echo "reconverse: $(git -C "$SRC/contrib/reconverse" rev-parse --short=10 HEAD 2>/dev/null)"
  echo "compiler: $(grep -m1 '^CMAKE_CXX_COMPILER:' "$B/CMakeCache.txt" | cut -d= -f2) ($("$(grep -m1 '^CMAKE_CXX_COMPILER:' "$B/CMakeCache.txt" | cut -d= -f2)" --version 2>/dev/null | head -1))"
  echo "build: $B"
  echo "shapes: $shapes"
} > "$out/meta.txt"

if [[ $list_only -eq 1 ]]; then
  for t in $tiers; do
    list_tier "$t"
    echo "== tier $t ($(wc -l < "$out/dirs-$t.txt" | tr -d ' ') directories)"
    cat "$out/dirs-$t.txt"
    echo "-- unreached by the aggregator Makefiles:"
    cat "$out/unreached-$t.txt"
  done
  rm -f "$helper"
  exit 0
fi

for t in $tiers; do
  results=$out/results-$t.tsv
  vacuous=$out/vacuous-$t.tsv
  rm -rf "$out/profraw/$t" "$out/logs/$t"
  mkdir -p "$out/profraw/$t" "$out/logs/$t"
  : > "$results"
  : > "$vacuous"
  list_tier "$t"
  n=$(wc -l < "$out/dirs-$t.txt" | tr -d ' ')
  echo "== tier $t: $n directories, shapes $shapes ($(date '+%H:%M:%S'))"
  start=$(date +%s)
  while read -r d mode; do
    [[ -z $d ]] && continue
    run_dir "$t" "$d" "$mode"
  done < "$out/dirs-$t.txt"
  secs=$(( $(date +%s) - start ))
  if [[ -f $out/tiers.tsv ]]; then
    grep -v "^$t"$'\t' "$out/tiers.tsv" > "$out/tiers.tsv.new"
    mv "$out/tiers.tsv.new" "$out/tiers.tsv"
  fi
  printf '%s\t%s\t%s\n' "$t" "$n" "$secs" >> "$out/tiers.tsv"
  echo "== tier $t done in ${secs} s; $(grep -c $'\tpass\t' "$results") passing runs," \
       "$(wc -l < "$vacuous" | tr -d ' ') ran nothing; unreached: $(wc -l < "$out/unreached-$t.txt" | tr -d ' ')"
done
rm -f "$helper" "$out/.marker"
