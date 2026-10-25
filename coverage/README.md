# Coverage of Charm++ and reconverse

Line, function and region coverage from clang source-based coverage
(`-fprofile-instr-generate -fcoverage-mapping`), measured over Charm++'s own
tests and examples on the reconverse runtime. It covers both Charm++
(`src/`) and the runtime under it (`contrib/reconverse`).

## Recipe

```bash
# 1. An instrumented build (needs clang; Apple clang on macOS).
./build charm++ reconverse-darwin-arm8 -j6 -g --coverage
#    (Linux: ./build charm++ reconverse-linux-x86_64 clang -j4 -g --coverage;
#     the 'clang' option is what makes charmc link with clang as well)

# 2. Build and run the tests, collecting profiles (about 20-25 min on a laptop).
coverage/run.sh reconverse-darwin-arm8 --tier all-tiers --shapes "1 2"

# 3. Merge, measure and write coverage/out/REPORT.md.
coverage/report.sh reconverse-darwin-arm8 coverage/out
```

`--coverage` adds the flags to `CMAKE_C_FLAGS`/`CMAKE_CXX_FLAGS` and the
linker flags before reconverse is configured, so libreconverse is
instrumented too, and records `CMK_COVERAGE=1` in `conv-mach-opt.sh`. charmc
reads that and instruments every program it compiles and links, so a plain
`make -C tests/charm++/<dir>` against the build is instrumented with no
`OPTS`, including `make CHARMC=<build>/bin/charmc` as CI runs it.

`run.sh` options: `--tier ci|all|ex|all-tiers`, `--shapes "1 2"` (process
counts; 2 runs through lcrun with `TESTRUN_PROCS=2`), `--out <dir>` (default
`coverage/out`), `--list` (print the tiers without running anything).
Timeouts: `COV_BUILD_TIMEOUT` (900 s) and `COV_RUN_TIMEOUT` (300 s) per
directory; `timeout` is used where it exists, `coverage/tmo.pl` otherwise
(macOS), and either kills the whole process group. Before a multi-process
run `run.sh` waits for any other lcrun job on the machine to finish
(two lcrun jobs at once stall each other on a laptop) and clears
`~/.tmp/lct_pmi_file-*`.

`report.sh` writes `REPORT.md`, plus per cumulative tier the merged profile,
the `llvm-cov report` text, `export -summary-only` JSON and an lcov export
under `<out>/report/`; `COV_HTML=1` adds `llvm-cov show -format=html` of the
last tier. It picks `llvm-profdata`/`llvm-cov` from `xcrun`, the compiler's
directory or `PATH`; set `LLVM_PROFDATA`/`LLVM_COV` to override.
`.github/workflows/coverage.yaml` runs all of this weekly on ubuntu-latest
and on demand, and uploads the output.

## Tiers

| tier | what | how it is found |
|---|---|---|
| `ci` | what CI runs | `TEST_DIRS` in `.github/workflows/reconverse-ci.yaml`, read at run time |
| `all` | every other test | directories under `tests/` that `make test` reaches through the aggregator Makefiles (`TESTDIRS`, else `DIRS`) and that have a `test` target |
| `ex` | examples | the same under `examples/` |

Tiers are reported cumulatively: `ci`, `ci+all`, `ci+all+ex`. Aggregators
are not run themselves; their leaves are. An aggregator that also runs a
program of its own (`examples/charm++/PUP`) is run with `DIRS` and
`TESTDIRS` emptied. Each tier's directories that have a `test` target but
that no aggregator reaches are listed in `unreached-<tier>.txt` and in the
report: they are not measured.

To add a test to the measured set, give its directory a `test` target and
list it in its parent's `DIRS`; it is picked up on the next run. To add it to
the CI tier, add it to `TEST_DIRS` in `reconverse-ci.yaml`.

`coverage/exclude.txt` lists directories that are skipped (one per line,
reason after `#`; a directory excludes everything below it). Every entry is
a directory that fails to build or to run on this line today; delete the
line when it is fixed.

## Reading the numbers: traps

- **Static libraries hide unlinked code.** Charm++'s libraries are static,
  and llvm-cov sees only what some program links. A source no test links
  (most trace modules, CkLoop, several load balancers) adds nothing to any
  denominator, so the percentage looks better than the tree is. The report
  lists these files under "Compiled sources absent from every binary".
  Reconverse is measured through `libreconverse`, a shared library, so all of
  it is counted.
- **Tests that run nothing.** A `test` target can pass without starting a
  program: a Makefile that does not include `tests/common.mk` expands
  `$(call run, ...)` to nothing, and `testrun` skips a run whose PE count
  does not divide by the process count. `run.sh` checks that each passing
  run wrote a profile and lists those that did not under "Tests that ran
  nothing".
- **"N functions have mismatched data"** from llvm-cov is expected: every
  test defines its own `main` and handlers under the same names, and header
  inlines exist in several binaries. List the names with
  `llvm-profdata show --all-functions <profdata> | grep ':$' | sort | uniq -d`.
  On 2026-10-25 the only runtime functions among them were
  `CmiMemoryInit` and `CmiMemoryIsomallocDisablePop`, which
  `src/conv-core/memory.C` defines once per `-memory` variant; the rest
  were test and example functions.
- **Denominators move.** Which binaries were built, header instantiations
  and build options all change the totals. Compare percentages loosely, and
  covered-line counts of one file where it matters.
- A test that aborts, segfaults or times out writes no profile; its partial
  run is not counted.
- `bin/charmxi` is built uninstrumented, so `src/xlat-i` is not measured.
- Headers reach the profile under several paths (`src/`, the build's
  `include/` copies, the `include/` symlink). The report attributes each copy
  to its `src/` or `contrib/reconverse/` original and each generated
  `.decl.h`/`.def.h` to the directory of its `.ci` file (marked `[gen]`).

## Last verification run

2026-10-25, macOS (Apple Silicon, Apple clang 16), charm `8d7638f51`,
reconverse `a8efcfc23`, `./build charm++ reconverse-darwin-arm8 -j6 -g
--coverage` (105 s), shapes 1 and 2. Tiers: ci 21 directories (247 s), all
36 (185 s), ex 81 (378 s); `report.sh` 68 s; 150 binaries measured.

Line coverage (functions and regions are in REPORT.md):

| tier | Charm++, all files | Charm++, hand-written | reconverse |
|---|---|---|---|
| ci | 21.8% (17893 / 82065) | 27.1% (12696 / 46906) | 45.5% (3911 / 8595) |
| ci+all | 30.0% (24600 / 82071) | 36.0% (16896 / 46912) | 47.2% (4054 / 8595) |
| ci+all+ex | 36.1% (29628 / 82067) | 45.4% (21300 / 46908) | 47.3% (4067 / 8595) |

The 2026-10-04 measurement with ad hoc scripts (charm `6eeeae95b`) had
36.3% and 46.9% for the last tier. The hand-written denominator grew by
about 5000 lines since then, mostly `src/conv-core` (memory.C, isomalloc.C,
memory-gnu-internal.C: 4129 lines now, 1 then), which the measured programs
now link. 89 of 219 compiled sources are linked by no program.
Runs that passed without running anything: `dynamic_insertion_deletion`
(ci tier) and `delegation/multicast` with 2 processes (every PE count is
odd), `tests/charm++/collide` (no Collide library in a charm++ build) and
`examples/charm++/allGather` (its Makefile does not include `common.mk`).
Nine `unreg` zerocopy examples fail with 2 processes only ("Network
completion failed for peer rank 0", plain build too).
