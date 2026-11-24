# Migration ledger — how to read and use it

`reconverse-migration-ledger.tsv` is a pure 4-column TSV (GitHub renders
it as a table): every file differing between the reviewed line and
`reconverse-specific-build`, its changed-line count at generation time,
a disposition, and a note. `check-migration-ledger.sh` recomputes the
live delta and fails on any undispositioned file. Migration is complete
when only tombstone/superseded rows remain. Per-file original authors
and commits: `reconverse-provenance.tsv`.

Dispositions: `item-N` (plan item in reconverse-merge-plan.md),
`pr-NNNN`, `with-feature`, `superseded` (the reviewed line's version is
authoritative), `tombstone` / `tombstone-candidate`, `needs-judgment`.

## GPU-series sub-stages (2026-08-28)

- `item-9.1-DONE` — landed via branch `gpu-stage1-hapi` (HAPI portable
  core; hybridapi naming unified across all three build paths).
- `item-9.2-DONE` — landed via branch `gpu-stage2-d2d` (GPU-direct D2D
  transport). Its boundary is spelled `!CMK_RECONVERSE`, in
  `ckrdmadevice.C` and at every site referencing a symbol it defines
  (today just the `loopback_handler` registration in `init.C`). Those
  guards have to move together: drifting apart is what broke the first
  reconverse+GPU link. `grep -rn "loopback_bridge\|loopback_handler"
  src/` enumerates the referencing sites.
- `item-9.3-DONE` — landed via branch `gpu-stage3-examples`
  (examples/benchmarks tier). Both sub-stages were billed as
  whole-file boundaries — take the branch version, no line selection.
  That held for 9.2 and did **not** hold for 9.3: roughly a third of
  the branch's delta here is development scratch (a hardcoded personal
  `CHARM_DIR`, `DataType` switched to `float`, tests and guards
  commented out, per-iteration `printf` debugging, an allocation added
  inside a benchmark's timing loop, a stray Slurm log). Each 9.3 row's
  note records what was taken and what was not.
  Two things 9.3 turned up that outlive it:
  - Nothing built `examples/charm++/cuda/gpudirect` or
    `benchmarks/charm++/cuda`. Neither was in a parent Makefile's
    `DIRS`, so every GPU-direct example was staged into the build tree
    and compiled by nothing — and had in fact stopped compiling after
    9.1. Both are now wired into CI.
  - `CkDeviceBufferPost`'s stream field is `hapi_stream` on the
    reconverse line and `cuda_stream` on the classic one, so no single
    source compiles against both. The examples paper over it with a
    `POST_STREAM` macro, the same shim `tests/charm++/cuda/d2dtest`
    uses. Delete both when the classic line goes.
- Acceptance test for the GPU series: `tests/charm++/cuda/hapitest`,
  one self-checking program (device mapping, buddy allocator, N async
  callbacks). Build-only in CI per #3950; reviewers run it on their
  own GPUs. It is backend-neutral, so an NVIDIA run and an AMD run are
  the same verdict on the same source.
- **item 11 (GPU-LB) is a within-file boundary, not a file list**:
  activate by defining `HAPI_CUPTI_LB` (cmake + charmc) together with
  its ck-ldb half (`LBHasBalancersRegistered`, `setObjGPUTime` /
  `getObjGPUTime` on migratables). The complete boundary is
  enumerable: `grep -rn HAPI_CUPTI_LB src/`
- item 10: the memory-daemon excision markers in `hapi_impl.cpp` say
  "returns with the shrink/expand series" — grep those when staging
  it; port-vs-tombstone decided there (newer work may avoid the OS
  daemon entirely).
