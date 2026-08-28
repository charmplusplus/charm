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
- `item-9.2` / `item-9.3` — whole-file boundaries: take the branch
  versions of these files onto the post-9.1 base; no line selection
  needed. 9.2 = GPU-direct D2D transport (validate on a multi-GPU
  Anvil node, `-A asc050025-gpu`); 9.3 = examples/benchmarks tier.
  The 9.2 boundary is spelled `!CMK_RECONVERSE`, in `ckrdmadevice.C`
  and at every site referencing a symbol it defines (today just the
  `loopback_handler` registration in `init.C`). Those guards have to
  move together: drifting apart is what broke the first reconverse+GPU
  link. `grep -rn "loopback_bridge\|loopback_handler" src/` enumerates
  the referencing sites.
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
