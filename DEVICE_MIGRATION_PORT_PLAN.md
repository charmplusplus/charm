# Porting `DeviceMigrationStrategy` onto `cupti_lb_reconverse`

Replace reconverse's stage-then-send GPU migration path with the
export-and-pull path from `cupti_lb_new`, while keeping reconverse's
buddy-allocated LB buffer.

## Why

**Reconverse today (stage-then-send).** Source allocates an LB buffer
(`alloc_comm_buffer`, `cklocation.C:3257`), copies the chare's device data into
it during `pupElementsFor`, runs a full `cudaDeviceSynchronize()`, then ships
the staging buffer. The destination allocates its own LB buffer
(`cklocation.C:3341`) to receive.

Per migration: **2 device allocations, 2 device copies, 1 device-wide sync.**

**`DeviceMigrationStrategy` (export-and-pull).** The source only *describes*
its buffers — raw pointer (MEMCPY), `hapiIpcGetMemHandle` (IPC), or an RDMA tag
— and the destination pulls straight from the source's live buffers into one
destination buffer. Completion is reported per-stream via `hapiAddCallback`.

Per migration: **1 device allocation, 1 device copy, no device-wide sync.**

Roughly half the device bandwidth and half the peak device memory, and it
removes a full-device sync from the migration path — which matters when many
chares migrate at once, since the current version serializes on repeated syncs.

The memory point is not just efficiency: reconverse's staging comes from a
fixed-size LB buffer that **hard-aborts when exhausted** ("Not enough memory on
device Load balance buffer") at *both* ends (`cklocation.C:3259`, `:3343`).
Removing source-side staging deletes one of those two failure points.

## Key design decision: do NOT swap wholesale

`cupti_lb_new` allocates the destination buffer with raw `cudaMalloc` /
`cudaFree` per migration (`cklocation.C:3565`, `:3710`). Reconverse
deliberately moved away from that — see its commits "malloc for load balancing
buffers", "remove one extra level of cudaMalloc/cudaFree calls during load
balancing", "lb buffer size not restricted to power of 2". Per-migration
`cudaMalloc`/`cudaFree` are expensive and implicitly synchronizing.

**Port the protocol, keep reconverse's allocator.** Substitute
`dm->alloc_comm_buffer(...)` / `free_comm_buffer` for `cudaMalloc` / `cudaFree`
in the pull path. This is a hybrid, not a replacement.

## The protocol

1. **Source** pups host data only (no GPU), collects `(ptr, size)` pairs for its
   device buffers, calls `strategy.fillSourceHandles(toPe, srcPtrs, handles)`,
   sends `CkArrayElementMigrateHandleMessage` (handles + `src_pe`), and **holds**
   the live chares in `heldChares[id]` instead of destroying them.
2. **Destination** (`immigrateGPUHandle`) allocates one contiguous buffer of the
   summed sizes, calls `strategy.issuePulls(...)`, records `pendingPulls[id]`.
3. **On pull completion** `finalizeGPUMigrate(CkLocMgrFinalizeMsg*)` runs on the
   destination: rebuilds the chare from the pulled buffer, closes any opened IPC
   pointers, then sends `ackGPUMigrate(id)` to the source.
4. **Source** (`ackGPUMigrate`) destroys the held chares — only now is it safe.

Transport selection is `DeviceMigrationStrategy::forPes(srcPe, dstPe)`:
MEMCPY (same process) / IPC (same node, cross process) / RDMA (cross node).
Contrast with reconverse's single `did_inter_node_gpudirect_rdma` boolean
applied *after* staging.

## Phases

### Phase 0 — prerequisite (DONE — result below)
Settle the `RandCentLB` `destPe: -1` abort using the `CHARM_NO_INTRAPROC_MIGRATE`
escape hatch (A/B, two short runs).

**Result: the failure is NOT the intra-process fast path. The base is broken.**

Same binary, same arguments:

| | fast path ON | fast path OFF (`CHARM_NO_INTRAPROC_MIGRATE=1`) |
|---|---|---|
| pic2d + RandCentLB | `Reason: Particle count not conserved at iteration` | crashes in `CkIndex_CkLocMgr::_call_immigrateGPU_marshall5` |
| jacobi2d + RandCentLB | aborts (signal 6) | — |

With the fast path disabled, migration still fails, *inside reconverse's own
GPU migration path* (`immigrateGPU`) — code untouched by this work. So migration
of GPU-resident chares is already broken on `cupti_lb_reconverse`. The earlier
`destPe: -1` was almost certainly a downstream symptom of corrupted location
state, not a root cause.

The fast path is not exonerated: it fails differently (loses particles rather
than crashing). **Both paths are broken.**

Consequences:
- This makes the port a **correctness fix**, not just an optimization. It also
  explains why every LB run reports 0 migrations: the GPU migration path has
  never actually been exercised.
- There is **no known-good baseline**. Phases 1-4 will be debugged against a
  base that already fails, so a Phase 3 lifetime bug cannot be distinguished
  from the pre-existing breakage by symptom alone.

Recommended follow-up (not yet done): establish a minimal working migration case
— a non-GPU chare array migrating under RandCentLB on this branch — to confirm
the host-side migration machinery is sound and the breakage is confined to the
GPU path.

### Phase 1 — types and declarations
- `CkDeviceMigrateHandle` → `ckrdmadevice.h`
- `CkArrayElementMigrateHandleMessage`, `CkLocMgrFinalizeMsg` → `cklocation.h` / `.ci`
- Entry methods: `immigrateGPUHandle`, `finalizeGPUMigrate`, `ackGPUMigrate`
- `_initGPUMigrateHandlers` initproc + `_gpuMigrateRdmaCompleteHandlerIdx`

All runtime dependencies verified present on reconverse: `hapiIpcGetMemHandle`,
`hapiIpcOpenMemHandle`, `hapiIpcCloseMemHandle`, `CmiSendDevice`,
`CmiDeviceBuffer`, `hapiMemcpyAsync`, `hapiAddCallback`, `gpu_size`,
`CkDeviceBuffer`.

Compiles; unreachable.

### Phase 2 — strategies
Port `DeviceMigrationStrategy` + `MemcpyStrategy` / `IpcStrategy` /
`RdmaStrategy` + `forPes()`. Swap the allocator per the decision above.
Still unreachable.

### Phase 3 — the handshake (highest risk)
Add `HeldMigratingChares` / `heldChares` and `PendingPull` / `pendingPulls`;
wire the four-step protocol. Reconverse's `emigrate` destroys immediately after
packing, so deferred destruction must be threaded through `duringMigration`,
`deleteElt`, and the location-cache epoch handling — precisely where the two
branches have diverged.

### Cross-node transport: correction (found during Phase 3)

The donor's `RdmaStrategy` is written against `CmiSendDevice`/`CmiRecvDevice`.
Those are charm-layer functions (`conv-core/conv-rdmadevice.C`, listed in
`cmake/converse.cmake`), but under `RECONVERSE=1` the `converse` target is
replaced wholesale — no `libconverse.a`, no `conv-rdmadevice.o` — and the
functions forward to `LrtsSendDevice`, which exists only in charm's ucx/mpi
machine layers that reconverse also replaces. So `RdmaStrategy` cannot link
here without porting charm's LRTS device-RDMA stack onto reconverse's LCI
backend. That is out of scope for this port.

**This does not mean cross-node GPU migration is unavailable.** It already
works: the existing path pushes a `CkDeviceBuffer` through the `nocopydevice`
entry-method parameter (`sendGPUMsg`, `cklocation.C:3042`), carried by
reconverse's comm backend (LCI with `LCI_USE_CUDA=ON`). None of the LRTS stack
is involved.

Consequences:
- `RdmaStrategy` is gated behind `CMK_DEVICE_RDMA_MIGRATE`
  (`CMK_GPU_COMM && !CMK_RECONVERSE`) and is dead code under reconverse.
- **Phase 4 must NOT route cross-node migrations into `forPes()`.** Doing so
  would turn working cross-node migration into an abort — a regression.
  Cross-node must keep using the existing `CkDeviceBuffer` path until a
  `CkDeviceBuffer`-based strategy replaces `RdmaStrategy`.
- A proper cross-node port means reimplementing the transport on
  `CkDeviceBuffer` (push, source-driven) rather than the donor's
  `CmiSendDevice` pull. The staging-copy win is still available there: push
  directly from the chare's own device buffers instead of packing into a
  staging buffer first.

### Phase 4 — rewire and remove
Split reconverse's `emigrate` into host-only pup + handle path, keeping the
intra-process fast path in front of both. Then delete `GPUMigrateData`,
`sendGPUBuffers`, `sendGPUMsg`, `did_inter_node_gpudirect_rdma` and the
source-side staging: 25 hits in `cklocation.C`, 5 in `.h`, 1 in `.ci`.

Keep the new path behind an env flag until confirmed on real multi-GPU
hardware; only then delete the old code.

### Phase 5 — validation
Compile; jacobi2d + pic2d single-GPU for no-regression; force migrations with a
random LB to exercise the handle path on the same-GPU transport.

## Open items

- **`pool_buff_mem_remaining` changes meaning.** Reconverse reports LB-buffer
  free size into `ProcStats` as a capacity signal. Removing source-side staging
  roughly halves LB-buffer demand, so `lb_buffer_size` sizing becomes
  conservative. Revisit once the path lands — it is the same signal the proposed
  LB GPU-memory constraint would consume.

## Validation limits on current hardware

Single GPU means only `MemcpyStrategy` is reachable — and it carries a comment
saying it should never be hit once the intra-process fast path exists.
`IpcStrategy` needs ≥2 processes on distinct GPUs; `RdmaStrategy` needs multiple
nodes. Compilation plus a single-GPU no-regression run is the ceiling here.
Phase 3 bugs will be lifetime/ack bugs, which surface as corruption under load
rather than clean aborts.
