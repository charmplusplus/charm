# MetisLB on GPUDirect LeanMD: migration memory/slot exhaustion, and the fix

Status: diagnosed 2026-09-13; implemented 2026-09-14 as part of the memory contract
under `+gpupool` (commits 95959779a..cdb754540, branch `rate-aware-gpu-lb`), and
awaiting the four-GPU check in `pool_contract.sbatch`.

What was built is broader than the window below, which became the pacing layer of
three:

- **Batches** (`LBBatchPlanner`, LBMemoryContract.h). The planner now sees the pool
  (capacity = pool free + device free in whole arenas) and splits a central step so
  each batch's payloads, landing arenas and per-PE IPC slots fit. This is what
  bounds MetisLB's repartition.
- **The admission gate** (`requestLanding`, cklocation.C). A landing is granted only
  above the floor of payloads the destination process still owes; otherwise it waits.
- **The window** (`ckWindowTake`, cklocation.C). As proposed below, per PE, bounded by
  the IPC slot budget and one arena of in-flight payload bytes; overrides
  `CHARM_LB_MIGRATE_WINDOW` and `CHARM_LB_MIGRATE_WINDOW_MB` (environment, not a `+`
  flag).

Three corrections to the diagnosis below:

- MetisLB is a CentralLB. Its issue loop is `CentralLB::ProcessMigrationDecision`,
  which was already batch-aware; the `DistBaseLB` loop cited below is DiffusionLB's.
  The batches never split because the stats reported pool free plus device free as
  the staging reserve.
- A failed pool arena growth prints "Failed to allocate GPU memory", not "Fatal CUDA
  Error [2] out of memory". That message comes from a hapiCheck'd call and names its
  file and line, which the logs should be read for.
- IPC event slots are sliced per PE, not per device.

## Summary

MetisLB cannot be used on the GPUDirect LeanMD benchmark: at the first load-balancing
step it aborts, either running out of CUDA IPC event slots or running the GPU out of
memory. The cause is not the pool size. MetisLB does a *global* repartition, so it
migrates most objects at one instant, and the runtime issues all of those device
migrations at once. Each in-flight migration holds an IPC event slot and a staged
device payload until its destination acks, so thousands of simultaneous migrations
exhaust both the slot pool and device memory. The fix is to bound the number of
concurrent in-flight migrations — a migration window — and drain the rest in waves.

This matters because on a concentrated, essentially static imbalance (LeanMD's
`-density gradient`), a global cut like Metis balances in a single step, whereas the
incremental DiffusionLB stalls at a fixed point (~1.43 max/avg on the device
dimension) because its bordering-only flow cannot reach the under-loaded far nodes.
Metis is the right tool for this load shape; it just needs to migrate affordably.

## Reproduce

    examples/charm++/cuda/gpudirect/leanmd
    ./leanmd 8 8 8 100 20 20 -computemap local -density gradient \
        +pe 32 +setcpuaffinity +gpushm +gpuipceventpool <N> +gpupool \
        +balancer MetisLB +balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim
    # launched -N 1 -n 4 --ntasks-per-node=4 (4 GPUs, 8 PEs/GPU), NUMA-pinned.

8x8x8 = 512 cells, ~14 computes/cell => ~7168 Compute objects. MetisLB repartitions
all of them at LB step 20.

## Symptom (measured)

    +gpuipceventpool 256    -> abort: "PE 0, device 0: no free CUDA IPC event slot
                               after 60s ... the transfers that would release these
                               slots cannot run" (slot exhaustion + a busy-wait deadlock)
    +gpuipceventpool 1024   -> "Fatal CUDA Error [2] out of memory"
    +gpuipceventpool 2048   -> "Fatal CUDA Error [2] out of memory"
    +gpuipceventpool 4096   -> "Fatal CUDA Error [2] out of memory"
                               (device pool had grown to arena 4 of 256 MB and beyond)

DiffusionLB alone (no Metis) does not abort — it migrates incrementally — but it does
not balance this load: it stalls and gives only a few percent over noLB.

## Root cause

1. `DistBaseLB::ProcessMigrationDecision` (src/ck-ldb/DistBaseLB.C, ~line 271) loops
   over every move and calls `lbmgr->Migrate(obj, to_pe)` back-to-back, synchronously.
   For a global balancer this fires all migrations at the barrier at once.

2. Each device migration stages its payload from the +gpupool arenas and claims a CUDA
   IPC event slot from the per-device pool. Both are held until the destination acks
   the transfer. src/ck-core/cklocation.C (~line 4433) states the hazard directly:
   "holding device memory until a peer acks is unbounded in the number of concurrent
   migrations."

3. With thousands of migrations live simultaneously:
   - a small slot pool (256) is exhausted; the host then busy-waits in
     `acquireIpcSendSlot` for a slot that cannot free, because the peers that would
     free it are stuck in the same wait -> 60 s deadlock abort;
   - a large slot pool (>=1024) has enough slots, so all payloads proceed, and the
     device pool grows arena after arena until CUDA is out of memory.

   Neither pool size helps because the quantity that must be bounded is *concurrency*,
   not pool size.

## Fix: a bounded migration window

Cap the number of concurrent in-flight (staged-but-not-yet-acked) device migrations at
K, and issue the remaining moves in waves as earlier ones complete.

- In `ProcessMigrationDecision`, stop issuing the whole move list synchronously. Issue
  up to K, and as each migration's destination-ack completion fires (the event that
  frees its pool payload and its IPC slot), issue the next queued move. The plumbing
  already exists: `ledgerOpen`/`ledgerClose` account for outstanding moves and
  `ledgerClose` already waits for every deferred move before the step completes, and
  the `DeferredMigrateMsg` path already defers a move and fires it later. The change is
  to gate issuance on an in-flight counter instead of firing the list in one pass.

- Size K as the minimum of two budgets, per device (the pool is per device, shared by
  the PEs on that GPU, so divide by PEs-per-device for a per-PE cap):
    * the per-device CUDA IPC event slot pool, and
    * a device-memory reserve for staged payloads — the "reserve the balancer can plan
      against" that the cklocation.C comment and `heldTransport()` already refer to.
  A tunable such as `+LBMigrateWindow K` (default derived from `+gpuipceventpool` and
  the arena budget) lets it be set explicitly.

This keeps Metis's full global decision — every move it chose still happens — and only
paces the transport, so Metis balances the gradient in one LB step while the migration
footprint stays inside a bound. It also helps any balancer that moves many objects at
once, not just Metis.

## Alternatives considered and rejected

- Raise `+gpuipceventpool`: does not bound memory; OOMs at 1024 and above.
- Cap Metis's move count (a move budget like GreedyRefine's `+LBPercentMovesAllowed`):
  mechanically avoids the flood but discards the balance quality that is the reason to
  use Metis — a partial Metis cut does not fix a concentrated imbalance the way a full
  one does.
- Fix DiffusionLB's reach instead (relax the Sep-10 bordering-only flow, or make its
  `+LBDiffusionRemapAbove` scratch-remap trigger): addresses the wrong balancer for
  this load shape; diffusion has a fixed point above balance on a concentrated static
  imbalance regardless of step count (measured: device max/avg plateaus at ~1.43 over
  20 static steps in the LB simulator).

## Validation

Cannot be checked in the login-node LB simulator (`tests/charm++/load_balancing/lbdriver`),
which replays only the balancer's decision, not the transport. It needs a GPU run:
rerun the reproduce command with MetisLB and confirm no slot-exhaustion abort and no
CUDA OOM at a modest pool size, and that the post-LB steady-state step time drops
(the placement is actually balanced).

## References

- src/ck-ldb/DistBaseLB.C  `ProcessMigrationDecision` — the all-at-once issue loop.
- src/ck-core/cklocation.C ~4433 — "unbounded in the number of concurrent migrations".
- src/ck-core/cklocation.C `acquireIpcSendSlot` region — the 60 s slot busy-wait.
- Balancer stall context: DiffusionLB commit 339904c20 "flow only to bordering
  neighbours" (Sep-10) is why DiffusionLB cannot substitute for Metis here.
