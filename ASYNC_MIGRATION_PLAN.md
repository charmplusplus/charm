# Async device-migration pulls: plan

Goal: restore `hapiAddCallback`-based completion in `MemcpyStrategy` /
`IpcStrategy` (`cklocation.C`), so the destination PE is not blocked for the
duration of the pull, without reintroducing the hang that the synchronous fix
(`fb7c40ad2`) resolved.

## What actually goes wrong with async completion (grounded, partially verified)

With async completion, `immigrate()` — which constructs the chare — is deferred
until the CUDA pull finishes. Between the source releasing the chare and the
destination constructing it, the chare exists on no PE. Peers resume from the
load balancer independently (observed on Delta: PE 5 resumed while PEs 11/24/31
were still in `finalizeGPUMigrate`) and send the next iteration's messages to
the migrated chare.

What happens to such a message on the destination (read from `ckarray.C`, not
yet traced at runtime):

- `CkArray::recvMsg`: `lookup(id)` fails; `locMgr->whichPe(id)` returns THIS PE,
  because with `CMK_GLOBAL_LOCATION_UPDATE` every PE's location cache was
  already updated at `ReceiveMigration`, before the chare physically arrived.
- `sendToPe(msg, myPe, type)` with `type == CkDeliver_queue` re-enqueues the
  message to this PE via `CkArrayManagerDeliver` (ckarray.C ~line 1929).
- The redelivered message repeats the cycle: **livelock**. The scheduler spins
  redelivering the same messages forever; the run hangs with all PEs busy.
  This matches the observed hang signature (busy scheduler, no progress).
- The existing buffer path (`bufferedIDMsgs[id]`, ckarray.C ~1955) is only
  reached on the INLINE delivery path, not the queued path, so it never
  engages.

Replay machinery for `bufferedIDMsgs` already exists and runs at element
creation (ckarray.C ~1114). So the fix is routing, not new infrastructure.

## Phase 0 — verify the livelock diagnosis (one Delta run)

The livelock mechanism above is read from code; verify it before building on
it. Add env-gated (`CHARM_DEBUG_MIGRATE`) counters in `CkArray::recvMsg` /
`sendToPe`:

- when `lookup(id)` fails and `whichPe(id) == CkMyPe()`, print once per id with
  a repeat count (id, ep, hops).

Run the async version (temporarily re-enabled behind an env var, see Phase 2)
on Delta, 4 proc / 32 PE with the balancer. Expected if the diagnosis is right:
a handful of ids with exploding repeat counts on the PEs that were still
pulling. If instead messages are buffered or dropped, STOP and rediagnose.

## Phase 1 — break the livelock: buffer for in-flight chares

Destination-side "expected arrivals" registry in `CkLocMgr`:

- Register id when the destination first learns a chare is coming:
  - `immigrateGPUHandle` (device handles arrived first), and
  - `immigrate()`'s buffering branch (host message arrived first,
    `bufferedHostMigrateMsgs`).
- Deregister when `immigrate()` completes (after `ckJustMigrated`).
- In `CkArray::sendToPe` (queued-delivery, `pe == CkMyPe()`, `lookup` fails):
  if the id is registered as in-flight, push onto `bufferedIDMsgs[id]` instead
  of re-enqueueing. Existing replay at element creation then delivers in order.
- The queued-path re-enqueue for ids NOT in the registry stays exactly as is
  (it covers unrelated races that presumably resolve).

Notes:
- `bufferedIDMsgs` replay ordering vs SDAG refnums is safe: SDAG buffers by
  refnum internally, so delivery order does not matter for correctness.
- The registry must live in `CkLocMgr` (per PE) but be consulted from
  `CkArray::sendToPe`; both are per-PE groups, accessor via `locMgr`.
- Messages can also arrive at the SOURCE (stale cache on a third PE that missed
  the update); source forwards via its updated cache to dest as today. Only the
  dest needs the registry.

## Phase 2 — re-enable async completion, switchable

Both strategies in `cklocation.C`:

- Keep the current overlapped-async copies + single `hapiStreamSynchronize` as
  the DEFAULT (known good).
- Add `CHARM_ASYNC_MIG_PULL=1` to switch to `hapiAddCallback` completion
  (`finalizeGPUMigrate` via callback, code shape from git history at
  `fb7c40ad2^`). One build, A/B on Delta by env var — no rebuild per
  hypothesis.
- Use a dedicated migration stream (one per DeviceManager, created lazily),
  NOT stream 0: legacy stream 0 synchronizes against all blocking streams, so
  pulls on it serialize with application streams and vice versa.
- Flip the default to async only after Phase 3 passes; then remove the switch.

## Phase 3 — validation

Local (single process; I can run):
- pic2d + GreedyRefineCentralGPULB, handle path forced
  (`CHARM_NO_INTRAPROC_MIGRATE=1 +gpushm ...`), both env settings, 10+ runs.
- jacobi2d -z and non-z regression, both env settings.

Delta (cross-process; requires Aditya):
- 2 proc / 2 GPU jacobi2d and pic2d, no LB (regression: cross-process transfer
  still works).
- 4 proc / 32 PE pic2d + balancer, `CHARM_ASYNC_MIG_PULL=1`, repeated ≥5 times
  (the failure is a race): check `RESUME count == 64` and completion, via
  `> run.log` + grep on the file, never piping the live run.
- Same with `CHARM_ASYNC_MIG_PULL` unset (control).

## Risks / open questions

- AtSync barrier epoch: with async completion the chare registers with the
  destination barrier AFTER that PE's `ResumeClients` may have advanced
  `curEpoch`. Reading `CkSyncBarrier::addClient`, a new client at the current
  epoch is simply not yet counted and blocks the barrier until it calls
  `AtSync` — correct as long as the chare eventually runs, which Phase 1
  guarantees by delivering its buffered messages. Verify explicitly in Phase 3
  by running ≥2 LB steps (`-b` small enough for a second AtSync).
- `hapiPollEvents` head-of-line: completion events are polled FIFO per PE; a
  long pull event at the queue head delays app callbacks behind it. Bounded by
  pull duration; acceptable, but worth knowing.
- Demand-creation arrays (`CkArray_IfNotThere_createhere/...`): the registry
  check must run BEFORE any demand-create decision for in-flight ids, or a
  spurious second element could be created.
- Expected payoff is small: pulls are 590 KB–5.9 MB (25–240 us over PCIe),
  once per migrated chare per LB step. This is correctness-hardening plus
  hygiene, not a measurable speedup on pic2d. The 50% balancer regression is a
  separate, larger issue (objGPU ~0.1% of step; communication locality not in
  the cost model).

## Explicitly out of scope

- Cross-node (RDMA) migration: unchanged, still the staged path.
- The balancer cost model / communication locality.
- Removing the synchronous fallback before Phase 3 passes on Delta.
