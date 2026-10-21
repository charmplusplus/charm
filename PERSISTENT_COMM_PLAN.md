# Persistent device communication: implementation plan

Date: 2026-10-17. Branch: rate-aware-gpu-lb at e53a99c5d. Written from a read of the tree; line
numbers are pointers into today's files and will drift.

## 0. What exists today, and exactly where it stops

The API is `CkDevicePersistent` (upstream "Persistent Intra-node GPU Messaging", 1cf289600),
declared at `src/ck-core/ckrdmadevice.h:14-63`, implemented at `src/ck-core/ckrdmadevice.C:867-960`,
exercised by `examples/charm++/cuda/gpudirect/persistent` and
`benchmarks/charm++/cuda/gpudirect/latency-persistent`.

An endpoint is `{ptr, cnt, cb, cb_msg, hapi_stream, pe, hapi_ipc_handle, ipc_ptr, ipc_open}`.
`open()` is one `cudaIpcGetMemHandle` on the user pointer. `get(src)` / `put(dst)` call
`findTransferModeDevice(src.pe, me)` and then:

| mode | what happens | status |
|---|---|---|
| MEMCPY (same process) | `hapiMemcpyAsync` on the initiator's stream | works |
| IPC (same physical node, other process) | `cudaIpcOpenMemHandle` once per struct copy (`ipc_open` latch), then a device-to-device copy | works only without the pool, see below |
| RDMA (other node) | `CkAbort("... not supported for inter-node messages")` | missing |

Completion: both sides' `CkCallback`s are queued on the **initiator's** stream with `hapiAddCallback`,
so the passive side learns of a transfer only through the callback message. There is no receiver
notification entry, no sequence number, no flow control, no byte count (the whole `cnt` is copied).

### 0.1 Why it does not work in this branch's default configuration

1. **`+gpupoolalloc vmm`** (the default pool backend off-node, [[vmm-pool-backend]]): `cudaIpcGetMemHandle`
   cannot name `cuMemMap`'d memory. The branch's direct sends carry a `HapiVmmIpcHandle` overlay instead
   (`hapi_impl.cpp:4761-4800`) produced by `hapiIpcExportBuffer` and consumed by `hapiIpcImportBuffer`.
   `CkDevicePersistent::open()` calls the raw `hapiIpcGetMemHandle` inside `hapiCheck`, so on pool memory
   it aborts.
2. **Buddy pool**: the raw handle names the whole arena. Each struct copy opens its own mapping with
   `cudaIpcOpenMemHandle`, bypassing the process-wide import cache (`hapiIpcImportBuffer`, keyed by
   exporting base and validated by the handle). A region already imported by the cache cannot be opened
   again (`cudaErrorAlreadyMapped`), and the reverse order leaves the cache unable to open it. The
   GPU_IPC_PROTOCOL_SWITCH_PLAN already lists "fold `CkDevicePersistent` onto the shared import cache"
   as the follow-up cleanup; it was never done.
3. **Inter-node**: aborts. Every primitive needed exists elsewhere in the tree: reconverse's one-sided
   `CmiIssueRput` / `CmiIssueRget` (`reconverse/src/conv-rdma.cpp:474-535`, LCI put/get with the
   mr/rmr written by `CmiSetRdmaBufferInfo`), device registration through `acquireDeviceRegistration`
   (arena registered once under `+gpupool`, `ckrdmadevice.C:519-577`), and an RDMA **put from device
   memory with a target notification**: the restage put (`issueDeviceRestagePut`, opMode
   `CMK_DEVICE_RESTAGE_PUT`, `notifyDeviceRestagePut`, `ckrdmadevice.C:2074-2160`), which also shows the
   one ordering rule an off-node put has (issue it from the producer stream's completion callback,
   because the NIC reads outside every CUDA stream).

### 0.2 Why it does not survive migration

- `pup()` ships `ptr, cnt, pe, cb, hapi_ipc_handle` by value. Nothing invalidates a peer's copy when the
  owner moves; the header says "needs to be recreated and exchanged again". The example closes the
  peer's handles in `ckAboutToMigrate` and re-runs the whole exchange after every LB step behind
  `CkWaitQD` (a job-wide barrier per LB step).
- A peer's copy names a **PE**. After the owner's element migrates, the pointer is dead at the new home
  (device state is re-allocated at unpack) and the PE is wrong; the mode decision
  `findTransferModeDevice(src.pe, me)` is made against the stale PE, so a put can target a process
  that no longer holds the element.
- The migration gate counts only `CkDeviceBuffer` sends (`outstandingDeviceSends`,
  `noteDeviceSendPosted/Done`, `deviceRecvParked`, `cklocation.C:3041-3130`). A persistent put in
  flight into an element that is being packed is invisible to it.

### 0.3 What a per-message device send costs today (what a channel removes)

Per `nocopydevice` payload: sender `CkRdmaDeviceOnSender` (location lookup, mode choice, registration
or IPC export lookup, IPC slot claim, drain-then-sync or deferred prepare), an envelope with an
~80-byte descriptor per buffer; receiver `CkRdmaDeviceIssueRgets` (admission, mode re-check, landing
resolution, copy issue through the batch, completion by event ring or pinned flag), an ack message back
to the sender, the sender's completion callback, the LB graph weight. Two messages per payload plus,
off-node, a receiver-initiated rget. A channel put is one write and one notification.

## 1. Where it can pay, and where it cannot

- **sph3d 98M / 8 A40s**: kernels-only traces put the GPU idle at 18-22%, nearly all of it the two
  exchange-round waits per step with the partner GPUs busy (skew), not transport
  ([[sph3d-exchange-waits]], 2026-10-10: "a persistent one-sided channel would attack the 19% that is
  not skew: not worth it"). What a channel can buy on sph3d is **host time**: the two-node step tracks
  the busiest PE's host time ([[n2-step-tracks-busiest-pe-host-time]]); 52 device messages per patch
  per step ([[sph3d-exchange-waits]]) = ~13k messages per step job-wide, ~420 per PE per step at
  ~40 ms/step, one every ~95 us of PE time. The saving is 420 x (send + receive host cost per message).
- **leanmd 16x8x8**: ~1000 cells, ~14 computes per cell; per step each compute receives one or two
  position arrays and returns a force array to each cell, and cells exchange migrants in 26 directions:
  of the order of 80k small device messages per step over 64 PEs (~1300 per PE per 1.0-1.5 s step).
  Fixed partner sets between LB events; ~750 Compute migrations per LB event every 20 steps is the
  migration stress test. This is the app whose message handling is the largest share of the step.
- **moe**: ~0.35 ms per message receive cost, open ([[moe-example]]).
- **sph2d small sizes**: host-bound below 60k particles per patch ([[sph2d-lb-loses-on-migration-cost]]).

The number that decides the application work is the per-message host saving measured by the
microbenchmark of step 2 and the per-message host cost of the current path (step 0). Nothing in this
plan is worth doing for sph3d's GPU idle; it is worth doing for host time on leanmd and sph3d, and for
moe's receive cost.

## 2. Design

A runtime-managed persistent channel between two chare-array elements (or group branches), one
fixed destination buffer per channel on the receiver, one-sided puts by the sender in all three transfer modes,
completion delivered to the receiver element as an entry-method call, migration handled by the runtime
with no application re-exchange, traffic visible to the balancers.

### 2.1 Endpoint descriptor (what a peer holds)

`{owner (array/group id + element id), home PE, generation, device index, the destination buffer (ptr in the owner's
address space and capacity), fixed for the channel's life, IPC export (handle + offset + base from hapiIpcExportBuffer: works for
cudaMalloc, buddy and vmm pool memory, served by the export cache), RDMA registration (CmiNcpyBuffer
from acquireDeviceRegistration: the arena's one registration under +gpupool, rmr in layerInfo),
notification callback (CkCallback to the owner element's entry)}`.

Everything is filled once at `open()`; a put reads it and issues, which is what makes the channel
persistent. `generation` is bumped at every re-open (migration); a peer's copy with an older generation
is stale by definition.

### 2.2 `put(src_ptr, bytes, stream)` by mode

| mode | write | source readiness | completion to the receiver | what disappears vs today |
|---|---|---|---|---|
| MEMCPY | `cudaMemcpyAsync` on the sender's stream into the destination buffer | the stream carries the producer | a `hapiAddCallback` on the sender's stream sends the notification message (channel id, seq, bytes) to the owner element | envelope + descriptor, receiver copy issue, ack |
| IPC | import the peer's export once (import cache), peer-to-peer `cudaMemcpyAsync` from the sender's stream into the mapped slot | same | same | the per-message IPC event slot handshake (`hapi_ipc_event_shared src_ready / dst_flag`, slot claim and reclaim), the receiver's copy, the ack |
| RDMA | `CmiIssueRput` from the sender's registered source into the peer's registered destination | the put is issued from the producer stream's completion callback (as `deviceRestagePutReady` does today) | a new opMode `CMK_DEVICE_CHANNEL_PUT` beside `CMK_DEVICE_RESTAGE_PUT` (`converse.h:995`): the initiator's completion sends the notification to the target PE, delivered to the element | the envelope, the receiver-initiated rget, the ack |

The notification is always a Charm message to the element (so it is forwarded by the location
manager if the element has moved, and so SDAG / plain-entry receive patterns keep working); the
device-side guarantee is that it is sent only after the write completed. The sender's source is free
when its own stream callback runs (MEMCPY/IPC) or when the put's local completion fires (RDMA); the
sender's optional callback fires then.

### 2.3 Receiver side: one buffer per channel, release as the credit

A channel has one destination buffer, fixed at `open()`. That is what makes it persistent and what
lets a put carry only `(seq, bytes)`. A put writes `bytes <= capacity`; the receiver's entry gets
`CkDeviceChannelEvent{seq, bytes}`. The sender must not overwrite the buffer while the receiver may
still read it, so the receiver calls `release()` once the kernels that read the put are ordered behind
it (a stream event or pinned flag the app records, or simply after the step that consumed it);
`release` sends one small credit message, and the sender's next put on that channel waits for it,
queued on the sender, never blocking the PE. An application that wants step t+1's data written while
step t's is still being read opens two channels on two buffers and alternates them; the runtime does
nothing special for that.

One put per channel per step with consumption before the next step (sph3d halos, leanmd positions
and forces) waits on a single buffer only when the receiver is behind by less than the credit round
trip, since the next put comes a whole step later. Later option, not in this plan: in a symmetric
exchange the reverse put is issued after the consuming kernel and could stand in for the credit.

### 2.4 Ordering of data and notification

MEMCPY/IPC: the notification is sent from the stream callback after the copy; the copy's completion
makes the bytes visible in the destination (peer mappings included). RDMA: the notification follows the
put's local completion, exactly the ordering the restage put relies on today; whether LCI/CXI local
completion of an RMA write implies remote visibility must be confirmed in LCI's put completion
semantics (step 1 verifies it with a checksum on the receiver at 2 nodes). If it does not, the
notification is sent as an LCI message on the same endpoint, which CXI orders after the write.

### 2.5 Migration

- **Registry**: an endpoint registers with the owner's `CkLocRec` at `open()` (as device registrations
  do: `CkRdmaDeviceDropRegistrations(owner)`), together with the list of peers that received its
  descriptor (the exchange goes through a runtime entry, so the owner knows every holder).
- **Departure (two-phase close, runtime-driven, before pack)**: when the element is chosen to migrate,
  the runtime sends `closing(channel, generation)` to every holder and the element stands down (the same
  stand-down `outstandingDeviceSends` gives today) until each holder acks; a holder acks when it has no
  put in flight to that endpoint and marks its copy stale. Puts a holder issues meanwhile queue behind
  the re-resolution below. Then the endpoint is closed (export invalidated with `hapiIpcInvalidateExport`,
  registration dropped if it is not the arena's) and the element packs. Cost is at LB events only, a few
  thousand tiny messages per event for sph3d (52 peers per patch) or leanmd (dozens per element), on top
  of an event that costs 10-40 ms today; it is measured in step 3.
- **Sender side at `put()`**: compare the held descriptor's PE with the location manager's current
  answer for the owner (`whichPe` / last known) and its generation with the last seen; if stale or
  unknown, hold the put (the source stays live and counts as an outstanding device send on the sending
  element, so the sender does not migrate mid-put) and send `resolve(channel)` to the owner element; the
  reply carries the fresh descriptor (new home, new ptr, new exports, generation+1) and the held put is
  issued. One round trip on the first post-migration put, nothing after, which is what the correction
  protocol costs today for a `CkDeviceBuffer` send.
- **Arrival**: the element re-opens its endpoints at the new home when its `pup` re-registers the landing
  buffer on unpack (`open()` on the new pointer; exports and registration are computed then). Landing
  data is not migrated by the runtime: a buffer the app had not consumed before `AtSync` is the app's
  state to pup, as it is today.
- **In flight across the close**: a put that was issued before `closing` arrived completes into the old
  home's buffer and its notification is forwarded to the new home by the location manager. The
  two-phase close exists precisely so that the element does not pack while such a put is in flight: the
  holder acks only after its completion. The notification then arrives at the new home for the
  old buffer: the receiver treats a notification whose generation is older than its endpoint's as
  "consumed before migration" and drops it. The put's data is lost to the receiver, so the sender must
  re-put after re-resolution: the sender keeps the last put per channel until the receiver's `release`
  credit (not the completion) acknowledges it, and re-issues it into the new endpoint. This is the only
  piece of the design that is new protocol rather than reuse, and it is what makes the app code
  migration-free.

### 2.6 Balancers and instrumentation

Each put records its bytes against the sending element's edge to the owner, as device sends do
(`CkRdmaDeviceTakePendingSendBytes` feeds the LB comm graph), so MetisLB's bytes-weighted cuts and
DiffusionLB's comm term see channel traffic. CUPTI attribution is unchanged (copies run on the app's
stream). `+gputrace` L-records and the message tier histograms get a channel kind so sph3d's tier
accounting and gap_report.py keep working. `CHARM_ZC_STATS` counters gain a channel slot.

### 2.7 API sketch

```
// receiver: one fixed destination buffer per channel
CkDeviceChannel ch(d_halo_in[dir], capacity_bytes,
                   CkCallback(CkIndex_Patch::haloArrived(nullptr), thisProxy[thisIndex]));
ch.open();                                   // exports and registers once; registers with my CkLocRec
thisProxy[peer].takeChannel(dir, ch);        // descriptor travels by pup (runtime records the holder)

// sender
CkDeviceStatus s = peer_ch[dir].put(d_src, bytes, stream /*, optional CkCallback src_done */);

// receiver entry: consume, then free the buffer for the next put
void haloArrived(CkDeviceChannelEvent* e) { /* e->seq, e->bytes */ ...; ch.release(); }

// double buffering = two channels on two buffers, alternated by the application
```

`CkDevicePersistent` is kept as a thin layer over the channel for the existing example and benchmark,
or removed once they are ported; its `get()` becomes a put requested from the other side (a one-sided
read is not needed by any app here and the RDMA read path has no target notification).

## 3. Steps, in order, each with its test

Allocations follow the standing rules: builds and single-node tests on one A40 node; cross-node arms
ride a 2-node allocation already granted for something else, one arm per configuration, in-session
control, steady state only.

**Step 0 (no code, one extra arm on the next 2-node allocation).** Measure the current per-message host
cost: `CHARM_ZC_STATS=1` on a 1000-step sph3d control arm and a leanmd arm. The `[zc-host]` line gives
send mean_us and recv-issue mean_us per message; multiplied by the messages per PE per step this is the
upper bound of the host saving. Decide from it which application goes first and whether step 4 is worth
doing at all.

**Step 1: make the existing API correct in this branch (~1 day).** Route `open/get/put` through
`hapiIpcExportBuffer` / `hapiIpcImportBuffer` (pool, vmm, import cache); add the RDMA mode with
`CmiIssueRput` and a `CMK_DEVICE_CHANNEL_PUT` opMode whose initiator completion sends the notification
(copy of the restage-put pattern), source readiness from the producer stream's callback. Test:
`examples/.../persistent` at 2 PEs in one process, 2 processes on one node with `+gpupool` buddy and
vmm, and 2 nodes with a receiver-side checksum (the visibility check of 2.4). Deliverable: the API
works in all three modes with the pool.

**Step 2: the channel object and the microbenchmark (~1-2 days).** `CkDeviceChannel`, descriptor pup,
notification handler to the element's callback with `{seq, bytes}`, the release credit, LB graph
bytes. Rewrite `latency-persistent` into a put-based ping-pong plus a one-directional rate test and
compare with the per-message `nocopydevice` path at 1 KB-4 MB in the three modes: message rate per PE
and host us per message. This number against step 0 decides step 4.

**Step 3: migration (~2-4 days, the investment).** Endpoint registry on `CkLocRec`, generation,
stale detection at `put()`, `resolve` round trip, two-phase close before pack with holder acks,
outstanding-send accounting for held puts, lazy re-open on unpack, redo of the last unreleased put,
generation check on notifications. Test: the persistent example's `-l` mode with the application-side
re-exchange removed, under RotateLB every step, then under Metis+Diffusion with `+LBAsync`; one node,
then the 2-node arm. Kill criterion for the design: a hang or a lost put under RotateLB at one node.

**Step 4: first application.** leanmd if step 2 shows the saving is per-message (many small messages,
fixed Cell<->Compute partner sets, migrations every 20 steps); sph3d halos if it is per-byte (26
directions x 2 payload kinds per patch, capacity = the halo maximum). sph3d lands ghosts today at a
base that moves with the migrant count, so a channel version gives each direction a fixed ghost
region and the layout pass reads 26 segments instead of one range: an application change inside this
step. Measured with the
standing protocol: 2 nodes, in-session control, sync and async with equal period and lag, Metis once
then Diffusion, second-half ms/step.

**Step 5: hardening.** Interplay with the correction protocol (a channel put and a `CkDeviceBuffer`
message to the same element around a migration), the async LB landing gate (channel buffers are
app-owned, not landing-gate allocations: nothing to reserve), tiers and traces, zc-stats, the stall
watchdog, close and deregister at exit, the multi-GPU-per-process layouts (one import per peer heap,
per-device exports).

## 4. Risks and open questions

- RDMA remote visibility at local put completion (2.4): verify in LCI before step 2 builds on it.
- Registration cost of app buffers outside the pool: LCI registers on every LCI device
  ([[lci-registers-on-all-devices]]); paid once per open, so irrelevant at steady state, but a channel
  per direction per patch is 52 registrations per patch unless the landing buffers are pool
  allocations (they are, under `+gpupool`: the arena's one registration covers them).
- Flow control is new behaviour for the apps: a put before the previous one was released queues, so
  the credit round trip is on the receiver's path whenever the receiver is behind by less than that
  round trip. Two channels on two buffers remove it; the API must print a diagnostic when a put has
  waited longer than a step.
- The two-phase close adds messages to every LB event; it is bounded by the holder count and must be
  measured against today's 10-40 ms event (step 3).
- The gain ceiling on sph3d is host time, not GPU idle; if step 0 shows send + receive under ~5 us per
  message, the sph3d gain is under 2 ms/step and only leanmd and moe justify step 4.

## 6. Status

**2026-10-18 00:05, built and unvalidated** (the allocation for the tests is still queued):

- Step 1 done in code: `CkDevicePersistent` exports through `hapiIpcExportBuffer`, peers import through the
  cache, cross-host put/get through `CmiIssueRput/Rget` with opMode `CMK_DEVICE_PERSISTENT_OP` (added to
  `reconverse/include/converse.h` and the installed copy; passed through by the machine layer), completion
  fires both endpoints' callbacks from the initiator, a put is issued from the producer stream's callback.
  The example allocates from the pool under `+gpupool`. Test: `examples/.../persistent/test_persist.sh`.
- Step 2 done in code: `CkDeviceChannel` (one fixed buffer, `put(src, bytes, stream)`, `event()` /
  `release()`, one queued put, LB graph bytes) and the microbenchmark
  `benchmarks/charm++/cuda/gpudirect/channel` (`run_channel.sh <jid> <memcpy|ipc|rdma>`: zero-copy entry
  vs channel, one-way latency and one-in-flight rate with the sender's issue cost per message).
- Step 3 done in code, with one change to the plan: there is no two-phase close over a holder list the
  receiver learns from puts. A sender resolves the channel before its first put (a RESOLVE event through
  the `arrived` callback, so it reaches the receiver wherever it lives; the reply carries the endpoint,
  the generation and whether the buffer is free), which is how the receiver learns its sender; the
  receiver's migration (`CkRdmaDeviceChannelsBeforeMigrate` from `CkLocMgr::emigrate`) sends `closing`
  to that sender and charges one outstanding device send until the ack (a holder acks when it has no
  put in flight; a holder that has left its PE is acked for); the receiver reopens on its new buffer in
  its pup (`reopen`), answers resolves that arrived while it was moving, and the sender re-resolves. A
  put's bytes that landed before the pack are in the buffer the application pups; the notification is
  forwarded. A put counts as an outstanding device send of the sending element, so a sender never
  migrates mid-put. No redo of the last put is needed under these rules.
  Test: `examples/.../chantest/test_chantest.sh` (two elements, verified exchange, `migrateMe` swaps every
  3 iterations in each transport; RotateLB does not exist in this tree).
- Step 0 ready: `sph3d/n2_zcstats.sh <jid>`. All of it runs from `sph3d/after_head1.sh <jid>` in the
  leftover minutes of the reproduction allocation.

**2026-10-18 03:40, two 2-node allocations later (jobs 22965711 leftover, 22974411):**

- Step 1 VALIDATED: `test_persist.sh` passes all 7 arms on two node pairs (memcpy plain/vmm, ipc plain/buddy/vmm,
  rdma plain/vmm), each verifying the received values in the completion callback.
- Step 3 across processes VALIDATED: `chantest` ipc_lb and rdma_lb pass (24 iterations, 14 element migrations by
  `migrateMe`, channels never re-exchanged by the application). The in-process handoff (`emigrateIntraProcess`,
  no pup) needed leave/arrive hooks and a process-wide element->channels registry; its first run failed because
  element id 0 was treated as "no element" (fixed 03:09, unvalidated). A receiver that migrates between event()
  and release() now carries the taken event in its pup (the async leanmd arm aborted on that).
- Step 2 numbers (`channel` microbenchmark, CHARM_ZC_STATS=1): same process, two PEs: one-way 1 KB-64 KB zc 23-25 us
  vs channel 16-18 us, 512 KB 41-46 vs 34-36, 4 MB tie at 190; one-in-flight rate 43k vs 59-74k msg/s; sender issue
  2.3 us vs 0.1 us. Two nodes: one-way tie at 61-63 us up to 64 KB, channel slower at 512 KB (96-99 vs 83-85) and
  4 MB (379-388 vs 306-309: the put is issued from a stream callback, and an LCI put against the zc path's rget);
  the receiver's release() costs 12 us (the off-node converse send of the credit). The ipc layout of the benchmark
  hangs before its first measurement in both zc and channel modes (chantest's ipc arms pass): open.
- Step 0 (sph3d async control, 1000 steps, 39.2-39.4 ms/step): [zc-host] per process send 0.3-0.7 us/msg,
  recv-issue 1.7-7.5 us/msg (mean ~4.5), 0.7-1.5M messages per process per 1000 steps (~250 per PE per step): the
  zc host work a channel removes is 1-2 ms/step per PE, 3-5% of the step, as the plan's bound said.
- Step 4 (leanmd `-channels`, binary leanmd.ch, 16x8x8 100 steps, 4,1,8): smoke 8x4x4 passes its energy check;
  noLB TIES the control (1495.8 vs 1495.6 ms/step over steps 21-100; 1505.6 vs 1510.0 over 42-100): at 1.5 s per
  step the message handling is under 1% of the step. The LB arms failed on two channel bugs: the async arm on the
  migrating receiver's release (fixed), the sync arm on one migration standing down forever (one outstanding count
  never reached zero; diagnostics added, next allocation). The channel run's balancer also moved 8 objects against
  the control's 8510: the comm-graph edge was recorded to the sender itself; fixed to the channel's owner.
- HEAD reproduction (chain head1): sph3d 2000 steps noLB 56.6 / async 38.7 / sync 40.9 (2nd half 56.9/38.8/41.0),
  (1,4,4) noLB 57.0; leanmd none 1494.7/1510.3, async 1139.9/1019.2, sync 1123.1/989.4; all within the 10-17
  references. pic2d 2-node aborts at step start on HEAD in all modes (index compress overflow, a garbage 2D index
  on a location path right after a pool growth: the open pic2d heap-corruption item, not new). The sph3d tier print
  is still wrong in multi-GPU processes (process-local vs host-local device index); diagnostic only.

**2026-10-18 05:35, end of the night (third allocation, job 22979444):**

- Step 3 holds: `chantest` passes all six arms on three node pairs, including the in-process handoff
  (element id 0 fix) and migration of a receiver between event() and release().
- Two more runtime fixes from the leanmd arms: a resolve that arrives while the receiver's migration is pending is
  answered at once and the sender closed right after (holding it deadlocked the split barrier), and a cross-host put's
  completion is forwarded to the initiating PE (LCI raises it on whichever PE runs progress; the sender's bookkeeping
  is per PE, and its element's outstanding count climbed for ever).
- leanmd `-channels`: no-LB ties the control at 16x8x8 (1495.8 vs 1495.6) and 8x4x4 (203.4 vs 202.1 ms/step). Every
  LB arm still hangs at the end of the first balancing wave (AtSyncWait at step 36): all stand-downs were released and
  kicked, 4411 pup migrations and 1730 handoffs completed, but the migration window's second batch never completes on
  any PE. Not a channel stand-down. Suspects: the full-capacity slot pups under `-channels` (392-728 KB per Compute
  against nPart-sized pups before) against the migration window, the landing gate or the batch planner; or a lost
  window give-back on the handoff path. Next: `CHARM_DEBUG_MIGRATE=1` with the window prints on `ch_async_8x4x4`
  (hangs within 30 s), and the control's EMIG sizes for comparison; alternatively pup `nPart` entries and reopen the
  channel on the smaller slot. During steps 21-36 after the first wave the channel run stepped at 867 ms (16x8x8) and
  105 ms (8x4x4) against the control's 1112-1156 and 141-148, which is suggestive but not a result from a finished run.
- Open besides: the `channel` benchmark's ipc layout hangs before its first measurement in both zc and channel modes
  (group chares; the array-based chantest ipc arms pass); the receiver's release costs 12 us off-node (the converse
  send of the credit); pic2d at two nodes aborts on HEAD before any of this (open pic2d heap-corruption item).

**2026-10-18 08:20, fourth allocation (job 22981955): the leanmd channel runs complete, and they are faster.**

- Root cause of the LB hangs: element ids repeat across arrays (a Cell and a Compute share id values; ids are
  unique only within one location manager), and the channel registries and `ownedBy()` were keyed by id alone, so a
  Compute's migration closed and charged the same-id Cell's 55 channels, the acks credited the Cell, and the Compute
  stood down for ever. Everything is keyed by (location manager, id) now. Also fixed: cross-host put completions are
  forwarded to the initiating PE; a resolve during a pending migration is answered at once.
- leanmd `-channels`, (4,1,8), 100 steps, steady state ms/step (steps 42-100), control in the same allocation:
  16x8x8 async 866.7 vs 1025.8, sync 872.3 vs 1001.6; 8x4x4 async, nine clean pairs, 106.9-110.4 vs 125.2-182.1
  (about -25%); no-LB ties at both sizes; energy drifts equal to the control's.
- Caveat on attribution: the two modes reached different mappings. In the channel run MetisLB's partition passed the
  cut gate (cross-host share 34 -> 37%) and moved 7162 objects; in the control the gate refused Metis (25 -> 62%) and
  DiffusionLB moved 8793. Part of that was a bug: the communication edge of a put was recorded when the transfer was
  issued, so a put issued from a credit or resolve handler was charged to whatever object ran then (7672/9683 MB
  external/internal against the control's 2791/14454). It is recorded at the put call now. The fifth allocation runs
  both modes with the Metis gate off (`+LBMetisMaxCutRise 1.0`) so they take the same kind of mapping.
- One hang remains, intermittent: 1 of 2 channel async runs at 16x8x8 stopped at step 47 (mid lag window), 1 of 10
  at 8x4x4 at step 24. Every channel arm of the fifth allocation runs with `CHARM_DEBUG_MIGRATE=1` (the per-PE
  channel stall scan) and a 5-minute timeout.

**2026-10-18 08:35, before the fifth allocation: the protocol assumed ordered delivery between two PEs; it has none.**

- Messages between two PEs are not delivered in order here (several PEs of a process poll the fabric and push
  what they receive onto the target PE's queue), so every pair of channel messages that the protocol relied on
  arriving in order was a hole, and each is a way for the hangs seen at steps 24 and 47 to happen after a migration
  wave. Four were found by reading the protocol, all fixed in `ckrdmadevice.{h,C}` (rebuilt 08:34, every test
  binary and `leanmd.ch` recompiled, the latter installed before the chain's allocation starts):
  1. A sender's RESOLVE crossing its own close-ack at the owner: the owner answered the resolve (sender resolved
     again), then took the ack as "closed" and migrated with a resolved sender, whose put then landed in a buffer
     that no longer existed. Now `resolved_during_closing`: an ack that finds a newer resolution closes that one
     too, posted before the ack's count-down so the element's outstanding count never touches zero in between.
  2. Resolved overtaken by its own Closing at the sender: the sender acked the closing, then adopted the stale
     endpoint and put into the old home. Now every resolution has a number (`epoch`, counted by the owner) carried
     by Resolved, Closing and Credit; the sender keeps `closed_epoch` and drops a Resolved of an acked resolution.
  3. A Credit of a closed resolution counted after re-resolving: the owner released after sending its closing, the
     sender re-resolved at the new home with `free=1` and then also took the late credit: two puts into one buffer.
     A credit now counts only when the sender is resolved and the credit names its current resolution.
  4. A sender that resolved while the element stood down for another channel's ack was never closed (the pending
     migration was only noted on channels that had a sender when the migration began). `closingForMigration` now
     notes the pending migration on every owned channel first, so event() closes a late resolver at once.
- Debug prints under `CHARM_DEBUG_MIGRATE`: `[CHAN-RESOLVE]` per answered resolve, `[CHAN-STALE]` for every
  dropped stale message and for a put of an older generation taken after a pup (its bytes travelled with the
  slot); the stall scan prints epoch and closed_epoch.
- The fifth allocation (job 22982782, est. 09:13) runs chantest with LB first, then the leanmd arms as planned.

**2026-10-18 10:25, fifth allocation (job 22982782, gpub[051,084], 08:34 binary = epoch protocol): the hang is gone.**

- chantest 6 of 6 (same process, cross-process, two nodes; 14 migrations each with LB) on the epoch protocol.
- 12 channel LB arms in a row completed (chan4: 1 hang in 10). The debug counters name the live hole: each
  16x8x8 run dropped 783 and 1364 stale credits (hole 3: a credit of a closed resolution arriving after the
  re-resolution, which before the fix made two puts share one buffer); the resolve/ack crossing (hole 1) and the
  overtaken Resolved (hole 2) never occurred in these runs. ~3700 puts per run were taken one generation late
  (their bytes travelled with the slot's pup), as designed.
- Steady-state ms/step (steps 42-100), (4,1,8), 100 steps, control in the same allocation:
  16x8x8 async 860.5 and 852.1 vs 1047.0; 16x8x8 sync 879.8 vs 1003.7; 8x4x4 async seven pairs 109.0-110.4 vs
  138.6-183.8; 8x4x4 no-LB 207.6 (control 202.1 in chan3). `+LBMetisMaxCutRise 1.0` pairs: 8x4x4 107.7 vs 129.2
  (both on Metis mappings: 795 and 961 moves); 16x8x8 863.5 vs 1044.4 (the control's cut rose 108.7% and was
  refused at the 100% ceiling, so it fell to Diffusion again). The "+LBMetisMaxCutRise ... not parsed by the RTS"
  warning is false: MetisLB's constructor reads the flag after that check.
- STILL CONFOUNDED: every channel run took a Metis mapping and (except the 8x4x4 gate-off pair) every control a
  Diffusion one, because the channel run's comm graph differed (16x8x8: 7535 MB external / 9570 internal against
  the control's 2807 / 14591, equal totals). Root cause found in the first arm's log and fixed 10:06: the put's
  comm edge named its destination under the SENDER's object manager with the OWNER's element id, so a
  Cell->Compute put pointed at the Cell of the same id (CkArray::recordSend keys by the destination array's
  manager). `deviceChannelRecordSend` now uses the owner's manager (`owner_mgr`, pup'd with the descriptor).
  Not in the chan5 binary; `n2_chan6.sh` (queued 10:21, after chan5's node was released early at the user's
  word) runs ctl/ch pairs at the default gate and at `+LBMetisMaxCutRise 2.0` at both sizes, then sync, so both
  modes see the same graph and the mapping is equal.

**2026-10-18 12:45, sixth allocation (job 22985435, gpub[040,051]): equal mappings, the open items, and a smaller
channel effect than the earlier numbers said.**

- Two more bugs before the arms ran: (a) the chain's first channel arm died at startup with
  `std::bad_array_new_length`: `CkDeviceChannel`'s copy constructor and assignment never copied `owner_mgr`, so a
  sender's copy taken from the handshake parameter carried an uninitialized group id into the new owner-keyed
  comm-graph call (the pup path carries it, which is why chantest passed). Fixed 12:26, the runner replaced on the
  same allocation (`chan6b.sh`). (b) the benchmark runner's new binary knob was named `BIN`, which the chain
  exports as `leanmd.ch`; renamed `CHBIN`.
- With the owner-keyed graph the two modes see the same graph (16x8x8: external 2726 vs 2791 MB) and take the same
  balancer path. Steady-state ms/step (steps 42-100), (4,1,8), 100 steps, control first:

  | config | default gate (both on Diffusion) | `+LBMetisMaxCutRise 2.0` (both on Metis) |
  | --- | --- | --- |
  | 16x8x8 async | 1014.9 vs 969.5 (-4.5%) | 844.3 vs 853.8 (tie) |
  | 8x4x4 async | 162.3 vs 140.7 (-13%) | 134.3 vs 137.4 (tie) |

  So the 15-25% of chan4/chan5 was the mapping (the channel runs' scrambled graph passed the Metis gate, the
  controls' did not); the channels themselves are worth 4-13% on the Diffusion mappings and nothing on the
  Metis ones, where host time per step is a smaller share. chantest 6/6 again.
- Open items: pic2d at two nodes passes in all three modes on HEAD (none 22.8, async 28.6 with 127 migrations,
  sync 28.9 ms/iteration with 205) once rebuilt from clean: its objects dated 09-18 and were only relinked after
  the 10-17 header changes (the "index compress overflow" was a 2D index holding two packed shorts: ABI skew).
  The benchmark's two-process hang is NOT a group-chare issue: the group version completed in 25 s, the array
  version hung once after 2 sizes and completed twice; intermittent, under a stack hunt in the last minutes.
  The credit's cost is the inter-process converse send itself: `CmiSyncSendAndFree` of the 40-byte credit costs
  12-13 us to another process on the same node AND on another node, and 0 us within a process (alloc 0.00 us in
  every case); the zero-copy path's own sender issue (which also sends one converse message) is 3.7-4.8 us, so
  the send path, not the protocol, is where the 12 us is. The two-node large-message gap is NOT NIC striping:
  `+lci_ndevices 1` leaves both unchanged (4 MB: zc 298-299 us, channel 384-386; 512 KB: 83 vs 97). The gap
  grows with size (+0.7 us at 1 KB, +2.6 at 64 KB, +14 at 512 KB, +85 at 4 MB): the single one-sided put runs
  at ~11 GB/s against the zero-copy path's ~14 GB/s. Candidates: the receiver-driven get of the zero-copy path
  (RDMA read) against the sender's RDMA write from device memory, or chunking/concurrency in the zero-copy
  transfer; test by splitting the put into 2-4 concurrent puts, or by a receiver-side get on notification.
- Cross-process on one node (array version, the two sizes that completed): zc 28.9-29.2 us vs channel 33.1-33.4
  one-way; the channel's receive handler is 12.8 us (the credit send), its issue 0.1 vs 4.7 us.
- 12:46 hang hunt (two-process layout, one node, 2 PEs per process): a stalled run has every PE idle in the
  scheduler (gdb on both ranks: CsdScheduler, deviceRecvBatchPoll, no thread blocked), so an event is lost, not a
  lock taken. Rate: 3 hangs in 9 runs with the four modes interleaved (zc and channel alternating on the SAME two
  device buffers), 0 in 6 runs of either path alone (`-m 10` channel modes, `-m 5` zero-copy modes). Lead for the
  next session: the switch between the paths on one buffer pair -- the zero-copy receive imports the sender's
  source buffer, the channel put imports the receiver's destination buffer, and the import cache or the batched
  receive's completion detection loses one event at the switch. Reproduce with `-m 15` (about 1 in 3), then trace
  the last put/receive of the stalled size with the channel debug prints plus the zc batch prints.
- Allocation 22985435 released at 12:47 with 6 minutes left: every planned arm and the hunt were done.

**2026-10-18 17:25: the credit's 12 us led to a build defect. Both LCI builds on this machine (the stock install and
the vmm-fixed library every multi-process run preloads) were compiled with no -O flag. An -O3 build is swapped in
(`/u/bhosale/lci/build-cuda/lib/liblci.so`, originals kept as `.O0`); the Charm runtime itself was already -O3.
Every communication number above was measured on -O0 LCI. `n2_chan7.sh` (job 22995223) measures again on the
optimized library: a bare converse send (benchmarks/.../sendcost: send call, handler, one-way, in one process /
two processes / two nodes), the persistent put as 1/2/4/8 concurrent one-sided writes (+gpuputchunks), and leanmd
control vs channels under the runtime's own balancing defaults (no flags).**

**2026-10-18 18:25: the 12 us was libfabric's registration cache, not the protocol. Stack samples inside the burst
(job 22995223) showed every small send going through the CXI provider's `cxip_txc_copy_from_hmem`: a
`cuPointerGetAttributes` to learn the buffer's memory type, a `cxip_map` (NIC address mapping, one syscall), the
copy, and a `cxip_unmap` (another syscall), because `FI_MR_CACHE_MONITOR=disabled` had been set in every script
since the memhooks monitor deadlocked with CUDA. With `FI_MR_CACHE_MONITOR=uffd` a bare converse send costs 1.26 us
instead of 10.3 (one-way 4.0 instead of 23.4); `kdreg2` is not available here and silently behaves as disabled.
Every runner now sets uffd. On it (optimized LCI too), one-way latency zc vs channel: two nodes 22.0 vs 19.5 us at
1 KB, 24.8 vs 22.6 at 64 KB, 47.8 vs 57.3 at 512 KB, 263 vs 342 at 4 MB; two processes on one node 28.8 vs 20.1,
29.9 vs 22.6, 46.2 vs 39.3, 194.5 vs 194.8; the credit's send is 0.5 us (other host) and 2.2 us (same host).
The channel now wins everywhere except large messages on two nodes, where the sender's single RDMA write runs
slower than the zero-copy path's receiver-side read (chunking into 2/4/8 concurrent writes changed nothing, so it
is read-vs-write, to be tested with a receiver-pull variant). leanmd under the runtime's balancing defaults, still
on the disabled monitor: 16x8x8 async 851.4 vs 875.4 (channels 2.8% slower), 8x4x4 async 125.5 vs 121.6 (3%
faster), 8x4x4 none 206.5 vs 207.8; `n2_chan8.sh` repeats them on uffd.**

**2026-10-18 20:30, on the fixed message path (job 22996172, gpub[080,091], uffd + optimized LCI).**

- chantest 6 of 6 (posts 1.5-3 us). Microbenchmark one-way, zc vs channel: two nodes 21.5 vs 18.2 (1 KB), 23.9
  vs 21.5 (64 KB), 44.0 vs 54.9 (512 KB), 260 vs 345 (4 MB); the receiver-read variant (`-g`) 20.9 vs 12.4,
  23.5 vs 15.2, 43.7 vs 35.0, 261 vs 247: a read beats the write at every size on this fabric, and beats the
  zero-copy path at every size. Two processes on one node 30.1 vs 19.9, 31.6 vs 24.3, 48.0 vs 40.1, tie at 4 MB.
  The channel's cross-host tier is therefore announce-and-read now (built 20:22, `n2_chan9.sh` validates): the
  sender registers its source, announces once its stream is complete (ANNOUNCE event through the element's
  control entry, carrying the pup'd source), the receiver reads when its buffer is free (queued announces travel
  with a migrating element and are read at the new home), delivers the DATA event locally and sends DONE, which
  is the sender's source-done; no credits on that tier.
- Applications on uffd, runtime defaults: pic2d async 15.9 / none 20.5 ms/iteration (27.7 / 27.8 before: the
  most message-bound one); sph2d async 1.649 / none 2.866; sph3d async 38.0 / none 57.6 (unchanged: bound
  elsewhere). The one-node bare send between two processes no longer hangs (it hung twice with the cache
  disabled).
- leanmd control vs channels under the runtime's defaults: 16x8x8 async 883.4 vs 825.5 (channels 6.6% faster);
  8x4x4 none 206.5 vs 206.5; 8x4x4 async, 13 pairs: control 120-141 (median 127), channels 122-176 (median
  143). The loss is not the transport: in the window after the Metis partition (steps 21-39) the channel runs
  are 1-2% FASTER (99-104 vs 101-102 ms/step); after DiffusionLB's later waves they fall behind, and those waves
  are 1.5-2x larger in the channel runs (100-200 objects per event vs 50-115). Either DiffusionLB moves more
  under channels (its inputs differ: comm bytes recorded at the put, lower host load per object) or each move
  costs more (a Compute's pup carries both full position slots under channels, and the re-resolve round trip
  follows every migration). `n2_chan9.sh` runs a Metis-only pair (`+balancer MetisLB`, one partition, no later
  waves) beside the default pairs to separate the two.
- The cost-table recalibration (lbcalib on uffd) was killed mid-run (rc 143) and produced invalid fits; open.

**2026-10-18 21:10, the announce-and-read protocol: three holes found on leanmd, all fixed, validation queued.**

- chantest 6 of 6 on the protocol (fully rebuilt binaries; the first chantest failures were stale objects after
  the class grew, not the protocol). Benchmark two nodes with the runtime's own announce-and-read: channel 18.0
  vs zc 21.4 us (1 KB), 20.6 vs 23.6 (64 KB), 41.0 vs 44.4 (512 KB), 253 vs 260 (4 MB); two processes on one
  node 19.7 vs 29.0, 23.7 vs 30.3, 39.5 vs 47.2, 196 vs 195. The channel is ahead everywhere except the 4 MB
  copy-engine tie on one node.
- leanmd hung at step 21 (after the first migration wave) three times; the stall scan named each cause:
  1. the receiver treated its buffer as free until the application took the DATA event, so a second announce
     was read over unconsumed data: `busy` is set when the read completes, cleared at release().
  2. a Compute migrated onto its Cell's PE: the force channel went from announce-and-read (cross host) to
     put-and-credit (in process), but the Cell's `sender_pulls` flag from the previous sender made release()
     skip the credit: the flag is cleared on every new resolution and set again only by an announce.
  3. my own migration-cost change: regrowing the position slots to the channel's capacity inside the pup
     called ensureDevice and hapiAddCallback from the unpack, which can run inside the device poll's callback,
     and every stream callback on that PE stopped (senders stuck with one put in flight at 16x8x8). The
     regrowth and the reopen now happen in Compute::ckJustMigrated; an arriving owner registers its channel
     at unpack so resolves and announces that reach the element first are queued, not aborted.
  Also: a resolution answered after a pushed one withholds its free credit while a put of the previous one is
  uncredited (`uncredited` counter), reopen() pushes the new resolution to the last sender, a moved Compute's
  sender copies resolve at arrival, and only live atoms travel in its pup.
- With fixes 1-2 and the regrowth still in the pup: 8x4x4 async channels 141.3 vs control 137.2 (DiffusionLB
  moved 76/67/34 against 150-200 before), 16x8x8 hung (fix 3). `n2_chan10.sh` (job 22998802) validates all
  three: chantest, benchmark, leanmd 16x8x8 and 8x4x4 pairs, no repeat loop.

**2026-10-18 21:50, state at the end of the day.**

- Validated (job 22998802): chantest 6 of 6 on the announce-and-read protocol; benchmark, zc vs channel one-way:
  two nodes 22.3 vs 19.0 us (1 KB), 25.2 vs 21.9 (64 KB), 45.0 vs 41.9 (512 KB), 268 vs 260 (4 MB); two
  processes on one node 27.1 vs 17.5, 28.7 vs 20.7, 45.0 vs 36.6, 193.0 vs 192.7. The channel is ahead at every
  size in every layout. The benchmark's own pull (announce sent at once) is 5 us ahead of the runtime's at small
  sizes (13.8 vs 19.0): the runtime announces from the source stream's callback, which goes through the device
  poll; announcing at once when the stream is already idle (a stream query) is the next cut.
- leanmd with channels: 8x4x4 async completes (141.3 vs control 137.2, one pair, DiffusionLB waves 76/67/34);
  16x8x8 async still hangs after the first migration wave (steps 21-25) after three fixes. The last stall scan
  on it (21:06, before the regrowth left the pup) showed thousands of in-process senders with one put in flight
  whose stream callback never fired, i.e. a device poll that stopped on the receiving PEs. Two of my own
  migration-cost changes ran inside the pup or inside a stream callback (ensureDevice, hapiAddCallback,
  hapiFreeMigratable); both are now in entry methods (Compute::ckJustMigrated, Compute::slotCopyDone). The
  arrival mark (`CkDeviceChannel::arriving()` from the element's pup) replaced the unpack-time registration that
  dangled for parameter copies and aborted the 21:16 arm. `n2_chan11.sh` (queued 21:48) runs the 16x8x8 channel
  arm with the stall scan first, then chantest, then the pairs if it completes.
- Operational: four core files of 50-155 GB (sph2d/sph3d balancing-off aborts at 16:42, run without
  `ulimit -c 0`) and twenty-nine smaller ones filled the home quota at 21:34 and broke a build and a patch;
  deleted, and `run_lmd.sh` sets the ulimit now (run3d.sh and run_channel.sh already did).

**2026-10-18 23:50, bisection of the 16x8x8 hang and the safe state.**

- chan11 (job 22999545): the 16x8x8 channel arm with the stall scan and the receiver-read protocol hung again at
  the first migration wave. Bisection on the same node, 100 steps each, layout (4,1,8), runtime defaults:
  V1 = sender write across nodes + the 20:08 full-slot pup: completes (channels 861.8, control 843.7 ms/step,
  mean42-100; the earlier pair of the day was 825.5 vs 883.4, so the 16x8x8 spread between runs is ~40 ms and
  the two protocols are within it). Write + the live-atom pup with regrowth in ckJustMigrated: hangs.
  V2 = receiver read + the 20:08 pup: hangs without prints and COMPLETES with CHARM_DEBUG_MIGRATE=1, i.e. a
  timing race, not a logic error: a stream callback lost on in-process senders during the wave (the stall scan
  showed senders with one put in flight and no callback). Not found tonight.
- Safe state, installed and in the source: `pull_mode = false` in CkDeviceChannel::onResolved (the read path
  stays in the code; the benchmark's -g variant and chantest exercise it), Compute::pup ships the full slot
  under -channels, Compute::ckJustMigrated only re-resolves the force channels' sender copies; slotCopyDone and
  arrivedSlots removed from Compute.h/.ci/.cc. Rebuilt libck, leanmd, chantest and the channel benchmark from
  clean at 23:40.
- Safe build on the node's last minutes (after_safe.sh): 8x4x4 async channels 138.2 vs control 126.7 ms/step
  (waves 84/66/63 vs 70/103/73; the 9% loss of the earlier 13 pairs again). The 16x8x8 channel arm HUNG at
  step 47 (rc=124, one Diffusion wave, 0 cross-node moves) -- with the sender-write protocol and the full-slot
  pup. So the bisection's conclusion was wrong: the hang is not the read path's. Tally for the write protocol
  at 16x8x8 async: completed 20:08 arm, chan5's debug arms, after_v1 (861.8); hung chan4's repeat (step 47)
  and after_safe (step 47). Both protocols share the race; it sits in the in-process/IPC put path or in the
  migration protocol after Diffusion's first wave, and the CHARM_DEBUG_MIGRATE prints hide it.
- Watchdog (23:48, src/ck-core/ckrdmadevice.C, always on, no switch): a per-PE count of channel events
  (put, release, resolve replies, credits, completions, closings, announces); every 10 s a timer compares it;
  after 20 s without an event the PE prints `[CHAN-WATCH]` (a tally of stuck senders by cause and stuck owners)
  and up to 12 `[CHAN-STALL]` lines, now with the source stream's state (idle/busy/error) and the tier
  (memcpy/ipc/rdma) for a put in flight; again once a minute. A healthy run prints nothing. n2_watch1.sh
  (queued 23:49) runs chantest, then the 16x8x8 channel arm up to 5 times and stops at the first hang: its
  watch lines are the result.
- A mistake of mine: I deleted leanmd.v1 while after_v1.sh was still about to run its two 8x4x4 arms with it;
  both failed with rc=127 (no binary). Rule: never remove or overwrite a binary while a chain that names it is
  alive (the same rule as for chain scripts).
- Open, in order: (1) the step-47 hang at 16x8x8 under either protocol (read the watch lines from n2_watch1;
  candidates from the code: a put completion whose sender token is no longer registered on the PE is dropped
  silently in deviceChannelSrcDoneFn; a sender copy whose first put ran with _currentLocRec NULL is never
  held for migration); (2) announce at once when the source stream is idle
  (benchmark -g is 5 us ahead at 1 KB); (3) leanmd migration cost with channels: live-atom pups and pushed
  resolutions, redone without any stream or device work inside the pup; (4) the 8x4x4 DiffusionLB-wave loss
  under channels (medians 143 vs 127 over 13 pairs earlier today); (5) lbcalib on uffd and the built-in cost
  table; (6) the `[am-send]` instrumentation in reconverse's issueAm (remove or keep). Nothing committed.

**2026-10-19 00:30, ROOT CAUSE of the post-wave hang (both protocols), FIXED.**

- Caught by the always-on watchdog on job 23003213 three times (watch1 at step 25, probe1 at step 25, probe2 at
  step 22, each on the first arm): in the whole job, 2-40 sender copies with `resolved=1 credits=0 in_flight=0
  pending=1(tag 22) epoch=2 closed=0`, all cross-node (node 0 senders, node 1 owners and the reverse), nothing
  stuck on any owner. The owner probe (a stuck sender asks its owner to print its view) showed every owner
  released (`busy=0 have_last=0`), its last event the moved copy's put 21 from the OLD PE, `epoch=2`, and
  `sender=<new pe>/<new token>` -- i.e. the owner had replied to the moved copy's resolve while still holding put
  21 (free=0), then released and sent credit(epoch 2) to the right copy. The refused-credit counter closed it:
  on every stuck PE "credits refused = stuck senders", last one "put 21 of resolution 2 while current 1
  resolved 0 resolving 1": the credit reached the sender BEFORE the reply it belonged to, was refused as not
  resolved, and the reply then arrived with free=0. Two messages from one PE thread to one PE, cross-node,
  delivered out of order (Slingshot adaptive routing or CXI inject vs DMA sends; LCI's backlog keeps order, so
  it is below LCI). Dozens per wave at 16x8x8, so the hang rate was ~2 in 3.
- Fix (ckrdmadevice.C, no header change): a credit whose epoch is later than the copy's current one is kept in
  a per-PE table by sender token (`device_channel_early_credits`) and applied in onResolved when that epoch's
  resolution lands (`credits = 1`); entries retire with the token (destructor, handoff). Any credit for an
  older epoch is still refused. The same reorder cannot hit Resolved vs Closing (closed_epoch drops the stale
  reply) and credit-after-release of the previous epoch is correctly refused.
- Validation: fix1 arm on job 23003213 completed 16x8x8 async channels at 858.4 ms/step (steps 42-100, waves
  7/0/0, no watchdog output); n2_fix1.sh queued: chantest, 2 more 16x8x8 channel arms, the control, the 8x4x4
  pair. Diagnostics kept in the build: [CHAN-WATCH] tally + [CHAN-STALL] examples + [HAPI-QUEUES] dump after
  20 s without a channel event, the owner probe ([CHAN-PROBE]), refused/dropped credit counters. All silent in
  a healthy run.
- Still to do: the read protocol (pull_mode) was switched off for the same hang -- re-test it now; the 8x4x4
  wave loss; the lesson for every other two-message control sequence in the runtime (a reply followed by a
  state change to the same PE): none may assume order across nodes.

**2026-10-19 03:50.** The second fixed arm (fix2, job 23003213, 00:27) hung at step 45 (after the Diffusion wave)
with a NEW signature: one sender on PE 2, `gen 3 owner_pe 61 resolved=1 credits=0 pending(tag 43) seq=42
epoch=3 closed=2`, and the probe at PE 61: no owner registered there. The owner reopened twice (gen 3 at PE
61, epoch 3 to this copy) and then left PE 61 without this copy learning (no Closing of epoch 3: closed=2; no
pushed resolution of gen 4) -- a handoff within process 7 (leaveForHandoff unregisters, arriveFromHandoff
re-registers at the new PE and pushes epoch 4) or a pup migration whose arrival never reopened. Diagnostics
added (probe3, scratch build, installed only after the chain): the probe searches every PE of the owner's
process and the location manager's record of the owner element; each PE's watchdog tick is visible to its
siblings, so a spinning PE is reported by the others. Tally of the early-credit build at 16x8x8: 1 completion
(858.4), 1 hang (step 45). n2_fix1.sh (job 23005325, pending since 00:30) runs the write build; after_stop1.sh
installs probe3 and runs 2 arms if it stops on a hang; after_pull2.sh runs the read build (leanmd.pull3, with
probe3) if it completes.
- Found 03:55 while reading reopen(): the push of the new resolution to the last sender never happens --
  `push = sender_known && sender_closed` is computed after both were cleared, and packForMigration (which runs
  BEFORE the element's pup, cklocation.C ~4767) clears sender_known anyway, so the pupped value is false. Every
  moved owner therefore costs each of its senders a resolve round trip before the next put (part of the
  post-wave loss). Fix later, separately: packForMigration keeps sender_known/pe/token (the sender was closed
  and acked before the pack), reopen computes push before clearing. Not changed during the hang hunt.

**2026-10-19 07:13.** n2_fix1 (job 23005325, granted 07:04): chantest 6/6; 16x8x8 async channels 854.7 then a hang
at step 24 (three senders on PE 59, owners' buffers opened on PE 55, `gen 2 epoch 2 closed=1 seq=22 pending 23`,
tier ipc; no owner registered at PE 55). probe3 build then completed twice (849.9, 838.5). Tally of the
early-credit build at 16x8x8: 4 completions (838-858 ms/step), 2 hangs (steps 24 and 45). The remaining hang:
the owner was closed-and-acked at epoch 1 (pup-style prep), then moved by a HANDOFF within its process
(generation 2, same buffer, dst.pe still the opening PE, pushed resolution epoch 2), the sender's next put
landed, and the owner's credit for it never reached the sender (refused/dropped counters on the sender's PE
show nothing of it). Next build (probe4, installed 07:13) prints a credit that lands on a PE it was not
addressed to and the counters on every quiet PE; after_probe4.sh runs up to 7 arms and stops at the first hang.

**2026-10-19 07:26.** probe4 (misdelivery print, counters on every quiet PE): 7 of 7 completions on job 23005325
(843.3-857.7 ms/step). Tally of the early-credit build at 16x8x8 async channels: 11 completions (838-858,
mean ~851), 2 hangs (00:27 step 45; 07:06 step 24), i.e. about 1 in 6; the two hangs had the same signature
(owner handed off within its process after a closed-and-acked resolution; the sender's next put landed and the
owner's credit for it never reached the sender; the sender had already issued the next put, so the owner had
integrated and released). No misdelivery or dropped credit for the stuck tokens has been seen yet (the probe4
build never hung). Next: n2_fix2.sh (queued 07:26) measures channels vs control on this build (16x8x8 x2 pairs,
8x4x4 x2 pairs) and stops on a hang with the full diagnostics; the read build gets one arm on the last minutes
of 23005325 (after_pull3.sh).
- 07:27: the read protocol (leanmd.pull3: cross-host tier announces and reads, on the early-credit fix) completed
  its first 16x8x8 arm at 853.7 ms/step (write build: 838-858). Its 2026-10-18 hangs were most likely the same
  reorder family. One arm only; not yet the default. probe5 build installed 07:31 (counters: early credits kept,
  resolutions/closings that found no copy, reply misdelivery print); n2_fix2b.sh runs the channels-vs-control
  chain on the pending lmd_fix2 allocation (job 23011983).

**2026-10-19 09:20, live-atom pup under -channels (user: "fix it").** The Compute's position slot is a channel
buffer of fixed capacity (2*maxCellParts+64) that its sender knows, so the 2026-10-18 pup shipped the whole slot
(twice: positions and forces, plus the energy partials). Now: (1) protocol -- the closing ack carries the
sender's last put (seq, bytes); the owner compares with the last put it took (`taken_seq`) and records
`landed_unseen` (a put whose event is still queued in the SDAG buffer at pack time: common, since posArrived is
an SDAG when and the Cell runs one put ahead); a blind ack (the copy had left) records "unknown"; the owner is
not free until that put is taken. Public `unseenBytes()` (header change: leanmd, chantest, channel rebuilt
from clean). (2) Compute::pup -- channels pupped first; d_pos ships max(nPart, unseen) entries (the whole slot
only when unknown); d_force and d_energyPartial are not shipped (regenerated every step); posCh.arriving() at
unpack. (3) Compute::ckJustMigrated -- allocates the three buffers at capacity from the pool, copies the live
positions on the chare's stream, reopens the channel on the new buffer, resolves the force copies, and frees the
arrived buffers from slotCopyDone (a stream callback entry). Validation: n2_pup1.sh on the pending lmd_fix2
allocation (job 23011983): chantest, 16x8x8 async channels x2 (must complete, energy drift as before), then the
no-LB pairs the user asked about (16x8x8 none ctl/ch x2, 8x4x4 none ctl/ch x2).

**2026-10-19 09:58, jacobi3d without LB (user: "test a simple jacobi3d application without LB").** The existing
examples/charm++/cuda/gpudirect/jacobi3d (modes: regular, -d zero-copy, -s the old persistent API) gained `-k`
device channels: per neighbour face, two owned channels (one per iteration parity: a neighbour puts iteration
i+1 while this block still reads i), copies handed out at init (initRecvCh), the face put behind the packing
kernels on comm_stream with the iteration as the event's reference number (SDAG `when recvGhostCh[my_iter]`),
unpack from the channel buffer, release of this parity's channels after the update. The example's post entry
used a field that no longer exists (cuda_stream -> hapi_stream; the transfer stays held behind the previous
unpack on that stream). Runner run_j3d.sh (leanmd's split and pinning; 32 blocks = 4 per GPU on 8 A40s);
after_j3d1.sh runs zc/ch pairs x2 at grids 512^3 (128 KB faces), 1024^3 (512 KB), 2048^3 (2 MB), 200 iterations
after 10 warm-up, on the node n2_pup1.sh keeps (job 23011983, est. 10:25). Expectation from the message-path
benchmark with the sender-write protocol: channel ahead at 128 KB, behind at 512 KB and 2 MB across nodes.
- 10:25 jacobi3d corrections (user review): (1) the zero-copy mode had NO source callback (CkDeviceBuffer(ptr,
  count, stream)) and one receive buffer per face landed at message arrival -> both modes now pass a source
  callback whose reference number is the iteration and wait for them before packing (zero-copy: the previous
  iteration's; channels: the one before last), and receive by iteration parity; (2) an end-of-run checksum
  (sum of every block's interior, reduced) printed by Main, with a host-staged reference arm in the chain;
  (3) every device allocation now comes from the runtime pool (hapiMalloc/hapiFree) instead of raw cudaMalloc,
  so the arena is registered and exported once. Build 10:25:18, before the allocation (est. 10:25:30).

**2026-10-19 10:45, n2_pup1 (job 23011983) results.**
- The live-atom pup hung 2 of 2 at step 22-23 under LB, and the probe said why at once: the stuck senders' owners
  were "registered on PE 4 of this process, not here", gen 3, `sender_known=0`, taken_seq 22, released. A move
  within the process is a pointer HANDOFF (same GPU: nothing packed, no unpack): the runtime's arriveFromHandoff
  had already re-registered the channel (gen 2) and pushed epoch 2 to the sender; my ckJustMigrated then ran as
  if after a pup -- allocated new slots, reopened (gen 3, sender_known cleared, push dead) and freed the buffer
  the sender was still writing. The sender kept epoch 2, its put 22 landed in the freed buffer, the owner took
  and released it with no sender to credit. Fix (10:45 build, installed): `arrivedByPup`, set by the unpack;
  ckJustMigrated regrows/reopens only then, and after a handoff only re-resolves the force copies.
  n2_pup2.sh (queued 10:45): 16x8x8 async ch x3, LB control, 8x4x4 async pair.
- Channels vs control WITHOUT LB (runtime defaults, (4,1,8), mean42-100): 16x8x8 none: control 1505.5 / 1506.5,
  channels 1520.0 / 1520.3 ms/step (+1%); 8x4x4 none: 206.4 / 206.6 vs 206.3 / 206.4 (tie). The 16x8x8 channel
  arms show rc=124 only because 100 x 1.52 s crossed the 150 s timeout at exit (both printed their exit lines
  and the full 100 steps). Per-PE [am-send] at exit: on the channel arm, PE threads 5-7 of rank 0 posted at
  14-15 us with ~44 LCI retries per send (680k retries / 15.5k sends) against 2 us and ~2 retries elsewhere --
  the credit/control traffic backs up on some PEs; not on the critical path at +1%, but it is where a per-step
  cost would come from.
- jacobi3d (pool build): host-staged reference 512^3 831.1 us/iteration, checksum 8.762191026712e+06; the
  zc/ch pairs follow (after_j3d1.sh).
- 11:02 jacobi3d on job 23011983: the host-staged mode ran (831 us/iteration at 512^3, checksum 8.762191026712e+06);
  the zero-copy AND the channel mode (both with my source-callback waits and parity receive buffers) stalled after
  init and never finished an iteration. Common to both: the `when sendDone[my_iter-1]()` wait on the source
  callbacks (an empty message with the iteration as reference number). A debug build prints block 0's progress per
  iteration (-v); the one-node request for it was refused (one job per user in this QOS), so after_j3d_dbg.sh
  runs reg/zc/ch at 256^3 x 3 iterations on the pup2 allocation after its leanmd arms (est. 11:59).
  The 1% without LB at 16x8x8: both none arms' per-thread [am-send] show that channels add ~15.5k transport
  messages per PE (the DATA note and the credit of every cross-process put, same-node pairs included), where
  zero-copy same-node sends go through +gpushm's slot flags with no transport message at all; on 8 of 64 PE
  threads (rank 0 and rank 4, threads 4-7) those posts took 14-32 us with 40-106 LCI retries each. The rebuilt
  libreconverse (10:56) counts retries by reason (lock/nopacket/nomem/backlog) and the longest post; the channel
  stats (CHARM_ZC_STATS) now also give put() host time by tier, puts queued, events; after_pup2.sh runs one
  channel and one control none arm with both.

**2026-10-19 14:17, same-host credits and notes through shared memory (user: "go ahead").** ckrdmadevice.C: one
POSIX shm segment per host and job ("charm-chan-<host>-<job>", zero pages = empty state, no rendezvous, unlinked
at exit), one area per PE (8192 credit slots {channel, epoch, count} + a ring of 8192 arrival notes {ready
ticket, event, the arrival callback}). Owner: replyResolve to a same-host other-process sender allocates a slot
(CAS on channel, count 0, epoch), the Resolved message carries (area, slot); release() increments the count
instead of sending a credit; the slot is retired at the next reply, at the closing ack, at pack, close and
destruction. Sender: onResolved keeps (area, slot); put() reads the word when it has no credit; a queued put
joins the PE's waiter list, drained by a scheduler-loop poll; reads check channel and epoch on both sides of
the count (a freed or reused slot is ignored). Notes: a same-host other-process put's completion writes a ring
entry instead of sending the CkDataMsg; the owner PE's poll turns entries into the local event (the callback
travels in the entry, so a handed-off or moved owner still gets it through the array manager); a full ring
falls back to the message. Cross-host and in-process pairs unchanged. Header change: leanmd, chantest, channel
and jacobi3d rebuilt from clean 14:17; the previous build kept as leanmd.msgcredit for the A/B.

**2026-10-19 14:52, put with immediate data across hosts (user: "JUST BUILD THE PUT WITH IMM").** LCI
(build-cuda-rel, installed over build-cuda/lib/liblci.so, backup .pre-imm): the write with immediate packs the
sender's rank above the 32 immediate bits as sends do; a remote-write completion now carries rank and length to
progress_remote_write, which passes size to the handler. reconverse: comm_backend::issueRputSignal (post_put
with .tag and a second registered remote completion), setSignalHandler; Converse CmiIssueRputSignal(info, tag)
and CmiSetRputSignalHandler. Channel runtime: a per-process table of 65536 signal slots {owner PE, arrival
callback, channel, generation, sender, last seq}, allocated by the owner in replyResolve for a cross-host sender
and carried in the Resolved reply with the owner process's registered 1 MB header region (16 B per slot: seq,
bytes, tag); the sender's issue() asks the transfer for the signal; persistentRputReadyFn writes the header by
an inject-sized put from the PE's registered scratch, then the data write with the slot as its immediate; the
network thread's handler pushes the completion to the owner's PE, whose bridge waits for the header's sequence
to advance (the two writes are unordered on this fabric), builds the event and sends it to the element. Slots
retire with the resolution (closing ack, pack, close, destructor; at a new reply only if the previous
resolution was closed). Fallback to the note message: tag >= 65536, size >= 4 GB, table exhausted, same host.
Builds (header change, all from clean): leanmd.imm (NOT installed as leanmd), chantest/channel/jacobi3d (the
chain's chantest exercises the signal across nodes first). after_pup2.sh: imm, ctl, msg, shmwrite arms.

**2026-10-19 15:08, job 23014814 (granted 14:59).** chantest on the IMMEDIATE build: 6 of 6 PASS, including the
rdma modes with migration (the signal path across nodes works). leanmd on the shared-memory + write build
(installed leanmd), runtime defaults, (4,1,8), mean42-100: 16x8x8 async channels 833.9 / 853.9 / 851.2 ms/step
(3 of 3 complete; live-atom pup with the handoff guard, under Metis + Diffusion waves), control 854.9; 8x4x4
async channels 143.4 vs control 142.3 -- the 9% wave loss of yesterday is gone (the full-slot pup and the
message credits were it). The four-way no-LB comparison (imm, ctl, msg, shmwrite) and the jacobi3d traces follow.

**2026-10-19 15:22.** Four-way no-LB at 16x8x8 (job 23014814): imm 1518.4, ctl 1508.0, msg 1520.4, shmwrite 1520.0
ms/step -- all three channel builds equal because NEITHER new path had engaged: CkRdmaDeviceChannelInit runs before
init.C's CmiInitCPUTopology, so "PEs on this host" equalled "PEs in this process" and the shared-memory segment
and the signal table were never set up; this morning's "shared-memory" leanmd results were message-credit results.
Fixed 15:14: both are set up at first use (deviceChannelLateInit), each process prints what it enabled and the
first credit/note/signal of each kind. Retry reasons on the slow threads: lock (LCI's endpoint lock shared by the
8 PE threads of a process) with some nomem (network queue full) -- a separate lever: more LCI devices per process.
jacobi3d traces: both device modes stop at the wait for the previous iteration's source callbacks -- a
parameterless entry reached through a callback loses the reference number in the SDAG closure; fixed 15:22 with
sendDone(int iter) issued from a C-function callback with setRef. The user's correction stands: the live
allocation is the scarce thing; its remaining minutes now run the immediate build's leanmd LB arms and jacobi3d.

**2026-10-19 15:36, results on job 23014814 (granted 14:59, expired 15:29).**
- jacobi3d (immediate build; corrected source callbacks and parity receives; checksums bit-identical to the
  host-staged reference 8.762191026712e+06 at 512^3): 32 blocks on 8 A40s, 200 iterations after 10 warm-up,
  average iteration: 512^3 (128 KB faces) zero-copy 1291.0 us vs channels 702.5 us (-46%); 1024^3 (512 KB)
  5817.8 vs 5477.9 (-6%); 2048^3 (2 MB) 42989.6 vs 42405.5 (-1.4%). The cross-host tier in these runs was the
  READ (the pull_mode line had been left on from the read-retest builds; found 15:26), same-host pairs through
  shared memory.
- leanmd, immediate build (late init; cross-host read as above), LB, (4,1,8): 16x8x8 async channels 847.0
  ms/step (migr 11/0/0, complete); control 854.9 (earlier arm) -- the 8x4x4 pair and the 16x8x8 control arm
  were cut short by the node's end; the 8x4x4 channel arm's first-use lines showed shared-memory credits and
  notes engaged on all 64 PEs.
- The immediate path itself: with the read on it never engaged; with the write restored (15:26) the first puts
  with immediate data were issued and the run aborted at once: "remoteDeregHandler: neither source nor
  destination node" after "CmiAlloc: size <= 0" -- the header transfer went through CmiIssueRput with an
  NcpyOperationInfo whose completion reached the zero-copy API's deregistration path. 15:35: the header now goes
  by CmiRdmaPutRaw (reconverse: comm_backend::issueRput with a no-op completion, no operation record, no ack);
  the data write with immediate unchanged. Built into leanmd.imm, chantest, channel, jacobi3d; UNTESTED.
- Queued (one-hour request, lmd_imm1): chantest; leanmd.imm under LB 16x8x8 x2 + control, 8x4x4 pair; no-LB
  16x8x8 channel + control; jacobi3d zc vs ch at 512/1024/2048. Mistakes of the day, recorded: deferring
  requested runs to a later allocation (hours of queue) and 30-minute requests; a kill pattern that matched my
  own shell (three times); the read protocol left on in the source by a retest build.

**2026-10-19 18:05, job 23020057 (granted 17:56, one hour).** The immediate path with the raw-put header: chantest
rdma modes FAIL (segfault/abort), leanmd and jacobi3d channel arms abort after the first signalled write. With the
completion poll instrumented (every completion carrying remote data printed): NO FI_REMOTE_WRITE completion ever
reaches the target; every remote-data completion is an ordinary message to the active-message handler (index 9);
the signal handler (10) never fires. LCI's endpoint asks for FI_RMA | FI_MSG only; adding FI_RMA_EVENT changed
nothing. So CXI through this libfabric delivers no write-with-immediate completion to LCI as configured (LCI
itself defaults its rendezvous to "write + FIN message", not "writeimm"); making it work means binding LCI's
memory regions for remote events in the backend, not doable in this allocation. The sender's crash after its
first signalled write is unexplained (no stack). Immediate notes DISABLED by a constant in issue()
(CHAN_IMMEDIATE_NOTES); the whole machinery stays in place. The node's remaining 50 minutes run the comparison
on the write protocol with shared-memory same-host credits/notes (after_now2.sh: chantest rdma, leanmd LB
16x8x8 ch/ctl + 8x4x4 pair, no-LB 16x8x8 ch/ctl, jacobi3d zc/ch x3).

**2026-10-19 18:16, results on job 23020057 (after_now2.sh, 18:05-18:15): write protocol, shared-memory same-host
credits and notes, message notes across hosts (immediate path disabled).** chantest rdma plain/LB PASS.
- leanmd (4,1,8), 16x8x8 async LB: channels 848.6 vs control 843.8 ms/step (migrations 0/0/0 vs 0/0/8; both
  complete); 8x4x4 async LB: 138.1 vs 125.8 (migrations 116/90/145 vs 38/132/69 -- different LB histories, the
  wave loss seen 18:08 is a migration-count difference, not a channel cost); 16x8x8 no LB: 1519.6 vs 1509.1
  (+0.7%, the small no-LB loss persists). No CHAN-WATCH, no faults.
- jacobi3d, 32 blocks on 8 A40s, 200 iterations, zero-copy vs channels (checksums identical): 512^3 1162.1 vs
  850.3 us (-27%); 1024^3 5796.8 vs 5481.3 (-5%); 2048^3 42868.6 vs 42413.1 (-1%). Against the 15:36 numbers
  (read across hosts): the write makes zero-copy faster at 512^3 (1291 -> 1162) and channels slower (702 -> 850),
  so at 128 KB faces the cross-host READ beat the write by ~150 us per iteration in channel mode.
- LCI rebuilt 18:10 with FI_RMA_EVENT in the hints, the granted caps printed, and every memory region registered
  with FI_RMA_EVENT when granted (fi_cxi(7): optimized MRs generate no remote events; FI_CXI_OPTIMIZED_MRS=false
  makes all MRs standard). The arms from 18:11 on ran on that library without change. Imm-ON variants
  (CHAN_IMMEDIATE_NOTES=true) built as chantest.immon, leanmd.immon, jacobi3d.immon (18:10); libck restored to
  imm off. after_imm2.sh (18:15, same node): chantest rdma on chantest.immon, then with FI_CXI_OPTIMIZED_MRS=false
  if no immediate completion arrived, then leanmd.immon 16x8x8 async/none + 8x4x4 async and jacobi3d.immon
  ch/zc x3 with whichever setting delivered.

**2026-10-19 18:27, the immediate path's crash found (job 23020057).** after_imm2.sh (18:15): chantest rdma on
chantest.immon FAIL at the first signalled put, default and FI_CXI_OPTIMIZED_MRS=false alike; leanmd.immon and
jacobi3d.immon abort the same way. LCI's granted caps: rma_event=1, remote_cq_data=0. gdb on the live node (the
script and output must live on /u, not the login node's /tmp): task 0 dies in remoteDeregHandler ->
CmiDeregisterMem, and a breakpoint on CmiInvokeRemoteDeregAckHandler never fires on either rank -- nobody sent a
dereg message. The message that reached the handler was the immediate-data note itself: init.C registered the
probe bridge but never device_channel_signal_bridge, so device_channel_signal_handler stayed 0 and
deviceChannelSignalFromNet pushed every note with handler index 0, which is whatever Converse registered first
(here the dereg handler; "CmiAlloc: size <= 0" in other runs was the same message through another handler).
The 18:05 verdict ("CXI delivers no write-with-immediate completion") was wrong about the cause: LCI itself turns
support_putimm OFF for the cxi provider (backend_ofi.cpp) and emulates a put with remote completion as the write
followed by a zero-length message carrying the immediate (progress_write: post_sends(imm_data_rank, nullptr, 0,
imm_data)); the target gets it as a message completion that LCI routes to the registered rcomp -- no
FI_RMA_EVENT or FI_REMOTE_CQ_DATA needed, so the LCI MR change of 18:10 was unnecessary (harmless; it stays).
Fix: init.C registers the signal bridge. Rebuilt imm-on (CHAN_IMMEDIATE_NOTES=true still in the source),
relinked chantest.immon, leanmd.immon, jacobi3d.immon 18:25; after_imm3.sh running: chantest rdma, leanmd.immon
16x8x8 async/none, jacobi3d.immon ch x3, leanmd.immon 8x4x4 async.

**2026-10-19 18:37, immediate path after the handler fix (after_imm3.sh, 18:26-18:30).** chantest rdma plain/LB
PASS on chantest.immon, both ranks "first put received as immediate data ... waited 0.0 us for the header" (one PE
per process). With several PEs per process it fails: leanmd.immon (4,1,8) 16x8x8 async and none: a segfault on
one rank at step 1-2 (no abort text); jacobi3d.immon (4 PEs/process) 512^3: CmiAbort "Cannot receive a message
for an array without a local branch" (CkSendMsgArray: the callback's array id has no branch, i.e. the signal
entry's stored callback is not a live one); leanmd.immon 8x4x4 async ran 62 steps at 113.6 ms/step (message
notes 138.1, control 125.8) through two LB rounds, then stalled after the migrations with every sender's
watchdog reporting "credits refused by a copy here ... credits 0" -- the LB path of the immediate notes is a
second, separate bug. Every immediate note also printed "CmiAlloc: size <= 0" (4-12k per rank): LCI's
progress_recv allocates the zero-length message's buffer through the user allocator (reconverse's CmiAlloc), which
refuses 0 bytes; patched LCI to allocate only when the message has bytes (installed 18:35, previous copy
liblci.so.rmaevent). Under gdb neither multi-PE failure reproduces (leanmd none ran 81 steps clean before the
timeout, jacobi3d exited normally): timing races. Note for gdb on the compute nodes: scripts and outputs must be
on /u (the login node's /tmp is not mounted there), and breakpoints on library symbols need
"set breakpoint pending on" in batch mode (without it the 18:24 breakpoint on CmiInvokeRemoteDeregAckHandler
was never set; the conclusion held because the handler fix made chantest pass). after_imm5.sh (18:35): leanmd
none 16x8x8, jacobi3d ch x3, leanmd async 16x8x8 on the LCI without the zero-byte allocation.

**2026-10-19 18:41, immediate notes VALIDATED (after_imm5.sh, job 23020057, LCI without the zero-byte
allocation).** The multi-PE failures were that allocation: LCI handed the zero-length immediate message to the
Converse allocator, which refused 0 bytes and returned NULL; with the patch every arm is clean (zeroalloc=0).
Same build/LCI otherwise as the 18:05-18:15 write-protocol comparison (message notes), so the three columns compare:

| arm | immediate notes | message notes | control / zero-copy |
|---|---|---|---|
| leanmd (4,1,8) 16x8x8 no LB | 1518.1 | 1519.6 | 1509.1 ms/step |
| leanmd 16x8x8 async LB | 846.8 (migr 19/0/0) | 848.6 (0/0/0) | 843.8 (0/0/8) |
| jacobi3d 512^3 (128 KB faces) | 719.9 | 850.3 | 1162.1 us/iter |
| jacobi3d 1024^3 (512 KB) | 5480.6 | 5481.3 | 5796.8 |
| jacobi3d 2048^3 (2 MB) | 42562 | 42413 | 42869 |

Checksums identical in every jacobi3d mode. The immediate note removes a message per put: visible only where the
puts are small and frequent (jacobi3d 512^3: -15% vs the message note, -38% vs zero-copy, the 15:36 read-protocol
channel number 702.5 nearly matched); at 512 KB+ and in leanmd the note is not on the critical path. leanmd's
0.6% no-LB loss against the control stands. Source: CHAN_IMMEDIATE_NOTES = true is now the default (comment
rewritten), libck rebuilt 18:41; the standard binaries (leanmd, jacobi3d, chantest, channel) are relinked after
after_imm6.sh (which still uses jacobi3d = message notes for the band). after_imm6.sh: leanmd.immon 8x4x4 async
(the 18:29 post-migration stall) + the 2-repeat band jacobi3d 512/1024 message vs immediate.

**2026-10-19 18:48, band + the hardware test; node released.** 2-repeat band (after_imm4.sh), jacobi3d channels:
512^3 message notes 712.2 / 717.4 us, immediate 703.7 / 715.4; 1024^3 5481.4 / 5485.2 vs 5480.9 / 5546.4. The
18:14 message-note 850.3 was a single-run outlier; the two note kinds tie, and channels' gain over zero-copy at
128 KB faces is ~38%. leanmd.immon 8x4x4 async stalled again after LB round 2 (62 steps at 113.2 ms/step,
migrations 53/99; no stuck stream, no missing owner, only credit tallies): an element waits for a DATA event that
never arrived -- the immediate note is dropped at the old owner process once its slot is retired at pack
(deviceChannelSignalFromNet returns on !used, uncounted), and retired slots are reused LIFO so a late header can
land in a slot already given to another channel; which of the two needs a counter + one 60 s rerun (next
allocation). Hardware put-with-imm on cxi tested: support_putimm forced on -> fi_writemsg(FI_REMOTE_CQ_DATA)
"Flags not supported"; FI_REMOTE_CQ_DATA in hints caps -> ENODATA. So CXI 2.3.1 has no RMA write with immediate;
LCI's emulation (write + zero-length message) stays, the LCI source says why. Installed liblci.so = emulation +
zero-length allocation fix (copy liblci.so.emul). Standard binaries leanmd, jacobi3d, chantest, channel relinked
18:47 against libck with CHAN_IMMEDIATE_NOTES = true (same code as the .immon variants). Allocation 23020057
released 18:47 (mine; arms done).

**2026-10-19 18:58, the stall's cause (from the code; counters queued to confirm) and the fix (built, unvalidated).**
Mistake first: I released job 23020057 at 18:47 with this stall unexplained; the user had to wait for a new
node (23024413, requested 18:52). Cause: the owner retires its signal slot in packForMigration() (and in
onCloseAck() at the ack), and deviceChannelSignalFromNet() drops a note whose slot is not in use -- silently. A
put issued before the sender saw Closing has its data write, header and immediate message in flight; the data
lands (the closing ack's last_seq > taken_seq makes landed_unseen carry the bytes in the pup), but the note
reaches the old process after the retire and is dropped, so the element never gets the DATA event for that put
and waits forever. A message note never has this problem: it carries the owner's array callback and the array
runtime forwards it to the element's new home, where take() handles the stale generation. Second hazard of the
same retire: the free list is LIFO, so the slot is immediately regranted and a late header for the old channel
lands in another channel's slot. Diagnostics added (leanmd.diag): per-process counters of notes dropped at a
retired slot / out of range and of headers whose seq skipped or went backwards, in the watchdog tally.
Fix (ckrdmadevice.h/.C + init.C, built as leanmd.fix / chantest.fix / jacobi3d.fix): the slot is never freed at
pack or at the ack directly. pack (and a reply before the ack) stashes it (departed_signal_slot/pe, pupped with
the element); the closing ack, wherever it arrives, sends a slot-release message to the granting PE with the
sender's acked last_seq; there chanSignalRelease frees the slot if the bridge already took that seq, otherwise
records final_seq and the bridge frees it when it takes that put. A blind ack (UINT64_MAX) frees at once. A
second move before the first ack releases the older stash blind (counted). Tally line: "signal slots released:
N at the ack, N deferred to the acked put, N of those freed by it". after_stall1.sh (diag arm) runs when the node
starts; after_stall2.sh (chantest.fix rdma, leanmd.fix 8x4x4 async x2 + 16x8x8 async) follows it.

**2026-10-19 23:07, job 23024413 (granted 23:00): the dropped-note theory was WRONG; the stall is a resolution race.**
after_stall1.sh (leanmd.diag, 8x4x4 async): stalled at step 82 (migr 62/46/63) with ZERO immediate notes dropped
at a retired slot and zero headers going backwards (the "skipped a put" count only shows a fresh slot taking the
sender's running sequence). after_stall2.sh (leanmd.fix: slot kept until the acked last put is taken): chantest
rdma PASS; 8x4x4 async stalled at step 83 (migr 20/206/113), 16x8x8 async 853.8 ms/step complete, 8x4x4 async
repeat 121.8 ms/step complete (100 steps). The fix is sound (109 slots released at the ack, 1 deferred and freed by
the bridge) but not the cause. The stalled run has exactly ONE stuck channel (20000000072): sender copy on pe 10
"resolved=1 epoch=3 closed=2 credits=0 pending=1 (tag 83) seq=82, tier ipc"; the owner on pe 2 (same host, other
process) "epoch=4 gen 3 taken_seq=82 busy=0 have_last=0 sender=10/a00000000a4 sender_closed=0 closing_sent=0":
the owner took and released put 82 under resolution 4 (shared-memory credit slot of epoch 4) while the sender
still holds resolution 3, so the credit is invisible to it (chanShmReadCredit returns -1 for another epoch) and
its put 83 waits forever. The sender never saw Resolved(4) nor Closing(3) (closed_epoch 2), and pe 10 counted no
dropped resolution. Not immediate-note specific in mechanism (the stuck pair is an IPC pair); the immediate notes
only change the timing. after_stall3.sh: the same arm with CHARM_DEBUG_MIGRATE=1 (every resolution, closing and
ack printed), up to 3 tries, prints the stuck channel's trace from every rank.

**2026-10-19 23:12, ROOT CAUSE of the 8x4x4 stall, from the protocol trace (after_stall3.sh, 3rd try stalled at
step 64).** Channel 2a0000000091, owner on PE 42, sender on PE 50 (same host, other process; IPC tier, shared-
memory credits). Trace: resolution 1 -> closing sent -> ack -> HANDOFF to PE 41 (owner moved inside its process,
gen 2) -> PE 41 answers the SAME sender twice in a row: "resolution 2 to pe 50" then "resolution 3 to pe 50" (one
from the waiting list, one pushed). The sender ends at "epoch=2 resolved=1 closed=1 credits=0 pending=1 (tag
63)": the two replies arrived in the wrong order (messages between PEs are not ordered) and onResolved applied
resolution 2 after 3, overwriting the newer state. The owner (epoch 3) took put 62 and released its credit
under resolution 3's shared-memory slot; the sender reads the slot for resolution 2, gets -1, and waits forever.
Nothing to do with immediate notes (they only changed the timing), nor with dropped notes or slot reuse (the
18:58 fix stays: it is correct, and its counters confirmed 0 drops). Fix (leanmd.fix2, 23:11): onResolved ignores a
resolution whose epoch is <= the one the copy holds (counted: "resolutions ignored here as older than the one
held"); the skipped-a-put counter now excludes a fresh slot. after_stall4.sh: 8x4x4 async x3 + 16x8x8 async.

**2026-10-19 23:19, the fix validated (leanmd.fix2 = slot lifetime + older resolutions ignored), job 23024413.**
8x4x4 async LB: 7 of 7 runs complete (121.1, 125.6, 136.8, 135.6, 127.0, 114.6, 134.1 ms/step; migrations per
round up to 192) where 5 of 8 runs stalled before; 16x8x8 async 838.1; 16x8x8 no-LB 1520.5 (control 1509.1);
chantest memcpy/ipc/rdma plain+LB all PASS; jacobi3d channels 512^3 696.9 us, 1024^3 5589.1, checksums identical.
The guard's counter was not seen firing: it is printed only by the quiet-PE watchdog tally, which a healthy run
never reaches -- after_stall7.sh reruns 8x4x4 with CHARM_DEBUG_MIGRATE=1 (the "[CHAN-STALE] ...; ignored" print)
up to 4 times to see it take effect. Standard binaries leanmd, jacobi3d, chantest (23:13) and the channel
benchmark (rebuilt from clean) are this build.

**2026-10-19 23:25, CONFIRMED.** The guard's counter is only printed by the quiet-PE tally, so 11 clean runs
showed "ignored=0"; with the print made immediate (leanmd.fix3, prints only), run 3 of after_stall8.sh showed
"[CHAN-STALE 59] channel 31000000008b token 3b00000000be: resolution 2 after 3 (resolved 1, closed 1); ignored"
and completed (117.6 ms/step): the reorder happened and the guard held. 15 of 15 8x4x4 async runs complete on the
guarded build (114-153 ms/step) against 5 stalls in 8 before. No resolution was ever dropped for lack of a copy.

**2026-10-19 23:29, control bands under async LB (after_ctl1.sh, CHANNELS=0, same build):** 8x4x4 136.8 / 124.5 /
128.4 ms/step (migrations up to 175 per round) against the 15 channel runs at 114-153 (median ~127): channels
tie the control under LB at 8x4x4; the 18:08 "138 vs 126" was one run each. 16x8x8: control 848.6 against
channels 838.1 / 846.8 / 853.8: tie. after_ctl2.sh: 16x8x8 no-LB control x2 against channels' 1518-1521.

**2026-10-19 23:34, no-LB control band (after_ctl2.sh):** 1505.1 1506.4 ms/step against channels' 1518.1-1520.5 (four runs) and the 18:14 control 1509.1. Node 23024413 held with the minutes left; nothing queued.

**2026-10-20 10:40, the move window: five changes built, unvalidated (job 23047794 queued for one node).** From the
discussion of live migration (GPU_INITIATED_COMM_IDEAS.md, idea 2) the parts that do not need GPU-initiated transport:
1. Senders told the new location at emigrate: CkLocRec::recentSenders (the last 8 PEs whose messages the element took,
   noted in CkArray::deliverToElement) get an updateLocation when it leaves (CkLocMgr::informSenders, both the packed
   path and the handoff), so their next message goes to the new home instead of the forward and the repair.
2. The channel protocol's two waits measured, not changed: the sender's hold (closing received -> resolution applied)
   and the owner's close (closing sent -> ack), in the tally below. The redirect-before-arrival of the discussion is not
   implementable as designed: the new home's channel buffer is only known once the element unpacks there.
4. Four migration streams per device, round robin (DeviceManager::migration_streams): a pack's copies wait on that
   element's event, and a stream wait holds everything queued behind it, so one slow element no longer holds every
   later pack of the device.
5. PUP::DeviceLife::Stable: pup_buffer_device(ptr, n, PUP::DeviceLife::Stable) declares a device buffer the element
   does not write between the balancer's decision and the pack. recvMigrate (a deferred move, pool mode) runs a sizing
   walk, copies the Stable buffers into one pool block laid out as the unpack's stable region (PUP::snapshot, a sizer
   that copies) and ships the block as a zerocopy payload (immigrateStable) while the element keeps stepping; the
   pack skips them (manifest entries with DEVICE_MANIFEST_STABLE). Under CHARM_DEBUG_MIGRATE only (user: not in the
   steady path), each Stable buffer is hashed at both points (hapi_kernels.cu, nvcc-built into libhybridapi) and the
   device compares: a change aborts naming the buffer (ckStableCheckDone); otherwise nothing is hashed. The
   destination keeps the landed block (keyed by element and snapshot generation) until
   the host message arrives, parks the message otherwise, and rebinds the Stable buffers into it as the arena is rebound.
   A decision that changes, a handoff, or a destroyed element discards the snapshot (discardStable by generation).
6. CkMigratable::setObjRebuildCost(seconds): priced into every move by DiffusionCostModel::migrateCost (used by
   DiffusionLB and MetisLB) through a second LB user-data slot; the LB sizing (recordLBSizes) sizes a move by what the
   pack ships (Stable buffers skipped), the footprint slot floors at the whole device state.
Tally (CHARM_LB_MIGSTATS=1, [mig-window] lines at exit, per process): decide->try, stand-down, pack (device), ship
(dispatch -> ack), land (payload landed -> unpacked), settle (ckJustMigrated), chan-hold, chan-close, snapshot lead.
Tests: chantest -a (CkLocRec::recvMigrate under +LBAsync: the balancer's path without a balancer) with a Stable ramp and
a Mutable iteration buffer verified after every arrival, and -x (the ramp rewritten after the decision: must abort);
chantest/test_chantest_stable.sh. leanmd/logs/stable1_chain.sh: chantest, then leanmd 8x4x4 async on one node, new
build zero-copy and channels x2 against leanmd.prev (the 2026-10-19 23:25 binary), with the tally.

**2026-10-20 11:00, job 23047794 (gpub001, one node, 4 A40): the five changes validated; the move window measured.**
chantest (memcpy and ipc transports; 24 iterations, a move every 3): plain, migrateMe moves, the balancer's path
under +LBAsync (-a: CkLocRec::recvMigrate, the Stable ramp snapshotted at the decision and the Mutable iteration
buffer packed, both rebound and verified after every arrival), all PASS, 14 snapshots per run; the broken promise
(-a -x under CHARM_DEBUG_MIGRATE: the Stable buffer rewritten after the decision) aborted naming the buffer in both
transports, as required. leanmd 8x4x4 async LB, 100 steps, mean42-100 ms/step (three runs each unless noted):
| build | zero-copy | channels |
|---|---|---|
| previous (2026-10-19 23:25, leanmd.prev) | 230.6 239.5 238.0 | 226.4 219.0 231.7 |
| this build | 225.5 225.1 233.1 | 206.5 208.2 210.6 |
| this build, senders not told (ablation) | 222.5 | 209.0 206.8 |
| this build, one migration stream (ablation) | 239.5 | 217.4 206.3 |
Channels: the bands do not overlap, -8%. Zero-copy: a 3% lean, the bands touch (233.1 vs 230.6). The ablation at n=2
(two more runs each queued as chain stable4) says the four migration streams carry most of the channel gain (one
stream: 206-217) and the sender notification is within the band (not told: 207-209); the gain is in the migration
path, where both items act. The tally (CHARM_LB_MIGSTATS=1, per process, this build): zero-copy pack (copies
enqueued -> complete on the device) mean 2-95 ms by process, max 256 ms, and ship (dispatch -> ack) tracks it
(45-96 ms mean) since the transfer waits for the pack's copies; land 0.05 ms; settle 0; decision -> first attempt
17-115 ms mean (the lag). Channels: the sender's hold (closing received -> new resolution) 3-74 ms mean by process,
max 339 ms -- the element's absence as its partner sees it -- of which the owner's closing -> ack is 1.5-3.6 ms mean
(max 106 ms); stand-down 5-23 ms mean. So the window is the pack's wait for the element's device work and the
transfer behind it, not the protocol's round trips: removing the closing ack (idea 2's remaining piece) would buy
2-4 ms of a ~50 ms window per move. The pack stage, not the ack, is where the next gain is (the kick fires at the
zero crossing of outstanding sends, while the element's last kernels may still be queued; four streams cut the
head-of-line wait among elements, the element's own wait remains).
**11:05, correction after chain stable4 (two more runs of each channel arm, and jacobi3d).** The three late runs were all
slow together (previous build 208.2 then 222.4; one stream 208.3 then 218.0; senders not told 202.9 then 216.6), so the
previous build's channel band is 208.2-231.7 over five runs and overlaps this build's 206.5-210.6: the channel
result is a LEAN of about 5%, not the 8% written above, and the ablation cannot attribute it (one stream 206.3-218.0,
senders not told 202.9-216.6, both inside the spread). Run-to-run spread at this size is ~10 ms on one node, so a
difference under 5% needs more than five runs or a larger problem. What stands: no regression on either path, the
Stable lifetime and the sender notification correct (chantest), and the window measured. jacobi3d 512^3 on one node
(16 blocks): zero-copy 1633.7 us, channels 1412.6 us, checksums identical to last night's (8.762191026712e+06).
Node 23047794 released 11:05 (user: "release the node after you are done"); the standard binaries leanmd, jacobi3d,
chantest are this build; leanmd.prev (2026-10-19 23:25), leanmd.noinform and leanmd.onestream are the control and
the two ablation variants. Nothing committed.

### 2026-10-20 13:40: the runtime presets CUDA_DEVICE_MAX_CONNECTIONS=32; today's bands were 8 queues

None of the leanmd, jacobi3d or chantest runs from 2026-10-18 to 2026-10-20 (the stable* chains, the after*/n2_*
chains, both builds, both ablations) set `CUDA_DEVICE_MAX_CONNECTIONS` or `CHARM_LB_HELPER`: only scale.sh carried
the 32-queue setting (since 2026-09-20) and only the sph3d runners enabled the LB helper thread. So every number in
the two entries above is at the driver default of 8 hardware queues with the CUPTI load build and the strategy on a
PE. The comparisons are like-for-like (both arms, both builds, the ablations), but the absolute numbers do not belong
next to the September scaling series, and the 16x8x8 references of 2026-10-19 were two-node runs.

Change: `hapiPresetEnvironment()` (hapi_impl.cpp) sets `CUDA_DEVICE_MAX_CONNECTIONS=32` unless the variable is
already in the environment; `charm_main` (init.C) calls it before `ConverseInit`, because the machine layer's LCI
calls `cuInit(0)` during ConverseInit and the driver reads the environment then. The HAPI Config banner now ends
with `N hardware queue(s) per device (runtime default | CUDA_DEVICE_MAX_CONNECTIONS from the environment)`.
scale.sh exports the variable only when `CONNECTIONS=` is given. Built 13:34-13:36 (hybridapi, ck, LB modules;
leanmd, jacobi3d, chantest relinked and carry the symbol). The LB helper thread stays opt-in (`CHARM_LB_HELPER`):
its last leanmd measurement (2026-09-21) was 3.4% slower with the cause never found.

Unvalidated on hardware. `tests/util/streamorder/connpreset.cu` (built) measures the effective queue count after
an in-process setenv placed before cuInit, after cuInit, or after the primary context exists; the next node runs
`examples/charm++/cuda/gpudirect/leanmd/logs/conn1_chain.sh <jobid>`: connpreset (kill criterion: "before" must
give 32), chantest, leanmd 8x4x4 async zc x3 / ch x3 at the preset plus a CONNECTIONS=8 pair as the same-session
reference, 16x8x8 async zc x2 / ch x2 on one node, and if time allows the 16x8x8 no-LB pair.

### 2026-10-20 15:30: job 23053532 (gpub002, 14:39-15:25): the preset validated; channels lose at 16x8x8 on one node

**Placement check** (`tests/util/streamorder/connpreset`, rewritten: a two-stream pair no longer aliases on driver
595.71.05, queues are assigned dynamically; the signal is 64 streams x 2 dependent kernels depth-first): setenv
before cuInit = 32 queues (3.63 T, implied 24.3), after cuInit = 8 (8.98 T), after the context = 8. The driver reads
the variable at cuInit, so charm_main (ahead of LCI's cuInit in ConverseInit) is the only correct place. Every
leanmd/jacobi3d banner read "32 hardware queue(s) per device (runtime default)", and CONNECTIONS=8 read "from the
environment".

**leanmd 8x4x4 async, one node, mean42-100 ms/step:** 32 queues zc 203.3 / 217.8 / 219.1, ch 209.2 / 209.9 / 208.9;
8 queues (same session) zc 223.8, ch 213.0; this morning at 8: zc 225.5 / 225.1 / 233.1, ch 206.5 / 208.2 / 210.6.
Channels do not move with the queue count; zero-copy gains ~5% at 32, which closes most of its gap to channels here.
The move window is unchanged (pack 1-93 ms means at both counts).

**leanmd 16x8x8 async, one node (new; the 2026-10-19 16x8x8 runs were two nodes):** zc 1537.6 / 1548.7, ch 1624.5 /
1567.6 (channels 1-6% behind). Pre-LB steps 1-20 tie (2401-2406); the quiet window after the Metis moves (25-40)
nearly ties (zc 1585/1551, ch 1591/1587); the loss is in the Diffusion phase with moves in flight, where the channel
arm's device time per interval is LOWER (28.1-28.7 s vs 28.5-29.8 s) while its steps are longer: waiting, not work.
The channel arm's tallies: chan-hold 27,411 holds for 5,941 moves, 305 ms mean, 3.5 s max (worst process 782 ms
mean in run 1, 1141 ms in run 2); stand-down 43-375 ms per move (zero-copy: 0); chan-close 56 ms mean. Sync balancer:
zc 1626.5, ch 1672.1 (2.8% behind); no LB: zc 2882.1, ch 2947.1 (2.3% behind, one pair). jacobi3d 512^3: ch
1416/1415 us at 32 and 1414 at 8, zc 1634/1676 at 32 and 1613 at 8, checksums identical.

**Cause, from the code.** A channel move is a chain of message hops through scheduler queues that are 50-400 ms
deep at 448 objects per PE: closing to every sender and an ack back before the owner may pack (stand-down); after
the element lands, reopen on the new PE and the resolution back to the sender before a parked put can go. A put on
a closed channel parks with the application's source pinned, and leanmd's cell then waits for forces from the
compute that cannot run until the parked put is finally issued. Zero-copy forwards a message once through the
location manager and never parks a sender. `CkRdmaDeviceChannelsBeforeMigrate` also runs before the runtime decides a
move is a same-device handoff, where nothing is packed. (The CHARM_ZC_STATS pair, 15:18-15:22, was wasted: the
channel put counters were never printed; fixed below.)

**Implemented (uncommitted, built 15:30, NOT yet run on a node):**
1. *No sender hold* (`CkDeviceChannel::issueStaged`): a put while the channel is closed (or behind such puts) is
   copied to a pool staging buffer on the producing stream, the source is released to the application once the
   copy has read it, and the staging buffer is announced to the receiving element -- an element message the
   location manager parks and forwards -- which reads it into the reopened buffer in put order (`startPull` now
   requires seq == taken_seq + 1). The receiver's DONE frees the staging buffer through a per-PE table keyed by
   (token, seq) and the sender debits one credit (`onStagedDone`); every release credits, so the balance holds.
   A put parked for a credit when the closing arrives is converted the same way. The sending element is charged
   one outstanding device send per staged put until its DONE, so it does not move with one unread.
2. *Urgent control messages*: closing, close-ack (and the blind ack), resolution, message credits and DONE go by
   `CmiSyncSendAndFreeUrgent`; reconverse serves the urgent queue first in every scheduler iteration, whichever way
   the message arrived. Each hop now costs the entry method running at the destination, not the queue depth. This
   also covers the same-device handoff (its closing/ack round trip becomes ~1 entry method), so the handoff keeps
   its handshake: skipping it would need the acks forwarded to the new PE for slot and credit retirement.
3. The ack-free pack was NOT implemented: with urgent acks the stand-down is one entry method; removing the acks
   needs deferred frees under the memory contract and a late-put forwarder, for little left to gain.
4. `[chan-stats]` prints the channel put counters at exit under CHARM_ZC_STATS=1 (per tier mean us, queued,
   staged, events).

Validation (next node): chantest all arms (memcpy and ipc, lbx aborts), then leanmd 16x8x8 async one node zc x2 /
ch x2 (target: the 1538-1549 vs 1568-1625 pair; the staged counters in the tally), 8x4x4 async zc x2 / ch x2 (the
win there must survive), and the no-LB 16x8x8 pair with CHARM_ZC_STATS=1 for the 2.3% steady-state gap.

### 2026-10-20 17:15: job 23058109 (gpub001, 16:21-17:12): the protocol change validated; the no-LB gap profiled

**chantest:** all 8 arms pass on the 15:37 build (memcpy and ipc; plain, lb, lba; both lbx aborts), 14 moves each.
**leanmd 16x8x8 async, one node, mean42-100 ms/step (this node runs ~5% slower than gpub002):** zero-copy 1639.4 /
1579.5 / 1638.7, channels 1548.2 / 1638.9 / 1569.1. Before the change (gpub002, 15:00) channels were 1-6% behind on
both pairs; now the bands overlap (channel means 2% lower): the loss is gone, a win is not demonstrable at this
noise (single steps swing 800 ms; a 59-step mean carries ~2%). Per process 327-3,301 puts went out as staged
announces (chan-stats "staged"), no CHAN-STALL line anywhere; stand-down rose to 173 ms mean because an element
now also waits for its own staged puts to be read (they count as outstanding sends) before it may move.
**leanmd 8x4x4 async:** zero-copy 209.8 / 206.4, channels 203.1 / 220.9: a tie (the 8x4x4 "win" of the morning was
within this noise too). **No LB 16x8x8:** zero-copy 2884.9, channels 2946.5 (+2.1%; +2.3% on gpub002): the
steady-state gap is reproducible and unrelated to migration. The fourth extra pair was cut at the user's request
("stop doing repeat runs"): repeats past 3 per arm add nothing at this noise.

**Profile (nsys, rank 0, 30 steps, no LB, channels; `/scratch/mzu/bhosale/nsys/conn2_ch/`):** cudaMemcpyAsync
422,912 calls (7.0 us host each), cuMemsetD32Async 952,584 calls (2.8 us host each) = the completion-flag writes
(+gpuflagmemset is the default flag path), on the GPU 952,840 memsets of 1.33 us = 56.5% of all copy-engine time,
380,160 D2D memcpys of 20 KB (2.26 us) = 38%. Each channel put adds TWO stream callbacks (source done + arrival
note, persistentStreamCallbacks) = two flag writes, and one unbatched memcpy. The zero-copy receive path batches its
landings (one memcpy batch + one flag per batch, deviceRecvBatchDone) and adds one flag per send (ipcSrcReadyFn).
Per PE per step that is ~1,750 puts x (2 flags + 1 copy) on the channel path: ~2.7% of the run on the copy engine
and in stream serialization, the size of the gap. Candidates, not implemented: (a) one flag per put (fire src_done
from the arrival-note callback: the copy's completion implies the source is done); (b) batch the puts a PE issues
in one entry method into one memcpy batch + one flag, as the zero-copy receive path does.
The zero-copy profile could not be written: under nsys with +nvtx the zero-copy run hangs after "Exit called" on
every PE (twice; 420 s and 300 s timeouts; --duration=115 did not help), the channel run exits normally. Its log
also shows "a 13200-byte device buffer is being sent over direct CUDA IPC with no completion callback" (leanmd's
zero-copy path, pre-existing). Node released 17:12.

### 2026-10-20 19:00: job 23063562 (gpub002, 18:26-): flags counted on both paths; the one-flag put; the nsys "hang"

**Flag writes per process, 16x8x8 no LB, 30 steps, 421,376 messages:** channels 952,576 (2.26 per put), zero-copy
305,294-344,367 (0.8 per message). **One-flag put** (`deviceChannelPutDoneFn`: on the stream tiers the source-done
callback delivers the note too; across hosts the completion handler fires both without a flag): channels 537,856
flags (1.28 per put), put host cost 10.8 -> 8.5 us, chantest 8/8, 16x8x8 async 1546.8 (zc on this node 1538-1549),
8x4x4 async 216.1 -- but the no-LB mean is unchanged, 2584.3 vs 2588.2 with two flags vs zero-copy 2502.4: the
flags were a cost, not the 3% gap. Built as leanmd.oneflag (the chain's leanmd untouched); libck and chantest
carry it.
**The nsys "hang" was not a hang.** The zero-copy application exits normally (all four ranks print their exit
counters); nsys keeps draining the agent's trace for minutes afterwards (qdstrm files of 690-994 MB in the node's
/tmp against an 87 MB channel report: the zero-copy path makes ~10x the CUDA API calls per message), and every
timeout killed it mid-drain, leaving "incomplete or invalid" streams that QdstrmImporter refuses. Fix: fewer steps
and a long allowance (conn3_zc10.sh: 10 steps, 1200 s). The chain's stack capture had matched nsys's own command
line (it carries the app's arguments); c3_appstack.sh matches the process by name.

### 2026-10-20 19:10: the zero-copy profile, like for like (rank 0, 10 steps, no LB), and the one-flag profile

Reports: `/scratch/mzu/bhosale/nsys/conn3_{ch10,oneflag10,zc10}/rank_0.nsys-rep` (zero-copy's is 616 MB for 10 steps
against 87 MB for 30 channel steps: 17.9 MILLION cudaEventQuery calls in 10 steps, the idle PEs polling events).
| per 10 steps, rank 0 | channels, two flags | channels, one flag | zero-copy |
|---|---|---|---|
| kernels total | 4393.6 ms | 4384.4 ms | 4357.3 ms |
| copy engine total | 779.8 ms | 599.7 ms | 401.5 ms |
| memsets (flags) | 317,448 / 424.7 ms | 179,208 / 257.1 ms | 96,934 / 129.9 ms |
| D2D copies (126,720 each) | 302.0 ms (2.38 us) | 290.0 ms | 232.5 ms (1.83 us, batched: 13,140 cudaMemcpyBatchAsync) |
| host API total | 5.0 s | 4.6 s | 13.4 s (8.3 s of it cudaEventQuery while idle) |
The kernels agree within 0.8%; the difference is the copy engine, flags first and unbatched copies second, and the
one-flag put closes half of it on rank 0 -- yet the no-LB step time did not move (2584 vs 2588 vs zc 2502, 30-step
runs), so rank 0's copy engine is not what sets the step. Rank 0 is the lightest rank under the density gradient
(its PEs idle 95%); the step is set by rank 3's GPU. conn3_rank3.sh profiles rank 3 on both paths (19:09).

### 2026-10-20 19:15: the busiest rank (rank 3, top of the density gradient), 10 steps, no LB

`/scratch/mzu/bhosale/nsys/conn3_r3_{ch,zc,oneflag}/rank_3.nsys-rep`. Rank 3's force kernel averages 701 us (118 on
rank 0) and its kernels fill the step (26.1 s per 10 steps on a 2.45 s step: the GPU is saturated); this rank sets
the step.
| rank 3, per 10 steps | channels (two flags) | zero-copy |
|---|---|---|
| kernels total | 26,132.8 ms | 26,128.8 ms (equal: no interference) |
| copy engine total | 917.5 ms | 512.0 ms |
| memsets (flags) | 317,448 / 458.3 ms (1.44 us) | 111,088 / 167.5 ms (1.51 us) |
| D2D copies, 126,720 each | 393.0 ms (3.10 us each) | 286.8 ms (2.26 us each, batched) |
So on the GPU that sets the step, channels spend 41 ms more per step on the copy engine, which is where the puts'
chain (copy, flag, host poll, note, consumer's launch) runs: 2.9x the flag memsets and single copies 37% slower each
than the batched ones. That is the 2-3% no-LB gap's size (75 ms of a 2.5 s step). The one-flag put takes out about
half the memset part; the rest needs the puts batched the way the zero-copy receive path batches its landings
(cudaMemcpyBatchAsync: 14,715 batches for 126,720 copies here). Host API time under nsys is distorted (stream
creation 1.8 ms per call, batch calls 155 us) and the step means under nsys are not comparable.
One-flag put on rank 3 (`conn3_r3_oneflag`): copy engine 713.4 ms per 10 steps (memsets 179,208 / 281.1 ms, copies
368.3 ms at 2.91 us), kernels 26,108 ms (unchanged): halfway from 917.5 to zero-copy's 512.0. The standard leanmd
and jacobi3d are relinked against the one-flag library (19:11); leanmd.oneflag is the same code. Next for the
steady-state gap: batch a PE's puts (the copies, and one flag per batch). Node 23063562 released 19:11.

### 2026-10-20 19:20: channel puts batched (built; validation queued as job 23067051, conn4_chain.sh)

`deviceChannelBatchAdd/Flush` (ckrdmadevice.C): on the stream tiers (same process, same host) a put no longer issues
its own copy and flag; it joins this PE's open batch on its producing stream (a cell's thirteen position puts, a
compute's two force puts). The batch is one `cudaMemcpyBatchAsync` on that stream (one `cudaMemcpyAsync` per put if
the driver refuses) with ONE completion flag, whose callback delivers every put's note and source-done. A batch
flushes when the producing stream changes, at 256 puts, at every scheduler iteration and at idle, so it never
outlives the entry method that opened it; the cross-host tier keeps its one-sided write. The one-flag merge of 17:30
is subsumed (a batch of one). Expected on rank 3 per 10 steps: ~36k batches for 138k puts, flags ~77k against
179k (one flag) / 317k (two) / 111k (zero-copy), copies at the batched rate (2.26 us). `[chan-stats]` prints
batches, puts per batch, the largest, and refusals. chantest, the no-LB pair, the async ties and the rank-3 profile
run on the next node; chain: leanmd/logs/conn4_chain.sh.

### 2026-10-20 20:20: batched channel puts validated (job 23067051, gpub002)

chantest 8/8. **No LB, 16x8x8, 30 steps, mean 11-30, same node and pair:** channels batched 2571.6 vs zero-copy
2587.5 ms/step -- the first no-LB pair of the day with channels not behind (per-put 2947 vs 2882 at 16:42,
one-flag 2584 vs 2502 at 18:29; zero-copy itself drifted 2502 -> 2588 on this node over two hours, so the earlier
"3% gap" carried drift as well as cost). Per process: 125,184 batches for 414,720 puts (3.3 per batch: a compute's
two force puts dominate, a cell's thirteen position puts make the largest, 27), no batched-copy refusals, flag
writes 248,320 (two-flag 952,576, one-flag 537,856, zero-copy 307-344k), put host cost 10.8 -> 0.4 us.
**Rank 3 profile, 10 steps, copy engine:** per-put 917.5, one-flag 713.4, batched 537.9, zero-copy 512.0 ms;
memsets 89,352 (zc 111,088); D2D copies 331 ms at 2.61 us (zc 287 at 2.26); kernels identical (26.1 s).
**Async, batched channels:** 16x8x8 1618.0, 8x4x4 229.8 ms/step; their zero-copy controls run next on this node
(the node is ~3% slower than at 15:00-18:50, when zc was 1538-1549 and 203-219).

### 2026-10-20 21:05: batching attributed to the element; the "hang" was a stale shared-memory segment

**8x4x4 async with batching (20:18-20:27):** channels 229.8 / 232.1 vs zero-copy 198.4 / 209.0 (12% behind, where
every earlier 8x4x4 pair tied); no-LB 8x4x4 tied (322.9 vs 323.7). The channel arm's balancer moved 20-30% more
objects than the zero-copy arm's: a batch flushed from the scheduler loop launches its copies under no entry
method, so the balancer's per-object device load no longer carried them. Fix: `CkRdmaDeviceChannelBatchFlush()`
from `CkCallstackPop`, before the entry's CUPTI correlation is popped (only a batch opened inside an entry; one
opened from a converse handler waits for the scheduler loop).
**Then four runs in a row "hung" before step 1 (20:31-20:53), on three builds including the plain revert.** Stacks:
every PE idle in the scheduler; channel debug: opens and resolutions complete, "first credit through shared
memory", then nothing. Cause: the channel's shared-memory segment (`charm-chan-<host>-<job>`) is created once per
JOB and relies on zero pages as its empty state; a step I killed by hand at 20:30 left it behind with live ring and
credit state, and every later step opened it. `rm /dev/shm/charm-chan-*` on the node and the same binary ran. Now
named per job STEP (`-<SLURM_STEP_ID>`). The flush-at-pop revert of 20:51 was undone; nothing was wrong with it.
**Final build (20:56; chantest 8/8):** 8x4x4 async channels 209.5 vs zero-copy 219.4 (migrations 1360 vs 1243, the
attribution restored); 16x8x8 async channels 1603.1 vs zero-copy 1663.3 (20:22 control, same node); no-LB 16x8x8
batched 2571.6 vs 2587.5 (20:13); rank-3 copy engine 538 vs 512 ms per 10 steps. Node 23067051 released 21:03.
Open: more than one pair per claim; the two-node and 16x8x8-two-node configurations on this protocol.
