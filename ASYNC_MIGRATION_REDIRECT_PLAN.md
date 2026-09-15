# Async LB: redirect a migrating element's arrivals instead of parking

Status: steps 3 and 5 implemented 2026-09-15 as `CkRdmaDeviceStageParked`
(ckrdmadevice.C) plus the park-before-cache in `CkArray::recvMsg` (ckarray.C).
Steps 1, 2 and 4 (redirect from the decision) are not implemented: the
timestamped runs showed the moves DO fire by kick; what they wait on is the
kick handler's place in the scheduler queue and the pack copies' stream
ordering, neither of which a redirect changes. Section 2 below is therefore
superseded: elements do reach zero, the kick loses a race. Written 2026-09-14
from the LeanMD async measurements below.

## 1. Symptom

Under `+LBAsync` the LB event is supposed to disappear into the overlap
window. It does, sometimes. Same binary, same node, same command
(job 22074542, LeanMD 8x8x8, 42 A cells / 2744 atoms, `-lblag 16`):

| run | mean 21-100 | mean 42-100 | step 21 (first LB step) |
|---|---|---|---|
| sync                | 1194.0 | 1137.2 | 2156 |
| async lag 2         | 1258.3 | 1214.3 | 2401 |
| async lag 16, run 1 | 1165.9 | 1104.2 | **1135** |
| async lag 16, run 2 | 1191.0 | 1124.6 | 2227 |

Run 1 hid the entire event: step 21 cost an ordinary step. Run 2 paid a full
spike. The mechanism works and then misses, about half the time.

Lag matters because the park is the deadline: `-lblag 2` gives the event two
steps before the app blocks, and it was costing 7-8% against both sync and
lag 16. Every LeanMD async number before this was taken at lag 2. The cap is
`ldbPeriod - LB_INSTRUMENT_WINDOW - 1` = 16 for a 20-step period (Main.cc).

## 2. Cause

A move is safe when `outstandingDeviceSends == 0` for the element -- it counts
sends still sourcing from its buffers and receives deferred into them
(`CkNoteDeviceRecvDeferred` / `CkNoteDeviceRecvComplete`, cklocation.C). When
the strategy's move arrives, the element either is device-quiet at that instant
and emigrates, or `pendingMigrateTo` is set and the move waits for
`noteDeviceSendDone` to reach zero, which enqueues a kick
(`_deferredMigrateHandler`).

A *running* element has no reason to reach zero. Each step a Compute takes two
position buffers in and puts two force buffers out, so new in-flight work
arrives before the old completes. The code says this itself, in the comment
above `migrateAtParkOnly()`:

> The park is not what makes a move safe. outstandingDeviceSends is ... What
> the park adds is a guarantee the count REACHES zero -- admission control
> bounces new device receives for a parked element with a move pending, so it
> cannot be fed faster than it drains.

So the element is fed as fast as it drains and falls through to the park, where
all ~1350 drains serialize into one point. Whether any given element happens to
cross zero earlier is a race against its own completion traffic -- hence the
intermittency, and hence why it got worse when objects grew ~5x (2744 atoms).

### Ruled out

- **The app is not gating migration.** LeanMD never calls `ReadyMigrate`; it
  sets `usesAtSync = true` and `setMigratable(false)` on Cells. Nothing in the
  runtime or the balancers calls `ReadyMigrate(false)` either, so the
  safe-to-pack gate is always open and the `[KICK-HOLD]` re-buffer path is dead
  code for this app.
- **Not detection lag.** The source callback decrements the counter as soon as
  the send completes; there is no polling delay to remove.
- **Not instrumentation contamination.** The window is the 3 steps before each
  LB step (`Compute::updateInstrumentation`), and with lag 16 the park is at
  step 36 against a window of 37-40. No overlap.

## 3. Design

As soon as a PE knows an element must migrate, stop the element being fed, by
moving its arrivals to where it is going rather than holding them where it is.

1. **At the decision** (`pendingMigrateTo` set): remove the element from the
   local location map and install a forwarding entry to `toPe`.
2. **Arrivals are redirected**, not buffered at the old PE. A device post is a
   rendezvous control message -- no bytes have moved -- so the redirect is a
   control hop.
3. **At the new home** the element has not arrived yet. A message for an
   element that is not here would normally be bounced to its home PE, which
   would send it back. A bit on the message (or on a placeholder record) marks
   it "stay here, the element is coming", and it is held locally.
4. **The sender's location cache is updated on the forward**, so from its next
   message onward it targets the new PE directly. The redirect cost is
   therefore one hop per sender per move, plus whatever was already in flight.
5. **Redirected receives are staged.** The new home allocates a landing buffer
   of the size in the descriptor and performs the pull itself -- an ordinary
   direct receive that lands in a runtime buffer rather than the element's. The
   sender's completion fires on that transfer. When the element arrives, its
   posted buffer is filled from the staged one with a local device copy.
6. **Meanwhile the element drains and goes.** Nothing new is arriving for it,
   so its outstanding count decreases monotonically to zero, the kick fires,
   and it emigrates mid-window instead of at the park.

### Why 5 is required

Two reasons, one of them correctness:

- **Deadlock.** A direct send keeps the sender's own buffer live until the
  receiver reads it. If the receiver is migrating and the post is merely held,
  the sender's stand-down never clears; if that sender is itself migrating,
  neither can leave. Any cycle in the "outstanding send" graph among
  simultaneously migrating elements wedges. Staging releases the sender at the
  transfer, so no element's departure depends on another's arrival and the
  graph that could contain a cycle stops existing. (LeanMD cannot hit this --
  Computes send only to Cells and Cells never migrate -- so a prototype here is
  safe, but the general fix needs it.)
- **Sender stalls.** A non-migrating sender (a Cell) would otherwise wait
  through the whole drain-pack-move of the element it sent to. Staging
  completes its send immediately.

Staged receive is not new machinery: it is the path that predates
`+gpuipcdirect`.

### Payload repair is already written

A send prepared for a co-resident target carries a process-local pointer and
cannot be forwarded verbatim to another process. That case is already handled
by the forward-time repair in `ckrdmadevice.C` ("Forward-time repair of a
memcpy-prepared payload", ~line 1047): in the process that owns the source, it
re-exports the source, claims an IPC event slot, records the event behind the
memcpy event and rewrites the descriptor, with the same treatment for a forward
off the physical node. `requestDeviceRestage` (~line 1450) is the older
correction round trip and remains the fallback.

## 4. Correctness notes

- **Nothing is stranded.** Every `AtSyncStart` owes an `AtSyncWait` and the LB
  step does not complete until every migration lands, so a held message cannot
  outlive the step. Remaining case: element deletion, which LeanMD never does.
- **Ordering is not a constraint.** Charm guarantees no ordering between
  messages, so a later message overtaking an earlier one via the redirect
  changes no semantics. (LeanMD tags positions with `stepCount` and consumes
  them through SDAG `when` in any case.)
- **The park stays** as the backstop for anything that does not cross.
  `CHARM_LB_MIGRATE_AT_PARK_ONLY` already restores park-only behaviour for
  bisecting.

## 5. Open question

`+gpupool` is direct-only and deliberately has no comm/LB buffer, so the
landing buffer for a staged redirected receive has to come from the pool, sized
from the descriptor at arrival. If the pool can serve it, 5 works as written.
If it cannot, the redirect still works but the deadlock backstop reverts to the
park -- which is today's behaviour, so nothing regresses.

## 6. Where it lands

- `src/ck-core/cklocation.C` -- install the forwarding entry when
  `pendingMigrateTo` is set; the hold bit and the "don't bounce home" rule;
  release held messages when the element arrives.
- `src/ck-core/ckrdmadevice.C` -- choose the staged path for a redirected
  receive; reuse the forward-time repair for the descriptor.
- `src/ck-core/cklocrec.h` -- state for the forwarding entry and the hold.

## 7. Validation

1. **Confirm the premise first.** `CHARM_DEBUG_MIGRATE=1` prints
   `[KICK pe] id=... movable=0/1 toPe=...`. Compare a fast run and a slow run
   on how many of the ~1350 moves went by kick versus fell to the park, and how
   long each element's count stayed above zero. If elements are continuously
   non-zero from decision to park, this plan is the fix. If they cross early
   and the move is not taken, the bug is in the kick path instead and this is
   the wrong change.
2. After the change, the LB-step spike should be absent on *every* run at
   lag 16, not one in two, and async should beat sync rather than tie it.
3. Controls: `CHARM_LB_MIGRATE_AT_PARK_ONLY=1` for the old behaviour; the lag
   sweep (2 / 8 / 16) should flatten, since the park deadline stops mattering
   once moves are taken mid-window.

## 8. Measurements this rests on

- Size sweep, gpub023, stock granularity (600 atoms/cell), mean 42-100:
  64 cells/GPU 138 noLB / 106 balanced; 128: 278 / 214; 192: 428 / 390;
  256: 577 / 577. noLB flat at 0.54-0.56 ms per cell at every size; balanced
  flat at 0.417 up to 128 then decays to the noLB line.
- Granularity change (`CELL_MARGIN` 4 -> 16, `PERDIM` 10 -> 14, densities
  550/2200/2744, cutoff unchanged at 26): 8x8x8 step 278 -> 1388 ms, device
  share 38% -> 48-50%, launch 29% -> 19-20%, energy drift 1.7e-16 (test passes).
- Sep 7 triplicate at the stock granularity had lag 16 *worse* than lag 2
  (196.0 vs 187.6 vs sync 183.8, mean 42-100). That reverses at the new
  granularity, which is consistent with the park deadline being the binding
  constraint: bigger objects drain slower, so a 2-step deadline is hopeless and
  a 16-step one is merely unreliable.
