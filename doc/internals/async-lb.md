# Asynchronous load balancing

A balancing step is joined in two halves, `AtSyncStart()` and `AtSyncWait()`;
`AtSync()` is the two back to back. Under `+LBAsync` an element keeps running
between them, so the strategy and the migrations overlap the application. This
chapter describes how an element joins and is handed back a step, how a
distributed step knows its moves are done (the move ledger), how a centralized
decision reaches the PEs (`+LBPeerDecision`), how MetaBalancer agrees on when to
balance from GPU loads (`+MetaLBGpuTrigger`), and how loads are measured while
elements keep running. User-facing behavior is in the manual, section "Load
Balancing Chare Arrays".

## Joining and handing back a step

`CkMigratable` (src/ck-core/ckmigratable.h, cklocation.C) keeps, per element:

| field | meaning |
|---|---|
| `lbStepPending` | the element has joined a step that has not been handed back |
| `waitParked` | the element is stopped in `AtSyncWait()` |
| `lbWaitEpoch` | `LBManager::resumeEpoch()` when it joined |
| `lbMeasurementClosed` | its measurement window is closed (below) |
| `lbStepSeen`, `lbIterNo`, `lbStepBlocked`, `metaLBStreamJoinedPe` | GPU trigger state (below) |

All but `metaLBStreamJoinedPe` are pupped on migration (not on checkpoint, like
the barrier epoch).

`AtSyncStart()` (`lbStart(wait, split=true)`) returns `Continue` when no
balancer is listening or no step is due, `Blocked` when MetaBalancer stops the
element, and otherwise joins: `lbJoinStep` sets `AsyncMigrate`/`ReadyMigrate`,
records sizes, closes the window, sets `lbStepPending`, and calls
`LBManager::AtLocalBarrier` last. `AtSync()` calls `lbStart(wait, false)` then
`AtSyncWait()`; only the split form aborts when called with a step pending, so
an `AtSync()` repeated before a resume joins the next step, as it always did.

The step can end inside the `atBarrier` call that completes the PE's barrier
(no balancer: `LBManager::InvokeLB` resumes inline), so the element is resumed
by exactly one of:

- `AtSyncWait()` with `lbStepPending` clear: no step, or the step is over.
  Opens the window, calls `ResumeFromSync()`.
- `ResumeFromSyncHelper` (the barrier's resume) with `waitParked` set: the
  element is stopped in `AtSyncWait()`.
- `ResumeFromSyncHelper` with neither flag set: MetaBalancer's wall-time path,
  which stops the element (`Blocked`) and joins it through `metaLBCallLB()`, or
  the second step of a repeated `AtSync()`.
- `lbCheckWaitRelease` on arrival (below).

`ResumeFromSyncHelper` with `lbStepPending` set and `waitParked` clear only
clears `lbStepPending`; `AtSyncWait()` later finds the step over.

`AtSyncWait()` re-asserts `ReadyMigrate(true)` for a step joined with
`waitForMigration`, so a move buffered in `nextPe` by an element that narrowed
its window departs when its entry method returns (`checkBufferedMigration`). A
step joined with `AtSyncStart(0)` leaves readiness to the application (TCharm
moves its thread only at `allow_migrate`).

`+LBAsync` forces `+LBSyncResume`, so every PE runs `LBManager::ResumeClients`
once per step from a global reduction and `resume_epoch` advances in step on
all PEs. An element stopped in `AtSyncWait()` and moved by the step it waits
on carries `lbStepPending`/`waitParked`/`lbWaitEpoch`; on arrival
(`immigrateWithDevice`, `processAfterActiveRgetsCompleted`)
`lbCheckWaitRelease` hands it back if its new PE's epoch is already past
`lbWaitEpoch`.

The framework is one step at a time: CentralLB and DistBaseLB both hold the
AtSync barrier off (`RegisteringObjects`) from the start of a step until its
clients are resumed, so an element still running cannot start the next step.

Under `+LBAsync`, `CkLocRec::recvMigrate` does not pack the element inline in
the decision loop: it posts an `LBDeferredMoveMsg` to the PE's own queue
(`_lbDeferredMoveHandler`, registered in init.C), which takes the move by
element id when it runs, or buffers it in `nextPe` if the element has closed
its window since.

Arrival accounting: the migration message carries "ordered by the balancer"
(`ignoreArrival` false) and "asynchronous" (`asyncArrival`, from
`AtSync(0)`) separately. An ordered asynchronous arrival is reported as
`Migrated(h, false)`; CentralLB counts it in `future_migrates_completed` and
holds the end of the step (`CheckMigrationComplete`) until all are in. TreeLB
refuses asynchronous arrivals.

## The move ledger

Under `+LBSyncResume` a DistBaseLB step ends on the ledger in `LBManager`
(src/ck-ldb/LBManager.{h,C}) instead of arrival counts:

1. `DistBaseLB::ProcessMigrationDecision` calls `ledgerOpen(step)` before its
   `Migrate` loop.
2. Each move reaches `CkLocRec::recvMigrate` synchronously, which calls
   `ledgerRecord(locMgrGroup, elementId, toPe)` and tags the record
   (`lbLedgerStep`) before any deferral.
3. `ledgerClose(fn)` after the loop; `fn` (`MigrationDone(1)`) runs when the
   ledger is empty.
4. `CkLocMgr::emigrate` stamps `lbLedgerSrcPe`/`lbLedgerStep` into the
   migration message and clears the tag; the destination, once the element
   has registered, sends `ledgerRetire(locMgrGroup, id, step)` to the source.
5. A record destroyed with its tag set retires its entry locally.

An ack that matches no entry aborts; `ledgerWatch` (once a second) prints the
outstanding entries of a ledger that has not drained after ten seconds. The
key is (location manager group, element id) because element ids are unique
only within one location manager. A source's MigrationDone no longer waits for
its own arrivals; the global resume waits for every source's ledger, which
covers every arrival. Without `+LBSyncResume`, DistBaseLB counts its own
arrivals as before. CentralLB never opens a ledger.

## Peer decision

With `+LBPeerDecision` (forced by `+LBAsync`) `CentralLB::ApplyDecision` calls
`InitiateScatter` instead of broadcasting. The decision is split by source PE
down a binary tree of `LBScatterMsg`s (`ScatterMigrationResults`); each
carries, per PE of its span, the arrivals the balancer waits for
(`numMigratesPerPe`) and those it does not (`numFutureMigratesPerPe`), and the
next step's decision PE (`next_lb`). Spans of at most `+LBScatterThreshold`
PEs (32) are broadcast. `ProcessMigrationDecision` sets the two expected
counts and issues its own moves; a move whose object died since the
statistics sends `MissMigrate` (`LBDatabase::Migrate` returns 0 for an
unregistered handle).

## MetaBalancer's GPU trigger

With `+MetaLBGpuTrigger` (src/ck-ldb/MetaBalancer.{h,C}) each sample index
closes per PE once all its elements have called `AtSyncSample()`
(`sampleMetaLBLoad` -> `AddLoad` -> `ContributeStats`). The PE's contribution
carries its device id and its share of that device's load
(`hapiPrepareCuptiLoads(epoch)`, `GetTotalObjGPULoad`) and goes point to point
to PE 0 (`ReceiveGpuSample`), which accumulates by sample index rather than
through a reduction: a free-running stream cannot keep every PE's Nth
contribution describing the same sample. PE 0 evaluates a complete index
(`EvaluateGpuSample`), skipping samples flagged as taken during a step; at
max/mean above `+MetaLBGpuTolerance` it opens a vote:

1. `GpuCollectIter(seq, tentative)` to all PEs: set `_lbStepRequested = seq`,
   `_lbStepTentative`, `_lbStepPeriod = 0`; reply with the largest `lbIterNo`
   reached locally (`GpuReportIter`).
2. PE 0 broadcasts `RequestLBStep(seq, max + 1)`, which sets `_lbStepPeriod`
   and releases elements stopped at the floor (`MetaLBResumeWaitingChares` ->
   `recvLBPeriod`).

`AtSyncStart` increments `lbIterNo` on every call; with a step requested it
continues until the agreed count, or stops at the tentative floor
(`Blocked`, `lbStepBlocked`) while the count is open. `recvLBPeriod` joins a
stopped element whose count is reached (under +LBAsync handing it back at
once; otherwise marking it stopped in `AtSyncWait` before joining) or lets it
continue.

An element joins its PE's sample stream at its first sample, and at its first
sample after arriving on another PE (`metaLBStreamJoinedPe`); the source's
stream lets it go in `~CkMigratable`. The stream is not reset at a step's
resume. The wall-time path keeps its reduction (`lbDataCollection`, with a
device tail in the payload) and its old stream handling.

## Measuring while elements run

Under `+LBAsync` an element's window closes at its join (under the barrier a
joined element waits, and it is never closed) and opens when the step is handed back
(`lbSetMeasurementClosed`), PE-locally in three places: `LBObj::joinedStep`
stops `IncrementTime`/`IncrementGPUTime`, `LBDatabase::Send`/`MulticastSend`
stop recording its traffic and `EstObjGPULoad` ignores a declared device load
(counted under `+LBDebug`), and HAPI pushes the no-object correlation for its
entry methods (`hapiCuptiObjectJoinedStep`), leaving its kernels unattributed.

`LBObj` banks the time its window was open (`LDObjData::windowTime`, LB format
6) and the span it was active, from its first `StartTimer` after `ClearLoads`
to its join. `LBDatabase::GetObjData` multiplies host and cpu load by
`activeSpan / windowTime` (`sampleScale`, at most 10), `GetCommData` each
record by its sender's factor (`LBCommTable::GetCommData(data, scale)`). Under
the barrier span and window coincide. Device load is not scaled.

Under `+LBAsync` HAPI flushes CUPTI without `CUPTI_ACTIVITY_FLAG_FLUSH_FORCED`
(`hapiCuptiSetAsyncLB`, `hapiCuptiFlushFlag`): a forced flush returns buffers
with incomplete records that CUPTI later completes from inside a launch on
another, still running PE, after the buffer has been freed. The cost is that
the kernels in the buffer being filled are read one round late.

## Helper thread

`LBOffThread.h`: one persistent thread per process (Linux), started on first
use and pinned by `CHARM_LB_HELPER` (or `CHARM_LB_CUPTI_OFFTHREAD`,
`CHARM_LB_STRATEGY_THREAD`). `CentralLB::InvokeLB` runs
`hapiPrepareCuptiLoads` on it and fans out `gpuLoadsReady` from a
`CcdSCHEDLOOP` poll; `CentralLB::LoadBalance` runs `strategyCompute` (work)
on it for a balancer with `strategyOffThreadOk` (MetisLB) and continues in
`strategyFinishOffThread` on the PE. On the helper, `CkMyPe()` reads 0.

## Limits

- A balancer that ends its phases by quiescence (DistributedLB's gossip)
  finishes under `+LBAsync` only once the application is quiet, such as with
  its elements stopped in `AtSyncWait()`.
- Device transport: an element with device zerocopy sends or receives in
  flight is not held back from a deferred move here; the runtime's existing
  device-channel deferral (`outstandingChannelWork`) still applies.
- The GPU trigger needs a GPU build; without one it never fires.

## Where in the source

- src/ck-core/ckmigratable.h, cklocation.C: `CkMigratable::AtSync`,
  `AtSyncStart`, `lbStart`, `lbJoinStep`, `AtSyncWait`, `AtSyncSample`,
  `AtSyncPending`, `ResumeFromSyncHelper`, `lbCheckWaitRelease`,
  `lbSetMeasurementClosed`, `recvLBPeriod`, `sampleMetaLBLoad`;
  `CkLocRec::recvMigrate`, `takeDeferredMove`, `_lbDeferredMoveHandler`;
  `CkLocMgr::emigrate`, `immigrateWithDevice` (ledger stamps and acks,
  asynchronous arrivals).
- src/ck-core/cklocation.h, cklocrec.h: migration message fields, `lbLedgerStep`.
- src/ck-ldb/LBManager.{h,C,ci}: `+LBAsync`, `CkLBSetAsync`, `resume_epoch`,
  the ledger, `+LBPeerDecision`, `+LBScatterThreshold`, `+MetaLBGpuTrigger`.
- src/ck-ldb/DistBaseLB.C: barrier hold, ledger use.
- src/ck-ldb/CentralLB.C, BaseLB.{h,ci}: scatter, helper thread.
- src/ck-ldb/MetaBalancer.{h,C,ci}: GPU sample stream and vote.
- src/ck-ldb/LBObj.{h,C}, LBDatabase.{h,C}, LBComm.{h,C}, lbdb.h: window,
  scaling.
- src/arch/cuda/hybridAPI/hapi.h, hapi_impl.cpp: joined-object exclusion,
  flush flag.
- tests/charm++/load_balancing/async_arrival.
