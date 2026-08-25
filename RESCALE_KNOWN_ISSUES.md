# Known latent issues: collectives under rescale

> Companion: `BARRIERLESS_RESCALE.md` explains the full barrier-less
> rescale pipeline these issues live in.

Status as of 2026-08-25 (was 2026-08-23), after the era-mixed gcount fix (post-rescale
structural-predicate tolerance + `CkDriveReductionsAfterRescale` end-of-restore
re-drive), and after the AMPI shrink/expand work of 2026-08-25 (R2b, R6b, R7).
Entries marked FIXED are fixed; the rest are documented, not fixed. Ordered by
severity within each section.

## Background

The no-restart cut (longjmp rescale) preserves survivor group state across the
transport rebuild, so reduction and broadcast bookkeeping spans the cut.
Reduction trees are rebuilt (`rebuildTreeForRescale`), array-manager counters
rebased (`rebaseCountersForRescale`, called from `CkArray`), and a round left
completable by the tree shrink is re-driven at the end of the restore. The
first post-rescale round at the root completes on the structural predicate
(all current-tree children reported + all locals in) instead of the era-mixed
gcount arithmetic. Everything below is what that machinery does NOT cover.

## Reduction

### R1. Interior doomed PE holding a partial (FIXED 2026-08-24 -- doomed-exit flush + generation stamp)

The original issue: at >=8 PEs a doomed PE can be an interior reduction-tree
node holding child A's subtree contribution while waiting on child B; that
partial was state, not a message -- invisible to the transport flush, dying
at the cut. Verified live on the first 8->7 interior-hole shrink campaign
(TopoTree at 8 PEs: PE 4 is interior with kids 5,6,7).

Fix, three parts (all landed):

1. **Doomed-exit flush** -- `CkReductionMgr::flushForDoomedExit`, invoked on
   departing nodes by reconverse's ConverseCleanup through
   `CmiRescaleDoomedFlushFn` before the exit flush loop. Every held message
   is forwarded to the tree parent: kid subtree messages as-is (preserving
   their `fromPE`), local contributions merged into one partial stamped with
   the doomed PE, future-round messages forwarded for the parent to
   future-queue. The manager then enters pass-through (`doomedPassThrough`):
   any straggler arriving during the exit pump is relayed upward unmerged.
   Chains of doomed PEs compose hop by hop.

2. **Generation stamp** -- `CkReductionMsg::worldGen`. PE numbering changes
   at every commit (an interior hole renumbers every PE above it), so a
   `fromPE` is only meaningful within its generation. RecvMsg translates a
   previous-generation `fromPE` through the old->new survivor mapping
   (`CmiRescaleOldPeToNew`, computed from the availability vector that drove
   the cut) before the per-kid gate consults it; a doomed sender maps to -1,
   which no gate consults; messages two or more generations old are dropped.
   Without this the gate could pass without the flushed subtree (silently
   partial reduction) and the late replay then aborted.

3. **Dead-world location keys** (found by the same campaign; not reduction
   bugs but interior-hole exposures): a carried element id whose embedded
   home PE left the world is stripped to the by-index path in
   `CkArray::handleUnknown`; `CkLocCache::requestLocation` refuses to send
   to an out-of-range decoded home; buffered location-request tables
   (`bufferedIdRequests`, `bufferedLocationRequests`) are cleared at rescale
   reset; and a replayed dead-world request's `peToTell` is range-guarded on
   the home side. Each of these was an observed "Destnode 7 out of range 7"
   abort before its guard.

Verified: repeated 8->7 interior-doom shrinks clean (including runs where the
doomed-exit flush demonstrably forwarded a held subtree message), plus the
full 4-PE battery. Residual: nodegroup reductions (R4) never got any of this.

### R2. Structural predicate could mask a lost contribution (FIXED 2026-08-24: per-kid gate)

Caught live (1/25 double-shrink cycles once the evacuate-only placer changed
round timing): the original gate compared `nRemote` (which counts OLD-tree
messages) against `treeKids()` (the NEW tree), so the root could complete the
spanning round while a *surviving* kid's contribution was still in flight --
the round finished with that subtree's data missing, and the contribution
aborted with "Recv'd late remote contribution!" when it landed post-restore.

Fix: `kidsSeen` (per-round set of kid PEs whose subtree message actually
arrived, recorded in `RecvMsg` from `m->fromPE`, cleared at every round
completion). For the round with `postRescaleRound` set, completion requires
every current-tree kid to be in `kidsSeen` or inactive for the round; the
era-mixed `nRemote` count is ignored. Verified 25/25 double-shrink cycles.
Residual: a kid re-parented from a doomed interior PE whose round-N message
died with that PE still hangs the gate -- that is R1, unchanged.

`fromPE` is now initialized to -1 in `CkReductionMsg::buildNew` (it was
garbage on locally-built messages; only tree-forwarded messages stamp it).

### R2b. Structural predicate truncated rounds that were not era-mixed (FIXED 2026-08-25)

Follow-on to R2, and the cause of roughly one failed expansion in three.
`postRescaleRound` made the root complete the first round after a cut on the
structural predicate for *every* such round, not only for rounds that actually
span the cut. An element migrating during that round -- evacuation before a
shrink, the populate round after an expand -- belongs to neither PE's locals,
so the shortfall it causes is indistinguishable from era-mixed noise. The round
closed without it and its contribution then landed on a closed round:
"Recv'd late remote contribution!". Reproduced on plain Charm++ (jacobi2d) as
readily as on AMPI.

Fix: `eraMixedRound`, set in `RecvMsg` only when a message built in the
previous generation is folded into this round, and required alongside
`postRescaleRound` before the predicate may truncate. A round with no
old-world message has consistent counts, and the ordinary
`totalElements > nSources` test then correctly waits for the migrant.

### R6b. A barren PE promised not to contribute, then contributed (FIXED 2026-08-25)

The remaining share of the same failure, seen at >=6 PEs on an interior node
whose kid was the newcomer. `informParentInactive` announces "inactive from
redNo N", which the parent reads as a promise covering N and every later round.
A newcomer cannot keep that promise: it is barren only until the post-rescale
load balancing round populates it, and an element landing there contributes to
the very round the PE excused itself from.

Fix: `_rescaleReductionSettling`, a window from the restore until the end of
the load balancing round that follows it. Inside it `checkIsActive` never
declares inactivity and `sendReductionStartingToKids` prompts every kid rather
than only the quiet ones -- the optimization is simply off, which is correct by
construction at the cost of one empty message per barren PE per round.

Two placement details are load-bearing. The window must open at the very top of
`CkRecvGroupROData`: a newcomer's reduction managers are constructed inside
`CkPupGroupData`, and a barren manager announces itself the moment it exists,
so opening the window after that misses the only announcement that matters.
And the all-kids prompt must be range-guarded, because the window opens before
the trees are rebuilt and `kids[]` can still name a PE that left in this
rescale ("Destnode N out of range N").

### R7. Zero-copy PUP raced its own bookkeeping (FIXED 2026-08-25)

Not a rescale bug -- it needs only load balancing -- but it surfaces first
under the migration traffic a rescale generates, and it aborted AMPI runs
within a few thousand rounds. `zcPupIssueRgets` issued its Rgets and then
recorded them in `pendingZCOps` and `bufferedActiveRgetMsgs`. A transport that
can satisfy an Rget without going to the network -- shared memory between two
processes on one host, the ordinary case for a local run on LCI -- invokes the
completion handler from inside `CmiIssueRget`, which looked the object up in
tables that did not exist yet and aborted with "zcPupGetCompleted: object not
found". Both tables are now populated before the first Rget is issued.

### R3. Only the FIRST post-rescale round is tolerated (LOW-MEDIUM)

`postRescaleRound` clears at the first completion. If two rounds are in
flight at the cut (streamable/partial reductions, pipelined contributes with
multiple outstanding redNos), the second round's counts can be era-mixed too
-- same hang, no tolerance. Current apps contribute one round at a time.
Streamable partial merges at interior nodes across a cut are entirely
untested (`partialReduction` path in `finishReduction`).

### R4. NodeGroup reductions have no gcount rebase (UNTESTED)

`rebaseCountersForRescale` is called only from `CkArray` (ckarray.C:1795).
`CkNodeReductionMgr` gets `resetReductionForRescale` and
`driveCompletionAfterRescale`, but no counter rebase and no structural
tolerance: a node-group reduction in flight at the cut has the same era-mixed
exposure with no recovery path. No node-group reductions exist in the current
test apps.

### R5. Plain Group reductions rely on full flush-reset (BY DESIGN, sharp edge)

Non-array `CkReductionMgr`s reset via `flushStates` (redNo -> 0 on all
survivors together). Runtime-internal groups tolerate this because their
protocols restart from scratch after the cut. An *application* Group running
its own reduction across the cut would silently lose the in-flight round
(contributions discarded by the flush). Documented as a usage constraint
until someone needs it.

### R6. Contributed-then-migrated element on the spanning round (UNOBSERVED -- wedge)

An element that contributes to round N on PE A and then migrates to PE B
before the cut normally reconciles via `adjVec` (`contributorLeaving` on A /
`contributorArriving` on B). `rebaseCountersForRescale` clears `adjVec` and
sets `gcount = lcount`: after the cut, B's `lcount` includes the migrant but
B's `nContrib` does not (its contribution is travelling up the OLD tree as
remote data), so B's local gate `nContrib < lcount` can never be satisfied
for round N -- a hang. Not yet observed; requires an element to contribute
and migrate within the evacuation window. Evacuation is exactly such a
migration, so this is plausible under load. Fix direction: fold the in-flight
adjustment into the rebased counters (preserve `adj(redNo).lcount` for the
spanning round instead of clearing it), or give the local gate the same
post-rescale treatment as the remote gate.

## Broadcast

A broadcast cannot hang the job the way a stuck reduction does -- there is no
completion barrier -- so the worst case in every scenario below is a missed or
stale delivery to individual elements, or unbounded memory, not a wedge.

### B1. Newcomer broadcast-epoch misalignment (TOP broadcast risk -- expand only)

Survivors keep `bcastSendEpoch` and the `storedBcasts` circular queue
(`headEpoch`..`curMaxEpoch`) across the longjmp; a newcomer starts at epoch 0
with an empty queue. Consequences on a broadcast-active app:

- A migrant element arriving on the newcomer with `elBcastNo = N` (cluster
  epoch) makes `bringUpToDate` look up epochs the newcomer's queue has never
  seen.
- The first incoming broadcast stamped ~N inserts at offset `N - headEpoch(0)`,
  forcing the storage to resize to next-pow2(N): unbounded growth with
  cluster uptime, and the delivery/epoch checks around the empty slots are
  unaudited.

Fix direction: align the newcomer's broadcast epoch at admission -- pup
`bcastSendEpoch` and initialize `headEpoch` to the cluster's current epoch.
Same class as the already-fixed `CkSyncBarrier::curEpoch` pup. Not hit today
only because the test app broadcasts strictly after the post-restore reduction
completes, so no broadcast ever spans the cut or lands on a cold newcomer.

### B2. Broadcast in flight at the cut (believed covered; not stress-tested)

Reconverse broadcasts relay when the holding PE *processes* the message
(`CmiBcastForward` via the swapped handler), so the doomed-PE pump triggers
forwarding and the forwarded copies are flush-counted sends: no reduction-
style silent state loss. The replay of broadcasts captured in `_buffQ` during
the restore (old-world epoch stamps entering the new world) has not been
exercised under heavy broadcast traffic.

### B3. springCleaning across the rescale window (LOW)

`storedBcasts` cleaning frees broadcasts older than `oldMaxEpoch`; a periodic
cleaning firing between the rescale decision and the cut could free a
broadcast a mid-evacuation migrant still needs for catch-up. Plain LB
migration has the same exposure; the rescale window (and especially
overlapped evacuation) widens it.

### R8. Hold-boundary never released its clients (FIXED 2026-08-25)

`+rescaleholdboundary` -- the mode that removes the in-flight-at-the-cut
exposure entirely, by holding every chare through the drain and cut -- did not
work: the rescale completed and the application never took another step, so
the safe mode was not usable and every campaign ran in the risky one.

`CkSyncBarrier::resetForRescale` clears `startedAtSync` so the next AtSync
round can fire, and `startedAtSync` was the only record that a round had fired
whose clients had not yet been resumed. Once a round fires, every client's
epoch equals `curEpoch` and `atCount` is back to zero, so nothing else in the
barrier's state tells "parked in AtSync, waiting" from "idle".

Fix: `clientsAwaitingResume`, set where the round fires in `checkBarrier` and
cleared in `resumeClients()`. It survives the rescale reset, and
`resumeClientsIfHeld` consults it. With it, hold-boundary is reliable: AMPI
5/5 and 4/4 on 4-PE chains and 3/3 on a six-step 6-PE chain; the Charm++
reference 5/5 where early release gave 3/5.

### R9. Address-space randomization now affects every job (2026-08-25, BY DESIGN)

Isomalloc is in the build as of the AMPI work, so every job negotiates a global
address range -- including jobs that never create a migratable thread. A
process that joins later probes its own free region, and with randomization on
it can differ; it is then refused, correctly, with the range it found and the
range the job needs. This had been showing up as intermittent expansion
failures on plain Charm++.

Every process in a rescaling job needs `setarch $(uname -m) -R`. The examples
carry `jacobi1d_norand` / `jacobi2d_norand` wrappers because ranks are started
through ssh and do not inherit the launcher's personality. Measured on the
Charm++ reference, 4-3-4-3-4 in early-release mode: 5/6 with randomization on,
6/6 with it off.

### R10. A PE left barren by the populate round never said so (FIXED 2026-08-25)

The settling window from R6b is a promise-suppressor: while it is open, no PE
declares itself inactive. But `checkIsActive` only runs on events --
contributorLeaving, contributorArriving, round completion -- and for a
newcomer that the post-rescale load balancing round leaves with *zero*
elements, every one of those events happens inside the window. When the window
closed it had no occasion to announce, so it was neither on its parent's
inactive list (which would have got it prompted) nor contributing.

The parent, back on the ordinary `nRemote >= treeKids()` gate, waited for it
forever. The whole job froze, and the SIGUSR2 dump named it exactly: the root
at `redNo=1840 inProg=1 nContrib=4 nRemote=3 kids=4 inactive=0` while every
other PE sat at `redNo=1841` with nothing in flight.

Fix, two parts:

1. `CkReEvaluateReductionActivity()`, called where the window closes in
   `LBManager::ResumeClients`: sweep every reduction manager and give it the
   `checkIsActive` it would otherwise never get.

2. `checkAndAddToInactiveList` now prompts a newly-quiet kid whenever its
   announced round is at or before the one in progress (`red_no <= redNo`),
   and prompts it for the round the parent is actually in. Requiring the two
   to be equal lost the case where a kid's announcement is a round behind its
   parent, which is exactly what a late-populated newcomer produces.

Charm++ 6-5-4-5-6 with `+rescaleholdboundary` went from 0/3 to 3/3.

### R11. AMPI's startup depends on message order (2026-08-25, BY DESIGN)

Found by the randomized message queue added to Reconverse
(`+randomizedqueue <window>`, see `AMPI_SHRINK_EXPAND_PLAN.md` A10). With
delivery order perturbed from the first message, AMPI does not start: PEs die
in `ampiParent::init` on a null `ampiPeMgrProxy.ckLocalBranch()`, and in
`ampi::findParentAfterCreation` on a null `parentProxy[thisIndex].ckLocal()`.

AMPI brings itself up as a chain -- the `ampiPeMgr` group, the `TCharm` array,
`ampiParent` bound to it, `ampi` bound to that -- and each link reaches the
previous one synchronously through a local lookup. Nothing enforces that the
previous link exists; it holds because the creations are broadcast in order
and delivered in order. In an optimized build the assumption is not even
checked: the `CkAbort("AMPI can't find its parent!")` beside the lookup is
inside `#if CMK_ERROR_CHECKING`, so the symptom is a segmentation fault with
no diagnostic.

Not treated as a bug to fix. A bootstrap is a fixed sequence that runs once,
not concurrency, and reordering it does not model anything that can happen in
a real run. Instead the randomized window starts suspended and each layer
resumes it when its own startup is finished -- Charm++ in `_initDone`, AMPI in
`ampiInit` once `MPI_COMM_WORLD` exists -- and a rescale suspends it again for
the restore. Worth recording because it is a real constraint on the runtime:
any change that made group and array creation messages arrive out of order,
or that delivered them through a different path, would break AMPI startup in
this way.

### R12. A migrated rank can resume before its bound MPI_COMM_SELF arrives (FIXED 2026-08-25)

Seen once in three runs of AMPI 6 PE, 6-5-4-5-6, under `+randomizedqueue 8`.
A rank that the rescale's load balancing round moved to a different PE
segfaults on the far side, inside `AMPI_Migrate`:

```
removeUnimportantArrayObjsfromPeCache()   ampi.C:1213
AMPI_Migrate
AMPI_Main
```

`AMPI_Migrate` calls that function whenever `TCHARM_Migrate()` returns on a
different PE than it left. It does

```c
arrayObjs.erase(getAmpiInstance(MPI_COMM_SELF)->ckGetID().getID());
```

and `getAmpiInstance` is `comm2ampi`, which for anything but `MPI_COMM_WORLD`
resolves through `getCommStruct(comm).getProxy()[thisIndex].ckLocal()`. The
disassembly confirms the fault is the dereference immediately after
`ampiParent::comm2ampi(1000001)` returns: the rank's `MPI_COMM_SELF` element
is not local on the new PE at the moment its thread resumes.

So AMPI assumes that when `TCHARM_Migrate()` returns, every array element
bound to the rank has already arrived and been constructed. There is no null
check -- the `#if CMK_ERROR_CHECKING` guard that would have caught it is
elsewhere -- so the symptom is a bare SIGSEGV, the same shape as R11.

Reordering is what brings it out. The same campaign, run both ways:

| | Result |
|---|---|
| without `+randomizedqueue` | 6/6 |
| with `+randomizedqueue 8` | 5/6, and 2/3 in the matrix -- 2 failures in 9 |

Two failures in nine against none in six is not, by itself, a decisive count.
What makes it convincing is that the mechanism is an arrival-order dependency
and the two crashes are identical -- same instruction, different PE each time.

**Fixed** by guarding the lookup: 10/10 afterwards, with reordering on and
every run confirmed to have rescaled four times. The function is cache pruning, and an
element that has not arrived has nothing in this PE's cache to prune, so
skipping it is not a workaround for a missing wait -- there is nothing to wait
for. The other two erases are the running thread's own objects and cannot be
absent.

## Residual failures on long chains (2026-08-25, OPEN)

With R2b, R6b, R7, R8, R10 and the R9 requirement understood, campaigns pass
repeatedly:

| Configuration | Result |
|---|---|
| Charm++ 6 PE, 6-5-4-5-6, held | 3/3 (was 0/3) |
| Charm++ 4 PE, 4-3-4-3-4, held | 3/3 |
| Charm++ 4 PE, 4-3-4-3-4, early release | 5/6 (was 2/5) |
| AMPI 4 PE, 4-3-4-3-4, defaults | 3/3 |
| AMPI 4 PE, 4-3-2-3-4, defaults | 3/3 |
| AMPI 6 PE, six-step chain, defaults | 2/2 |

One thing remains.

**Early release can still end a spanning round with "ERROR! Too many
contributions at root!"**, during the expand populate round. This is R6
observed rather than hypothesised: `rebaseCountersForRescale` clears `adjVec`,
so an element that migrated while the round was in flight is reconciled
nowhere.

Partly addressed: the rebase now marks the manager `eraMixedRound` when it was
mid-round at the cut, because the partial it is holding was built against the
counters the rebase just moved and no later message would flag it. That took
the 4-PE 4-3-4-3-4 campaign in early-release mode from roughly 4/6 to 5/6 --
an improvement, on a sample too small to call it more than that. What is left
needs the send gate `BARRIERLESS_RESCALE.md` calls for, plus preserving the
spanning round's adjustment across the rebase rather than only flagging it.

Hold-boundary avoids the whole class by construction, and is what AMPI uses by
default.

Measured pass rates, five runs each, driven by
`examples/ampi/shrink_expand` and `examples/charm++/shrink_expand/jacobi2d-iter`:
AMPI 4 PE shrink+expand 4/5, AMPI 4 PE 4-3-4-3-4 5/5, Charm++ 4 PE
shrink+expand 5/5, Charm++ 4 PE 4-3-4-3-4 2/5. The comparison is the useful
part: on the deeper campaign the plain Charm++ application fails more often
than the AMPI one, so what remains is in the shared machinery.

## Test gaps that would exercise these

1. >=8 PE shrink dooming an interior reduction-tree PE (R1, R2).
2. Broadcast-every-iteration app with a barrier-less cut mid-iteration (B1, B2).
3. Expand followed immediately by broadcasts, before the first post-restore
   reduction completes (B1).
4. A node-group reduction app across a cut (R4).
5. Pipelined / streamable reductions across a cut (R3).
