# AMPI under no-restart shrink/expand, and AMPI on Reconverse

> Companions: `SHRINK_EXPAND_ARCHITECTURE.md` (the no-restart cut),
> `BARRIERLESS_RESCALE.md` (the two arming modes),
> `RESCALE_KNOWN_ISSUES.md` (deferred collective bugs).
>
> Started 2026-08-24 as a design; both parts were implemented over
> 2026-08-24/25 on `paw-atm26-shrinkexpand`, and the document was kept as
> the running record. Sections that were predictions are marked as such,
> next to how each turned out.

## 0. The short version

Two questions, and they were not independent.

**Both are answered, and both are working.** AMPI builds and runs on
Reconverse with migratable threads, and an AMPI job changes its process
count while running: `MPI_COMM_WORLD` keeps its size, every rank keeps
its number, and the application sees only a pause inside `AMPI_Migrate`.

Where to look:

| | |
|---|---|
| What Reconverse gained, and what it cost | **B3**, **B5** |
| What shrink/expand needed on top of that | **A7** |
| Every bug fixed, in one list | **A11** |
| What is missing and what is still broken | **A12** |
| How it was measured, and how the measurement itself was wrong at first | **A7 (Measured)**, **A10** |

**Can AMPI do shrink/expand?** Structurally, yes — better than plain
Charm++, because AMPI is already over-decomposed and `MPI_COMM_WORLD`
does not change width when the PE count does. The application sees a
pause at an `AMPI_Migrate()` call and nothing else: same rank count, same
communicator, same `MPI_Comm_rank`. That is the headline result and it is
worth stating plainly, because every other elastic-MPI story (ULFM, MPI
Sessions, malleable MPI) changes the communicator and therefore the
application.

The work was expected to be in three places, and mostly was: (1)
AMPI/TCharm's boot-once state is not survivor-restart-safe, (2)
Isomalloc's globally negotiated address region is renegotiated at every
rescale and a newcomer can shrink it out from under live data, and (3)
the expand-populate round must be a barrier round. What that estimate
missed is that most of the *effort* went somewhere else entirely —
reductions spanning the cut, and a dozen pieces of runtime state that
were stale rather than wrong on the far side of the longjmp. A11 is the
full list; the majority of it fixes plain Charm++ as much as AMPI.

**What would AMPI on Reconverse take?** Much more, and that estimate held
up. The branch's build was Reconverse-only, and Reconverse had user-level
threads but no *migratable* ones: no Isomalloc, no `CthPup`, no
`CthCreateMigratable`, no isomalloc-backed heap or TLS. AMPI without
migratable threads is AMPI without load balancing, checkpointing, or
shrink/expand — i.e. not the AMPI anyone wants.

So the ordering was forced: **Isomalloc + migratable threads in
Reconverse is a prerequisite for AMPI, which is a prerequisite for AMPI
shrink/expand.** The 6.4 kLOC estimate for what had to find a home in
Reconverse turned out to be roughly 4x too pessimistic, because the
layering split (B2) let `isomalloc.C` stay where it was and compile
essentially unchanged. What Reconverse actually gained is about 950 lines
across 11 files, listed in B5.

---

# Part A — AMPI + shrink/expand

## A1. Why AMPI is the easy case (and the hard one)

The easy part. A rescale in this system is:

1. quiesce the objects at an agreed iteration,
2. let the load balancer evacuate the doomed PEs,
3. cut the transport, longjmp, rebuild, resume,
4. run one more LB round to populate newcomers.

Step 2 is *exactly* AMPI's existing `AMPI_Migrate(ampi_load_balance=sync)`
path. AMPI ranks are `TCharm` array elements, `TCharm` sets
`usesAtSync=true` (`tcharm.C:219`), and `TCHARM_Migrate()` is
`AtSync(); stop();` (`tcharm.C:435-446`). There is no new evacuation
mechanism to build.

And `MPI_COMM_WORLD` is a `CkArray` of fixed extent. Its size is `-vp N`,
not `CkNumPes()`. Shrinking from 8 PEs to 6 moves ranks; it does not
delete them. The application's world is invariant across the cut. This is
the whole reason AMPI is the right vehicle for elastic MPI.

The hard part. AMPI's per-rank state is a *user-level thread with a live
stack and a private heap at a fixed virtual address*. Everything that
makes that work — Isomalloc, PIEglobals, TLS swapping — assumes the
process's virtual address layout is negotiated once at job start and
never changes. A rescale re-runs `ConverseInit` and admits processes that
were not part of that negotiation.

## A2. The application contract

*As designed. The signature changed slightly in the build — the flag is
an out-parameter, so the call works from Fortran and from the funcptr
shims. See A7 for what shipped.*

```c
for (int iter = 0; ; iter++) {
    compute();
    exchange();                       /* MPI_Isend/Irecv/Waitall */

    if (AMPI_Rescale_check(iter))     /* new; wraps CkRescaleCheck() */
        AMPI_Migrate(lb_sync_info);   /* existing */
}
```

`AMPI_Rescale_check(iter)` is a thin wrapper over
`CkMigratable::checkRescale(iter)` (`cklocation.C:1954`), reached through
`TCharm::get()`. Semantics are exactly as documented in
`src/ck-ldb/rescalepoint.h`: returns false at every boundary but one, at
the cost of a comparison; when it returns true the rank must quiesce and
call `AMPI_Migrate`.

The `iter` must be globally meaningful — same value at the same logical
step on every rank. For a rank-loop MPI code that is the natural
iteration counter. For codes with no such counter,
`AMPI_Rescale_armed()` (wrapping `CkRescaleArmed()`) is the weaker
fallback.

Declining to call it is safe: the rescale then lands at the job's next
ordinary `AMPI_Migrate`, which is the behaviour without the call. Cost of
forgetting is latency, not a hang.

**Constraint: boundary mode only.** `+rescalebarrierless` must be
rejected when TCharm is in the job. A barrier-less cut fires while ranks
are running, and evacuating a doomed PE then means migrating a thread
suspended mid-`MPI_Recv` — which `TCharm::pup` does not support and which
would in any case need the in-flight-at-cut guarantees that
`RESCALE_KNOWN_ISSUES.md` says are not there. Reconverse's
`ConverseCleanup` flush-to-quiescence
(`reconverse/src/rescale/rescale.cpp:420-455`) helps a great deal here —
it drains the transport until cluster-wide send/arrival counters agree —
but the counters do not cover MPI messages already *delivered into* a
doomed PE's queues for a rank that has left. Boundary mode makes the
question moot: at the cut every rank is parked in `AMPI_Migrate` and its
posted communication has completed.

## A3. What already works, unchanged

These fall out of the existing rescale machinery and need no AMPI work:

| Concern | Why it is already handled |
|---|---|
| `ampi`, `ampiParent`, `TCharm` location state | `CkLocMgr::resetForRescale` (`cklocation.C:2416`) loops over all `managers`; the three arrays are bound (`TCHARM_Attach_start` → `opts.bindTo`) and share one loc mgr |
| `localElems` / `array_objs` re-key | `CkArray::reKeyLocalElem` is called per manager in the same loop |
| Reduction state (`MPI_Allreduce`, `MPI_Barrier`) | AMPI contributes through the standard `CkReductionMgr`; `rebuildTreeForRescale`, `rebaseCountersForRescale`, `checkIsActive` all apply |
| `ampiPeMgrProxy` readonly | pup'd by `CkPupROData` in the rescale broadcast |
| `ampiPeMgr` group | `[migratable]`, pup is a no-op, `localAmpiParents` repopulates as ranks arrive |
| `Builtin_kvs` (`mype`/`numpes`/`numnodes`) | rebuilt in `ampiProcInit`, which *does* re-run on survivor restart (`init.C:1766`) |
| AMPI sequence numbers, `AmpiSeqQ`, request lists | keyed by **rank**, not PE — rescale-invariant by construction |
| `MPI_Wtime` | `TCharm::timeOffset` is per-element and only adjusted at pup; the `CmiWallTimer` epoch is already preserved across the longjmp |
| PE-local / node-local message fast paths | decided at send time from live `CkMyPe()` |

## A4. Blockers, as predicted — and how each turned out

*Written before any of it was built. Kept because the predictions were
mostly right and the two that were wrong are the interesting ones. The
detail behind each verdict is in A7.*

| Predicted blocker | Outcome |
|---|---|
| B1 `ampiNodeInit` not idempotent | **Right, and it was the whole fix.** A guard on the existing flag |
| B2 Isomalloc renegotiates its range at the cut | **Right, and worse than predicted.** A newcomer did not merely narrow the range, it joined a node reduction the running job was not in — crashing in `CmiNodeReduceHandler`. Expansion did not work at all until this was fixed |
| B3 Expand-populate must be a barrier round | **Right, but the real problem was next door.** The round is barrier-gated; what broke was the reduction *predicate* used across it (A7/A4-A5) |
| B4 `CthInit` re-runs and rebuilds the main thread | **Did not bite.** Reconverse's `CthInit` is idempotent enough on the survivor path |
| B5 PIEglobals is the only supported privatization | **Right, and still true.** `-tlsglobals` links and does nothing |
| B6 Second-order staleness | **Right in kind, wrong in list.** The things that went stale were not the ones guessed; see A11 |

Two blockers nobody predicted turned out to matter more than three that
were: the zero-copy PUP race (A7/A3), which had nothing to do with
rescale at all, and the post-rescale reduction predicate (A7/A4-A5),
which took two rounds of root-causing.


### B1. `ampiNodeInit` is not idempotent — hard abort / silent corruption

`_initCallTable.enumerateInitCalls()` runs **initnode and initproc calls
on every survivor restart** (`init.C:1766`, definition at
`init.C:1334-1344`). This is deliberate and load-bearing: Converse resets
the handler table on restart, so handlers registered by initcalls must be
re-registered in the same order or survivor and newcomer handler indices
diverge.

`TCharm::nodeInit` guards itself (`tcharm.C:66-69`). `ampiNodeInit` does
not (`ampi.C:974`). On the second incarnation:

- `CkAssert(AMPI_threadstart_idx == -1)` (`ampi.C:1058`) fires.
- `AmpiReducer = CkReduction::addReducer(...)` (`ampi.C:1056`) appends a
  *second* entry to the reducer table. `addReducer` returns
  `reducerTable().size()` (`ckreduction.C:1907-1913`) — it has no dedup.
  Survivors then hold `AmpiReducer = k+1` while a newcomer, booting
  fresh, holds `AmpiReducer = k`. Every user-defined `MPI_Op` reduction
  after that dispatches to the wrong reducer on one side of the cluster.

**Fix.** Give `ampiNodeInit` the same `static bool` guard `TCharm::nodeInit`
has, but split it: the parts that must re-run on restart (nothing, today)
from the parts that must not (all of it). Simplest correct form:

```c
static void ampiNodeInit() noexcept
{
  if (ampi_nodeinit_has_been_called) return;   // survivor restart
  ...
  ampi_nodeinit_has_been_called = true;
}
```

The flag already exists (`ampi.C:833`) and is currently only read as an
assertion helper (`ampiimpl.h:3059`).

Then audit every other initnode in the AMPI/TCharm/ROMIO link for the
same shape. The generic hazard is *any* initnode that appends to a
globally-indexed table (reducers, thread-start functions, PUPables,
chare/entry registration). This is a class of bug, not one instance.

### B2. Isomalloc region renegotiation at the cut — data loss

This is the deepest one.

`CmiIsomallocInit` is called unconditionally from `ConverseCommonInit`
(`convcore.C:4301`) and therefore re-runs on every survivor restart.
Inside `CmiIsomallocInitExtent` (`isomalloc.C:747`):

- the *probe* is skipped on survivors (`isomallocStart != nullptr`,
  `isomalloc.C:754`) — good, `IsoRegion` is preserved;
- but the *synchronization* is not. `else if (CmiNumNodes() > 1)`
  (`isomalloc.C:853`) runs a node reduction that **intersects** every
  node's region and broadcasts the result, and then unconditionally
  assigns `isomallocStart/isomallocEnd` from it (`isomalloc.C:937-941`).

On a shrink this is a no-op (survivors all carry the same region). On an
**expand it is not**: the newcomer arrives with whatever region its own
`find_largest_free_region` probe produced. If that is narrower or offset
— different kernel ASLR draw, different `ulimit`, a differently-sized
binary, a machine with a different memory map — the cluster's region
shrinks. Consequences on live data:

- `CmiIsomallocContextCreate(myunit, numunits)` (`isomalloc.C:2567`)
  partitions `[isomallocStart, isomallocEnd)` by unit. Moving the ends
  moves every future partition boundary. Existing `Mempool`s keep their
  own recorded `start`/`end`, so live allocations do not move — they just
  stop being inside the region.
- `isommap::pup` range-checks against the region (`isomalloc.C:996-1002`)
  and errors out. Any rank migrating after the expand fails to unpack.
- `CmiIsomallocInRange` starts answering false for live pointers, which
  mis-routes frees in `memory-isomalloc`.

For plain Charm++ this has been invisible: nothing uses Isomalloc. For
AMPI every rank's stack, heap **and** (under PIEglobals) code+data
segment live there.

**Fix, two halves.**

1. *Survivors skip the sync.* Gate the `CmiNumNodes() > 1` branch on
   `_reuseRegistrationStateOnRestart`, the same way `CmiTimerInit`'s
   `inithrc()` and `CmiInitHwlocTopology` are gated. Note the memory of
   this already exists as a perf item — the first-rescale isomalloc
   re-sync costs ~15 ms — so this is a latency win as well as a
   correctness one.

2. *Newcomers adopt, never negotiate.* The cluster's `IsoRegion` must
   reach a newcomer **before** `ConverseCommonInit` runs, which is before
   `_initCharm` and therefore before the RO/group broadcast. The only
   channel that early is the coordinator handshake. Concretely: PE 0
   publishes `IsoRegion` to the coordinator at `REGISTER_INITIAL` time;
   `REGISTER_NEWCOMER_REPLY` / `INTEGRATE` carry it back; the newcomer
   stashes it in a global that `CmiIsomallocInitExtent` consumes instead
   of probing, and **aborts the join** (rather than the job) if it cannot
   `mmap` that range. This is the `__FAULT__ CmiIsomallocRestart` path
   (`isomalloc.C:810-836`, which reads a `.isomalloc` file on a shared
   FS) generalized to an in-band channel.

   Rejecting a newcomer that cannot map the region is the right failure
   mode: an elastic scheduler can hand back the node and try another.

### B3. The expand-populate round must be a barrier round

For expand, `_rescaleResumeCb = LBManager::StartLB()`, and
`CentralLB::StartLB()` is `thisProxy.ProcessAtSync()`
(`CentralLB.h:149`) — it builds stats and migrates *without* going
through `CkSyncBarrier`.

Today this is safe for expand by accident: `CheckForRealloc`'s early
release is gated on `pending_realloc_state == SHRINK_IN_PROGRESS`
(`CentralLB.C:1296-1305`), so on expand the clients stay held at the sync
barrier across the cut, and the populate round finds them quiescent.

For AMPI that accident must become a guarantee, because migrating a
`TCharm` element whose thread is *not* parked at `AtSync` is not
supported:

- `TCharm::pup` refuses a running thread and, when `isStopped == false`,
  skips the `activateThread`/`deactivateThread` bracket around the user
  pup routines (`tcharm.C:255-300`) — so user data would be pup'd with
  the wrong Isomalloc context active;
- `sema.size() > 0` aborts outright;
- `opts.setAnytimeMigration(false)` (`tcharm.C:686`) is only an
  optimization hint (it sets `stableLocations`), *not* an enforcement —
  `ckEmigrate` prints a warning and migrates anyway
  (`ckarray.h:368-375`).

**Fix.** Add an assertion, not a mechanism: before the populate round,
assert every migration candidate is at the barrier when any registered
object manager declares AtSync-only migration. Cheapest form is a
`TCharm::pup` `CkAbort` when `!isStopped` and we are inside a rescale
round — turning a silent corruption into a loud failure — plus making
`_rescaleHoldBoundary` implied whenever TCharm is linked.

### B4. `CthInit` re-runs and rebuilds the main thread

`CthInit(CmiMyArgv)` is called unconditionally from `ConverseRunPE`
(`machine-common-core.C:1582`). `CthBaseInit` resets
`CthCpvAccess(CthDatasize)` to 0 and `CthInit` allocates a **new**
main-thread `CthThread` (`threads.C:1689-1700`, `threads.C:642-657`).

The recovery is real but subtle, and it should be written down rather
than rediscovered: `CtvInitialize` finds `CtvOffs##v != -1` (the offsets
are `Csv`, preserved) and therefore takes the `CthRegistered(off+size)`
branch (`converse.h:1673-1677`), which grows the *current* (new main)
thread's data block back to the right size. Because initprocs re-run in
the same order, the offsets and the final `CthDatasize` come out
identical. Existing user threads keep their own already-sized `data`
blocks and are untouched.

So this works — but only as long as (a) initcalls keep re-running on
restart, and (b) no `CtvInitialize` is ever conditionalized. Both are
worth an explicit comment and a test. The old main-thread object leaks
one allocation per rescale; harmless, worth freeing.

### B5. PIEglobals is the only supported privatization

Per `doc/ampi/03-using.rst`: PiPglobals and FSglobals explicitly do not
support migration or checkpointing (`03-using.rst:472`, `:486`).
TLSglobals and PIEglobals do. Of those, **PIEglobals** is the right
default here because it puts the rank's code+data segment *inside its
Isomalloc context* (`ampi_globals_pie.C:238-300`), so globals migrate
with the thread and a newcomer needs no per-rank setup at all.

FSglobals would additionally make expand slow: `AMPI_Node_Setup` dlopens
one copy of the user binary *per rank* on every node
(`ampi_globals_fs.C:136-166`), and a newcomer runs that at join time.
With 512 ranks that is 512 `dlopen`s on the critical path of an
expand — the same shape of problem as the GPU `hapiInit` barrier.

**Action:** make `-pieglobals` the documented and tested mode for
elastic AMPI; error out at startup if a rescale-capable build is combined
with `-pipglobals`/`-fsglobals`.

Note PIEglobals also takes an Isomalloc partition of its own —
`CmiIsomallocContextCreate(numranks*2, (numranks+1)*2)`
(`ampi_globals_pie.C:257`) — indexed by *rank count*, not PE count. That
is the good news buried in B2: **the Isomalloc partitioning scheme is
already rescale-invariant.** `TCharm` partitions by `thisIndex` out of
`numElements+1` (`tcharm.C:203`). Neither depends on `CkNumPes()`. Only
the region *endpoints* are at risk, which is exactly what B2 fixes.

### B6. Second-order: things that become stale, not wrong

- `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` colors by
  `CmiPhysicalNodeID(CkMyPe())` / `CkMyNode()` (`ampi.C:9613-9619`). After
  a rescale the resulting communicator no longer reflects co-location.
  This is already true after any ordinary LB migration, so it is a
  documentation item, not a rescale bug.
- `TCharm::nChunks` defaults to `CkNumPes()` (`tcharm.C:79`). Because
  `nodeInit` is guarded, survivors keep the original value; a **newcomer**
  re-runs `nodeInit` and would compute a different default. Elastic AMPI
  jobs must therefore pass `+vp N` explicitly, and the newcomer spawn must
  carry the full original argv. Worth a hard check: abort a newcomer whose
  computed `nChunks` disagrees with the cluster's.
- `ampiProcInit` reconstructs `CkpvAccess(msgPool)` on every restart
  (`ampi.C:1095-1096`), leaking the previous pool. Small, but per-rescale.

## A7. What was built, and what it took

*This is the main record for Part A. A11 lists every fix in one table;
A12 lists what is missing and what is still broken.*

Implemented 2026-08-25. An AMPI job now changes its process count while
running: `MPI_COMM_WORLD` keeps its size, every rank keeps its number,
and the application sees only a pause inside `AMPI_Migrate`.

### The application contract, as built

```c
int rescaleNow = 0;
AMPI_Rescale_check(iter, &rescaleNow);   /* new */
if (rescaleNow)
    AMPI_Migrate(hints);                 /* existing: ampi_load_balance=sync */
```

`AMPI_Rescale_check(iteration, &flag)` and `AMPI_Rescale_armed(&flag)`
are in `ampi.C` and declared in `ampi_functions.h`, so the funcptr shims
and every privatization mode pick them up. They wrap
`CkMigratable::checkRescale` / `CkRescaleArmed` through the rank's TCharm
thread. Example: `examples/ampi/shrink_expand/jacobi1d.c`, a 1-D Jacobi
with a halo exchange and an `MPI_Allreduce` every iteration, which checks
its rank, world size, heap and stack after every rescale and aborts if
any of them moved.

### The five things that had to be fixed

**A1. `ampiNodeInit` ran again on every survivor (FIXED).** As predicted:
initnode calls re-run across the cut, and this one asserts
`AMPI_threadstart_idx == -1` and appends a second `AmpiReducer`. A guard
on the existing `ampi_nodeinit_has_been_called` was the whole fix.

**A2. Isomalloc renegotiated its address range at every cut (FIXED).**
Also as predicted, and worse than predicted: a newcomer does not merely
risk narrowing the range, it enters a node reduction that the running
job is not in. That crashed inside `CmiNodeReduceHandler` and corrupted
the heap — expansion did not work at all until this was fixed.
`skipSyncForRescale()` sits both rescale roles out of the collective, and
the agreed range now travels in the restore broadcast, where
`CmiIsomallocAdoptRegion` installs it on the joining process before any
context can exist. A process that cannot map the job's range is refused
with a message saying so, rather than failing later and elsewhere.

**A3. Zero-copy PUP raced its own bookkeeping (FIXED).**
`zcPupIssueRgets` issued the Rgets and *then* recorded them. A transport
that can satisfy an Rget without touching the network — shared memory
between two local processes, the common case for a local run — calls the
completion handler from inside `CmiIssueRget`, which found nothing and
aborted with `zcPupGetCompleted: object not found`. Reproducible within a
few thousand load balancing rounds, with no rescale involved. The two
tables are now populated before the first Rget goes out.

**A4. The post-rescale structural predicate truncated sound rounds
(FIXED).** The root completes the first round after a cut on "every kid
reported and all locals are in" rather than on the contributor count,
because the count is era-mixed. But an element migrating *during* that
round — evacuation before a shrink, the populate round after an expand —
is counted in neither PE's locals, and the shortfall looks exactly like
era-mixed noise. The round closed without it and its contribution then
arrived at a closed round: `Recv'd late remote contribution!`, roughly
one expansion in three, in plain Charm++ as much as in AMPI. The
predicate is now gated on `eraMixedRound`, set only when the round
actually contains a message built before the cut. With no such message
the counts are consistent and the ordinary test correctly waits for the
migrant.

**A5. A barren PE promised not to contribute, then contributed (FIXED).**
The remaining failures were an interior node closing a round without the
newcomer below it. Declaring inactivity is a promise — "I will not
contribute to this round or any later one" — and a newcomer cannot keep
it while the populate round is still moving elements toward it. The
promise is now suspended for the settling window between the restore and
the end of the load balancing round that follows it: nobody goes quiet,
parents wait for every kid, and a barren kid answers each round with an
empty contribution. The window has to open at the very top of
`CkRecvGroupROData`, before `CkPupGroupData` constructs the newcomer's
managers — a barren manager announces itself the moment it exists, and
opening the window later missed exactly that announcement.

Two smaller ones fell out of the same work: prompts to kids are now
range-guarded (the window opens before the trees are rebuilt, so `kids[]`
can still name a departed PE — `Destnode N out of range N`), and
`CkSyncBarrier::resumeClientsIfHeld` now releases clients that are
genuinely parked rather than bailing on a `startedAtSync` flag that the
rescale reset has already cleared.

### Measured

Campaigns are driven by a CCS client against a live job: shrink by
dooming the highest-numbered PE, expand by registering a newcomer and
requesting one. After every step the application must still be making
progress and its own invariant checks must still hold.

| Configuration | Result |
|---|---|
| AMPI 4 PE, 16 ranks, 4-3-4-3-4 | 5/5, 3/3 |
| AMPI 4 PE, 16 ranks, 4-3-2-3-4 | 5/5, 3/3 |
| AMPI 6 PE, 24 ranks, 6-5-4-3-4-5-6 | 3/3, 2/2 |
| Charm++ 6 PE, 6-5-4-5-6, hold-boundary | 3/3 (was 0/3) |
| Charm++ 4 PE, 4-3-4-3-4, hold-boundary | 3/3 |
| Charm++ 4 PE, 4-3-4-3-4, early release | 5/6 (was 2/5) |
| AMPI, no rescale, 20k iterations / 100 LB rounds | clean |

Re-measured 2026-08-25 with deliberate message reordering
(`+randomizedqueue 8`, A10) and with the harness fixed to require a
coordinator `COMMIT` per step. Every run below is confirmed to have
actually rescaled.

| Configuration, with reordering | Result |
|---|---|
| AMPI 4 PE, 4-3-4-3-4 | 5/5 |
| AMPI 4 PE, 4-3-2-3-4 | 3/3 |
| AMPI 6 PE, 6-5-4-5-6 | 2/3, then 5/6 -- see R12; 10/10 once fixed |
| Charm++ 4 PE, 4-3-4-3-4 | 3/3 |
| Charm++ 6 PE, 6-5-4-5-6 | 3/3 |

The one failing configuration was 6/6 without reordering, which is how
R12 was identified as an ordering dependency rather than noise. With the
null guard it is 10/10 with reordering on.

(AMPI rows are with default settings, which now means hold-boundary;
Charm++ rows are with `setarch -R`. Before the A8 work the same AMPI
rows read 4/5, 3/4 and 2/3, and the Charm++ early-release row 2/5.)

Rescale cost, PE 0, 4 PEs: shrink ~6 ms, expand ~60 ms including
newcomer integration.

The comparison row is the point: on the deepest campaign the plain
Charm++ reference application fails more often than the AMPI one. What
residual flakiness remains is in the rescale machinery both share, not in
anything AMPI added.

### A8. Debugging the intermittent failures

Picked up after the first pass left AMPI at 3-5 runs in 5 on the deeper
campaigns. Two root causes, and the second was mine.

**Hold-boundary mode had never released its clients.** `+rescaleholdboundary`
is the mode that removes the whole exposure -- it holds every chare
through the drain and the cut, so nothing is in flight when the transport
goes -- and it did not work: the rescale completed and the application
never took another step. `CkSyncBarrier::resetForRescale` clears
`startedAtSync` so the next round can fire, and `startedAtSync` was the
*only* record that a round had fired whose clients were still waiting.
Once a round fires, every client's epoch equals `curEpoch` and `atCount`
is back to zero, so nothing else in the barrier's state distinguishes
"parked in AtSync, waiting to be resumed" from "idle". A separate
`clientsAwaitingResume` flag, set when the round fires and cleared in
`resumeClients()`, survives the reset and is what
`resumeClientsIfHeld` now consults.

With that, hold-boundary is reliable: AMPI 5/5 and 4/4 on the 4-PE
chains, 3/3 on the 6-PE six-step chain, and the Charm++ reference 5/5
where the default mode gave 3/5.

**Address-space randomization now matters to plain Charm++ too.** Adding
Isomalloc to the build (Part B) gave every job a negotiated global
address range, including jobs that never create a migratable thread. A
process that joins later probes its own free region, and with
randomization on it can differ -- the A2 refusal fired on a Charm++
newcomer with `the job uses 10000000 - 5b3938000000, and only
10000000 - 5ad5de000000 is free here`. That is the check doing its job,
but the requirement is new, and it had been showing up as intermittent
expansion failures. Every process in a rescaling job needs
`setarch $(uname -m) -R`; the examples carry `jacobi1d_norand` and
`jacobi2d_norand` wrappers because ranks are started through ssh and do
not inherit the launcher's personality.

**What the two together are worth**, on the Charm++ reference running
4-3-4-3-4 in the default (early-release) mode: 2-3 in 5 before, 5/6 with
randomization on, 6/6 with it off.

**AMPI now asks to be held.** A TCharm thread can only move where it is
quiescent, which makes the early release's premise -- "nothing the
application does from here on changes the rescale" -- false for it: a
released rank resumes into MPI calls and can put a message on the wire
toward a PE that is leaving. `TCharm::procInit` therefore turns
`_rescaleHoldBoundary` on unless the command line said otherwise, and
`+rescaleearlyrelease` is the new way to say otherwise. It costs the
application the drain, a few milliseconds; the stop window is if anything
shorter, because holding skips the drain entirely (measured 2.8 ms held
against 4.2 ms released).

### A9. The hold-boundary expand stall, root-caused

The one reproducible failure left after A8 was a 6-PE Charm++ expand under
hold-boundary that left the application frozen: the populate round ran
(`migrations allowed 16 out of 16`) and no iteration followed.

Chasing it by inference went nowhere — the load balancing round completed
on every PE, `ResumeClients` fired everywhere including the newcomer, and
every barrier client was released. What settled it was the runtime's own
`SIGUSR2` state dump, taken while the job was frozen:

```
PE 0 (root):  redNo=1840 inProg=1 nContrib=4 nRemote=3 lcount=4 kids=4 inactive=0
PE 1,2,3:     redNo=1841 inProg=0 nContrib=0 nRemote=0 lcount=4 kids=0
elements everywhere: redNo=1841
```

Three of four kids had reported; the fourth never would. That fourth was
the newcomer, which the balancer had left with **zero elements** — and
`inactive=0` said it had never told the root so.

The cause was the settling window from A8's sibling fix. While the window
is open no PE declares itself barren, and `checkIsActive` only runs on
events: contributorLeaving, contributorArriving, round completion. For a
newcomer that ends the populate round with nothing, all of those happen
inside the window. When it closed, there was no occasion left to
announce, so the newcomer was neither on the root's inactive list (which
would have got it prompted for each round) nor contributing.

Fixed in two places: `CkReEvaluateReductionActivity()` sweeps every
reduction manager where the window closes, and
`checkAndAddToInactiveList` now prompts a newly-quiet kid whenever its
announced round is at or before the one in progress, for the round the
parent is actually in — requiring equality lost exactly the
one-round-behind announcement a late-populated newcomer produces.

Charm++ 6-5-4-5-6 with `+rescaleholdboundary`: 0/3 → 3/3, with no
regression anywhere else.

The general lesson is worth keeping: every reduction failure in this work
has been the same shape — a PE that owes the tree a contribution and a
parent that does not know whether to wait for it. The state dump names
that in one line, and is much faster than reasoning about the protocol.

### A10. Randomized message queues in Reconverse

Everything above was found by running campaigns and watching what broke.
That finds the bugs the machine's own timing happens to expose, and a
laptop running every PE as a thread in one process has fairly regular
timing. Classic Converse had an answer for this — `CMK_RANDOMIZED_MSGQ`
in `src/conv-core/msgq.h`, a compile-time build option that shuffles the
scheduler's ready queue — and Reconverse, being a fresh implementation,
had nothing equivalent. Neither did any branch worth checking:
`fix-anytime-migration-reduction-hang`, `fix-recordreplay` and
`recordreplay-fixes` are all still on the old Converse, so their
randomization does not apply to anything built here.

So it was added. `+randomizedqueue <window>` keeps a per-PE reordering
window in front of the scheduler's local queue: messages are pulled from
the queue into the window until it holds `window` of them, and the one
that runs next is drawn uniformly from the window rather than taken from
the head. `+randomizedqueueseed N` fixes the stream. Four properties
matter:

- **It is a runtime flag, not a build option.** The same binary runs
  perturbed or not, so a failure found under randomization can be
  re-run without it to see whether ordering was the cause.
- **The window drains rather than holds.** When the queue runs dry the
  window empties before the PE goes idle, so no message is delayed
  indefinitely and quiescence still means quiescence.
- **It says how much it actually did.** On the way out each process
  reports how many of the messages it scheduled ran out of FIFO order.
  Without that, "the flag was on" is not evidence that any order
  changed — a window only perturbs anything when it holds two messages
  at once, and how often that happens is the application's business,
  not the flag's. Jacobi2d on 4 PEs reorders about 77% of its messages;
  the AMPI Jacobi about 35%.
- **It survives a rescale.** A survivor re-enters `converseRunPe` after
  the longjmp, and `CmiGetArg*` has already consumed the flag from
  `argv` — so on the second pass the parse finds nothing. Taking that
  at face value would silently switch randomization off at exactly the
  first rescale, which is the part under test. The depth persists
  instead, the per-PE buffer is kept rather than reallocated (it may
  still hold messages that arrived before the cut), and the departing
  PE's drain in `CmiRescalePumpQueuesOnce` empties the window too — a
  PE that exits still holding messages would otherwise call itself
  quiet while holding work.

**Bootstraps are exempt, and that is the interesting part.** The first
thing randomization did was stop AMPI from starting at all: PEs died in
`ampiParent::init` with a null `ampiPeMgrProxy.ckLocalBranch()`, and in
`ampi::findParentAfterCreation` with a null `parentProxy[thisIndex]
.ckLocal()`. Both are the same shape. AMPI brings itself up as a chain —
the `ampiPeMgr` group, then the `TCharm` array, then `ampiParent` bound
to it, then `ampi` bound to that — and each link reaches the one before
it *synchronously*, through a local lookup that assumes it is already
there. Nothing enforces that; it holds because the creations are
broadcast in order and delivered in order. Reorder them and the lookups
return null. (In an optimized build they are not even checked: the
`CkAbort("AMPI can't find its parent!")` next to that lookup is inside
`#if CMK_ERROR_CHECKING`, so what you get is a segmentation fault.)

That is a genuine ordering assumption, and worth recording as one — but
it is not a bug a randomized queue can usefully hunt, because a
bootstrap is not concurrency. It is a fixed sequence, it happens once,
and perturbing it tests nothing that could happen in a real run. So the
window starts suspended and each layer resumes it when its own startup
is done: Charm++ at the end of `_initDone`, AMPI later still, in
`ampiInit` once `MPI_COMM_WORLD` exists. Suspensions nest and are
counted, because AMPI takes its suspension (in `ampiProcInit`) before
Charm++ drops the initial one. A rescale suspends at the cut in
`ConverseCleanup` and drops every suspension on the far side of the
restore, since re-running the proc-inits takes fresh ones that nothing
would ever pair with.

The seed is per PE (`seed + 7919 * (pe + 1)`), so PEs make different
choices rather than the same one, and a survivor whose PE number changed
picks up a different stream after the rescale.

**A harness defect the flag exposed, worth recording separately.** The
first campaign run under randomization came back 17/17 and it was
worthless: the AMPI rows had no `+balancer GreedyRefineCentralLB`, so
`manager_init()` never ran, the `set_bitmap` CCS handler was never
registered, and every rescale request was accepted by the socket and
dropped. The harness scored those runs as passes because it only asked
whether the application was still making progress -- which stays true
when nothing happens to it. It now also requires a new `COMMIT` in the
coordinator's log for every step, so a request that goes nowhere fails
instead of passing.

Two more came out of the same pass, both of which had been quietly
poisoning runs. The CCS and coordinator ports are now allocated per run
and checked free before launching — a socket held from a previous run
makes the next one fail to bind, which reads as a hang at the first step
rather than as the setup problem it is. And cleanup was killing the wrong
process: `BIN` is a `setarch` wrapper that exec's the real binary, so the
ranks carry *its* name and `pkill -f jacobi1d_norand` matched nothing.
Every run leaked its ranks, and a leaked rank keeps its CCS socket, which
is where the port collisions were coming from. It now kills by the real
binary name and by port.

The general lesson is worth stating plainly, because it applies to any
rescale campaign: **a harness that asks only whether the application is
still running cannot fail.** The application is always still running when
the request was dropped. Something that only a real membership change
produces has to be checked — here, a new `COMMIT` in the coordinator's
log, cross-checked against `rescale points taken N` in the application's
own output.

## A11. Every fix, in one place

Everything that had to change for an AMPI job to shrink and expand while
running. Most of it is not AMPI code: the majority of these are in the
shared rescale machinery and fix plain Charm++ as much as AMPI. The
`R`-numbers are entries in `RESCALE_KNOWN_ISSUES.md`.

**AMPI and TCharm**

| Fix | Where | What went wrong |
|---|---|---|
| `ampiNodeInit` idempotence | `ampi.C` | initnode calls re-run across the cut; this one asserted `AMPI_threadstart_idx == -1` and appended a second `AmpiReducer` |
| `AMPI_Rescale_check` / `AMPI_Rescale_armed` | `ampi.C`, `ampi_functions.h` | new API; wraps `CkMigratable::checkRescale` through the rank's TCharm thread |
| Hold-boundary is the default with TCharm | `tcharm.C` | `procInit` sets `_rescaleHoldBoundary` unless the user was explicit; a barrier-less cut cannot evacuate a thread parked mid-`MPI_Recv` |
| Migration arrival restores page protections | `tcharm.C` | `CmiIsomallocContextJustMigrated` ran only on the synchronous path, so a PIEglobals rank arriving asynchronously resumed with its own text mapped read-write. Moved to `ckJustMigrated`, the arrival hook for both paths |
| `MPI_COMM_SELF` cache prune tolerates a late element (R12) | `ampi.C` | `AMPI_Migrate` pruned the PE cache the moment a rank resumed on a new PE, dereferencing a null `ckLocal()` for an element that had not arrived |

**Isomalloc and the memory module**

| Fix | Where | What went wrong |
|---|---|---|
| Region is not renegotiated at a cut | `isomalloc.C` | `skipSyncForRescale()` sits both rescale roles out of the collective; the agreed range travels in the restore broadcast and `CmiIsomallocAdoptRegion` installs it before any context exists |
| A process that cannot map the range is refused | `isomalloc.C` | it used to fail later and somewhere else |
| ASLR is detected and announced | `isomalloc.C` | `read_randomflag()` existed and nothing called it; now consulted the first time a migratable thread is created in a multi-process job |

**Reductions** — the deepest seam, and the one that produced the most failures

| Fix | Where | What went wrong |
|---|---|---|
| `eraMixedRound` gates the structural predicate (R2b) | `ckreduction.C` | the post-cut root completed on "every kid reported" rather than on the count, which is era-mixed. An element migrating *during* that round is in neither PE's locals, and the shortfall looked identical to era-mixed noise: `Recv'd late remote contribution!`, about one expansion in three |
| Settling window suspends inactivity promises (R6b) | `ckreduction.C`, `LBManager.C` | declaring inactivity promises never to contribute again, and a newcomer cannot keep that while the populate round is still moving elements toward it. Window opens at the very top of `CkRecvGroupROData` — a barren manager announces itself the moment it exists |
| `CkReEvaluateReductionActivity` at window close (R10) | `ckreduction.C`, `LBManager.C` | a newcomer the populate round left with zero elements had every announcement opportunity *inside* the window, so it was neither on its parent's inactive list nor contributing. The parent waited forever |
| Prompt a kid whose round is behind (R10) | `ckreduction.C` | `checkAndAddToInactiveList` required the announced round to equal the parent's; a late-populated newcomer is a round behind |
| Range-guard prompts to kids | `ckreduction.C` | the window opens before the trees are rebuilt, so `kids[]` can still name a departed PE — `Destnode N out of range N` |
| `gcount` rebase on shrink survivors | `ckreduction.C` | doomed PEs' `gcount` is lost; survivors rebase or get "Too many contributions at root!" |
| Obligation-free round completion | `ckreduction.C` | ported from upstream #3939; the sibling of R10 for `lcount > 0` |
| `resetForRescale` virtual renamed | `ckreduction.C` | any group naming a method `resetForRescale` silently overrode the reduction reset and its reductions hung after the first rescale |

**The cut, the restore, and everything that was stale across it**

| Fix | Where | What went wrong |
|---|---|---|
| Zero-copy PUP records before it issues | `ckrdma.C` | `zcPupIssueRgets` issued the Rgets and *then* recorded them. A transport that satisfies an Rget without touching the network — shared memory between local processes — calls the completion handler from inside `CmiIssueRget`, which found nothing: `zcPupGetCompleted: object not found`. Reproducible within a few thousand LB rounds, no rescale involved |
| `_bufferHandler` at the longjmp landing | `init.C` | the exit flow left `_charmHandlerIdx` on `_discardHandler`, correct for kill-and-restart and silently fatal here: survivors ate peer messages through the whole restore |
| Group-message stashes are drained | `ck.C`, `ckcheckpoint.C` | survivors install nothing at restore, so messages stashed on table entries sat there forever |
| `CkSyncBarrier::resetForRescale`, `clientsAwaitingResume` | `cksyncbarrier.C` | hold-boundary never released its clients; `startedAtSync` was the only record a round had fired, and the rescale reset clears it |
| `curEpoch` is pup'd on expand | `cksyncbarrier.C` | a newcomer at epoch 0 had its kicks discarded as stale; the cluster wedged on the second post-expand LB step |
| `LBManager::resetForRescale` | `LBManager.C` | `lb_in_progress` stuck, `bufferRealloc` queue undrained |
| CentralLB migration counters reset | `CentralLB.C` | `migrates_completed`/`lbdone`/`startedAtSync` persisted past the longjmp; the next ordinary LB step hung at the MigrationDone equality check |
| `CkLocMgr::resetForRescale` | `cklocation.C` | location cache, home PE recomputation, re-keyed local recs, `informHome` — without it sends went to a killed PE |
| `localElems` and `array_objs` re-key | `ckarray.C` | the loc mgr re-encodes ids on rescale; each `CkArray`'s tables must be re-keyed too, or ghost messages buffer forever. The PE-level `array_objs` hash needed it as well — that one showed up as a use-after-free on the first LB step after a *second* rescale |
| `CkArray::resetForRescale` skips `flushStates` | `ckarray.C` | survivors must keep `contributorInfo::redNo` across the longjmp |
| `CmiAssignOnce` is idempotent | `convcore.cpp` | the survivor's `_initCharm` overwrote a handler slot on each restart, and a newcomer then took a SIGSEGV in `CmiHandleMessage` |
| The wall-clock epoch survives the cut | `convcore.cpp` (reconverse) | the survivor path must not re-stamp `Cmi_startTime`. Application code and the load balancer compare `CmiWallTimer()` readings taken before a rescale against readings taken after, and re-stamping makes the clock jump backwards by however long the job has been running |
| CCS handler table survives a restart | `conv-ccs.C` | `CcsInit` reset `ccsTab`, wiping the `set_bitmap`/`realloc` handlers that drive the next rescale |
| No disk checkpoint on a rescale | `ckcheckpoint.C` | the gate trusted a PE-0-only variable, so PEs 1+ wrote Groups/arrays/NodeGroups to disk on every rescale |

**Found by deliberate message reordering** (A10)

| Fix | What went wrong |
|---|---|
| R11 — AMPI's bootstrap is order-dependent | the `ampiPeMgr` → `TCharm` → `ampiParent` → `ampi` chain reaches each previous link synchronously. Not fixed by design: a bootstrap is a fixed sequence, not concurrency, so the reordering window is suspended across it |
| R12 — `MPI_COMM_SELF` prune, above | fixed with a null guard; 6/6 without reordering, 2 failures in 9 with it, 10/10 after |

## A12. Missing features and remaining bugs

### Not implemented

- **`-tlsglobals` does not privatize on Reconverse.** It links and is
  silently a no-op: Reconverse has no TLS-segment swapping, so
  `CmiTLSCreateSegUsingPtr` / `CmiTLSSegmentSet` have no counterpart in
  `CthInterceptionsImmediateActivate`. Porting `src/util/cmitls.C` (642
  lines) is the remaining item. PIEglobals is the working mode.
- **Barrier-less rescale is rejected with TCharm in the job**, and should
  stay rejected until the transport-level send gate in
  `BARRIERLESS_RESCALE.md` exists. Evacuating a doomed PE while ranks run
  means migrating a thread suspended mid-`MPI_Recv`, which `TCharm::pup`
  does not support.
- **Fortran bindings and the `MPIX_` aliases** for `AMPI_Rescale_check` /
  `AMPI_Rescale_armed`. The C entry points exist and every privatization
  mode picks them up; the Fortran spellings were never added.
- **ROMIO / MPI-IO across a rescale is untouched.** A survivor's open file
  handles live through the longjmp because the process never exits, but a
  rank migrating *to a newcomer* carries a descriptor that does not exist
  there. Probably wants `AMPI_Migrate` to be a no-file-open point, which
  it effectively already is for checkpointing.

### Known bugs

- **`-pieglobals` does not privatize the runtime heap** (B4). `RTLD_DEEPBIND`
  on the user's PIE binary binds its `malloc` to libc instead of the
  interposing definition in the executable, so a rank's heap does not
  travel with it. Stack and globals are Isomalloc'd; the heap is not.
  A program whose per-rank state is heap and stack — like the example
  here — is fine; a program with per-rank *globals* and a migrating heap
  is not. Believed pre-existing upstream; not yet verified against a
  classic build, which is the cheap next step.
- **Early release can still end a spanning round with "Too many
  contributions at root!"** (R6). `rebaseCountersForRescale` clears
  `adjVec`, so an element that migrated while a round was in flight is
  reconciled nowhere. Hold-boundary avoids it by construction, which is
  why AMPI defaults to holding. Partly improved — the rebase now marks a
  manager era-mixed when it was mid-round at the cut — taking 4-PE
  early-release from roughly 4/6 to 5/6, a sample too small to call it
  more than that.
- **An expand with no newcomer registered wedges the job.** The commit
  admits nobody and the clients stay held at the barrier they stopped
  for. The scheduler must register the newcomer before it asks; the
  runtime should refuse the request instead.
- **AMPI's bootstrap assumes in-order message delivery** (R11). Not a bug
  that bites any real run — group and array creation messages are
  broadcast and delivered in order — but it is an undocumented constraint
  on the runtime, and in an optimized build the assumption is not checked:
  the `CkAbort("AMPI can't find its parent!")` beside the lookup is inside
  `#if CMK_ERROR_CHECKING`, so violating it gives a bare SIGSEGV.

### Untested, and most likely to bite next

- **Several reduction trees in flight at once.** R3 says only the first
  post-rescale round is tolerated. A rank-loop code doing `MPI_Allreduce`
  on `MPI_COMM_WORLD` *and* on a row communicator in the same iteration
  is exactly the untested case, and AMPI makes it easy to write. This is
  the single most likely source of a post-rescale hang.
- **`MPI_Bcast` immediately after an expand.** AMPI's broadcast goes
  through the array broadcast path, and the newcomer broadcast epoch is
  listed as an untested gap in `RESCALE_KNOWN_ISSUES.md`. AMPI would
  exercise it on the first iteration after every expand.
- **SMP.** Reconverse forces `CMK_SMP=1`. AMPI in SMP mode, plus
  Isomalloc, plus rescale is three interacting things none of which has
  been tested together. Note that the randomized-queue suspension counter
  is per PE specifically so that this case is not broken by construction.
- **Sub-communicator collectives spanning a cut** more generally. Every
  campaign here uses `MPI_COMM_WORLD` only.

### Environmental requirements, not bugs but easy to forget

- **ASLR must be off** for every process in a rescaling job (`setarch -R`,
  or `randomize_va_space=0`). A migrated thread's stack holds return
  addresses into the program's own code and Isomalloc has no say over
  where the loader put it. Isomalloc now warns the first time a migratable
  thread is created in a multi-process job. Building Reconverse statically
  into a non-PIE executable would likely remove the requirement; untried.
- **A CentralLB-family balancer must be in the job** — `set_bitmap` is
  registered by `manager_init()`, which only CentralLB and TreeLB call.
  Without it the CCS rescale request is accepted by the socket and
  dropped, with no error anywhere. This cost a whole campaign; see A10.

## A13. Phasing, as planned

*Historical. Every phase below is done except the parts of Phase 5 noted
in A12. Kept because the estimates are worth comparing against what it
actually took: Phase 1 was indeed a day, Phase 2 was the substantial one
as predicted but for a different reason (the newcomer joining a node
reduction, not the coordinator protocol), and Phase 5 is where the
unknowns were — all five of A7's fixes came out of it.*

**Phase 0 — get AMPI building again (blocked on Part B).**
Un-comment the TCharm block (`CMakeLists.txt:1094-1123`), build
`TARGET=AMPI` against whatever Converse layer exists. Success criterion:
`tests/ampi/migration` passes with ordinary load balancing. Until this
runs there is nothing to rescale.

**Phase 1 — idempotence.** B1. Guard `ampiNodeInit`; sweep every
AMPI/TCharm/ROMIO initnode for append-to-global-table shapes. Add a
debug-build assertion in `CkReduction::addReducer` and
`TCHARM_Register_thread_function` that fires if called during a survivor
restart. Test: two consecutive rescales of a trivial AMPI job with a
user-defined `MPI_Op`, checking the reducer index agrees on every rank.

**Phase 2 — Isomalloc region stability.** B2. Survivor gate first (pure
win, testable alone: assert `IsoRegion` is bit-identical before and after
a shrink). Then the coordinator-carried region for newcomers, with the
map-or-refuse failure mode. Test: expand where the newcomer is forced to
a different probe result (`CMK_MMAP_START_ADDRESS`, or a wrapper that
pre-maps a hole).

**Phase 3 — the application API.** `AMPI_Rescale_check(int)` /
`AMPI_Rescale_armed()` in `ampi.C` + `ampi.h` + `ampi_functions.h` +
Fortran bindings, plus the `MPIX_` aliases. Reject `+rescalebarrierless`
when TCharm is present. Imply `_rescaleHoldBoundary`.

**Phase 4 — quiescence guarantees.** B3. `TCharm::pup` abort on
`!isStopped` during a rescale round; confirm the expand-populate round is
barrier-gated rather than accidentally quiescent.

**Phase 5 — the real test.** Port `tests/ampi/jacobi3d` (it already has an
`AMPI_Migrate` LB step) to call `AMPI_Rescale_check` every iteration.
Run the same shrink/expand campaigns the Charm++ jacobi2d test goes
through: 8→6, 6→8, chained, interior-doomed PE.

Rough ordering weight: Phase 1 is a day. Phase 2 is the substantial one —
a week, mostly in the coordinator protocol and in convincing yourself
about the failure modes. Phases 3-4 are a few days. Phase 5 is where the
unknowns are.

## A14. Risks I would not assume away

*Written up front, before anything was built. Three of the four are
still live and have moved to A12; only the SMP one is untested rather
than known-bad.*

- **`RESCALE_KNOWN_ISSUES.md` R1/R3 under MPI collectives.** *Still live,
  still the top risk — A12.* AMPI does far
  more reduction traffic than jacobi2d, and `MPI_Allreduce` on
  sub-communicators means *several* independent reduction trees in flight.
  R3 explicitly says only the first post-rescale round is tolerated and
  that multiple in-flight rounds have no tolerance. A rank-loop code doing
  `MPI_Allreduce` on `MPI_COMM_WORLD` and on a row communicator in the
  same iteration is exactly the untested case. This is the single most
  likely source of post-rescale hangs.
- **B1 in `RESCALE_KNOWN_ISSUES.md` (newcomer broadcast epoch).** *Still
  untested — A12.* AMPI's
  `MPI_Bcast` goes through the array broadcast path. Expand + immediate
  `MPI_Bcast` is listed there as an untested gap; AMPI would exercise it
  on the first iteration after every expand.
- **ROMIO / MPI-IO.** *Still untouched — A12.* Open file handles on a survivor survive the longjmp
  (the process never exits) but a rank migrating to a newcomer carries a
  file descriptor that does not exist there. Untouched by this plan;
  probably needs `AMPI_Migrate` to be a no-file-open point, as it
  effectively already is for checkpointing.
- **SMP.** *Still untested — A12.* Reconverse forces `CMK_SMP=1` (`CMakeLists.txt:298`). AMPI in
  SMP mode plus Isomalloc plus rescale is three interacting things none of
  which has been tested together here.

---

# Part B — AMPI on Reconverse

## B0. The starting point, precisely

*As of 2026-08-24, before any of Part B. The CMake exclusions described
here have since been reversed — `src/conv-core` builds again as the
Charm++-side Isomalloc libraries, and the TCharm block is un-commented.
Kept because it is the measurement of what had to be closed.*

This branch had moved Charm++ onto Reconverse as the *only* Converse
layer:

- `cmake/converse.cmake:222-243` — the `converse` library target is
  commented out entirely; `add_custom_target(converse)` is now a pure
  dependency alias onto `reconverse`, `topomanager`, `charm_cxx_utils`,
  `ckrescale`, `conv-ccs`.
- `CMakeLists.txt:986-990` — `src/QuickThreads` only when
  `NOT RECONVERSE`; `src/conv-core` never.
- `CMakeLists.txt:1094-1123` — the entire TCharm build block is inside
  `#[[ ]]`.
- `reconverse-linux-x86_64/lib/` contains no `libmoduleampi.a`, no
  `libmoduletcharm.a`, no `libconverse.a`; the only `threads` object in
  the tree is `_deps/reconverse-build/.../src/threads.cpp.o`.

So `src/conv-core/isomalloc.C` (2788 lines), `threads.C` (2232),
`memory-isomalloc.C` (223), `global-elfgot.C` (435) and
`src/util/cmitls.C` (642) — 6.3 kLOC — are simply not compiled.

## B1. The missing surface

Every `Cth*` / `CmiIsomalloc*` symbol referenced by TCharm and AMPI,
checked against `reconverse/include/converse.h` and `reconverse/src/`:

**Present** (7): `CthThread`, `CthVoidFn`, `CthCreate`, `CthFree`,
`CthSelf`, `CthSuspend`, `CthAwaken`, `CthTraceResume`, `CmiMemoryIs`.
Also present and needed: `CthAddListener` / `CthThreadListener`,
`CthRegister` / `CthRegistered` / `CtvInitialize` (Ctv works; there is a
`tests/ctv`).

**Missing** (28):

| Group | Symbols |
|---|---|
| Migratable threads | `CthCreateMigratable`, `CthPup`, `CthMigratable`, `CthStackOffset`, `CthPointer` |
| Isomalloc core | `CmiIsomallocEnabled`, `CmiIsomallocInRange`, `CmiIsomallocRegion`, `CmiIsomallocContext` |
| Isomalloc contexts | `…ContextCreate`, `…ContextDelete`, `…ContextMalloc`, `…ContextMallocAlign`, `…ContextPermanentAllocAlign`, `…ContextProtect`, `…ContextPup`, `…ContextEnableRandomAccess`, `…ContextEnableRecording`, `…ContextJustMigrated`, `…ContextGetUsedExtent`, `…EnableRDMA`, `…GetRecordedHeap`, `…GetThreadContext` |
| Memory interception | `CmiMemoryIsomallocContextActivate` |
| Thread interceptions | `CthInterceptionsDeactivatePush/Pop`, `CthInterceptionsTemporarilyActivateStart/End` |
| Global swapping | `CtgInit` (and the whole `Ctg*` family behind `CMI_SWAPGLOBALS`) |

Structurally, `reconverse/src/threads.cpp` has no `isMigratable` field on
`CthThreadBase`, and `CthAllocateStack` ignores its `useMigratable`
argument and calls plain `malloc` (`reconverse/src/threads.cpp:205-218`).
There is no memory-wrapper layer at all — Reconverse builds do not link
`memory-*.a`.

## B2. Three ways to close it — the answer was a split

*What was built takes Option 1 for threads and Option 2 for Isomalloc,
which none of the three anticipated. Migratable threads went into
`reconverse/src/threads.cpp` because that is where threads are; Isomalloc
stayed in Charm++'s tree because it is written against PUP, which lives
above Reconverse, and the two are joined by an ops table Charm++
registers. That split is what made it cheap: `isomalloc.C` compiles
against Reconverse's `converse.h` essentially unchanged, and the only
new shape anyone had to invent was `CmiPupStream` (B5). Option 3 was
rejected on the merits and the rejection still stands.*

**Option 1 — port Isomalloc into Reconverse (recommended).**
Move `isomalloc.C`, `memory-isomalloc.C`, and the migratable half of
`threads.C` into `reconverse/src/`. Isomalloc is close to
self-contained: it needs `CmiNodeReduceStruct` / `CmiSyncNodeBroadcast`
(Reconverse has collectives), `CmiGetPageSize`, `CmiNodeAllBarrier`,
`CmiBarrier` — all present. `memory-isomalloc` needs a malloc-wrapping
hook, which Reconverse does not have and would have to acquire.

Argues for itself on two grounds: it is where the code belongs
long-term (Isomalloc is a Converse-level service), and the new
region-based Isomalloc in this tree is already the cleaner
context-per-object design rather than the old global slot map, so it
ports more or less whole.

Cost: ~3 kLOC moved plus a malloc interception layer. The interception
layer is the actual risk — `memory-isomalloc.C` is only 223 lines but it
sits under `memory.C`'s wrapper machinery, which Reconverse has replaced
with nothing.

**Option 2 — compile the legacy Converse pieces alongside Reconverse.**
Re-enable just `isomalloc.C` + `memory-isomalloc.C` + the migratable
parts of `threads.C` as a small static library that links against
Reconverse's `Cmi*` API rather than legacy Converse's. Faster to a
working build; leaves Charm++ depending on two thread implementations
(`reconverse/src/threads.cpp` for scheduling, `conv-core/threads.C` for
migration), which is not a state to stay in. Reasonable as a
Phase-0 spike to de-risk the AMPI work while Option 1 lands.

**Option 3 — migratable threads without Isomalloc.**
Reconverse uses boost `fcontext`. In principle a thread's stack could be
pup'd by copy (`CMK_THREADS_USE_STACKCOPY`) rather than by fixed address.
Rejected: it breaks every pointer into the stack, breaks
`CthStackOffset`/`CthPointer` (which TCharm's `UserData` is built on,
`tcharm_impl.h:119-123`), and breaks PIEglobals and TLSglobals entirely,
since both depend on the isomalloc'd segment. It would give AMPI
migration only for programs with no pointers into their own stacks — not
a useful subset.

## B3. Phasing for Part B — **DONE, except TLS**

Implemented and verified 2026-08-24. The layering turned out cheaper than
B2 estimated: `isomalloc.C` compiles against Reconverse's `converse.h`
essentially unchanged, so nothing had to be rewritten against a bespoke
serializer. Isomalloc stays in Charm++'s tree (where PUP lives) and
registers itself with Reconverse's thread layer through a small ops table.

1. **Isomalloc — done.** Built as a Charm++-side library (`libisomalloc`)
   from the unmodified `isomalloc.C` plus `isomalloc-reconverse.C`, the
   glue that registers `CthIsomallocOps` and adapts `CthPup(pup_er, ...)`
   onto Reconverse's serializer-agnostic `CthPupThread`. The region
   negotiation was rewritten onto Reconverse's message-shaped
   `CmiNodeReduce` (classic Converse's `CmiNodeReduceStruct` has no
   equivalent). Three primitives were added to Reconverse:
   `CmiGetPageSize`, `CmiInitMsgHeader`, `CmiMemoryIsSetFlag`.
   *Verified:* "Isomalloc> Synchronized global address space." across 2
   and 4 processes.
2. **Malloc interception — done.** `memory-os-isomalloc` and
   `memory-os-wrapper` build for Reconverse; `charmc` links the requested
   memory module again (it had been dropped from `ALL_LIBS`), and
   `CmiMemoryInit` is called from the Isomalloc init hook, since
   Reconverse has no `ConverseCommonInit` to call it from.
   *Verified:* an AMPI rank's `malloc` returns an address inside the
   Isomalloc region.
3. **Migratable threads — done.** `CthThreadBase` gained `isMigratable`,
   an Isomalloc context and an interception depth;
   `CthCreateMigratable`, `CthPupThread`, `CthStackOffset` /
   `CthPointer`, `CthMigratable`, and the four `CthInterceptions*` calls
   are implemented in `reconverse/src/threads.cpp`. Reconverse also
   gained `CmiThreadIs` with the `CMI_THREAD_IS_*` characteristics,
   `CmiPrintStackTrace`, and the no-op `Ctg*` globals-swapping interface.
   *Verified:* `tests/charm++/migratable_threads` — four threads each
   allocate from their own heap, record a stack address, suspend, are
   packed out and unpacked back in, and resume with stack address, stack
   contents and heap contents all intact.
4. **TLS — not done.** `-tlsglobals` links but does not privatize:
   Reconverse has no TLS-segment swapping, so `CmiTLSCreateSegUsingPtr` /
   `CmiTLSSegmentSet` have no counterpart in
   `CthInterceptionsImmediateActivate`. PIEglobals is the working
   privatization mode; porting `cmitls.C` is the remaining item.
5. **TCharm and AMPI — done.** The TCharm block in `CMakeLists.txt` is
   un-commented and `TARGET=AMPI` builds the whole toolchain (`ampicc`,
   funcptr shims, fs/pip/pie globals).
   *Verified:* an MPI program running 4/8/16 virtual ranks over 1/2/4
   processes with `MPI_Allreduce` and `MPI_Barrier`; ranks migrating
   through the load balancer with heap and stack preserved; and
   `tests/ampi/migration` (the upstream test, built `-pieglobals`)
   passing on 1, 2 and 4 PEs.

Header plumbing that had to change along the way, all of it because
`conv-config.h` begins with the machine layer's `conv-common.h` and a
Reconverse build has no machine layer: `memory-isomalloc.h`, `cmitls.h`,
`memory-os-wrapper.C` and `ampi.h` now reach for `conv-autoconfig.h` (and
`conv-mach-opt.h`) instead. `charm-api.h` picks up `conv-autoconfig.h`
too — without it `FTN_NAME` silently fell back to its unmangled spelling
and every Fortran entry point defined a name no caller looks for.

## B4. Bugs found on the way

### Migration arrival never restored page protections (FIXED)

`CmiIsomallocContextJustMigrated` — which replays the `mprotect` calls
recorded in the context, and so is what makes an Isomalloc'd **code**
segment executable again after it is unpacked — was called only from
`TCharm::ResumeFromSync`. That covers the synchronous path. The
asynchronous one (`TCHARM_Migrate_to` / `AMPI_Migrate_to_pe`) resumes
from `TCharm::ckJustMigrated` and never passed through it, so under
PIEglobals — where the rank's code lives in its context — the arriving
rank resumed with its own text mapped read-write and faulted on the first
instruction.

Moved to `ckJustMigrated`, which is the arrival hook for both paths and
runs before the resume; `CmiIsomallocContextJustMigrated` also grew a
null-context guard. This is not Reconverse-specific — the same gap exists
on the classic Converse path — and it is what stood between
`tests/ampi/migration` failing and passing here.

### PIEglobals does not privatize the runtime heap (OPEN)

Measured, in one AMPI rank:

| privatization | stack | globals | heap |
|---|---|---|---|
| none / `-tlsglobals` | Isomalloc | not privatized | **Isomalloc** |
| `-pieglobals` | Isomalloc | Isomalloc | **glibc arena** |

The cause is `RTLD_DEEPBIND` on the `dlopen` of the user's PIE binary
(`ampi_globals_pie.C:296-300`). The user object has `NEEDED libc.so.6`
and an undefined `malloc@GLIBC_2.2.5`; DEEPBIND puts its own dependency
chain ahead of the global scope, so `malloc` binds to libc rather than to
the interposing definition the memory module puts in the executable. A
rank's runtime heap therefore does not travel with it, and any PIEglobals
program that allocates and migrates loses its heap contents. Dropping
DEEPBIND is not the fix — pieglobals depends on it, and the load crashes
without it.

Believed pre-existing upstream rather than a consequence of this port:
the dlopen flags, the interposition mechanics and the autoconf answer for
`CMK_HAS_RTLD_DEEPBIND` are all identical on the classic path. Not
verified against a classic build, which is the next thing to do.

### Migration between processes requires ASLR off

A migrated thread's stack holds return addresses into the program's own
code, and Isomalloc has no say over where the loader puts that. With
`libreconverse.so` a shared library the addresses differ per process, and
resuming a migrated thread jumps into whatever is mapped there — a
segmentation fault with no other symptom. `setarch -R` (or
`randomize_va_space=0`) fixes it; every migration result above was
obtained that way. Building Reconverse statically into a non-PIE
executable would remove the requirement and is worth trying.

Isomalloc had a `read_randomflag()` that nothing called. It is now read
at init and consulted the first time a migratable thread is created in a
job of more than one process, which is exactly the population that can be
bitten, so the failure announces itself instead of arriving as a bare
crash.

## B5. Everything Reconverse gained

The complete surface added to Reconverse for this work, so the cost is on
the record in one place. `reconverse/` totals roughly 950 changed lines
across 11 files.

**Migratable threads** (`src/threads.cpp`, +239) — `CthThreadBase` gained
`isMigratable`, an Isomalloc context and an interception depth.

| Added | For |
|---|---|
| `CthCreateMigratable`, `CthMigratable` | creating a thread whose stack comes from Isomalloc |
| `CthPupThread`, `CthRegisterIsomallocOps`, `CthGetIsomallocContext` | serializing one, through an ops table registered by the layer that owns PUP |
| `CthStackOffset`, `CthPointer` | relocating stack-relative pointers |
| `CthInterceptionsDeactivatePush/Pop`, `CthInterceptionsTemporarilyActivateStart/End` | keeping the runtime's own allocations out of a rank's context |
| `CmiThreadIs` + `CMI_THREAD_IS_*` | the characteristics query TCharm expects |
| `CtgCreate`, `CtgInit`, `CtgInstall`, `CtgUninstall`, `CtgCurrentGlobals`, `CtgGetSize` | the globals-swapping interface, as no-ops |

**The PUP-shaped serializer.** Reconverse must not depend on PUP, which
lives above it, so `CthPupThread` takes a `CmiPupStream` — a `user`
pointer, a `bytes` callback and a mode word (`CmiPupIsSizing`,
`CmiPupIsPacking`, `CmiPupIsUnpacking`, `CmiPupIsDeleting`). Charm++'s
`isomalloc-reconverse.C` supplies the callback and recovers the real
`PUP::er` from `user`. This is the one piece of shape that had to be
invented rather than ported.

**Primitives Isomalloc needed** — `CmiGetPageSize`, `CmiInitMsgHeader`,
`CmiMemoryIsSetFlag`, `CmiPrintStackTrace`, `registerIsomallocInit` (so
`CmiIsomallocInit` runs at the right point in per-PE startup, since
Reconverse has no `ConverseCommonInit`).

**Rescale support** (`src/rescale/rescale.cpp`, `src/convcore.cpp`) —
`CmiRescaleOldPeToNew`, `CmiRescaleDoomedFlushFn`, the AM
sent/received counters (`CmiRescaleAmSent`, `CmiRescaleAmSentTo`,
`CmiRescaleAmRecv`, `CmiRescaleResetAmCounters`) that let
`ConverseCleanup` flush to genuine cluster-wide quiescence, and
`CmiRescalePumpQueuesOnce` for the departing process's drain.

**Deliberate message reordering** (`src/scheduler.cpp`, +223) —
`+randomizedqueue`, described in A10. `CmiRandomizedQueueSuspend`,
`Resume`, `ResumeAll`, `Take`, `Enabled`, `Stats`.

**Not added, and still missing** — TLS segment swapping. See A12.

## B6. What Reconverse gives back

Worth noting, because it is not all cost. Reconverse's rescale path is
materially better than the UCX one for AMPI's purposes:

- `ConverseCleanup` flushes to genuine cluster-wide quiescence — it loops
  on `sent == arrived` across two stable rounds rather than trusting
  `drain()` + barrier (`reconverse/src/rescale/rescale.cpp:420-455`).
  Eager-AM injection-vs-arrival is exactly the hazard that loses MPI
  messages at a cut.
- `CmiRescaleDoomedFlushFn` gives the layer above one chance to forward
  held state before the doomed process exits — the mechanism that fixed
  R1's interior-doomed reduction partial.
- `CmiRescaleOldPeToNew` gives a principled old→new PE translation, which
  any AMPI-level bookkeeping that caches PE numbers would need.

---

# C. Where it stands

```
  Part B (Reconverse)                     Part A (AMPI rescale)
  ───────────────────                     ─────────────────────
  B1  isomalloc.C port      DONE
  B2  malloc interception   DONE
  B3  migratable threads    DONE ─┐
  B4  TLS globals           OPEN  │  (pieglobals works; tlsglobals does not)
  B5  TCharm + AMPI build   DONE ─┴──►  A0  AMPI builds, LB works    DONE
                                        A1  initnode idempotence     DONE
                                        A2  Isomalloc region stable  DONE
                                        A3  AMPI_Rescale_check API   DONE (no Fortran)
                                        A4  quiescence guarantees    DONE (hold-boundary)
                                        A5  rescale campaign         DONE (jacobi1d, not jacobi3d)
```

The result that was worth chasing has landed: **an MPI application with
unmodified semantics changes its process count mid-run, with a fixed
`MPI_COMM_WORLD`, in a few milliseconds.** Shrink costs about 6 ms on
PE 0 at 4 PEs; expand about 60 ms, nearly all of it the newcomer's own
startup rather than anything the running job waits on.

Two claims in this document are worth reading with the caveat attached.
The campaign numbers are from a laptop running every PE as a separate
process over LCI on loopback — the failure modes exercised are the
runtime's, not the network's. And the deepest campaign is four rescale
steps on a 1-D Jacobi with one `MPI_Allreduce` per iteration; the
collective patterns most likely to break are the ones A12 lists as
untested, not the one that was tested.

What to do next, in the order I would do it:

1. **Verify the PIEglobals heap gap against a classic build.** One
   afternoon, and it decides whether that is this port's bug or
   upstream's. Everything about the mechanism says upstream.
2. **Multiple concurrent reduction trees across a cut.** The top risk in
   A12, easy to provoke — an `MPI_Allreduce` on `MPI_COMM_WORLD` and one
   on a sub-communicator in the same iteration — and cheap to add to the
   existing campaign harness.
3. **Port `cmitls.C`** to close `-tlsglobals`, or decide deliberately that
   PIEglobals is the supported mode and make `-tlsglobals` refuse rather
   than silently do nothing. The second is an hour and removes a trap.
4. **Build Reconverse statically** and see whether the ASLR requirement
   goes away. If it does, that is the single biggest usability win here.
