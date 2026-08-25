# No-Restart Shrink/Expand Architecture

## 0. The fundamental shift

Old model: shrink/expand was implemented as "checkpoint to disk → exit all
processes → restart with the new PE count → restore from disk." That works
but pays the price of a full process tear-down + cold restart on every
rescale (seconds), and the GPU/IPC state isn't trivially survivable.

New model: **the surviving processes never exit.** A rescale becomes:

1. PE 0 negotiates the new membership with a TCP coordinator.
2. The doomed PEs `_exit(0)` cleanly.
3. Every survivor's UCX endpoints are rebuilt against the new view.
4. Every survivor does a `longjmp` back to a `setjmp` planted at the top
   of `charm_main`, re-runs `ConverseInit` (which skips a lot of "already
   done" work), and resumes the application via a stashed callback.
5. Newcomers join as fresh processes that bypass `main()` and slot
   directly into the broadcast that PE 0 sends.

Wall-clock overhead per rescale on this build is **~10 ms**, dominated by
UCX endpoint reinit (was 547 ms before the optimizations).

## 1. Components

```
                  ┌─────────────────────────────────────────────────┐
                  │                Application                       │
                  │  Charm++ chares, RO data, groups, LB, reductions │
                  └────────────────────────┬────────────────────────┘
                                           │
       ┌───────────────────┐               │           ┌───────────────────┐
       │   charmrun_ssh    │               │           │  charm_coordinator│
       │  (ssh fan-out, no │               │           │   (standalone TCP │
       │   long-lived      ├───── ssh ─────┤           │     membership    │
       │   parent)         │               │           │     service)      │
       └───────────────────┘               │           └─────────▲─────────┘
                                           │                     │
       ┌───────────────────────────────────▼─────────────────────┴────────┐
       │                       UCX machine layer                          │
       │  - LrtsInit (PMI or coord-bootstrap)                             │
       │  - per-PE worker + endpoints                                     │
       │  - LrtsExit's rescale branch (the no-restart core)               │
       │  - UcxReInitEpsFromView, UcxTreeBarrier                          │
       └──────────────────────────────────────────────────────────────────┘
```

Three independent processes:

- **Application binary** (`jacobi2d` etc.): one process per PE. Holds
  chares, runs computation.
- **charm_coordinator**: a single TCP server that maintains the canonical
  cluster membership. Knows everyone's UCX address. Drives the
  shrink/expand handshake. Built at
  `src/util/coordinator/coordinator.cpp`. Linked deliberately without
  `charmc` (no Charm runtime dependencies).
- **Launcher** (`mpirun` / `prterun` / `charmrun_ssh`): spawns the initial
  cluster, then steps out of the way. `charmrun_ssh` is the new launcher
  with no supervised-daemon mesh (so a vacated host disappearing doesn't
  tear the job down).

## 2. The coordinator wire protocol

`src/util/coordinator/protocol.h` defines a small frame format. Key flows:

**Initial rank → coord:**
```
-> REGISTER_INITIAL { nodeId, ucxAddr }      (nodeId = launcher-assigned rank)
<- REGISTER_INITIAL_REPLY { nodeId, epoch, members[] }
                                              (sent after coord collects all
                                               expected-count ranks)
```

**Newcomer → coord (expand):**
```
-> REGISTER_NEWCOMER { ucxAddr }
<- REGISTER_NEWCOMER_REPLY { epoch, members[] }   (snapshot of current cluster)
<- INTEGRATE { nodeId, epoch, members[] }         (pushed at COMMIT — final view)
```

**PE 0 driving a reconfig:**
```
-> QUERY_PENDING                              (how many newcomers are queued?)
<- QUERY_PENDING_REPLY { count }
-> COMMIT { epoch, kills[], take }
<- COMMIT_REPLY { newNodeId, epoch, killedOldIds[], added[] }
```

**Coord pushes to killed members:**
```
<- DIE { }                                    (received by doomed PEs)
```

**All-ranks barrier (shutdown path only):**
```
-> BARRIER { epoch, nodeId }
<- BARRIER_REPLY                              (after all alive ranks check in)
```
Used only on the job-shutdown path in coord-bootstrap mode (as the
substitute for PMI's `runtime_barrier` in LrtsDrainResources / LrtsExit /
LrtsCleanup). The rescale path does NOT use it — post-rescale
synchronization is `UcxTreeBarrier` over the new endpoints, which replaced
the O(N) coord TCP star (each round-trip cost ~80 ms under
Nagle + delayed-ACK before TCP_NODELAY; the tree is 2·log N UCX hops and
the coord does zero work).

Survivors learn about the reconfig via a UCX **chain-broadcast** from PE 0
(`src/arch/ucx/machine.C` `UcxReconfigChainForward` /
`UcxRecvReconfigBytes`), not via TCP — that turned the original O(N)
coord-to-each-survivor TCP fan-out into a tree-broadcast.

## 3. Bootstrap modes (initial cluster)

The UCX `LrtsInit` at `src/arch/ucx/machine.C:328` has three entry paths:

### A. Survivor restart (post-longjmp)

```c
if (_shrinkexpand_restarting) {
    *numNodes = _shrinkexpand_new_numnodes;
    *myNodeID = _shrinkexpand_my_node;
    return;     // UCX already re-initialized in ConverseCleanup's rescale branch
}
```
This is the path taken on every rescale by every survivor. `LrtsInit` is
essentially a no-op because the endpoints were already rebuilt before the
`longjmp`.

### B. Newcomer (expand)

```c
if (CmiGetArgFlagDesc(*argv, "+newcomer", ...)) {
    // Init UCX locally
    // coord::register_newcomer(...) → snapshot of current cluster
    // Build speculative eps to existing members (handshake overlaps with wait)
    // coord::await_integrate(...) → block until COMMIT pushes final view
    // Diff speculative eps against final view, create/keep/close as needed
    // UcxTreeBarrier(myNewId, newNumNodes)   ← sync with survivors
    // return
}
```

A newcomer is spawned as a normal binary invocation but with
`+newcomer +coordinator HOST:PORT +restart /dev/shm`. The `+restart` flag
flips `faultFunc = CkRestartMain` so the newcomer doesn't run `main()` —
instead it waits for the in-memory RO+group broadcast that PE 0 builds.
The same handler-registration sequence must run as on survivors to keep
handler indices aligned.

### C. Initial launch — two sub-modes

#### C1. PMI bootstrap (mpirun/prterun)

```c
runtime_init(myNodeID, numNodes);            // PMIx_Init → JOB_SIZE, rank
UcxInitEps(*numNodes, *myNodeID);            // KVS publish + fetch UCX addrs
// then register with coord for membership tracking
```

#### C2. Coord-bootstrap (charmrun_ssh) — the new alternative

```c
if (+nodeId, +numNodes, +coordinator all present) {
    _coord_bootstrap_mode = true;
    *myNodeID = arg_nodeId;
    *numNodes = arg_numNodes;
    // ucp init
    // coord::register_initial(nodeId, myUcxAddr) → REGISTER_INITIAL_REPLY
    // Build all endpoints directly from view.members[] — no PMI KVS round-trip
}
```

This bypasses PMI entirely. Useful because mpirun/prterun tear the whole
job down on any orted disconnect — incompatible with spot-instance reclaim
of vacated nodes. With charmrun_ssh + coord-bootstrap, there is no
long-lived supervised parent.

## 4. The shrink flow (8 → 6, walkthrough)

Trigger: a user-space CCS request (`./client`) calls `set_bitmap` on PE 0
with a `char[N]` vector (`1`=keep, `0`=kill).

```
Application
   │
   │ AtSync (or next LB period)
   ▼
CentralLB::ProcessAtSync  →  build stats msg
   ▼
CentralLB::Strategy → GreedyRefineCentralLB::work  →  ReceiveMigration → MigrationDone
   ▼
CentralLB::CheckForRealloc            (pending_realloc_state != NO_REALLOC)
   │
   │ Prints "Load balancer invoking charmrun to handle reallocation on pe 0"
   │ Sets _rescaleResumeCb = CkCallback(...ResumeClients, _lbmgr)  [SHRINK]
   ▼
CkArmRescaleCut(basedir, cb, avail_vector)
   │
   ▼
CkCheckpointWriteMgr::ArmRescaleCut  ← broadcast to all PEs
   │
   │ set_shrinkexpand_exit(true)     ← broadcast-set everywhere
   │ each PE's Checkpoint() prints "Shrink in progress on PE0" (PE 0 only)
   ▼
post-Checkpoint barrier  (contribute over thisgroup → SendRestartCB on PE 0)
   │
   │ Prints "Rescale snapshot (no-op) finished in ...s, sending out the cb..."
   ▼
restartCB.send()  →  CentralLB::RescaleCutArmed
   ▼
WillIbekilled (per-PE, computes new PE number)
   ▼
StartCleanup → CkCleanup → ConverseExit → ConverseCleanup → LrtsExit
   │
   ▼
   LrtsExit's shrink/expand branch  (get_shrinkexpand_exit() == true)
   │
   │ PE 0:                              Non-initiator survivors:
   │   coord::query_pending → pending     try_read_frame: DIE? → gotDie         Doomed:
   │   coord::commit(epoch, kills,        OR receive UCX reconfig chain          gotDie path
   │                 take, oldMembers,    → rebuild eps from delta               → close coord,
   │                 &view) → newView                                              → _exit(0)
   │   UcxBuildReconfigPayload                                                    [process dies]
   │   UcxReconfigTreeForward (binary-tree fanout)
   │
   ▼
   Survivor common path:
     UcxReInitEpsFromView(view, oldNum, myNode)
     UcxTreeBarrier (tree-collective UCX barrier over the NEW endpoints —
                     survivors + newcomers; the only post-rescale barrier)
   │
   ▼
   longjmp(_shrinkexpand_jmpbuf, 1)   ← skips the rest of LrtsExit
   │
   ▼
   setjmp returns nonzero in charm_main
   │
   │ _reuseRegistrationStateOnRestart = true
   │ _exitStarted = false; _mainDone = false
   │ set_shrinkexpand_exit(false)
   │ ST_RecursivePartition_clearCache() ; _topoTree = NULL
   │ rebuild argv with +restart added (if not present)
   │
   ▼
   ConverseInit runs again
     (most subsystems gate on _reuseRegistrationStateOnRestart and skip)
   │
   ▼
   _initCharm → faultFunc(_restartDir, msg)  →  CkRestartMain
   │
   ▼
   On PE 0: build in-memory broadcast (RO data + groups + nodeID + cb)
            CmiSyncBroadcastAllAndFree
   On other PEs: receive that broadcast in CkRecvGroupROData (their LrtsInit
                 already returned with _shrinkexpand_restarting==true)
   │
   ▼
   CkRecvGroupROData restores RO + groups, then invokes _rescaleResumeCb
   │
   ▼
   LBManager::ResumeClients (for shrink)
   │
   ▼
   Application resumes its iteration loop
```

That whole sequence costs about 10 ms in the optimized build. The dominant
components:

- `ep reinit` (UCX): ~6 ms
- `ConverseCommonInit`: ~1.1 ms
- coord COMMIT round-trip: 0.5 ms
- `post-register` (topology rebuild): ~1 ms
- everything else: sub-ms

## 5. The expand flow (6 → 8)

Same skeleton, with two extra moving parts:

**Newcomer registration (parallel to PE 0's COMMIT):**

```
Newcomer process                    Coordinator                     PE 0 driver
─────────────────                   ───────────                     ───────────
spawn jacobi2d +newcomer            ...                              user iteration
  +coordinator H:P +restart DIR
   │
   │ ucp_init, get_address
   │
   ▼
coord::register_newcomer(addr)  →  REGISTER_NEWCOMER (queued in pendingFds_)
   ◄ REGISTER_NEWCOMER_REPLY {snapshot}
   │
   │ Build speculative eps to current members (overlaps with PE0's RTT)
   │ Block in coord::await_integrate(...)
   ⋮
                                                                     set_bitmap triggers
                                                                     CentralLB::ProcessAtSync
                                                                     → CheckForRealloc (EXPAND)
                                                                     → CkArmRescaleCut
                                                                     → ArmRescaleCut broadcast
                                                                     → barrier → SendRestartCB
                                                                     → WillIbekilled → CkCleanup
                                                                     → ConverseExit → LrtsExit
                                                                     → coord::query_pending
                                    ◄ QUERY_PENDING ----- ...
                                    -> QUERY_PENDING_REPLY {count=N}
                                                                     → take = min(reqested, N)
                                                                     → coord::commit(kills=[],
                                                                                     take)
                                    ◄ COMMIT ----------- ...
                                    handleCommit:
                                       compact-renumber survivors
                                       append `take` newcomers
                                       send INTEGRATE to each consumed newcomer
                                       send COMMIT_REPLY to PE 0
   ◄ INTEGRATE {nodeId, members[]}
   │
   │ Diff speculative eps vs final view; create/keep/close
   │ Newcomer-side _shrinkexpand_my_node, _new_numnodes set
   │
   ▼
   UcxTreeBarrier(myNewId, newNumNodes)
   │                                                              ▼ COMMIT_REPLY arrives on PE 0
   │                                                              UcxReconfigTreeForward
   │                                                              (survivors receive delta on UCX)
   │                                                              All survivors run
   │                                                              UcxReInitEpsFromView
   │                                                              UcxTreeBarrier
   │                                                              longjmp
   │
   ▼ (synchronizes with survivors arriving at the same barrier)
   Newcomer returns from LrtsInit normally (no longjmp).
   Continues into ConverseInit → _initCharm → faultFunc==CkRestartMain
   → newcomer receives the RO+groups broadcast that PE 0 sends after its longjmp
   → its _rescaleResumeCb is the standard one (LBManager::StartLB for expand)
```

Crucial detail: the newcomer never runs `main()`. It enters `charm_main`,
but `faultFunc != NULL` (because `+restart` was injected) means
`_initCharm` calls `faultFunc(_restartDir, msg)` instead of `main()`. On
`CkRecvGroupROData` it's recognized as `rank >= _numPes` (pre-broadcast
world) and absorbs the broadcast as state.

## 6. State preservation across the longjmp

This is the bulk of the engineering. Every survivor's in-memory state must
remain valid after `longjmp` even though `ConverseInit` runs again. The
list of subsystems that needed careful gating:

| Subsystem | What needed gating |
|---|---|
| Handler table | `_reuseRegistrationStateOnRestart` skip in init.C; `CmiAssignOnce` idempotent |
| CCS handler table | `CcsInit` skips `ccsTab` reset on survivor restart (else `set_bitmap`/`realloc` handlers vanish) |
| CkReductionMgr per-group | `resetForRescale` in `CkRecvGroupROData` (stale `redNo`, in-flight contributions) |
| `LBManager::lb_in_progress` | reset in `CkRecvGroupROData`; drains `bufferRealloc` queue |
| `CentralLB` counters | `migrates_completed`, `lbdone`, `startedAtSync` reset via `flushStates` + `CkSyncBarrier::resetForRescale` |
| `CkSyncBarrier::on` | rescale longjmp skips the end-receiver; survivors stuck with `on=false` without reset |
| `CkSyncBarrier::curEpoch` | pup'd on expand so newcomer adopts cluster's epoch (else kicks stale) |
| `CkLocMgr` | clear location cache, recompute home PE, re-key local recs, `informHome` |
| `CkArray::localElems` / PE-level `array_objs` | re-key on rescale (chained-rescale UAF without this) |
| Reduction `gcount` rebase on shrink | survivor's `gcount = lcount` (else "Too many contributions at root!") |
| Reduction tree `checkIsActive` | `rebuildTreeForRescale` must call it so expand newcomer leaves with `lcount=0` inform parent |
| `CmiTimerInit` HRC epoch | skip `inithrc()` on survivor restart (else `CmiWallTimer` jumps backwards) |
| `CmiInitHwlocTopology` | skip on survivor restart (saves 13 ms) |
| Disk-checkpoint gate | `isRescale` trusts `shrinkexpand_exit` (broadcast-set), not `pending_realloc_state` (PE 0 only) |
| `_exitStarted` / `_mainDone` flags | reset after longjmp (else next exit silently dropped) |
| `_topoTree` cache | clear (it references old node count) |
| Argv | rebuild from `Cmi_argvcopy`, ensure `+restart` present |

The unifying pattern: **anything that's a one-shot at boot must check
whether this is a survivor restart and skip its work.** The gate is the
global `bool _reuseRegistrationStateOnRestart` set inside the `setjmp`
block in `charm_main`. The complement — anything that has live state that
needs *adjusting* (not re-creating) for the new topology — gets a
`resetForRescale` method invoked from `CkRecvGroupROData` once PE 0's
broadcast restores readonlies.

## 7. Tag layout (UCX) and the device-tag bit

UCX endpoints don't carry sender IDs natively; Charm encodes the source in
the message tag. The format:

```
|---- top bits: source node ID ----|---- bottom bits: message-type tag ----|
```

With `cParams.tag_sender_mask = 0` the receiver doesn't filter by sender;
it gets it from the matched tag. Across a rescale, source node IDs change.
The pre-rescale receive tags don't decode correctly post-rescale, so all
preposted receives are reposted with the new id layout
(`UcxPrepostRxBuffers`).

## 8. The startup banner

Survivor's `_initCharm` runs again and *would* print the Charm banner,
group registration prints, etc. These are gated by
`_reuseRegistrationStateOnRestart` so post-rescale logs look like a brief
restart banner rather than a full boot. The output you see post-rescale
(`Number of PE: N -> N`, `CkPupPerPlaceData sizing/packing`) is from the
explicit rescale-status prints, not from `_initCharm`'s normal boot
sequence.

## 9. Performance characteristics (current)

Optimized 6-PE shrink, on the local test:

```
total = 0.011150 s
   orchestration   (cb -> ConverseCleanup)            : 0.000104 s
   coord COMMIT    (cleanup -> commit returned)       : 0.000554 s
   ep reinit       (commit -> UcxReInitEpsFromView)   : 0.007949 s   ← dominant
   post barrier    (ep reinit -> UcxTreeBarrier)      : 0.000011 s
   to longjmp                                         : 0.000000 s
   longjmp -> init                                    : 0.000004 s
   ConverseInit                                       : 0.001225 s
   restore (in-memory)                                : 0.000495 s
   post-restore                                       : 0.000093 s
```

The ep-reinit time scales with the number of UCX endpoints to close +
create. Everything else is a roughly fixed overhead.

## 10. Launcher independence

Three options, all interoperable with the same binary:

- **mpirun / prterun**: traditional PMI bootstrap (`runtime_init`),
  supervised-daemon model. Works for single-host or static clusters.
  Daemon-loss tears the job down — unsuitable for spot reclaim of vacated
  nodes.
- **prterun (PRRTE)**: softer daemon-loss semantics than mpirun, but the
  TCP-close path is still a hard abort.
- **charmrun_ssh**: ssh fan-out, no parent process, no supervised daemons.
  Combined with coord-bootstrap (`+nodeId`/`+numNodes`/`+coordinator`),
  the ranks register only with the coord. Vacated hosts disappearing is a
  non-event for the surviving cluster.

## 11. Invariants worth knowing

- **PE 0 can never be in the kill set.** It drives the COMMIT and
  broadcasts state to newcomers.
- **All PEs must take the same branch in barrier-collapse-on-rescale
  logic** in `CmiTimerInit` — else deadlock.
- **The coord's `pendingFds_` queue is FIFO**; newcomers are integrated in
  registration order.
- **`shrinkexpand_exit` is broadcast-set in `ArmRescaleCut`** and
  reset post-longjmp in `charm_main`. It's the ground truth for "we're
  rescaling," not `pending_realloc_state` (which is PE-0-only).
- **The coord listens on `INADDR_ANY`** (`coordinator.cpp:78`) so
  reachability is purely a launcher-side concern: the launcher must
  advertise a routable IP to the ranks.
- **No disk I/O on rescale.** "Rescale snapshot (no-op)" means exactly
  that — the entire state transfer is in-memory broadcast from PE 0 to
  newcomers (and live state on survivors).

This is the whole picture. The complexity isn't in any one piece — it's
that every subsystem that has "boot-once" state had to learn the
"boot-once-per-incarnation, with the incarnation able to longjmp back to
itself" lifecycle.
