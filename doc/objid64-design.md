# 64-bit object id redesign: detailed design (charm #3994)

Status: draft 1, for discussion on the `objid-redesign` branch. Merging is deferred
until the reviewed-with-reconverse line is stable with users and the test suite is
more comprehensive; PR 0 of section 9 can land independently.

Draft 1, 2026-09-29 (Kale + Claude). Line references: reviewed-with-reconverse at
ebdc49c66. Companion analysis (why): doc/objid64-analysis.md. Decisions fixed by Kale
on 2026-09-29: stay at 64 bits; C = 12 build-time (AMPI may need 21 or 24); the
hashed payload is home key (width set per run from the process count and an
expand factor) plus a unique number handed out in tranches, no creator field; explicit bounds only, warn otherwise;
staged PRs; chares stay bound to individual PEs; drone mode is orthogonal.

## 0. Goals

G1 A compressible (bounded, packable) index uses 49 payload bits instead of 16.
G2 One home per element, computable from the id alone or from the index alone,
   on any PE, under the current process count; no PE number stored in the id.
G3 Delivery by id never needs the index off the source PE; no linear reverse scan.
G4 Ids are stable across checkpoint/restart including shrink/expand; directories
   are rebuilt, not restored.
G5 Local (same-PE) insertion stays synchronous; creation path unchanged in shape.
G6 Directory and location cache become per process (PR 3); per-PE tables hold only
   id -> object pointer.
G7 Every capacity limit aborts with a message naming the build flag to raise.

## 1. Id layout (src/ck-core/objid.h)

    bit 63..61  TYPE_TAG        3   (unchanged)
    bit 60..49  COLLECTION      C   default 12   -DCMK_OBJID_COLLECTION_BITS (build time)
    bit 48..0   PAYLOAD         64-3-C = 49

    PAYLOAD, packed-index kind (array has a compressor):
        the packed index, up to 49 bits (FixedArrayIndexCompressor budget)

    PAYLOAD, hashed kind (no compressor):        widths fixed PER RUN, not per build
        top H bits   HOMEKEY   H = ceil(log2(numProcs * expandFactor)), clamped to [1, 24]
        low U bits   UNIQUE    U = 49 - H; a number unique within the collection,
                               handed out in tranches (section 3.1); no creator field

Revised 2026-09-29 18:12 (Kale): the creator | counter split is gone. H is computed
at first launch from CkNumNodes() and the expand factor (+objid_expand F, default 8;
Kale's presumed ceiling on expansion), stored in the checkpoint, and reused by every
restart, because keys are embedded in live ids. If a restart has more than 2^H
processes it still works (home = key % numProcs is always defined); only the
directory spread is limited to 2^H processes, and the runtime prints a warning.

CMK_OBJID_HOME_BITS is removed. Build-time static_asserts: 1 <= C <= 40. Runtime
checks at startup and restart: H >= 1, U >= 16.

The three TYPE_TAG bits stay reserved and unused (Kale, 2026-09-29 22:33): they were
meant for a universal object handle (analysis section 7); a use may turn up in a
corner of the system, or inside this scheme, e.g. to mark a kind of payload. They
are not reclaimed for capacity.

Which kind a payload is comes from the collection: every PE's CkLocMgr branch knows
whether its array has a compressor. ObjID itself does not need a kind bit.

Capacities at the default C = 12: 4,096 collection ids (each chare array consumes
2-3: CkArray + CkLocMgr + CkLocCache, plus map/mcast if created); packed index up
to 2^49 (1D 5.6e14; 3D 65536^3 needs 48). Hashed-kind capacity per array per run:

| processes | expand 8 -> H | U | ids per array | initial tranche per process |
|---|---|---|---|---|
| 2 | 4 | 45 | 3.5e13 | 8.8e12 |
| 64 | 9 | 40 | 1.1e12 | 8.6e9 |
| 1,024 | 13 | 36 | 6.9e10 | 3.4e7 |
| 18,816 (Frontier, 2/node) | 18 | 31 | 2.1e9 | 32,768 |
| same, expand 2 | 16 | 33 | 8.6e9 | 131,072 |

The initial tranche is half of the unique space divided by the process count
(section 3.1); the other half is the allocator's pool for refills and for every
process after a restart. The small-process validation case (Kale's concern) is
solved: at 2 processes an array can hold trillions of elements. At full Frontier
scale the initial tranche is small, and its size is governed by the expand factor,
so the factor is a command-line choice, not a build constant.

Collection ids are a global counter and are not recycled today, so a program that
creates arrays in a loop reaches 4,096 after ~1,500 arrays; the existing abort in the
ObjID constructor (objid.h:56) names the flag. Recycling CkGroupID is a separate
improvement (to file).

AMPI (when supported): --ampi-only currently sets C = 29 (buildcmake:619). With
C = 24: payload 37; AMPI's own rank arrays are 1D bounded, hence packed-kind, so
the hashed kind is unused by AMPI itself, and U = 37 - H still leaves a usable
unique space at any scale. Decide the exact C when AMPI lands.

ObjID API: getCollectionID() unchanged in meaning; getElementID() returns the
49-bit payload (this is what CkArrayMessage::array_element_id() and the location
tables key on); getHomeID() is deleted (its only reader was CkLocCache::homePe,
cklocation.h:378). New: a process-wide `ck::objid::Layout {H, U}` set at startup
or restored at restart; helpers hashKey(payload) = payload >> U,
unique(payload) = payload & ((1<<U)-1), makeHashed(key, unique). Constructor abort
messages updated to name the payload split.

## 2. Packed-index kind (arrays with bounds)

Unchanged mechanism, wider budget:
- FixedArrayIndexCompressor::make (ckarrayindex.h:367): budget test becomes
  `sum > ck::ObjID::bits::PAYLOAD_BITS`. compress() bound check likewise.
- CkLocMgr::lookupID (cklocation.h:572, 597): `id = compressor->compress(idx)`.
  The home is no longer added; the two CmiAssertMsg on home/id size go.
- lookupIdx for this kind = decompress (unchanged).
- Home: homeProc(idx) = CkNodeOf(map->homePe(mapHandle, idx)); homeProc(id) =
  homeProc(decompress(id)). Directory PE (PR 1) = map->homePe(idx) exactly as today,
  so compressible arrays see no behavioural change in PR 1 beyond the wider budget --
  except the sender-repair rule of 4.3, which applies to both kinds: a forwarded delivery
  now teaches the sender the location (measured in doc/objid64-perf.md).
- checkInBounds unchanged.
- Warning (decision: no bounds inference): in CkLocMgr's constructor, if
  `compressor == nullptr` and `opts.numInitial` or `opts.end` is non-empty, print
  once per array on PE 0: "Array <gid> was sized (<dims>) but has no bounds; ids
  use the hashed scheme. Call CkArrayOptions::setBounds(...) if the index space is
  fixed." Also when bounds exist but exceed the budget: "bounds need <sum> bits,
  budget is <PAYLOAD_BITS>". Both are CkPrintf, not aborts.

## 3. Hashed kind (arrays without a usable compressor)

### 3.1 Minting (creator side, synchronous; tranches)

Each process holds, per collection, a current tranche [cur, end) with an atomic
cursor, and a slot for the next tranche. getNewObjectID(idx) (cklocation.C:2642),
non-compressible branch:

    key = ck::indexHashKey(idx)                    // section 3.2, H bits
    u   = tranche.cursor.fetch_add(1)              // any PE of the process
    if (u >= tranche.end) -> swap in the next tranche if it has arrived (CAS on the
                              slot), retry; else fall back (below)
    if (u - tranche.cur >= tranche.len/2 and no request in flight) -> request next
    id  = ObjID::makeHashed(key, u)
    insertID(idx, id)

Tranche state lives in a process-shared table (CksvAccess) keyed by CkLocMgr gid,
created under a CmiNodeLock when the CkLocMgr branch is constructed. The per-PE
`idCounter` member and its pup (cklocation.C:2454, 2515) are removed.

Unique space [0, 2^U) per collection:
- Initial region [0, 2^(U-1)): at first launch process p owns [p*S, (p+1)*S) with
  S = 2^(U-1-ceil(log2 numProcs)); no message needed, so populateInitial and any
  insertion in the array's constructor phase mint immediately on every process.
- Allocator region [2^(U-1), 2^U): handed out top-down by the allocator, which is
  the CkLocMgr branch on PE 0 (its cursor `trancheTop` is pupped there; PE 0's
  group copy is what shrink/expand restarts restore, ckcheckpoint.C:879, so the
  allocator state survives every restart consistently). Entry methods:
  `requestTranche(int proc, int wantLog2)` on PE 0, reply `grantTranche(base, len)`
  to rank 0 of proc, which installs it in the next-tranche slot. Grant size:
  max(S, requested), doubling on each successive request from the same process,
  capped so the region is not exhausted by one requester.
- After ANY restart every process discards its old tranche and takes its first one
  from the allocator: rank 0 of each process sends requestTranche during
  CkPupArrayElementsData unpacking, and the restart path waits for all grants
  before _initDone releases user code (restart already barriers there).

Refill: requested when half of the current tranche is used, so a process has half a
tranche of headroom while the request is in flight. Exhaustion before the grant
arrives (a bulk insertion loop of more than S/2 elements without returning to the
scheduler, at large scale) falls back to deferred insertion: the creation is queued
in the process's pending list and completed by grantTranche. Only a local `[inline]`
insert observes this (the element is not there when insert returns); the runtime
prints a one-time warning naming +objid_expand and the tranche size. Remote
insertions are asynchronous anyway.

Possible future mitigation (Kale, 18:22): block the inserting call until the next
tranche arrives, by running the scheduler re-entrantly or by making the allocator
reply through a path the blocked PE can poll. Complicated (re-entrancy inside an
entry method, and the allocator process may itself be inside a long entry method);
not part of this design. The deferred-insert fallback stays the v1 behaviour.

Uniqueness: u is unique within the collection by construction (disjoint tranches).
Two different indices may share a key; the key takes no part in uniqueness.

Implementation notes (PR 1b, 2026-09-30), where the code departs from or refines the
text above:
- Deferred insertion is a re-send, not a queue of half-built elements: when minting
  fails, CkLocMgr::registerNewElement returns null and CkArray::insertElement hands
  the constructor message and index to CkLocMgr::deferInsertion on that PE;
  grantTranche (rank 0 of the process) wakes every PE that deferred
  (resumeDeferredInsertions), and the PE re-runs insertElement. Demand creation and
  remote insertion go through the same entry, so they are covered.
- The fast path is lock-free: read `end` (acquire), then fetch_add the cursor; a
  tranche is installed by storing the cursor first and `end` (release) second, so a
  number taken against a stale cursor fails the bound test and is wasted, never
  reused. The slow path (swap in the spare, request, register as waiter) holds
  _nodeLock. At most one spare is held; the request goes out when half of the
  current tranche is used, and again whenever a process is exhausted.
- The allocator hands out the pool bottom up in powers of two: the requester asks
  for twice its current tranche, or for 2^16 (FIRST_GRANT_LOG2) when it has none;
  PE 0 clamps between 2^16 and region / (8 * processes). A zero-length grant means
  the pool is spent and the requester aborts with a message naming setBounds,
  +objid_expand and CMK_OBJID_COLLECTION_BITS. (Until 2026-10-10 the floor was the
  launch share, which is 8x the cap, so every grant was the cap and each restart in
  a chain of restarts spent 1/8 of the pool: the ninth aborted. Aditya's review of
  #4021.)
- Restart does not wait for grants, and does not request any: a restored CkLocMgr
  (restoredFromCheckpoint, set in its pup) starts with an empty tranche, and the
  first insertion on a process asks the allocator for 2^16 numbers and is deferred
  until the grant arrives. A process that never inserts after the restart costs the
  pool nothing. Per-process cursors are not checkpointed at all; only the allocator
  cursor is. An array created after a restart is not restored, so it takes its share
  of its own (untouched) initial region as at launch, split by the current process
  count rather than the launch count (after an expand the launch share would run
  into the allocator's half).
- `+objid_tranche_log2 N` caps every tranche at 2^N numbers so tests can see refills
  and deferral at small scale; it is pupped with the layout.

### 3.2 Index hash key

    ck::indexHashKey(const CkArrayIndex& idx) -> uint32 in [0, 2^H)
        h = splitmix64 mix over the words (nInts, dimension, index[0..nInts-1]);
        return h >> (64 - H);          // H from ck::objid::Layout

A real mixer, not CkArrayIndex::hash() (ckarrayindex.h:145, a sum of rotated words
that clusters on structured indices). Must be a pure function of the index bytes;
identical on every PE and across restarts (no PE count, no map state).

Optional map hook (later, not PR 1): `virtual uint32 CkArrayMap::homeKey(int
arrayHdl, const CkArrayIndex&)` defaulting to indexHashKey, for maps that want the
directory co-located with a placement pattern. The runtime requires only that it
is a pure function of the index and returns < 2^H.

### 3.3 Home

    homeProc(id)  = ObjID::hashKey(payload) % CkNumNodes()
    homeProc(idx) = indexHashKey(idx)        % CkNumNodes()      // identical by construction
    PR 1 directory PE: homePe(x) = CkNodeFirst(homeProc(x))      // rank 0 of the process
    PR 3: the process's shared table; requests go to the process (node queue)

CkArrayMap::homePe is NOT consulted for hashed-kind arrays (it defaulted to procNum;
only CldMap overrides it in the tree). procNum is still used for placement
(findInitialHostPe, populateInitial). This is the one user-visible semantic change
for hashed-kind arrays: the directory PE is rank 0 of a hashed process rather than
procNum(idx). The cost profile is unchanged: a source without an id already round-
trips to the home before the first send (bufferForLocation -> requestLocation(idx)).

### 3.4 Reverse lookup

lookupIdx(id) keeps two branches: compressor->decompress, or a local record
(elementNrec(id)->getIndex()). The idx2id scan and the bare CkAssert (cklocation.h:640-
648) are deleted; if neither branch applies it aborts with "index of id %llx is not
known on PE %d" -- and no delivery path may call it (section 4). Remaining callers:
deliverInline's bound-sibling demand creation (ckarray.C:1959; the sibling's record
is local, so the record branch serves) and the LB UpdateLocation path (rewritten,
section 4.4).

One scan survives, off the delivery path (PR 1a fixes, 2026-10-09):
CkLocMgr::recoverIndex(id, idx, scanAtHome). Demand creation needs the index (the
home approves by index; createhere creates at the original sender), and a message
can reach a PE by id for an element that has no record there: a deleted element
whose id and stale location the sender still held. The compressor or a local record
serve first; failing both, the home of a hashed-kind element scans its own idx -> id
bindings (kept by reclaimRemote for exactly this reason), and a non-home PE hands
the message to the home. handleUnknownByID uses it only for messages that ask for
creation; a buffer-kind message never recovers the index.

## 4. Location manager and delivery

### 4.1 One home function

    int CkLocMgr::homePe(const CkArrayIndex& idx) const   // exists; semantics per kind
    int CkLocMgr::homePe(CmiUInt8 id) const               // exists but dead; becomes live:
        compressor ? homePe(compressor->decompress(id))
                   : CkNodeFirst(ObjID::hashKey(id) % CkNumNodes())

CkLocCache gets a back-pointer `CkLocMgr* mgr` (set by CkLocMgr's constructor and by
its pup; the cache is created first, ckarray.C:653-660; bound arrays share both).
CkLocCache::homePe(id) (cklocation.h:378) becomes `mgr->homePe(id)`; the ObjID home
bits it read no longer exist. CkLocCache::requestLocation(id) (cklocation.C:2394) is
otherwise unchanged. CMK_RANK_0 wrappers are dropped (drone mode orthogonal).

### 4.2 Registration (unchanged protocol, now consistent)

createLocal -> informHome(idx, pe) -> homePe(idx); emigrate -> informHome;
reclaim -> reclaimRemote at homePe(idx); updateLocation(idx, entry) at the home
binds idx -> id (insertID) and id -> location (cache). Because homePe(id) ==
homePe(idx) for both kinds, no second registration exists and the creator holds
no special entry. reclaimRemote keeps the idx -> id binding (as today, so late
messages are not stranded) and erases the location.

### 4.2a Epochs (Kale's question, 2026-09-29 22:36)

The epoch mechanism carries over unchanged in meaning: it is a per-element migration
counter, not a property of the home. It lives in the element's location entry at its
current PE, travels in the migrate message as getEpoch(id)+1 (cklocation.C:3074), is
installed at the destination by cache->insert (:2440), and every update that reports
a migration carries the epoch of that migration; a receiver applies an update only if
its epoch is newer (:2421). Migrations of one element are serialized by the element
itself, so its epochs are totally ordered, and two informs racing to the home from
consecutive migrations resolve to the later one whatever their arrival order. None
of this depends on which PE the home is, so PR 1 keeps the code as is.

Three places need care:
- PR 3, same-process migration (section 7.3 R3): today the source runs
  recordEmigration (pe = dest, epoch++, with an assert that pe == self) AFTER sending
  the migrate message. With a shared table the destination thread can process the
  message first, so the increment-in-place and its assert are wrong. recordEmigration
  becomes an epoch-tagged compare-write {dest, e+1}, idempotent with the
  destination's insert of the same value; no in-place increment anywhere.
- CMK_GLOBAL_LOCATION_UPDATE (section 4.4): today UpdateLocation on every PE
  fabricates the epoch as its OWN cached epoch + 1 (cklocation.C:106), which is not
  the element's epoch; a bystander with a stale cache stores a low epoch and a later
  older reply can overwrite the newer location. The redesigned update must carry
  the true epoch from the emigrating PE (it has it: the same value put in the
  migrate message), as the !CMK_LBDB_ON branch at :3133 already does.
- Restart: entries are rebuilt with epoch 0 (createLocal's default) on every PE,
  which is sound because every table is empty at that point; no pre-restart epoch
  survives anywhere.
- Aditya's intra-process live-pointer fast path must bump the epoch exactly as the
  packed path does when it is cherry-picked (one compare-write of {dest, e+1}).

### 4.3 Delivery by id (ckarray.C)

recvMsg (ckarray.C:1852) cache-miss branch: replace `lookupIdx(id); handleUnknown(idx)`
with `handleUnknownByID(msg, id, type, opts)` (Aditya's e417584dd function, folded in
verbatim except that homePe(id) is the section 4.1 function). Adopt
requestLocationOnce with its pending set keyed by array_element_id(). Enable the
commented-out rule at ckarray.C:1885: after more than one hop with a stale cache
entry, forward to homePe(id) rather than chase the chain. sendMsg / handleUnknown /
bufferForLocation on the source PE keep their index-keyed shape.

Multi-hop repair (multiHop -> cache->requestLocation(id, srcPe)): now fired by the
fast delivery path for ANY forwarded message (hops >= 1; recvMsg counts a hop per
forwarding PE, a direct delivery arrives with 0), so a sender learns the location from
the first forwarded delivery, cold or one-hop stale; the explicit requestLocationOnce
of the first 1a draft is gone (PR 1b, 2026-10-01; numbers in doc/objid64-perf.md).

### 4.4 Load balancer global update (CMK_GLOBAL_LOCATION_UPDATE)

UpdateLocation (cklocation.C:90) becomes id-keyed: cache->updateLocation(entry); if a
local record or the compressor yields the index, also fire the index listeners.
Never calls lookupIdx on a PE that has no record. The entry's epoch must be the
element's true epoch, sent by the emigrating PE (section 4.2a), not each
receiver's cached epoch + 1 as today (:106).

As implemented (PR 1a fixes, 2026-10-09, after Aditya's review of #4017): the
synchronous update stays, through CkLocMgr::updateLocationFromLB(id, pe), so every
PE but the source addresses the element at its destination before the move is acted
on (the CentralLB invariant). The balancer's decision
(MigrateInfo) does not carry the element's epoch, so the entry is counted from the
receiver's cache as before; that count never exceeds the true epoch (both advance by
one per migration, and a bystander's starts no higher), so the destination's insert
still applies over it. The emigrating PE then broadcasts the entry with the true
epoch (CkLocMgr::emigrate), which closes the stale-reply window of the fabricated
count. Cost: one group broadcast per migration, in this build mode only. Carrying
the true epoch in MigrateInfo (through LDObjData) would remove the broadcast; left
for later.

### 4.5 Envelope

Unchanged in PR 1-3. The envelope's array union still carries id (payload plus
collection), the array gid, hop count and ifNotThere. (The gid duplicates the
collection bits and ifNotThere is an entry-method property; reclaiming those bytes
is noted for later, not needed at 64 bits.)

## 5. Creation, migration, deletion

- Dynamic insertion (ckInsertIdx -> findInitialHostPe -> insertElement on the host):
  unchanged. The host mints (section 3.1) or computes (section 2) the id, builds the
  record, cache entry and element table, runs the constructor, informs the home.
  `[inline]` local insertion stays synchronous.
- Demand creation: unchanged (approval through homePe(idx), which is the same PE
  the id-keyed path will consult).
- Migration: CkArrayElementMigrateMessage carries idx and id (cklocation.h:94);
  immigrate binds idx -> id before createLocal (cklocation.C:3198). Unchanged. The
  intra-process fast path on Aditya's branch (emigrateIntraProcess) is compatible:
  it carries the same fields.
- Deletion: reclaim -> reclaimRemote at homePe(idx). Unchanged.

## 6. Checkpoint, restart, shrink/expand

Ids never change: packed-kind ids are functions of the index; hashed-kind ids are
restored verbatim (restore/resume already call insertID(idx, id), cklocation.C:3320,
3332). Section cookies (ckmulticast ObjKeyList) therefore stay valid.

Layout persists: `ck::objid::Layout {H, U}` is written by PE 0 with the readonly/
group data and restored before any locmgr is unpacked; a restart never recomputes H.
If newNumProcs > 2^H the runtime warns that directory spread is limited.

Directories are rebuilt, not restored: resume/restore call createLocal with
notifyHome = true on every restart path, so each restored element registers at its
home under the NEW process count. The __FAULT__ branch of CkLocCache::pup
(cklocation.C:2347-2392) that ships remote entries is removed (it encoded the old
PE count).

Tranches: the allocator cursor `trancheTop` is part of the PE 0 CkLocMgr branch's
pup, so it is restored on every restart path (same count or shrink/expand). Every
process discards its pre-restart tranche and obtains a fresh one from the allocator
at its first insertion after the restart (section 3.1). The initial region is considered fully
consumed after the first launch; the allocator never hands out from it. This
replaces the per-PE `p | idCounter`, which under shrink/expand is restored from
PE 0's copy on every PE and can collide.

Startup/restart checks: U >= 16 (else abort naming +objid_expand and
CMK_OBJID_COLLECTION_BITS); allocator region not exhausted (abort with the same
message when a grant cannot be made).

## 7. Process-level directory and cache (PR 3)

Analysis added 2026-09-29 18:22 at Kale's request: which tables stay per PE, and how
the shared table is kept race-free.

### 7.1 Tables today (all per PE) and where they go

| table | keyed by | holds | today | PR 3 |
|---|---|---|---|---|
| CkArray element table (getEltFromArrMgr) | id | ArrayElement* | per PE, per array | **stays per PE** |
| CkpvAccess(array_objs) (ck.C:1331) | full id | ArrayElement* | per PE | **stays per PE** |
| CkLocMgr::hash | id | CkLocRec* (index, LB handle, timing) | per PE | **stays per PE** |
| CkLocCache::locMap | id | {pe, epoch} | per PE | **per process** |
| CkLocMgr::idx2id | idx | id | per PE | **per process**, plus a per-PE read cache |
| CkLocMgr::bufferedLocationRequests | idx | requesting PEs | per PE | **per process** |
| CkArray::bufferedIDMsgs / IndexMsgs / CreationMsgs | id / idx | messages | per PE | stay per PE, plus a per-process waiter set |
| tranche cursor (3.1) | gid | atomic | -- | per process |

**A separate PE-level id -> object pointer map is needed, and it is the hot path.**
Every delivered message does one lookup of the recipient id on the owning PE
(ck.C:1331, then CkArray::recvMsg -> lookup(id)). A thread-local table is read with
no synchronization at all. Putting the pointer into the shared table would make
every delivery take a shard lock or at least an atomic read of a cache line that
other cores write, on a table 64 PEs share; that is measurable at millions of
messages per second per node and buys nothing, since the pointer is only usable on
the owning PE anyway. So: delivery consults the per-PE pointer table FIRST and
touches the shared table only on a miss (element not on this PE), exactly the
order recvMsg has today. CkLocRec stays per PE with the element for the same
reason (it carries the LB handle and timing state that only the owning PE uses).

### 7.2 The shared table

One object per process per CkLocMgr, reached through a CksvAccess map keyed by
locmgr gid. Created by whichever rank constructs its CkLocMgr branch first
(double-checked under a CmiNodeLock, since group creation runs on all ranks
concurrently); refcounted by ranks; freed when the last rank's branch is deleted.

    struct SharedLoc {
      struct Shard {
        CmiNodeLock lock;                                  // pthread mutex on reconverse
        std::unordered_map<CmiUInt8, Entry> loc;           // id -> {pe, epoch}
        std::unordered_map<CkArrayIndex, CmiUInt8> idx2id; // hashed kind only
        std::unordered_map<CmiUInt8, uint64_t> waiters;    // id -> bitmask of ranks
        std::unordered_map<CkArrayIndex, std::vector<int>> pendingIdxRequests;
      } shards[NSHARDS];                                   // NSHARDS = 256, by id hash
      Shard& of(CmiUInt8 id); Shard& of(const CkArrayIndex&);
    };

Every operation takes exactly one shard lock for its duration, and does its
read-compare-write inside it. No operation holds two shard locks (the idx2id and
loc entries for one element live in the shard chosen by the id, and index-keyed
operations first map idx -> id under the idx shard, then release, then take the id
shard; a stale idx -> id binding is impossible because bindings never change).
Non-SMP: one rank, locks uncontended; identical code.

### 7.3 Writers, readers, and each race

Writers of a location entry: W1 creation (owning PE), W2 emigration (source PE:
pe = dest, epoch+1), W3 immigration (dest PE: pe = self, epoch from the message),
W4 update from a remote (home inform, multi-hop repair, LB global update, request
reply), W5 erase on death (reclaim / reclaimRemote). Readers: whichPe(id) on every
send-side miss from any rank; the home answering a request from any rank.

R1 Two W4 updates racing with different epochs. Resolved by the epoch compare being
   a locked read-compare-write ("newer epoch wins"); today's per-PE code compares
   without a lock because nothing was concurrent. Same-epoch updates are idempotent.
R2 A reader sees an entry that is about to change (element migrating). Benign and
   identical to today's stale cache: the message reaches the old PE, misses the
   per-PE pointer table, consults the shared table (now updated), forwards; the
   multi-hop repair (multiHop) then corrects the sender. No new failure mode.
R3 Intra-process migration b -> c. W2 on b and W3 on c write the same value
   {c, e+1}; order does not matter ONLY because both are epoch-tagged compare-writes:
   today's recordEmigration increments in place and asserts pe == self after the
   send, which fails if c's thread processes the migrate message first (section
   4.2a). With Aditya's live-pointer fast path (emigrateIntraProcess on his branch)
   it is one write. Messages in b's queue for the element after b's W2 take the R2
   path to c: one extra local hop, no loss.
R4 Insert or erase concurrent with a read. The map structure is only ever touched
   under the shard lock, so a reader never sees a torn bucket. (A lock-free read of
   the entry value is possible later because {pe, epoch} packs into one 64-bit
   word; not in v1 -- measure first.)
R5 Death vs a late update. W5 erases; a late W4 with an older epoch would re-create
   a ghost entry, because updateLocation today writes `locMap[id]` unconditionally
   (cklocation.C:2420). This exists per PE already. PR 3 keeps a tombstone
   {pe = -1, epoch = death epoch} so an older update is rejected by the epoch
   compare; tombstones are dropped when the idx -> id binding is dropped (never, as
   today, or by a later reclaim policy).
R6 Waiter registration vs fill. Rank b buffers a message for id and must not miss
   a fill that lands "between" its check and its registration. Both happen inside
   one critical section on the id's shard: {look up entry; if absent, set bit b in
   waiters[id]}. The filler, in its own critical section, writes the entry and
   takes the waiter bits out. It then sends one local wake message per waiting rank
   (CmiSyncSend to that PE; the filler's own rank flushes inline). A wake that
   arrives after the rank already flushed is a harmless no-op.
R7 idx2id for hashed-kind arrays is on the send path of index-addressed sends. A
   per-PE thread-local read cache (idx -> id, filled on miss from the shared shard)
   keeps the hot path unsynchronized; bindings never change, so the cache never
   goes stale. Memory: only indices this PE has sent to.
R8 Tranche cursor: fetch_add; next-tranche slot: compare-and-swap (3.1).
R9 Home requests are delivered to the process (node queue) and answered by whichever
   rank dequeues them, reading the shard. If the home has no entry yet, the request
   is parked in pendingIdxRequests / waiters of the shard (today's per-PE code has a
   TODO and drops such requests, cklocation.C:2406); the rank that later performs
   W1/W3/W4 for that id answers them from inside the same critical section's
   collected list (sends happen after the lock is released).
R10 Group-creation ordering: a request can arrive at the home process before its
    CkLocMgr branches exist. Today's lookupGroupAndBufferIfNotThere handles the
    per-PE case; node-level requests use the same buffering keyed by gid at the
    node queue level (charm's nodegroup creation path already does this).

### 7.4 What must not be shared

The LB database (per PE, registers CkLocRec handles), the reduction managers, the
per-PE buffered message queues (messages are PE-owned memory and are sent from the
PE that buffered them), and the CkMigratable_initInfo per-PE construction state.

### 7.5 Cost expectations and what to measure

Send-side miss: one shard lock per miss instead of a message per PE-miss; cache
fills once per process. Delivery hot path: unchanged (per-PE table). Migration:
one shared write per migration per process touched instead of per PE. Measure on a
64-PE node: directory requests per second before/after, whichPe latency under
contention (all ranks sending to the same 1,000 elements), and a migration-heavy
run (megatest 4x8 x3, anytime_migration) for correctness under R1-R6.

### 7.6 Shortcuts the shared table must provide (Kale, 2026-09-30)

The point of the shared table is that a PE never asks the home about anything it
could learn from its own process. Stated as guarantees, so PR 3 can be tested
against them:

S1 **Resident elements resolve locally, always.** Every element currently in the
   process has its idx -> id binding (hashed kind) and its id -> {pe, epoch} entry
   in the shared shards, written by the PE that created it (W1) or received it
   (W3) before that PE runs any entry method of it. So a send by index from any PE
   of the process to an element on another PE of the process costs: per-PE read
   cache miss, one shard read for idx -> id, one shard read for id -> pe, then
   CmiPushPE to that rank. No message leaves the process and the home is not
   consulted. Today the same send on a PE that has never seen the index goes to
   the home and back (bufferForLocation -> requestLocation(idx)), even when the
   element is one core away; objid_insert's cold-send phase measures exactly this.
   The packed kind needs no idx2id at all (compress is a pure function), so S1
   for it is the location entry alone.

S2 **An element that left the process leaves a forwarding entry.** Emigration (W2)
   writes {dest, epoch+1}; it is never erased by departure, only by death (W5,
   as a tombstone) or by a newer location. A PE of the old process that still
   addresses the element pays one hop to dest, which repairs the sender (multiHop)
   -- no trip to the home. Intra-process migration (R3) is the degenerate case:
   the forwarding entry is the final location.

S3 **One request per process, not per PE.** When no PE of the process knows the
   element, the first PE to miss sends the request and registers as a waiter; the
   others find the pending request in the shard (R6) and only register. The reply
   fills the shard once and wakes every waiting rank.

S4 **The home answers from the shard, on any rank.** A request arriving at the home
   process is answered by whichever PE dequeues it (R9); the index -> id binding
   and the location are both in the shard, so the home never needs the element's
   own PE to be the one that answers.

S5 **Same-PE delivery stays as it is.** sendMsg on the owning PE finds the element
   in the per-PE pointer table and delivers inline; the shared table is not
   touched. Chares remain PE-bound; S1 shortens the lookup, not the queue hop.

What S1 needs that the per-PE design does not have: the per-PE idx2id of the
hashed kind is today filled only by local creation, immigration and home replies;
in PR 3 the shared idx2id is filled by every creation and immigration on any rank
of the process. The per-PE read cache in front of it (R7) is fill-on-miss and
never invalidated, which is sound because a binding never changes.

### 7.7 Bounded caches (Kale, 2026-10-01)

Every fill-only table above grows with the number of distinct elements a PE or a
process has ever addressed, and a long-running program that keeps migrating and
keeps sending to new elements never stops adding entries. Today's per-PE
CkLocCache::locMap already has this property; the redesign must not make it worse
and should fix it while the tables are being rebuilt.

Classify entries as authoritative or cached. Authoritative entries are never
evicted: the id -> location and idx -> id entries of elements RESIDENT in the process
(S1 depends on them), the home-directory entries for elements whose home is this
process (the home must always answer), tranche state, and the per-PE element pointer
and record tables. Everything else is a cache that can be refetched from the home at
the cost of one request: locations of remote elements learned from replies, repairs
and forwarding entries left by departed elements, tombstones, and the whole per-PE
idx -> id read cache (R7).

Policy, cheap on the hot path: a size cap per shard for cached entries and a cap for
the per-PE read cache; when a cap is hit, drop the cached entries of that shard (or
the whole read cache) in one sweep rather than keeping LRU state per entry. A
two-generation variant (entries carry the generation they were filled in; a sweep
drops the older generation) halves the refetch burst after a sweep at the cost of
one byte per entry. Caps default to a few hundred thousand entries per process and
are runtime options (+objid_cache_cap), with a counter of sweeps in the diagnostics.

What eviction may cost: a later send to an evicted element pays one request to the
home (hashed kind) or one forwarded delivery (packed kind), which the repair rule of
4.3 then fixes again. Dropping a forwarding entry or a tombstone is safe: a message
for the element goes to the home instead of along the chain; a late, older location
update that would have been rejected by a tombstone re-creates a cached entry that
is wrong only until the next forwarded delivery repairs it, and that entry is itself
evictable. Authoritative entries are never in that position.

Measure before choosing the per-PE read cache at all: if one shard read per
index-addressed send is cheap enough on a 64-PE node, the read cache can be dropped
and only the shared table needs the cap.

## 8. Diagnostics and limits (PR 0, can land first)

- Abort in getNewObjectID when the per-PE counter would exceed the element mask
  (today: silent overflow into the home field at 65,535 per PE). PR 0 only; PR 1
  replaces the counter with tranches.
- The no-compressor note (section 2) at array creation.
- Existing collection-overflow abort (objid.h:56) keeps working; message updated.
- PR 1 adds: U >= 16 check; allocator exhaustion abort; the one-time deferred-
  insertion warning; the directory-spread warning after a large expand.

## 9. PR sequence

PR 0  Diagnostics (section 8, minus the process check) + coverage test: a migrating
      hashed-kind array (6D custom index with -DCK_ARRAYINDEX_MAXLEN=6, or
      tests/charm++/anytime_migration made unbounded), run with a balancer and with
      anytime migration; also under checkpoint/restart with a different PE count.
      Independent of the design; useful today.
PR 1  Layout (section 1), packed budget (2), hashed minting with per-process
      tranches and the PE-0 allocator (3.1-3.2), home function and cache
      back-pointer (4.1), delivery by id with e417584dd folded in (4.3), LB update
      by id (4.4), lookupIdx scan removed (3.4), persisted layout, allocator cursor
      and directory rebuild on restart (6). Home PE = rank 0
      of the hashed process for hashed-kind arrays; per-PE caches unchanged.
      Landed as PR 1a (#4017, layout/home/delivery, per-process initial tranche
      only) and PR 1b (allocator, refills, deferred insertion, restart with any
      process count).
      Tests: PR 0's test, bcastred, megatest, pingpong, ckpt tests (tests/charm++/
      chkpt, shrink/expand example), sections test after restart.
PR 2  Manual: id layout, build flags, what "compressible" means, the sizing note,
      shrink/expand guarantees.
PR 3  Process-level directory and cache (section 7). Measure: directory request
      counts and location-miss latency on a 64-PE node before/after.
PR 4  Cleanup: delete CMK_OBJID_HOME_BITS remnants, CMK_RANK_0, __FAULT__ cache
      pup, tryLookupIdx variants; optional CkArrayMap::homeKey hook.
PR 5  (optional, later) Id-carrying CkCallback variant. CkCallback's array form
      stores the index (ckcallback.h:121), which predates the object id; invoking
      it costs an index -> id step at the sender, and for a hashed-kind element the
      sending PE has never dealt with, a round trip to the home. A variant built
      where the id is known (the element itself, or a PE holding its record) sends
      by id directly; it is the first user of a kind field in the tag bits if
      callbacks to chares and groups are ever folded into the same handle.

### 9.1 Merge gate into the reviewed line (Kale, 2026-10-09)

The redesign is not merged into reviewed-with-reconverse all at once. The cut is
PR 1 as a whole: 1a (#4017, merged into objid-redesign 2026-10-09), its fixes
(#4028, from Aditya's review of 1a), and 1b (#4021), together. 1a alone leaves
hashed-kind arrays with the initial tranche only, an abort when it runs out, and an
abort on a restart with a different process count; 1b removes all three, so 1a is
never merged to the reviewed line without 1b. The gate:

1. #4028 approved by Aditya and merged into objid-redesign.
2. #4021 rebased onto that, its tests (including the restart ones) passing, merged
   into objid-redesign.
3. ChaNGa, which creates a ~262k-element bounded array and today sees the PR 0
   "needs 17 bits ... hold 16" note, rebuilt and run against objid-redesign by
   Kale or Ritvik (Tom Quinn's schedule is not to be depended on): the note gone,
   the run unchanged.
4. A full sweep, because the id layout is a deep change: every test under
   tests/charm++ and tests/converse, every example under examples/charm++ with a
   test target, the benchmarks we run (pingpong, jacobi), the ten PPL mini-apps
   listed at https://charmplusplus.org/miniApps/ (LeanMD, AMR, Barnes-Hut, DenseLU,
   HPCCG, Kripke, TS, FFT, RA, EP Stream; repositories under github.com/UIUC-PPL),
   and the applications paratreet2 and ChaNGa, each in at least single-process multi-PE and
   multi-process configurations, with a balancer where the program supports one,
   and checkpoint/restart where it does. On the Mac and on one Linux cluster
   (Anvil). CI's subset is not enough for this merge.

Then #4015 (objid-redesign -> reviewed-with-reconverse) leaves draft and merges.
PR 2 (manual) follows as a small PR straight to the reviewed line. PR 3 and PR 4
are their own PRs against the reviewed line after PR 1 is in, not passengers on
objid-redesign. A checkpoint written before PR 1 cannot be restored after it (id
layout and checkpoint format change; pupLayout's marker makes that a clear abort):
the release notes must say so.

## 10. Test matrix

| test | kind | exercises |
|---|---|---|
| pingpong, megatest, hello/1darray (fixed) | packed | wider budget, unchanged paths |
| new: 6D custom-index migration (PR 0) | hashed | minting, homePe(id), delivery by id, LB |
| anytime_bcastred | packed | multi-hop repair |
| chkpt + restart same P | both | id stability, counter restore |
| shrink/expand example, P -> P/2 and P -> 2P | both | directory rebuild, counter file, key % newN |
| sections (ckmulticast) across restart | both | stable cookies |
| tiny layout (+objid_expand huge, small U; C=4 build) | hashed | every abort, deferred-insert fallback, allocator refill |
| startupTest, hello/4darray, leanmd | hashed | the sizing note, no behaviour change |

## 11. Open questions for Aditya

Q1 Which shrink/expand path is in use on rate-aware-gpu-lb: the ckcheckpoint.C
   CkResumeRestartMain route (+shrinkexpand, Cmi_myoldpe) or something newer? Does
   the LB database survive a restart (ids held across it)?
Q2 The AMR index: custom CkArrayIndexT with how many ints (CK_ARRAYINDEX_MAXLEN)?
   (Kale, 2026-09-29 22:26: the initial insertion burst is not large and fits the
   initial tranche; it is the refinements afterwards that add chares, and those run
   between steps, where a tranche refill has time to land. So the deferred-insert
   fallback of 3.1 should stay unreached in AMR; the index width is still open.)
Q3 Does any device or LB code read ObjID home bits or assume home == creator PE?
   (Audit on the reviewed line says no; his branch adds handleUnknownByID and
   emigrateIntraProcess, both id-keyed.)
Q4 (withdrawn 2026-09-29 22:26.) The question asked about ids stored across LB
   steps; ids never change within a run, today or in this design, so that part was
   moot. The only case where an id changes today is a restart with a different PE
   count for a compressible array (analysis D4), and this design removes it. Nothing
   in application state needs auditing.
Q5 Is rank 0 of the home process as the interim directory PE (PR 1) acceptable for
   his runs, or does he need PR 3 before the AMR work can use this?
