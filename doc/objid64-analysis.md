# 64-bit object id: the resolution chain today, and the schemes considered (#3994)

This is the analysis behind doc/objid64-design.md: what the code does today, the
defects that follow, and each scheme considered on 2026-09-29 with why it was kept
or dropped. Sections 3, 3b-3f are the discussion trail; the design document holds
the result.

Draft 2026-09-29 (Kale + Claude; for discussion with Aditya).
Line numbers: reviewed-with-reconverse at ebdc49c66.

## 1. The chain as the code actually runs it

Two "homes" exist and are computed differently:

- H_map(idx) = CkArrayMap::homePe(idx), default procNum(idx) (cklocation.h:149),
  PE-count dependent, recomputable from the INDEX on any PE.
- H_id(id)  = the 24 home bits of the id (CkLocCache::homePe, cklocation.h:378).
  Compressible arrays: H_map(idx) at the time lookupID minted the id (cklocation.h:572).
  Non-compressible: CkMyPe() of the CREATING PE (cklocation.C:2647), which is not
  H_map(idx) whenever an element is inserted away from its map home.

Where each is consulted:

| step | who holds what | lookup | code |
|---|---|---|---|
| send by index (source only) | idx | lookupID: compressible = H_map(idx)<<16 + compress(idx); else idx2id hash | ckarray.C:1823, cklocation.h:597 |
| known id, location cached | id | cache.getPe(id) | ckarray.C:1827 |
| known id, no location | id | buffer by id, requestLocation(id) -> **H_id** | ckarray.C:2062, cklocation.C:2396 |
| unknown id (never seen idx) | idx | buffer by idx, requestLocation(idx) -> **H_map** | ckarray.C:2071, cklocation.C:2721 |
| small msg, has id | idx | forward to **H_map(idx)** (not H_id) | ckarray.C:2003-2010 |
| receive, not local, cache miss (every hop) | id | lookupIdx(id): decompress, or local rec, or LINEAR SCAN of idx2id, then handleUnknown(idx) -> H_map(idx) | ckarray.C:1873, cklocation.h:629 |
| creation / migration / death notify | idx | informHome -> H_map(idx); reclaimRemote -> H_map(idx) | cklocation.C:2585, 2792 |
| multi-hop repair | id | cache.requestLocation(id, srcPe): the DELIVERING pe answers, no home involved | cklocation.C:2853 |
| LB global update (CMK_GLOBAL_LOCATION_UPDATE) | id | lookupIdx(id) then updateLocation(idx, e) | cklocation.C:101 |

So Kale's chain "ID -> idhome -> idx -> indexhome -> location" is literally the
receive-side cache-miss path: the id's own home bits are NOT used there; the code
recovers the index (linear scan for non-compressible) only to recompute H_map(idx).
CkLocMgr::homePe(id) (cklocation.h:553) is dead (its caller is commented out at
ckarray.C:1885-1888, with the comment "home of an id being different than the home of
an index"). The ONLY live reader of the home bits is CkLocCache::requestLocation(id).

## 2. Defects that follow from two homes

D1. Non-compressible arrays: registrations go to H_map, location REQUESTS by id go to
    H_id = creator. The creator's cache entry is updated only by its own
    recordEmigration (first migration). Later migrations inform H_map only, so the
    creator answers with a stale location; delivery then relies on multi-hop repair.
    reclaimRemote erases at H_map only; the creator keeps the stale entry forever.
D2. Shrink: restored non-compressible ids keep H_id = old creating PE number
    (restore/resume call insertID(idx, id) with the checkpointed id, cklocation.C:3320,
    3332). If old PE number >= new CkNumPes, requestLocation(id) sends to a PE that
    does not exist.
D3. Shrink/expand + non-compressible: idCounter is restored from PE 0's group copy on
    every PE (ckcheckpoint.C:879 "restore from PE0's copy if shrink/expand"), but the
    ids restored on new PE k include old PE k's (k<<16)+c for c up to old PE k's
    counter. New inserts on PE k mint (k<<16)+counter0 and can collide.
D4. Compressible arrays under shrink/expand: the id CHANGES on restart (lookupID
    recomputes H_map with the new PE count; the checkpointed id is discarded since
    insertID is a no-op). Anything that stored an id is stale: section cookies in
    ckmulticast (ObjKeyList from lookupID, ckmulticast.C:316,355,412). CkCallback is
    safe (stores the index, ckcallback.h:121).
D5. Per-PE counter overflow into the home field (65,535 per PE per collection; 255 in
    --ampi-only), unguarded (#3994 point 3).
D6. lookupIdx's linear scan + CkAssert compiled out in production (#3994 point 4).
D7. The compressor gets 16 bits (#3994 point 1).

## 3. Proposed design: no home in the id; two directories, each keyed by what the asker holds

Principle: a "home" is a directory PE that anyone can compute from what they hold,
under the CURRENT PE count. Never store a PE number that is interpreted as a PE.

Id layout (default): 21 collection + 3 tag + **40-bit element key**.
- Compressible array: key = packed index (budget 40 bits: 1D up to 2^40; 3D 8192^3 = 39 bits;
  3D 1000^3 = 30 bits; today's 16 bits stop at 1D 65536 or 3D 40^3).
  Id is a pure function of the index. Stable across restarts and PE counts.
- Non-compressible array: key = (creator tag : counter), default split 20 : 20
  (configurable like CMK_OBJID_COLLECTION_BITS). The creator tag is a UNIQUENESS tag
  only, never dereferenced as a PE. Abort when the counter would overflow.
  (--ampi-only: collection 29 -> key 32 -> 16:16.)

Two directory functions:
- H_map(idx) as today (user map, procNum default): answers "idx -> id, location".
  Used only on the source PE the first time it sends to an index it has no id for,
  and for creation/insertion placement. Unchanged.
- H_id(id): for compressible ids = H_map(decompress(id)), so the same PE as H_map and
  no second registration. For non-compressible ids = CMK_RANK_0(creatorTag % CkNumPes)
  (any fixed hash of the key works; creatorTag % P coincides with today's behaviour
  whenever the PE count is unchanged, and is well defined after shrink).
  Answers "id -> location".

Registration: createLocal / emigrate / reclaim inform H_map(idx) (today) AND, for
non-compressible arrays only, H_id(id) when it differs (one extra small message per
create/migrate/death of a non-compressible element; zero for compressible arrays).

Delivery (receive side, cache miss): forward to H_id(id) directly. No lookupIdx, no
linear scan, no index anywhere off the source PE. If this PE IS H_id and has no entry,
buffer by id (the id exists, so its registration is in flight) -- this is Aditya's
handleUnknownByID (e417584dd) with H_id defined as above instead of the raw home bits.
The commented-out "after two hops go home" rule (ckarray.C:1885) becomes valid.

Source side: unchanged shape. lookupID(idx) hit -> cache -> send; miss ->
requestLocation(idx) to H_map, which replies updateLocation(idx, entry) that fills
idx2id and the cache as today.

Reverse map id -> idx: needed only (a) locally (CkLocRec has it), (b) compressible
(decompress), (c) demand creation at a non-home PE, which can carry the index in the
request message. idx2id keeps only the forward direction; the linear scan is deleted.

Shrink/expand and restart: ids never change (compressible: index-derived; non-
compressible: restored verbatim). Both directories are rebuilt from the restored
elements themselves (createLocal -> informHome to H_map and H_id under the NEW PE
count), not from checkpointed cache contents. Counter safety after restart: on each PE
set idCounter = 1 + max(counter of restored ids whose creator tag == this PE's tag),
computed once at the end of CkPupArrayElementsData. Section cookies (D4) become stable.

CkLocCache: give it its manager pointer (created 1:1, ckarray.C:653-660; bound arrays
share the pair) so requestLocation(id) can compute H_id; or move that method into
CkLocMgr. Header TODO at objid.h:19 ("can be determined at runtime") is resolved by
construction rather than by sizing the field at runtime.

## 3b. Kale's refinement (2026-09-29 17:15): make the id home EQUAL the index home, both as a hash

For a non-compressible array, put a hash of the INDEX in the id instead of a PE number:

    id = collection(21) | homeKey(24) | counter(16) | tag(3)
    homeKey = hash24(idx)                 (map-independent mix of the index bits)
    home(id) = home(idx) = CMK_RANK_0(homeKey % CkNumPes)
    counter  = assigned by the home, per homeKey (disambiguates hash collisions)

Consequences:
- One directory, at one PE, reachable from the index (source) and from the id (any
  hop) by the same arithmetic. D1 disappears without a second registration.
- Shrink/expand: the id is unchanged; both sides recompute homeKey % P'. Each new home
  rebuilds idx2id and its per-key counters (max+1) from the elements restored to it.
  D2, D3, D4 all disappear.
- The 16-bit counter now bounds indices that COLLIDE in a 24-bit hash bucket per
  collection (65,536 of them), not elements per PE. With N = 1e9 elements and a decent
  mixer that is ~60 per bucket. D5 becomes unreachable in practice; still guard it.
- Today's default index home for a dynamically inserted array is already a hash of the
  index: RRMap::procNum = ((i.hash()+739) % 1280107) % CkNumPes (cklocation.C:352). So
  the default behaviour is preserved in kind. No map in src/, examples/, tests/ or
  benchmarks/ overrides homePe except CldMap. Placement (procNum) is untouched.
- Cost: ids must be minted at the home. Creation of a non-compressible element on a PE
  other than its home routes the creation through home (one extra hop, dynamic
  insertion only; populateInitial already creates at procNum). This also closes the
  known race at ckarray.C:1050 ("What if it's remote but we don't know it?"): home is
  the serialization point, as it already is for demand creation.
- Optional map hook `homeKey(idx)` (default = hash24) lets a map co-locate the directory
  with its placement (e.g. block number) if it wants; the runtime only requires that
  home = homeKey % P on both sides.
- Use a real mixer for hash24, not CkArrayIndex::hash (ckarrayindex.h:145, a sum of
  rotated words; structured indices will cluster).

Compressible arrays are unchanged from section 3: id = packed index (40 bits),
home(id) = home(idx) = map->homePe(decompress(id)); no minting, no counter.

## 3c. The cost of home-minting for LOCAL insertion (Kale's question, 17:26)

Facts (ckarray.C, ckarray.ci):
- Constructor messages carry NO id. The envelope's array union (id, arr, hopCount,
  ifNotThere) is valid for ArrayEltInitMsg and the id field is left unset by
  prepareCtorMsg (:986). It is available.
- The index is NOT in the constructor message either. It travels as a marshalled
  parameter of the CkArray group entry `insertElement(CkMarshalledMessage,
  CkArrayIndex, int listenerData[])` (ckarray.ci:13), with the ctor message nested.
- insertElement is `[inline]`. `ckInsert(m, ctor, CkMyPe())` therefore runs
  synchronously: the element exists (and ckLocal() works) when the call returns.
  ckInsertIdx (:732) -> findInitialHostPe (:993): host = known location, else the
  proposed PE, else procNum(idx).
- At construction the id is already used: PE-level array_objs table (:240), LB
  RegisterObj, the element table, the location record, the cache.

Under section 3b (id minted at the hash home), a local insert whose home is elsewhere
becomes a round trip (2 messages per element) and asynchronous. For the common real
pattern -- a group member inserting all of its own elements under a user-designed
partition (paratreet, ChaNGa, AMR) -- that is 2 messages per element at startup and a
semantic change (element not there on return). Rekeying after a provisional local id
is possible but touches LB registration, array_objs, the element table and the record,
and 40-bit index hashes collide too often at 1e9 elements to skip it.

Trade-off, non-compressible arrays only:

| scheme | creation | migration / death | directories |
|---|---|---|---|
| 3b single hash home, home mints | +2 msgs when host != home; async local insert | as today | one |
| 3 two maintained homes: idx home = map, id home = creator % P; creator mints | as today (sync, 0 extra) | +1 small msg per migration not leaving the creator, +1 per remote death | two |

Recommendation: section 3 for non-compressible arrays (keeps synchronous local
insertion, no rekeying, minimal diff from today: two extra informs). Section 3b's
single-directory property can be added later for arrays declared as remotely inserted,
if ever wanted. Compressible arrays: one home either way, no minting.

## 3d. Both schemes as a per-array creation option (Kale, 17:34)

Yes. The choice belongs to the location manager, which is already per array (shared by
bound arrays) and already makes a per-array id decision at construction: `compressor =
FixedArrayIndexCompressor::make(bounds)` from CkArrayOptions (cklocation.C:2476), pupped
via `bounds` (:2532). Add an id-scheme field to CkArrayOptions, pupped next to bounds,
so every PE's branch knows the scheme and can compute home(id) locally:

    enum IdScheme { Compressed /* auto when bounds fit 40 bits */,
                    CreatorMinted /* default otherwise: section 3 */,
                    HomeMinted   /* opt-in: section 3b */ };

The rest of the runtime treats the 40-bit key as opaque; only CkLocMgr interprets it,
through one small strategy interface:

    bool   mintLocally(idx, id&)     // Compressed: compute; Creator: counter; Home: false
    int    homeOfId(id)              // map(decompress) | creatorTag % P | hash24 % P
    int    homeOfIdx(idx)            // map->homePe | map->homePe | hash24(idx) % P
    void   registrationTargets(idx, id, out)  // {idxHome} | {idxHome, idHome} | {home}
    void   rebuildCounters(restored) // none | per creator tag | per hash key

Scheme-specific code paths:
- CreatorMinted: informHome/reclaim also to idHome (two informs). Nothing else new.
- HomeMinted: ckInsertIdx routes the insertElement group message to homeOfIdx when the
  host is not the home; home mints, records (idx -> id, location = host), stamps the id
  into the nested ctor message's envelope id field, forwards. Host buffers by id when
  the location is itself but the element is absent (deliverInline's bound-sibling
  buffer, ckarray.C:1949, already does this). Insert is asynchronous for such arrays
  even when host == CkMyPe(); document it. Demand creation: home stamps the id in the
  approval message.
- Compressed: as section 3, budget 40 bits; no minting, one home.

Shared by all three: no home field in the id, the cache reaches its manager, delivery
by id goes to homeOfId, restart rebuilds directories under the new P, overflow guard.

Default: Compressed when bounds fit, else CreatorMinted (today's creation semantics
preserved). HomeMinted is opt-in (`CkArrayOptions::setIdScheme`), for arrays that are
inserted remotely or migrate heavily and want a single directory.

## 3e. Creator-minted id that hashes to the index home (Kale, 17:37) -- the bit accounting

The scheme: id = homeKey(idx) | creator | counter, minted synchronously on the inserting
PE. home(id) = homeKey % P = home(idx) = hash(idx) % P. One directory, no round trip,
no rekeying, shrink-stable because homeKey is fixed width and both sides reduce mod P
at use time. The only obstacle is width:

    tag 3 + collection C + homeKey H + creator Cr + counter N
    H  >= log2(max PEs that may serve as homes)   20-24
    Cr >= log2(P)                                 20
    N  = elements per creator per collection      20+
    -> 63 + C bits; with C = 21: 84.

64-bit variants all compromise: (a) leased counters instead of creator bits
(allocator hands out blocks; first block = PE rank, refill by request, prefetch at
half) makes uniqueness ~32 bits: 3 + C + 20 + 32 -> C <= 9 (512 collections), or with
H = 16 (only 65,536 PEs ever serve as homes) C <= 13. Feasible, cramped, plus a lease
protocol.

128-bit id (Kale: 40 bits is not sacrosanct). Layout proposal:

    tag 3 | collection 21 (29 AMPI) | payload 104
    payload, standard index (nInts <= CK_ARRAYINDEX_MAXLEN = 3):
        the CkArrayIndexBase itself: 96 index bits + nInts(2 bits) + dimension(3 bits)
        -> every built-in 1D..6D index is the id, NO BOUNDS NEEDED, no compressor,
           home = map->homePe(idx) read straight from the id.
    payload, oversized custom index (MAXLEN raised, e.g. Aditya's AMR):
        homeKey 24 | creator 24 | counter 32 | spare 24; home = homeKey % P.

Measured cost, envelope: sizeof(envelope) = 64 on reconverse-darwin-arm8 (type union
16 bytes, ALIGN_BYTES 16, no slack). s_array today = id 8 + arr 4 + hopCount 1 +
ifNotThere 1. `arr` duplicates the collection bits already inside the id
(setRecipientID(ObjID(thisgroup, id)) and setArrayMgr(aid) get the same gid); hopCount
fits the spare byte in s_attribs (3 of 4 bytes used); ifNotThere is a property of the
entry method (ckarray.C's own TODO at :2081 says so) and can be looked up. So s_array
= one 128-bit id, the union stays 16 bytes, the envelope stays 64 bytes. (The
CMK_SMP_TRACE_COMMTHREAD config already pushes the union to 24 and is unaffected in
kind.)

Remaining costs: 16-byte keys in the id-keyed hash tables (locrec hash, cache,
element table, array_objs, bufferedIDMsgs); +8 bytes per object in the LB database
(lbdb.h ids are CmiUInt8) and in section key lists; the typed-id refactor surface
(Mikida's unmerged 2021 branch 64_bit_update: 19 files, +340/-164, introducing
ck::BaseID) -- that surface has to be touched for any width change and is the natural
vehicle. AMPI: ranks are 1D bounded, index-as-id, unaffected.

What the 128-bit design removes: the compressor and its bounds requirement, the
setNumInitial/setEnd trap, idx2id for all standard arrays, lookupIdx entirely (the
index is in the id), the home field, D1-D7. What it keeps: the hash-home path only for
oversized custom indices.

## 3f. Process-level homes and one location table per process (Kale, 17:42)

Idea: hash to a PROCESS, not a PE. One directory + cache table per process, shared by
its PEs; the per-PE tables keep only id -> object pointer (which already exist: the
CkArray element table and the PE-level array_objs at ckarray.C:240). This is Kale's
long-standing process-level location cache (Kale's long-standing agenda item) and it
composes with every scheme above.

What it buys on its own:
- A location miss is paid once per process, not once per PE (64x fewer directory
  requests on a 64-PE node for all-to-all patterns); the same for stale-location
  repairs after migration.
- Directory entries are stored once per process instead of once per PE that has
  cached them.
- Home lookups are answered from shared memory by whichever PE of the home process
  receives the request (node-level message, CmiSyncNodeSend / CsdNodeEnqueue exist on
  reconverse: contrib/reconverse/include/converse.h:492,596).
- Non-SMP builds degenerate to today (one PE per process); nothing lost.

Bits: home key width becomes log2(max processes): Frontier 9,408 nodes x (1..8
processes) -> 14-17 bits, so H = 16 is comfortable where per-PE needed 20-24. If the
creator tag is also per process (per-process atomic counter, or process tag +
rank-in-process bits) it drops the same way. Saving: 8-12 bits in total.

Does that rescue the 64-bit single-home creator-minted scheme? Not without leasing:
3 + C + H 16 + Cr 16 + N -> N = 29 - C; C = 12 gives 131,072 elements per process per
collection, C = 21 gives 256. With leased uniqueness (32 bits, no creator tag):
3 + C + 16 + 32 -> C = 13. So in 64 bits the choice stays: leases + small collection
field, or two homes. In 128 bits it is simply roomier, and the process-level table is
a pure efficiency win.

Design points the process-level table has to settle (not blockers):
- Concurrency: written by any PE of the process (creation, migration, cache fill),
  read by all. Sharded map with per-shard locks, or a lock-free map; reads dominate.
  CmiNodeLock is a pthread mutex on reconverse (converse.h:48).
- The entry still names a PE, with the epoch as today. Chares stay bound to
  individual PEs (Kale, 17:44); only the directory and cache are per process.
  Message delivery is unchanged: the sender addresses the PE named in the entry.
- Wake-ups: bufferedIDMsgs / bufferedIndexMsgs are per PE. When PE a's request fills a
  shared entry, PE b's buffered messages must be flushed: keep a per-entry waiter set
  (PE bitmask) in the shared table and notify those PEs (local CmiSyncSend or a
  node-queue wake), or have buffering PEs register a listener keyed by id.
- Home = key % numProcesses (shrink/expand: recomputed both sides as before).
  Drone mode (CMK_RANK_0, cklocation.h:330) is orthogonal and not part of this.
- CkLocCache is a per-PE group today (cklocation.ci:10); it becomes a per-process
  object reached through a nodegroup or a CsvAccess structure keyed by locmgr gid.

## 4. What does not change

Epochs and CkLocEntry; the location cache as an id -> (pe, epoch) map; multi-hop
repair; LB's use of ids as opaque keys (lbdb.h:90); device paths (ids as map keys);
the user-visible CkArrayMap interface. Groups/nodegroups never build an ObjID.

## 5. Sequencing (each PR independently mergeable)

P0 diagnostics: abort on counter overflow; warn when setNumInitial/setEnd left an array
   non-compressible. Coverage test: migrating non-compressible array (6D custom index,
   -DCK_ARRAYINDEX_MAXLEN=6, or unbounded anytime_migration) run under shrink/expand.
P1 make H_id a function (CkLocMgr::idHome(id)), route the receive-side miss through it,
   register non-compressible elements at H_id, delete the linear scan. Home bits still
   present but only read through idHome(). Fixes D1, D6, unblocks Aditya's path.
P2 reinterpret home bits as creator tag; H_id = tag % P; restart counter fix.
   Fixes D2, D3.
P3 move the compressor budget to the full 40-bit key; id independent of home. Fixes
   D4, D7. (One-line budget change once P1-P2 hold; the assertions in lookupID go.)

## 6. Questions for Aditya

Q1 Shrink/expand path in use: the ckcheckpoint.C CkResumeRestartMain route (+shrinkexpand,
   Cmi_myoldpe) or something newer on rate-aware-gpu-lb? Does it restart the LB database
   (ids held across the restart)?
Q2 AMR index: custom CkArrayIndexT with how many ints (CK_ARRAYINDEX_MAXLEN)? Elements
   per PE per collection at peak (D5 exposure)?
Q3 e417584dd keys its pending-request set by the element id; with H_id defined as above,
   is anything in his device/LB code reading the home bits directly? (Audit says no on
   the reviewed line.)
Q4 Any stored ids in application state across LB steps or restarts (sections, callbacks)?
