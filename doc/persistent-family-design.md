# The persistent family: regions, counters and put (design, 2026-11-16)

Status: a design for discussion. There is no code yet; implementation would follow the phases of
section 15.

Facts carry file references. Working assumptions are marked W (section 2). Points that are proposed but
not settled say so where they appear, and are listed in section 16.

Two appendices hold the evidence, to read only to check a claim:
- Appendix A (behind sections 8 and 10): the unmerged `CkDeviceChannel` on branch rate-aware-gpu-lb,
  and its measurements;
- Appendix B (behind section 17): the communication calls of ChaNGa, SpECTRE, Quinoa and Enzo-E, with
  file:line references.

FINDINGS.md, cited in section 12.2 and elsewhere, is the commperf project's measurement log. It is not in
this repository; the numbers this document uses are quoted where they are cited.

Contents: 1 goals; 2 working assumptions; 3 objects and API; 4 semantics (4.9 readiness patterns, 4.10 other
runtime services); 5 the counter (atomicity
across shared memory and network); 6 host data paths by locality; 7 strided layouts; 8 device memory;
9 sender side and batched notifications; 10 lifetime, teardown, migration; 11 error checking;
12 messages (eager, per-pair ring, rendezvous); 13 what happens to today's APIs; 14 tests and
measurements; 15 implementation phases; 16 open questions; 17 existing applications (G6); 18 draft
manual section, which call to use when; Appendix A, device channels on branch rate-aware-gpu-lb; Appendix B, application surveys.

---------------------------------------------------------------------------------------------------

## 1. Goals and non-goals

Goals:

- **G1 put and notify.** A repeated transfer into a buffer the receiver named once costs one put and one
  notification, with no handshake, no copy and no runtime readiness protocol.
- **G2 put and decrement.** Contributions from many senders into one or more receiver buffers
  (contiguous or strided) produce one scheduler entry on the receiver when all have arrived.
- **G3 one mechanism in every locality.** Same PE, PEs of one process, processes on one host, different
  hosts. The application code is the same; the runtime picks the data path per pair.
- **G4 host and device memory** through the same objects; memory kind is a property of a region.
- **G5 a smaller API surface.** Replace the direct API, CkDevicePersistent, the user-facing reconverse
  persistent API and CkDirect with one set of objects (section 13).
- **G6 no source change in existing applications**. ChaNGa, SpECTRE, Quinoa, Enzo-E
  and the other production codes keep compiling and running unchanged. They change code only to adopt
  the persistent family deliberately. The exception is NAMD, which uses the reconverse persistent API;
  we will port it ourselves (12.3). The survey of what those codes call is in Appendix B (section 17).

Non-goals: device-initiated communication (puts issued from GPU kernels, as in NVSHMEM);
collective operations; fault tolerance; pointer-based data (it stays with messages and PUP).

## 2. Working assumptions

- **W1 readiness is the application's.** A contributor puts into a region only when the application's
  logic guarantees the receiver no longer needs the region's previous contents.
  The runtime does not check this in production builds; error-checking builds detect the violations
  that are detectable (section 11).
- **W2 iteration.** Regions, counters and their handles are set up once and used for many rounds. Setup
  may cost round trips; a round may not.
- **W3 handles travel in ordinary messages.** A handle is a small value that can be PUP'd into any entry
  method. Setup needs no special protocol beyond that.
- **W4 migration tears down and rebuilds**. No forwarding of puts to a migrated owner.
- **W5 strided arrays use layout descriptors; PUP-style packing is for pointer-based data**.
  Variable-size contributions are a special case to support (section 4.6).
- **W6 the counter is one atomic 64-bit word** that contributors in every locality update (section 5 shows that shared-memory and network contributions work together on it).
- **W7 bookkeeping runs at network poll time where feasible**. Feasible means the decrement
  only. Completion handlers run inside `lci::progress_x` while the PE holds that device's trylock, with
  `g_in_progress_callback` set (comm_backend_lci.cpp:463-478). So a copy at poll time would stop every
  other PE from polling the device, and sends from that context are restricted. Copies and scatters run
  on the owner PE instead, which also puts the data in the owner's cache and NUMA domain.
  The application's action always runs on the owner PE through its scheduler.
- **W8 reconverse facts** (survey of bbb92e1):
  - no comm thread; each PE polls the network from its scheduler loop, and a completion handler runs on
    whichever PE thread of that LCI device won the device trylock, not necessarily the owner
    (scheduler_registered.cpp:98-113, comm_backend_lci.cpp:328-334, 462-480);
  - `CmiPushPE` onto another PE's queue is safe from any thread (moodycamel MPSC, queue.h:66-70);
  - every process on a host maps every peer's IPC segment (shm or xpmem), at different addresses, so
    links are offsets (doc/internals/shared-memory-ipc.md:10-18); each PE drains its own lock-free
    receive stack `rq[rank]` in the owner's segment once per scheduler iteration (scheduler.cpp:74-80);
  - xpmem mode exports each process's whole address space (cmixpmem.cpp:87-88) but attaches only the
    pool today; CMA is compiled out (conv-rdma.cpp under `CMK_USE_CMA`, never defined for reconverse);
  - zerocopy get/put between two processes on one host goes through LCI RDMA, i.e. the NIC
    (conv-rdma.cpp:403-413);
  - no backend call yet carries a remote-completion signal on a put (comm_backend.h:72-80); the
    prototype branch persist-put-signal adds `issueRputSignal` (LCI `post_put_x(...).remote_comp(rcomp)
    .tag(signal)`: ibv RDMA write with immediate, ofi `fi_writedata`), with a 16-bit signal
    (`max_imm_tag` 65535).
- **W9 machines.** Anvil (HDR100 InfiniBand, LCI ibv; GPU partition A100 with CUDA 11.4, whose missing
  dmabuf API breaks LCI's CUDA backend, anvil.md:810-816), Delta (Slingshot-11 cxi, A100 and A40),
  Frontier (Slingshot-11 cxi, MI250X). The Mac (ofi over TCP) is for development only.
- **W10 no NIC atomics.** Network contributions decrement the counter with a CPU atomic in the owner's
  process, triggered by the put's immediate data. NIC atomic operations are not atomic with respect to
  CPU atomics on the same word on most HCAs, so they cannot be mixed with shared-memory contributions.

## 3. Objects and API

Naming: regions, counters and put/get form the **persistent family**. Everything in it is
set up once, used for many rounds and torn down (W2), which is what "persistent" means here, close to
MPI's persistent requests. Family 1 (section 12) is the **message family**. Consequences: the existing
reconverse persistent API is replaced, not kept (12.3), so the word keeps one meaning; `CkDevicePersistent`
already fits the name and becomes the device case of the family.

### 3.1 Objects

| object | what it is | owned by | travels as |
|---|---|---|---|
| region | memory registered once: base, bytes, memory kind, optional pieces with layouts | one PE | region handle |
| counter | an atomic count, an action, a re-arm mode | one PE | counter handle |
| layout | a strided description of a piece of a region (section 7) | inside a region | inside the region handle |
| import | a contributor's prepared view of a handle (rkeys, xpmem attachment, IPC mapping) | the contributor's process | not sent |

The **owner** of a region or a counter is the receiver: the PE (in Charm++, the object on that PE) that
creates them, receives the data and runs the counter's action. A **contributor** is a sender: it puts
data into an owner's region and decrements the owner's counter.

A **piece** is a part of a region that puts write into, numbered from 0 in the order the owner declares
the pieces. Each piece is an offset in the region plus either a byte length (one contiguous range) or a
layout (a strided set of ranges, section 7). Pieces let one region, registered once and sent as one
handle, receive several contributions at different places. For example, a block's ghost buffer is one
region and each of its 6 ghost faces is a piece. A put names the piece by its number, plus an offset
within the piece. A counter is independent of regions: one counter may
count contributions into several regions, and a region may be paired with two counters (double
buffering by parity uses two regions and two counters). The pairing is made by each put, not when the
region or the counter is created: `CmiPut` names the destination region (through its import), the
piece and the counter to decrement. So the owner sends each contributor a region handle, a piece and a
counter handle, and the contributor names all three in every put (example in 3.3).

### 3.2 Converse API (C, reconverse)

```c
typedef enum { CMI_MEM_HOST, CMI_MEM_DEVICE } CmiMemKind;

/* Regions */
CmiRegion  CmiRegionCreate(void *base, size_t bytes, CmiMemKind kind, int devId, int flags);
void      *CmiRegionAlloc(size_t bytes, CmiMemKind kind, int devId, int flags, CmiRegion *out);
int        CmiRegionAddPiece(CmiRegion r, size_t offset, size_t bytes, const CmiLayout *layout);
void       CmiRegionDestroy(CmiRegion r);
void       CmiRegionGetHandle(CmiRegion r, CmiRegionHandle *h);     /* POD up to 8 pieces; see below */

/* Counters */
CmiCounter CmiCounterCreate(int64_t n, CmiCounterAction action, int flags);
                     /* flags 0: this counter is re-armed explicitly; CMI_COUNTER_AUTO_REARM:
                        this counter re-arms itself before its action runs (4.2) */
void       CmiCounterRearm(CmiCounter c, int64_t n);  /* where the round's data is no longer used */
int64_t    CmiCounterValue(CmiCounter c);
void       CmiCounterDestroy(CmiCounter c);
void       CmiCounterGetHandle(CmiCounter c, CmiCounterHandle *h); /* POD, fixed size */

/* Contributors */
CmiImport  CmiImportRegion(const CmiRegionHandle *h);   /* once per handle per process; registers
                                                     with the owner (10.1) */
void       CmiReleaseImport(CmiImport imp);
int        CmiImportRegions(const CmiRegionHandle *h, int n, CmiImport *out);  /* batch forms: one   */
void       CmiReleaseImports(CmiImport *imps, int n);                         /* message per owner */
int        CmiPut(CmiImport dst, int piece, size_t offset,
                  const void *src, size_t bytes, const CmiLayout *srcLayout,
                  const CmiCounterHandle *remote,      /* may be NULL */
                  CmiCounter local,                     /* may be NULL */
                  void *stream);                        /* device: the producer's stream, else NULL */
int        CmiGet(void *dst, size_t bytes, const CmiLayout *dstLayout,
                  CmiImport src, int piece, size_t offset,
                  CmiCounter local, void *stream);
int        CmiSignal(const CmiCounterHandle *remote);   /* a contribution with no data: decrements
                                                     the remote counter only (4.9, pattern 3) */
```

`CmiCounterAction` is one of: a Converse handler index with a preallocated message (the runtime
allocates it when the counter is created, so firing never allocates); a C function and argument run on
the owner PE from the scheduler; or none (the owner tests `CmiCounterValue`).

`CmiRegionAlloc` lets the runtime choose memory that every locality can reach without per-pair work:
host regions come from the owner's IPC segment when IPC is on (so host peers can write them in shm mode)
and are NIC-registered once; device regions come from the registered device arena (CkDeviceMalloc's
pool, ckrdmadevice.C:204-270). `CmiRegionCreate` accepts any user memory (section 6 says what that
costs in each locality).

The IPC segment does not scale to application data. It defaults to 8 MiB
(`kDefaultSegmentSize`, cmi-shmem-common.h:24), and FINDINGS.md already records it filling. Ghost layers
and pencil buffers live inside application arrays, and the first adopters' faces are parts of their own
arrays (17.4). So in practice regions are user memory. The same-host direct path for user memory is
xpmem, which becomes a requirement for that path (6); `CmiRegionAlloc` from the segment is for small
regions.

A region handle is a fixed-size POD with room for 8 pieces and their layouts; a region with
more pieces gets a variable-size handle, which is PUP'd the same way.

### 3.3 Charm++ API (C++)

A block that receives its 6 ghost faces from 6 neighbors:

```c++
// Owner (the receiving block): one ghost region with a piece per face, and one counter for 6 faces.
CkRegion ghosts(ghostBuf, ghostBytes, CkMem::Host);  // or CkRegion::alloc(bytes, CkMem::Device, dev)
int piece[6];
for (int f = 0; f < 6; f++)
  piece[f] = ghosts.addPiece(faceOffset[f], CkLayout::box3d(...));
CkCounter halos(6, CkCallback(CkIndex_Block::halosDone(), thisProxy[thisIndex]));
                // a third argument CkCounter::AutoRearm makes this counter re-arm itself (4.2)
// Tell the neighbor on each face where its data goes. Handles are PUP-able values.
for (int f = 0; f < 6; f++)
  neighborProxy[f].takeHandles(ghosts.handle(), piece[f], halos.handle());
// ... in the code that last reads the ghost data, or where it contributes (4.9): halos.rearm(6);

// Contributor (a neighbor), once, in takeHandles(regionHandle, piece, counterHandle):
dst = CkImport(regionHandle);  myPiece = piece;  ownerCounter = counterHandle;
// Contributor, every iteration: put its face into its piece and decrement the owner's counter.
CkPut(dst, myPiece, 0, src, bytes, ownerCounter, &localCounter /* or nullptr */, stream /* device */);
```

Six separate regions, one per face, work the same way: each neighbor gets its own region's handle and
the same counter handle. The action may be an entry method, so an SDAG `when halosDone()` waits for a round. A convenience
`CkTarget` bundles one region, one counter and its pieces for the common case, and `CkChannel` bundles a
one-piece region, its counter and a release credit (4.9, pattern 3; section 13). `CkSignal(counterHandle)`
is the C++ form of `CmiSignal`.

## 4. Semantics

### 4.1 Visibility

When a counter's action starts on the owner PE, every byte of every contribution counted toward that
round is visible to the owner PE's thread; for device regions, also to device work the action launches
afterwards (section 8.4).

### 4.2 Counting, firing and re-arming

The counter word holds the number of contributions still expected in the current round. A contribution
has a weight k: 1 for a put, k for a batch of k same-host device puts (8.2).

- A contribution does `old = fetch_sub(k, acq_rel)`. If `old == k`, it completed the round and it
  **fires**: it queues the counter's action on the owner PE (section 5.3 says how, per locality).
  Error-checking builds abort if `old < k` (section 11).
- **Re-arm is explicit by default.**
  - The owner calls `CmiCounterRearm(c, n)` (C++ `c.rearm(n)`) at the point where it has finished using
    the round's data. That point can be:
    - after the last read, in the action or in a later entry method;
    - after the device work that reads the data has been ordered (8);
    - where the owner contributes to the reduction that separates iterations (4.9).
  - Re-arm does `fetch_add(n, acq_rel)`. If the result is 0, the next round already arrived in full and
    the counter fires again.
  - Manual re-arm allows a different N per round, for adaptive codes whose neighbor counts change.
    In production builds it is the same single atomic as automatic re-arm, and no message is added.
- **Re-arm is legal only after that round's action has started**. Before that, a second
  firing would push the counter's one preallocated action message while it is still queued (5.3). In
  case (b) that corrupts the owner's intrusive IPC receive stack. Error-checking builds check it.
- **Automatic re-arm** is opt-in for each counter: the flag `CMI_COUNTER_AUTO_REARM` in
  `CmiCounterCreate` (C++ `CkCounter::AutoRearm`). It is not a build option; counters with and without
  it coexist. The runtime re-arms the counter just before running its action. It is safe only when both hold:
  - the action is the last use of the data, i.e. it copies or consumes the data before returning;
  - no contributor can issue the next round's put before the action has run.

  It is not the default because a put that lands after an automatic re-arm, while the owner still uses
  the data, passes every check of section 11.
- A contribution that arrives between firing and re-arm takes the word below zero. It does not fire,
  and after the re-arm it counts toward the next round, so counting stays correct even then. Such an
  arrival means the contributor did not wait for readiness (W1), so error-checking builds abort on it.

The rule that whoever moves the word to exactly zero fires covers both the last contribution and an
early round. A round fires exactly once.

### 4.3 Ordering

None between contributions. A contribution is whole when counted: its bytes are all visible.

### 4.4 Several regions or several counters

Allowed in any combination. A put names one destination piece and at most one remote counter.

### 4.5 Source completion

The local counter given to `CmiPut` is decremented when the source buffer may be reused. Local
completion never sends a message (section 9). A put with no local counter must not have its source
modified until something else tells the application it is safe (for example the next round's data
from the same neighbor, the double-buffering argument in charm-notes, charm_best_practices.md:2198).

When the source may be reused depends on the machine. A write completes locally when the
NIC has read the source. Where the runtime carries a cross-host put out as announce-then-read (8.2), the
source stays in use until the owner's get completes, at least a round trip later. A code that reuses
its source "when the next round arrives" relies on the timing; give `CmiPut` a local counter instead.

### 4.6 Variable-size contributions

A piece has a capacity. A put of fewer bytes is allowed; its length must reach the owner. The signal
has no room for it (16 bits name the counter), so the length travels with the data: each piece declared
`CMI_PIECE_VARSIZE` reserves an 8-byte length word just before its data. The 8 bytes must belong to the
region: a piece allocated by `CmiRegionAlloc`, or a user piece declared with its own header space. They
cannot be taken from inside a user array, and import checks refuse it. The runtime writes
`[length | data]` as one transfer, so the length is visible whenever the data is. The action reads
`CmiRegionPieceLength(r, piece)`. A put larger than the capacity is refused at the contributor (it knows
the capacity from the handle) with an error; growing a piece is a teardown and rebuild (section 10).

### 4.7 Get

`CmiGet` reads a piece of a remote region into local memory and decrements a local counter when the
bytes are in place. The owner is not told unless it asks (section 9.2). Gets serve data that stays valid
while readers fetch on demand; puts serve everything iterative.

### 4.8 Quiescence detection

- **Created count.** A put, or a signal (`CmiSignal`), increments it on the contributor's PE when it is
  issued.
- **Processed count.** It is raised in bulk on the owner PE when the round's action runs from the
  scheduler. The amount is the round's N: with manual re-arm, the N that round was armed with. The
  poll-time handler and the other-process contributors of case (b) never touch QD counters.
- **The action message** is not counted separately; the bulk step stands for it. Between the firing and
  the action, the created count exceeds the processed count, so QD waits for the action, as it should.
- **A partly filled round** holds quiescence off, because work is outstanding. A counter destroyed in
  mid-round adds the contributions it has received (N minus the remaining count) to the processed
  count; otherwise QD would never complete.
- **No remote counter.** A put without one is counted processed at its local completion on the
  contributor.
- **Gets.** A get is created at issue and processed at its completion, both on the reader.

SpECTRE, Enzo-E and ChaNGa start phases with `CkStartQD` or `CkWaitQD` (section 17), so this is required
before any of them adopts puts.

### 4.9 Readiness patterns

W1 leaves readiness to the application. The design serves two patterns, and the argument for each is
written out so that users can check their own code against it. In both, remember that a put is not a
message: it lands when it arrives, whatever the owner is executing.

1. **A reduction between iterations.**
   - Conditions: no object starts iteration i+1 until the reduction of iteration i has completed, and
     every object contributes to that reduction only after it has finished using the data it received
     in iteration i.
   - Result: no put of iteration i+1 can land while an owner still reads iteration i's data, so one
     buffer is enough.
   - The order inside an object matters. With `contribute(); compute();` a neighbor's next put can land
     while `compute()` still reads.
   - Rule for users: re-arm where you contribute.
2. **Two buffers by parity**, for codes without a reduction every iteration.
   - Setup: each owner has two regions per neighbor (and two counters), and iteration i lands in region
     i mod 2.
   - Argument: A puts its iteration i+2 data into the region that held iteration i only after computing
     iteration i+2's boundary. That boundary needs B's iteration i+1 data, and B sends it only after its
     last read of the iteration i data that A put. So the region is free when the put lands.
   - Rule for users: send your next round only after your last read of the round before.
3. **A release credit**, opt-in, for producer-consumer streams whose logic gives no such guarantee (a
   producer may run ahead of its consumer, as in a stream of MoE tokens). It is the credit of the
   unmerged `CkDeviceChannel` (Appendix A.1), built from the objects of this design.
   - Setup: the producer creates a credit counter with N = 1 and sends its handle to the consumer along
     with the region's handle.
   - When the consumer has finished with a round's data, it re-arms its own counter and calls
     `CmiSignal` on the producer's credit counter. `CmiSignal` is a contribution with no data.
   - The producer issues its next put only after its credit counter fires. Until then it keeps the data
     itself. `CkChannel` (section 13) does this queuing for the application.
   - The application still decides when the buffer is free, so W1 holds. The runtime only carries the
     signal and holds no put of its own accord.
   - Cost: one small transfer per round. The release credit of the device channels measured 0.5 us to
     another host and 2.2 us within a host, with the libfabric registration cache on (Appendix A.4.5).
   - Rule for users: release when you are done with the data, and use this pattern when neither of the
     first two patterns applies.

Applied to the two uses the design claims:

- **Halo exchange** (the example of 7.1).
  - Pattern 1 applies if the code reduces every iteration (Jacobi with a convergence test); pattern 2
    otherwise.
  - With neither, putting faces straight into the ghost layer of the live array is the jacobi3d bug
    fixed in #3980.
- **FFT transpose** (NAMD's PME, G2).
  - A contributor sends its step s+1 data to a pencil only after it has received that pencil's step s
    result in the backward transpose.
  - The pencil sends that result only after its last read of its step s buffer, because its forward FFT
    and onward send come first.
  - This is pattern 2's argument, with the backward transpose in the role of the partner's next round.
    One buffer per direction is enough.

### 4.10 Other runtime services

- **Load-balancer communication records.**
  - A put records its bytes on the contributor's edge to the owner object when it is issued, as a
    message send does, so comm-aware balancers see the traffic.
  - The edge must be keyed by the owner's own array manager. Aditya's channels first recorded the edge
    under the sender's manager with the owner's element id, which pointed it at the wrong object
    (Appendix A; PERSISTENT_COMM_PLAN.md, 2026-10-18 10:25).
- **Projections.** Two events: put issued (contributor) and round fired (owner, when the action runs).
- **Record-replay.** Out of scope for the first version. Which contribution completes a round varies
  between runs, and region contents are not recorded. A `+record` run that creates a region prints a
  warning.
- **`CkExit` with puts in flight.** Treated like messages in flight at exit. The runtime does not wait for
  them, skips release notices, and deregisters memory only after the network is shut down, so a late
  completion cannot touch freed memory.
- **Attribution of copies.** Same-host copies run in the contributor's entry method and are charged to
  it. Scatters run on the owner PE (W7), charged to the owner. Poll-time work is only the decrement.

## 5. The counter## 5. The counter: one atomic word for every locality

### 5.1 Where the word lives

Each process has a counter table of fixed-size entries (one 64-byte line each, to avoid false sharing
between counters): the atomic word, the owner PE, the re-arm count, the action, a generation number and
the preallocated action message reference.

- With IPC on, the table lives in the process's own IPC segment: a new fixed area after the
  `ipc_shared_` header (cmi-shmem-common.h:97-115), so every host peer reaches entry `i` at
  `segment_base(owner) + table_offset + 64*i` through its existing mapping.
- With IPC off, the table is ordinary process memory, and host peers in other processes contribute
  over the network path (5.2 c), so no other process ever touches the word.

Default size 4096 entries (256 KiB); `+counter_table N` up to 65536 per owner PE (5.5).

### 5.2 Who updates the word

| contributor | who executes the decrement | how it reaches the word |
|---|---|---|
| (a) same process (same or other PE) | the contributor's thread, after its copy | process address |
| (b) other process, same host | the contributor's thread, after its copy | its mapping of the owner's segment |
| (c) other host (or same host without IPC) | the LCI remote-completion handler in the owner process, on whichever PE thread is polling that device | process address |

All three are CPU atomic read-modify-write operations on the same 64-bit word, from cache-coherent cores
of one host. They are therefore linearizable with respect to each other, whatever mix of contributors a
round has. This answers whether shared-memory and network contributions can be combined on one counter:
they can, because both end in a CPU atomic on the owner's host; what cannot be combined is a CPU atomic
with a NIC atomic, which this design never uses (W10).

Requirements to make (b) correct and checked:

- `std::atomic<int64_t>` must be lock-free (then it is address-free, so two mappings of the same page
  work). Add `static_assert(std::atomic<int64_t>::is_always_lock_free)` (true on x86-64 and aarch64);
  reconverse's IPC queue already relies on the same property without asserting it (survey, item 5).
- The word must not be shared with any process-local lock or non-atomic data.

### 5.3 Firing in each case

The contribution that took the word to zero fires:

- (a) and (c): `CmiPushPE(owner, actionMsg)`. In (c) this runs at poll time, on the polling PE: the step 1
  prototype already pushes from the same place.
- (b): the firing thread is in another process, so it cannot use `CmiPushPE`. It pushes the action
  message onto the owner PE's IPC receive stack `rq[ownerRank]` in the owner's segment, which the owner
  drains every scheduler iteration (cmishmem.cpp:111-135). For that, the action message must already be
  an IPC block in the owner's own segment: the owner allocates it when the counter is created. Today
  `CmiAllocIpcBlock` refuses the caller's own segment (cmishmem.cpp:290-292); it needs a variant that
  allocates from it. The push needs a variant too: `CmiPushIpcBlock` asserts that the pushing process
  allocated the block (`meta->mine == block->sender_proc`, cmishmem.cpp:273), and here the owner
  allocated it and a contributor pushes it.
- On the same PE the firing contribution also queues (never calls the action inline), so an action never
  runs inside a put call.
- One preallocated action message per counter is enough: the counter cannot fire again until the owner
  has taken that message off its queue, because re-arm (4.2) is legal only once the action has started.
  The message is runtime-internal (a trampoline handler reads the counter index, re-arms if automatic,
  and calls the user's action), so the user never owns or frees it.
- For an entry-method action, the preallocated message is the final Charm++ envelope, with the target
  and priority of the action's callback. It is not a Converse message that then queues a second Charm++
  message, which would cost two scheduler entries and lose the priority (charm #4022).

### 5.4 Why the owner sees the data (memory ordering)

- Shared memory, (a) and (b): each contributor copies, then `fetch_sub` with release. All decrements are
  read-modify-write operations on one atomic, so each later decrement continues the release sequence of
  every earlier one; the firing decrement is acq_rel, so it synchronizes with every earlier
  contributor. The push to the owner's queue (release) and the owner's pop (acquire) carry that to the
  owner PE. Cross-process, this holds because the atomics are address-free (5.2).
- Network, (c). This is the one place the design states the argument for each provider.
  - **InfiniBand RC (ibv, Anvil).** RDMA write with immediate is one operation. Its completion at the
    target is generated only after the write's data is placed in memory, and the CQ poll has the needed
    read barrier. The handler then decrements with release as above.
  - **Slingshot (cxi, Delta and Frontier).** CXI 2.3.1 has no RMA write with immediate: it refuses
    `FI_REMOTE_CQ_DATA` (Appendix A.4.2). The LCI that reconverse pins (a8127fa) uses native
    `fi_writedata` on cxi when `FI_CXI_ENABLE_WRITEDATA=1` and libfabric is 2.5 or later
    (backend_ofi.cpp:75-79, 220-227); then the signal is one operation, as on ibv, untested so far.
    Otherwise LCI emulates the signal as the write, followed,
    after the write's *local* completion, by a zero-length message carrying the signal. Local completion
    of a write does not by itself mean the bytes are visible at the target (8.1;
    charm_best_practices.md, "A notification after a one-sided put must take the same channel as the
    put"). So the decrement in that message's handler is correct only if one of these holds:
    - the write's local completion is delivery-complete (`FI_DELIVERY_COMPLETE`; LCI's emulated write
      already requests it, backend_ofi_inline.hpp:307, but whether cxi honours it is not established), or
    - the provider orders the send after the write on the same endpoint (send-after-write ordering,
      `FI_ORDER_SAW`), and the signal goes on the same LCI device as the write.

    With `+lci_ndevices 2` (used on Delta) there are several routes. Neither condition has been
    established. Aditya's checksums on Delta passed, but payload checks alone did not catch the
    equivalent misrouting on Delta in 2026-09.
  - Establishing which condition holds on cxi, or making LCI request one of them, is an exit condition of
    phase 1 (section 15). Until then, phases 2-3 are built and tested on ibv.
  - On cxi without native `fi_writedata`, G1's "one put and one notification" is two network operations, so the Anvil prototype's
    numbers (FINDINGS.md) do not carry over to cxi. They are measured there in phase 1.
  - For device memory, see 8.4.

### 5.5 Signal encoding

On the network the 16-bit signal is the counter's table index. The remote-completion id (`rcomp`,
15 bits) chooses what the index means. LCI's fast-path immediate data is the 16-bit tag plus the 15-bit
`rcomp` (lci src/core/progress.cpp:21-22). One `rcomp` per owner PE and per kind makes the 65536 limit a
per-PE limit, not a per-process one. Per process it would be tight: 64 PEs with
over-decomposition and several counters per object. Proposed `rcomp`s: counter decrement; strided staging slot
(section 7.3); per-pair message ring (section 12.2). Reconverse registers one `rcomp` today, for active
messages (comm_backend_lci.cpp:49-53, 273-274). Generation numbers
do not fit in the signal; section 10 makes reuse of an index safe without them.

## 6. Host data paths by locality

| locality | region from `CmiRegionAlloc` | region from `CmiRegionCreate` (user memory) | decrement |
|---|---|---|---|
| same PE | memcpy by the caller | same | (a) |
| same process, other PE | memcpy by the contributor's thread | same | (a) |
| same host, other process, xpmem | memcpy through the mapping | memcpy through an xpmem attachment made at import (the address space is already exported) | (b) |
| same host, other process, shm mode | memcpy through the owner's segment (the region is in it; small regions only, 3.2) | put through the NIC (loopback RDMA; the region is NIC-registered) | (b) / (c) |
| same host, IPC off | put through the NIC | put through the NIC | (c) |
| other host | RDMA write with immediate data into the region | same | (c) |

Notes:

- Within a process there is no message at all: one memcpy and one atomic. Today's zerocopy APIs memcpy
  and also send messages in this case.
- Across processes on a host, the copy is done once, by the contributor, straight into the destination.
  Today `+ipc` copies a message into the receiver's pool and the application then copies it again.
- CMA (`process_vm_writev`) would serve user memory in shm mode, but it is compiled out of reconverse
  and depends on ptrace permissions; not assumed (W8).
- **xpmem is the requirement** for direct same-host writes into user memory. It must be
  confirmed on Anvil, Delta and Frontier in phase 3. Without it, same-host puts into user memory go
  through the NIC.
- **Not verified, to check in phase 2**:
  - on ibv, an RDMA write with immediate consumes a pre-posted receive that is shared with eager
    traffic, which matters when there are many puts;
  - one memory registration per user region runs against cxi registration limits and the kdreg2 cost
    seen on Frontier. One arena registration serves `CmiRegionAlloc`, but user regions each need their own.
- Large contributions within a host: a single-thread memcpy of 16 MB took 501 us on Anvil (33 GB/s), more
  than the NIC's 12.5 GB/s; the direct copy stays the default at every size until measured otherwise.

## 7. Strided layouts

### 7.1 Descriptor

`CmiLayout` describes up to three nested levels: element block bytes `b`, and for each level a count and
a byte stride. Example, a 64^3 array of doubles with one ghost layer stored in a 66^3 array, C order:
- the two i-faces are contiguous (one block of 66*66*8 bytes);
- the two j-faces are 64 runs of 64 doubles (b = 512 B, stride 66*66*8);
- the two k-faces are 64*64 single doubles (b = 8 B, strides 66*8 and 66*66*8).

A contributor may also pass a source layout, so neither side packs by hand.

Putting faces straight into the ghost layer of the live array, with one counter, is safe only under one
of the readiness patterns of 4.9. With neither, it is the jacobi3d bug fixed in #3980.

### 7.2 Within a host

The contributor copies source layout to destination layout directly (a strided copy; device regions use
`cudaMemcpy3DPeerAsync` / `hipMemcpy3DPeerAsync`, section 8), then decrements once for the piece. No
staging, no packing.

### 7.3 Across hosts

An RDMA write targets one contiguous remote range, so a strided destination needs either one write per
block or a contiguous staging area at the receiver plus a scatter. Proposal:

- **Default: gather, one put, scatter** (the staging slot doubles the receiver memory for strided pieces
  that have remote contributors). The runtime gathers the source into a registered contiguous
  buffer (or posts the write with a scatter-gather list on the source side when the block count is at
  most the device's limit, removing the sender copy; whether LCI exposes this is to be checked), puts
  it into a per-piece staging slot at the receiver (allocated when the piece is declared, registered
  once) with the staging `rcomp` and the slot index as signal. The handler only queues a scatter task on
  the owner PE (W7). The owner scatters from the slot into the piece using the layout, then decrements
  the counter once. This is layout-driven by the runtime, not application pack/unpack (W5).
- **Large blocks: one write per block, straight into the region**, when the block size is at least a
  threshold `B_put` (working value 4 KiB, to be measured with commbench halo3d). Each block write carries
  a signal for the piece's hidden sub-counter (count = number of blocks), and the sub-counter reaching
  zero decrements the user's counter once. Relying instead on in-order completion of writes (signal on
  the last write only) holds for InfiniBand RC on one queue pair but not with several LCI devices or on
  cxi without requested ordering, so it is not assumed.
- **No scatter at poll time.** It would run the copy under the device trylock (W7). Scattering on the owner
  costs one extra scheduler entry per strided piece with remote contributors.
- **Gate (proposed; open question 7).** Gather, put and scatter make the same two copies as a message's pack and
  unpack. So phase 5 goes ahead only if commbench `halo3d` shows strided puts beating messages, with the
  eager limit raised (12.2).

## 8. Device memory

### 8.1 Today (survey)

- nocopydevice: same process D2D copy; same host D2D through a staging comm buffer only with `+gpushm`
  (two copies, two events; measured 13-46% slower than RDMA get at 2-16 KB, #3960), otherwise silently
  through the NIC; other hosts GPUDirect RDMA get. Completion through hapi event polling or an ack.
- CkDevicePersistent (#4228): same process D2D; same host CUDA/HIP IPC handle opened once, copy straight
  into the mapped buffer, no staging; other hosts a get, and a put is emulated as a message plus a
  destination-issued get plus an ack, because local completion of a put does not make the data visible
  at the destination (charm_best_practices.md:2968-2985). RDMA paths block the PE in
  `hapiStreamSynchronize` (#4232).
- No remote-completion signal anywhere on recharm; no GPUDirect flush anywhere; LCI forces zero-copy for
  device memory registrations and turns a put with remote completion into `post_putImm`
  (communicate.cpp:171-176, 488-505), untested on device memory over ibv. On cxi, LCI has no hardware
  put-with-immediate and emulates it as the write plus a zero-length message (next item).
- Unmerged: Aditya's `CkDeviceChannel` (charm branch rate-aware-gpu-lb; Appendix A).
  - It is a device put-and-notify with one fixed buffer per channel, a release credit, and migration handled by the runtime.
  - Against `nocopydevice` it is 3-10 us faster per message (27.1 to 17.5 us one way between two processes on one A40 node).
    jacobi3d at 128 KB faces is 38% faster per iteration (1162 to 704-720 us, 8 A40s); the gain is 1-5% at 512 KB and 2 MB, and nothing in leanmd.
  - The measured gain is the removed per-message host work: descriptor, lookup, landing decision, ack. It is not bandwidth.

### 8.2 Proposed paths

The counter is always the host word of section 5; the action always runs on the owner PE's host thread.

| locality | data movement | when the decrement happens |
|---|---|---|
| same process (same or other GPU) | `hapiMemcpyAsync` D2D (3D variant for strided) on a runtime stream of the contributor, ordered after the producer's stream with an event | when the copy's event completes (hapi event polling), by the contributor's PE: (a) |
| same host, other process | CUDA/HIP IPC handle of the destination opened once at import (ref-counted per process, as `persistIpcPtr` does, ckrdmadevice.C:676-715); D2D copy straight into the mapped region | when the copy's event completes, by the contributor's PE, on the shared word: (b) |
| other host, GPUDirect available | RDMA write with immediate data from device memory to device memory | in the remote-completion handler: (c) |
| other host, no GPUDirect | device to pinned host staging, host put with signal into a pinned staging slot at the receiver, host to device copy issued by the owner PE (W7), decrement after that copy's event | (c), deferred to the event |

The source side never blocks: the put waits for the producer's stream through an event that the
poller checks (the fix #4232 proposes for CkDevicePersistent).

Three rules from Aditya's channel measurements (Appendix A.4.1, A.4.2, A.4.6):

- **The cross-host verb is chosen per machine.** `CmiPut` across hosts is carried out either as a
  write with signal or as announce-then-read:
  1. the contributor sends a small announce message carrying its source registration;
  2. the owner's poller issues a get into the region;
  3. the decrement happens at the get's local completion on the owner's side.

  The write into device memory measured 10.9 GB/s (3.7 GB/s per direction with both hosts writing) against
  16.1 and 12.5 GB/s for a read on A100 nodes over Slingshot; Frontier runs every verb at 24 GB/s. The choice
  comes from a startup probe or a machine table. The API does not change. With announce-then-read the source
  stays in use until the owner's get completes, so the local counter fires later.
- **On cxi a put with signal is two network operations, unless native `fi_writedata` is enabled (5.4).** LCI emulates it as the write, then after the
  write's local completion a zero-length message carrying the 16-bit signal. Whether the data is visible
  when that message's handler decrements depends on the conditions of 5.4, which are not yet established.
- **Same-host device puts are batched.** A contributor's puts on one producing stream go into one
  `cudaMemcpyBatchAsync` with one completion event. The batch is flushed when the stream changes, at 256
  puts, when the entry method returns, and at idle. The contributor then decrements each counter once by
  the number of its puts in the batch. With one event per put, leanmd's copy engine ran 41 ms per step
  longer than with the batched zero-copy path.

### 8.3 Strided device data

Within a host, 3D peer copies handle the layout. Across hosts, a gather kernel on the sender's stream
into a registered device staging buffer, one put, and a scatter kernel launched by the owner PE at the
receiver (W7); the decrement waits for the scatter kernel's event.

### 8.4 Visibility of GPUDirect writes

The CPU seeing the completion does not by itself guarantee that later GPU work sees data the NIC wrote
into GPU memory. CUDA reports per device whether it orders such writes (device attribute
`cudaDevAttrGPUDirectRDMAWritesOrdering`) and provides `cudaDeviceFlushGPUDirectRDMAWrites` (CUDA 11.3 and
later) for devices that do not. The runtime queries the attribute once and, where needed, flushes once per
firing, on the owner PE, before running the action, not once per contribution.
For ROCm an equivalent has not been identified; open question. Nothing in charm, reconverse or LCI calls
a flush today (survey).

### 8.5 Device risks to retire first

1. A device-to-device put with immediate data on Delta (cxi, A100) and Frontier (cxi, MI250X); on Anvil
   the CUDA 11.4 dmabuf gap must be checked first.
   - Aditya ran it on Delta A40s over cxi (2026-10-19), using LCI's emulation.
   - First LCI must stop passing the zero-length signal message to `CmiAlloc(0)`, which returns NULL.
     He patched this locally; it is not upstream. Our put-with-signal prototype has run only on Anvil's ibv
     and would hit this on cxi.
2. Cost of the flush per firing.
3. CUDA IPC handles on Delta with one GPU visible per task fail (`+gpushm` abort in `hapiInit`,
   #4228's one failing step); the import must fall back to the NIC path, not abort.

## 9. Sender side and batched notifications

### 9.1 Puts need no acknowledgements

Source completion of a put is local: LCI reports it to the sender, the runtime decrements the local
counter. A neighbor that puts 6 faces uses one local counter with N = 6 and gets one notification,
which also gives #3966's one-callback-per-send result for free. No message goes back to the sender in
any locality.

### 9.2 Where a message must go back, batch it with the #3961 rule

The September work (charm #3960, #3961; charm_best_practices.md:2198) settled: notices whose only effect
is to release something (a registration, a staging block) are deferred, carried in a block of ids on the
next message going back to that PE, and sent alone only when the block fills (8 ids) or the PE goes idle;
notices that start application work (a source callback) are never deferred. Measured on Frontier at
2-16 KB: a standalone release cost about 4 us per message, a carried one about 0.3 us, within 1% of no
release at all.

In this design the cases that need a message back are:

1. a get whose source owner asked to be told (release-only: deferred; with a callback: immediate);
2. per-pair ring credits (section 12.2), which are release-only and need not be messages at all;
3. teardown acknowledgements (section 10).

The host zerocopy path's per-buffer acknowledgement messages (#3964, open) are outside this design but
the same id-block code would serve them.

## 10. Lifetime: setup, teardown, migration

### 10.1 Setup

- The owner creates regions and counters and sends their handles in entry methods (W3).
- **Imports register with the owner**. `CmiImportRegion` does three things:
  - records the remote keys;
  - attaches with xpmem, or opens the CUDA/HIP IPC handle, when the owner is on the same host;
  - sends one registration message to the owner process.

  The owner keeps the list of importing processes for each region. Handles may still be forwarded
  freely, but every process that imports one registers, so the owner always knows who may put. That is
  one message per importing process per region at setup, which W2 allows.
- **Caching.** Imports are cached keyed by (owner process, region id, generation).
- **Batch forms.** `CmiImportRegions` and `CmiReleaseImports` take a whole neighbor set and send one
  message per owner process. A rebuild after AMR touches every neighbor at once.

### 10.2 Teardown

1. The application stops putting into the region; its logic knows the last round.
2. Each importing process calls `CmiReleaseImport` (or the batch form). The call waits for that process's
   own puts into the region to complete locally, detaches the mappings, and sends one release notice
   to the owner. The notice is release-only, so it is carried per 9.2 when possible.
3. `CmiRegionDestroy` and `CmiCounterDestroy` complete when every registered import has been released.
   The runtime then deregisters the memory, frees runtime-allocated memory, and returns the counter
   index to the free list. Indices are reused least recently freed first, and only after the releases.
   A stray late signal can therefore hit a reused index only if step 1 was broken, which checking builds
   detect (section 11).

Two rules from Aditya's channels (Appendix A.4.4, A.4.8):
- **No message order.** No step of setup or teardown may rely on two messages arriving in order. His
  migration protocol had six such holes, and each one hung leanmd.
- **Urgent queue.** Setup, teardown, registration and release messages go through reconverse's urgent
  queue (`CmiSyncSendAndFreeUrgent`). In leanmd, ordinary messages waited 50-400 ms per hop behind deep
  scheduler queues.

### 10.3 Migration and other events that invalidate handles

Any migration is accompanied by teardown and setup of persistent communication, and that is the user's
responsibility; automation is a later question (10.4). There is a real tension between migration at any
time and persistent communication, and the manual says so (10.5). There are three cases.

1. **Migration at a collective point.**
   - Where: AtSync, plus the rebuild points where the application knows no put is in flight. Partners
     also change at AMR (Enzo-E at every adapt phase, SpECTRE, Quinoa's `resizePostAMR`) and at ChaNGa's
     domain decomposition.
   - Procedure: the object tears down before it moves, sets up after it arrives, and sends new handles.
     Its contributors release their imports and import again.
   - Cost: each migrated owner makes each of its importers release and re-register, about two messages
     per importer plus the handle delivery, done in parallel across neighbors.
   - Break-even: if a round saves s per contribution and a rebuild costs S per neighbor pair, the
     persistent family pays when objects stay in place for more than S/s rounds. For s, Aditya measured
     3-10 us for device puts against `nocopydevice`. S is to be measured in phase 4; tens of microseconds
     are expected.
2. **Migration at any time.**
   - Where: `migrateMe` (Enzo-E uses it), balancers without AtSync (Aditya's `+LBAsync`), Quinoa's
     non-blocking load-balancing mode, evacuation.
   - It is incompatible with owning a region: a put through a stale handle corrupts memory silently.
   - **Proposed guard (open question 5).** An object that owns a live region or counter
     cannot migrate. The attempt aborts in all builds, with a message naming the region.
   - A migrating contributor that owns nothing only has to wait for its own puts to complete and import
     again after it arrives.
   - The alternative, deferred, is a runtime hold on puts to a moving owner. The candidate is the
     resolve/closing/epoch protocol of the unmerged device channels (Appendix A.4.4, A.4.9), whose
     message-ordering holes have been found and fixed. It is kept for this purpose (section 13).
3. **Events that invalidate every handle**, i.e. disk checkpoint followed by restart.
   - The per-object release protocol cannot run after a restart.
   - The user tears down before checkpointing. Whether a restart on the same number of nodes could
     keep handles is deferred.
   - Option for later: write handles into the checkpoint as an "invalid" marker. A program
     that continues after the checkpoint keeps its regions; a restarted one must run setup again, and a
     put through an invalid handle aborts.

Two further items:
- **Shrink and expand.** A new implementation (Aditya's, expected in about two weeks) readjusts runtime
  structures in memory without checkpoint and restart. What regions need depends on whether process
  ranks are renumbered (open question 6):
  - handles name the owner by process rank and counter index;
  - if surviving processes keep their ranks, only moved objects rebuild; otherwise every object does;
  - on expand, new processes and their host peers must map each other's IPC segments before same-host
    puts work.
- **One hook for all cases** (proposed). Each object has a generation number that increases whenever
  its handles become invalid. The runtime tells the application which objects are affected: the
  migrated ones after migration, all of them after a restart or a shrink or expand. The application then
  re-runs setup for those objects.

### 10.4 Automation, later

What can be automated:
- owner-side destroy and re-create through PUP of `CkRegion` and `CkCounter`. This is clean only for
  runtime-allocated memory; user memory must be re-bound;
- delivery of the new handles to the registered importers.

What cannot:
- knowing that no puts are in flight;
- pausing contributors. The balancer's resume would have to wait until the re-imports are acknowledged.

The first version is therefore user-managed plus the guards. A helper comes after one or two
applications have done it by hand. Aditya's runtime-managed channels needed six ordering fixes, which
argues for this order.

### 10.5 Advice for the manual

"Use the persistent family when objects stay in place for many rounds between collective
load-balancing points. Use message sends for objects that migrate at arbitrary times, or when most
objects move at every load-balancing step. Readiness is your responsibility: follow one of the patterns
in 4.9."

## 11. Error checking

Readiness stays the user's responsibility (W1). The runtime's job is to detect violations
in checking builds and print an error that tells the user to use a message send there.

Production builds check only what is free:
- a put beyond a piece's capacity returns an error (the contributor knows the capacity from the handle);
- a put through a handle marked invalid (10.3, case 3) aborts;
- the migration guard (10.3, case 2), if adopted.

Error-checking builds add the following:
- **Count check.** Abort when a decrement finds `old < k` (4.2). It catches a put that arrives after its
  round fired and before the owner re-armed.
- **Round check.**
  - Each contributor numbers its own puts per counter.
  - Each contribution is followed by a small check message to the owner, carrying the counter index,
    the generation and the round number.
  - It catches two errors: a put from a later iteration counted in place of a missing one from the
    current iteration (the count stays positive, so the count check misses it), and puts through stale
    handles after teardown or migration (by the generation).
  - One message per contribution is acceptable only in these builds; the signal itself has no room for
    the generation.
- **Not caught by either check:** a put that lands after an automatic re-arm while the owner still uses
  the data. That is why re-arm is explicit by default (4.2).
- **Re-arm legality.** Re-arm before the round's action has started aborts (4.2).
- **Import checks.**
  - The source's memory kind matches the declared kind.
  - The layout fits inside the region.
  - `CMI_PIECE_VARSIZE` is accepted only on pieces with their own header space (4.6).

Every abort names the counter, the owner PE and the round. The message reads: "if the application's
logic cannot guarantee this transfer's readiness, use a message send here".

## 12. Messages: runtime-owned destination

### 12.1 API

`send(entity, payload, source = copy | hold(callback))`: today's entry-method invocation, where `hold`
replaces the `nocopy` and `nocopydevice` keywords (the runtime may read the source directly and calls
back when it may be reused). Variable-size and pointer-based data stay here (W5).

### 12.2 Three transport protocols, chosen by the runtime

| protocol | receiver memory | network crossings | receiver copies | range |
|---|---|---|---|---|
| eager | receive packets shared by all senders, per process | 1 | 1, out of the packet | up to the eager limit |
| per-pair ring | a ring per active pair of processes, registered once | 1 (write with immediate into the ring) | 1, out of the ring | mid sizes, pairs that repeat |
| rendezvous | a fresh `CmiAlloc` buffer per message | 3 | 0 | large |

Measured on Anvil (job 22247447, FINDINGS.md): raising the LCI packet size moves the eager limit exactly
to the packet size; at 64 KiB packets (pool bytes unchanged) a 32 KiB message drops from 10.87 to 7.92 us
and 48 KiB from 12.28 to 9.89 us, with 8 B latency and memory unchanged. Eager stays faster than
rendezvous up to about 115-120 KiB by extrapolation.

**Eager limit (proposed).** Make 32 or 64 KiB packets the reconverse default, after a many-to-one and
message-rate test (each small message occupies a whole packet; 64 KiB packets leave 8192 packets in
the same pool).

The same sweep on Delta (Slingshot, cxi; job 24045055, 2026-11-16) gives three results:
- **Where the jump falls.** The eager limit is again the packet size minus 76 B.
- **Rendezvous costs far more on cxi.** The step from eager to rendezvous is 10.0 / 9.7 / 7.1 / 6.8 us at
  8 / 16 / 32 / 64 KiB packets, against 3.9 / 3.6 / 3.0 / 1.9 us on Anvil. So raising the limit gains more
  there: 16 KiB goes from 15.1 to 5.5 us one way, 32 KiB from 15.8 to 9.1 us.
- **64 KiB packets have a cost on cxi.** They are 0.4 us slower at small sizes (8 B: 3.50 against
  3.10 us) and 1.1 us slower at rendezvous sizes, which Anvil does not show. On Delta, 32 KiB packets give
  most of the gain without that cost.

So the default may need to differ by fabric. A second step inside eager, between 16 and 24 KiB on Delta,
is libfabric's own cxi rendezvous threshold, confirmed on Delta (job 24046124):
- With `FI_CXI_RDZV_THRESHOLD=131072` the step disappears, and every eager size above 16 KiB is
  2.6-2.7 us faster. For example, 32512 B goes from 10.04 to 7.35 us.
- So reconverse should raise that threshold, at least to the LCI packet size, whenever it raises the eager
  limit on cxi, as it already sets the cxi receive-matching mode itself (reconverse #281). Filed as
  reconverse #334: set it to max(16384, packet size), with overwrite off so that a user's value wins.

**Per-pair ring** (MPI's middle protocol), for network pairs only (host pairs
already use the IPC pool).

**Deferred (proposed; open question 7).** On cxi, rendezvous costs about 10 us more than eager (above), so the
ring would gain more there than on Anvil; the deferral should be decided per fabric. On Anvil, with 64 KiB packets and eager faster than rendezvous up to about 115-120 KiB,
the ring's range would be 64-120 KiB. Rule M1 (17.2) also forces the ring to copy out, so it saves only
the rendezvous round trips in that range. It is built only if the eager change has been measured and a
gap remains. The design as it would be built:

- One ring per (sender process, receiver process, LCI device), in receiver memory registered once,
  created after a pair has sent a threshold number of mid-size messages (working value 16 within a
  window), within a per-process memory budget, and revocable.
- Contiguous byte ring: messages are written back to back with a length header, delivered in arrival
  order of their signals, copied out into `CmiAlloc` buffers, and the space is freed in order, so the
  credit is just the receiver's head position.

  The copy-out is required, not just chosen: applications keep received messages for an iteration or a
  phase, write into them, add references to them, re-send them, and in places never free them
  (section 17, rule M1). So ring slots are never handed to the application, which rules out design (b)
  of the earlier proposal.
- Credits return lazily: the receiver writes its head position into a word in the sender's memory with
  a plain RDMA write when half the ring has been consumed; the sender reads that word only when its
  cached head says the ring is full, and if still full sends this message by rendezvous instead of
  waiting. No credit message is ever sent.
- Signal: the ring's `rcomp` with the pair's ring index.

**Receiving in place for `nokeep` entries (proposed; open question 8).** An entry method marked `nokeep`
does not own its message. The runtime frees the message when the method returns, and the method may not
keep a pointer into it, modify it or free it (manual, section "Entry Method Attributes"). Every entry
method with marshalled parameters is `nokeep` automatically, except threaded, inbox and coroutine entries
(src/xlat-i/xi-Entry.C:214-218). So `nokeep` covers most of the traffic of the applications in section
17.

For these deliveries, rule M1 (17.2) does not apply, and the runtime may deliver straight from the memory
the data arrived in, then reclaim it when the entry returns. That memory can be an eager packet, a ring
slot, an IPC pool block or a rendezvous landing buffer, and in each case the receive-side copy is saved.
- **Saving.** A message of 64 KiB, copied out of a packet at about 33 GB/s, costs about 2 us, the same
  order as the network time saved by raising the eager limit.
- **Ring slots.** The ring may hand its slots to `nokeep` entries without copying, and copy out only
  for entries that own their message.
- **References.** Code that keeps a reference past the return (SDAG buffering a message for a later
  `when`, or an explicit `CmiReference`) leaves the reference count above one when the entry returns. The
  runtime then copies the message into a `CmiAlloc` block and releases the transport memory, so a kept
  message never holds a packet or a slot.
- **Cost.** A delivery held in transport memory occupies it until the entry runs, which includes the
  wait in the scheduler queue. With fewer, larger packets (12.2, eager limit), the runtime falls back to
  copying out when free packets run low.

### 12.3 The reconverse persistent API

Replaced, with no compatibility layer. It is recent (reconverse #192, 2026-07-30;
extended for NAMD in #205, 2026-08-24, with constants and compression-header variants), no one depends on
it beyond NAMD, and NAMD will be changed to use the persistent family directly; its PME transposes are the
put-and-decrement case (G2), the use CkDirect was built for. Its mechanism (fixed runtime-owned slots per
PE pair, written with a put) is design (b) of the per-pair ring in 12.2, so whatever of it survives does
so inside that ring, chosen by measurement. Consequence for the step 1 prototype: only its network part
(put with signal in the backend) is worth a PR; its changes to persist-comm.cpp speed up an API that is
going away. That network part also exists on Aditya's reconverse branch rate-aware-gpu-lb (b7b5139,
2026-10-19: `issueRputSignal`, a signal handler given rank, tag and size, `CmiRdmaPutRaw`). The PR should be
one version reconciled with his, and it needs the LCI zero-length-message fix to work on cxi (8.5).

## 13. What happens to today's APIs

| today | becomes |
|---|---|
| plain send, marshalled params | unchanged |
| `nocopy`, `nocopydevice` | `send(..., hold)` |
| `nocopypost` | kept for receivers that can choose a buffer only after seeing the message. Once #4285 is merged (approved 2026-11-16), the runtime registers the posted buffer for each message and releases it after, so `deregMode` has no effect there; one registration kept across messages is a region. |
| direct API (`CkNcpyBuffer` get/put) | region + import + `CkPut`/`CkGet` with N = 1 counters; kept as a thin compatibility layer |
| `CkDevicePersistent` | device region + `CkPut` + N = 1 counter; its IPC import code moves under the import |
| reconverse persistent | removed; NAMD moves to the persistent family (12.3) |
| CkDirect (removed 2018) | region + `CkPut` + N = 1 counter |
| `+ipc`, `+gpushm` | transport choices made by the runtime; options stay for tuning |
| `CkDeviceChannel` (unmerged, branch rate-aware-gpu-lb) | not added as a separate API; see below |

G5 holds only after phase 8. Until then the API surface grows: the persistent family is added,
while `nocopypost`, the direct API and `CkDevicePersistent` remain. The intended deprecation order is:
1. the reconverse persistent API, removed once NAMD moves (phase 8);
2. the direct API and `CkDevicePersistent`, deprecated once their compatibility layers over regions
   pass their tests, and removed a release later;
3. `nocopy` / `nocopydevice` as spellings of `send(..., hold)`, kept as aliases.

**The device channels** (Appendix A) are not added as a second API with a transport of their own. A
channel is a region with one piece plus a counter with N = 1, and its measured gains come from removing
per-message host work, which the persistent family removes as well. Two APIs for device put and notify
would undo G5. What is kept:
- **A convenience class**, `CkChannel`, of about a hundred lines over a region, a counter and a release
  credit (4.9, pattern 3). The examples on that branch (leanmd, jacobi3d, sph3d, moe) can port with
  little change.
- **The release credit**, as readiness pattern 3 (4.9).
- **The runtime-managed migration protocol**, as the candidate runtime hold for owners that move at any
  time (10.3, case 2; open question 5).
- **Its mechanisms**, already part of this design:
  - the cross-host verb chosen per machine (8.2);
  - same-host device puts batched with one completion flag (8.2);
  - control messages through the urgent queue (10.2);
  - same-host credits through shared memory (5.3, case b);
  - load-balancer edges keyed by the owner's manager (4.10);
  - the hardware-queue preset (Appendix A.4.10).

## 14. Tests and measurements

Reconverse tests (CTest, 1 and 2 processes, `+ipc` on and off):

- `region_counter_local`: same PE and same process; explicit re-arm with varying N, automatic re-arm,
  re-arm before the action started (must abort in checking builds).
- `region_counter_mixed`: one counter receiving contributions from the same PE, another PE, another
  process on the host and (on clusters) another host in the same round; thousands of rounds; checks
  every byte and that each round fires exactly once.
- `region_counter_early`: an error-checking build must abort on an early contribution, and on a
  later-round put counted in place of a missing one (round check, 11).
- `region_qd`: `CkStartQD` with puts in flight, a partly filled round, and a counter destroyed mid-round
  (4.8).
- `region_strided`: the three face kinds of 7.1 in every locality.
- device variants of the first two on Delta.

commbench additions (patterns, with JSON output as today):

- `halo3d`: 6 faces (2 contiguous, 2 row-strided, 2 element-strided), host and device, sizes from 16^3
  to 256^3 per block; compares put+counter against marshalled messages and against today's direct API.
- `transpose`: P contributors into one pencil buffer (NAMD's FFT case), P from 4 to 256.
- `manytoone` and `rate`: for the eager-limit decision.

Metrics: time from the first contributor's put to the owner's action, per round (median of at least 5
interleaved repeats); scheduler entries per round on the owner; bytes copied (recorded, as commbench
already does for copies).

## 15. Implementation phases

Each phase is a PR with tests; measurements gate the next phase.

1. **reconverse: put with signal in the backend.** The network part of branch persist-put-signal:
   `issueRputSignal`, the signal handler registry, `isPutSignalCapable`. It is reconciled with Aditya's
   b7b5139 (12.3). Test on Anvil (ibv) and Delta (cxi). Exit conditions:
   - the ordering condition of 5.4 is established for cxi, or LCI is made to request it;
   - LCI's zero-length signal message no longer goes through `CmiAlloc(0)` (8.5);
   - a test that detects a signal arriving before its data, not just payload checks: under many
     concurrent puts and `+lci_ndevices 2`, each payload ends in a sequence number that the handler
     checks when the signal arrives;
   - put-and-notify latency measured on cxi, both emulated (two network operations) and with
     `FI_CXI_ENABLE_WRITEDATA=1` where libfabric is 2.5 or later, before phases 2-3 rely on the Anvil
     numbers.
2. **reconverse: counters and contiguous host regions, network and same-process paths.** Counter table,
   firing, re-arm, error checks, `CmiPut`/`CmiGet`, imports without IPC.
3. **reconverse: same-host paths.** Counter table and `CmiRegionAlloc` in the IPC segment, IPC-block
   action messages in the owner's own segment, xpmem import of user regions; xpmem confirmed on Anvil,
   Delta and Frontier. The mixed-locality test.
4. **charm: `CkRegion`, `CkCounter`, `CkImport`, `CkPut`, `CkGet`**, PUP of handles, SDAG use, QD (4.8),
   LB records (4.10), tests in TEST_DIRS; commbench `halo3d` (contiguous faces) and `transpose`; the
   rebuild cost S of 10.3, including the batch import and release.
5. **strided layouts** (both sides), staging slots, scatter on the owner PE, per-block puts; commbench
   `halo3d` with all faces. Gated on `halo3d` beating messages with the raised eager limit (7.3).
6. **device memory** after the three checks of 8.5.
7. **messages:** eager default change (after the many-to-one and rate tests); per-pair rings only if a
   gap remains (12.2).
8. **NAMD's PME on the persistent family, then remove the reconverse persistent API;** compatibility
   layers for the direct API and documentation.

## 16. Open questions

1. Counter table: 4096 entries per PE by default, at most 65536 per PE (5.5). Is that enough for
   NAMD-scale pencil decompositions?
2. The per-block put threshold `B_put` (working value 4 KiB, 7.3) is to be measured. Is it a reasonable
   default until then?
3. The ROCm equivalent of the GPUDirect flush (8.4).
4. Names inside the persistent family: region, counter, import and put, or other terms (target, handle,
   contribute)? A C++ namespace (`ck::persistent::Region`) or prefixed names (`CkPersistentRegion`)?
5. Migration at any time (10.3, case 2). The proposed guard: an object that owns a live region or counter
   cannot migrate, and the attempt aborts in all builds. Is that acceptable, including under the
   asynchronous GPU-aware balancer if it is merged? Or should the runtime hold puts to a moving owner?
6. Shrink and expand (10.3): are process ranks renumbered?
7. Defer the per-pair ring (12.2) until the eager-limit change has been measured, and build strided
   cross-host puts only if commbench `halo3d` shows a gain over messages (7.3)?
8. Receiving in place for `nokeep` entries (12.2): worth the bookkeeping? Measure the copy saved against
   the packets held while a message waits in the scheduler queue, at the raised eager limit.

## 17. Existing applications (G6)

**What was surveyed (2026-11-16).** Four public codes, read at their current default branches. Each per-app
report has counts and file:line references.

| app | branch and commit | report |
|---|---|---|
| ChaNGa | master 8fd8207 (2026-11-13) | Appendix B.1 |
| SpECTRE | develop e734530 (2026-11-13) | Appendix B.2 |
| Quinoa | develop 3b14062 (2026-11-12) | Appendix B.3 |
| Enzo-E | main 4798687 (2025-08-13) | Appendix B.4 |

**What was not surveyed.**
- NAMD is not public (its GitLab needs a login); we port it ourselves (12.3).
- Also not surveyed: OpenAtom, ParaTreeT, Loimos, charm4py programs, and AMPI programs (AMPI itself is inside charm).

### 17.1 What they use, against what the redesign changes

| | ChaNGa | SpECTRE | Quinoa | Enzo-E |
|---|---|---|---|---|
| direct API, `nocopy*`, `CkDevice*` | none on master; the group's unmerged alltoall branch uses `nocopypost`, `CkPostBuffer`, `CkMatchBuffer`, `CkSendBuffer` | none | none | none |
| reconverse persistent API, CkDirect, CmiDirect | none | none | none | none |
| Converse sends, handlers, `CmiAlloc` | none; one `CmiReference` on a received cache fill | none | none | none |
| messages with their own pack/unpack | no | 1 (sent only by tests) | no | 9 classes, all the bulk data |
| received messages kept, written, re-sent | cache fills kept as storage for the iteration, written into, given references, re-sent | tests only | 3 kinds never freed | kept for a phase, freed later |
| priorities, `[expedited]` | used throughout (cache requests and fills) | none | tests only | none |
| relies on message order | no | no (one possible case at startup) | no (tested under randomized queues) | no |
| QD | `CkWaitQD`, `CkStartQD` | `CkStartQD` at every phase | debug option only | `CkStartQD` at adapt phases |
| when partners change | AtSync, domain decomposition | single AtSync indicator after QD, AMR | AtSync (also a nonblocking mode), AMR | AtSync, `migrateMe`, AMR insert and destroy |

So on their main branches none of the four calls an API that the redesign removes or reimplements.
G6 then depends only on the message family staying invisible. The rules below make that precise.

### 17.2 Rules the message family keeps

- **M1. A delivered message is an ordinary `CmiAlloc` block,** with the standard header and reference count,
  and the application owns it. This applies to entry methods that own their message. A `nokeep` entry
  method, which includes every non-threaded entry with marshalled parameters, may be given transport
  memory instead (12.2), as long as anything that is still referenced at return is copied then. It may be:
  - kept indefinitely;
  - written into;
  - given references with `CmiReference`;
  - re-sent;
  - freed with `CkFreeMsg` or `delete`;
  - leaked.

  No transport resource may stay tied to a delivered message. Eager already copies out of the packet,
  the ring copies out of its slots (12.2), and rendezvous lands in a fresh `CmiAlloc` buffer.
- **M2. Nothing above the transport moves.** Marshalling and `pack()` run inside the send call on the
  sender's thread. Enzo-E's messages point at live field arrays until `pack()` copies them.
  Same-PE delivery and local `[inline]` calls are not packed: Enzo-E reads the sender's data directly,
  and SpECTRE's BoundaryMessage test asserts pointer identity. Delivery to another PE in the same process
  is packed; Enzo-E would race on the sender's arrays if it were not. The envelope layout stays: ChaNGa
  calls `UsrToEnv` and `CmiReference`, and Enzo-E's `CkAllocBuffer` copies an envelope.
- **M3. Priorities, queueing modes and `[expedited]`** keep their meaning on every protocol. ChaNGa's
  cache depends on them for performance.
- **M4. No ordering promise is added or removed.** None of the four relies on order between two objects.
  - SpECTRE's two startup broadcasts from Main to its GlobalCache nodegroup are the one unguarded case
    (spectre-survey C9). Its readers are only inside SpECTRE's own Parallel layer, so the risk is low.
  - Reconverse already reorders through the IPC receive stack.
- **M5. QD counts every transfer,** puts included (4.8).
- **M6. Eager limit, ring and rendezvous choices are invisible** to the application. Only time changes.

### 17.3 Charm++ 9 build items found on the way (not caused by the redesign)

1. **SpECTRE.** It patches charmxi's generated `.def.h`/`.decl.h` files, with one patch set per Charm++
   version (7.0.0, 7.0.1, 8.0.0). `tools/patch_charm_modules.sh` skips the step silently when no set exists
   for the version, and recharm reports 9.0.0. A build against it therefore:
   - loses SpECTRE's limit of 64 nested `[inline]` calls, which guards against stack overflow;
   - fails to compile the message-typed `receive_data`, which only tests use.

   Either charmxi offers an inline depth limit itself, or SpECTRE adds a 9.0.0 patch set.

   **What the limit does** (src/Parallel/MaxInlineMethodsReached.hpp). It is not a depth counter. A
   `thread_local` count goes up on every local inline `receive_data` and is never decremented on return.
   At 64 the call returns true and the count is reset to 0. The patched proxy then takes the non-inline
   branch: the arguments are marshalled into a message and enqueued for the local element through the
   scheduler. So every 64th local inline call on a thread goes through the queue, nested or not. That
   bounds any nesting chain at 64 frames, at the price of an occasional deferral of a call that was not
   nested. Under `SPECTRE_PROFILING` the function always returns true, so every call goes through the
   queue and Projections sees it. Nothing aborts when the limit is reached.

   **Plan.** Adding a 9.0.0 patch set is a simple change for SpECTRE, so we remind
   them. A limit native to charmxi needs more thought. One option is a real depth count: increment before
   the direct call, decrement after it, and enqueue only when the depth exceeds a limit set at run time.
2. **Quinoa.**
   - **MPI.** It needs the `mpi` machine layer: it calls MPI from entry methods (Zoltan2 inside an `[exclusive]`
     nodegroup entry, Omega_h, `MPI_Allreduce`). To run on reconverse, MPI would have to be usable from entry
     methods in a reconverse run (reconverse #318 added interop at startup and exit only).
   - **Plan.** Quinoa will move through MPI interoperability: MPI phases alternate with
     Charm++ phases, and control returns to MPI between them. So its MPI calls have to move out of entry methods
     into the MPI phases. Moving it will need hands-on help from us.
   - **Pinned Charm++.** It is pinned to charm `374907eb8` on branch charm-7.0.1, and its `collidecharm.C`
     patch does not apply to recharm.
   - **Runtime internals.** It writes `chareIdx = -1` in 20 files to work around the 8.0.1 `chare_objs` table.
3. **Enzo-E.**
   - Its CMake requires `charmrun`, which a reconverse build does not have; the workaround is a
     `-DCHARM_RUN` cache setting.
   - It reads `CMK_RANDOMIZED_MSGQ` and `CMK_SMP` from `../tmp/conv-autoconfig.h`; the reconverse build has both.
4. **ChaNGa.**
   - Master already handles builds without `charmrun` (#242).
   - It includes headers from the charm build tree: `$CHARM/tmp`, the cache and `threadsafe_hashtable` directories, and `hybridAPI`.
   - It includes internal LB headers, and requires `CMK_LB_USER_DATA` (the `ChaNGa` charm target supplies it).

### 17.4 First adopters of the persistent family

| app | exchange | partners | size | how readiness and counting are done today |
|---|---|---|---|---|
| SpECTRE | DG mortar and ghost data, every step | 6-24 face neighbors | about 40 KB per face (N=10, 50 components) | inboxes keyed by temporal id; the nodegroup path already does an atomic decrement and one message at zero within a node |
| Enzo-E | ghost refresh | 26 neighbors (uniform level), up to 56 | face 320 KB (n=32, g=4, 10 fields), edges 20-40 KB, corners 5 KB; strided across fields | a `Sync` counter, and a neighbor handshake after each refresh (which is W1) |
| Quinoa | shared-node and ghost-cell exchanges; conjugate-gradient iterations | node or face neighbors | per shared node 8 B plus ncomp reals; payloads would have to be flattened, with the id list sent once per rebuild | per-exchange counters and SDAG `when` |
| ChaNGa | remote moments; the particle shuffle per node pair (alltoall branch) is variable-size, bounded only by particle counts, and is better left on messages | SFC neighbors; node pairs | fixed-size records; shuffle bounded by the particles that leave | particle counts or QD |

Three observations:
- **All four count arrivals by hand today.** The counter of G2 replaces code they already have, which is
  the strongest argument that G2 matches practice.
- **Layouts matter.** Enzo-E's faces and Quinoa's shared nodes are not contiguous, so the layout
  descriptors of section 7 decide whether those codes gain a copy-free path.
- **Partner sets change at AMR and domain decomposition,** not only at load balancing, so rebuilds must be
  cheap (section 10).
- **Migration decides adoption, not just static patterns**. Enzo-E uses `migrateMe`, and SpECTRE,
  Enzo-E and Quinoa change partners at AMR. So the guard of 10.3 and the measured rebuild cost S decide
  whether these codes can adopt the persistent family. A pattern like NAMD's, with fixed partners, does not
  test this.

## 18. Draft manual section: which call to use when

The text below is meant for the Charm++ manual, as a section "Choosing how to communicate".
- The advice on today's calls is in the manual PR #4287 (section "Choosing How to Communicate").
- The advice on regions and counters goes in with phase 4 (section 15).
- The numbers are one-way latencies between two nodes, one process each, measured on two networks:
  Anvil (HDR100 InfiniBand, 2026-11-15) and Delta (Slingshot-11, 2026-11-16, job 24046124). They apply to
  today's default eager limit of 8 KiB. When the limit is raised (12.2), the thresholds of the first three
  rows move up with it.

### Start from the data and its destination

| your situation | use | why, and what it costs |
|---|---|---|
| Data up to the eager limit (8 KiB today), or sent once, or with changing partners | **marshalled parameters** (plain entry methods) | At 8 B: 3.0 us one way on Anvil, 3.8 us on Delta, and 0.5 us within a process (a pointer is passed, no copy). It is also the right choice for pointer-based or variable data, through PUP. Above the eager limit a plain send pays the network layer's rendezvous: at 16 KiB, 9.9 us on Anvil and 17.6 us on Delta. |
| You want to keep, modify or forward the received data without copying it | **a message** (`CMessage_` class) | You own a received message: keep it, write into it, re-send it, or free it. Marshalled parameters are owned by the runtime (below). |
| A source buffer above the eager limit that you can leave untouched until told | **Zero Copy Entry Method Send API** (`nocopy` parameter, `CkSendBuffer(ptr, callback)`) | It saves the sender's copy into a message, at the cost of a callback and a round trip. Below the eager limit it is 3-4 times slower than a plain send (8 B: 10.2 against 3.4 us on Anvil, 13.4 against 3.8 us on Delta). Above it, it is 0.84-0.96x a plain send on Delta (16 KiB to 1 MiB), where rendezvous is costly, and 1.06-1.22x on Anvil. Within one process it copies once where a plain send passes a pointer. |
| The receiver can choose where the data goes only after seeing the message (its size or destination depends on the content) | **Zero Copy Entry Method Post API** (`nocopypost`, `CkPostBuffer` / `CkMatchBuffer`) | The data lands in the receiver's buffer, at the cost of a round trip for the metadata. It costs about what the Send API costs. |
| Data that must start and end in your own buffers, repeated with the same partners | today: **Zero Copy Direct API** put or get (`CkNcpyBuffer`), with the handles exchanged once; after phase 4: **a region and a counter** | Against messages plus your own copies, a put is faster from 16 KiB on Delta (1.45x at 16 KiB, 2.31x at 1 MiB, 3.46x at 16 MiB) and from 64 KiB on Anvil (1.33x, 1.69x at 1 MiB, 2.30x at 16 MiB). At 4 KiB and below, plain messages still win (8 B: put 7.4 against 3.8 us on Delta); put with signal (phase 1) removes the acknowledgement behind that. Get and put cost about the same: a get is 2-6% faster on Delta from 4 KiB to 256 KiB, and a put is up to 7% faster on Anvil. |
| Many senders filling one buffer, with one action when all have arrived (halo faces, FFT transposes) | after phase 4: **one counter with N contributions** (put and decrement); today: messages, counted by hand | One scheduler entry for the whole round instead of one per message, and no copies. Every application surveyed in section 17 counts arrivals by hand today. |
| GPU buffers, one transfer | **`nocopydevice`** (`CkDeviceBuffer`) | Device-to-device without staging through the host where the hardware allows it. |
| GPU buffers, repeated with the same partners | today: **`CkDevicePersistent`**; after phase 6: **a device region and a counter** | The handles are opened once. No measurement against `nocopydevice` yet; the unmerged device channels (Appendix A), which add a fixed buffer and a release credit, measured 3-10 us less per message. |

### Rules that apply whichever call you use

- **Do not modify or keep marshalled parameters.** Entry methods with marshalled parameters are
  `nokeep`: the runtime owns the buffer, frees it when the method returns, and for a broadcast hands
  the same buffer to every receiver in the process. Copy what you need to keep. Use a message if you
  need to own the data.
- **The kind of chare does not matter for speed.** Chares, groups, node groups and arrays differ by
  3% or less; choose by design.
- **Between processes on one host, run with `+ipc`.**
  - Messages up to the IPC cutoff (32 KiB by default) are 3-9 times faster: 8 B goes from 3.08 to 1.00 us
    on Anvil, and 16 KiB from 15.2 to 1.9 us on Delta.
  - Above the cutoff a message goes through the network card: 64 KiB takes 17.4 us on Delta.
  - The zero-copy APIs gain less.
- **For regions, readiness is yours.** A put lands whenever it arrives. Use one of the three patterns
  of section 4.9: a reduction between iterations, two buffers by parity, or a release credit.
  Error-checking builds detect most violations (section 11).
- **For regions, migration is yours too.** Use them for objects that stay in place for many rounds
  between collective load-balancing points. Tear down before an object moves and set up again after
  it arrives (section 10.5). Use messages for objects that migrate at arbitrary times.

---------------------------------------------------------------------------------------------------

## Appendix A. Device channels on branch rate-aware-gpu-lb (Aditya Bhosale; read 2026-11-16)

### Sources

None of this is merged, and there is no PR for it.

- **Charm++ branch** `rate-aware-gpu-lb` in charmplusplus/charm.
  - 358 commits by Aditya that are not on `recharm` or `classic`.
  - Tip 625a6d48f (2026-11-09); merged with recharm at 5e3f7e418 (2026-10-30).
  - The plan and measurement log is `PERSISTENT_COMM_PLAN.md` (1282 lines, 2026-10-17 to 2026-11-07).
  - Also on the branch: `GPU_INITIATED_COMM_IDEAS.md`, `GPU_IPC_PROTOCOL_SWITCH_PLAN.md`, and
    `examples/charm++/cuda/gpudirect/moe/README.md` ("Measured 2026-11-07").
- **Reconverse branch** `rate-aware-gpu-lb` in charmplusplus/reconverse:
  - b7b5139 (2026-10-19): put with remote completion, raw put, signal handler.
  - 488c57e (2026-09-16): urgent queue for expedited sends.
- **Machines.** NCSA Delta: A40 nodes (gpub*), 4 GPUs per node, Slingshot-11 (cxi). Later also A100 nodes and Frontier.
  Most comparisons used two nodes with the control in the same allocation.

### A.1 What was built

"Persistent device communication" was built in two stages.

1. **Make `CkDevicePersistent` work in his branch's configuration** (2026-10-17/18).
   - It now handles pool and vmm memory, using the IPC import cache.
   - It works across hosts through `CmiIssueRput` / `CmiIssueRget`, with a new opMode `CMK_DEVICE_PERSISTENT_OP`.
   - All 7 arms passed.
   - Charm #4228 (merged 2026-11-12) later did the cross-host part independently on recharm.
2. **A new object, `CkDeviceChannel`** (src/ck-core/ckrdmadevice.{h,C} on the branch).
   - The receiver owns one fixed destination buffer per channel.
   - The sender calls `put(src, bytes, stream)`.
   - The receiver's entry method gets `{seq, bytes}`, then calls `release()`, which gives the sender a credit for its next put.
   - Double buffering means opening two channels on two buffers.
   - Migration is handled by the runtime, with no re-exchange by the application. The protocol uses resolve,
     closing, ack and credit messages, with generations and epochs.

How a put travels depends on locality:

| locality | data | notification |
|---|---|---|
| same process | `cudaMemcpyAsync` on the sender's stream | stream callback, then a message to the element |
| same host | a peer copy into the IPC-mapped buffer | as above. From 2026-10-19 the credits and arrival notes go through a per-host POSIX shm segment instead of messages. |
| other host | either a write with the note as immediate data (commit 88d4f724d), or announce-and-read: the sender announces, the receiver gets and then sends DONE | the default changed twice; see A.4.1 |

### A.2 The measurements

**jacobi3d** (examples/charm++/cuda/gpudirect/jacobi3d, new `-k` channel mode). 32 blocks on 8 A40s over 2 nodes,
200 iterations after 10 warm-up, average per iteration. Checksums matched the host-staged reference in every mode.

| grid (face size) | zero-copy (`nocopydevice`) | channels | change |
|---|---|---|---|
| 512^3 (128 KB) | 1162.1 us | 703.7-719.9 (three runs) | -38% |
| 1024^3 (512 KB) | 5796.8 | 5480.6 | -5% |
| 2048^3 (2 MB) | 42869 | 42413-42562 | -1% |

- One node (16 blocks, 4 A40s), 512^3: 1633.7 vs 1412.6 us (-13.5%).
- An earlier two-node run using the read path gave 1291.0 vs 702.5 (-46%). Zero-copy itself varied between 1162 and 1291 across runs.

**Microbenchmark** (benchmarks/charm++/cuda/gpudirect/channel), one-way latency in us, zero-copy / channel.
This is the final protocol of 2026-10-18 21:50 (announce-and-read across hosts, libfabric cache monitor fixed, LCI built with -O3).

| layout | 1 KB | 64 KB | 512 KB | 4 MB |
|---|---|---|---|---|
| two nodes | 22.3 / 19.0 | 25.2 / 21.9 | 45.0 / 41.9 | 268 / 260 |
| two processes, one node | 27.1 / 17.5 | 28.7 / 20.7 | 45.0 / 36.6 | 193.0 / 192.7 |

Host time per message:
- Sender: zero-copy 2.3-4.7 us; channel 0.1 us (0.4 us with batching).
- Receiver: zero-copy receive issue averaged about 4.5 us (sph3d step 0).

**Applications:**
- **leanmd** (16x8x8 and 8x4x4, with and without load balancing): a tie, or within noise. Message handling is under 1% of a 1.5 s step.
  The 0.7-2.3% loss without LB disappeared once puts were batched (2571.6 vs 2587.5 ms/step).
- **sph3d:** bound elsewhere. The zero-copy host work a channel could remove was 3-5% of the step.
- **moe:** tokens sent as channel puts. This was not compared against zero-copy in isolation.

So "significantly faster" holds in two places:
- latency at small and mid sizes, 3-10 us per message, with the largest gain between processes on one host (27 to 17.5 us);
- a face exchange that is latency-bound: jacobi3d at 128 KB faces, -38% per iteration.

It does not hold at 512 KB and above, or in codes where communication handling is a small share of the step.

### A.3 Where the gain comes from

**Zero-copy device send**, per payload (plan section 0.3):
- Sender work in `CkRdmaDeviceOnSender`: location lookup, mode choice, registration or IPC export, slot claim.
- An envelope carrying an 80-byte descriptor per buffer.
- Receiver work in `CkRdmaDeviceIssueRgets`: admission, landing, copy issue, completion.
- An ack message back, then the sender's callback.
- Across hosts, an rget issued by the receiver.
- Within a host under `+gpushm`, a two-copy staging path with an event handshake.

**Channel put:** one copy or write, then one notification. The destination is fixed at open, so there is no
per-message descriptor, lookup, landing decision or ack.

This is the case for G1 (section 1), measured on devices.

### A.4 Findings that bear on our design

#### A.4.1 Whether write or read is faster across hosts depends on the machine

Measured with rma_bw, 128 MB transfers:

| machine | write into device memory | read into device memory |
|---|---|---|
| A100 pair, Slingshot | 10.9 GB/s alone; 3.7 GB/s per direction with both hosts writing | 16.1; 12.5 |
| Frontier (MI250X, Cassini) | 24 GB/s for every verb and memory class | 24 GB/s |

His branch therefore made announce-and-read the cross-host default again (2026-11-07). The write path stays
behind `CHARM_CHAN_WRITE=1`.

On the A40 pair, the read also beat the write at every size. At 4 MB the read took 247 us against 345 us for the write.

#### A.4.2 Slingshot (cxi) has no RMA write with immediate data

With CXI 2.3.1:
- forcing `fi_writemsg(FI_REMOTE_CQ_DATA)` returns "Flags not supported";
- putting `FI_REMOTE_CQ_DATA` in the hints returns ENODATA.

By default LCI turns `support_putimm` off for cxi. It emulates a put with remote completion as the write,
followed after its local completion by a zero-length message that carries the immediate data. So on Delta,
Frontier and Perlmutter-class machines, put with signal costs a message after the write completes, not one operation.
The LCI that reconverse pins (a8127fa) turns it on when `FI_CXI_ENABLE_WRITEDATA=1` is set and libfabric
is 2.5 or later (backend_ofi.cpp:75-79, 220-227). The libfabric versions on Delta and Frontier have not
been checked, and the native path has not been tested.

#### A.4.3 A defect on that path affects our step 1 prototype

LCI allocates the zero-length message through the user allocator. That is reconverse's `CmiAlloc`, which
refuses 0 bytes and returns NULL ("CmiAlloc: size <= 0"). With several PEs per process, the result was
segfaults and handlers run with the wrong index.

Aditya patched LCI on Delta. The patched build is under his home directory there; the branch shows no upstream LCI change.

Our prototype (reconverse 2d674d3, local branch persist-put-signal) has only run on Anvil's InfiniBand, where write with immediate is
native. On Slingshot it would hit this unless LCI or `CmiAlloc` is fixed first.

#### A.4.4 Messages between two PEs are not delivered in order

This holds across nodes (adaptive routing on Slingshot, below LCI) and within reconverse (several PEs poll the
device and push what they receive). His control protocol had six ordering holes, and each one hung leanmd after a
migration wave. Examples:
- a credit arrived before the resolution it belonged to;
- resolution 2 arrived after resolution 3;
- a resolve crossed a close ack.

Our design has no control message in the steady state (W1: no credits). Teardown (section 10) must still be written
so that no step relies on the order of two messages.

#### A.4.5 The credit cost 12 us until the libfabric registration cache was turned on

`FI_MR_CACHE_MONITOR=disabled` had been set in his scripts, so every small send mapped and unmapped its buffer
(`cxip_map`). A bare Converse send cost 10.3 us. With `uffd` it cost 1.26 us, and the credit cost 0.5 us across
hosts and 2.2 us within one host.

charm-notes/machines/frontier.md already records the cxi monitor settings. Under W1 our design has no credit at all.

#### A.4.6 Device puts must be batched on the copy engine

At first each put had its own stream callbacks. The callbacks wrote a flag with `cuMemsetD32Async`: 2.26 flag writes
per put, against 0.8 per message for zero-copy. Zero-copy batches its landings with one `cudaMemcpyBatchAsync` and
one flag per batch. On the GPU that set the step, the unbatched puts cost 41 ms per 2.5 s step, which was the size
of leanmd's loss.

Batching closed the gap. A PE's puts on one producing stream go into one `cudaMemcpyBatchAsync` with one flag.
The batch is flushed:
- when the stream changes;
- at 256 puts;
- when the entry method returns (from `CkCallstackPop`, so that the balancer charges the copies to the right object);
- at idle.

For section 8.2: batch same-host device puts the same way. A contributor whose batch holds k puts to one counter then decrements it once by k.

#### A.4.7 Same host: a shared-memory word instead of messages

He used one per-host POSIX shm segment, with one area per PE (8192 credit slots and a ring of 8192 arrival notes).
That is case (b) of sections 5.2-5.3, done with a separate segment. Two traps he hit:
- **Leftover segment.** The segment relied on zero pages as its empty state, so a killed job step left a live one
  behind and the next run used it. He now names it per job step.
- **Initialization order.** The channel initialization ran before `CmiInitCPUTopology`, so "same host" silently
  meant "same process", and the shared-memory path never engaged in a whole set of runs.

Reconverse's IPC segment is set up after the topology. Our implementation must check the same ordering.

#### A.4.8 Control messages need the urgent queue

Ordinary messages sat behind scheduler queues 50-400 ms deep (448 objects per PE in leanmd 16x8x8). Each hop of a
migration protocol cost that much: the sender held a put for 305 ms on average, and up to 3.5 s.

Sending closing, ack, resolution and DONE with `CmiSyncSendAndFreeUrgent` fixed this; reconverse serves its urgent
queue first in every scheduler iteration (488c57e). Our setup, teardown and import messages should be urgent for the same reason.

#### A.4.9 Migration while other objects run

His runtime-managed migration caused most of the bugs and, until fixed, a 1-6% loss in leanmd 16x8x8. Kale's W4
(tear down and rebuild at AtSync) avoids that protocol. However, the GPU-aware asynchronous balancer on the same
branch (`+LBAsync`) moves elements while their neighbors keep stepping, with no AtSync.

If that balancer is merged, regions owned by migrating elements need one of two things:
- a hold on puts to the moving element, which the application cannot provide without runtime help;
- a version of his resolve/closing protocol.

This is a decision for later, recorded here so the design does not assume AtSync everywhere.

#### A.4.10 Two smaller items

- The runtime sets `CUDA_DEVICE_MAX_CONNECTIONS=32` before `cuInit`, from `charm_main` before `ConverseInit`.
  Zero-copy leanmd gained about 5% from it.
- On cxi, registering 2 GB or more fails without `FI_RMA_EVENT`.

### A.5 Overlap with our work

- **The same reconverse backend primitive exists twice.**
  - Aditya's b7b5139 (2026-10-19, unmerged) adds:
    - `comm_backend::issueRputSignal` and `setSignalHandler`, whose handler receives (rank, tag, size);
    - `CmiIssueRputSignal` and `CmiSetRputSignalHandler`;
    - `CmiRdmaPutRaw`, a put with no operation record or ack.
  - Ours, 2d674d3 (2026-11-15), adds `issueRputSignal`, a handler receiving (src_rank, signal),
    `isPutSignalCapable` and `getMaxPutSignal`.
  - Both use a 16-bit tag.
  - The step 1 reconverse PR should be one PR that reconciles the two, made with Aditya. His version reports the
    size; ours lets a caller find out whether the backend supports the signal.
- **`CkDeviceChannel` is the device case of G1** (put and notify, N = 1), with credit flow control and
  runtime-managed migration added. The persistent family generalizes it to a counter with N contributions, host
  and device memory, and every locality, and drops the credit by W1. His measurements are the evidence that G1
  pays on devices at small and mid sizes.

---------------------------------------------------------------------------------------------------

## Appendix B. Application surveys (2026-11-16)

Each survey inventories one application's Charm++ and Converse communication calls, read from a clone of its
default branch; nothing was built or run. Section 17 is the synthesis. Within a survey, "B.k.n" refers to
its own numbered categories.

### B.1 ChaNGa

- **App:** ChaNGa. **Repo:** https://github.com/N-BodyShop/changa, master, commit `8fd8207654667b086c368b73a4002ba1fc437fd4` (2026-11-13 10:20 -0800, "Makefile: handle charm builds without charmrun (reconverse) (#242)"). Shallow clone: a shallow clone.
- **Utility:** https://github.com/N-BodyShop/utility, master, `a97fe5d2e32bab23de89069fa66830ef799eb5e8` (2025-09-17), shallow clone. Its only Charm++ use is `PUP::er` operator| overloads, `charm++.h` includes, and CmiFopen/CmiFclose (structures/TipsyReader.h:275-308). It does no messaging.
- **Scanned:** 256 files (173 ChaNGa, 83 utility): .ci/.cpp/.h/.c/.cu/.hu, configure.ac, cuda.ac, Makefile.in, cuda.mk.in, CI yml, READMEs. Read and grep only, 2026-11-16.
- **Excluded:** `.git/`, `utility/xdr/` (vendored XDR library), `data/` (tables). `changa/OLD/` (5 files, not built) was scanned, but its hits are left out of the counts. `cuda/intrinsics/*.hu` (CUDA shuffle wrappers) has no Charm++ use. The generated `configure` was not scanned.

#### Summary
- **Main loop:** Main is a threaded mainchare. It broadcasts to the TreePiece 1D array, `treeProxy.X(..., CkCallbackResumeThread())` (about 183 broadcast sites, 327 ResumeThread uses). Results come back through `contribute` (134 calls) and 12 custom reducers.
- **Gravity and SPH remote data:** all of it goes through CkCache. A high-priority CkCacheRequestMsg goes to the owning TreePiece. A CkCacheFillMsg comes back to the requester's CkCacheManager group, which calls [local] entries on the requesting TreePiece. A received fill message stays in the cache, as its storage, until the end of the iteration. The default node cache is charm's process-shared CkTreeCacheManager.
- **Walk chunking:** the walks are split by prioritized self-messages (ComputeChunkMsg, dummyMsg). The same message object is re-sent each time.
- **Domain decomposition:** a Sorter chare runs histogram reductions. Particles then move in varsize ParticleShuffleMsg sends. The ORB variant uses marshalled arrays.
- **Tree build:** fixed-size marshalled `requestRemoteMoments` / `receiveRemoteMoments` messages to SFC-neighbor pieces.
- **Node level:** the DataManager nodegroup and PE-level merges work through shared memory (ckLocalBranch, CmiLock). The CUDA path uses HAPI.
- **LB and checkpoint:** AtSync with ChaNGa's own CentralLB subclasses, which use LB user data. Checkpoints go to disk.
- **Master has no zerocopy, no Converse sends, no handlers and no CmiAlloc.**

#### B.1.1 Interface files
- **11 .ci files:** ParallelGravity.ci (642 lines), Reductions.ci, 7 LB modules (HierarchOrbLB, MultistepLB, MultistepLB_SFC, MultistepLB_notopo, MultistepNodeLB_notopo, Orb3dLB, Orb3dLB_notopo), and testonestar/testimf.ci and testsnia.ci (PUPables only).
- **Chare kinds:**
  - mainchare 1 (Main); chare 1 (Sorter).
  - groups: 7 app groups (ProjectionsControl, ReductionHelper, Writer, PETreeMerger, DumpFrameData, IntraNodeLBManager, PEList [CUDA]) and 7 LB groups.
  - charm templated groups the app instantiates: `group CkTreeCacheManager<KeyType, GravityNodeTraits>` (ParallelGravity.ci:8) and 3 `CkCacheManager<KeyType>` instances.
  - nodegroup 1 (DataManager).
  - arrays: 2, both `[1D]`: TreePiece, and LvArray (3 instances bound to TreePiece, used only for reductions). No custom index.
- **Messages (6):** dummyMsg, ComputeChunkMsg, fillGPUMsg [CUDA], `ORBSplittersMsg{pos[],dim[]}`, `ParticleShuffleMsg{loads[],parts_per_phase[],particles[],pGas[],pStar[]}`, `BucketMsg{buckets[],particles[]}` [PUSH_GRAVITY, off by default]. No `[packed]` or `[varsize]`.
- **Entry attributes:**
  - threaded 7 (Main 6, Sorter::startSorting); notrace 6.
  - local 5: receiveNodeCallback, receiveParticlesCallback, receiveParticlesFullCallback (ParallelGravity.ci:519, 527, 528), registerTP, finishedTPWork.
  - expedited 3: fillRequestNode, fillRequestParticles, fillRequestSmoothParticles (:518, 524, 525).
  - nokeep 3: evaluateBoundaries on ReductionHelper :242 and on TreePiece :342, newOrder :416.
  - inline 1: requestRemoteMoments :499. It applies only without MERGE_REMOTE_REQUESTS, so the default build has none.
  - immediate, sync, exclusive, reductiontarget, createhere, createhome, aggregate, whenidle, python, accel: **none**.
- **nocopy, nocopypost, nocopydevice: none. SDAG: none.**
- **Templates:** no templated entry methods; the only templated chares are the charm CkCache groups above. 74 `PUPable` declarations.

#### B.1.2 Messages in C++
- **CMessage_ subclasses: 6** (ParallelGravity.h:225-347). The app also uses charm's `CkCacheRequestMsg<KeyType>` and `CkCacheFillMsg<KeyType>`.
- **CkFreeMsg 4:**
  - CacheInterface.cpp:46 frees a kept particle fill when the cache evicts it.
  - CacheInterface.cpp:140 frees a smooth fill after copying it.
  - CacheInterface.cpp:444 frees a node fill once per node.
  - ParallelGravity.cpp:1693 `CkFreeMsg(cb.thread_delay())`.
- **CkCopyMsg 1:** CacheInterface.cpp:239 copies an outgoing fill once per buffered requester. **CkAllocMsg and CkReferenceMsg: none.**
- **Received messages stored instead of deleted:**
  - Fill messages become cache storage until the end of the iteration (CacheInterface.cpp:37-41).
  - Node fills (CacheInterface.cpp:357-445): the nodes live inside the message buffer. The message pointer is stored in the 8 bytes before each node, the vtable pointer is overwritten, and offsets are turned into pointers.
  - `std::vector<ParticleShuffleMsg*> incomingParticlesMsg` holds shuffle messages until all particles arrive (TreePiece.cpp:1210, freed at 1266).
  - `savedLiveVizMsg` (TreePiece.cpp:6485).
- **Received messages re-sent:**
  - To self: ComputeChunkMsg (TreePiece.cpp:4522); dummyMsg in nextBucket :4132, calculateEwald :4219, and smooth.cpp:440, 957, 1189.
  - To another TreePiece: a received CkCacheRequestMsg is forwarded when the node is not local (TreePiece.cpp:3331, 3355, 3362).
- **Static pack(), unpack() or alloc(): none.** CkSetRefNum and CkGetRefNum: **none.** Envelope: UsrToEnv 1 (CacheInterface.cpp:399).
- **Priority:**
  - CkPriorityPtr 17 lines (15 sets, 2 debug prints) and CkSetQueueing(CK_QUEUEING_IFIFO) 15, in CacheInterface.cpp, TreePiece.cpp, smooth.cpp, DataManager.cpp.
  - Values: node requests -110000000, particle requests -100000000, fills -10000000, positive per-piece values for self-messages.
  - CkEntryOptions::setPriority 7 sites (TreePiece.cpp:2857-3528, Compute.cpp:2119) for the moments exchange.

#### B.1.3 Zerocopy and RDMA
- **None on master.** None of CkNcpyBuffer, CkSendBuffer, CkDeviceBuffer, CkDevicePersistent, CkNcpyMode, CK_BUFFER_, CkPostBuffer, CkMatchBuffer, CkRdma*, CkDirect or CmiDirect, and none of the words ncpy, zerocopy or zero-copy. The only matches are `strncpy`. The alltoall branch adds zerocopy (see the end).

#### B.1.4 Converse-level calls
- **Messaging:** no Cmi*Send, CmiSync*Broadcast, CmiPushPE or CmiPushNode: **none.**
- **Handlers and memory:** CmiRegisterHandler, CmiSetHandler, CmiAlloc, CmiFree: **none.** CmiReference **1**: CacheInterface.cpp:399 `if (!isRoot) CmiReference(UsrToEnv(msg))` adds one reference to a received fill per non-root node. This happens only when CHANGA_SMPCACHE=0.
- **Scheduler** (CsdEnqueue, CqsEnqueue, CmiDeliverMsgs, CsdScheduler, CsdSchedulePoll): **none.** **Persistent API: none.**
- **Locks:**
  - CmiCreateLock 7, CmiLock 31, CmiUnlock 37, CmiDestroyLock 3.
  - The nodegroup `__nodelock`: 12 live uses (DataManager.cpp:171-1409).
  - They protect the star and memory logs, the pinned-host pool (MemoryPool.cpp) and the DataManager tree merge.
- **Barriers and fences: none.**
- **Topology:**
  - CkMyNode 29, CkNumNodes 12, CkNodeFirst 5, CkNodeSize 5, CkMyRank 4, CmiMyNode 3.
  - TopoManager in 6 LB files.
  - IntraNodeLBManager.cpp:55 reads another PE's Cpv with `CpvAccessOther(tpGravDone, i)`.

#### B.1.5 Aggregation, multicast and delegation
- **TRAM and comlib: none.**
- **Multicast, PUSH_GRAVITY only (off by default):**
  - TreePiece.cpp:4796-4798: `CProxySection_TreePiece::ckNew` and `ckSectionDelegate(CkMulticastMgr)`.
  - TreePiece.cpp:4886, 4898: CkGetSectionInfo and section `mgr->contribute`. BucketMsg derives from CkMcastBaseMsg.
- **Other delegation: none.** CkMulticast is still linked as a module.

#### B.1.6 Collectives
- **contribute(: 134.** TreePiece.cpp 66, InOutput 14, Sph 13, collision 11, DataManager 9, others.
- **Custom reducers:**
  - `addReducer` 14: 12 in Reductions.cpp:182-194 and 2 under COSMO_STATS at ParallelGravity.cpp:4501-4502.
  - `CkReductionMsg::buildNew` 8. No tupleElement or setElement.
  - Built-in reducers: sum_double, sum_int, sum_long, max_int, max_long, concat, random, nop, logical_or.
- **CkCallback forms:**
  - CkCallbackResumeThread 327; entry-method `CkCallback(CkIndex_X::m(), proxy)` 38; `CkCallback::resumeThread` 3 plus `thread_delay` 1.
  - Function pointer 1 (DataManager.cpp:1309, HAPI completion); `CkCallback::ignore` 1; liveViz function index 1.
  - No reductiontarget. CkExit 18.
- **Broadcasts:**
  - About 183 `treeProxy.<entry>(` sites, 83 of them in ParallelGravity.cpp.
  - About 19 `dMProxy.<entry>(` nodegroup broadcasts.
  - Also writeProxy, prjgrp and the cache groups (collectStatistics).
  - Sorter broadcasts ORBSplittersMsg and the evaluateBoundaries key arrays to the ReductionHelper group (always built, Makefile.in:130).

#### B.1.7 Local access and node-level parallelism
- **ckLocal( 23.** Main reads `treeProxy[0].ckLocal()` at ParallelGravity.cpp:1499, 1567, 2575-2578, which assumes element 0 is on PE 0. The bound LvArrays are reached through ckLocal to contribute.
- **ckLocalBranch( 56, CkLocalBranch( 1, CkLocalNodeBranch( 19.**
- **CkLoop (SMP only):** CkLoop_Init 1 (ParallelGravity.cpp:1455), CkLoop_Parallelize 2 (TreePiece.cpp:4281, 4687).
- **`#pragma omp`, std::thread, Kokkos: none.**

#### B.1.8 Migration, LB, checkpoint, QD, futures
- **Load balancing:**
  - AtSync: 2 calls (TreePiece.cpp:5517, 6613; 6 grep matches with comments).
  - ResumeFromSync is an entry (ParallelGravity.ci:532). usesAtSync=true (ParallelGravity.h:1377, 1451). No migrateMe.
  - LB internals: about 40 lines use `CkpvAccess(_lb_obj_index)` with getObjUserData and LBRegisterObjUserData. Also setObjTime and LBTurnInstrumentOn/Off.
- **PUP:** `pup(` 228; PUPable: 74 in .ci, 77 PUPable_decl/abstract.
- **Checkpoint:** `CkStartCheckpoint(file, Main::restart, true)` after CkWaitQD (ParallelGravity.cpp:3283-3285). No memory checkpoint or CkRestart.
- **Quiescence detection:**
  - CkStartQD 2 live (TreePiece.cpp:732, 741: `shuffleAfterQD` when the splitters are reused).
  - CkWaitQD 4 live (InOutput.cpp:1457; ParallelGravity.cpp:1689, 2085, 3283).
- **Futures and threads:** CkFuture, CthSuspend, CthAwaken: **none.**
- **Load balancers:**
  - Default: `MultistepLB_notopo` (configure.ac:351, testcosmo/runscript `+balancer MultistepLB_notopo`).
  - Also built: MultistepLB, MultistepLB_SFC, MultistepNodeLB_notopo, Orb3dLB, Orb3dLB_notopo, HierarchOrbLB.
  - Linked from charm: RefineLB, GreedyLB, RotateLB.

#### B.1.9 GPU (`--with-cuda`; -DCUDA -DSPCUDA -DPINNED_HOST_MEMORY)
- **HAPI:** `hapiAddCallback(stream, CkCallback*)` 11 live sites (HostCUDA.cu:132-1372). HAPI_TRACE_BEGIN/END 13 each, HAPI_MEMPOOL ifdef. hapiWorkRequest and hapiEnqueue appear only in commented-out code (HostCUDA.cu:555-564).
- **CUDA calls:** cudaMemcpyAsync 11 (host to device except HostCUDA.cu:548), cudaMallocHost 7 (the MemoryPool.cpp pinned pool), cudaStreamCreate 4. Device choice: `cudaSetDevice(CmiMyNode() % numGpus)` (DataManager.h:427, 445).
- **Device memory and messaging:** no CUDA-aware messaging, cudaIpc, cudaMemcpyPeer, hip, Kokkos/RAJA, CK_DEVICE, CMK_CUDA or nocopydevice. The device path is node-local: the DataManager gathers TreePiece data into pinned buffers, and HAPI callbacks re-enter DataManager or PEList.
- **Raw pointers through entry methods:** `updateParticles(intptr_t)` (DataManager.cpp:1345) and `flushInteractionsToGpu(intptr_t, intptr_t)` (Compute.cpp:386, 955; TreePiece.cpp:4149, 4537, 4569).

#### B.1.10 Internals
- **Charm++ headers:** charm++.h, charm.h, converse.h (keytype.h:1), pup.h, pup_stl.h, cklists.h, ckio.h, liveViz.h, ckmulticast.h, CkLoopAPI.h, CkCache.h, CkTreeCache.h, TopoManager.h, BaseLB.h, CentralLB.h, HybridBaseLB.h, Refiner.h, ckBIconfig.h, ck128bitHash.h, hapi.h.
- **Build-tree paths:**
  - Makefile.in:107-109: `-I$CHARM/tmp`, `-I$CHARM/tmp/libs/ck-libs/cache` and `.../threadsafe_hashtable`.
  - cuda.mk.in: `-I$CHARM/tmp/hybridAPI`.
  - configure.ac:44 sources `$CHARM/tmp/conv-config.sh`.
- **CMK_ tests:**
  - CMK_LB_USER_DATA: `#error` if missing (TreePiece.cpp:39).
  - CMK_SMP: ParallelGravity.cpp:1030, DataManager.cpp:477, TreePiece.cpp:4280, Makefile.in:79.
  - CMK_LBDB_ON in 5 LB files; CMK_BALANCED_INJECTION_API (DataManager.h:23, 399); CMK_HAS_INT16 (keytype.h:4); CMK_VERSION_BLUEGENE (gravity.h); CMK_TRACE_ENABLED (configure.ac:355).
  - CMK_ERROR_CHECKING, CMK_WITH_STATS, CMK_CHARMDEBUG are only printed (ParallelGravity.cpp:2704-2714).
  - CMK_SSE, CMK_USE_SSE2 and CMK_USE_AVX are ChaNGa's own flags.
- **Runtime internals used:**
  - `CkpvAccess(_lb_obj_index)`; CpvDeclare, CpvAccess and CpvAccessOther (IntraNodeLBManager.cpp); `__nodelock`.
  - CkLocMgr::iterate with CkLocIterator subclasses (IntraNodeLBManager.cpp:32, PETreeMerger.cpp:13, PEList.cpp:16, TreePiece.cpp:6712).
  - None of `_entryTable`, `_chareTable`, CksvAccess or `_register*`.
- **Maps:** CkArrayOptions with DefaultArrayMap (RRMap under ROUND_ROBIN_WITH_OCT_DECOMP) and `bindTo` (ParallelGravity.cpp:1386-1408). No custom CkArrayMap and no CkArrayIndex subclass.
- **CkCache internals read directly:** DataManager.cpp:915-921 iterates the cache's internal maps (`getCache()`, CkTreeCacheView) for the GPU.

#### B.1.11 Build and version
- **Charm build:** the charm target `ChaNGa` (README; it supplies `--enable-lbuserdata`). CI (.github/workflows/changa.yml, changa-smp.yml) runs `./build ChaNGa netlrts-linux-x86_64 [smp] --with-production --enable-error-checking` on v7.0.1.
- **Version checks:** MINIMUM_CHARM_VERSION=60800 (configure.ac:38). `CHARM_VERSION > 61002` in 5 LB files, `> 60401` in DataManager.h, `LB_MANAGER_VERSION`.
- **Modules:** `-module CkCache CkIO CkMulticast RefineLB GreedyLB RotateLB liveViz CkLoop`, the 7 ChaNGa LB modules, and `-balancer MultistepLB_notopo`. **Patches to Charm++: none.**
- **CkTreeCache:** `--enable-smpcache` defaults to yes when `$CHARM/include/CkTreeCache.h` exists (configure.ac:170-191). The header is present in charm's `src/libs/ck-libs/cache`.
- **Reconverse:** Makefile.in:251 handles charm builds without charmrun ("e.g. reconverse"). No other reconverse or LCI mention.

#### B.1.12 Ordering assumptions
- **No code says it relies on message order between two objects. No sequence numbers.**
- **How the code copes with arrival order instead:**
  - Requests that arrive early are buffered: `smPartRequests` when `sSmooth==NULL` (CacheInterface.cpp:253-263), and `momentRequests` (TreePiece.cpp:3425-3432).
  - Shuffle completion is counted against `dm->particleCounts` (TreePiece.cpp:1224) or found by QD.
  - The keys broadcast is separated from the shuffle by a DataManager reduction (DataManager.cpp:147-149).
- **Priority:** the code depends on IFIFO priorities and expedited dispatch for performance, not for correctness.

#### Compatibility points
1. **CacheInterface.cpp:357-445 (:388, :399, :444): node fills kept as storage when `--enable-smpcache=no`.**
   - What the code does: it keeps each received CkCacheFillMsg for the iteration. It writes into the buffer (offsets to pointers, vtable restore), takes one CmiReference(UsrToEnv) per node, and calls CkFreeMsg once per node.
   - Requirement: the receive buffer must be a CmiAlloc block with a reference count that the app may keep and write.
   - What breaks it: ring-buffer slots or reused rendezvous buffers. The envelope layout also matters.
   - In SMP-cache mode the per-node reference goes away, but CkTreeCacheManager still keeps the message and frees it once per line.
2. **CacheInterface.cpp:37-46: particle fills kept until cache eviction.** Same exposure as item 1, without reference counting. These are mid-size messages, the range the new ring buffer targets.
3. **TreePiece.cpp:1210-1266: received ParticleShuffleMsg (varsize, can be large) held until all of a piece's particles arrive.**
4. **Received messages re-sent:** TreePiece.cpp:4132, 4219, 4522; smooth.cpp:440, 957, 1189 (to self); TreePiece.cpp:3331, 3355, 3362 (CkCacheRequestMsg forwarded to another PE). Each one needs the runtime to accept a re-send of a message from its receive memory.
5. **[nokeep] entries** (ParallelGravity.ci:242, 342, 416): marshalled broadcasts shared among the local elements.
6. **Priority and expedited dispatch:** CkPriorityPtr/CkSetQueueing on 15 messages, CkEntryOptions on 7, [expedited] fillRequest* entries. A new eager or rendezvous path must keep both.
7. **Build-tree includes** (`$CHARM/tmp`, `tmp/libs/ck-libs/cache`, `threadsafe_hashtable`, `tmp/hybridAPI`, `tmp/conv-config.sh`) and the internal LB headers (Refiner.h, BaseLB.h, CentralLB.h, HybridBaseLB.h).
8. **CMK_ macros and runtime internals:** CMK_LB_USER_DATA is required; CMK_BALANCED_INJECTION_API and ckBIconfig.h (GNI); `CkpvAccess(_lb_obj_index)`, `CpvAccessOther`, `__nodelock`.
9. **Migration:** AtSync and ResumeFromSync at load balancing, plus a particle re-shuffle at every domain decomposition. A persistent region on a TreePiece's particle or tree arrays must be rebuilt after each tree build, not only at load balancing.
10. **Same-process pointers through entry methods:** `updateParticles(intptr_t)`, `flushInteractionsToGpu(intptr_t, intptr_t)`, and the [local] callbacks with `void *source`. They depend on intra-process delivery of entry methods.
11. **CkCache contract (a library we control):** ChaNGa's EntryType `unpack`/`free` take ownership of the fill message, and DataManager reads the cache's internal maps. A CkCache rewrite on the persistent family must keep these or change them together with ChaNGa.

#### Candidates for the persistent family
1. **Node and particle cache fills (gravity), every step.**
   - Pattern: request then reply to the TreePieces the walk opens: a mostly stable spatial neighborhood plus the top of the tree.
   - Node fill size: up to 2^(d+1)-1 = 31 nodes at nCacheDepth d=4, each `ALIGN_DEFAULT(sizeof(BinaryTreeNode)+8)`. Estimated from the fields, not measured: about 0.4-0.5 KB each with hexadecapole moments, so about 15 KB or less.
   - Particle fill size: one bucket of up to nBucket=12 ExternalGravityParticle (40 B each), plus 16 B, about 0.5 KB.
   - Fit: a one-sided get from an owner-exposed region. The owner's arrays are rebuilt every step, so the region needs re-registration each step.
2. **SPH cache with writeback.** The fills are as in item 1. `flushSmoothParticles` sends bucket data back to the owner, which combines it there: a put-with-combine. One bucket of particles per message.
3. **Remote moments at every tree build.** Fixed-size records (MultipoleMoments, 2 OrientedBox<double>, ints) to the few SFC neighbors that share boundary nodes. The partners stay the same across substeps until the next domain decomposition.
4. **Boundary keys:** `recvBoundary(Key, dir)` to exactly the ±1 SFC neighbors (TreePiece.cpp:2696-2749). Tiny, fixed size.
5. **Domain decomposition shuffle (ParticleShuffleMsg).** Many-to-few, mostly to SFC neighbors. Sizes are bounded by the particles that leave a piece. The alltoall branch already makes it one nocopypost transfer per node pair. Moving it to per-node-pair regions with put plus a counter would need each region sized for the largest shuffle and set up again after each decomposition, so section 17.4 leaves it on messages.
6. **Output gathers** (Writer `receive`, `oneNode*` to rank 0). Infrequent; low priority.

#### Unmerged branches
Read-only git was used. Local origin/master is `39fc62a7` (2026-09-29). Neither local clone has the cloned master `8fd8207`, so merge bases are against the local origin/master. "Not in master" means the branch adds the line (absent from its merge base) and the cloned master does not have it.

##### Branch alltoall-aggregation (the research group's clone)
- **HEAD:** `4e9d0811` (2026-10-08). Merge base `39fc62a7`.
- **Branch commits (log -30, top 7):** 4e9d0811 "zero-copy only; require Charm++ 7.0.1"; 9abeed18 "probe for the zero-copy post API at configure time"; 0e93dc3c "report the bin count"; 44f0185c "hand intra-process particles over by pointer"; c1bab8e8 "zero-copy post API transfer (bShuffleZeroCopy)"; 843f21cf "Aggregate the domain decomposition particle exchange per node"; 39fc62a7 (base). The other 23 entries are master history.
- **35 hits not in master:**
  - ParallelGravity.ci:259-264: DataManager entries `startNodeShuffle(CkReductionMsg*)`, `acceptNodeShuffle(..., ShuffleBin bins[nBins], double loads[], unsigned parts[], ..., nocopypost char data[nBytes])` and `nodeShuffleSent(CkDataMsg*)`. TreePiece entries `fillNodeShuffle()` and `acceptNodeShuffleBin(intptr_t bufAddr, int iBin)` (:330).
  - DataManager.cpp:281-284: `dMProxy[n].acceptNodeShuffle(..., CkSendBuffer(b->data, cb))`, one message per node pair.
  - DataManager.cpp:294-295: the completion entry reads a `CkNcpyBuffer*` from `CkDataMsg::data`.
  - DataManager.cpp:343-344: `CkMatchBuffer(ncpyPost, 0, srcNode); CkPostBuffer(b->data, nBytes, srcNode)` into the app's own NodeShuffleBuf.
  - DataManager.cpp:371: passes a node-local pointer as `intptr_t`.
  - `lockShuffle` CmiCreateLock/CmiLock (:51 and others); CkMyNode and CkNumNodes bins.
  - configure.ac: `MINIMUM_CHARM_VERSION=70001` for the zero-copy post API.
  - New parameter `bNodeShuffle` (default 1) and NodeShuffle.h (`PUPbytes(ShuffleBin)`).
- **Exposure:** nocopypost with CkPostBuffer and CkMatchBuffer (the Post API, kept unchanged, section 13), and a CkSendBuffer completion callback that reads the `CkNcpyBuffer` from `CkDataMsg::data` (the Send API's sender side; section 13 does not yet say whether that callback keeps its form when `nocopy` becomes `send(..., hold)`). The branch needs no change under the redesign.

##### Branch treecache-nodecache (the research group's clone)
- **HEAD:** `b59fb571` (2026-09-26). Merge base `2b45e2f0`.
- **Branch commits (top 4):** b59fb571 "registerStatistics: register the reducers once per process"; 1d54138c "Walkers hand the node cache the slot they missed at"; 87517857 "Node cache: use charm's process-shared CkTreeCacheManager"; c0560a2b "Restore the vtable pointer of received cache nodes". The other 26 entries are master history.
- **Hits not in master: none.** 4 lines differ from master, but they are in the merge base and master removed them later (EwaldMsg and three Ewald/DataManager priority lines).
- **Status:** the branch's change (CkTreeCacheManager node cache, `requestDataAt`, no per-node CmiReference) is on master behind `CHANGA_SMPCACHE` / `--enable-smpcache`. On the branch it is unconditional.

### B.2 SpECTRE

| | |
|---|---|
| App | SpECTRE (numerical relativity, DG + finite-difference subcell) |
| Repo / branch | https://github.com/sxs-collaboration/spectre, `develop` (shallow clone) |
| Commit | `e734530bcd051e257eb9c9a579e877e954ba3aea`, 2026-11-13 11:47:41 -0500 (merge of PR #7242) |
| Files scanned | 7,484 files outside `.git` and `external/`: 5,614 C++ (`.hpp/.cpp/.tpp`), 11 `.ci`, 0 `.ci.in`, 878 CMake files, plus docs, CI workflows, containers, support scripts. `src/Parallel` (113 files, 14,453 lines) and `cmake/` (104 files) grepped in full; their core files read. |
| Excluded | `external/` (148 vendored files: brigand, libsharp, SPHEREPACK, slatec, EffectiveSource, GravitationalEffectiveSource, Licenses); `.git` |

#### Summary
- All Charm++ use sits in SpECTRE's own layer in `src/Parallel`. There are four templated chare kinds: `AlgorithmArray`, `AlgorithmGroup`, `AlgorithmNodegroup`, and `AlgorithmSingleton` (a one-element array). They share generic templated entry methods: `simple_action`, `threaded_action` (nodegroup only), `receive_data`, `reduction_action`, `perform_algorithm` and `start_phase`. Code outside `src/Parallel` calls the `Parallel::` wrappers. It almost never calls Charm++ directly.
- Data is sent as marshalled, PUP-serialized C++ objects (`std::tuple`, `DataVector`, `BoundaryData`). `receive_data` unpacks into new heap objects. It moves them into a per-object "inbox" keyed by a temporal id (`TimeStepId`, iteration id). The runtime frees the message when the entry method returns, and no message pointer is kept.
- `receive_data` is `[inline]`. For a local receiver, the generated proxy code calls the C++ method directly with the sender's forwarded objects, so no message is built. SpECTRE patches the generated `.def.h`/`.decl.h` files, per Charm++ version, to stop after 64 nested inline calls.
- The dominant traffic is a nearest-neighbour exchange every time step. DG elements send mortar and ghost data to each face neighbour. The elements are either array elements or are packed into a nodegroup ("DgElementCollection"). The nodegroup path uses lock-free SPSC inboxes and sends one runtime message per receiver per step.
- Collectives: custom-reducer reductions (PUP bytes in a `CkReductionMsg`); section reductions (sections are used only for reductions); Main to all-element broadcasts; phase changes driven by `CkStartQD`; load balancing triggered through a one-element `AtSync` indicator array; disk checkpoints via `CkStartCheckpoint`.
- Absent: zerocopy, Converse-level calls, `CMK_` tests in source, TRAM, delegation and GPU messaging. One custom message type (`BoundaryMessage`, with its own `pack`/`unpack`) exists, but only tests send it.

#### B.2.1 Interface files
- **11 `.ci` files**: 9 in `src`, 2 in `tests`. No `.ci.in`, and no CMake generation of `.ci` files.
  - `add_charm_module` (`cmake/SetupCharmModuleFunctions.cmake:19-21`) copies the `.ci`, runs `charmc`, then runs `tools/patch_charm_modules.sh` on the output.
  - 77 executables write `CkRegisterMainModule()` by hand, calling `Parallel::charmxx::register_main_module` (`src/Elliptic/Executables/Poisson/SolvePoisson.cpp:18`).
- **Chare kinds**: mainchare 4 (`Main` [migratable, templated] `Main.ci:14`, `ParallelInfo`, `TestMainCharm`, `Test_GlobalCache`); chare 0; group 2 (`AlgorithmGroup` [migratable] `AlgorithmGroup.ci:11`, `PeGroupReporter`); nodegroup 3 (`GlobalCache` [migratable] `GlobalCache.ci:16`, `AlgorithmNodegroup` [migratable], `NodeGroupReporter`); array [1D] 3 (`AtSyncIndicator`, `PrinterChare`, `TestArrayChare`); readonly 2 (`Printf.ci:17-18`).
  - Array with custom index 2: `AlgorithmArray` and `AlgorithmSingleton`, indexed by `[SpectreIndex_detail::ArrayIndex<SpectreArrayIndex>]` (`AlgorithmArray.ci:25`). This is a deliberate charmxi workaround, documented at `AlgorithmArray.ci:14-23`.
- **Messages**: 1, `template <size_t Dim> message BoundaryMessage;` (`BoundaryMessage.ci:9`). It has no varsize arrays and no `[packed]`/`[varsize]` attribute, but defines its own static `pack`/`unpack` (see B.2.2).
- **Entry attributes** (85 entry lines):
  - nonzero: inline 4 (all `receive_data(temporal_id&, ReceiveData_t&, bool)`, e.g. `AlgorithmArray.ci:78`); reductiontarget 6 (`reduction_action` x4, `Main.ci:19,25`); local 1 (`AlgorithmArray.ci:85` `invoke_iterable_action`, called only when `SPECTRE_CHARM_PROJECTIONS` is defined).
  - zero: expedited, immediate, nokeep, threaded, sync, exclusive, createhere, createhome, aggregate, whenidle, notrace, python, accel.
  - SpECTRE's "threaded_action" is an ordinary nodegroup entry method, not Charm++ `[threaded]`.
- **Parameter keywords**: nocopy 0, nocopypost 0, nocopydevice 0.
- **SDAG** (when, serial, overlap, forall, case): 0.
- **Templates**: 22 templated entry methods, 9 templated chares and 1 templated message. One message-typed entry exists: `template <ReceiveTag, MessageType> entry void receive_data(MessageType*)` (`AlgorithmArray.ci:75`).

#### B.2.2 Messages in C++
- **Classes deriving from `CMessage_`**: 1, `evolution::dg::BoundaryMessage<Dim>` (`src/Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.hpp:31`). Production code never sends it. Only `tests/Unit/Parallel/Test_AlgorithmMessages.cpp:249` and `Test_AlgorithmCore.cpp:483` send it, and its inbox tag `BoundaryMessageInbox` (`InboxTags.hpp:160`) has no users in `src`.
- **CkAllocMsg / CkFreeMsg / CkCopyMsg / CkReferenceMsg**: 0 / 0 / 0 / 0. Related: `CkAllocBuffer` 1 (`BoundaryMessage.cpp:73`, inside `pack()`); `delete in_msg` 1 (`BoundaryMessage.cpp:108`, frees the original sender-side message after `pack()` copies it into the new buffer).
- **Stored received messages**: `BoundaryMessageInbox::insert_into_inbox` takes ownership in a `std::unique_ptr<BoundaryMessage>` (`InboxTags.hpp:181`), keeps it until the time step consumes it, then frees it with `delete` (tests only). `CkArgMsg` in `Main::Main` (`Main.hpp:280`) is never deleted. Re-sent received messages: none.
- **Static pack()/unpack()/alloc()**: `BoundaryMessage::pack` (`.cpp:53`) and `unpack` (`.cpp:113`); no `alloc()`.
  - When `owning == false`, the message holds raw pointers into the sender's `DataVector`. `pack()` then copies that data into a new `CkAllocBuffer` and sets `owning = true`.
  - `unpack()` points the pointers back into the received buffer.
  - The design depends on the runtime calling `pack()` only when a message leaves the process. `Test_AlgorithmMessages.cpp:306-358` asserts that a same-node message has `owning == false` and the identical data address.
  - The message is registered by hand: `MessageType::base::__register(name, sizeof, pack, unpack)` (`src/Parallel/CharmRegistration.hpp:137-140`).
- **Envelope access** (UsrToEnv, EnvToUsr, `envelope`): 0 in app source. Of the 37 `envelope` hits, the source hits are physics terms and one is a doc sentence (`Concepts.md:260`). `envelope env; env.setMsgtype(ForArrayEltMsg); env.setTotalsize(0)` appears only as context lines in the generated-code patches (B.2.10).
- **CkSetRefNum / CkGetRefNum**: 0.
- **Priority and queueing**: 0 calls to CkSetQueueing, CkPriorityPtr, setPriority, setQueueing, CK_QUEUEING_ or CkSetMsgPriority. `CkEntryOptions` 2, used only for `setGroupDepID` so that groups are created after the GlobalCache nodegroup (`Main.hpp:533-534`; test `Test_GlobalCache.cpp:602`).

#### B.2.3 Zerocopy and RDMA
- **Calls**: 0 for CkNcpyBuffer, CkSendBuffer, CkDeviceBuffer, CkDevicePersistent, CkNcpyMode, CK_BUFFER_, CkPostBuffer, CkMatchBuffer, CkRdmaWrapper, CkRdma, CkDirect and CmiDirect.
- **Words** ncpy / zerocopy / zero-copy: 0 relevant. The 11 hits are `NoCopy<>` test structs and a `strncpy`.
- **Intent to avoid copies** shows up in three places instead: the BoundaryMessage "same node, no copy" design (`InboxTags.hpp:150-152`); `[inline]` receive_data; the shared-memory inbox insertion in DgElementCollection (B.2.7).

#### B.2.4 Converse-level calls
- **Messaging, scheduler, handlers**: 0 for CmiSyncSend, CmiAsyncSend, CmiSyncSendAndFree, CmiFreeSend, CmiSyncBroadcast, CmiSyncNode*, CmiPushPE, CmiPushNode, CmiRegisterHandler, CmiSetHandler, CmiAlloc, CmiFree, CmiReference, CsdEnqueue, CqsEnqueue, CmiDeliverMsgs, CsdScheduler and CsdSchedulePoll.
- **Persistent API** (CmiPersistent, CmiCreatePersistent, CmiUsePersistentHandle, CmiDestroyPersistent, CmiDestoryPersistent, CmiCreateCompressPersistent): 0.
- **Locks and barriers** (CmiLock, CmiCreateLock, CmiNodeBarrier, CmiNodeAllBarrier): 0. `Parallel::NodeLock` wraps SpECTRE's own `std::atomic` Spinlock (`src/Parallel/NodeLock.hpp`). `.claude/rules/Cxx.md:10` bans `lrtslock.h`.
- **Fences** (CmiMemory*): 0. The app uses `std::atomic` memory orders (`AtomicInboxBoundaryData.cpp`, `StaticSpscQueue.hpp`, `MultiReaderSpinlock.hpp`).
- **Topology**: Charm++-level only, in one wrapper file, `src/Utilities/System/ParallelInfo.cpp:11-37`. It calls CkNumPes, CkMyPe, CkNumNodes, CkMyNode, CkNodeSize, CkMyRank, CkNodeFirst, CkNodeOf, CkRankOf and CkWallTimer. Everything else uses `sys::`/`Parallel::my_node()`.
- `converse.h` is included in 4 places (`DistributedObject.hpp:7`, `IO/Observer/Tags.hpp:7`, 2 test helpers), but no Cmi calls follow. The only Cmi text is a doc comment (`DistributedObject.hpp:137`).
- **Converse in the generated-code patches**: `CpvAccess(curPeEvent)` and `CpvInitialized(curPeEvent)`.

#### B.2.5 Aggregation, multicast, delegation
- **TRAM** (MeshStreamer and variants): 0. **Delegation** (ckDelegate, CkDelegateMgr): 0. **Comlib**: 0.
- **Sections**:
  - `CProxySection_` appears 3 times, only as type aliases (`Algorithm{Array,Group,Nodegroup}Declarations.hpp:56`). `CkSectionInfo` is stored as a cookie in `Parallel::Section` (`Section.hpp:94`). CkGetSectionInfo, ckSectionDelegate and CkSectionID: 0 calls (CkGetSectionInfo appears in comments only).
  - Section proxies are created with `cproxy_section::ckNew(arrayID, indices)` in 3 places: `ParallelAlgorithms/Amr/Actions/ElementsRegistration.hpp:219,227` (multigrid grid-index sections) and `Evolution/DiscontinuousGalerkin/Initialization/SetupEqualRateRegions.hpp:165` (local-time-stepping regions).
  - Sections are used only for reductions, through `SectionProxy::contribute(..., cookie, cb)` (`Reduction.hpp:315`). In Charm++ that call goes through `CkMulticastMgr`.
  - `SetupEqualRateRegions.hpp:167-177` explains why: "Charm sections are buggy ... we avoid doing anything other than reductions". The code uses an array broadcast instead of a section broadcast.
  - `Reduction.hpp:304-314`: the section cookie is never refreshed, so section reductions may break after migration.

#### B.2.6 Collectives
- **`contribute(`**: 11 hits, all in `src/Parallel` plus `ParallelInfo.cpp`: `Reduction.hpp:286,291,315` (generic); `ContributeToPhaseChangeReduction.hpp:51,79,89`; `DistributedObject.hpp:1233` (`logical_and_bool` termination check); `GlobalCache.hpp:479` (an empty contribute used as a barrier).
- **Wrapper call sites**: `Parallel::contribute_to_reduction` is called 21 times in 15 files outside `src/Parallel` (per-iteration solver reductions, observers, control system, memory monitor).
- **Custom reducers**: `CkReduction::addReducer` on every node at init (`CharmMain.tpp:102`, registered through `_registerInitCall`), kept in a `CkReduction::reducerType` map (`CharmMain.tpp:35`). `ReductionData::combine(int, CkReductionMsg**)` reads `msg->getData()` with `PUP::fromMem` and builds the result with `CkReductionMsg::buildNew` (`Reduction.hpp:35,166-178`). The reductiontarget wrapper (`redn_wrapper_reduction_action`) unmarshals the PUP bytes of one `ReductionData` argument. tupleElement / setElement: 0.
- **CkCallback forms**:
  - entry method plus proxy: `CkIndex_Main::execute_next_phase()` and others (`Main.hpp:588,709,835,840,854,878,947,1030`);
  - reductiontarget: `CkReductionTarget(Main, did_all_elements_terminate)` (`DistributedObject.hpp:1231`) and `redn_wrapper_*` (`Reduction.hpp:275`, `ContributeToPhaseChangeReduction.hpp:44,70`);
  - ckExit 0 (`CkExit` is called directly in `Utilities/System/{Exit,Abort}.cpp`); resumeThread 0; function pointer 0.
  - `Parallel::Callback` is SpECTRE's own `PUP::able` class, not CkCallback (`Callback.hpp:51`).
- **Broadcasts**: about 30 of 87 `simple_action`/`threaded_action`/`receive_data` sites target a whole-component proxy. Some of those targets are singletons, so the count is an upper bound for broadcasts. Recurring broadcasts:
  - linear/nonlinear-solver residual monitors to all elements every iteration (9 sites, e.g. `LinearSolver/Gmres/ResidualMonitorActions.hpp:91,149,243`);
  - `Parallel::mutate` to the GlobalCache nodegroup (11 sites; control-system function-of-time updates, e.g. `ControlSystem/UpdateFunctionOfTime.hpp:238`);
  - `start_phase`/`set_terminate` from Main at each phase.

#### B.2.7 Local access and node-level parallelism
- **`ckLocal(` / `ckLocalBranch(`**: only in `src/Parallel/Local.hpp:36,38,52` (plus mocks). `.ckLocal()` is banned elsewhere by a rule. The wrappers are used 13 times (`Parallel::local`) and 60 times (`Parallel::local_branch`) outside `src/Parallel`.
- **`local_synchronous_action`**: a direct C++ call on the local group/nodegroup branch, with no message (`Invoke.hpp:104`). 14 sites.
- **CkLocalBranch, CkLocalNodeBranch, CkLoop, std::thread in src**: 0. `std::thread` appears only in 3 test files.
- **`#pragma omp`**: 6, all in `IO/Exporter/PointwiseInterpolator.cpp` (a standalone exporter).
- **Kokkos**: 251 hits, optional (`SPECTRE_KOKKOS`). It is used for on-node math (`DataStructures`, `Hydro`) and `Kokkos::initialize` in `Main.hpp:282`. No Kokkos data goes through messages.
- **Node-level DG**: `src/Parallel/ArrayCollection/*` keeps many elements in one nodegroup. In `SendDataToElement.hpp:51-101`, a same-node sender inserts into the receiver's inbox in shared memory. The inbox is lock-free: an SPSC queue (capacity 20) per direction plus an atomic `missing_messages` counter (`AtomicInboxBoundaryData.hpp`). A runtime message (`threaded_action<ReceiveDataForElement>` to the node's branch) is sent only when that counter reaches zero. Cross-node data goes as a `threaded_action` with the data marshalled.

#### B.2.8 Migration, load balancing, checkpoint, QD, futures
- **AtSync and migration**:
  - `AtSync(` 1 call (`Main.hpp:268`), from the one-element `AtSyncIndicator` array (`usesAtSync = true`, not migratable; `Main.hpp:262-263`). `ResumeFromSync` returns control to Main (`Main.hpp:272`). migrateMe / CkMigrate: 0.
  - Every SpECTRE array element sets `usesAtSync = false; setMigratable(true)` (`DistributedObject.hpp:528-529,559-560`). The design relies on Charm++ moving those elements when the single indicator calls AtSync. Main runs `CkStartQD` before load balancing (`Main.hpp:835`) and before checkpoints (`Main.hpp:840`).
  - Element pup deregisters and re-registers with observers when the phase is LoadBalancing (`DistributedObject.hpp:715-723`). The nodegroup DgElementCollection refuses load balancing (`:698-703`).
- **PUP**: 1,409 `pup(` hits in `src` (971 files), 1,076 `pup(PUP::er` definitions. `PUP::able` appears 679 times in 554 files; `WRAPPED_PUPable_decl` 359 times.
- **Checkpoint**: `CkStartCheckpoint` 1 call (`Main.hpp:877`). Restart uses `+restart <dir>` (`docs/Tutorials/CheckpointRestart.md:36`). CkStartMemCheckpoint and CkRestart: 0.
- **Quiescence detection**: `CkStartQD` 6 calls (`Main.hpp:709,835,840,854,947,1030`); every phase transition is QD-driven. CkWaitQD: 0.
- **Futures and threads** (CkFuture, CthSuspend, CthAwaken, CkCallbackResumeThread): 0.
- **Dynamic insertion and deletion**: `insert` in AMR (`Amr/Actions/CreateChild.hpp:58,66`, `CreateParent.hpp:54`), `ckDestroy` in 4 places, `doneInserting`. Charm++ versions below 8.0.0 disable h-refinement (`cmake/SetupCharm.cmake:41`).
- **Load balancers named**: RecBipartLB (recommended), ScotchLB (optional `-DUSE_SCOTCH_LB`), MetisLB, RefineLB, GreedyLB, NullLB (`docs/DevGuide/LoadBalancingNotes.md`). The notes say comm-unaware balancers drop efficiency to 20-30%.

#### B.2.9 GPU
- 0 hits for: hapi/HAPI (the 80 hits are the string "BaumgarteShapiro"), hapiAddCallback, hapiMalloc, cudaMemcpy, cudaIpc, hipMemcpy, CK_DEVICE, CMK_CUDA, nocopydevice, RAJA and CUDA-aware messaging.
- Kokkos is used for compute only (B.2.7).

#### B.2.10 Internals
- **Charm++ headers included**: `pup.h` (757), `pup_stl.h` (109), `charm++.h` (31), `converse.h` (4), `ckarrayindex.h` (2: `Parallel/ArrayComponentId.hpp:7`, 1 test), `charm.h` (`Informer/Informer.cpp:7`), `ckcallback.h` (`ParallelInfo.cpp:7`), plus the generated `*.decl.h`/`*.def.h` (the def.h files are included with `CK_TEMPLATES_ONLY`, e.g. `AlgorithmArray.hpp`). `envelope.h` and other internal headers: 0.
- **`CMK_` macros tested in source**: 0. Patch context lines contain `CMK_LBDB_ON` and `CMK_CHARMDEBUG`. The macOS build line passes `-DCMK_NO_ISO_MALLOC=1` (`.github/workflows/Tests.yaml:1058`; `docs/Installation/Installation.md:530`).
- **Runtime internals used**: generated registration functions called by hand (`CkIndex_X::__register` at `CharmRegistration.hpp:174,220`; `ckindex::idx_*` and `redn_wrapper_*`; `CMessage_X::__register` at `:137`); `_registerInitCall` (`CharmMain.tpp:127-140`); a hand-written `CkRegisterMainModule`. `_entryTable`, `_chareTable`, CkpvAccess, CpvAccess, CksvAccess: 0 in source (`CpvAccess` appears only in the patches).
- **Custom index and options**: CkArrayIndex subclass `Parallel::ArrayIndex<Index>` (`src/Parallel/ArrayIndex.hpp`) placement-news a 12-byte POD (ElementId) into `CkArrayIndex::index`, sets `nInts`, and `static_assert`s `CK_ARRAYINDEX_MAXLEN == 3`. CkArrayMap and CkLocMgr: 0. `CkArrayOptions` 1, `opts.bindTo(...)` for bound arrays (`Main.hpp:567-568`).
- **App-maintained patches of charmxi output**: `tools/CharmModulePatches/src/...`, 12 unified diffs keyed `_v7.0.0`, `_v7.0.1` and `_v8.0.0`, applied by `tools/patch_charm_modules.sh`.
  - `AlgorithmArray`/`AlgorithmSingleton` `.def.h`: the local branch of the `[inline]` receive_data becomes `if (obj != nullptr and not Parallel::detail::max_inline_entry_methods_reached())`. The depth limit is 64, kept in a thread_local counter (`MaxInlineMethodsReached.hpp`).
  - Template parameters are reordered; `_TRACE_*` arguments are wrapped in parentheses; `CpvAccess(curPeEvent)` is guarded with `CpvInitialized`; `CMessage_MessageType::__idx` becomes `MessageType::base::__idx`; `BoundaryMessage.decl.h` gets a fix to the constructor name.

#### B.2.11 Build and version
- **Build lines**: `./build charm++ <arch> --with-production --build-shared --disable-tls`, Charm++ v8.0.0 (`docs/Installation/Installation.md:113-118`). CI and containers use:
  - `multicore-linux-<arch> gcc -g -O2 --build-shared --with-production --disable-tls` (`containers/Dockerfile.buildenv:486-492`);
  - `mpi-linux-x86_64-smp clang ... --build-shared --with-production --disable-tls` (`Tests.yaml:636`);
  - `multicore-darwin-arm8 ... -DCMK_NO_ISO_MALLOC=1` (`Tests.yaml:1056`).
- **Cluster setups** use `./build LIBS mpi-linux-x86_64-smp --with-production` with Charm++ 7.0.0 (`support/Environments/setup/{anvil,expanse,frontera}_gcc.sh`, `mbot_install.sh:2399`; Ocean2 uses Charm++ 8.0.0). Runs use `++ppn` with `+setcpuaffinity` or `+pemap/+commap`; `+no_isomalloc_sync` appears on CaltechHpc. Profiling builds add `--enable-tracing`. Spack: `charmpp@7.0.0: backend=multicore`. CUDA options: none.
- **Version checks**: `SPECTRE_REQUIRED_CHARM_VERSION 7.0.0` (`cmake/SetupCharm.cmake:4`); below 8.0.0, h-refinement is disabled (`:41`); the docs say "7.0.0, or later (8 preferred)"; the version is parsed from `charm-version.h` (`cmake/FindCharm.cmake`).
- **Patches to Charm++ itself** (`support/Charm/v6.7.patch`, `v6.8.patch`, `v7.0.0.patch`, `v7.0.1.patch`; none for 8.x): 7.0.x removes `-ftls-model=initial-exec` (checked by `SetupCharm.cmake:97-110` for Python bindings), fixes `ckhashtable.h` for C++20, and improves a PUP::able error in `pup_util.C`; 6.x changes pup.h/pup_stl.h. The charmxi-output patches are listed in B.2.10.
- **Modules**: `-modules EveryLB`, plus `ScotchLB` when `USE_SCOTCH_LB` (`SetupCharm.cmake:36-40`, `FindCharm.cmake:96-98`). Options `-nomain -nomain-module`, optional `-tracemode projections|summary`. The moduleinit file is generated from `charmc -save`.
- **Reconverse / LCI**: no mention. UCX/OFI appear only in a FindCharm comment (`:270`).

#### B.2.12 Ordering assumptions
- **Stated policy** (no ordering assumed): `docs/DevGuide/ParallelExecutable/Concepts.md:124-133` (actions run nondeterministically, and B must check that A completed); `docs/DevGuide/Amr.md:89-95` (AMR actions handle out-of-order execution).
- **Coping mechanisms**:
  - all inboxes are keyed by temporal id (`InboxInserters.hpp`, `AtomicInboxBoundaryData` `messages` map);
  - `LinkedMessageId {id, previous}` lets sequences be rebuilt "even when elements are received out-of-order" (`DataStructures/LinkedMessageId.hpp:16-19`). It is used by the control system, the apparent-horizon finder, Cce and the time tags (about 40 files).
  - `PiecewisePolynomial` keeps an `update_backlog_` for function-of-time updates that "come out of order" (`Domain/FunctionsOfTime/PiecewisePolynomial.cpp:155-163`);
  - the interpolation target tolerates late duplicate requests (`InterpolationTargetDetail.hpp:415-425`).
- **Tolerated race**: `ApparentHorizonFinder/CurrentTime.cpp:52`, "technically prone to async issues where times arrive out of order", for observation-only horizon finds.
- **Possible implicit order** (no guard in code): Main sends two broadcasts to the GlobalCache nodegroup from its constructor, `set_resource_info` (`Main.hpp:491`) and then `set_parallel_components` (`:592`). Nothing checks that the first has arrived before the cache's resource info is read through `Tags::ResourceInfoReference`; no users of that tag were found outside `src/Parallel`. Group creation uses `setGroupDepID`, not message order.
- **Phase transitions** use `CkStartQD`, not message order.

#### Compatibility points

| # | Site | What it does | Change that could affect it |
|---|---|---|---|
| C1 | `tools/CharmModulePatches/**` (12 files), `tools/patch_charm_modules.sh`, `cmake/SetupCharmModuleFunctions.cmake:21` | Line-based diffs of charmxi output. Context lines include `envelope env; env.setMsgtype(...)`, `_TRACE_*`, `CpvAccess`, `CMK_LBDB_ON`, `CkRegisterEp`. | Any change to charmxi output for `[inline]` marshalled entries or message-typed entries makes the patches fail (build error). recharm reports **9.0.0** (recharm's `charm-version.h`, a recharm build), and no `_v9.0.0` patch exists, so a recharm build silently applies none of them: the inline-depth guard is lost, and `receive_data(MessageType*)` fails to compile (`CMessage_MessageType::__idx` is still emitted by `xi-Entry.C:2503`; only tests instantiate it). In recharm `_TRACE_CREATION_DETAILED` is an inline function (`ck-perf/trace.h:409`), so the parentheses fix is not needed there. |
| C2 | `AlgorithmArray.ci:78` and 3 siblings; `Invoke.hpp:24-46` | `[inline]` receive_data: a direct call with forwarded C++ objects when `ckLocal()` is non-null; otherwise a marshalled PUP message. | Transport changes do not reach the inline path. A change to `[inline]` code generation or `ckLocal` semantics interacts with C1. |
| C3 | `BoundaryMessage.cpp:53-145`, `CharmRegistration.hpp:137`, `InboxTags.hpp:170-187`, `Test_AlgorithmMessages.cpp` | Custom `pack`/`unpack`. A same-node message carries raw pointers to the sender's data (`owning == false`). The receiver keeps the message in a `unique_ptr` until the step is consumed, then `delete`s it. `CkAllocBuffer` is called in `pack`. | It requires that `pack()` runs before any copy of message bytes and that same-process delivery passes the pointer unchanged (the test asserts this). An intra-process ring buffer or copy would break the test and the no-copy design. Rendezvous/ring receive buffers must survive long retention and `delete` (CkFreeMsg). Only tests use this path today. |
| C4 | `Reduction.hpp:35,101-104,166-178`; `CharmMain.tpp:102` | Custom reducer reads `CkReductionMsg::getData()` and builds the result with `buildNew`. The reductiontarget wrapper unmarshals one PUP argument from the reduction payload. | The marshalling format of a single-argument reductiontarget must stay "plain PUP bytes". A change to the reduction message layout would affect it. |
| C5 | `Reduction.hpp:301-316`; `SetupEqualRateRegions.hpp:165-178`; `Amr/.../ElementsRegistration.hpp:219,227` | Section reductions through `CkMulticastMgr`, with a cookie that is never refreshed. | Migration and load balancing, if section internals change. |
| C6 | `Main.hpp:228-273,835-878`; `DistributedObject.hpp:528,559,656-740` | Load balancing through a one-element AtSync indicator while all elements have `usesAtSync=false`. A QD precedes LB and checkpoints. Element pup re-registers with observers. | Persistent regions torn down and rebuilt at LB. Because QD runs first, no messages are in flight at LB or checkpoint time. |
| C7 | `ArrayIndex.hpp` (custom `CkArrayIndex`, `CK_ARRAYINDEX_MAXLEN == 3` assert); `ArrayComponentId.hpp` (`CkArrayIndex::hash()`) | Index layout dependence. | Not a comm-layer item, but any change to the CkArrayIndex layout (e.g. the objid work) affects it. |
| C8 | `Main.hpp:533-534` | `CkEntryOptions::setGroupDepID` orders group creation after the GlobalCache nodegroup. | Must keep its semantics. |
| C9 | `Main.hpp:491,592` | Possible implicit FIFO between two startup broadcasts to the same nodegroup (B.2.12). | The redesign must not reorder them, or SpECTRE needs a guard. Low impact: no readers found outside `src/Parallel`. |
| C10 | `support/Charm/v7.0.*.patch` | Patches Charm++'s configure, `ckhashtable.h` and `pup_util.C`. | Only relevant if those files change; none touch messaging. |

Not present, recorded as absent: zerocopy and RDMA APIs, `nocopy*`, `CkDevice*`, Converse sends/handlers/CmiAlloc/CmiFree, the reconverse persistent API, CkDirect, `[nokeep]`, envelope access in source, CkCopyMsg/CkReferenceMsg, message re-send, `CMK_` tests in source, and reliance on message order between two objects.

#### Candidates for the persistent family

Sizes are given as formulas because the code computes them from meshes. Numbers marked "e.g." are illustrations, not values from the code.

1. **DG boundary/mortar plus ghost exchange (evolution, every step or substep).**
   - Code: `Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp:954` (array) or `:945` (nodegroup collection); subcell ghosts at `DgSubcell/Actions/ReconstructionCommunication.hpp:302,629`.
   - Partners: 2·Dim face neighbours (6 in 3D), up to `maximum_number_of_neighbors(3) = 24` with h-nonconforming faces (`Domain/Structure/MaxNumberOfNeighbors.hpp:15-26`). They are fixed between AMR and LB phases.
   - Payload: `BoundaryData<Dim>` (`BoundaryData.hpp:43-51`) = meshes + `TimeStepId` + mortar `DataVector` (mortar points × boundary-correction components × 8 B) + optional ghost `DataVector` (ghost-zone width × face points × variables, plus 2×variables of RDMP data).
   - Sizes are bounded and change only on p-refinement or a DG/FD switch. E.g. N=10 per dimension and about 50 components gives about 40 KB per face.
   - The nodegroup path already does put + atomic decrement + notify on zero inside a node (`SendDataToElement.hpp`, `AtomicInboxBoundaryData`). This is the put-decrement pattern the persistent API targets.
2. **Elliptic DG operator mortar exchange, every linear-solver iteration.** `Elliptic/DiscontinuousGalerkin/Actions/ApplyOperator.hpp:278`. Partners: face neighbours. Size: mortar points × fields × 8 B. Partners and sizes are fixed for a given grid.
3. **Schwarz subdomain overlap exchange, every Schwarz iteration.** `LinearSolver/Schwarz/Actions/CommunicateOverlapFields.hpp:150` and `Schwarz/ElementActions.hpp:506`. Partners: face neighbours. Size: overlap extent × face points × fields.
4. **Multigrid restriction and prolongation, every cycle.** `LinearSolver/Multigrid/Actions/RestrictFields.hpp:154` (children to parent) and `Multigrid/ElementActions.hpp:430` (parent to children). Up to 2^Dim children per parent, with volume field sizes.
5. **Limiter neighbour data, every step.** `ParallelAlgorithms/Actions/LimiterActions.hpp:186`. Partners: face neighbours. Size: limiter `PackagedData`, which is small (means and extents).
6. **Worldtube coupling, every step.** `CurvedScalarWave/Worldtube/ElementActions/SendToWorldtube.hpp:164` sends spherical-harmonic coefficients from elements to the worldtube singleton (many-to-one). `SingletonActions/SendToElements.hpp:82` sends 1 or 4 coefficients × 2 fields back (one-to-few). The partner set is the fixed set of elements next to the worldtube.
7. **Cce worldtube boundary data, every step or interpolation time.** `Cce/Actions/BoundaryComputeAndSendToEvolution.hpp:123,224,231,291,391`. The singleton receives a fixed size determined by l_max.
8. **Scalar reductions and broadcasts every solver iteration.**
   - Elements send inner products to a ResidualMonitor singleton through `contribute_to_reduction` (`Gmres/ElementActions.hpp:116,375,456`; `ConjugateGradient/ElementActions.hpp:84,159,217`; `NewtonRaphson/ElementActions.hpp:208,470`).
   - The monitor broadcasts a few doubles back to all elements through `receive_data` (`ResidualMonitorActions.hpp` sites).
   - Partners are fixed and sizes are a few bytes. This is a counter/notify candidate rather than a bulk one.
9. **Many-to-one gathers at observation or horizon-find intervals.**
   - observers: element → group → node writer → writer on node 0 (`IO/Observer/VolumeActions.hpp:149,675`; `ReductionActions.hpp:219,475`);
   - apparent-horizon finder: element volume data → singleton (`ApparentHorizonFinder/Events/FindApparentHorizon.hpp:195`);
   - interpolation: elements → interpolation target (`Interpolation/Events/InterpolateWithoutInterpComponent.hpp:441`).
   - Partners are fixed between AMR/LB phases. Sizes vary with the observed fields.

### B.3 Quinoa

- **App:** Quinoa (executables `inciter`, `meshconv`, `unittest`). **Repo:** https://github.com/quinoacomputing/quinoa, branch `develop`.
- **Commit:** `3b14062ba321ae3ee1689337ac6f871f60579c6f`, 2026-11-12 16:43:03 -0700 ("testing the overset restart"). Clone: a shallow clone.
- **TPL repo:** quinoa-tpl `master` `5a34b5a47f38c42823faae53e02ef584a361e00d`, 2026-11-12 ("update charm to fix overset/collide restart capability"). Clone: a shallow clone, no submodules.
- **Scanned:** 601 source files (.ci, .cpp, .hpp, .h, .in): 568 under `src/` and 33 under `tests/unit/`. Also scanned: the 22 CMakeLists.txt in src/ and tests/unit, `cmake/*.cmake`, `tests/regression/**/CMakeLists.txt`, `doc/pages/*.dox`, `tools/docker`, and quinoa-tpl `CMakeLists.txt`.
- **Excluded:** no third-party code is vendored in the Quinoa repo; TPLs are git submodules of quinoa-tpl and were not cloned. Regression data (meshes, baselines), `doc/images` and `.git` were not scanned.
- **Method:** grep only. Nothing was built or run.

#### Summary
- **Charm++ features used.** All communication uses marshalled entry methods (parameters are STL containers serialized with PUP), SDAG `when` joins, and reductions. None of the following appear: CMessage_ classes, zerocopy, Converse calls, TRAM, sections, delegation, CkLoop, GPU messaging, CMK_ macros in C++.
- **Array layout.** There are 14 1D chare arrays. Per mesh, the solver array (ALECG, DG, FV or OversetFE) and the ALE, ConjugateGradients, BiCG, Ghosts, Refiner, Sorter and TransferDetails arrays are bound to Discretization (`bindTo`). Elements call each other through `ckLocal()`. Arrays are filled by dynamic `insert`/`doneInserting`.
- **Main time-step traffic** is halo exchange with the neighboring chares:
  - Node-based schemes (ALECG, OversetFE) send shared-node partial sums to node-sharing neighbors (`NodeCommMap`), twice per Runge-Kutta stage, 3 stages per step.
  - Cell-based schemes (DG, FV) send ghost-element data to face neighbors (`Ghosts::m_sendGhost`), 2-5 exchanges per stage.
  - Each payload is a `std::vector<size_t>` of node or element ids plus a `std::vector<std::vector<real>>`. The id list is re-sent on every exchange.
- **Collectives.** Each step does a min-dt allreduce whose result is broadcast to the whole array. Every diagnostics interval there is a custom PUP-serialized reduction to a singleton chare (Transporter), followed by a broadcast. The CG and BiCG linear solvers use per-iteration allreduces.
- **Overset meshes** (OversetFE) transfer the solution every step through the Charm++ `collidecharm` library. Collisions are funnelled to PE 0 by `CollideSerialClient`.
- **Field output** goes many-to-one inside each node, to the MeshWriter group branch on `CkNodeFirst(CkMyNode())`.
- **Runtime services:**
  - AtSync load balancing every `lbfreq` steps, with an optional non-blocking mode;
  - disk checkpoint/restart (`CkStartCheckpoint`);
  - AMR, which rebuilds the communication maps.
- **Build:** Charm++ must use the **MPI machine layer**, because Zoltan2/Trilinos, Omega_h and `MPI_Allreduce` in exception handling call MPI directly from inside entry methods. Charm++ is pinned to a charm-7.0.1-based branch plus a patch to the collide library.

#### B.3.1 Interface files
- **.ci files:** 37, all under `src/`. 13 are unit-test only (`src/UnitTest`) and 2 are unit-test mainmodules (`src/Main`).
- **Chare kinds:**
  - mainchare: 5 (Main ×4, `m2mtransfer.ci:12` LibMain).
  - chare: 13. Transporter (`transporter.ci:21`), execute ×4, and the rest are unit tests.
  - group: 6. MeshWriter, M2MTransfer, LBSwitch, ChareStateCollector, and the two `CkArrayMap`s (LinearMap, UnsMeshMap).
  - nodegroup: 4. Partitioner (`partitioner.ci:28`), QuietCerr, and `MPIRunner<Proxy>` ×2.
  - array [1D]: 14. Discretization, ALECG, DG, FV, OversetFE, ALE, Ghosts, Refiner, Sorter, ConjugateGradients, BiCG, TransferDetails, plus CGReceiver and TestArray in unit tests.
  - No 2D/3D arrays and no custom index. 9 declarations are `[migratable]`.
- **Messages:** no `message` declarations; `[packed]`/`[varsize]`: none.
- **Message-pointer entries:** CkReductionMsg* in 7 entries; CkDataMsg* in 5 (`ale.ci:39-41`, `cgreceiver.ci:25-26`); CkArgMsg* in 5.
- **Marshalled array parameters (3):**
  - `transporter.ci:68` `boxvol(tk::real meshdata[n], int n)`;
  - `transferdetails.ci:35` `Collision colls[nColls]`;
  - `transferdetails.ci:39` `PotentialCollision colls[nColls]`.
- **Entry attributes:** reductiontarget 50, exclusive 5 (`partitioner.ci` partition/addMesh/recvMesh, `mpirunner*.ci` rungroup), expedited 2 (`tutsuite.ci:57`, `tutsuiteinciter.ci:59`; unit-test output). inline, immediate, nokeep, threaded, sync, local, createhere, createhome, aggregate, whenidle, notrace, python, accel: **0**.
- **Other declarations:** initnode 7 (six `registerReducers`, one `quiet`); initproc 1 (`lbswitch.ci:19`); readonly 27.
- **Parameter keywords:** nocopy, nocopypost, nocopydevice: **none**.
- **SDAG:**
  - 40 SDAG entries (`wait4*`), 40 `when`, 40 `serial`, 1 `forall` (`transporter.ci:94`, which matches on reference number `[meshid]`); overlap and case: 0.
  - 94 `*_complete` trigger entries; each exchange is joined by `when own*_complete(), com*_complete()`.
- **Templates:** templated chares `TUTTest<Proxy>` (chare) and `MPIRunner<Proxy>` (nodegroup), unit-test harness only. Templated entry methods: none.

#### B.3.2 Messages in C++
- **CMessage_ subclasses, CkAllocMsg, CkReferenceMsg:** none.
- **CkCopyMsg 2, CkFreeMsg 2.** `LoadBalance/LinearMap.cpp:85,88` and `UnsMeshMap.cpp:190,193`. `populateInitial` copies the array-creation message for each local element, then frees it. Both maps are used only in unit tests (`tests/unit/LoadBalance`).
- **`delete msg`: 7 sites.** CkArgMsg in the 4 Main constructors and LibMain (`M2MTransfer.cpp:53`). CkReductionMsg in `Transporter.cpp:1334` (pdfstat), `:1547` (diagnostics) and `Main/Init.hpp:151` (dumpstate). Each is deleted after `PUP::fromMem(msg->getData())`.
- **Received messages never freed** (none are stored in containers; none are re-sent):
  - `Transporter.cpp:1442` `collectDtAndForces(CkReductionMsg*)`: a tuple reduction, once per OversetFE step;
  - `ALE.cpp:604,638,716` `applied/helmholtz/solved(CkDataMsg*)`: once per ALE linear solve;
  - `tests/unit/LinearSolver/TestConjugateGradients.cpp:64,72`.
- **Messages built by the app:**
  - `CkDataMsg::buildNew(sizeof(real), &x)` ×5: `ConjugateGradients.cpp:241,575`, `BiCG.cpp:267,615,793`, sent through a stored CkCallback;
  - `CkReductionMsg::buildNew` ×4 in custom reducers;
  - `CkReductionMsg::buildFromTuple` ×1 (`OversetFE.cpp:1119`, then `contribute(advMsg)`).
- **`reinterpret_cast<CkArgMsg*>(CkMigrateMessage*)` ×4** at `Main/Inciter.cpp:248-254`. The Main restart constructor reads argc/argv from the migrate message.
- **Static pack/unpack/alloc, envelope access (UsrToEnv/EnvToUsr/envelope), CkSetRefNum/CkGetRefNum, priorities and queueing (CkEntryOptions, setPriority, CK_QUEUEING_):** none.

#### B.3.3 Zerocopy and RDMA
- **Calls:** CkNcpyBuffer, CkSendBuffer, CkDeviceBuffer, CkDevicePersistent, CkNcpyMode, CK_BUFFER_, CkPostBuffer, CkMatchBuffer, CkRdmaWrapper, CkRdma, CkDirect, CmiDirect: **none**.
- **Words:** ncpy, zerocopy, zero-copy: **none**, including comments. "persistent", "RDMA" and "one-sided": none.

#### B.3.4 Converse-level calls
- **Messaging:** sends (CmiSyncSend etc., CmiPushPE/Node), handlers and message memory (CmiRegisterHandler, CmiSetHandler, CmiAlloc, CmiFree, CmiReference): **none**.
- **Scheduler:** CsdEnqueue, CqsEnqueue, CmiDeliverMsgs, CsdScheduler, CsdSchedulePoll: **none**.
- **Persistent API:** CmiPersistent, CmiCreatePersistent, CmiUsePersistentHandle, CmiDestroyPersistent, CmiDestoryPersistent, CmiCreateCompressPersistent: **none**.
- **Locks, barriers, fences:** CmiLock, CmiCreateLock, CmiNodeBarrier, CmiNodeAllBarrier, CmiMemory*: **none**.
- **Topology:** no Cmi-level calls. The Ck-level node calls CkMyNode, CkNumNodes and CkNodeFirst have 15 hits in 5 files. Examples:
  - `Discretization.cpp:990` and `Refiner.cpp:1058` send field output to `m_meshwriter[CkNodeFirst(CkMyNode())]`;
  - `Partitioner.cpp:318,655` map chares to nodes and send mesh chunks to the target node.

#### B.3.5 Aggregation, multicast and delegation
- **TRAM, sections (CProxySection_, CkMulticast, CkSectionInfo), delegation (ckDelegate, CkDelegateMgr), comlib:** **none**.
- **Charm++ collide library** (`-module collidecharm`). 6 calls:
  - `M2MTransfer.cpp:58` `CollideCreate` + `CollideSerialClient`;
  - `TransferDetails.cpp:47` `CollideRegister`;
  - `TransferDetails.cpp:131,160` `CollideBoxesPrio`;
  - `Discretization.cpp:218` `CollideSerialClientRestart`.
- **How collide is used:**
  - Internally the library aggregates voxel messages and reduces the collision list with `concat` to PE 0.
  - `collisionHandler` then runs on PE 0 and sends `processCollisions` to every destination chare (`M2MTransfer.cpp:137`).
  - Each destination chare sends `determineActualCollisions` to the source chares (`TransferDetails.cpp:221`), and the source chares reply with `transferSolution` (`:279`).

#### B.3.6 Collectives
- **`contribute(`:** 56 calls in 16 files.
- **Reduction operators:** nop 27, sum_double 13, sum_ulong 7, min_double 5, max_double 2, max_ulong 2, tuple 1 site (7 elements). Several sites contribute a `std::vector` directly, e.g. `Refiner.cpp:661`, `Discretization.cpp:268`.
- **Custom reducers:** 4 `addReducer` calls.
  - `Discretization.cpp:594` (PDF merge), `ElemDiagnostics.cpp:50` and `NodeDiagnostics.cpp:45` (diagnostics), `ChareStateCollector.cpp:39` (hash map).
  - Each reducer deserializes with `PUP::fromMem` and reserializes with `PUP::sizer`/`PUP::toMem` into `CkReductionMsg::buildNew`, e.g. `DiagReducer.cpp:63-142`.
  - Tuple reduction: `OversetFE.cpp:1110-1123`; read back with `toTuple` at `Transporter.cpp:1453`.
- **CkCallback forms:**
  - reductiontarget `CkCallback(CkReductionTarget(X,m), proxy)`: 56;
  - entry method `CkCallback(CkIndex_X::m(), proxy[i])`: 46;
  - ckExit, resumeThread, function pointer: 0;
  - callbacks stored in members and later invoked with `.send(...)`: 20 sites.
- **Broadcasts:**
  - Transporter calls `m_scheme[meshid].bcast<...>()` at 13 sites and makes whole-proxy calls on Refiner, Sorter, Partitioner and MeshWriter at about 12 sites (mostly setup).
  - Per step: the min-dt result is broadcast to the whole array (`ALECG.cpp:1001`, `FV.cpp:594`), and `bcast<refine>` follows each diagnostics reduction (`Transporter.cpp:1647`).
  - CG and BiCG broadcast reduction results every iteration (`ConjugateGradients.cpp:227,464`).
  - In setup, `Sorter.cpp:357` `thisProxy.offset(...)` is a broadcast from every element, nchare² deliveries.

#### B.3.7 Local access and node-level parallelism
- **ckLocal(): 45 hits in 12 files.** Calls go between bound arrays: Discretization to ALE (`Discretization.cpp:291-336`), ALE to ConjugateGradients (`ALE.cpp:297-731`), solver to Discretization (`OversetFE.hpp:428`), and in TransferDetails (`M2MTransfer.cpp:98,105`).
- **ckLocalBranch(): 13 hits in 5 files.** stateProxy (Ghosts), m2mtransferProxy (M2MTransfer, TransferDetails).
- **Not used:** CkLocalBranch, CkLocalNodeBranch, CkLoop, std::thread.
- **`#pragma omp`:** only `declare simd`, 4 sites in `Base/Vector.hpp` and `Mesh/DerivedData.hpp`.
- **Kokkos:** `Kokkos::initialize()`/`finalize()` in the Partitioner nodegroup (`Partitioner.cpp:99,688`), for Trilinos. A build-level Kokkos CUDA option exists (`src/CMakeLists.txt:115`), but the source contains no Kokkos kernels.

#### B.3.8 Migration, load balancing, checkpoint, quiescence detection and futures
- **Load balancing:**
  - `usesAtSync=true` ×4 and `AtSync()` ×4 (`ALECG.cpp:1675`, `DG.cpp:2185`, `FV.cpp:1052`, `OversetFE.cpp:1718`). AtSync is called when `It % lbfreq == 0` or `It == 2`.
  - With the `nonblocking` flag, `next()` is called right after `AtSync()`.
  - ResumeFromSync is defined 4 times and calls `next()` unless nonblocking (`ALECG.cpp:507-516`).
  - migrateMe and CkMigrate (other than CkMigrateMessage constructors): 0.
  - LB is turned off at startup: `TurnManualLBOn` in initproc `LBSwitch.cpp:39`. It is turned back on after setup by creating the LBSwitch group (`Transporter.cpp:1230`).
- **CkMigrateMessage constructors:** 23 hits in 20 files.
- **PUP:** `pup(` 192 times in 71 files. PUP::able: none (comment only, `Scheme.hpp:31`). Proxies are kept in a `std::variant` inside Scheme, which is pupped.
- **Checkpoint:** `CkStartCheckpoint` at `Transporter.cpp:1683`, every `rsfreq` steps, to disk. CkStartMemCheckpoint: 0. Restart goes through `Main(CkMigrateMessage*)`.
- **Quiescence detection:** `CkStartQD` at `Main/Init.hpp:125`, only under the `-q` debug option; quiescence is reported as a logic error. CkWaitQD: 0.
- **Futures and threads (CkFuture, CthSuspend, CthAwaken, CkCallbackResumeThread):** none.
- **Load balancers:** CMake links `-module CommonLBs`. Regression tests use `+balancer RandCentLB` (10) and `GreedyRefineLB` (1), with `+LBDebug 1 +cs`.

#### B.3.9 GPU
- hapi, cudaMemcpy, cudaIpc, hipMemcpy, CK_DEVICE, CMK_CUDA, nocopydevice, RAJA: **none**.
- Kokkos CUDA appears only as a compile and link wrapper (`src/CMakeLists.txt:115-140`, `Main/Inciter.cmake:50`). There is no device-memory messaging.

#### B.3.10 Internals
- **Headers:**
  - `charm++.h`, `charm.h`, `pup.h` and `pup_stl.h`, each included through a wrapper in `src/NoWarning/`;
  - `collidecharm.h` (`M2MTransfer.hpp:7`, `TransferDetails.cpp:18`);
  - generated `*.decl.h`/`*.def.h` files, through NoWarning wrappers;
  - `mpi.h` (`NoWarning/mpi.hpp`).
- **CMK_ macros:** none in C++. CMake tests two:
  - `CMK_SMP` (`cmake/FindCharm.cmake:150`, from conv-mach-opt.h), which sets `EXCEPTIONS_WRITE_TO_CERR`;
  - `CMK_RANDOMIZED_MSGQ` (`FindCharm.cmake:41-49`, from conv-autoconfig.h), which is detected and reported only.
- **_entryTable, _chareTable, Cpv/Ckpv/Cksv, _register\*:** none.
- **Runtime internals the app touches:**
  - `chareIdx = -1` in 20 files (22 hits). It writes Chare's internal member in every array element's constructor and migrate constructor. Per the comment at `Refiner.cpp:163-170`, this works around the charm 8.0.1 `chare_objs` table.
  - `CkGroupID(p).idx` is used as a map key at `M2MTransfer.cpp:76,93,97,104`.
  - The CkArgMsg cast described in B.3.2.
- **Custom array maps:** `tk::LinearMap` and `tk::UnsMeshMap` (CkArrayMap with procNum and populateInitial). They are used only in unit tests.
- **CkArrayIndex subclasses, CkLocMgr:** none.
- **CkArrayOptions:** `Scheme.hpp:132` `bindTo(discproxy)`; `M2MTransfer.cpp:78-80` `bindTo` + `setNumInitial`.

#### B.3.11 Build and version
- **Build line** (`quinoa-tpl/CMakeLists.txt:604-631`):
  - `buildold LIBS mpi-linux-x86_64 [CHARM_EXTRA_ARGS] [compiler] --build-shared --with-production -jN`. `--with-production` is used for Release builds; `--disable-charmdebug` is added on Alpine.
  - The machine layer is `mpi` (`:219,241`), or `multicore` when AMPI is enabled.
- **Options from the docs:**
  - SMP: `CHARM_EXTRA_ARGS="smp"` (`doc/pages/build.dox:193`), run with `+ppn`.
  - Randomized-queue debug build: `--enable-error-checking;--with-prio-type=int;--enable-randomized-msgq` (`build.dox:185`).
  - Projections: `--enable-tracing;--enable-commthread-tracing`.
- **Version:** submodule `src/charm` points to UIUC-PPL/charm `374907eb8` ("Migratable collidev701 (#3922)", 2026-06-03), on branch `charm-7.0.1` (v7.0.0 + 3 commits). There is no minimum-version check in CMake.
- **MPI requirement:** an MPI machine layer is required (`doc/pages/build_system.dox:61`).
- **Patch:** `cmake/charm-suppress-debugout.patch`, applied with `patch -p1` (quinoa-tpl `:617`). It comments out 17 `std::cout` lines in `src/libs/ck-libs/collide/collidecharm.C`; communication is unchanged. It does not apply to the recharm tip, whose collidecharm.C has no `std::cout`.
- **Modules:** `-module CommonLBs -module collidecharm` (`src/Main/CMakeLists.txt:13`; `src/CMakeLists.txt:301`).
- **reconverse, LCI:** no mention.

#### B.3.12 Ordering assumptions
- No comment or code relies on message order between two objects, and no exchange uses sequence numbers.
- **Synchronization method.** Exchanges use per-exchange counters (`if (++m_nrhs == NodeCommMap().size())`, `ALECG.cpp:1150`) and SDAG `when` joins. Stage separation comes from the dependency chain: a neighbor cannot send stage s+1 data until it has our stage s data. FIFO delivery is not used.
- **`fromstage` in DG.** `DG::comsol` carries `fromstage` (`DG.cpp:537`), but only to decide whether p-adaptive ndof/interface data is present at stage 0.
- **Randomized-queue testing.** The project tests explicitly under randomized message queues (`build.dox:177-185`, `contributing.dox:187`, `tests/regression/.../asynclogic/CMakeLists.txt:9`), plus RandCentLB migration tests.
- **Restart ordering.** `Discretization.cpp:211-221` adds a reduction barrier because collision messages could reach a PE before its collideMgr branch was restored.

#### Compatibility points
1. **CkCopyMsg/CkFreeMsg on the array-creation message** (`LinearMap.cpp:85-88`, `UnsMeshMap.cpp:190-193`). Change that could affect it: received-message allocation. The message comes out of the locally held CkMarshalledMessage, and the maps are used in unit tests only.
2. **Received messages never freed:**
   - `Transporter.cpp:1442` (CkReductionMsg, per OversetFE step);
   - `ALE.cpp:604,638,716` (CkDataMsg, per ALE CG solve);
   - `TestConjugateGradients.cpp:64,72`.
   - If received buffers come from a per-pair ring and are released on free, these hold ring slots indefinitely. The messages are small and usually same-PE (ALE is bound to CG, so its CkDataMsg is a local send), so whether they use the ring depends on the design.
3. **`reinterpret_cast<CkArgMsg*>` of the CkMigrateMessage** in the Main restart constructor (`Main/Inciter.cpp:248-254`). It depends on how restart builds the mainchare's migrate message.
4. **Runtime internals:** `chareIdx = -1` (20 files) and `CkGroupID(p).idx` (`M2MTransfer.cpp:76-104`). Neither is messaging, but both break if Chare or CkGroupID internals change.
5. **Collide library** (`-module collidecharm`; `TransferDetails.cpp`, `M2MTransfer.cpp`, `Discretization.cpp:218`).
   - It is Charm++-internal messaging: aggregator, voxel array, `concat` reduction to PE 0.
   - It must keep working, including across migration and restart. The app pins a custom collide-fix branch and patches it.
6. **MPI interop.** Zoltan2 is called inside the `[exclusive]` `Partitioner::partition` on all nodes (`Partitioner.cpp:225`, `ZoltanInterOp.cpp:150`). Omega_h uses `MPI_COMM_WORLD` (`Omega_h_MeshReader.cpp:66`), and `MPI_Allreduce` runs in ExceptionMPI. The new transport must coexist with direct MPI calls from entry methods on the mpi layer. Quinoa cannot run on reconverse or another non-MPI layer as built today.
7. **Migration.**
   - AtSync/ResumeFromSync on the 4 solver arrays, with 8 bound arrays migrating together.
   - Nonblocking mode keeps the halo exchanges running during LB: `AtSync(); if (nonblocking) next();` at the 4 sites listed in B.3.8.
   - Any adoption of persistent regions must handle teardown and rebuild while sends are in flight. AMR (`resizePostAMR`, `Discretization.cpp:506-529`) also replaces `NodeCommMap`, so it is a second rebuild point.
8. **`[exclusive]` nodegroup entries** (5) and **`[expedited]`** (2, unit-test only). Their semantics must survive the transport change, since an SMP build is supported.
9. **Ordering:** no assumptions (B.3.12). The app is a good randomized-order test case.
10. **Not touched by Quinoa:** zerocopy, nocopy*, CkDevice*, the reconverse persistent API, CkDirect, Converse sends, handlers, CmiAlloc and envelope access. None of the APIs that will be reimplemented or removed are used.

#### Candidates for the persistent family
All of these send to the same partners every iteration, with sizes fixed between AMR and LB events. Payloads today are `vector<size_t> gid` (re-sent every time) plus `vector<vector<real>>`, so they would need flattening into a contiguous region; the gid list could be exchanged once per rebuild.

| Pattern | Code | Partners | Per-message payload | Frequency |
|---|---|---|---|---|
| ALECG/OversetFE node partial sums: gradients `comChBndGrad`, rhs `comrhs` | `ALECG.cpp:1042-1046, 1119-1123`; OversetFE same | node-sharing chares, `NodeCommMap().size()` | shared nodes × (8 B gid + 3·ncomp reals) and × (8 B + ncomp reals); ncomp=5 for CompFlow | 2 exchanges × 3 RK stages per step |
| DG/FV ghost-cell exchange `comsol`, `comreco`, `comlim` (DG also `comale`, `comrefine`, `comsmooth`) | `DG.cpp:518-537, 869-882, 970-984`; FV same | face-neighbor chares, `m_sendGhost.size()` | ghost elements × (8 B tetid + ncomp·rdof reals for u + nprim·rdof for prim [+2 size_t]); rdof 4 (P1) or 10 (P2) | 2-3 per RK stage, 2-3 stages per step |
| CG solver q=A·p halo `comq` (+ `comres`/`comx` at start/end) + 2 allreduces/iter | `ConjugateGradients.cpp:409-418, 464, 227` | `NodeCommMap` neighbors | shared nodes × (8 B + ncomp reals) | every CG iteration (ALE mesh smoothing) |
| BiCG halo `comq`, `comt` (+ `comres`, `comx`, `comx2`) + 4-5 allreduces/iter | `BiCG.cpp:449, 643, 759` | `NodeCommMap` neighbors | as CG | every iteration (implicit/IMEX DG) |
| ALE node exchanges `comvort`/`comdiv`/`compot`/`comfor` | `ALE.cpp:413, 428, 666, 828` | `NodeCommMap` neighbors | shared nodes × (8 B + 1-3 reals) | per ALE mesh-velocity update |
| Min-dt allreduce + broadcast; diagnostics many-to-one (custom reducer → Transporter) | `ALECG.cpp:1000`, `NodeDiagnostics.cpp:167` | all elements | 8 B; NUMDIAG × ncomp reals | every step; every diag interval |
| Field output many-to-one per node (chares → MeshWriter on node's first PE); DG/FV `comnodeout` halo first | `Discretization.cpp:990`, `DG.cpp:668` | chares on node → 1 PE | whole chare mesh + all fields (large, variable) | output interval; poor fit (sizes change with AMR) |
| Overset transfer through collide | `TransferDetails.cpp:221, 279` | geometry-dependent, changes as the mesh moves | `vector<SolutionData>` (index + ncomp reals each) | every OversetFE step; poor fit (partners vary) |

### B.4 Enzo-E

| | |
|---|---|
| App | Enzo-E (framework Cello in `src/Cello`, application Enzo in `src/Enzo`) |
| Repo, branch | https://github.com/enzo-project/enzo-e, `main` (shallow clone) |
| Commit | `4798687a1c0e46489af679a555e22ad3f992fede`, 2025-08-13 16:54:54 -0400 ("Merge pull request #428 from gregbryan/double-fix") |
| Files scanned | 645 source files in `src/` (7 `.ci`, 271 `.cpp`, 312 `.hpp`, 5 `.h`, 46 `.F`, 2 `.c`, 1 `.y`, 1 `.l`) plus CMake, `config/`, `.circleci/`, `test/`, `tools/` (704 files in all); 96 `doc/**/*.rst` for build instructions and keywords |
| Excluded | `src/External/` (vendored pngwriter: `pngwriter.cpp/.h`, `pngconf.h`) |
| Other branches | only `main` and `pre-commit-ci-update-config`; none suggest communication work |

#### Summary

- All data movement is ordinary entry-method invocation on chare-array elements (`Block`/`EnzoBlock`, one element per AMR octree block, custom index `Index`). There is no Converse-level messaging, no zerocopy API, no TRAM, no sections and no GPU code.
- Bulk data (ghost zones, particles, fluxes, refine/coarsen payloads, output, checkpoint) travels in 8 custom-packed message classes (`MsgRefresh`, `MsgRefine`, `MsgCoarsen`, `MsgAdapt`, `MsgOutput`, `MsgInitial`, `MsgState`, `EnzoMsgCheck`). Each holds a `DataMsg` that points at the sender's live `FieldData`. The static `pack()` serializes that live data into a buffer from `CkAllocBuffer`. The static `unpack()` allocates a new message with `CkAllocBuffer` and keeps the raw received buffer (`buffer_`) so that ghost arrays can be read directly from it. The app frees that buffer later with `CkFreeMsg`.
- The app tells same-PE delivery from remote delivery by whether `unpack()` ran (`is_local_`). For a same-PE send it reads the sender's field arrays through raw pointers at receive time.
- Synchronization uses counters (class `Sync`, plus `sync_count_`/`sync_max_` in `Block`), early-arrival lists, reductions (`contribute`, 33 sites), and `CkStartQD` for the adapt phases. One varsize message, `FieldMsg`, serves the multigrid solvers.
- Ghost refresh works like this. Each block sends one `MsgRefresh` per neighbor face, edge and corner (26 neighbors in 3D at uniform resolution, up to 56 next to finer blocks). It sets its expected count to the number of messages it sent. It processes arrivals and then does a zero-payload neighbor handshake (`p_control_sync_count`) before it continues.
- Load balancing: Charm++ `AtSync` (TreeLB by default) and an app-driven `migrateMe` space-filling-curve balancer. AMR inserts blocks with `insert()` and destroys them with `ckDestroy()`.

#### B.4.1 Interface files

- **.ci files: 7.** `Cello/{main,main_enzo,main_mesh,main_simulation,mesh,simulation}.ci`, `Enzo/enzo.ci`. The mainchare declaration is in `Cello/main_include.h`, which four `.ci` files `#include`.
- **Chare kinds.**
  - mainchare: 1 (`Main [migratable]`, main_include.h:9).
  - chare: none.
  - group: 5. `Simulation`, `EnzoSimulation : Simulation`, and `MappingArray`/`MappingTree`/`MappingIo : CkArrayMap`. All are `[migratable]`.
  - nodegroup: none.
  - array: 7.
    - `Block` and `EnzoBlock : Block` use `array[Index]`.
    - `EnzoLevelArray` uses `array[Index3]`.
    - `IoReader`, `IoWriter`, `IoEnzoReader` and `IoEnzoWriter` use `array[1D]`.
- About 184 entry methods: 48 take `CkReductionMsg*`, 1 takes `CkArgMsg*`, and 16 take app messages. The rest are marshalled. The marshalled ones include arrays: `char a[n]` in `p_refresh_child`, `p_output_write`, `p_method_infer_*`; `long double dot[n]` plus `std::vector<int>` in `p_dot_recv_*`; `enzo_float field_data[nf]` in `p_transfer_data`.
- **Messages: 9 declared** (mesh.ci:85-94, enzo.ci:123).
  - `FieldMsg { char a[]; }` is the only varsize message.
  - The other 8 are plain `message X;` declarations whose C++ classes define static `pack`/`unpack`. The charmxi registration `X::pack` picks those up, so they act as packed messages.
  - The keywords `[packed]` and `[varsize]` occur 0 times.
- **Entry attributes** (`entry *[`): 0 hits. inline, expedited, immediate, nokeep, threaded, sync, local, exclusive, reductiontarget, createhere, createhome, aggregate, whenidle, notrace, python and accel are each 0.
- **Parameter keywords:** nocopy 0, nocopypost 0, nocopydevice 0.
- **SDAG:** when, serial, overlap, forall and case are each 0.
- **Templates:** no templated entry methods or chares.
- **Other declarations:** 13 `initnode` functions (6 register custom reducers, 6 create node locks, 1 registers the turbulence reducer) and 22 `readonly` declarations.

#### B.4.2 Messages in C++

- **`CMessage_` subclasses: 9.** `Cello/charm_FieldMsg.hpp:11`, `charm_Msg{Adapt,Coarsen,Initial,Output,Refine,Refresh,State}.hpp`, `Enzo/enzo-core/EnzoMsgCheck.hpp:19`.
- **Allocation and free calls.**
  - CkAllocMsg 0, CkCopyMsg 0, CkReferenceMsg 0.
  - **CkFreeMsg 13**: 7 in destructors and 6 in `update()` methods, all on `buffer_`, the raw received buffer.
  - **CkAllocBuffer 16**: one in every `pack` and one in every `unpack`.
  - Representative call sites:
    - `charm_MsgRefresh.cpp:66`: `CkAllocBuffer(msg,size)` allocates the packed buffer. The buffer carries a copy of `msg`'s envelope. `delete msg` follows at :77.
    - `charm_MsgRefresh.cpp:97-113`: `unpack` calls `CkAllocBuffer(buffer,sizeof(MsgRefresh))`, does a placement `new`, and saves `buffer_ = buffer`. `data_DataMsg.cpp:279` sets `field_array_u_` to a pointer *into the received buffer*.
    - `charm_MsgRefresh.cpp:127`: `CkFreeMsg(buffer_)` after the ghost data has been copied into the field (in `update`).
- **Static pack and unpack:** 8 classes, 16 functions. No static `alloc`. `FieldMsg` uses the generated `new (narray) FieldMsg` (EnzoSolverMg0.cpp:949, :1029; EnzoSolverDd.cpp:468).
- **Unusual details.**
  - Every `pack` except `MsgState::pack` begins with `if (msg->buffer_ != nullptr) return msg->buffer_;` (for example charm_MsgRefine.cpp:126). If a received message were re-sent, this would return the old received buffer with its old envelope.
  - The app avoids re-sending received messages. `problem_MethodOutput.cpp:366` says "copy incoming message and delete old (cannot reuse!)": the copy constructor resets `buffer_`.
  - `MsgState::unpack` (charm_MsgState.cpp:87-107) never frees the received buffer. That buffer leaks once per PE per restart.
- **Received messages stored instead of deleted.**
  - `Block::refresh_msg_list_` (mesh_Block.hpp:1045). `MsgRefresh` messages that arrive before the block enters that refresh wait here (control_refresh.cpp:221) until `refresh_wait` (:133-148).
  - `Block::adapt_msg_list_` (control_adapt.cpp:607-608) holds `MsgAdapt` until `adapt_ready_`.
  - `EnzoSimulation::msg_refine_map_` (EnzoSimulation.cpp:113) holds `MsgRefine` until the inserted block's constructor fetches it (EnzoBlock.cpp:44 → `get_msg_refine`).
  - Mg0 and Dd solvers store `FieldMsg*` in block scalar data (EnzoSolverMg0.cpp:648, :756; EnzoSolverDd.cpp:204, :318) until the `Sync` count completes.
  - Each of the 7 classes that keep `buffer_` holds the raw received buffer from `unpack` until `update()` or the destructor.
  - Neither list is PUPed. `Block::pup` (mesh_Block.cpp:339-345) clears `refresh_msg_list_` on unpacking, so the code assumes the lists are empty at migration.
- **Re-sent messages:** none. Output forwarding copies the message first (problem_MethodOutput.cpp:367, :409).
- **Envelope access** (UsrToEnv, EnvToUsr, `envelope`): 0 direct uses. `CkAllocBuffer` reads and copies the envelope internally.
- **CkSetRefNum and CkGetRefNum:** 0.
- **Priority and queueing** (CkSetQueueing, CkPriorityPtr, CkEntryOptions, setPriority, CK_QUEUEING_, CkSetMsgPriority): 0.

#### B.4.3 Zerocopy and RDMA

- **Calls:** CkNcpyBuffer, CkSendBuffer, CkDeviceBuffer, CkDevicePersistent, CkNcpyMode, CK_BUFFER_, CkPostBuffer, CkMatchBuffer, CkRdmaWrapper, CkRdma, CkDirect and CmiDirect each have 0 hits, in source, docs and comments.
- **Words** `ncpy`, `zerocopy`, `zero-copy`, `rdma`, `persistent`: 0 hits (the only `ncpy` matches are `strncpy`).
- **Related intent:** comments "WARNING: double copy" at EnzoSolverMg0.cpp:951 and :1031 (field data → temporary array → `FieldMsg`).

#### B.4.4 Converse-level calls

- **Messaging and handlers.** CmiSyncSend and its variants, CmiPushPE/CmiPushNode, CmiRegisterHandler, CmiSetHandler, CmiAlloc, CmiFree and CmiReference: all 0.
- **Scheduler.** CsdEnqueue, CqsEnqueue, CmiDeliverMsgs, CsdScheduler, CsdSchedulePoll: 0.
- **Persistent API.** CmiPersistent, CmiCreatePersistent, CmiUsePersistentHandle, CmiDestroyPersistent, CmiDestoryPersistent, CmiCreateCompressPersistent: 0.
- **Locks and barriers.** CmiCreateLock 6, CmiLock 8, CmiUnlock 8 (`CmiNodeLock`), in 6 files: problem_MethodCloseFiles.cpp, mesh_Hierarchy.cpp, problem_InitialValue.cpp, data_FieldFace.cpp, EnzoSolverBiCgStab.cpp, EnzoInitialMusic.cpp.
  - data_FieldFace.cpp:23 forces `#define CONFIG_SMP_MODE`, and :446 takes a node-wide lock around `face_to_face`, the same-PE ghost copy.
  - mesh_Hierarchy.cpp:221 locks around node-wide block counters.
  - CmiNodeBarrier and CmiNodeAllBarrier: 0.
- **Fences.** CmiMemory*Fence: 0.
- **Topology.** Cmi* topology calls: 0. Ck-level calls: CkNumNodes 6, CkMyRank 1 (monitor_Monitor.hpp:114), and `CkNumPes()/CkNumNodes()` as node size (problem_MethodCloseFiles.cpp:75, EnzoInitialMusic.cpp:560).
  - The I/O throttle assumes PEs are numbered contiguously per node (`CkMyPe()/node_size`).
  - `cello::index_static()` = `CkMyPe() % CONFIG_NODE_SIZE` (cello.hpp:743) indexes per-PE static arrays.
- **Blocking inside entry methods.** `std::this_thread::sleep_for` throttles I/O inside entry methods (problem_MethodCloseFiles.cpp:85, EnzoInitialMusic.cpp:570), with a node lock held around file closes.

#### B.4.5 Aggregation, multicast and delegation

- TRAM (MeshStreamer and variants, tram): none.
- Sections (CProxySection_, CkSectionInfo, CkMulticast, CkSectionID, ...): none.
- Delegation (ckDelegate, CkDelegateMgr): none.
- Comlib: none.
- Enzo-E builds its own tree collectives over the block array with point-to-point sends. Examples: the BiCgStab dot product (EnzoSolverBiCgStab.cpp:1700, :1757) and the Morton/Hilbert weight and index passes (problem_MethodOrderMorton.cpp:107, :172).

#### B.4.6 Collectives

- **`contribute(`: 33 sites in 19 files.** Most are in EnzoSolverCg (5), MethodOrderMorton and MethodOrderHilbert (3 each), and control_initialize (3). `control_sync_barrier` (control_charm.cpp:201) is the generic phase barrier.
- **Custom reducers: 7 `CkReduction::addReducer`** (charm_reductions.cpp:8, 58, 108, 131, 159, 187; EnzoMethodTurbulence.cpp:187). There are 7 `CkReductionMsg::buildNew`, 14 `reducerType`, 0 `tupleElement`/`setElement`. Built-ins used: concat, sum_int, sum_double, min_double, max_int.
- **CkCallback forms.**
  - Entry method plus whole array (`thisProxy`). This is the common case, so most reduction results are broadcast to all blocks.
  - Entry method plus array element: `CkCallback(int ep, CkArrayIndexIndex(index_), thisProxy).send(NULL)` (control_charm.cpp:217, :319; compute_Solver.cpp:164). Blocks send these to themselves to continue after a sync.
  - Entry method plus group element on PE 0 (control_restart.cpp:187, EnzoMethodCheck.cpp:100, EnzoMethodBalance.cpp:75).
  - Entry method plus mainchare, as the `CkStartQD` target.
  - Entry-point ints from `CkIndex_X::y()` are stored in `Refresh` objects and PUPed.
  - No reductiontarget, ckExit, resumeThread or function-pointer callbacks.
- **Broadcasts: about 35 explicit sites.**
  - `Main` broadcasts phase transitions to `block_array()` (main.cpp:127-306, 14 sites).
  - Others: `p_method_balance_migrate`/`_done`, `p_check_done`, `p_restart_done`, `p_method_infer_exit`.
  - Group broadcasts: `proxy_enzo_simulation.p_initialize_state(MsgState*)` (control_restart.cpp:619) is a broadcast of a custom-packed message.
  - In addition, about 40 reduction callbacks broadcast their result to the whole array.

#### B.4.7 Local access and node-level parallelism

- `ckLocalBranch(`: 2 hits (cello.cpp:199, enzo.cpp:9). They return the local `Simulation` group branch, which the app uses everywhere, including to pass stored `MsgRefine`/`EnzoMsgCheck` pointers to new blocks.
- `ckLocal(`, CkLocalBranch, CkLocalNodeBranch, CkLoop: 0.
- `#pragma omp`: 2, both `simd` (EnzoRiemannUtils.hpp:300, EnzoRiemannImpl.hpp:332).
- std::thread: none created (only `this_thread::sleep_for`).
- Kokkos: 1 comment (view_ViewCollec.hpp:252).

#### B.4.8 Migration, load balancing, checkpoint, QD and futures

- **Load balancing and migration.**
  - `usesAtSync = true` (mesh_Block.cpp:80), `setMigratable(!is_root)` (:257).
  - `AtSync()` (control_stopping.cpp:231) and `ResumeFromSync()` (:237).
  - App balancer: `migrateMe(ip_next)` (EnzoMethodBalance.cpp:118). Each migrated block's migration constructor sends `p_method_balance_check` to `EnzoSimulation[0]` (EnzoBlock.cpp:26). That group element counts to `count_total+1` (EnzoMethodBalance.cpp:91, :131), then calls `doneInserting` and broadcasts `p_method_balance_done`.
  - Dynamic insertion: `block_array()[index].insert(ip, MsgType::msg_refine, ip)` (EnzoSimulation.cpp:116). Deletion: `ckDestroy()` (control_adapt.cpp:1004). `doneInserting` is called 5 times.
- **PUP:** 558 `pup(` hits. 132 `PUPable` declarations in `.ci` (48 mesh, 84 enzo) and 439 `PUPable_decl`/`PUP::able` lines in C++.
- **Checkpoint:** `CkStartCheckpoint(dir, cb, false, 1)` (main.cpp:112, the 4-argument form). CkStartMemCheckpoint 0, CkRestart 0. There is also an app-level HDF5 checkpoint (`EnzoMethodCheck`, `IoEnzoWriter`, `EnzoMsgCheck`).
- **Quiescence detection:** `CkStartQD` 1 site (control_charm.cpp:194). It is reached from 6 adapt-phase transitions per adapt cycle (control_adapt.cpp:166, :243, :245; mesh_Block.cpp:252; control_charm.cpp:83). CkWaitQD 0.
- **Futures and threads:** CkFuture, CthSuspend, CthAwaken, CkCallbackResumeThread: 0.
- **Load balancers named:** build defaults `-module CommonLBs -balancer TreeLB` (CMakeLists.txt:150-161). Docs mention `+balancer GreedyLB` (getting_started_frontera.rst:50). Inputs use the app's own "balance" method.

#### B.4.9 GPU

hapi, HAPI, cudaMemcpy, cudaIpc, hipMemcpy, CK_DEVICE, CMK_CUDA, nocopydevice, Kokkos and RAJA: no code. Docs mention GPUs only as a future direction. config/msu_hpcc_gcc.cmake loads a CUDA module but does not use it.

#### B.4.10 Internals

- **Charm++ headers included:** `charm++.h` and `pup_stl.h` only, plus the generated `*.decl.h`/`*.def.h`. No internal headers (`envelope.h`, `ck.h`, `register.h` and so on). `charm.hpp` is the app's own header.
- **`CMK_` macros tested in C++:** none. CMake reads two macros at configure time:
  - `CMK_RANDOMIZED_MSGQ`, read from `<include>/../tmp/conv-autoconfig.h` (FindCharm.cmake:130-150, called at :306). A missing file is a fatal error.
  - `CMK_SMP`, by compiling a probe against `conv-mach-opt.h`/`conv-autoconfig.h` (FindCharm.cmake:330-370).
  - A reconverse build satisfies both: `tmp` links to `include`, and `CMK_SMP` is 1.
- **Runtime internals** (`_entryTable`, `_chareTable`, CkpvAccess, CpvAccess, CksvAccess, `_register*`): 0.
- **Custom array index:** `CkArrayIndexIndex : CkArrayIndex` (mesh_Index.hpp:249-256) does a placement `new` of a 3-int `Index` into `CkArrayIndex::index` and sets `nInts=3`. `Index` is `PUPbytes`. `CkArrayIndexIndex3` is the same for `Index3`.
- **Custom array maps:** 3 `CkArrayMap` groups with `procNum` (charm_MappingArray.cpp:22, charm_MappingTree.cpp:22, charm_MappingIo.cpp:20). `CkArrayOptions.setMap` at mesh_Factory.cpp:71, control_restart.cpp:143, EnzoMethodCheck.cpp:60. CkLocMgr: 0.

#### B.4.11 Build and version

- **How Charm++ is built:**
  - CI (`.circleci/config.yml:10, 46`): Charm++ 7.0.0 with `./build charm++ netlrts-linux-x86_64 -j4 --with-production`.
  - Docs (getting_started.rst:39-75) recommend v7.0.0 built with CMake, `-DNETWORK=mpi` (clusters) or `netlrts` (laptops), `-DSMP=OFF`. Frontera, Pleiades and Expanse use mpi with `SMP=OFF`.
  - The app's `-Dsmp=ON` (default OFF) adds `CONFIG_SMP_MODE`. CUDA is not used.
- **Minimum version:** none checked in code. main.cpp:113-114 says the 2-argument `CkStartCheckpoint` was for "version < 7.0.0", so 7.0.0 is the de facto minimum.
- **Patches to Charm++:** none. CMakeLists.txt:83-92 has a workaround flag `bypass_charm_mem_leak` (ON, defines `BYPASS_CHARM_MEM_LEAK`) for a Charm++ memory leak when a message is passed to an array-element constructor through `insert(msg)`. The code now always uses the bypass (EnzoBlock.cpp:308-313 keeps the old form as a comment), so the macro is unused.
- **Modules:** `-module CommonLBs` and `-balancer TreeLB` (if `balance=ON`), `-tracemode projections` (if `use_projections`).
- **Reconverse and LCI:** not mentioned.
- **Build hazard:** `FindCharm.cmake` requires `charmrun` (`FIND_PACKAGE_HANDLE_STANDARD_ARGS(... CHARM_RUN ...)`), and tests launch through charmrun. A reconverse build has no `charmrun` in `bin/`, so configuring against it needs `-DCHARM_RUN=...`. That is a cache override, not a source change.

#### B.4.12 Ordering assumptions

- No comment or code claims to rely on FIFO order between two objects, and no message carries a per-sender sequence number.
- Mechanisms that guard against early arrival:
  - **Refresh:** `MsgRefresh` carries `id_refresh`. Arrivals before the block reaches READY are buffered (control_refresh.cpp:206-222). A neighbor cannot send round k+1 of the same refresh id before this block finishes round k, because every Method refresh ends in a `sync_neighbor` handshake (problem_Method.cpp:52-54; control_charm.cpp:206-262).
  - **Adapt:** `MsgAdapt` carries `adapt_step_`, and the receiver asserts it equals its own step (control_adapt.cpp:678-682). The commented-out early `return` at :677 shows that arrivals from other steps were once dropped. Phases are separated by QD.
  - **Morton order:** uses a reduction "to ensure counters initialized before first entry method can arrive" (problem_MethodOrderMorton.cpp:51-52).
- problem_MethodFluxCorrect.cpp:82 says "neighbor_flux causes synchronization errors; using neighbor_leaf". That is a count mismatch, not an ordering problem.

#### Compatibility points

Each item gives the site, what it does, and the planned change that could affect it.

1. **Raw received buffers kept after `unpack()` and freed later with `CkFreeMsg`.** All custom packed messages except `MsgState` do this: charm_MsgRefresh.cpp:113/:127/:35, charm_MsgRefine.cpp:309/:369/:93, and the same in MsgCoarsen, MsgAdapt, MsgOutput, MsgInitial and EnzoMsgCheck.
   - Ghost arrays are read in place from the received buffer (data_DataMsg.cpp:279, :372).
   - If the buffer is waiting in `refresh_msg_list_`, it is held for a whole compute phase.
   - **Affected change:** message family. A mid-size ring-buffer slot would stay occupied for that time. `CkFreeMsg` must accept a buffer whose message the runtime already replaced through `unpack`, and it must accept `nullptr`.
2. **`CkAllocBuffer(msg,size)` in `pack` and `CkAllocBuffer(buffer,sizeof(X))` in `unpack`** (16 sites). These rely on `CkAllocBuffer` copying the envelope of an existing message, and on `pack`/`unpack` being allowed to return a different pointer.
   - **Affected change:** message family internals and the envelope layout.
3. **The local versus remote decision depends on whether pack/unpack ran** (`is_local_`, charm_MsgRefresh.cpp:101; data_DataMsg.cpp:360-373; charm_MsgRefine.cpp:352-366).
   - For a same-PE send, the receiver reads the sender's `FieldData*`, `FieldFace*` and `Adapt*` directly (control_refresh.cpp:351; control_adapt.cpp:377, :934). The local child-creation path is `proxy_enzo_simulation[CkMyPe()].p_refine_create_block` (EnzoFactory.cpp:242).
   - `MsgOutput` carries raw `MethodOutput*` and `FileHdf5*` pointers across processes and back (charm_MsgOutput.cpp:138-139).
   - **Required by the message family:** a delivery to a different PE, including one in the same process, must call `pack`/`unpack`. A same-PE delivery must not.
4. **`pack()` reads the sender's live block data through pointers held in the message.** For example `DataMsg::save_data` → `face_to_array` (data_DataMsg.cpp:197).
   - **Affected change:** rendezvous. If the send path delayed calling `pack()` until after the send call returned, the message would carry data from a later state.
5. **Received messages stored for later.** `refresh_msg_list_`, `adapt_msg_list_`, `msg_refine_map_`, and the Mg0 and Dd `FieldMsg` slots (B.4.2).
   - **Affected change:** message family, for messages held a long time.
6. **`if (buffer_) return buffer_;` in pack** (charm_MsgRefine.cpp:126 and 6 others). This path is unused today because forwarding copies the message. If it ran, it would send a stale envelope.
   - **Affected change:** message family, for re-sent received messages.
7. **`MsgState::unpack` leaks the received buffer** (charm_MsgState.cpp:87-107). A leaked buffer in a ring-buffer slot would never be released.
   - **Affected change:** message family.
8. **`CkStartQD` drives the adapt phases** (control_charm.cpp:194). Any rendezvous or put traffic must be counted by QD.
   - **Affected change:** message family and persistent family.
9. **Migration and dynamic membership.**
   - `AtSync`/`ResumeFromSync` (control_stopping.cpp:231/:237) and `migrateMe` (EnzoMethodBalance.cpp:118).
   - `insert`/`ckDestroy` every adapt phase (EnzoSimulation.cpp:116, control_adapt.cpp:1004).
   - `Block::pup` drops buffered message lists (mesh_Block.cpp:339-345).
   - **Affected change:** migration. Regions must be rebuilt at all three kinds of event.
10. **Converse locks.** `CmiCreateLock`/`CmiLock`/`CmiUnlock` (6 files). These are not messaging, but they are Converse API. data_FieldFace.cpp:446 takes the lock on the same-PE ghost copy.
    - **Affected change:** Converse-level calls.
11. **Build-time probes and launcher.** `CMK_RANDOMIZED_MSGQ`, `CMK_SMP`, `../tmp/conv-autoconfig.h` and the `charmrun` requirement (FindCharm.cmake). LB modules (`CommonLBs`, `TreeLB`).
    - **Affected change:** none in the comm layer. This is a build compatibility point for reconverse.
12. **Entry methods that sleep.** Entry methods sleep with `sleep_for` (problem_MethodCloseFiles.cpp:85, EnzoInitialMusic.cpp:570). On reconverse, which has no comm thread, a sleeping PE makes no network progress.
    - **Affected change:** progress assumptions only.
- **Not present:** zerocopy APIs, nocopy keywords, [nokeep], CkReferenceMsg/CkCopyMsg, envelope access, internal headers, `CMK_` macros in C++, Charm++ patches, and pairwise-order dependence.

#### Candidates for the persistent family

Sizes assume double precision, a block of n³ cells, ghost depth g (inputs use 4 in 55 files and 3 in 14), and nf fields in the refresh's field list. The partner set is fixed between adapt phases. Adapt (and the `p_refresh_child`/`MsgCoarsen` traffic) rebuilds it. Migration moves partners but does not change them.

1. **Ghost refresh** (`refresh_load_field_face_`, control_refresh.cpp:337-357 → `p_refresh_recv`).
   - **Pattern:** every Method refreshes at least once per cycle (`ir_post_`, control_compute.cpp:57). The CG, BiCgStab and Jacobi solvers refresh every iteration (EnzoSolverCg.cpp:306, :340; EnzoSolverJacobi.cpp:276).
   - **Partners:** with `min_face_rank=0` there are 26 at uniform level in 3D. With finer neighbors there are up to 56 (4 per face, 2 per edge, 1 per corner). Several messages can go to the same coarser block. When `coarse_padding>0`, extra "coarse array" messages go to third blocks (control_refresh.cpp:363-622).
   - **Sizes:** about g·n²·nf·8 B per face (n=16, g=4, nf=10: 80 KB; n=32: 320 KB). About g²·n·nf·8 B per edge (20 to 40 KB). About g³·nf·8 B per corner (5 KB). Data to a coarser neighbor is restricted before sending (data_FieldFace.cpp:283). Data from a coarser neighbor is prolonged by the receiver (:376).
   - **Counting today:** `Sync::set_stop(count)`, where count is the number of messages this block *sent* (control_refresh.cpp:80-89). Asymmetric padded cases count receives explicitly ("only count receives", :442).
     - `p_refresh_recv` increments with `advance()` if the state is READY, or else buffers the message.
     - Completion means `stop==0 || (is_done && READY)` (:169-170). It runs `refresh_coarse_apply_` and then the callback through `control_sync`.
   - **Fit with the counter API:** this maps to a counter with N set at refresh start. Arrivals before the block started must not fire the action. The app already solves this elsewhere by adding 1 to N for its own call.
   - **Landing buffers:** the ghost zones are strided across several fields, so a put would land in a per-(neighbor, face) buffer, and the receiver would still run `array_to_face`.
2. **Neighbor handshake after each refresh** (`p_control_sync_count`, control_charm.cpp:252/:285).
   - **Pattern:** zero-payload messages to the same 26 to 56 neighbors, counted against `num_neighbors+1` (:260, :294-321). The action is a self-send through `CkCallback(ep, idx, proxy)`.
   - **Fit:** this is a pure remote counter increment. It is also exactly the app-level "receiver buffer is free" guarantee that a put-with-notify into a persistent landing buffer needs before the next round.
3. **Flux correction** (`MethodFluxCorrect`, problem_MethodFluxCorrect.cpp:120; `refresh_load_flux_face_`, control_refresh.cpp:1414-1505).
   - **Pattern:** once per cycle. Each block sends to its face neighbors only (6 in 3D at uniform level).
   - **Payload:** only toward a coarser neighbor, as coarsened `FaceFluxes`: nf·(n/2)²·8 B (n=32, nf=5: 10 KB). Same-level and finer neighbors get empty messages that only count. The refresh also carries the conserved fields.
4. **Multigrid restrict and prolong** (EnzoSolverMg0.cpp:584-588, :714-719, :909-1060; EnzoSolverDd.cpp:186, :294).
   - **Restrict:** children send to their parent, a many-to-one gather. There are 2^rank = 8 contributions plus 1 for the parent itself (`set_stop(1+num_children)`, :228). Each is about (n/2)³·8 B (n=32: 32 KB).
   - **Prolong:** the parent sends to its 8 children, about (n/2+2g)³·8 B each.
   - **Frequency:** every V-cycle.
   - **Copies:** each `FieldMsg` is a double copy (:951), so this is a direct put candidate.
5. **BiCgStab tree dot product** (EnzoSolverBiCgStab.cpp:1700, :1757).
   - **Pattern:** a gather from 8 children up the octree (count `num_children`, :388), then a scatter back down. Several times per iteration.
   - **Size:** n `long double` (tens of bytes) plus a `vector<int>`.
   - **Fit:** a counter-with-small-payload candidate.
6. **Lower-frequency exchanges** (per load-balance or adapt step):
   - The Morton/Hilbert weight gather and index scatter (8 children).
   - The migration count on `EnzoSimulation[0]` (N = number of migrating blocks).
   - The `MsgAdapt` level exchange with neighbors, which is repeated over adapt steps and buffered when the block is not ready.
   - These are counters, not data regions.
7. **Not good region candidates:**
   - Particle refresh (`particle_send_`, control_refresh.cpp:942-984) sends to every neighbor slot, but the size depends on how many particles leave the block. Empty messages are still sent for counting, so it needs a counter plus variable-size messages.
   - Output, checkpoint and restart traffic is not repeated with the same partners.
