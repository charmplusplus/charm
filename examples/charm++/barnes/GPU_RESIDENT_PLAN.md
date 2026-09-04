# Making barnes GPU-resident

A staged plan for moving the Barnes-Hut pipeline in this directory onto the
device. Written against the current tree at `examples/charm++/barnes/`, whose
port is described in `README`: "The tree walk stays on the host; only the force
evaluation moves to the device."

The organizing invariant for the finished state:

> **Nothing that scales with particle count crosses PCIe.** Particle and LET
> bulk data move PE-to-PE over the device transport and never touch host
> memory. Host↔device traffic per iteration is a few tens of KB of control
> data.

Symbols used throughout: `N` particles per PE, `B` buckets per PE, `T` tree
pieces, `P` PEs, `D` tree depth. At the current `-p=128` on 4 PEs with 50K
particles: `N` ~ 12.5K, `B` ~ 100, `T` = 106, `P` = 4, `D` ~ 15.


## 1. Where the device is used today

Only four things live on the GPU:

- `d_partPos` (`float4`, pos+mass) and `d_accel` (`float4`, accel+potential),
  owned per-PE by `GpuParticleStore` in `GpuBatch.h`.
- `d_srcs` / `d_descs`, the flattened interaction list, owned per-TreePiece by
  `GpuTraversalBatch`.
- `gravKernel` in `barnes.cu`.

Everything else is host work: KDK integration, SFC key generation, two
`quickSort` passes, 15 histogram rounds, the all-to-all particle exchange
through `ParticleMsg`, the pointer tree build (one `new Node[2]` per split),
the postorder moment pass, and the full tree walk for every bucket. Particles
round-trip host↔device every iteration through two O(N) conversion loops in
`GpuParticleStore::upload()` / `apply()`.

Two structural facts make device residency tractable here:

1. **`DataManager` is a group and never migrates**, and already owns the
   particles. Device residency lands in a chare that never moves, so the
   README's "nothing device-side survives a migration" invariant is preserved
   for free.
2. **Particles are SFC-key-sorted**, so every ownership range is contiguous —
   tree piece slice, node slice, destination-PE slice. This is what makes both
   a flat device tree and a zero-copy particle exchange with *no packing
   kernel* possible.


## 2. Host / device split

| Phase | Today | After | Crosses PCIe |
|---|---|---|---|
| Force evaluation | device kernel over host-built list | device, walk + smem list | — |
| KDK integration | host loop, `DataManager.cpp:1202` | device kernel | — |
| Energy + bounding box | host accumulation, same loop | device partial + host cross-PE combine | up: 7 floats |
| NaN check | host loop, `findMinVByA:1244` | device partial + host combine | up: 1 flag |
| Universe box | host (Charm) | **host (Charm)** | down: 7 floats |
| SFC keys | host loop, `hashParticleCoordinates` | device kernel | — |
| Sort by key | host `quickSort` x2 | device CUB radix sort | — |
| Histogram counting | host walk of sorting tree | device kernel (binary search per bin) | up: ~32 B x bins, per round |
| Histogram *decision* | host, PE 0, `receiveHistogram:221` | **host, PE 0** | down: refine set, per round |
| Decomposition plan | host | **host**, O(`T`) index arithmetic | — |
| Particle exchange | host memcpy into `ParticleMsg` | device-to-device zero-copy slices | — |
| Post-exchange sort | host `quickSort` | device | — |
| Tree build | host, `new Node[2]` per split | device, level-synchronous | up: O(`P`·`D`) node descriptors |
| Moments (local subtrees) | host postorder walk | device bottom-up by level | — |
| Moment exchange (boundary) | **host** messages | **host** messages | down: O(`P`·`D`) moments |
| Remote data | host request/reply round trips | **deleted** — replaced by LET | — |
| LET construction | — | device walk + compaction | — |
| LET exchange | — | device-to-device zero-copy | — |
| Traversal | host walk per bucket | device, warp-lockstep | — |
| Load balancing | **host** | **host** | — |

### 2.1 What stays on the host, and why it should

- **Every Charm++ entry method, reduction, and broadcast.** The device never
  initiates communication.
- **The histogram refine decision** (`receiveHistogram`). A global sequential
  choice over O(bins) data on PE 0. Moving it buys nothing and the
  `ActiveBinInfo` bookkeeping is fiddly.
- **The decomposition plan**: mapping key ranges to tree piece indices to PEs.
  O(`T`) integer arithmetic, and it is exactly what the load balancer perturbs.
- **The boundary moment exchange.** O(`P`·`D`) nodes — the spine, not the tree.
- **The load balancer, entirely.** Tree pieces still pup only `iteration` and
  `lbState`.
- **`senseTreePieces` / `processSubmittedParticles` element-set bookkeeping**,
  and the `AtSync` / `AtSyncStart` / `AtSyncWait` state machine.

### 2.2 What data becomes device-resident across the whole iteration

```c
// DataManager-owned. Persists across the iteration. Never migrates.
struct DeviceParticles {
  float4* posm;    // x, y, z, mass      (exists today as dPos)
  float4* accel;   // ax, ay, az, phi    (exists today as dAccel)
  float4* vel;     // vx, vy, vz, _
  Key*    key;     // SFC key
  int     n;
};

// Flat, index-linked tree. Replaces Node<ForceData> on the device side.
struct DeviceNode {
  float4 cm_mass;              // moments.cm + moments.totalMass
  float  rsq;
  int    firstChild;           // index into node array; -1 for a leaf
  int    partStart, partCount; // contiguous range of DeviceParticles
  int    type;                 // NodeType
  int    ownerStart, ownerEnd;
  Key    key;
};
```

`savedEnergy` also stays permanently in device memory: it is a `DataManager`
member carried to the next iteration and never contributed to any Charm
reduction, so it never needs to come up.

### 2.3 What the host stops owning entirely

Deleted or reduced to a stub on the device path:

- `myParticles` (the `CkVec<Particle>`) — replaced by `DeviceParticles`.
  `loadParticles` still reads the file on the host; that is one H2D at startup.
- The interior of `Node<ForceData>`. The host keeps only the **boundary/remote
  spine**: the O(`P`·`D`) nodes whose owner range spans more than this PE,
  which is what the cross-PE moment exchange needs.
- `nodeTable` for anything but that spine.
- After Stage 1: `nodeRequestTable`, `particleRequestTable`, `Request`,
  `RequestMsg`, `NodeReplyMsg`, `ParticleReplyMsg` — the entire suspend-on-miss
  remote cache.
- After Stage 5: `ParticleMsg`, and the `CkVec<ParticleMsg*>` inside
  `TreePieceDescriptor`.

That is most of `DataManager.cpp`.


## 3. Reductions: the two-level structure and the sync-point budget

Charm's `contribute()` takes a **host** buffer. There is no device-resident
reducer. So every quantity reduced across PEs is a two-level operation:

1. **Device**: reduce N particles to a small fixed-width tuple.
2. **D2H** of that tuple.
3. **Host**: `contribute()` it, Charm combines across PEs.

Only step 1 moves to the device.

### 3.1 What the device level actually reduces

`kickDriftKick` (`DataManager.cpp:1202`) is one pass over N particles doing two
things at once, and both halves move together into a single fused kernel. The
map half is elementwise and writes N outputs — kick, drift, kick, then zero
`acceleration` and `potential`. The reduce half collapses to eight scalars over
four different operators:

| Quantity | Operator | Source |
|---|---|---|
| `energy` | sum | `p->mass * p->potential` |
| `savedEnergy` | sum | `p->mass * velocity.lengthSquared()` after the kick |
| `box.lesser_corner` | min x3 | `box.grow(p->position)` |
| `box.greater_corner` | max x3 | same |
| `haveNaN` | logical OR | `isnan(acceleration.length())`, from `findMinVByA:1244` |

Four distinct operators over one pass, so a hand-rolled fused block reduction
is the right shape — four separate CUB `DeviceReduce` calls would be four
passes over N to save nothing. `findMinVByA` is an entire extra O(N) host pass
today whose only product is that OR; it folds in for free.

Note this is a *scalar* reduction (N to a tuple). Only the moment pass in
Stage 6 is genuinely vector-valued — one output tuple per node, each reducing
over that node's contiguous particle range.

Also note the histogram counting is **not** a reduction at all, despite
producing a vector. Because keys are sorted, each bin is two binary searches
into the key array: count is `hi - lo`, `smallestKey` is `key[lo]`,
`largestKey` is `key[hi-1]`. O(bins · log N), one thread per bin.

At the Charm level, `BoundingBoxGrowReductionType` is a 7-float scalar combine,
but `NodeDescriptorReductionType` (`DataManager.cpp:216`) is a true elementwise
vector reduction over a bins-length array. Both stay on the host; only their
*input* is produced on the device.

### 3.2 Sync-point budget

The bytes are irrelevant. What costs is that each host reduction is a hard
device-to-host synchronization point. Per iteration:

| Sync point | Count | Payload |
|---|---|---|
| `DtReductionType` (NaN) — `finishIterationTail:899` | 1 | 1 flag |
| `BoundingBoxGrowReductionType` — `advance:973` | 1 | 7 floats |
| `NodeDescriptorReductionType` — `sendHistogram:216` | **15** | ~32 B x bins |
| Boundary moment exchange | 1 | O(`P`·`D`) |

The first two are serialized rather than fused, because the NaN check gates
whether `kickDriftKick` runs at all and KDK zeroes `acceleration`. KDK could be
run speculatively with the flag checked afterward, but the NaN path calls
`markNaNBuckets()` / `printTree()`, which need the pre-KDK accelerations —
not worth an extra N-sized buffer for a path that terminates the run.

**The 15 are the ones that matter**, and they run every iteration:
`treePiecesReady` -> `startNextIteration` re-decomposes each step. Each round
is kernel -> D2H -> cross-PE reduction -> PE 0 decides -> broadcast -> H2D ->
kernel.

### 3.3 The reason Stage 4 is worth doing

Moving the counting to the device buys almost nothing on its own. It is the
round-trip collapse it *enables* that pays.

`ActiveBinInfo::processRefine` refines exactly one level per round, and that is
correct on the host: `node->refine()` partitions the particle array, O(N) per
level, so counting more bins per round costs real work. On the device the
particles are already key-sorted, so a bin's count is two binary searches and
there is no partition at all. Counting 2^k bins per round is nearly free.

Refining k=3 levels per round takes 15 rounds to ~5, at the cost of a few
thousand binary searches. That is the actual deliverable of Stage 4.

### 3.4 Two smaller points

- Use `hapiAddCallback` on the D2H rather than `cudaStreamSynchronize`. It does
  not shorten the latency but it lets the PE run other tree pieces' work across
  it, and with ~26 tree pieces per PE there is other work. Both idioms already
  exist here — `GpuTraversalBatch::flush()` currently blocks on
  `cudaEventSynchronize(h2dDone)`.
- The only way to keep the cross-PE combine device-side is NCCL, called
  directly rather than through Charm. Structurally viable — a communicator
  needs static ranks and `DataManager` is a non-migrating group — but for 7
  floats and a few KB of bin counts it buys nothing against a heavy dependency.
  **Not recommended.**


## 4. Stages

### Stage 0 — instrumentation and oracle (gates everything)

Per-phase timers on the `DataManager`; counters for remote requests issued per
iteration and for time spent with all traversals blocked on replies; enable the
existing `BARNES_WALK_REPORT`. An acceleration-dump comparison against the
`GPU=0` build, which becomes the correctness oracle for every stage after this
one.

The one thing this decides: **whether remote-fetch latency dominates.** If it
does, Stage 1 is worth more than Stages 2-7 combined. Right now roughly 85% of
the ~500 ms iteration is unaccounted for; from the logs I can rule out tree
build (~800 nodes/PE), the sorts (12.5K particles/PE), the histogram (15
reductions), interaction-list PCIe (~35 MB, ~2 ms) and the force kernel (270M
interactions, tens of ms) as accounting for more than ~15% of it. The leading
suspicion is the remote node/particle request round trips — latency-bound, with
~26 tree pieces per PE each running its own suspending remote traversal — but
it is a suspicion, not a measurement.

Also raise the problem size here. 12.5K particles per GPU is not a regime where
device residency means anything; the target is 2M-10M, which `plummer` already
generates (`big.bin` is 500K).

### Stage 1 — push-based LET, on the existing host tree

**This is a protocol change, not a device change**, and that is what makes it
the right thing to do first: it needs none of the rewrite below and it can be
validated against the oracle immediately.

Today a bucket's walk hits a `Remote` node, suspends, sends a `RequestMsg`, and
resumes when the reply lands — thousands of latency-bound round trips per
iteration.

After: each PE exchanges the top few levels of its tree, then walks its own
tree against each remote PE's coarse cells applying the same opening criterion
the remote walk would apply. Accepting a cell means shipping its multipole;
descending to a leaf means shipping its particles. The result is sufficient by
construction, so the remote walk can never miss. One bulk send per destination,
no round trips.

Deletes the entire request/reply machinery in section 2.3.

Risk: medium. The LET walk is O(`P` x remote cells) per destination, so cell
granularity is a real tightness-vs-cost tradeoff. Walking against individual
remote *buckets* gives the tightest LET but costs the most; walking against
their top-level cells is the standard compromise (Bonsai). At `P`=4 this is
trivially cheap; the granularity question only bites at scale.

Remote particles arrive as `float4` (pos+mass), which is exactly
`ExternalParticle` — no new fields needed.

### Track A — kernel quality (independent, lands any time)

No dependency on anything else. Each changes both the CPU and GPU paths, so the
oracle stays valid.

- **A1 `rsqrtf`.** The inner loop in `barnes.cu` does one `sqrtf` and two
  divides per interaction. Replace with
  `rinv = rsqrtf(drsq); phii = m*rinv; mor3 = phii*rinv*rinv`. Hours of work.
- **A2 bounding-box opening criterion.** `openCriterionBucket` in `gravity.h`
  uses center-of-mass to center-of-mass distance. The bucket's `data.box` is
  already computed and gives a tighter, more accurate min-distance test — fewer
  interactions at the same error.
- **A3 quadrupole moments.** `MultipoleMoments.h` is monopole-only
  (`rsq`, `totalMass`, `cm`). Adding quadrupoles typically permits theta ~
  0.7-0.8 instead of the current `DEFAULT_THETA 0.5` at equal force error,
  which cuts interaction counts substantially. Touches `MultipoleMoments.h`,
  `getMomentsFromChildren` / `getMomentsFromParticles`, `grav()`, and the
  kernel — and nothing structural. **This is the largest accuracy-per-flop
  lever available.**

### Stage 2 — persistent device particle store

Introduce `DeviceParticles` (section 2.2) on the `DataManager`. Move KDK, the
fused energy/box/NaN reduction (section 3.1), and SFC key generation onto it.

Deletes both O(N) host conversion loops in `GpuParticleStore::upload()` /
`apply()` and both O(N) transfers. Low risk.

### Stage 3 — device sort

CUB `DeviceRadixSort::SortPairs` on the keys against an index, then a gather of
position and velocity through it. Acceleration and potential need no
permutation: the integrator has just zeroed them, and they are the only other
fields a `Particle` carries.

This moves **one** of the two `quickSort()` calls, not both -- see section 8.1.
The one in `decompose` runs on keys already on the device, and the particles
were going back to the host anyway, so sorting first means the readback
delivers them in order and the host sort disappears outright. The one in
`processSubmittedParticles` runs on particles that have just arrived from other
PEs into host memory, with the host tree build reading them immediately after;
sorting those on the device would be an H2D and a D2H to save an O(N log N)
host pass. It stays until Stage 6.

Radix sort is stable, so ties keep upload order. The host `quickSort` made no
such promise, which is the one way the two builds can now disagree about the
decomposition -- and only for particles sharing a box of the
21-bits-per-dimension grid. Low risk.

### Stage 4 — fewer histogram rounds (**no device work**)

The device half of this stage was dropped. See section 8.3: the premise in
section 3.3 was wrong, and moving the counting to the GPU buys nothing.

What survives is the half that was always the real deliverable -- refining more
than one level per round, so the decomposition converges in fewer round trips.
`-decomplevels=k` (default 1, which is exactly the old behaviour) lets a bin
that is far over target be taken down up to k levels in a single round, with
the depth chosen from its particle count.

Splitting a clustered bin k levels deep can leave most of its children empty,
and an empty child still costs a tree piece, so the level count is backed off
until the tree-piece budget covers it rather than tripping the existing
`CkAbort("Need more tree pieces!")`. One level always fits if anything does,
and the next round picks the bin up again.

### Stage 5 — zero-copy particle exchange

After the sort, each destination's particles are already contiguous, so the
send is a `CkDeviceBuffer` over a slice with **no packing kernel**. A small
counts exchange goes first so receivers can size and post device buffers at
computed offsets; then the concatenation of sorted runs is re-sorted on device.

This is where the branch's device transport work pays off, and where migration
interacts, since the balancer changes which PE a key range lands on.
Medium risk.

### Stage 6 — flat device tree and moments

Build `DeviceNode[]` (section 2.2) level-synchronously: at each level every
active node binary-searches its split in the sorted key array, refining while

```c
(node->getOwnerEnd() > node->getOwnerStart()) ||
(node->getNumParticles() > (Real)globalParams.ppb*BUCKET_TOLERANCE)
```

— the existing predicate from `buildTree`, unchanged. About 10-15 levels, so
~15 kernels plus a compaction each.

The level structure is what makes the bottom-up moment pass trivial: at the
leaf level, a segmented reduction over particle ranges; at each level above, a
fixed `BRANCH_FACTOR`-way combine from `firstChild`.

Boundary nodes need remote contributions. The device computes moments for all
fully-local subtrees, the roots of those subtrees are D2H'd (O(`P`·`D`),
tiny), the host runs the existing `passMomentsUpward` / `receiveMoments`
exchange, and the completed boundary moments are scattered back H2D into the
device node array.

`startTraversal` splits `myBuckets` among tree pieces by `largestKey` binary
search; on the device the bucket array is compacted and the split points are
the same binary searches, read back as O(`T`) ints.

High risk. This is the structural piece.

### Stage 7 — device traversal

Warp-lockstep walk: one stack per warp in shared memory, `__any_sync` deciding
whether to open a cell, so the warp never diverges. The walk fills a
few-thousand-entry interaction list **in shared memory**, which is then
evaluated densely.

The list moves to shared memory — it does not stop existing. Fusing the walk
into the force loop is wrong: the walk is divergent pointer-chasing and the
force loop is dense SIMD, so fusing makes the force run at walk efficiency.

The LET from Stage 1 guarantees no misses, so there is no suspension machinery
on the device at all.

**Launch from `TreePiece` entry methods, gridded over that tree piece's bucket
range.** See section 5.


## 5. Constraints that must hold at every stage

1. **CUPTI attribution.** GPU time is attributed per tree piece only because
   every launch happens inside a `TreePiece` entry method — this is what
   `TreePiece::traversalDone()` bouncing through
   `thisProxy[thisIndex].finishGpuWork()` is for. A device-resident traversal
   naturally wants one launch per PE, which would charge everything to the
   group and make the entire `rate-aware-gpu-lb` branch unmeasurable. Keep one
   launch per tree piece, gridded over that tree piece's buckets. Non-negotiable.

2. **Migration stays free.** All device state lives in the `DataManager` group,
   which never migrates. `TreePiece::pup` stays `p | iteration; p | lbState;`.

3. **`GPU=0` remains a pure correctness oracle.** The device path gets its own
   `DataManager` methods rather than more `#ifdef GPU_GRAVITY` interleaving in
   the existing ones.

4. **Track A changes both paths**, so the oracle stays meaningful across them.


## 6. Dependencies

```
Stage 0 --+--> Stage 1 (LET, host tree) --------------------+
          |                                                 +--> Stage 7
          +--> Track A (A1, A2, A3) -- independent          |
          |                                                 |
          +--> Stage 2 -> Stage 3 -> Stage 4 -> Stage 5 -> Stage 6
```

Stage 1 and Track A proceed in parallel with 2-6; neither blocks the other.


## 7. Scope: what this plan deliberately does not chase

This codebase's contribution is over-decomposed migratable tree pieces with
load balancing. Bonsai and PKDGRAV3 have no answer to dynamic imbalance during
a step; that is the thing worth defending.

So: adopt the SOTA where it is cheap and self-contained (Track A, warp-lockstep
traversal, shared-memory interaction lists) plus push-based LET, because that
one is on the critical path for this branch's communication story. Explicitly
do **not** chase FMM (PKDGRAV3, ExaFMM), dual-tree mutual interactions
(falcON), or PM/tree hybrids (HACC). Those are different algorithms, not
optimizations of this one.


## 8. Status

Landed and building (both `GPU=1` and the `GPU=0` oracle, which is kept as
`barnes-cpu`). **Nothing below has been run yet** -- testing is deferred to the
end, so every claim here is a claim about the code, not a measurement.

| Item | State | Files |
|---|---|---|
| Stage 0 instrumentation | landed | `Profile.h` (new), `DataManager.{h,cpp}`, `TreePiece.cpp`, `Request.h` |
| A1 `rsqrtf` | landed | `barnes.cu` |
| A2 bounding-box criterion | landed | `gravity.h` |
| A3 quadrupole moments | not started | -- |
| Stage 1 push-based LET | not started | -- |
| Stage 2 device particle store | landed | `barnes_cuda.h`, `barnes.cu`, `GpuBatch.{h,cpp}`, `DataManager.{h,cpp}`, `barnes.ci` |
| Stage 3 device sort | landed, **one of the two sorts** | `barnes_cuda.h`, `barnes.cu`, `GpuBatch.{h,cpp}`, `DataManager.cpp`, `Makefile` |
| A3 quadrupole moments | landed, default on | `MultipoleMoments.h`, `Node.h`, `gravity.h`, `Worker.cpp`, `barnes_cuda.h`, `barnes.cu`, `GpuBatch.h` |
| Stage 4 fewer histogram rounds | landed, **host only**, default off, **measured a net loss** | `ActiveBinInfo.h`, `DataManager.cpp`, `Parameters.h`, `defaults.h`, `Main.cpp` |
| NUMA-aware launch wrapper | landed | `numa_wrap.sh` (new) |
| Loader 64-bit offsets + block reads | landed | `DataManager.cpp` |
| Acceleration dump + comparison harness | landed | `DataManager.{h,cpp}`, `compare_accel.py` (new) |
| Stage 5a PE-aggregated exchange | landed, **validated** | `DataManager.{h,cpp}`, `Messages.h`, `barnes.ci`, `ActiveBinInfo.h`, `Worker.cpp`, `TreePiece.{h,cpp}` |
| Stage 6 flat device tree | landed, **validated** | `barnes_cuda.h`, `barnes.cu`, `GpuBatch.{h,cpp}`, `DataManager.{h,cpp}` |
| Stage 7 device traversal (local half) | landed, **validated**, `-devwalk=1` | `barnes_cuda.h`, `barnes.cu`, `TreePiece.{h,cpp}`, `Parameters.h`, `Main.cpp` |
| Stage 5b device-buffer exchange | **not started** | -- |
| Stage 1 push-based LET | **not started** | -- |

Build: `./build.sh barnes` (new). Four things the stock Makefile got wrong for
this checkout: `CHARM_PATH`/`STRUCTURES_PATH` point at an in-tree layout that
does not exist here; `liblci.so` has an unresolved `DT_NEEDED` on `liblct.so`
beside it, which only `LD_LIBRARY_PATH` at link time resolves; `NVCC_ARCH`
listed `compute_70`, which CUDA 13.2 no longer accepts -- Volta is gone and
`compute_75` is the oldest; and the kernel translation unit had to move from
`-std=c++14` to `-std=c++17`, which CUB requires. That last one is contained:
`barnes.cu` sees no Charm++ header, so its dialect does not have to match the
rest of the application.

Measurement: `./run_stage0.sh` inside an A40 allocation. It sets
`BARNES_PHASE_REPORT` and `BARNES_WALK_REPORT` and runs 50K and 500K, no LB.

Correctness: `BARNES_ACCEL_DUMP=<prefix>` writes one file per PE on the last
iteration, `key ax ay az phi`, keyed by SFC key so the two builds need not
agree about which PE holds what. `compare_accel.py <ref-prefix> <prefix>...`
folds the per-PE files and reports the relative acceleration error against the
reference, normalised by the mean reference magnitude rather than per particle
-- particles near the centre of mass have near-zero acceleration and a
per-particle relative error there is all noise.

The reference to use is the code itself at a very small opening angle:
`-theta=0.05` opens almost every cell, so the walk degenerates to direct
summation and the result is exact to rounding. Run everything at `-killat=1`
so the dump lands on iteration 0, where every run still has identical
positions; at any later iteration the runs have drifted apart and the
comparison measures trajectory divergence rather than force error.

Three binaries are kept: `barnes` (GPU), `barnes-cpu` (`GPU=0`), and
`barnes-cpu-mono` (`GPU=0 -DMONOPOLE_ONLY`), which is what A3 is measured
against.

### 8.1 Two corrections these changes forced

**Stage 2 does not pay for itself; Stage 5 is what pays.** The plan above
described Stage 2 as deleting "both O(N) host conversion loops and both O(N)
transfers." That was wrong. The host still owns the sort and the all-to-all
exchange, so the particles have to come back every iteration regardless of
where the integrator runs -- `hashKeys()` ends with a D2H of position,
velocity and key. What Stage 2 actually removes is three O(N) *host passes*
(`kickDriftKick`, `findMinVByA`, `hashParticleCoordinates`) and the accel
readback; what it adds is 32 B/particle of D2H where there was 16.

So Stages 2-5 are one increment as far as performance goes, even though they
land and verify separately. The cut point where the host stops needing O(N)
data is Stage 5, not Stage 2. Stage 2 remains the right thing to build first
because 3, 4 and 6 all need the device arrays it introduces -- it is
scaffolding, and should be reported as scaffolding rather than as a win.

Stage 3 is the first piece with a saving that stands on its own, and it is a
small one: an O(N log N) host sort per iteration, gone. The same reasoning caps
it -- only the sort whose input was already on the device can move, which is
one of the two.

**A1 costs the exact oracle, and A2 costs time at fixed theta.** `rsqrtf` is
accurate to about 2 ulp, so the CPU build is no longer bit-comparable against
the GPU one on the force path; `-DGRAV_EXACT_RSQRT` restores the
sqrt-and-divide form when a tight comparison is wanted. And the bounding-box
opening criterion is strictly more conservative than cm-to-cm, so at
`theta = 0.5` it opens *more* cells and costs *more*. The deliverable is that
the error/cost curve moves; theta is then the knob, and `-DOPEN_CRITERION_CM`
keeps the old test for the comparison. Neither of these is a regression, but
neither is a free win either, and the earlier text implied both were.

### 8.3 Stage 4's premise was wrong

Section 3.3 argued that the histogram refines one level per round because
`node->refine()` partitions the particle array, O(N) per level, and that the
device could refine several levels at once because counting there is only a
binary search.

The second half is true and the first half is not. `refine()` calls
`findSplitters` (`util.cpp:53`), which is `BRANCH_FACTOR-1` binary searches
into an already-sorted array -- the same O(bins · log N) the device would do.
The host was never paying O(N) per level.

So there is nothing for the GPU to take over here, and the multi-level refine
that was the stage's actual deliverable needs no device work at all: it is a
change to `ActiveBinInfo` and to PE 0's decision in `receiveHistogram`, both
host code, and it is what landed.

The cost of the histogram phase was never the counting. It is fifteen
serialized round trips per iteration, each a kernel-to-host hop, a cross-PE
reduction, a decision on PE 0 and a broadcast back. Cutting the number of
rounds is the only thing that touches that, and it is worth about three times
more than anything the counting kernel could have saved.

The general lesson, and it now applies to three of the seven stages: *check
what the host code actually costs before assuming the device is faster.*
Stages 2 and 3 were overclaimed in the same direction.

### 8.2 Three splits Stage 2 introduced

Device work between two Charm reductions cannot fall through, so three entry
methods became two:

- `finishIteration` -> `forcesReady`: was the accel D2H, is now the NaN
  reduction. Same shape, less data.
- `advance` -> `advanceTail` (new): the integrator and its box/energy
  reduction sit between them.
- `decompose` -> `decomposeTail` (new): key generation and the particle
  readback sit between them.

The first decomposition still runs the host path, because nothing has been
uploaded at that point -- the upload happens in `processSubmittedParticles`,
further down the same pipeline.


## 8.4 Stage 0 ran. Placement was the whole story.

Measured on the A40, 50K particles, 4 ranks x 4 PEs, 30 iterations, no LB.

**Everything before this was measured wrong.** `+setcpuaffinity` with
`srun --cpu-bind=cores` and no explicit `+pemap` left the ranks bound to cores
with no relation to the NUMA node their GPU hangs off -- and on a gpuA40x4 node
the GPU order is *reversed* against NUMA (GPU0 on NUMA 3, GPU3 on NUMA 0), so
the obvious pairing is the worst one. Iteration time was bimodal, 0.64 s or
3.0+ s for the identical command. With `numa_wrap.sh`: 0.035 s, stable to
+/-2% over four repeats. Interaction counts are byte-identical across both
placements, so it is the same work -- 18x against the good mode, 60x against
the bad. Remote request latency: 22000 us -> 246 us.

| phase | bad placement | **correct placement** |
|---|---|---|
| traversal | 21% | **83.4%** (24.4 ms/iter) |
| histogram | 70% | **7.0%** (2.0 ms/iter) |
| moments | 4% | 3.0% |
| sorts + keygen + buildtree + h2d | ~1% | ~4.1% |
| iteration | 0.656 s | **0.035 s** |

So the conclusion I drew from the first Stage 0 run -- "the histogram is 70% of
the iteration, the device-residency stages target under 0.4%, Stage 4 is the
whole game" -- was an artifact. The histogram's global reductions are
latency-bound, and cross-NUMA contention inflated every round trip about 90x,
which made a 2 ms phase look like a 400 ms one.

**What the corrected profile says:**

- **The traversal is 83%**, and within it the walk is two thirds and blocked
  time one third (33% of the span, 28% of the iteration). Stage 7 is the right
  target after all, and Stage 1 matters for the blocked third.
- **Stage 4 is a net loss.** `-decomplevels=3` halves the histogram
  (2.03 -> 1.08 ms) but takes the traversal from 24.4 to 37.4 ms, because 125
  tree pieces instead of 106 means more remote work and more per-piece
  overhead: 0.0493 s/iter against 0.0351. It should stay at the default of 1.
  Multi-level refine only becomes interesting if the histogram ever dominates
  again -- which is a many-node question, where reduction latency grows.
- **Stages 2, 3, 5 and 6 together target about 4%**, not the 0.4% the bad
  measurement suggested, but still not where the time is.

**Nothing in this file should be trusted against a run that did not pin
placement.** See the `numa-aware-pemap` memory.

## 8.5 Results

Everything below was measured on one A40 node, 4 ranks x 4 PEs, NUMA-pinned
(section 8.4 -- unpinned numbers are meaningless).

**Correctness**, relative acceleration error against the host path:

| check | mean | notes |
|---|---|---|
| Stages 2/3/A3: GPU vs CPU build, 5 iterations | 1.46e-5 | |
| A3 quadrupole vs monopole at theta 0.5 | 4x more accurate | quad at 0.7 matches mono at 0.5 |
| Stage 5a: GPU vs CPU, 5 iterations, 50K | 9.7e-5 | decomposition rewrite |
| Stage 6: device tree vs host tree | exact structure | mass 1e-6, cm 2e-6, rsq 1e-5 |
| Stage 7: device walk vs host walk, 1 PE | 2.4e-6 | |
| **Stage 7: device walk vs host walk, 4 PEs** | **1.4e-6** | max 1.2e-5 |

**Runtimes**, seconds per iteration:

| config | host walk | device walk | |
|---|---|---|---|
| 50K, no LB | 0.0361 | **0.0262** | 1.38x |
| 500K, no LB | 0.5064 | **0.2776** | 1.82x |
| 500K, DiffusionLB async | 0.3423 | **0.2124** | 1.62x |

Best against the no-LB host-walk baseline: **2.38x**.

The traversal phase went from 24.4 to 15.8 ms/iter at 50K, and the host local
walk from 0.307 s to 0.0001 s -- it is gone. The remote walk is untouched.

**What that exposes.** Blocked time inside the traversal rose from 33% to 60%
of the span: with the local walk free, what remains is waiting on remote data.
Stage 1 is now the largest single item, and it is a measurement rather than the
suspicion it was in section 4.

Balancers, 500K (section 8.4 measured these with the host walk):
`DiffusionLB` with `+LBDiffusionCommOn +LBDiffusionGpuDim` and `-qd=0` beats
no-LB by 16%; `GreedyRefineCentralGPULB` loses 10%. Without LB, two of four PEs
do **zero** walk work -- a placement failure in the array map, not gradual
imbalance, and still open.

## 8.6 Bugs the validation caught

Every one of these was invisible to the compiler and would have been
attributed to the wrong layer if found later.

1. **`dKey` uninitialised on iteration 0.** `hashKeys()` fills it, but the
   first `decompose` runs before the first upload, so the first device tree
   split on garbage. The node *keys* are computed by the build itself, so the
   symptom was "keys match, shapes do not" rather than a crash. Fixed by
   uploading keys with positions and velocities.
2. **`levelStart` off by one.** `treeLevelMarkKernel` wrote `levelStart[level+1]`
   with the count *after* that level's children were allocated, so level 0's
   range was [0,3) -- the root and its own children, in one launch. The root
   read children as they were written and came out zero-mass. Missed by the
   first tree check because on 4 PEs the root is Boundary and was skipped.
3. **Boundary moments never reached the device.** A host Boundary node carries
   the complete moments from the cross-PE exchange; the device build only sees
   local particles. Accepting such a cell as a multipole is exactly where
   local-only is wrong: median error was already 1e-6 with a 1% tail at 4.6e-3.
   `patchDeviceBoundaryMoments` scatters them after the host pass completes.
4. **Two broken self-checks.** The tree check first read `nodeTable`, which
   holds only the nodes the remote-request path looks up -- it reported "host
   nodes 1" and compared nothing. Then it ran before `makeMoments()`, so every
   host moment was zero; only a "compared N nodes" counter made that visible.

## 8.7 Second session: placement, and what is left for scale

500K on 4 ranks, one A40 node, best config at each step:

| step | s/iter |
|---|---|
| start (unpinned) | ~0.545 |
| NUMA pemap (8.4) | 0.506 |
| Stage 7 device walk | 0.278 |
| `-p` sized to the used count, block map | 0.134 |
| `-chunkdepth=9` | 0.129 |
| shared-memory staging in the walk kernel | 0.115 |
| sender-count reduction (P^2 empty sends removed) | 0.106 |
| **best measured** | **0.104** |

**5.2x overall**, and most of it is placement and memory traffic rather than
anything algorithmic.

**Placement is a budget problem.** `-p` is a budget, not a count: 652 of 2048
used in a 500K run. Charm's map blocks the whole index range, so the used
prefix lands on the first PEs and the rest idle -- the `10.5/3.0/0.0/0.0` walk
profile. Sizing `-p` near the used count fixes it outright (0.134 against
0.506) and a note now prints when the ratio is above 2x. `-mapcyclic=1`
selects a node-aware block cyclic map for when the budget cannot be sized:
chunks dealt round robin across *nodes*, then across a node's PEs, so
neighbouring key ranges stay on one node. It is **opt in** -- with `-p` sized
right the stock map is better (0.111 against 0.136), because the cyclic map
trades locality for balance and there is no imbalance left to buy.

**Load balancing now loses.** With placement fixed there is nothing to
recover: DiffusionLB costs 7% at 500K. The earlier "+16% for DiffusionLB" was
it rescuing the placement failure, not correcting real imbalance. That result
is superseded.

### What scales badly, in multi-node terms

Every number above is 4 ranks on one node, where a remote hop is intra-node.
The phases that grow with rank count are not the ones that dominate here:

1. **Remote fetching -- Stage 1 (LET).** O(misses) round trips, each becoming
   inter-node, over a far field spanning more owners. Must be designed against
   O(P^2) messages from the start -- coarse cells for distant ranks -- not
   retrofitted.
2. **The moment exchange -- DONE, see below.** Already **14.4%** at 4
   ranks. `passMomentsUpward` is a serial dependency chain up the tree: the
   tree is not ready until the root's moments complete, a message round trip
   per level over an O(P*D) spine. Replace it with **one all-reduce over the
   boundary spine**: the spine is determined by `keyRanges` alone, so every PE
   can enumerate it identically, contribute its local partials in additive form
   (mass, sum m*x, the six second moments about a fixed origin, box min/max)
   and derive cm, rsq and the quadrupole afterwards by the parallel-axis
   identity. One collective of a few hundred nodes x 16 floats instead of a
   depth-D chain. Smaller than LET and the phase is already visible.
3. **The histogram.** 15 serialized collectives per iteration, every iteration.
   1.4% here; each costs log(P) more at scale. `-decomplevels>1` is already
   built and was a net loss single-node because the extra tree pieces cost more
   than the rounds saved -- that trade **flips** when rounds get expensive.
   Incremental decomposition (reuse the previous splitters, one verification
   round, refine only drifted bins) turns 15 collectives into ~1 and is still
   unimplemented.
4. **Rank-independent, already done:** rsqrtf, quadrupoles, the device tree and
   walk, shared-memory staging. These carry over unchanged.

## 8.8 The moment exchange is now one reduction

`passMomentsUpward` and the per-node `requestMoments` traffic are gone. In
their place: every PE enumerates the **ownership frontier** -- the nodes at
which the owner range narrows to a single tree piece -- by walking its own tree
and stopping there. The refinement that produced those ranges came from
keyRanges, which is global, so every PE gets the same list in the same order.
A frontier node belongs to exactly one tree piece, so exactly one PE can
compute its moments and everyone else contributes zeros: a plain sum is the
answer, and nothing has to be reduced in additive form.

Seventeen floats per frontier node: mass, cm, rsq, the five quadrupole
components, the bounding box, and the owner's node type. The box travels
because `getMomentsFromChildren` derives rsq from it. The type travels because
`copyMomentsToNode` used to set it, and it is what tells the receiver's walk
whether to ask for particles or for a subtree.

Measured: the moments phase went from **17.4 to 6.6 ms/iter**, 2.6x. Total
runtime at 500K is unchanged (0.1041 against 0.1040) -- on one node the old
chain's latency was overlapping with other work, so removing it only moves
where the wait happens. The win is structural: a chain of round trips whose
length is the tree depth became one collective, and that is a rank-count
problem, not a single-node one.

Four bugs on the way, all caught by the acceleration harness:
double counting (`fillBoundaryMoments` recomputed nodes `MomentsWorker` had
already done -- root mass 2.19 instead of 1); the missing box (boundary nodes
got a poisoned opening radius); the missing type (every remote node stayed
plain `Remote`, so the walk asked for subtrees where it wanted particles); and
a dropped `getOwnershipFromChildren`, which sent requests to the wrong tree
pieces and cost 11% force error.

## 9. Where this stopped, and why

Not started: Stage 1 (LET), the moment spine all-reduce, incremental
decomposition, and Stage 5b -- which stays blocked behind Stage 1, because the
host only stops needing particles once the remote walk is device-resident.

Everything above landed without a single run: the GPU partition was never
touched, so the only signal any of it has is that it compiles. For the stages
so far that was a defensible trade -- each is a local substitution with a
clearly bounded blast radius, and the CPU build is kept beside it as an oracle
for when there is an allocation.

The four that remain are not like that:

- **Stage 5** moves the particle exchange onto `CkDeviceBuffer`. The current
  path is two hops -- DataManager to tree piece to that tree piece's local
  DataManager -- because the tree piece is the migratable unit and the
  DataManager does not know which PE one is on. A device-buffer version has to
  keep that routing, add a counts phase so receivers can size and post, and
  interact correctly with a balancer that is moving those tree pieces at the
  same time. It also delivers particles to the *destination's device*, which
  the host tree build cannot read -- so on its own it makes things worse, and
  only pays once Stage 6 lands.
- **Stage 6** replaces the pointer tree with a level-synchronous device build,
  the postorder moment pass with a bottom-up one, and has to mirror the
  boundary spine back to the host for the cross-PE moment exchange. It is the
  structural piece, and it rewires `buildTree`, `makeMoments`,
  `passMomentsUpward` and `startTraversal` at once.
- **Stage 7** needs both of those plus Stage 1.
- **Stage 1** is host-only and self-contained, but it is a protocol change that
  deletes the entire suspend-on-miss remote cache and replaces it with a walk
  whose sufficiency argument is the thing that keeps the traversal correct.

Any one of these can be made to compile in an afternoon and be silently wrong
in a way that only shows up as a wrong force on one bucket on one PE every few
hundred iterations. Compiling is a very weak signal for a distributed tree
code, and it is the only signal available right now. Writing all four blind
would produce a large amount of plausible code with no way to tell which parts
work, which is worse than not writing it.

The gate is Stage 0. It is a couple of runs on an A40, `./run_stage0.sh` is
ready, and it answers both open questions at once: whether remote-fetch latency
dominates -- which decides whether Stage 1 or Stage 5 is next -- and whether
everything above is even correct.

## 9.1 The CPU oracle cannot run on a login node

This was assumed throughout and it is false. `barnes-cpu` aborts at startup on
a login node, twice over:

- The runtime calls `cuInit(0)` unconditionally from LCI's accelerator layer
  (`lci/src/accelerator/accelerator_cuda.cpp:169`), whatever the application
  was built with. `GPU=0` does not avoid it.
- Substituting the CPU-only LCI build with `LD_PRELOAD` gets past that and then
  fails in the OFI backend (`backend_ofi.cpp:261`, `cxil_map: write error`) --
  the Slingshot NIC is not available to a login node.

So the oracle needs an allocation exactly like the GPU build does. It can be a
small one -- a few thousand particles for a single iteration is seconds of work
-- but "keep a CPU build you can check anywhere" was not a real fallback, and
nothing in this branch has been validated by anything except the compiler.

## 10. Open items

- Nothing has been run. Stage 0's numbers are what decide whether Stage 1 or
  Stage 3 comes next, and until then that ordering is a guess.
- `asynclb.sbatch` targeted `gpuA100x4-interactive`; switched to
  `gpuA40x4-interactive`.
- ~4.8 GB of core dumps sit in this directory and the repo root
  (`core.2.1.251.*`, `core.HeapHelper.*`), still there. `run_stage0.sh` sets
  `ulimit -c 0`. Note `make clean` would delete them, since its rule includes
  `rm -f core*`.
- A3 widened `GpuSource` from 16 bytes to 48, tripling the interaction-list
  PCIe traffic. It should pay for itself by allowing a larger theta, but that
  trade is unmeasured; `-DMONOPOLE_ONLY` and `barnes-cpu-mono` exist to measure
  it.
- The `-decomplevels` default is 1, which is exactly the old behaviour, so the
  multi-level path has never executed. It needs a run at k=3 or 4 before it
  means anything.
- Everything in section 9's list of scaling blockers except the loader is still
  open: the build is single-node (`multicore`), and the inter-node
  deferred-receive slot leak noted in earlier work on this branch was only ever
  fixed for the same-node case.
