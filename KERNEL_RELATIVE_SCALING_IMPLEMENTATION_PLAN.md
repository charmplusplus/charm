# Kernel-Relative GPU Scaling: Implementation Plan

## Status and scope

Implementation status (2026-08-23): Phase 0 is complete. Phase 1's runtime
changes are implemented and compile in the CUDA build: full-object CUPTI
tokens, separate object/work-tag correlations, deterministic kernel and launch
identities, RAII work tags, device descriptors, and debug counters. The pure
identity/metadata tests pass without CUDA hardware.

Post-review fixes applied to Phase 1: the peak-rate prior now uses a
compute-capability lane-count table rather than SM count times clock, and
degrades in-units when clock or capability is unknown (design decision 3); the
process-wide token lock is no longer taken per entry method or per kernel
record.

Phase 2 is implemented: epoch wire types with a total-preserving component cap,
per-record demand attribution and per-`(object, kernel)` aggregation, transport
through `LDObjData`/`ProcStats`/`CLBStatsMsg`, `useMem` accounting, and
`LB_FORMAT_VERSION` 4. The Phase 1 debt above was paid as part of it: the
immediate-dispatch fast path is back and `LBKernelRecord` is down from 96 to 48
bytes, and a kernel whose class is known to be tagged but which arrives untagged
is excluded from the model rather than misfiled.

Phase 3 is implemented in shadow mode: `CentralLB` owns the model, registers
device types, selects and rebases the reference, learns from each epoch's
summaries after statistics assembly and before `Strategy`, prices every object
against every available type, and scores the previous epoch's claims into a
`GpuPredictionAccuracy` split by prior-only/mixed/calibrated. Nothing feeds
placement. `CentralLB::predictGpuCost` exists for Phase 4 to consume and falls
back to the measured scalar whenever the model cannot price a pair.

First hardware run (2026-08-25, RTX 3050 Laptop, CC 8.6, 16 SMs, CUDA 12.8),
`jacobi2d-imbalance` on 2 PEs sharing 1 GPU under `GreedyCentralLB
+LBGpuScaling +LBDebug 2`. What it confirmed:

- the rate prior takes the architecture-table path:
  `cc=8.6 sms=16 clock_khz=1057000 rate=2164736000 source=architecture-table`,
  which is exactly `16 * 128 lanes * 1057000`;
- attribution is complete: `kernels=320 attributed=320 unattributed=0
  unresolved_tokens=0 hash_collisions=0`;
- the Phase 2 reconciliation invariant holds. The debug check compares
  `gpuTime` against `components + residual` for every object every epoch and
  printed nothing across every run;
- the immediate-dispatch fast path works: `deferred=0` for an untagged
  application, and for a tagged one `deferred=256` on the first drain (while
  kernel classes are still unclassified) then `0` thereafter;
- the `CUSTOM0` work-tag path works end to end: `work_tags=128` parsed and
  consumed, distinct buckets rose from 10 to 16 once `load_iters` was tagged,
  `lost_work_tags=0`;
- shadow scoring runs and reports, at roughly 3% median and 9% mean APE.

What it did **not** show, and cannot: this machine has one GPU of one type, so
source type always equals destination type, every learned `E` is 1 by
definition, and `predictObjectCost` returns the observed cost unscaled. The
reported APE is therefore pure epoch-to-epoch workload variance, not model
error. An A/B of the work tag over five runs each found no accuracy difference
(tagged median 0.022-0.032, untagged 0.026-0.030), which is the expected result
rather than a disappointing one: bucketing cannot affect a prediction that
applies no scaling factor. Every claim about cross-GPU accuracy still needs
heterogeneous hardware.

Also still unexercised on hardware: nested tag scopes, tracing on/off
transitions, the component cap firing (`capped=0` throughout), reference
rebasing under a reference that actually disappears, and multi-process sharing
of one GPU. Rate-file loading remains pending, so `+LBGpuRateFile` is parsed and
ignored.

This plan turns the online per-kernel relative-scaling proposal into an
implementation sequence for the current `rate-aware-gpu-lb` branch.

The first supported consumer will be `GreedyRefineCentralGPULB`. The estimator
will live in the central-LB base layer so that other centralized strategies can
reuse it later. Distributed and hierarchical strategies are deliberately left
for a later phase because their statistics protocols do not currently carry
destination-dependent object costs.

The initial feature will be opt-in. With the feature disabled, load-balancing
decisions must remain unchanged on collision-free workloads. Correcting the
pre-existing raw-object-ID collision is an intentional prerequisite bug fix and
may change attribution for applications with multiple aliased collections even
when scaling is disabled.

This plan does not include:

- FLOP counting or hardware-counter collection;
- automatic inference of a kernel's arithmetic precision;
- a globally optimal unrelated-machines solver;
- mandatory exploratory migrations;
- HIP support in the first implementation.

## Outcome

At each load-balancing epoch, the runtime will be able to answer:

```cpp
double predictGpuCost(const LDObjData& object,
                      const BaseLB::ProcStats& destination) const;
```

The result will be a predicted whole-device GPU service demand in seconds for
the object on the destination GPU. `GreedyRefineCentralGPULB` will use this
value when evaluating every candidate GPU group instead of assuming that
`LDObjData::gpuTime` is independent of destination.

The model will start from a per-GPU hardware-rate prior and update from normal
CUPTI observations. It will track sample count and variance, survive PUP and
checkpoint/restart, and retain a prior-only prediction for unseen pairs.

## Existing pipeline to preserve

The current data path is:

```text
CUPTI kernel + external-correlation records
    -> LBKernelRecord grouped by raw object id
    -> per-device sweep-line normalization
    -> object -> scalar gpuTime
    -> LDObjData::gpuTime
    -> CentralLB statistics
    -> GreedyRefineCentralGPULB scalar packing
```

The new path will be:

```text
CUPTI kernel + object correlation + optional work tag
    -> uniquely identified object/kernel/work-bucket records
    -> per-device sweep-line normalization
    -> compact per-object kernel epoch summaries
    -> LDObjData::gpuCosts + existing scalar gpuTime
    -> CentralLB statistics
    -> persistent GpuScalingModel update
    -> predictGpuCost(object, destination GPU type)
    -> destination-aware greedy/refine assignment
```

The existing scalar must remain equal, within floating-point tolerance, to the
sum of the modeled kernel components plus the unmodeled residual. That invariant
lets the feature be introduced without changing current accounting.

## Design decisions

### 1. Use globally unique LB object identities

The current CUPTI correlation pushes `CkMigratable::ckGetID()`, which is only an
element ID, and `LBDatabase::SetObjGPULoad` strips collection bits to match it.
Different chare arrays can therefore alias in the CUPTI maps.

Replace that representation before adding model state:

- obtain the active object's `LDObjHandle` through its `CkLocRec`;
- identify an object by `LDObjKey { omID, objID }`;
- allocate/reuse a process-local 64-bit CUPTI token for each full key;
- push the token as the external correlation ID;
- retain `token -> LDObjKey` until the epoch's CUPTI records have been drained;
- key normalized loads and kernel summaries by the full `LDObjKey`.

Do not depend on collection bits always being enabled, and do not pack a local
`LDObjHandle::handle` index into the token. Local handle indices are not stable
object identities.

### 2. Define kernel class and work bucket separately

The default kernel class is a deterministic hash of the copied CUPTI kernel
name. Do not use `std::hash`; use a specified stable hash and retain a debug-only
ID-to-name table so collisions can be detected.

The default work bucket is a deterministic hash of:

- grid dimensions;
- block dimensions;
- dynamic shared-memory bytes.

That default is insufficient when work varies through kernel arguments while
launch geometry stays fixed. Add an optional HAPI launch annotation using a
second CUPTI external-correlation kind (`CUSTOM0`):

```cpp
class hapiCuptiKernelTagScope {
 public:
  explicit hapiCuptiKernelTagScope(uint64_t workTag);
  ~hapiCuptiKernelTagScope();
};
```

The tag is pushed immediately around the CUDA runtime launch, so CUPTI associates
it with the same runtime correlation ID as the kernel record. A convenience
macro may be provided, but the RAII form is the primitive so early returns and
errors cannot unbalance the CUPTI stack.

Applications should use the tag for logical work that CUPTI cannot observe. For
example, `jacobi2d-imbalance` must include `load_iters` in the tag because all
chares use the same launch geometry while executing different loop counts.

An untagged launch remains supported and uses the automatic launch-signature
bucket. Records with incompatible explicit and automatic metadata must not be
merged.

#### A lost work tag must not fall back to the untagged bucket

`hapiProcessCuptiBuffers` currently wipes both correlation maps at the end of a
round, so a `CUSTOM0` record whose kernel arrives in a later drain is gone by
the time that kernel is resolved. The kernel then keeps the bucket the parse
loop already gave it, `gpuStableWorkBucket(launch, false, 0)`.

That failure is qualitatively worse than the equivalent one on the object path.
A dropped object correlation costs one sample: the kernel lands in
`cupti_unattributed_kernels_`, counts as contention, and bills nobody. A dropped
work tag instead files the kernel under a bucket that the `hasExplicitTag`
discriminator guarantees is distinct from the one every correctly tagged
instance of that kernel uses. Since the tag exists precisely to separate work
that launch geometry cannot, the untagged bucket for that kernel accumulates a
mixture over every tag value whose correlation was lost, and its
`meanLogNormalizedDemand` is the geometric mean of that mixture.

The ratio estimator then consumes it. If two GPU types lose tags at different
rates or for different tag populations, `derivedLogE` differences two means
taken over different work-size mixtures. The identity that makes the estimator
valid requires `T_{k,r}` and `T_{k,g}` to describe comparable work; here they do
not, and the discrepancy is absorbed into `E_{k,g}`, which is supposed to be a
property of the hardware alone. The result is a biased estimate of a hardware
constant, learned with a healthy-looking sample count and variance, that under
`alpha_n = 1/n` needs as many good samples to wash out as it took bad ones to
build.

The runtime cannot tell "this launch was never tagged" from "this launch was
tagged and the tag went missing" by looking at one record, because a legitimately
untagged launch is supported and looks identical. Both correlation maps are also
cleared together at the end of a round, so the ordinary cross-round loss takes
the object correlation with it and the kernel lands safely in
`cupti_unattributed_kernels_` rather than in a bucket. The dangerous case is
narrower than a plain dropped tag: it needs the object correlation to be present
while the tag is not, which requires an asymmetry between the two record streams
-- a tracing on/off transition landing between the object push and the tag push,
a CUPTI detach generation change mid-entry-method, or CUPTI dropping one record
kind.

Detecting it therefore has to be done at the level of the kernel class rather
than the individual launch.

Implemented behavior:

- `GPUManager::cupti_tagged_kernel_classes_` remembers every kernel class that
  has been observed carrying an explicit tag. An untagged instance of a class in
  that set is the asymmetry, and its record is flagged `unmodelable`.
- an `unmodelable` kernel still earns load and still occupies SMs in the
  contention sweep; its demand goes to `unmodeledGpuTime`, where the hardware
  prior prices it, instead of into a bucket it does not belong in.
- within one drain, a kernel of a known-tagged class whose tag has not been
  parsed yet is parked rather than flagged, because the tag record may still be
  in an unparsed buffer. Only a kernel still untagged after every buffer has
  been parsed is flagged.
- the count is reported at `+LBDebug` level 1, not 2. A nonzero value means the
  model would have been learning a fiction, not that some bookkeeping was
  discarded.

Residual limitation: the first untagged instance of a class seen before any
tagged instance is not flagged, because the class is not yet known to be tagged.
The set persists across rounds, so this self-corrects after the first epoch.

### 3. Separate device instance from device type

Add a descriptor to each `DeviceManager`:

```cpp
struct GpuDeviceDescriptor {
  uint64_t instanceId;       // current job-wide physical-node/ordinal id
  uint64_t typeId;           // deterministic hardware-class fingerprint
  double peakRateScore;      // cold-start prior, positive and finite
  uint32_t smCount;
  uint32_t computeMajor;
  uint32_t computeMinor;
  uint32_t maxClockKHz;
  uint64_t totalMemory;
};
```

`typeId` should include enough properties to distinguish full GPUs from MIG
slices and otherwise unequal devices: product/device identity, compute
capability, SM count, and relevant memory/clock attributes. Identical instances
should produce the same type ID on different hosts.

Do not reuse `LBManager::ProcessorGPUSpeed()`. Its resident-thread-based FLOP
formula is not a GPU peak rate, its cache is process-wide, and it selects a
device independently of HAPI's mapping.

Implement one `GpuPeakRateProvider` with an auditable fallback chain:

1. user-provided rate for the type, when configured;
2. architecture-aware device table for known compute capabilities;
3. documented `SM count * max clock` proxy for unknown devices.

Log both the score and its source once per device type. The score is a prior;
after both types have observations, it cancels out of the learned timing ratio.

### 4. Preserve normalized GPU service demand

`LDObjData::gpuTime` is currently SM-utilization-normalized whole-device demand.
It deliberately does not equal the raw sum of overlapping kernel durations.
That is the quantity consumed by the GPU-group makespan objective and must remain
the primary predicted cost.

Extend the sweep-line calculation so each attributed kernel record accumulates
its own normalized service demand. Aggregate those values by object and kernel
key. Also retain raw log-duration moments for estimator diagnostics and for a
shadow raw-time predictor.

The initial scheduler-facing model will learn relative normalized service demand:

```text
D[k,b,g] = normalized whole-device seconds for one comparable invocation
```

This uses only timestamps and already-collected launch/device attributes, while
remaining compatible with the existing objective. During validation, report
prediction error for both normalized demand and raw duration. If raw duration is
the better predictor for a workload with no overlap, that is evidence for a
later phase/critical-path model, not a reason to discard normalized accounting.

### 5. Maintain per-type log-cost statistics, then derive relative factors

For each `(kernelClass, workBucket, gpuType)`, maintain mergeable Welford moments
of log cost:

```cpp
struct LogCostStats {
  uint64_t samples = 0;
  double meanLogCost = 0.0;
  double M2 = 0.0;
  uint64_t lastEpoch = 0;
};
```

Storing per-type means is preferable to arbitrarily pairing a target sample
with one reference sample collected at a different time. For reference type
`r`, derive:

```text
logE[k,b,g] = log(P[r]) - log(P[g])
              + meanLogCost[k,b,r] - meanLogCost[k,b,g]
```

This is the proposal's estimator applied to geometric-mean comparable costs.
It also makes reference rebasing a view change rather than a destructive rewrite
of every learned entry.

For a drifting mode, update the moments used for prediction with
`alpha = max(alphaMin, 1/n)`. Keep lifetime Welford moments separately for
confidence and diagnostics if `alphaMin > 0`.

### 6. Bound memory and message growth

Only compact summaries cross the LB statistics path; raw CUPTI records stay
inside their producing process.

Per object and epoch:

- aggregate all invocations of the same `(kernelClass, workBucket)`;
- retain at most a configurable number of modeled components, initially 64;
- select the largest components by normalized service demand;
- combine all remaining service demand into `unmodeledGpuTime`;
- predict the residual with the hardware prior rather than dropping it.

Age out global model entries that have not been observed for a configurable
number of epochs. Never discard an entry still referenced by the current epoch.

### 6a. The per-round capture path also has to stay bounded

Two Phase 1 changes multiply the memory held between load-balancing steps, and
both are paid whether or not `+LBGpuScaling` is on, because only the population
of the new fields is gated, not their presence.

`LBKernelRecord` grew from 24 to 96 bytes:

```text
before:  start_ns 0..8  end_ns 8..16  device_id 16..20  sms_used 20..24    = 24
after:   ...24  GpuKernelKey 24..40  GpuLaunchSignature 40..80
         explicit_work_tag 80..88  has_explicit_work_tag 88..89 (+7 pad)   = 96
```

And every kernel now transits the `pending` vector. Before, a kernel whose
external-correlation record had already been parsed -- the common case, since
the correlation is normally queued into the same buffer ahead of the kernel
record -- went straight into its destination vector inside the parse loop, and
`pending` held only stragglers. Peak live bytes for the record set therefore
went from roughly `N*24 + S*28` with `S << N` to `N*104` for `PendingKernel`
plus `N*96` in the destination vectors as resolution proceeds: about 200 bytes
per kernel at the crossover against 24 before. `pending` also has no `reserve`,
so geometric doubling transiently holds another 1.5x.

At a few hundred thousand kernels per LB period this is tens of megabytes and
does not matter. At several million it is close to a gigabyte, on a node that is
also holding the application's host and device working sets.

Required behavior:

- keep the immediate-dispatch fast path. Parking everything was done to handle a
  correlation record landing in a buffer that completes after the kernel's,
  which is real and now also true of the `CUSTOM0` tag, but parking is only
  needed when something is actually missing:

  ```text
  if object correlation present and (scaling off or work tag present):
      dispatch immediately
  else:
      park in pending
  ```

- `reserve` `pending` from the previous round's straggler count.
- drop `GpuLaunchSignature` from `LBKernelRecord`. `workBucket` is a hash of it
  and is the only thing consumed downstream, so storing both costs 40 of the 72
  added bytes for nothing. The raw geometry is only needed if the affine size
  model is adopted later, and that would be per `(kernel, bucket)`, not per
  record.

Three further tables grow for the lifetime of the process and need bounding
before a long run can be trusted. None of them is bounded by the live object or
kernel count; each is bounded by everything ever seen.

- `GpuObjectTokenTable` (`src/ck-ldb/lbdb.h`). Two maps, roughly 48 bytes per
  distinct LB identity, retained forever. `hapiClearCuptiData` deliberately does
  not clear it because a later epoch must reuse the same token for the same
  identity, so a workload that creates and destroys array elements leaks a token
  pair per dead element. Eviction cannot simply drop unused tokens: a kernel
  record for a departed object can still arrive in the next drain, and any
  scheme that renumbers or reuses a token silently misattributes that kernel.
  Options are (a) evict on the LB epoch boundary after the drain, once no
  correlation can still reference the token, keyed off the LBManager unregister
  path, or (b) keep interning but reclaim in bulk when the table crosses a
  threshold, using a generation counter so a stale token resolves to "unknown"
  rather than to the wrong object. Whichever is chosen must also invalidate the
  per-PE `cupti_local_object_tokens` caches in `hapi_impl.cpp`, which are safe
  today only because the table is append-only.
- `GpuScalingModel::entries_`. Keyed by `(kernelClass, workBucket, gpuType)`, so
  an application whose launch geometry varies per step adds an entry per step.
  This is the aging rule above; `GpuAdaptiveLogStats::lastEpoch` is already
  recorded for it but nothing reads it yet.
- `GPUManager::cupti_kernel_names_`. Diagnostic only, populated when
  `+LBDebug` is on, never cleared. Should be cleared with the rest of the round
  in `hapiClearCuptiData`, or capped.

## Data structures

Add PUP-able, CUDA-independent structures in `src/ck-ldb/GpuScalingModel.h`:

```cpp
struct GpuKernelKey {
  uint64_t kernelClass;
  uint64_t workBucket;
};

struct GpuKernelEpochCost {
  GpuKernelKey key;
  uint64_t calls;
  double normalizedDemand;
  double meanLogNormalizedDemand;
  double M2LogNormalizedDemand;
  double meanLogDuration;
  double M2LogDuration;
};

struct GpuObjectEpochCosts {
  uint64_t sourceInstanceId;
  uint64_t sourceTypeId;
  std::vector<GpuKernelEpochCost> components;
  double unmodeledGpuTime;
};

struct GpuScalingEntry {
  LogCostStats normalized;
  LogCostStats rawDuration;
};

class GpuScalingModel {
 public:
  void observe(const BaseLB::LDStats& stats, uint64_t epoch);
  double predictGpuCost(const LDObjData& object,
                        const BaseLB::ProcStats& destination) const;
  double predictKernelCost(const GpuKernelEpochCost& component,
                           uint64_t sourceType,
                           uint64_t destinationType) const;
  void selectOrRebaseReference(const BaseLB::LDStats& stats);
  void pup(PUP::er& p);
};
```

The exact ownership dependencies may require forward declarations or splitting
wire types from the model class. The wire structures must not include CUDA or
CUPTI headers so estimator unit tests can run in a CPU-only build.

### Link-order constraint

`GpuScalingModel.C` is compiled into `libck`, and charmc puts `-lhybridapi`
*after* `-lck` on the link line. An archive is scanned once, so anything the
CUPTI path in `hapi_impl.cpp` calls cannot be resolved out of `libck` and fails
at link time with an undefined reference -- in an unrelated example binary,
which makes it look like someone else's breakage.

Everything HAPI touches therefore has to be defined in the header:
`GpuKernelKeyHash`, `GpuKernelTypeKeyHash`, all of `GpuRunningMoments`,
`GpuKernelEpochCost`, `GpuObjectEpochCosts`, and the `gpuStable*` /
`gpuDerivePeakRateScore` helpers. `GpuAdaptiveLogStats` and `GpuScalingModel`
stay in the `.C` because only the LB side constructs them; if a later phase has
HAPI evaluate a prediction in-process, they have to move too.

Extend:

- `LDObjData` with `GpuObjectEpochCosts gpuCosts` under `CMK_CUDA`;
- `BaseLB::ProcStats` with the PUP-able GPU device descriptor fields;
- `LBObj`/`LBDatabase` with setters that install the current epoch's summary;
- `LDStats::useMem()` so nested vector storage is counted.

Bump `LB_FORMAT_VERSION` and guard new PUP fields so LB simulation files either
fail with a clear version error or load older data with empty GPU summaries.

## Phased implementation

### Phase 0: Pure estimator and build wiring

Files:

- add `src/ck-ldb/GpuScalingModel.h`;
- add `src/ck-ldb/GpuScalingModel.C`;
- update `src/ck-core/CMakeLists.txt`;
- update `src/scripts/Makefile` and installed header lists;
- add `tests/charm++/load_balancing/gpu_scaling_model/`;
- add the test directory to `tests/charm++/load_balancing/Makefile`.

Work:

- implement deterministic key ordering/hashing;
- implement Welford update and merge;
- implement stationary and hybrid log-space updates;
- implement prior-only, calibrated, and mixed prior/calibrated prediction;
- implement deterministic reference selection and non-destructive rebasing;
- implement PUP for all model types;
- add a feature flag and configuration fields to `CkLBArgs`:
  - `+LBGpuScaling` (default off),
  - `+LBGpuScalingAlphaMin <double>` (default `0`),
  - `+LBGpuScalingMinSamples <int>` (default `1` initially),
  - `+LBGpuRateFile <path>` (optional type-to-rate overrides).

Tests:

- exact two-device noiseless convergence;
- running mean equals the batch mean in log space;
- Welford merge equals serial observation;
- fixed-alpha drift response;
- unseen destination uses the peak-rate prior;
- rebasing preserves every pairwise prediction;
- invalid/nonpositive/NaN observations are rejected;
- PUP round trip preserves predictions and confidence.

Gate:

- all tests run without CUDA hardware;
- no existing runtime behavior changes;
- both CMake and classic Make builds include the component.

### Phase 1: Correct attribution and capture metadata

Files:

- `src/arch/cuda/hybridAPI/hapi.h`;
- `src/arch/cuda/hybridAPI/hapi_impl.cpp`;
- `src/arch/cuda/hybridAPI/gpumanager.h`;
- `src/arch/cuda/hybridAPI/devicemanager.h`;
- `src/ck-ldb/lbdb.h`;
- `src/ck-ldb/LBDatabase.h`.

Work:

- replace raw element-ID attribution with full-key token attribution;
- copy and deterministically intern/hash `kernel->name` before freeing the CUPTI
  activity buffer;
- store grid, block, shared-memory, and explicit work-tag metadata in
  `LBKernelRecord`;
- parse external-correlation records by correlation kind so object tokens and
  `CUSTOM0` work tags do not share a map;
- add balanced RAII push/pop functions for work tags;
- discover and cache one `GpuDeviceDescriptor` per `DeviceManager`;
- preserve the existing scalar normalization output exactly for collision-free
  object IDs; collision cases should change only by becoming correctly
  separated.

Tests and instrumentation:

- unit-test token allocation/reuse and full-key lookup;
- test two array collections with the same element IDs and confirm separate GPU
  ownership;
- test nested/unattributed entry methods and tracing on/off transitions;
- add debug counters for kernel records, object correlations, work tags,
  unattributed records, invalid durations, and hash collisions.

Gate:

- with `+LBGpuScaling` absent, existing `gpuTime` values on collision-free
  workloads match the pre-change implementation within floating-point
  tolerance;
- no object history is merged across array collections;
- tracing start/stop remains balanced under SMP.

### Phase 2: Produce and transport per-object summaries

Files:

- `src/arch/cuda/hybridAPI/hapi_impl.cpp`;
- `src/arch/cuda/hybridAPI/gpumanager.h`;
- `src/ck-ldb/lbdb.h`;
- `src/ck-ldb/LBObj.h` and `LBObj.C`;
- `src/ck-ldb/LBDatabase.h` and `LBDatabase.C`;
- `src/ck-ldb/BaseLB.h` and `BaseLB.C`;
- `src/ck-ldb/CentralLB.h` and `CentralLB.C`.

Work:

- attribute normalized sweep-line demand back to each kernel record;
- locally aggregate per-object/per-key call count, total demand, log-demand
  moments, and log-duration moments;
- exclude unmodelable records (design decision 2: a kernel that resolved an
  object token but lost its work tag) from the components and fold their demand
  into the residual, so a lost tag costs a sample rather than biasing a bucket;
- restore the immediate-dispatch fast path and drop `GpuLaunchSignature` from
  `LBKernelRecord`, per design decision 6a, before the aggregation pass makes
  the per-round record set larger still;
- enforce the component cap and compute the residual;
- install summaries alongside scalar `gpuTime` in each `LBObj`. Note the
  deliberate asymmetry with `SetObjGPULoad`: an object absent from the map has
  its summary *cleared*, where the scalar path leaves the previous epoch's
  `gpuTime` in place. A strategy must be able to tell "no GPU work this round"
  from "the same GPU work as last round", and a stale breakdown paired with a
  fresh scalar would not reconcile. Whether the scalar path should be changed to
  match is a separate question, since existing balancers may depend on the
  carry-over;
- PUP summaries through `LDObjData` and device descriptors through `ProcStats`;
- clear epoch summaries with load statistics while retaining persistent model
  state;
- update format versioning and simulation serialization.

Required invariant:

```text
abs(gpuTime - (sum(component.normalizedDemand) + unmodeledGpuTime)) <= tolerance
```

Tests:

- sequential kernels;
- overlapping half-device kernels;
- unattributed contention;
- multiple invocations of one key;
- component-cap residual accounting;
- stats/PUP round trip;
- message-size accounting in `LDStats::useMem()`.

Gate:

- the invariant holds for every object in debug builds;
- communication volume is proportional to distinct object/kernel buckets, not
  kernel launch count;
- existing scalar LBs still receive the same `gpuTime`;
- peak bytes held per kernel between LB steps do not exceed the pre-Phase-1
  figure by more than the summary structures themselves;
- a kernel whose work tag was dropped never appears in a tagged bucket's
  statistics, and the count of such kernels is reported at `+LBDebug` level 1.

### Phase 3: Online model in shadow mode

Files:

- `src/ck-ldb/CentralLB.h` and `CentralLB.C`;
- `src/ck-ldb/GpuScalingModel.h` and `GpuScalingModel.C`;
- `src/ck-ldb/LBManager.h` and `LBManager.C`.

API note: the model does not take `BaseLB::LDStats` or `ProcStats`, as the
sketch above suggests. It takes the wire types instead --
`observeObjectCosts(const GpuObjectEpochCosts&, epoch)` and
`predictObjectCost(const GpuObjectEpochCosts&, destinationType, metric, out,
weakest)` -- and `CentralLB` does the `LDStats` walk. Two reasons: `lbdb.h`
already includes `GpuScalingModel.h`, so depending on `BaseLB` from the model
would be circular; and keeping the model free of LB types is what lets the whole
estimator be exercised in a CPU-only build with no GPU and no Charm runtime.
`CentralLB::predictGpuCost(const LDObjData&, const ProcStats&)` provides the
signature from the Outcome section on top of that.

Ownership:

- add `GpuScalingModel` to `CentralLB` so all centralized strategies share the
  interface;
- update it after complete `LDStats` assembly and before `Strategy()`;
- PUP it in `CentralLB::pup` so checkpoint/restart and central-PE migration do
  not erase learning;
- ensure all concurrent strategy replicas observe the same summaries in the
  same deterministic order.

Reference policy:

- choose the most common available GPU type;
- break ties by deterministic `typeId` ordering;
- prefer the existing reference while it remains available;
- on disappearance, select a new reference without rewriting stored per-type
  moments.

Shadow behavior:

- calculate `predictGpuCost` for every object/current destination and for every
  available GPU type;
- do not feed predictions to placement yet;
- when a predicted object/key later runs on a destination, record absolute
  percentage error;
- distinguish prior-only, low-confidence, and calibrated predictions.

Diagnostics:

- device descriptor and rate-source dump once per type;
- per-epoch raw/valid/rejected observation counts;
- `(kernel, bucket, type)` samples, mean, variance, age, and derived `logE` at
  high `+LBDebug` levels;
- median, mean, and p95 prediction error by prior/calibrated state;
- estimator update and prediction wall time.

Gate:

- homogeneous runs predict identity scaling and make no decision changes;
- synthetic heterogeneous observations converge to the injected ratios;
- reference loss and a newly appearing type do not produce missing, NaN, or
  negative costs;
- shadow prediction overhead and statistics-message growth are measured.

### Phase 4: Destination-aware `GreedyRefineCentralGPULB`

Files:

- `src/ck-ldb/GreedyRefineCentralGPULB.h`;
- `src/ck-ldb/GreedyRefineCentralGPULB.C`.

Data changes:

- make `GObj` retain its `LDStats` object index and source type instead of one
  destination-independent GPU scalar;
- make `GPUGrp` retain its device instance/type descriptor;
- expose a local helper `cost(obj, group)` that calls `predictGpuCost` and has
  one explicit fallback to `obj.gpuTime` for disabled/invalid models.

Algorithm changes:

1. Build current group loads using predicted cost on each object's current
   group; this should reproduce measured `gpuTime` for the source.
2. Order objects by descending minimum predicted GPU cost across available
   groups, with prediction regret as a tie-breaker so constrained objects are
   placed early.
3. In greedy preprocessing, choose the group minimizing
   `groupLoad[g] + cost(object, g)`, not the group with minimum load alone.
4. In the real assignment, evaluate every feasible group with its destination
   cost and choose the smallest predicted completion time.
5. Apply migration-retention tolerance to predicted completion time, using the
   current group's own destination cost.
6. Add `cost(object, chosenGroup)` to that group; never add the source scalar.
7. Compute reported GPU makespan from the destination-aware group totals, then
   combine it with the existing per-PE CPU dimension.
8. Preserve non-migratable objects as current-group background.
9. Apply GPU-memory and staging-buffer feasibility checks before accepting a
   destination. Port the existing checks from `GreedyRefineCentralLB` rather
   than allowing a better predicted cost to create an impossible placement.

Feature behavior:

- `+LBGpuScaling` absent: execute the old scalar path;
- one GPU type: costs reduce to the existing source costs;
- unseen destination: use the hardware prior;
- invalid prediction: keep the object on its current group and count the
  fallback;
- model confidence affects migration aggressiveness, not availability of a
  prediction.

Tests:

- a small synthetic assignment matrix with a known best mapping;
- crossed preferences, where two kernels prefer opposite GPU types;
- all-prior cold start;
- partially calibrated matrix;
- identical types reproduce scalar placement;
- non-migratable background;
- memory-infeasible best destination is rejected;
- migration-budget and concurrent `(A,B)` solution selection still work.

Gate:

- feature-off decisions are unchanged on collision-free workloads;
- no accepted assignment has a worse predicted makespan than the retained
  current mapping unless allowed by the migration-tolerance policy;
- every decision uses the same cost function for sorting, candidate scoring,
  accumulation, and final objective reporting.

### Phase 5: Confidence, drift, and exploration policy

Work:

- use sample count, standard error, and age to classify confidence;
- add optional log-space blending toward the prior for low-confidence entries;
- enable the `alphaMin` drift mode and report both adaptive and lifetime
  statistics;
- reject or down-weight observations dominated by known overlap/contention;
- add prediction-error-triggered reset/decay for phase changes;
- add optional exploration, default off:
  - prefer ordinary initial placement across types;
  - otherwise allow a bounded migration only when information value and
    predicted makespan risk are within configured limits;
- ensure the reference and model survive shrink/expand checkpoint restart;
- initialize newly discovered types with prior-only entries;
- bound the three lifetime-growth tables described in design decision 6: token
  eviction with a generation counter and per-PE cache invalidation, epoch-based
  aging of `entries_`, and clearing `cupti_kernel_names_` with the round.

Gate:

- no exploration occurs unless explicitly enabled;
- stationary tests show decreasing variance;
- drift tests show bounded adaptation time;
- stale or highly uncertain data cannot trigger an unbounded migration wave;
- checkpoint/restart preserves predictions for existing types and admits new
  types safely;
- a long run that repeatedly creates and destroys array elements holds steady
  resident memory in the token table, and a kernel record arriving after its
  object was evicted lands in the unattributed bucket rather than on another
  object.

### Phase 6: Additional LB consumers

Only start after Phase 4 is validated on heterogeneous hardware.

- `GreedyCentralLB`: replace scalar candidate load with the shared prediction.
- `GreedyRefineCentralLB`: consolidate duplicated GPU-group logic with the GPU
  variant before adding another copy of the estimator calls.
- `DiffusionLB`: extend pseudo-load and neighbor protocols with GPU type/rate and
  enough model state to price a remote destination; a local scalar substitution
  is not sufficient.
- `TreeLB`: extend the level statistics protocol before claiming estimator
  support; it cannot currently transport the required destination data.
- retain an exhaustive or ILP solver for small synthetic cases as an evaluation
  oracle, not a production dependency.

## Runtime validation matrix

### CPU-only and homogeneous regressions

- build without CUDA;
- build CUDA without enabling the feature;
- existing `lb_test` suite;
- one GPU and multiple PEs;
- multiple identical GPUs;
- instrumentation window on/off transitions;
- no kernel observations in an epoch.

Expected result: no placement or scalar-load behavior changes when the feature
is disabled; homogeneous predicted ratios remain one when enabled.

### Heterogeneous estimator validation

Use at least two materially different GPU types and two kernels with different
scaling behavior.

1. Fixed work, isolated kernels: establish ground-truth timing ratios.
2. Fixed work, concurrent streams: compare raw-duration and normalized-demand
   predictors.
3. Multiple work buckets: verify no cross-bucket updates.
4. Prior-only destination: measure cold-start error.
5. One, two, four, eight, and sixteen observations: report convergence.
6. Throttling/contention: evaluate the hybrid update.
7. Remove the reference type and add a previously unseen type.

Report median, mean, and p95 APE plus confidence calibration.

### Application validation

- `jacobi2d-imbalance`: add a work tag containing `load_iters`; use it first for
  a controlled, single-dominant-kernel experiment.
- `leanmd`: exercise multiple kernel classes and underfilled kernels; bucket by
  launch geometry and any hidden atom/pair work not represented by the grid.
- `pic2d`: use for performance/load-balancing experiments only after resolving
  its documented homogeneous-GPU bit-identical initialization assumption.

For each application report separately:

- estimator APE;
- predicted and observed per-GPU makespan;
- iteration time;
- migration count and bytes;
- LB strategy time;
- CUPTI processing and summary-transport overhead;
- end-to-end runtime against scalar GPU LB and no migration.

## Failure handling and invariants

The implementation must enforce:

- `peakRateScore > 0` and finite;
- observed duration and normalized demand are positive and finite;
- no hash collision is silently accepted in debug/validation builds;
- external-correlation stacks remain balanced across tracing transitions;
- all PEs describing one GPU instance agree on its type and rate score;
- a source prediction reproduces the source epoch cost within tolerance;
- every object always has a prediction, even if it is prior-only;
- no estimator state is cleared by `hapiClearCuptiData()`;
- model updates happen once per complete LB epoch;
- feature-off behavior does not read or depend on model state;
- message and model memory are bounded.

On a violated runtime invariant, prefer a counted fallback to current scalar
cost and retain placement. Abort only for structural corruption such as an
unbalanced CUPTI stack, inconsistent device identity, or impossible PUP data.

## Expected file map

New files:

- `src/ck-ldb/GpuScalingModel.h`
- `src/ck-ldb/GpuScalingModel.C`
- `tests/charm++/load_balancing/gpu_scaling_model/Makefile`
- `tests/charm++/load_balancing/gpu_scaling_model/gpu_scaling_model.C`

Primary modified files:

- `src/arch/cuda/hybridAPI/hapi.h`
- `src/arch/cuda/hybridAPI/hapi_impl.cpp`
- `src/arch/cuda/hybridAPI/gpumanager.h`
- `src/arch/cuda/hybridAPI/devicemanager.h`
- `src/ck-ldb/lbdb.h`
- `src/ck-ldb/LBObj.h`
- `src/ck-ldb/LBObj.C`
- `src/ck-ldb/LBDatabase.h`
- `src/ck-ldb/LBDatabase.C`
- `src/ck-ldb/BaseLB.h`
- `src/ck-ldb/BaseLB.C`
- `src/ck-ldb/CentralLB.h`
- `src/ck-ldb/CentralLB.C`
- `src/ck-ldb/LBManager.h`
- `src/ck-ldb/LBManager.C`
- `src/ck-ldb/GreedyRefineCentralGPULB.h`
- `src/ck-ldb/GreedyRefineCentralGPULB.C`
- `src/ck-core/CMakeLists.txt`
- `src/scripts/Makefile`
- `tests/charm++/load_balancing/Makefile`

Application changes for validation should be separate commits from runtime
changes.

## Commit sequence

Keep each commit independently buildable and, where possible, behavior-neutral:

1. Add pure scaling-model types, algorithms, tests, and build integration.
2. Correct CUPTI object identity without changing scalar load results.
3. Capture deterministic kernel/work/device metadata behind the feature flag.
4. Produce per-kernel normalized summaries and prove scalar conservation.
5. PUP summaries and device descriptors through central statistics.
6. Update/persist the model in CentralLB and add shadow diagnostics.
7. Convert `GreedyRefineCentralGPULB` candidate evaluation to destination costs.
8. Add hard memory constraints and confidence-based migration safeguards.
9. Add annotated heterogeneous validation workloads and publish results.
10. Add drift, reference-loss, and optional exploration support.

Do not combine object-identity correction, new accounting, and changed placement
in one commit. Each affects correctness independently and needs its own regression
boundary.

## Definition of done for the first release

The first release is complete when:

- the feature is opt-in and feature-off placement is unchanged except where the
  prerequisite object-identity fix corrects an existing collection collision;
- object attribution is collision-free across chare collections;
- kernel class and comparable-work bucket reach the central LB in bounded
  summaries;
- per-kernel summaries exactly conserve existing normalized `gpuTime`;
- the model passes convergence, prior, rebasing, merge, and PUP unit tests;
- a destination-cost API is available to all `CentralLB` derivatives;
- `GreedyRefineCentralGPULB` evaluates and accumulates destination-specific cost;
- homogeneous GPUs reproduce scalar-LB behavior;
- a two-type heterogeneous run demonstrates lower calibrated prediction error
  than the peak-rate prior;
- at least one mixed-kernel application improves observed makespan or iteration
  time without violating memory/migration constraints;
- estimator, CUPTI, statistics-message, and strategy overheads are reported.
