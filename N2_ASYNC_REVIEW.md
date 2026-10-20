# N2 LeanMD async review, 2026-09-17

Reviewed `N2_LEANMD_OCCUPANCY_HANDOFF.md`, the preserved rank logs for
22151249, 22154226 and 22155894, and the current measurement/async source.
The initial review made no runtime changes. The accounting fix described below
is now implemented; no new cluster runs were made.

## Implemented fix and validation

`LBDatabase::Send` and `MulticastSend` now exclude traffic from an active object
whose `joinedStep` gate is closed. The check also covers forced array sends.
An open object on the same PE still contributes, and the existing resume path
reopens communication together with compute. Processor-origin forced sends and
forced sends from open objects retain their previous behavior. Self-message
filtering and the PE-wide instrumentation/trace switches are unchanged.

Validation:

- `make -C tests/charm++/load_balancing/lbdriver check-comm-window`: passes.
  This standalone test links the real LBDatabase send methods, LBObj compute
  accumulators and communication table with runtime stubs, without GPU/NIC
  initialization. It covers closed/open objects on one PE, forced sends,
  multicast, resume, disabled instrumentation, self sends and processor records.
- The same test compiled against a temporary copy with the two new gates
  removed fails its message/byte assertion. The regression catches the old bug.
- `cmake --build multicore-linux-x86_64-cuda --target ck moduleDiffusionLB -j4`:
  passes. Existing header macro and unrelated missing-return warnings remain.
- `make -C examples/charm++/cuda/gpudirect/leanmd -W Main.o`: relinked the normal
  LeanMD executable against the rebuilt runtime.
- `git diff --check`: passes. Changes are uncommitted; other workspace work was
  preserved.

This fixes the inconsistent communication sampling gate. It does **not** add
full-period extrapolation: the runtime has no explicit application-step sample
fraction, and deriving it from wall time would mix compute and waiting. The
existing one-off migration charge remains conservative for short async samples;
the cost-model header now states that limitation instead of asserting every
sample covers a full period. Double-buffered full-period measurement or explicit
sample-horizon metadata would be a separate change.

Next validation is the matched N2 comparison below. In the fixed lag-16 arm,
recorded traffic should shrink with the compute sample instead of remaining
near sync's full-period volume. Compare traffic per measured work as well as
mapping, runtime and GPU balance; do not treat lower recorded traffic as a
physical reduction in transferred bytes. Previously saved binaries and scripts
that select them will not contain this fix.

## Follow-up: confirmed accounting defect in the source

Compute and communication do **not** use the same sampling gate under async:

- `cklocation.C:2715` closes each object's gate at its join. Host/device
  accumulators check `joinedStep` (`LBObj.C:51,64`), and CUPTI suppresses its
  object correlation (`hapi_impl.cpp:4642`). The gate remains closed until the
  object's wait/resume, even if the strategy finished much earlier.
- `LBDatabase.C:131` (`Send`) and `:160` (`MulticastSend`) check the PE-wide
  statistics flag, but never the sender object's `joinedStep`.
- `ckarray.C:1961` (`recordSend`) also checks only the PE-level collection
  state, then calls `Send(..., force=1)`. That force path must be considered
  when fixing the gate; changing just the non-forced branch is insufficient.
- `DistBaseLB.C:395` reopens PE-wide instrumentation at migration completion.
  From that point until the lagged wait, communication is collected while the
  object's compute is excluded.
- `DiffusionMetric.C` consumes the raw message/byte counts and compares
  `loadBenefit - migrateCost - commDelta`. There is no sample-horizon correction.
  The same-window claim at `DiffusionCostModel.h:26` is therefore false under
  async.

Job 22154226 makes this visible quantitatively:

| Diffusion round | Sync recorded MB | Async recorded MB | Async/sync ratio of MB per declared GPU second |
| --- | ---: | ---: | ---: |
| 1 | 21047.5 | 19310.8 | 4.56 |
| 2 | 21047.5 | 20576.6 | 4.87 |
| 3 | 21047.5 | 21581.0 | 5.12 |
| 4 | 21047.5 | 19424.8 | 4.52 |

MB is the sum of the log's internal and external BEFORE-LB totals, not measured
physical-link traffic. The nopeer job independently shows ratios 4.49–5.08.
Thus the short compute window is not accompanied by a corresponding reduction
in recorded communication. Calibration constants can be correct while their
inputs describe different amounts of application work.

This biases both move ranking and acceptance: a positive communication delta is
too punitive relative to compute benefit; a negative delta is too attractive.
It does not merely suppress all moves. Migration amortization also needs the
explicit sample horizon described below.

Fix direction: use one per-object measurement epoch for compute and outgoing
communication, including forced array sends and multicast, then normalize all
sampled quantities to a common prediction horizon. Preserve intentional
non-object/explicit measurement semantics rather than indiscriminately dropping
every forced record. An alternative is double-buffered epochs that keep
measuring application work during async LB without modifying the frozen snapshot.

This confirms an accounting bug, not the cause of the entire 14.5% mean42 gap.
The slow async run has good aggregate GPU balance, and sampled donor summaries
show successful shedding rather than pervasive cost rejection. Runtime causality
still needs a controlled comparison after correcting the accounting.

## What the runs establish

| Job | Sync mean42 (ms) | Async mean42 (ms) | Sync total (s) | Async total (s) |
| --- | ---: | ---: | ---: | ---: |
| 22154226, cutrise | 1257.6 | 1440.0 | 129.832 | 136.998 |
| 22155894, nopeer | 1388.9 | 1307.5 | 134.374 | 132.537 |

The first pair's async penalty is 14.5% in mean42, but 5.5% in application
time. The second pair reverses the result. These are different placements as
well as different executions, even between the two identically configured sync
arms: Metis moves 4489 versus 4720 objects. Async Metis moves 5177 versus 4699.
Configuration equality does not establish placement equality.

For 22154226 the maximum/mean of per-process **kernel-duration sums** in the
five successive drains is:

- Sync: 1.761, 1.276, 1.120, 1.114, 1.113.
- Async: 1.742, 1.111, 1.070, 1.083, 1.102.

Async has better balance by this proxy throughout, and less cumulative kernel
time (803.3 versus 810.1 GPU seconds through the last drain). More GPU work or
worse aggregate GPU balance therefore does not explain this pair's slowdown.
Ordinary steps, excluding residues 0 and 1 modulo 20, average 1324.5 ms async
versus 1186.1 ms sync. The runtime loss is spread beyond the named LB boundaries.
These exclusions are not a complete async-overhead decomposition.

## The sampling-window discrepancy is concrete

All three inspected N2 async logs explicitly say **lag 16**, with period 20.
This is not the earlier lag-4 configuration that matched sync at N1.

`CkMigratable::lbJoinStep` closes the object's measurement gate.
`ResumeFromSyncHelper` returns without reopening it when an async object has
not parked; `AtSyncWait` subsequently reopens it. Thus a completed LB operation
can leave attribution closed until the application's lag expires. At lag 16,
only roughly four steps remain before the next period closes the gate again.

The reported Diffusion T_g values confirm that fivefold change in coverage:

| Diffusion round, job 22154226 | Sync T_g (s) | Async T_g (s) |
| --- | ---: | ---: |
| 1 | 14.985 | 3.016 |
| 2 | 15.858 | 3.183 |
| 3 | 16.076 | 3.219 |
| 4 | 16.208 | 3.307 |

This is not fivefold less physical GPU work. For example, async process 0's
second drain processes 290057 kernels, of which 57082 are attributed and
232975 unattributed; its cumulative kernel-time counter increases by 18.632 s.
The short window explains why comparing declared load with the whole wall
interval is misleading under async.

The cost model compares load benefit with communication delta plus amortized
migration cost (`DiffusionMetric.C`, `DiffusionCostModel.h`). A one-off migration
cost does not automatically shrink with a shorter sample. A proper fix should
give compute and communication measurements an explicit common horizon and
price migration over the actual placement lifetime. Multiplying GPU loads alone
would not suffice. This is a measurement/model concern, **not a demonstrated
cause of the observed runtime penalty**: the slow async arm is already balanced
by the kernel-sum proxy and accepted 107, 156, 2, 2 cross-process moves.

## Claims in the handoff that need qualification

1. `cupti-device-total kernel_s` accumulates kernel durations. It is not a union
   of busy intervals and not SM occupancy. Concurrent kernels can overlap.
   These totals alone cannot prove 97% occupancy, an exact idle-time difference,
   or that no GPU computed anywhere for 25% of a wall interval. The last claim
   needs aligned timelines across GPUs; averaging their durations cannot give it.
2. Better balance does not rule out placement quality. Endpoint distribution,
   critical-path dependencies and transfer traffic can differ between equally
   balanced placements. Both sync runs applied different Metis mappings.
3. Disabling peer decision did not appreciably change the whole-run ordinary-step
   mean (1324.5 to 1307.7 ms), but changed async mean42 by 132.6 ms. With different
   mappings and a 131.3 ms sync swing, that single comparison does not isolate or
   conclusively exonerate forwarding.
4. LeanMD prints its step timer before entering LB. Sync work at step 20 appears
   in step 21; async also has a lag-16 wait point at step 36, charged subsequently.
   Grouping only multiples of 20 and their successors misses that distinction.
5. The old N2 join experiment compared noLB with sync, not async. It establishes
   long dependency waits but cannot identify the cause of this async gap.

## Best next discriminator

Run a matched N2 sync / async-lag-4 / async-lag-16 comparison with the same
peer-decision policy, isolated logs and consistent instrumentation. Include a
repeat/order reversal because the existing sync spread is large. Record both
whole application time and mean42. No N1 run is necessary for this question.

Use the existing join trace in both modes to compare input readiness, compute
completion, and force consumption. Trace the lag-window phases as well as the
existing sample steps; those timers include scheduling/queue delay and cannot
alone distinguish network transfer from GPU starvation. Record a per-process
union-of-kernel-intervals audit with matching wall windows. To determine whether
the entire job is idle simultaneously requires aligned cross-process timelines.
Log actual mapping/cut information so a scheduling change is not confused with
a different placement.

The evidence currently locates the loss in application progress between LB
boundaries, despite good aggregate GPU balance. It does **not** yet identify
whether the remaining delay is transfer/dependency latency, runtime scheduling,
instrumentation, or forwarding. Calling any one of those the root cause would
go beyond these logs.
