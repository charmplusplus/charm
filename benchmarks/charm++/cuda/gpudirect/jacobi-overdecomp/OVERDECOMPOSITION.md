# Streams for Multi-GPU Charm++

What the hardware permits, and what the runtime must do about it, as we add
multiple GPUs per process.

Kale + Claude, 2026-08-31; measurements section added 2026-09-01.

This is the design analysis behind the `jacobi-overdecomp` benchmark that
sits beside it. Read `README.txt` for how to run the benchmark; read this
for what the runs are trying to settle.

## The numbers

Four limits get conflated. Only one of them governs how much actually runs
at once.

| Limit | NVIDIA | AMD | Governs |
|---|---|---|---|
| **Hardware queues** | `CUDA_DEVICE_MAX_CONNECTIONS` default **8**, max **32** | `GPU_MAX_HW_QUEUES` default ~**4**, max ~**8** | Independent host->device command paths. Streams beyond this count share a queue and acquire false dependencies. |
| Concurrent kernels | 128 (Volta and later) | comparable | Grids the device can track. Not reachable through per-stream host queues. |
| Copy engines | A100 reports `asyncEngineCount` **5**; a small fixed number in any case || Concurrent transfers. Extra streams do not add transfer bandwidth. |
| Occupancy | per kernel || Whether co-resident kernels fit. Small kernels overlap; a device-filling kernel serialises everything. |

**The operative number is hardware queues: 32 on NVIDIA, 8 on AMD.**
Everything else follows from that.

## How many streams per chare

The in-tree gpudirect examples give each chare **two** streams --
`compute_stream` and a higher-priority `comm_stream`
(`examples/charm++/cuda/gpudirect/jacobi2d/jacobi2d.C:297`); the
three-stream arrangement (separate H2D and D2H) applies where transfers
stage through the host. Take two per chare: at 10 chares per device that is
20 streams, inside NVIDIA's 32 and well outside AMD's 8, where the same
arrangement supports about four chares per device. That is not
overdecomposition.

**The transfer streams are the ones to give up.** Transfers serialise on the
copy engines no matter how many streams they arrive on, so private H2D/D2H
streams never bought transfer parallelism -- only ordering independence,
which costs one event instead of one queue:

    one shared transfer stream per device
      -> hapiEventRecord(ev_chare, xfer_stream)
      -> hapiStreamWaitEvent(compute_stream_chare, ev_chare)
      -> kernel fires on the device, no host round trip

That takes each chare to a single compute stream, so a device supports ~32
chares on NVIDIA and ~8 on AMD before queues are exhausted. Better, still
not enough for overdecomposition on AMD.

## Measured: what a stream costs

Measured on an A100 (Anvil, CUDA 13.1), since the pool's usual justification
is that creating streams is expensive:

| Operation | Cost |
|---|---|
| Context init (`cudaFree(0)`) | 18.7 us, once |
| First stream after init | 14.5 us |
| **Default (blocking) streams** | 2.6 us each at n=1, rising to **11-13 us each** by n=32 and beyond -- 1.6 ms to create 128 |
| **Non-blocking streams** | **~1.6-2.0 us each, flat** from n=8 to n=128 -- 0.25 ms for 128 |
| With priority | 5.1 us each at n=8 |
| Destroy | ~1.8-2.0 us each, flat |
| Create + destroy round trip | 3.8 us |

**Creation is not the reason to have a pool.** Ten chares with two streams
each is twenty creations -- well under a millisecond, once, against a run
measured in minutes. What justifies a pool is *bounding the number of
streams*, which is what the queue ceiling demands anyway.

The measurement did turn up something worth acting on. The pool creates
*default* streams via `hapiStreamCreate`, and default streams are both
markedly more expensive to create at scale (11-13 us versus a flat 1.6 us)
and semantically worse: they synchronise implicitly with the legacy default
stream, so any launch on the null stream serialises against all of them.
`hapiStreamNonBlocking` already exists in `hapi_portable.h`. The pool should
use it.

## Measured: overdecomposition on Anvil (2026-09-01)

First runs of this benchmark, on one Anvil node with 2x A100, `+pe 4`,
`CUDA_DEVICE_MAX_CONNECTIONS=32`. **All of these had `zerocopy=0`** -- the
`-z` flag was not passed, so they measure the host-staged ghost path (D2H,
host message, H2D), not D2D. They are a baseline to compare `-z` against,
nothing more.

**Multiple GPUs per process works.** One process, 4 PEs, 2 GPUs: HAPI
reported `2 device(s) per process, 2 PE(s) per device`, and chares split 32
on device 0 and 32 on device 1 (PEs 0,2 -> device 0; PEs 1,3 -> device 1).
The mapping did not collapse. Note *why* this benchmark survives multi-GPU
when the runtime pool would not: every chare creates its own streams and
never calls `hapiGetStream`, so it bypasses item 1 below entirely.

One process driving 2 GPUs beat 2 processes driving 1 each -- 1616 vs 1881
us per iteration at the same 4 PEs and 64 chares -- with per-chare ghost
send costs of 3.1-9.6 us against 3.6-18.8 us. The single-process case does
same-process copies where the two-process case pays cross-process costs.

**The overdecomposition sweep found an overhead floor, not a concurrency
ceiling.** Holding the 4096^2 grid fixed and shrinking the block:

| block | chares | chares/device | avg iter | per chare, per PE |
|---|---|---|---|---|
| 1024 | 16 | 8 | 419 us | 105 us |
| 512 | 64 | 32 | 1610 us | 101 us |
| 256 | 256 | 128 | 6414 us | 100 us |

The last column is the result. Cost per chare per iteration is flat at ~100
us even though a 1024^2 block does sixteen times the arithmetic of a 256^2
one, so the chare's actual work is invisible next to a fixed per-chare cost.
Time therefore tracks chare count almost exactly and overdecomposition only
multiplies the fixed cost.

Two things this refutes. The queue ceiling was **not** the binding
constraint -- the curve is already linear at 8 chares per device, far below
the ~16 that 32 queues divided by 2 streams per chare would predict. And
only 3-10 us of the ~100 us is in the ghost send, so the bulk is elsewhere.

Two candidate explanations, both untested as of this writing:

- The host-staged path itself (`zerocopy=0`), which is what `-z` settles.
- Stream flags. This benchmark creates both streams with
  `hapiStreamDefault`, not `hapiStreamNonBlocking`, so every one of them
  implicitly synchronises with the legacy null stream -- 256 such streams at
  128 chares per device. A one-line change tests it.

**What the sweep cannot answer.** Overdecomposition buys latency hiding, and
a single-node run has little latency to hide. The sweep above measures the
overhead floor. Whether overdecomposition ever pays has to be measured at
2+ nodes, where ghost exchange crosses the fabric.

## Multiplexing: yes, under runtime control

Private streams cannot be the mechanism for overdecomposition, because chare
count would stay tied to queue count. The runtime must be able to place
several chares' work on one stream -- deliberately, not by accident.

The distinction that makes it safe: an *accidentally* shared stream puts the
ordering edge in hardware, where nothing can revoke it, and a cycle with a
callback someone is waiting on becomes a deadlock. Work held in a runtime
queue has nothing queued behind it in the driver, so the edge stays in
software where the scheduler can reorder, defer, or drop it. Same apparent
dependency, categorically different failure behaviour.

The first policy worth building is simple: bound the outstanding work per
stream, and submit whichever chare's data is ready. That is the
message-driven principle applied to the device -- the one resource the
runtime currently hands out blindly.

## PE-to-GPU binding: the default, not a rule

PEs are partitioned across devices, each PE bound to one, and the chares on
a device owned by its PEs. That is the default and the path to optimise --
it is what NAMD needs, and it is what makes device-implicit calls
(`hapiGetStream()` returning a stream on *my* device) sound.

It should not be enforced, because the hardware is less restrictive than the
rule would be. Querying a stream or event owned by another device is
permitted; `hapiStreamWaitEvent` explicitly supports a stream on one device
waiting on an event recorded on another; async copies follow their pointers.
The one operation that requires the right current device is a **kernel
launch**, and that costs a `hapiSetDevice` immediately before it. So a PE
polling a completion on another device, or starting a transfer to it, is
legal work that a baked-in restriction would forbid for nothing.

What follows for the API: keep the device-implicit forms as the fast
default, add device-parameterised forms beside them (`hapiGetStream(dev)`),
and make sure the per-PE event queue tolerates events from more than one
device -- which is free, since cross-device query works. The thing to avoid
is baking device-implicitness any deeper; unbaking it later is an API break,
and that is exactly the position the current flat pool put us in.

## First version: mechanism only

Ship the mechanisms and let applications decide how many streams to use and
for what. Policy -- streams per chare, multiplexing, ownership -- waits for
measurements we do not have yet, especially on AMD.

This is already the idiom in practice: every CUDA example creates its own
stream per chare and never touches the runtime pool. Make that the
documented model. **Users create their own streams; the pool is
runtime-internal plus a convenience for casual use, unowned and not to be
held.** Acquire/release, release-at-migration, and exhaustion policy all
disappear from the first version.

Four items below are mechanism rather than policy, and a first version is
not usable without them: the device-indexed pool (1), no silent aliasing
between PEs (3), the queue-count variables (4), and the documented limits
above -- a user choosing streams needs to know 32 and 8. Item 5 is the one
that can wait.

## What multi-GPU support must change

1. **Move the stream pool into `DeviceManager`**, indexed by
   `my_device_id`. Today it is a flat array in the process-wide
   `GPUManager`: streams are created on whichever device was current and
   handed out without reference to the caller's device, so a process
   spanning two devices gives a PE the wrong device's stream. This is the
   item that blocks multi-GPU outright.
2. **Size the pool to the queue count** -- `min(concurrent-kernel limit,
   connections)`, so <=32 and <=8 -- not to the 128 resident-grid limit. The
   extra streams buy no overlap and increase aliasing.
3. **Never let two PEs hold one stream unknowingly.** Until the runtime
   schedules submission, partition the pool per PE (a per-PE cursor and a
   disjoint slice). Under SMP, which is the default on the reconverse line,
   the current rotating dispenser can hand the same stream to two PEs once
   its counter wraps.
4. **Set the queue-count variables during initialisation**, before the
   context exists: `CUDA_DEVICE_MAX_CONNECTIONS=32` and the ROCm
   equivalent, unless the user set them. The three-stream design silently
   loses its overlap at the default of 8, and the value is read at context
   creation.
5. **Create pool streams non-blocking.** `hapiStreamCreate` makes streams
   that implicitly synchronise with the legacy default stream, and that are
   ~6x more expensive to create at scale. `hapiStreamNonBlocking` costs
   nothing extra and removes a serialisation nobody asked for.
6. *(Later.)* **Add acquire/release for chares that hold streams**, keeping
   the existing `hapiGetStream()` as an unowned borrow so current callers do
   not leak. Release should record an event and withhold reuse until it
   completes; a held stream must be released before migration, since the
   destination PE may be on another device.

## Known blocking point

On the same-process (MEMCPY) path, `CkRdmaDeviceOnSender` calls
`hapiStreamSynchronize` on each buffer before the metadata message goes out
-- the receiver dereferences the pointer directly, so the producing work
must have retired. It is correct, and it is the one place in HAPI where a PE
waits on the GPU instead of taking a callback from the scheduler loop.

The cost lands exactly where overdecomposition should help: with several
chares per PE, one chare's send stalls the others for the length of a
kernel, so the cheapest transfer path is the one that serialises a PE.
Fixing it means sending the metadata from a completion callback rather than
inline, which the charmxi-generated send path has to learn to defer -- a
design change rather than a local edit. Deferred deliberately after
discussion; tracked as issue #3957.

Note for anyone measuring this benchmark: the application's packing kernels
and the runtime's transfer share one `comm_stream`, so that synchronise
waits for the packing too. With N chares per PE it serialises N times per
iteration. It is a candidate explanation for the ~100 us floor above.

## Left open deliberately

CUDA graphs are the way past the queue limit -- dependent nodes are
dispatched device-side rather than through per-stream host queues, and an
iterative application's per-step DAG is exactly the shape graphs replay
well. Worth doing after the above, not instead of it.

---

Companion: *How Charm++ Uses GPUs* -- a reading guide to the current
implementation (HAPI, the scheduler integration, device-to-device
transfers).
