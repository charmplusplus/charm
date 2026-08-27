# Size-based protocol switch for intra-node, cross-process GPU messaging

Add a second transport for `CkNcpyModeDevice::IPC` — a **direct IPC handle
exchange** that copies straight out of the sender's live buffer — and pick
between it and today's **staged** transport by payload size, with the threshold
set from an environment variable. Then measure the crossover with GPU-enabled
OSU latency/bandwidth benchmarks.

## Status: implemented

Everything below is built. Configure with `CHARM_GPU_IPC_THRESHOLD` (bytes,
`K`/`M`/`G` suffixes accepted) or `+gpuipcthreshold`; unset means always staged,
which is exactly the previous behaviour. Sweep with `scripts/ipc_crossover.sh`.

Verified: full Charm build, `verify` and `leanmd` (20 steps, two LB rounds,
energy conserved) on the local single-GPU multicore build, and both new
benchmarks build and refuse to run in the wrong process layout. **The direct
transport itself has not been executed** — the only complete local build is
`multicore`, which is one process, so nothing here can resolve to IPC. It needs
a cross-process run on Delta.

Four deviations from the plan as written:

1. **No callback gate (option B, not A).** Direct is chosen by size alone. The
   applications that relied on staging's implicit ordering are the ones that
   changed: `leanmd` now attaches a completion callback to every force send and
   drains the acknowledgements at the end of every step, not only on balancing
   steps. The runtime warns once per process when a direct send carries no
   callback. Still to convert, and unsafe above the threshold until they are:
   `jacobi3d` (6 sends), `pic2d` (4), `jacobi2d-imbalance` (4), `verify` (2 of
   3). `jacobi2d` was already correct.
2. **The import cache is not flushed at load balancing.** Closing a mapping
   while a copy is still reading it is an illegal access, and quiescing every
   device in the process to make it safe costs more than the problem is worth: a
   stale mapping leaks address space but cannot fault, because a reallocated
   buffer gets a fresh handle and the entry is simply never looked up again.
   `hapiIpcFlushImportCache()` exists and runs at exit.
3. **`CHARM_GPU_IPC_CACHE=0` synchronizes each receive** before closing the
   mapping, since closing an in-flight one faults. It prices open + close as an
   upper bound; it is not a transport option, and the sweep labels it as such.
4. **CUDA builds now link `-lcuda`.** `cuMemGetAddressRange` is the only way to
   find the base of the allocation holding an interior pointer, and a handle
   names an allocation, not an address. The stubs directory is appended last in
   `conv-mach-cuda.sh` so a driverless build host can still link without the
   stub winning over a real driver.

## What exists today

`findTransferModeDevice` ([conv-rdmadevice.C:7](src/conv-core/conv-rdmadevice.C#L7))
resolves a send to MEMCPY / IPC / RDMA from PE locality alone. Size never enters
the decision, and IPC has exactly one implementation.

**Staged (the current IPC path).** At init each process exports one IPC handle
per device — for the *whole* buddy-allocated device comm buffer — through POSIX
shared memory ([hapi_impl.cpp:1891](src/arch/cuda/hybridAPI/hapi_impl.cpp#L1891)),
and every other process opens it once
([hapi_impl.cpp:1966](src/arch/cuda/hybridAPI/hapi_impl.cpp#L1966)), landing in
`hapi_ipc_device_info::buffer`. Per transfer
([ckrdmadevice.C:1115](src/ck-core/ckrdmadevice.C#L1115) sending,
[:626](src/ck-core/ckrdmadevice.C#L626) receiving):

1. sender claims a comm-buffer block + an IPC event pair (`acquireIpcSendSlot`,
   [:880](src/ck-core/ckrdmadevice.C#L880)),
2. sender copies `src → comm_buffer[off]` (D2D, own device), records `src_event`,
3. message carries `{device_idx, comm_offset, event_idx}`,
4. receiver waits on the imported `src_event`, copies
   `peer_comm_buffer[off] → dst` (peer D2D), records `dst_event`, sets
   `dst_flag` in shm,
5. sender's next `reclaimCompletedIpcEvents` ([:778](src/ck-core/ckrdmadevice.C#L778))
   sees `dst_flag`, queries `dst_event`, frees the block and the slot.

**Two device copies, zero per-transfer handle operations.**

A direct-handle transport already exists, but only in the persistent API:
`CkDevicePersistent::open/get/put` ([ckrdmadevice.C:210-290](src/ck-core/ckrdmadevice.C#L210))
calls `hapiIpcGetMemHandle` on the user pointer, ships the handle in the pup, and
`hapiIpcOpenMemHandle`s it once on the peer (`ipc_open` latches it). That is the
shape to generalize — but it is a hand-managed, per-object arrangement with no
completion handshake, so none of it is directly reusable for the message path.

## The two protocols, and where a crossover comes from

| | staged | direct |
|---|---|---|
| device copies | 2 (`src→comm`, `comm→dst`) | 1 (`src→dst`, peer) |
| per-transfer handle ops | none | `IpcGetMemHandle` (sender), `IpcOpenMemHandle` (receiver) — both cacheable |
| comm-buffer pressure | one block per in-flight transfer | none |
| event-pool pressure | one pair | one pair (still needed) |
| sender may reuse `src` after | its own staging copy | receiver signals completion |

Cost model, `S` = payload, `B` = device copy bandwidth, `F` = fixed cost:

```
T_staged ≈ 2S/B + F_staged        (F_staged = buddy alloc + event slot + shm flag)
T_direct ≈  S/B + F_direct        (F_direct = amortized handle get + open)
crossover S* = (F_direct − F_staged) · B
```

The whole question is `F_direct`. A cold `cudaIpcOpenMemHandle` is ~100–500 µs;
at ~1.5 TB/s intra-device that puts `S*` in the hundreds of MB, i.e. staging
would win everywhere and the feature is pointless. **An import cache is
therefore not an optimization, it is the precondition** — with a warm cache
`F_direct` collapses to a hash lookup and `S*` drops into the range where the
extra staging copy actually matters (tens to hundreds of KB, machine-dependent).

So the benchmark must sweep **both** configurations: cache on (the steady-state
number that sets the shipping default) and cache off (which prices the handle
operations themselves, and answers the question the calibration idea in
`project-ipc-cost-switch` was really asking).

There is a second, non-performance motivation: a payload larger than the comm
buffer can never be staged. Today that send spins in `acquireIpcSendSlot` until
`CHARM_IPC_SLOT_TIMEOUT` and aborts. Direct-above-threshold gives large messages
a path that consumes no comm buffer at all.

## The one real hazard: source-buffer reuse

This is a semantic change, not just a performance knob, and it should be settled
before any code is written.

Staged is reuse-safe with no cooperation from the application: the sender's
staging copy is enqueued on `buffers[i]->hapi_stream`, so any later kernel the
app runs on that same stream is ordered after it. leanmd relies on exactly this
— `Compute::sendForces` hands out `CkDeviceBuffer(d_force)` with **no**
completion callback and calls `AtSync` immediately after
(see the comment at [ckrdmadevice.C:580](src/ck-core/ckrdmadevice.C#L580)).

Direct removes that decoupling: the receiver reads the sender's live buffer, and
nothing orders the sender's next kernel against a copy issued in another process.
The existing per-buffer completion callback (`CkDeviceBuffer(ptr, cb)`, delivered
through `CkRdmaDeviceRecvHandler`) is the correct signal, but it is optional
today and most call sites omit it.

Decision to make (my recommendation in bold):

- **A. Gate on the callback.** Use direct only when `cnt >= threshold` *and*
  `cb.type != CkCallback::ignore`; otherwise stage regardless of size. Safe by
  construction, needs no app changes anywhere, and the OSU benchmarks supply
  callbacks so the measurement is unaffected. Costs nothing except that
  callback-less senders never see the large-message win until they opt in.
- B. Threshold only, document the requirement. Simplest, and fine for a
  benchmark branch — but silently unsafe for leanmd-style senders if the
  threshold is ever set low enough to catch them.

Either way, log once per process when a send is downgraded from direct to staged
for lack of a callback, so a missing win is visible rather than mysterious.

## Design

### Wire format — `CmiDeviceBuffer` ([conv-rdmadevice.h:38](src/conv-core/conv-rdmadevice.h#L38))

```c++
enum class CmiIpcProtocol : char { NONE = 0, STAGED = 1, DIRECT = 2 };

CmiIpcProtocol      ipc_protocol;   // NONE unless the sender staged/exported
hapiIpcMemHandle_t  ipc_handle;     // DIRECT: handle for the *allocation*
size_t              ipc_offset;     // DIRECT: src_ptr − allocation base
```

Pup `ipc_protocol` unconditionally, and the handle + offset only under
`DIRECT` — the same pattern `data_stored` already uses at
[conv-rdmadevice.h:74](src/conv-core/conv-rdmadevice.h#L74). This keeps 64 bytes
off every staged descriptor. Conditional pup is compatible with the in-place
retarget in `CkRdmaDeviceIssueRgets`: it re-packs the same object with only
`ptr` changed, so the width is unchanged, and the existing
`patch.size() != field_len` abort ([:470](src/ck-core/ckrdmadevice.C#L470))
guards the invariant.

Replace the receiver's `sender_staged` test
([:587](src/ck-core/ckrdmadevice.C#L587)) with a switch on `ipc_protocol`, keeping
`device_idx != -1` as the compatibility condition for `STAGED`.

### Threshold

`CHARM_GPU_IPC_THRESHOLD` — bytes, accepting `K`/`M`/`G` suffixes, parsed once
into a `static const size_t` (the pattern used throughout this file):

- unset → `SIZE_MAX`, i.e. **always staged, today's behaviour exactly**. Nothing
  changes until the number is measured.
- `0` → always direct (the A/B arm for sweeps).
- `N` → direct when `cnt >= N`.

Also add `+gpuipcthreshold <bytes>` alongside `+gpucommbuffer` /
`+gpuipceventpool` ([hapi_impl.cpp:1326](src/arch/cuda/hybridAPI/hapi_impl.cpp#L1326))
for consistency, with the env var winning. Print the resolved value from PE 0
next to the other `HAPI>` lines.

`cnt` is set by the generated marshalling code immediately before
`CkRdmaDeviceOnSender` ([xi-Parameter.C:456](src/xlat-i/xi-Parameter.C#L456)), so
the size is available at the decision point. Decide **per buffer**, not per
message — a multi-buffer entry method can legitimately mix protocols.

### Export cache (sender)

`cudaIpcGetMemHandle` wants the base of the allocation, and the receiver's opened
pointer is that base — so carry an offset:

```c++
CUdeviceptr base; size_t sz;
cuMemGetAddressRange(&base, &sz, (CUdeviceptr)ptr);   // cuda.h already included
hapiIpcGetMemHandle(&h, (void*)base);
ipc_offset = (char*)ptr − (char*)base;
```

Cache `base → handle` in a process-wide map. `cuMemGetAddressRange` is a cheap
driver query but not free, and app buffers are long-lived, so the hit rate is
essentially 1 after the first iteration.

### Import cache (receiver) — the load-bearing piece

Process-wide map, `handle bytes → mapped base pointer`, with a `CmiNodeLock`.
Mirrors what `ipcHandleOpen` already does for comm buffers, so put it in
`GPUManager` next to `hapi_ipc_device_infos`
([gpumanager.h:55](src/arch/cuda/hybridAPI/gpumanager.h#L55)).

- Key on the full 64-byte handle (hash + `memcmp`). `cudaIpcOpenMemHandle`
  **fails with `cudaErrorAlreadyMapped` if the same handle is opened twice in a
  process without closing**, so the cache is required for correctness, not just
  speed.
- Follow `ipcHandleOpen`'s existing practice of a single process-wide mapping
  used from every PE regardless of which device each PE drives — that works
  today because P2P is enabled between all devices on the host
  ([hapi_impl.cpp:1408](src/arch/cuda/hybridAPI/hapi_impl.cpp#L1408)). If a
  mixed-device process ever misbehaves, key the cache on
  `(handle, importing device)` rather than switching devices around the open —
  device switching around `cudaIpcOpenEventHandle` is the wrong fix that cost
  hours before (see `project-multigpu-working-state`).
- **Staleness is the known hole.** If the exporter frees the allocation, every
  cached import of it dangles, and migration is precisely when device buffers
  get torn down and rebuilt. For this branch: never close on the fast path,
  and flush the whole import cache at the LB resume barrier, where cross-process
  reallocation actually happens. Add `CHARM_GPU_IPC_CACHE=0` to force
  open/close per transfer — slow but always correct, and it doubles as the
  cache-off arm of the measurement.

### Event slots and reclaim

Direct still needs the event pair — `src_event` orders the receiver's copy after
the sender's producing kernels, `dst_event` + `dst_flag` tell the sender the
buffer is free. Only the comm-buffer block goes away:

- split `acquireIpcSendSlot` ([:880](src/ck-core/ckrdmadevice.C#L880)) into
  "claim event only" and "claim event + block"; direct calls the former, so it
  cannot block on comm-buffer exhaustion,
- extend `event_pool_flags` with `2 = claimed, no block to free`.
  `reclaimCompletedIpcEvents` ([:778](src/ck-core/ckrdmadevice.C#L778)) currently
  aborts on any flag other than 1, and offset `0` is a legal block, so a
  sentinel offset will not do — it has to be the flag.

### Cross-protocol fallbacks

Both must keep working, since the delivery PE is only known on arrival:

- **direct message, delivered same-process** (target migrated into the sender's
  process): take the plain `src → dst` copy from `source.ptr`, then still record
  `dst_event` and set `dst_flag` so the sender reclaims the slot — exactly the
  shape of the existing MEMCPY-with-staging-anyway branch
  ([:604](src/ck-core/ckrdmadevice.C#L604)).
- **unconfirmed destination** (`dest_pe == -1`,
  [:1015](src/ck-core/ckrdmadevice.C#L1015)): stays as it is — stage IPC. Do not
  add direct here; its correctness depends on a completion signal the sender
  cannot reason about when it does not know where the message is going.

### Stats

Extend the `CHARM_ZC_STATS` block ([:336](src/ck-core/ckrdmadevice.C#L336)) with
staged/direct counts plus import-cache hits/misses, so a sweep can confirm the
switch actually fired at each size instead of inferring it from timings.

## Implementation order

Each step builds and is testable on its own; `examples/charm++/cuda/gpudirect/verify`
(2 processes, `+gpushm`) is the correctness gate at every step.

1. **Wire format + threshold plumbing.** `ipc_protocol`/`ipc_handle`/`ipc_offset`,
   conditional pup, env var + `+gpuipcthreshold` parsing, receiver switches on
   the protocol field. Sender still always sets `STAGED`. Behaviour unchanged;
   verify must pass byte-identically.
2. **Export + import caches.** Standalone, with a unit test in
   `tests/charm++/cuda/` that exports a buffer, imports it from a peer process
   twice, and asserts one cache miss and one hit.
3. **Direct send/receive path.** Event-only slot claim, flag value 2, receiver's
   handle→pointer resolve + single peer copy. Run verify with
   `CHARM_GPU_IPC_THRESHOLD=0` (all direct).
4. **Fallbacks + gating.** Same-process delivery of a direct message; the
   callback gate from option A above. Run the jacobi2d/leanmd migration cases
   with the threshold at 0 to exercise the delivery-mismatch paths.
5. **Stats + `CHARM_GPU_IPC_CACHE=0`.**
6. **Benchmarks** (below).
7. **Pick and document defaults** per machine once measured.

## Benchmarks

`examples/charm++/osu_latency` and `osu_bw` are host-message benchmarks
(`message LatencyMsg { char data[]; }`) — they never touch a device buffer, so
they cannot measure this path. Add GPU variants under
`examples/charm++/cuda/gpudirect/osu/`, modelled on `verify` and `jacobi2d`:

- **`osu_latency_gpu`** — two chares, `cudaMalloc`'d buffers, ping-pong over
  `entry void ping(int n, nocopydevice char buf[n])` with a `CkDeviceBufferPost`
  handler, half-round-trip timing, per-size skip/iteration counts.
- **`osu_bw_gpu`** — windowed sends (`-w`, default 64) plus an ack, same size
  sweep.
- Both: `-s`/`-e` size range (default 8 B → 64 MB, doubling), `-i` iterations,
  and a completion callback on every `CkDeviceBuffer` (required by the option-A
  gate, and honest — a real app must handle completion to reuse its buffer).

**Placement is the whole experiment.** The two endpoints must land in different
processes on the same physical node, or `findTransferModeDevice` returns MEMCPY
and the run silently measures nothing. Insert the two elements at explicitly
chosen PEs (`-peer <pe>`, default = first PE of process 1) and have the
benchmark abort at startup unless `findTransferModeDevice(pe0, peer)` is `IPC`.
Launch as 2 processes × N PEs on one node with `+gpushm`.

**Sweep driver** — `scripts/ipc_crossover.sh`: for each of
`{threshold=SIZE_MAX (staged), threshold=0 (direct, cache on),
threshold=0 + CHARM_GPU_IPC_CACHE=0 (direct, cache off)}`, run both benchmarks
over the size sweep and emit one CSV (`machine, protocol, cache, size,
latency_us, bw_MBps`). The crossover is where the staged and direct-cache-on
curves intersect; round down to a power of two for the default.

Also worth capturing, since it is the number the model turns on: a standalone
timing of `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle`
on the target machine. The cache-off arm gives this implicitly, but a direct
measurement makes the per-machine defaults defensible in writing.

Machines: local dGPU first for correctness (single GPU type — cross-device
paths stay unexercised, see `project-local-machine-limits`), then Delta A40s for
the real numbers, with the flag set from `project-multigpu-working-state`.

## Out of scope

- Migration (`DeviceMigrationStrategy`). The same import cache is what a staged
  vs. IPC-pull switch would need there, so build it to be reusable — but the
  routing decision is a separate change with a different safety argument.
- MEMCPY and RDMA modes.
- `CkDevicePersistent`. It already does direct handle exchange with a manual
  latch; folding it onto the shared import cache is a follow-up cleanup, not
  part of this.
- Automatic threshold calibration at startup. Get the measured numbers first;
  auto-calibration is only worth it if the per-machine values turn out to
  scatter more than a factor of two.
