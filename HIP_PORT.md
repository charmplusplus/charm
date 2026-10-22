# HIP / CUDA portability of the GPU runtime (2026-10-21)

The GPU runtime (hybridAPI, the device paths of ck-core, the balancers' device loads) builds for
NVIDIA (CUDA) or AMD (HIP/ROCm) from one source. This note records the design, how each tree is
built, and what is still vendor-specific.

## Design

- **`CMK_GPU`** (conv-config.h, and conv-mach-{cuda,hip}.h so that interface files run through
  charmxi see it; cmake/ci-files.cmake passes `-DCMK_GPU=1` to charmxi) is the guard for every
  vendor-neutral GPU path in the runtime (ck-core, ck-ldb, util, conv-core). `CMK_CUDA` and `CMK_HIP`
  appear only where the vendors differ: hapi_portable.h, the driver-level blocks of hapi_impl.cpp, the
  trace backends and the range markers. Both vendor macros are always defined (0 or 1): test them
  with `#if`, never `#ifdef`.
- **`hapi_portable.h`** is the vendor table: every runtime call the runtime makes has a `hapi*` name
  defined for both vendors (streams, events, copies, IPC handles, pointer attributes, device
  attributes, error codes), plus inline driver-level wrappers `hapiDrv*` taking integer device
  addresses (virtual-memory reservation, chunk creation, mapping, access, export/import, granularity,
  device probes), which hide CUDA's integer `CUdeviceptr` against HIP's pointer `hipDeviceptr_t`.
  Application code includes it through hapi.h; the three validation applications (chantest, leanmd,
  jacobi3d) use the `hapi*` names and compile their kernels with nvcc or `hipcc -x hip` as the
  Makefile reads `BUILD_HIP` from the tree's conv-mach-opt.mak.
- **Completion flags**: the pinned-flag writer is `cuMemsetD32Async` / `cuStreamWriteValue32` looked up
  as driver entry points on CUDA, and `hipMemsetD32Async` / `hipStreamWriteValue32` behind adapters
  with the same signatures on HIP (hapi_impl.cpp).
- **Trace backend** (`hapi_trace_backend.h`): the balancer's per-object device loads are built from
  three neutral record kinds (kernel, API call, copy). On CUDA, CUPTI's activity records are walked
  and converted in hapi_impl.cpp (`hapiTraceForEachRecord`); on HIP, `hapi_trace_rocprof.cpp`
  registers with rocprofiler-sdk before the first HIP call (`hapiTraceBackendEarlyInit` from
  `hapiPresetEnvironment`), traces kernel dispatches and the HIP runtime API without the polling
  calls, learns kernel names and register counts from the code-object callback, and queues one
  converted batch per buffer callback into the same queue the drain reads.
- **Range markers**: NVTX on CUDA, roctx on HIP (hapi_nvtx.h for the runtime's ranges, cknvtx.C for the
  entry-method ranges, through a small shim).
- **Device kernels of the runtime** (hapi_kernels.cu): one source, built by nvcc or `hipcc -x hip`
  (CMakeLists.txt); the warp shuffle has a per-vendor spelling.
- **Hardware queues**: the runtime presets `CUDA_DEVICE_MAX_CONNECTIONS=32` on CUDA; on HIP the
  variable is `GPU_MAX_HW_QUEUES` and the runtime leaves the driver's default until measured there.

## Building

CUDA tree (unchanged): `multicore-linux-x86_64-cuda`, `./build charm++ multicore-linux-x86_64 cuda
--with-production --with-lbuserdata ...`.

HIP tree: `./build charm++ multicore-linux-x86_64 hip --with-production --with-lbuserdata
--with-fetch-reconverse-dir=<reconverse clone> -DCMAKE_BUILD_TYPE=Release`, then in the tree
`cmake . -DLCI_DIR=<LCI HIP install>/lib64/cmake -DLCI_ROOT=<LCI HIP install> -DLCI_USE_HIP=ON
-DROCM_PATH=/opt/rocm -DCMAKE_HIP_ARCHITECTURES="gfx908;gfx90a"` and `make`. Two things to know:
`./build` forwards every `-D` it is given to the compiler flags as well (a value with a semicolon
becomes a stray linker input), so the LCI and ROCm cache entries go to cmake directly; and on a
Cray PE the compiler wrappers inject `-lcuda -lcudart -lmpi_gtl_cuda` while `craype-accel-nvidia80`
and `cudatoolkit` are loaded, so LCI (`-DLCI_USE_HIP=ON`, tests off) and the HIP tree must be
built with those modules unloaded (`craype-accel-host`), or every executable needs libcuda on the
AMD node. LCI's OFI backend needed one fix for libfabric 2.3.1 (no `device.rocr` field; FI_HMEM_ROCR
takes no device).

## Still vendor-specific or untested

- `cudaMemcpyBatchAsync` (the batched channel puts and receive landings) is CUDA 12.8+; HIP takes the
  per-copy fallback. `hipMemcpyBatchAsync` exists and could be wired the same way.
- hapi_memory_daemon (shrink/expand) is built on CUDA only.
- tests/util/streamorder and benchmarks/charm++/cuda/gpudirect/* still spell CUDA names.
- rocprofiler attribution, the virtual-memory pool and IPC on AMD, and cross-node device transfers
  through LCI's HIP path are validated only as far as the MI100 run below says.

## Validation, 2026-10-21

**A40 (job 23078963, gpub001, the ported CUDA tree):** chantest 8 of 8 (memcpy and ipc transports: plain, lb, lba,
and both lbx aborts as required). leanmd 16x8x8 no LB, 30 steps: channels 2583.3 vs zero-copy 2537.0 ms/step
(mean 11-30; last night 2571.6 vs 2587.5 on gpub002). jacobi3d 512^3: channels 1414.7 us/iteration (last night
1416.2), zero-copy 1687.3, checksums 8.762191026712e+06 on both, identical to last night's. leanmd 8x4x4 async:
channels 241.1 / 229.4 / 238.4 vs zero-copy 204.6 / 200.9 -- but the same-node A/B against the pre-port binary
(HEAD sources built from a stash, CHARM_GPU_LOAD_AUDIT=1) read pre-port 223.6 and 248.1 vs ported 215.6, with
the audit's unowned share of device time 0.66 on both and 959/873 vs 814 moves: the band is the node tonight,
not the port. `[chan-stats] put batches` now prints (125,184 batches for 414,720 puts, 3.3 per batch).
Found and fixed on the way: the hapi shared-memory segment `charm-hapi-host<id>` was left behind by an earlier
kill on the node and hung every two-process run before its first channel use (both PEs idle, no `[CHAN-INIT]`
line); it is now named per job and step like the channel segment.
Observation for later: at 8x4x4 async the load audit charges two thirds of the device's busy time to no object
on both builds (unowned_frac 0.66, occupancy 0.65-0.71).

**MI100 (job 23104951, gpud01, 13:38-13:46, the HIP tree):** chantest's memcpy-transport arms pass (24 iterations, with and
without a balancer). Its ipc-transport arms (two processes sharing device memory through the pool) aborted at the first
pool import: `vmm import: cuMemSetAccess(chunk 0) from device N: hipErrorInvalidValue (peer access between the devices is
required)`. HIP maps another device's physical memory into a device's address space only once peer access from that
device to the memory's device is enabled, which CUDA's virtual-memory path never needed. The pool's shared-memory entry
now carries the owner's device, and the import enables peer access from each local device to it before the mapping; a
device that cannot reach it is left out of the grant. leanmd 4x4x4 in one process: step 1 took 5.3-5.5 s, then 50-65
ms/step; the no-LB run made no step after step 6 (240 s timeout, no stacks taken), async LB ran 30 steps in 12 s, async
with channels was at step 22 when the allocation ended. 8x4x4 and jacobi3d did not run. The second pass with the fix
(the same arms, plus every process's stacks dumped after 90 s without a step) waits on the node, drained since 17:40.

## Frontier (MI250X, ROCm 7.14.1, Slingshot-11), 2026-10-22

Clones, installs, scripts and every log are under `/lustre/orion/csc710/scratch/adityapb/hip`: `env.sh`
(modules; do NOT load `craype-accel-amd-gfx90a`, the wrappers then link `libmpi_gtl_hsa`, which needs the
ROCm 6 `libamdhip64.so.6` beside ROCm 7's), `lci_install_hip` (UIUC-PPL/lci master 662b4736, HIP, ofi, no
CUDA), `charm-reconverse/multicore-linux-x86_64-hip` with the three applications built in `apps/`,
`irun.sh`/`wrun.sh` (a run on a `salloc --no-shell` allocation, `wrun` with a stall catcher that dumps every
process's stacks), `frontier_wrap.sh` (GCD-to-L3 pinning, the region's first core left to the OS),
`frontier_chain1.sh` (the batch chain), `vmmtest/` (standalone HIP repros of the driver findings below).

**Build corrections to the handover.** `./build` takes `--enable-lbuserdata` (the `--with-` spelling and
`-DCMAKE_BUILD_TYPE` are passed to the compiler as flags), needs `CC=cc CXX=CC FC=ftn` or it builds with the
system GCC 7.5, and `charmc` then calls bare `clang`, so `$ROCM_PATH/lib/llvm/bin` must be on PATH. The tree's
cmake needs `-DRECONVERSE_ENABLE_HIP=ON`, else reconverse builds with `CMK_HIP 0`. The first `./build` leaves
an autofetched non-HIP `liblci.so`/`liblct.so`/`liblci-ucx.so` in the tree's `lib/`, which shadow the HIP LCI
at run time (moved to `lib.stale/`). Clang rejects the `void*` arithmetic in `buddy_allocator.cpp:195`.
A one-node `srun` step needs `--network=single_node_vni` or cxi finds no VNI and LCI aborts in `fi_domain`.
Multi-process runs need `+gpudevs 1` for one GCD per process (the runtime otherwise takes every GCD it sees).

**Driver findings, each with a repro in `vmmtest/`.**
- `hipMemsetD32Async` as the completion-flag writer orders nothing at system scope: a kernel's or a blit
  copy's writes issued before it can still be in the device's L2 when the flag lands, and a peer device or
  another process reading on the flag reads stale data (32 KB fill on device 0, flag, copy from a device-1
  stream: stale 200 of 200; chantest's ipc puts and +LBAsync snapshots failed on it). An event recorded
  ahead of the memset carries the release (`hipEventDisableSystemFence` brings the staleness back), so the
  HIP writer records one per device before every flag (`hapiHipMemsetD32Async`). The memset writer stays the
  default: the front-end `StreamWriteValue32` deadlock under IPC is the same on HIP.
- ROCm resolves an imported `hipMemCreate` descriptor at `hipMemSetAccess`: the fd must stay open until the
  grant (`hipErrorInvalidValue` otherwise); the MI100 "peer access" diagnosis was this. Fixed in
  `hapiVmmImportLocked`; the pool's shared-memory entry now also carries the physical ordinal.
- ROCr cannot dma-buf-export `hipMemCreate` memory (`HSA_STATUS_ERROR_INVALID_ALLOCATION`), so the cxi
  provider cannot register the vmm pool: across nodes the pool must be `+gpupoolalloc buddy`, and NIC-registered
  arenas are capped at 2048 MB per arena (`+gpupoolsize 2048`).

**Runtime bugs found here, in shared code.**
- The channel protocol stalled at the step after a migration in half the two-node runs (and some one-node
  ones): the owner's DONE, Resolved and credit messages reach a sender in any order (timestamped: a DONE sent
  before the reply delivered after it, a credit sent after both delivered first), and `onResolved` reset the
  credit to `free ? 1 : 0`, which folded an early credit into the free flag or let a staged read's debit land
  after the reset. The reply now carries the last put that entered the buffer (taken, pulled, or landed
  before the pack), and the sender reconciles its in-flight direct puts, the DONEs seen while unresolved and
  the early credits against it by sequence number (`acct_seq`, `direct_issued`, `dones_seen`,
  `creditedPut`). Validated: chantest 8 of 8 on one node and 4 of 4 across two, leanmd 8x4x4 async channels
  60 steps, four two-node runs in a row with no watchdog line (31-35 ms/step).
- `hapiCuptiLanePrune` read an empty lane (segfault at the second async LB step on HIP, where the tag lane
  is never written).

**Profiler.** rocprofiler-sdk 1.3.2's inline queue interposition holds a queue's `gate_lock` in
`process_doorbell_impl` while `ring_buffer_writer` spins for space in a full hardware queue; with the queue's
head a barrier on another process's IPC event and that process's PEs blocked on the same lock, every
multi-process leanmd run hung within four steps (`CHARM_LB_NO_CUPTI=1` runs without device loads). The deeper cause is in the same file: the inline
interposition's async signal handler parks one pool thread in an active wait per in-flight kernel, and the
pool is `ROCPROFILER_ASYNC_SIGNAL_HANDLER_THREADS` (default `GPU_MAX_HW_QUEUES`, else 4), so a barrier on
another process's event starves it. Measured (leanmd 8x4x4, four processes, no LB, tracing on): a 128K-packet
AQL queue, 4 steps then the hang; 32 hardware queues, 3; a 64-thread pool, 2; the legacy intercept queue
(`ROCPROFILER_QUEUE_INTERPOSITION=false`, which still traces every dispatch) 60 steps at 29.5 / 29.8 ms/step
against 29.1 / 29.0 without tracing. The HIP trace backend now presets that variable before it registers
(`hapiTraceBackendEarlyInit`, the environment's own value wins). Source: rocm-systems
`gpuep-releases/therock-7.14`, `queue_interposition.cpp` (a copy in `ext/qi_714.cpp`).

**Batched copies.** `hipMemcpyBatchAsync` (ROCm 7.14, the 12.8 signature with the failure index, one attribute
per batch) is wired under `CMK_HIP` at both batch sites. leanmd 8x4x4 async channels on one node: 29,826
batches for 102,860 puts with 0 refusals and 0 fallbacks, against 27,126 refusals (every batch falling back
to per-copy issue) before; 31.2 / 34.8 vs 32.0 / 37.8 ms/step (steps 11-30 / 31-60, single runs).

**Performance, two nodes, eight processes of one GCD and 7 PEs (`perf_matrix.sh`, single runs, no LB,
no tracing, buddy pool).** leanmd 8x4x4 ms/step (11-30 / 31-60): 4 hardware queues 19.2 / 19.4, 8 queues
19.2 / 19.6, 16 queues 20.3 / 20.3; `+gpustreamwrite` 18.7 / 18.8; `FI_MR_CACHE_MONITOR=kdreg2` 18.8 / 18.9.
jacobi3d 512^3 (32 blocks) channels 793 us/iteration (4 queues), 828 (16 queues); zero-copy 929; checksum
8.762191026712e+06 throughout. The driver default of 4 queues is the right one; the flag writer and the MR
cache monitor are within noise of each other (kdreg2 is available here and costs nothing, uffd is what the
scripts set). With four processes in total (two per node) the same leanmd arm runs at 28.6-31 ms/step.

**Validation summary (2026-10-22, the final binaries).** One node: chantest 8 of 8 (memcpy and ipc transports,
plain/lb/lba, both lbx aborts as required); leanmd 4x4x4 in one process 30 steps (no LB, async, async channels);
leanmd 8x4x4 in four processes of one GCD, 60 steps: no LB 30.1 / 30.2 ms/step, async 40.6 / 31.5, async channels
31.2-34.5 / 34.8-41.1 (vmm pool), tracing on with the legacy intercept queue 29.5 / 29.8 (no LB); jacobi3d 512^3
channels 1403 us/iteration, zero-copy 1475, checksum 8.762191026712e+06. Two nodes (buddy pool, 2048 MB arenas):
chantest 4 of 4; leanmd 8x4x4 four processes no LB 28.6 / 28.8, async 29.3 / 28.8, async channels four runs in a
row 31-35 / 33-40 with no watchdog line, and with tracing on async 36.8 / 35.6 and async channels 30.9 / 44.8, the
balancer seeing device loads of 44-64% of the interval; jacobi3d 16 blocks channels 1447 us, 32 blocks (eight
processes) channels 793, zero-copy 929, checksums identical. The load audit (`CHARM_GPU_LOAD_AUDIT=1`, 8x4x4 async,
one node, tracing on) read util 0.67-0.95 per GCD and an unowned share of 0.76-0.81; the audit now names the
unowned kernels (`[gpu-audit]   unowned <s> <kernel>`, the kernel key kept on unattributed sweep entries), and
they are leanmd's own `pairForceBothKernel`: objects that have joined the balancer's step are left unattributed
for the rest of the round by design, which with `-lblag 8` on a 10-step period is most of it. With `-lblag 2`
the share is 0.128. Attribution itself works on HIP (the A40's 0.66 was another lag). The HIP runtime's blit
kernels (`__amd_rocclr_fillBuffer*`, `__amd_rocclr_copyBuffer*`: every completion flag and device copy, 19k
of the 24k dispatches per window) are now delivered as copy records, as CUDA's copy engine reports them.
Kokkos: `leanmd.kokkos` against the gfx90a Kokkos install runs 8x4x4 no LB at 30.6 / 31.1 ms/step, the
hand-written kernels' 30.1 (the Makefile wants `hapi_kokkos.h` and `hapi_blas.h` in `apps/`). The relinked
binaries with nothing profiler-related in the environment: leanmd 8x4x4 async traced 33.0 / 30.6 ms/step with
device loads at 54% of the interval, chantest ipc/lba pass.

**sph3d and moe.** Their hand-written kernel targets now build on HIP: the Makefiles take leanmd's device-compiler
branch (hipcc, `-ffast-math`; moe links `-lhipblas` from `$ROCM_PATH`), the host code uses the `hapi*` names
(`hapiEventElapsedTime` and `hapiMemset` added to the vendor table) and the BLAS shim's names
(`hapiBlasSetMathModeTf32`, `HAPI_BLAS_OP_*`, `hapiBlasStatusSuccess` added), sph3d's CUB comes from hipCUB
behind clang's `__HIP__` (CUB must precede converse.h, so no `CMK_` name exists there yet), and `ck` is built
with `CMK_NVTX=1` on HIP as on CUDA, so cknvtx.C's roctx branch provides the `+gputrace` hooks sph3d links
against. sph3d ran its defaults (2.3 M particles) for 40 iterations in four processes at 11.4 ms/iteration; moe builds (not run: it needs its model
arguments). On CUDA every renamed call expands to the identical CUDA name.

Open on Frontier: the vmm pool cannot be NIC-registered (ROCr dma-buf export), so two-node runs take the buddy pool;
`hapi_memory_daemon` is still CUDA-only; `tests/util/streamorder` and `benchmarks/charm++/cuda/gpudirect/*` still
spell CUDA names. Every number above is a single run. Nothing here is committed.

**The balancer on Frontier (2026-10-22, 16x8x8, `-computemap local -density gradient`, 7 PEs per process, one
GCD each, period 20, lag 16, device-load tracing on; `lb_experiment.sh` and the chained 200-step runs).**
Two things had to be true before it could act. The memory contract: with the one 2048 MB buddy arena the
cross-node registration cap allows, every node advertised 0 MB (free 602 MB below the 614 MB reserve) and no
object could move; two arenas (`+gpupoolarenas 2`, 4096 MB per process) advertise ~1.4 GB, and on one node the
8 GB vmm pool ~3.5 GB. The transfer-cost table: the built-in one is Delta's A40s, and MetisLB priced its first
plan on Frontier as a net loss (-13%); `lbcalib` (ported, `apps/lbcalib`) wrote `lbcost.frontier.conf` in 10 s
on two nodes (inter_node alpha 26.8 us, beta 41 ps/B, R2 0.9999; migration alpha 9.4 us; the tiers a one-GCD
layout does not exercise were filled from adjacent ones), passed with `+LBCostConfig`. With both, 200 steps,
ms/step over steps 11-30 / 61-100 / 121-200: two nodes (8 processes) no LB 134.2 / 134.3 / 134.9, async
132.7 / 167.4 / 115.3 (10,008 moves over the run; 14.5% faster once converged); one node (4 processes) no LB
247.3 / 250.1 / 250.7, async 257.9 / 253.4 / 199.5 (5,601 moves; 20% faster). The channels transport under
the balancer: one node 250.7 (no gain, 7,377 moves); two nodes failed in three of four 200-step runs, two
ways. (a) "the header of a put with signal never arrived": a sender answered twice after the owner's move
(the pushed reply and its own request, 10 us apart) leaves the first reply's signal slot stashed and the
second's current; `onCloseAck` released one or the other (`else if`), so the current slot stayed referenced,
the next reply's `retireSignalSlot` freed it blind while the acked put's note was still on its way, the LIFO
free list handed the same slot back to the same channel, and the allocation's "start clean" wiped the header
that had landed 11 ms earlier (traced with `CHARM_DEBUG_SIGNAL=1`, the new lean trace of the cross-host
signal path: slots, headers, notes, resolutions, closings). Both slots are released at the ack now, the
current one deferred by the acked sequence as designed. (b) A credit lost after an intra-process handoff
(the sender resolved with no credit while the owner had released its last put): the reply's taken_seq was
raised to `pull_ev.seq` whenever the buffer was busy without a taken event, but a direct put that landed
before the pack leaves the owner in that state too, and an owner that never read a staged put sent
`pull_ev`'s uninitialised seq (a pointer-sized number), so the sender's reconciliation discarded the early
credit it was holding for that put. Now `landed_unseen` says which case it is. After both fixes, six
two-node channels runs of 200 steps in a row completed with no watchdog line. The pool: cxi registers at most
2048 MB per arena (4096 and 8192 fail in LCI's registration), but four 2048 MB arenas per process register and
run, so Delta's per-process total does not apply here (`CK_GPU_ARENA_TOTAL_MAX_MB=16384`, now in env.sh,
with `+gpupoolarenas 4`). With that 8 GB the channels arm can move (15,514 moves) and ends at 146.0 ms/step,
8% slower than no LB, while zero-copy ends at 121.1, 10% faster: under this balancer the channels transport
loses what the moves gain, on one node as across two. Repeated (two nodes, 200 steps, steps 121-200, every run
with the 8 GB pool and the Frontier table): no LB 132.9 / 134.5 / 136.1 / 134.1 / 134.9; async zero-copy 123.6 /
106.4 / 117.2 / 121.1 / 115.3 / 126.9 (mean 118, 12% under no LB); async channels 148.8 / 160.8 / 120.8 / 146.0
(mean 144, 7% over no LB), with one channels run killed at step 145 and one zero-copy run at step 79 by a
segmentation fault in one rank after migrations (under investigation). The balancer's own migration tallies
(`CHARM_LB_MIGSTATS=1`) say where the channels' cost sits: per move, stand-down (closings acked) 26 ms against
0, ship (payload dispatched to the destination's ack) 97.5 ms against 6.0, land 1.7 against 0.07 ms, and a
channel is held closed for 167 ms on average (max 1.2 s) between its closing and its new resolution, during
which the sender's puts take the staged path (28 us each against 0.5 us resolved). The Compute's pup ships
less with channels (live positions only, bounded by `unseenBytes()`), so the bytes are not the cost; the
payload waits at the destination. A 40-second step was also seen once and not traced. The 41-60
window pays for the first migrations (193 ms/step at 100 steps). Launch-overhead knobs on the no-LB arm,
one node: `HIP_FORCE_DEV_KERNARG=1` 250.7 vs 252.7, `+gpusubmit` 252.0, both 258.7: nothing to take.

**Why the channels transport loses under the balancer (2026-10-22 afternoon; `chan_diag.sh`, `maxsends.sh`,
`maxsends2.sh`, `chain3.sh`, all two nodes / 8 processes unless said, `CHARM_ZC_STATS=1 CHARM_LB_MIGSTATS=1`).**
Per step the channels are not the problem: without the balancer they run 132.6 against zero-copy's 134.1 ms/step,
and a same-host put costs 0.5 us of host time. The loss is in the moves. The payload's "ship" (dispatched to the
destination's ack) was 20-113 ms per process against 2-13, and the cross-node rget that carries it averaged
64.5 ms posted-to-done for ~105 KB (88 us in the zero-copy arm), 205 ms in a later run. The reason is LCI's
send queue: reconverse builds LCI with `LCI_BACKEND_MAX_SENDS_DEFAULT` 64 per device, every cross-host channel
put is three network operations (the header's raw put, the data put, and the zero-length message LCI emulates
immediate data with), the posts came back `retry_nomem` ("network queue full") 11-13k times per PE with single
posts of 6-8 ms (20 ms at worst), and the rget waited in the device's backlog behind them. A deeper queue
(`LCI_ATTR_NET_MAX_SENDS`, cxi's own tx size is 1024) removes the nomem retries but not the backlog: 1024 gave
132.6 ms/step, rget 62 ms, hold 262 ms; 4096 (with `FI_CXI_DEFAULT_TX_SIZE`) rget 41 ms, 135.5 ms/step --
because the one device's try-lock is contended by seven PEs (330-500k `retry_lock` per process) and a failed
try-lock also parks the operation in the backlog. One LCI device per PE (`+lci_ndevices 7`) is the fix for the
contention: zero retries of any kind, and the rget falls to 0.4-1 ms. The registration caches, by contrast,
are not where the time is: LCI's own cache (`lci_install_hip_rc`, built with `LCI_USE_REG_CACHE=ON` plus a
bypass of the UCX rcache for device buffers in `device_inline.hpp`, which it cannot register) took the
other-host put from 40.8 to 37 us of host time, both caches 36 us, the steps unchanged.

With per-PE devices the two transports stand (steps 121-200): zero-copy 110.8 ms/step at the default 64 sends
(cross-node rget 28 us; 118.1 in an earlier run) -- the best Frontier figure, 17% under no LB -- against 138.3
at 256 sends after a 17.6k-move burst and 270 at 1024 (the deeper per-device queue halves the same-process
copy rate: MEMCPY receives 1.3 ms against 0.6); channels 146.7 at 64, 132.1 at 256, 120.4 at 1024 (that run
stalled at step 170, below). So `+lci_ndevices 7` and the default queue depth; `LCI_ATTR_NET_MAX_SENDS` is
not a knob to raise. What the channels still pay with the contention gone is the protocol itself, measured per
move: the stand-down (the owner cannot pack until every sender's put in flight has landed and acked the
closing, and a put stays in flight until the GPU runs its copy behind the step's kernels) 45-54 ms across
nodes and 154 ms on one node; the hold (closing received to the new resolution, during which the sender's
puts go staged) 146-181 ms across nodes, 344 on one; and the ship 63-157 ms even with a 0.4 ms rget, so the
rest of that window -- the payload message's wait for its pack copies, the destination's scheduling -- is
where a `ship-wait` tally now splits it (`[mig-window] ship-wait`, next run). On one node the IPC payload copy
itself takes 3.5 ms (0.6 in the zero-copy arm). The zero-copy arm has none of this because a message's landing
buffer is allocated at the receiver when the data is ready, so the receiving element is free to move at any
time; a channel binds the destination buffer ahead.

**The intermittent segmentation fault, caught.** An `LD_PRELOAD` crash catcher (`ext/segvbt.so`; `irun.sh`
appends it, `SEGVBT=0` disables) attaches gdb to the faulting process and leaves every thread's stack in
`logs/int/<tag>/crash_<host>_<pid>.txt`. The first catch (step 5, before any migration, so not the moves): the
HSA async-events thread in rocprofiler-sdk 1.3.2's `pool<signal_t>::release()` dereferencing null while a PE
thread in `WriteInterceptor` grew the signal pool. `release()` reads the pool's `stable_vector` (a vector of
chunk pointers that reallocates on growth) holding only the availability mutex, `acquire()` grows it under the
pool mutex: a plain data race, still in upstream `develop`. The pool grows by 4096 signals (a constexpr)
whenever every signal is in flight, and a probe (`ROCPROFILER_LOG_LEVEL=info`) shows it growing 6-65 times per
rank in a run, to 270k signals in a burst over twelve steps -- ROCr's single async-events thread waits on every
registered signal at once and falls behind, so signals pile up. Each growth is one race window, which is why
the crash is rare and comes at ramp-up or after a migration burst. The two-line fix (take `m_pool_mtx` in
`release()`, same order as `acquire()`) is `ext/rocprofiler_pool_release_lock.patch`, applied to a sparse
clone of rocm-systems `gpuep-releases/therock-7.14` in `ext/rocm-systems`, built with ROCm's clang (the login
node's g++ has no `<filesystem>`; libdw from the spack elfutils 0.194) into `ext/rocprofiler_patched`
(`ext/rocm-systems/build2`), the same 77 public `rocprofiler_*` symbols as ROCm's 1.3.2. `irun.sh` puts it
ahead of ROCm's copy (`ROCPROF_PATCHED=0` for the stock one); rocprofiler's registration log names the path it
loaded. Verified: four traced runs (three channel, one zero-copy) through 60, 160, 78 and 3 pool growths -- each
a race window before -- with no crash.

**The credit stall on the ipc tier.** Five traced channel runs (`CHARM_DEBUG_SIGNAL=1`, 3.8 GB of log each)
completed clean; the untraced per-PE-device run stalled at step 170 in the same shape as before: sender PE 31
resolved under epoch 4 with put 165 queued for a credit, owner PE 35 free, last put 164 taken from the copy's
earlier token on PE 28 (an intra-process handoff in between), the credit for 164 released and never seen. The
channel objects now keep a ring of their last 24 protocol events (puts, resolutions, credits by message and
by shared memory, closings, acks, handoffs, pack/unpack) at no cost, and the stall watchdog and the owner's
probe print it with the raw shared-memory credit word (`[CHAN-HIST]`). The fifth untraced run of `chain3.sh`
stalled at step 151 and the histories said it in two lines. Owner PE 49: `reply s147 e4 pe46 sl956 busy
@29.077`, `release s147 e4 pe46 sl956 @29.079` -- the credit for put 147 written into shared-memory slot 956
two milliseconds after the reply, word reading `epoch 4 count 1`. Sender PE 46: `unpack @29.044` (it had just
arrived by pup), `resolve-req @29.044`, `pend s147 @29.081` (put 148 parked, credits 0), `resolved e4 sl956
credits 0 @29.116`; and `seen 0` against that `count 1` forever. The sender's put parked BEFORE its resolution
arrived, and `put()` lists a copy for the per-PE shared-memory credit poll only when it parks resolved;
`onResolved` issued a parked put when the reply carried a credit and otherwise left it, so a credit that
reached the word after the reply was computed (or, as here, before the reply was applied) was never read.
The same gap followed the last staged put (`onStagedDone`). Both now call `awaitShmCredit()`: list the copy
for the poll and read the word once at that moment. The message-credit path never had the problem (it is
event-driven), which is why the stall was only ever seen on the ipc tier. Four channel runs with the fix
completed (128.6-157.6 ms/step; `ship-wait`, the payload's wait for its pack copies, 0.08-0.10 ms, so the
rest of the ship window sits at the destination between the envelope's arrival, the ~1 ms rget and the
entry that acks -- the PE's message queue, which the channel arm fills with a note per put); the fifth
segfaulted in `pollShmCredit` from the per-PE waiter walk, a bug the fix made frequent: the walk polled
`w[i]`, the poll issued the put, `issue()` unlisted the copy by swapping it with the back and popping, and
the walk's own removal then read `w.back()` of an empty vector and polled the garbage. The walk now runs
over a copy of the list and purges afterwards; three channel runs with both fixes completed clean
(138.5-151.8 ms/step, `chain5.sh`). The HIP calls on the put path are not where a put's host time goes
(`vmmtest/batchbench`, one GCD: `hipMemcpyBatchAsync` of 4 x 64 KB 3.8 us against 4.7 for four
`hipMemcpyAsync`, 16 copies 5.3 vs 18.8; `hipEventRecord` 0.35 us; `hipMemsetD32Async` 1.2; `hipIpcGetMemHandle`
0.8; `hipMemGetAddressRange` 0.1), so the 37 us of a cross-host put is the runtime's own work.

**Where a put's host time goes, and why the per-step gain is small here (`[chan-cost]`, `[hapi-cost]`,
both printed under `CHARM_ZC_STATS=1`).** Inside the run every HIP enqueue costs 20-35 us: the pinned-flag
write behind each completion callback 24-35 us, the batched-copy closure 21-27 us, the persistent put
(context, op info, callback) 25-42 us; the source endpoint's open is 0.5-1.5 us. With tracing off
(`CHARM_LB_NO_CUPTI=1`) the same calls cost 4-5 us (flush 13, persistent put 5.5): the 6-7x is rocprofiler's
interception of every HIP call and every packet (the legacy intercept queue, with seven PEs issuing at once),
and both transports pay it -- the zero-copy arm's flag writes cost the same 29.6 us -- at ~1.9M calls per
process per run, 20-26% of all PE time. `+gpusubmit` moves the calls to one submitter thread and makes the
step 2-3x slower under tracing (one thread cannot issue 1.9M calls at 25 us), so it is not the answer here;
the patched profiler library is not either (the tax is the interception itself). What the tax does to the
step is small, because the step is not host-bound in this configuration: the balancer's accounting has the
device loads explaining 53-58% of the interval and host work 10-17%, and under the density gradient the
heaviest GCD sets the step, which is why no LB gives 134 whatever the transport and the balancer gains 17%.
The fair per-step comparison is uniform density without the balancer (100 steps, steps 51-100): channels
152.9 against zero-copy 161.9 ms/step with tracing off (5.6% faster; a same-host put 0.77 us, a cross-host
put 5.9 us, a zero-copy send 1.0 us but 11% more of them with their receive-side work), 159.0 against 162.5
traced (2%). So the channels do win per step, by the host work they save, and on Frontier that saving is a
few percent of a step that the GPU and the cross-node latency chain mostly set. The sub-timers stay in the
code at no cost when the statistics are off.

**The fix for the channel transport under the balancer (2026-10-22 evening).** Every hop of a move is
`[expedited]` and served from reconverse's urgent queue, and the scheduler-loop work between entries is
negligible (channel polls 0.8 ms at most, LCI progress and post loops under 3 ms, `hapiPollEvents` under 8 ms),
yet a per-hop trace (`CHARM_MIG_TRACE=1`, `migt.py`; only same-node hops, since the nodes' clocks differ by
seconds) put the destination's landed-to-entry at a median of 34 ms and a 90th percentile of 455 ms with the
profiler off. The entry histogram (`CHARM_EP_TIMES=1`) and a full HIP-API trace (`hipapi_trace.sh`, rocprofv3)
named the blocker: `hipStreamCreateWithFlags`, 3,853 calls of 2 ms or more in a 60-step one-node run, 13 ms each
(84 ms at worst), plus `hipHostFree` at ~1 s per call. Both are on the migration path: `hapiAcquireStream`
creates a stream whenever the device's free list is empty, and a balancer round's net arrivals on a device each
need one, so the destination PE spends the burst inside stream creation while every urgent hop of every other
move waits behind it; `Cell::freeDevice` freed two pinned slabs per move. On Delta those cost 20 us and 23 ms
(Compute's slabs were already pooled there for that reason). The zero-copy arm pays the same, but its ack
precedes the unpack and nothing waits on its reopen; the channel's closing-to-resolution does. With the
runtime's existing cap of shared application streams (`+gpustreams 32`, now the HIP default; CUDA keeps one
stream per chare) the channel arm's windows fall to grant 3.5 ms, ship 4.4, landing 2.6, settle 0.15, hold 111
(stand-down 42), and the step from 146.7 to **108.6 ms** (steps 121-200; 101.2 over 61-100). Cell's pinned
slabs now come from a never-freed pool and the arrival copy's synchronize is a stream callback (13 ms per
unpack on a shared stream otherwise); alone that changes little (the creations were the cost), with the cap:
channels 116.2, zero-copy 101.2 (92.0 over 61-100), single runs. What remained of the channel hold was the
owner's stand-down, ~40 ms, and a tally at the first deferral (`CHARM_LB_MIGSTATS`, `sd-own-inflight`,
`sd-own-pending`, `sd-acks`) said what it waits for: never its own puts (0.000 in flight or parked), only
its ~2 senders' closing acks -- which arrive in 3 ms (`chan-close`). The rest was the re-drive: when the last
ack brings the outstanding count to zero, `noteDeviceSendDone` posts the deferred-migration kick as a FIFO
self-message, by design (a 2026-09-15 Delta measurement on the zero-copy path: an urgent kick moved the element
ahead of the messages already queued for it, which then had to be parked and forwarded, and FIFO won by 4%),
and on a PE whose queue holds tens of milliseconds of work that is the stand-down. For a channel owner the
trade is the other way -- its senders are held for all of it -- so a count that reaches zero by a closing ack
now kicks urgently (`CkDeviceChannel::onCloseAck` -> `noteDeviceSendDone(urgentKick)` ->
`CmiPushPEUrgent`); the zero-copy path keeps its FIFO kick. Stand-down 44 -> 6.7-10 ms, hold 130 -> 64-97.
Final, two nodes, 8 processes, 200 steps, steps 61-100 / 121-200, the finished binary (patched profiler,
per-PE LCI devices, 32 shared streams, pooled pinned slabs, the credit and waiter fixes, the urgent kick):
zero-copy 98.4 / 100.1 ms/step (5,749 moves), channels 97.8 / 103.9 (10,086 moves; 88.0 / 113.0 in the run
before), against the day's start of 118 / 146 and a no-LB 134. The two transports are now within the
run-to-run spread of each other under the balancer, where the morning had channels 20-30% behind; per step,
without the balancer, channels lead by 5.6% untraced and 2% traced (above).

**HIP-specific tunings, behaviour unchanged (2026-10-22, 16:10-16:20; `hipopt.sh`, `hipopt2.sh`).** On the
per-step comparison (uniform density, no balancer, channels, traced, steps 51-100, one run each):
`HSA_ENABLE_SDMA=0` 157.9 against 160.3 (nothing); `GPU_MAX_HW_QUEUES=8` 155.0 and 16 154.9 (a 3% lean with
the 32 shared streams, and 87.4 against 82.3 under the balancer -- not taken); zero-copy ties channels at 160.5.
The completion-flag writer: HIP's flag write was an event record (the system-scope release hipMemsetD32Async
lacks) and a blit, 26.5 us of host time traced; a one-thread kernel with `__threadfence_system()` and a
system-scope atomic store (`hapi_kernels.cu`, `hapiFlagKernelImpl`) is one launch, 11.4 us, and on the
cross-device stale-read repro (`vmmtest/flag_kernel`, `COPYKB=32 PULL=1 CPDEV=1`: device 0 writes 32 KB and the
flag, a device-1 stream copies the data the moment the host sees the flag) it is clean in 0 of 200 trials for
kernel-written and for blit-written data, against 200 of 200 stale for the memset without its event. It is the
HIP default now (`+gpuflagmemset` restores the event-and-memset writer; CUDA is untouched). The profiler's
HIP-API tracing was registered for every runtime call but the polls, 541 operations; the attribution of a
kernel to its chare rides on the launching call's correlation id, so only the 98 dispatching operations
(launch, memcpy, memset) are traced now (`CHARM_TRACE_ALL_API=1` restores all): the flag write's event record
is no longer intercepted (26.5 -> 15.4 us before the kernel writer). `hipEventDisableSystemFence` on the
ordering events was considered and rejected: the repro shows that fence is the cross-device release. The flag
ring (`HAPI_FLAG_SLOTS` 4096) never filled in any run.

Final figures on the finished binary (two nodes, 8 processes, 200 steps, ms/step over steps 11-30 / 31-60 /
61-100 / 121-200, one run each): channels 119.4 / 78.7 / 89.1 / **82.3** (790 moves, device loads 74% of the
interval), zero-copy 119.4 / 77.6 / 92.9 / 103.5 (8,538 moves, 51%); the pre-balancer step itself fell from
130 to 119 for both transports with the kernel-written flags (the event's cache writeback and the blit are
gone from every flag). Against the day's start -- channels 146 and zero-copy 118 at steps 121-200, no balancer
134 -- the channels transport is now the faster one under the balancer on Frontier, as on the A40s.
