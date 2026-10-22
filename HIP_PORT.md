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
