******************** HYBRID API ********************
*                                                  *
* Originally by Lukasz Wesolowski (June 2008)      *
* Redesigned by Jaemin Choi (July 2017)            *
*                                                  *
****************************************************

1. Introduction
---------------

  Hybrid API, or HAPI, provides an interface for asynchronously offloading work
to a CUDA-enabled GPU device.

  The original design by Lukasz was as follows:
  1) Create a workRequest object and populate it with information about buffers
     and kernel execution parameters.
  2) Enqueue the workRequest into a queue managed by GPU Manager, which is a
     process-shared entity.
  3) All PEs (schedulers) poll the queue at a certain interval.
  4) Progress of the first 3 elements are examined and moved forward as needed.
     3 CUDA streams are used for this: 1 for host-to-device data transfer,
     1 for device-to-host data transfer, and 1 for kernel execution.
  5) When a workRequest has progressed to the last state (all data transfers
     and kernel execution complete), the associated Charm++ callback function
     is invoked.

  This design allows the two data transfers to overlap with each other and with
kernel execution, and frees up the CPU to execute other entry methods when one
has offloaded work to the GPU. However, it unnecessarily intervenes with GPU
operations; we can associate a separate stream for each chare and have the
chares directly call CUDA APIs on them. With the recent support of concurrent
kernel execution in CUDA, this allows kernels of multiple chares to execute
concurrently, in addition to the overlap with data transfers.

  The new design consists of thin wrappers of CUDA API calls so that the runtime
can provide support for profiling and leave room for optimizations in the
future. A critical factor in both the old and new designs is the support for
Charm++ callback functions; it allows offloaded work on the GPU to proceed
concurrently with entry methods of other objects on the CPU.

  Two schemes for supporting Charm++ callbacks are currently implemented in
HAPI, one that uses CUDA events and polling, and one that uses the CUDA
callback feature. As of now, the CUDA events based scheme performs better
and is set as the default.

2. Usage
--------

  1) Offload part of the computation as a CUDA kernel.
  2) Include hapi.h in the CUDA source code.
  3) (Optional) Replace CUDA API calls with corresponding HAPI calls.
     e.g. cudaMemcpyAsync() -> hapiMemcpyAsync()
  4) Place hapiAddCallback() after a CUDA API call to invoke the given Charm++
     callback when the corresponding GPU operation is complete.
     e.g. after cudaMemcpyAsync() or kernel invocation

3. Files
--------

<hapi.h>
  Contains declarations of HAPI functions.
  To be included in user's CUDA source code.

<hapi_nvtx.h>
  Contains functions that utilize NVIDIA's NVTX feature.
  To be included in user's CUDA source code.

<hapi_src.h>
  Contains function declarations used by the Charm++ runtime.

<hapi_src.cu>
  Contains the actual implementation of HAPI functionalities.

4. NVCC Flags
-------------

  The following #define flags can be passed to NVCC in the Makefile:

  HAPI_CUDA_CALLBACK
    Use CUDA callback based scheme for Charm++ callback support. Default scheme
    is based on CUDA events.

  HAPI_MEMPOOL
    Allocate a pool of page-locked memory in advance to draw allocations from.
    Can be useful when page-locked memory are frequently allocated/freed during
    program execution.

  HAPI_TRACE
    Time for invocation and completion of GPU events (e.g. memory allocation,
    data transfer, kernel execution) are recorded.

  HAPI_INSTRUMENT_WRS
    Records work request start (WRS) and end times. This includes time spent in
    each phase (data in, kernel exec, and data out) and other relevant data
    such as the chare that launched the kernel, etc.

  HAPI_NVTX_PROFILE
    Turn on NVTX profiling for HAPI functions.

  HAPI_CHECK_OFF
    Turn off error checking of CUDA API calls.

  HAPI_DEBUG
    Print debugging output during execution.

  HAPI_MEMPOOL_DEBUG
    Print mempool-related debugging output.
