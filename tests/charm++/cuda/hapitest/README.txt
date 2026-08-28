hapitest -- acceptance test for the HAPI portable core (migration plan 9.1)
==========================================================================

One self-checking program. Every check is a CkEnforce, so a failure aborts with
a file and line; success prints exactly one line:

    hapitest PASS: <PEs> PEs, <n> per-PE checks, <n> GPU callbacks, <t> s

It is meant to be the thing a reviewer runs on their own GPU. Per the runtime
validation protocol (#3950) CI for this series is build-only; this test is what
makes "reviewers run on GPU types of their choice until satisfied" concrete,
and it is deliberately backend-neutral so an NVIDIA run and an AMD run are
comparable verdicts on the same source.

What it checks
--------------
  1. hapi_portable.h as user code sees it. hapitest.C is compiled by charmc --
     a plain host compiler -- and calls hapiMalloc / hapiMemcpyAsync /
     hapiStreamCreate directly. Only the kernel launch is in hapitest.cu. If
     the header stops being self-sufficient outside the Charm++ build (the
     failure its __HIP_PLATFORM_AMD__ fallback exists to prevent), this test
     does not compile.

  2. Per-PE device mapping: the device hapiMyDevice() reports is the one the
     thread is actually set to, is in range, carries this PE's physical node in
     its high half, is stable, and -- under the default round-robin mapping --
     matches hapiMapping()'s own arithmetic. Then, across PEs: the mapping
     reaches min(devices, PEs) distinct devices per physical node and spreads
     them evenly. That last pair is what catches every PE landing on device 0.

  3. The buddy allocator, driven directly: power-of-two rounding, blocks that
     do not overlap, whole-region allocation, oversize refusal, and full
     coalescing back to one region under three different free orders. Plus the
     load-balancing sub-region's separate accounting.

  4. N asynchronous callbacks: each chare issues -k independent
     H2D/kernel/D2H chains and hangs a hapiAddCallback off each, then verifies
     every result against a closed form out of a sentinel-filled buffer, so a
     callback delivered before its copy landed fails rather than passes.
     Completions are counted per chare and summed. Quiescence is started while
     the work is still in flight, so QD reached early -- the QdCreate/QdProcess
     bracketing in hapiAddCallback going wrong -- is also a failure.

Building
--------
This directory is not in tests/charm++/Makefile's DIRS, matching how the CUDA
examples are handled: a non-GPU build cannot compile it.

    make GPU=hip  CHARM_DIR=../../../../reconverse-linux-x86_64-amd
    make GPU=cuda CHARM_DIR=../../../../netlrts-linux-x86_64-cuda CUDA_ARCH=sm_80

HIP_ARCH defaults to gfx90a (MI250X), CUDA_ARCH to sm_80 (A100).

Running
-------
    ./hapitest +pe 4

On Frontier, one node (--network=single_node_vni is required for single
physical node runs and must be omitted for multi-node ones):

    srun -A CSC710 -N1 -n1 -c56 --gpus-per-node=8 --network=single_node_vni \
         ./hapitest +pe 8

Flags: -c chares (default 2*PEs), -k chains per chare (default 8),
       -n doubles per chain (default 1024), -v verbose,
       -M skip the exact mapping formula (use with +gpumap block / none,
          which the formula does not model).

Known gap
---------
There is no "allocate until exhausted, then ask for one more" case in the buddy
allocator section. That path runs allocator::malloc's bucket scan off the end
of the bucket array -- `buckets[bucket].empty() && bucket < bucket_count` tests
emptiness before the bound. It predates stage 9.1, so this test does not assert
on it; it is noted here so the omission is not mistaken for coverage.
