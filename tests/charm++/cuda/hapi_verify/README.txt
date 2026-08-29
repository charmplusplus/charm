hapi_verify -- GPU-effect verification test for the HAPI layer
==============================================================

Companion to ../hapitest, written independently of it as a
cross-examination instrument during the stage-9.1 review; kept because
the two tests catch different failure classes by construction. What
this one uniquely checks:

  - that kernels ACTUALLY EXECUTE and produce correct data: every device
    buffer is filled by a kernel, copied back asynchronously, and
    verified element-by-element. A silently non-executing kernel (e.g. a
    fatbin with no cubin for the local GPU and a failed/ineffective JIT
    -- observed in the wild on an A100 with nvcc-default sm_75 objects)
    fails here and nowhere else; hello's empty kernel and callback-only
    tests cannot see it.
  - pinned-host pool / buddy-allocator behavior under interleaved
    varied-size alloc/free, with byte-pattern verification before every
    free (the overlap-corruption class).
  - per-PE device-mapping assertions, and N async completions delivered
    through hapiAddCallback.

Build:  make CUDATOOLKIT_HOME=<cuda dir> [CUDA_ARCH=sm_XX]
Run  :  ./hapi_verify +p2        (classic, via charmrun ++local)
        ./hapi_verify +pe 2      (reconverse)
Success is exactly one line:  HAPI_VERIFY PASS: ...
Every check is a CkEnforce; -w <sec> watchdog (default 60) aborts stalls.

Reviewer advice (per the GPU-series validation protocol in
doc/reconverse-merge-plan.md): for PRs touching GPU paths, run BOTH
hapitest and hapi_verify on the GPU hardware of your choice and record
where in the PR template's HPC-site field. Not in CI: hosted runners
have no GPU, and a CUDA-enabled reconverse binary cannot even start
without a driver (LCI calls cuInit at startup).

Pedigree, kept as advice for future GPU-test authors: six defects were
found in this program before it was sound -- a charmxi attribute
spelling, a hardcoded charmc path, -std too old for the toolkit, group
ckNew() misuse, hapiGetStream before hapiCreateStreams, and a missing
-arch flag whose failure mode was a kernel that never ran and never
errored. The last one is why the data verification exists.
