GPU-direct (device-to-device zerocopy) examples
===============================================

These examples move data straight between GPU buffers on different chares,
without staging it through host memory. The transfer is expressed with the
nocopydevice entry-method parameter and CkDeviceBuffer on the sending side, and
a post entry method that hands the runtime a destination pointer and a stream
on the receiving side. The runtime picks the actual mechanism per message --
a device-to-device copy within one process, an IPC handle between processes on
one host, or an RDMA get between hosts -- and the example does not change.


Building
--------

Each subdirectory builds against either GPU backend from one source. The
backend defaults to whatever the Charm++ build it is pointed at was configured
with, so from inside a build tree no flags are needed:

    make -C <build>/examples/charm++/cuda/gpudirect

Building out of the source tree, or overriding the backend, is explicit:

    make GPU=hip  CHARM_DIR=../../../../../reconverse-linux-x86_64-amd HIP_ARCH=gfx90a
    make GPU=cuda CHARM_DIR=../../../../../netlrts-linux-x86_64-cuda   CUDA_ARCH=sm_80

HIP_ARCH defaults to gfx90a (MI250X) and CUDA_ARCH to sm_80 (A100); set them
for the device you actually have, or the kernels may fail to launch with
nothing but an unchecked error to show for it.


Running
-------

`make test` in the parent directory deliberately does NOT descend here. Every
one of these needs real GPUs, and the transfer mode being exercised depends on
how the ranks are laid out: one process gets device-to-device copies only, two
processes on one host add IPC, and two hosts add RDMA. Running them therefore
belongs to the HPC-site validation protocol rather than to an automated test
sweep. Each subdirectory still has a `test` rule for running it by hand.

To exercise all three transfer modes, run across at least two hosts with more
than one PE per process, e.g.

    srun -N2 -n4 ./verify


Known limitations, as measured on Frontier (2026-09-01)
-------------------------------------------------------

These are properties of the runtime, not of the examples, but they decide how
the examples can be run, so they are recorded here.

1. Inter-node transfers wedge with exactly one PE per process. Two hosts and
   two PEs per process works; two hosts and one PE per process hangs with no
   output. This is not specific to these examples -- the stage 9.2 acceptance
   test, tests/charm++/cuda/d2dtest, hangs identically at that shape and passes
   at every other -- so give each process at least two PEs.
   (+backend_poll_thread 1, which is already the default, does not help, so
   this is not simply a question of who drives network progress.)

   verify, persistent, and the latency/bandwidth benchmarks all hard-require
   exactly 2 PEs, which makes one-PE-per-process their only inter-node shape.
   Until the above is fixed they can only be run on a single host, where they
   still cover both the memcpy (one process) and IPC (two processes) paths.

2. Load balancing and GPU-direct transfers cannot be combined across processes.
   The sender resolves the destination PE before the transfer; if the
   destination chare has since migrated to another process, the receiver
   rejects the buffer and aborts with "Destination process does not match the
   one the sender determined". Inside a single process, migration and
   GPU-direct transfers work together. jacobi2d warns about the combination at
   startup. The runtime's own message names CMK_GLOBAL_LOCATION_UPDATE, which
   is referenced in ck-core but not defined by any build; giving the device
   path real location updates is the fix.

3. The bandwidth benchmark's zerocopy phase exhausts LCI's memory registrations
   at the default 4 MB maximum size ("register_memory_impl ... No space left on
   device"). It reallocates its device buffers per message size and the
   registrations are not released. -x 65536 stays under the limit.

4. +gpucommbuffer, +gpulbbuffer and +gpuipceventpool are only read when +gpushm
   is also given. Passed on their own they are silently ignored, and the only
   sign is a generic "not parsed by the RTS" warning.


The examples
------------

verify        Correctness check: sends a known pattern between a chare array,
              a group and a nodegroup, and verifies every element on arrival.
              Must be run with exactly 2 PEs.

sdag          The same transfer driven from structured dagger, mixing a
              nocopydevice parameter and an ordinary array parameter in one
              entry method.

persistent    Uses CkDevicePersistent to set up a channel once and reuse it,
              rather than re-establishing the transfer per message.

jacobi2d      2D 5-point Jacobi stencil over a 2D chare array, exchanging halo
              rows and columns each iteration. -z selects the GPU-direct path;
              without it the ghosts go through host memory, which is the
              comparison the example exists to make. -y runs the synchronous
              variant, where Main drives each iteration.

              Load balancing is off unless -l is given. With it, the chares
              migrate at iteration boundaries: -f sets the first iteration at
              which that may happen and -l the interval. Note that the device
              data is staged through pinned host memory by hand in pup(); PUP
              has no device mode yet, so a chare owning GPU memory has to move
              it itself.

jacobi2d-imbalance
              jacobi2d with a graded per-chare load: blocks toward the
              bottom-right of the grid do up to -m extra stencil passes,
              blocks at the top-left do none. A block mapping therefore leaves
              the last PEs overloaded, which gives the balancer something to
              move. This is the variant to run when the point is the load
              balancer rather than the transfer, e.g.

                  ./jacobi2d +pe 8 -z -i 40 -f 10 -l 10 +balancer GreedyRefineLB

jacobi3d      The 3D counterpart of jacobi2d, exchanging six faces. -d selects
              the GPU-direct path and -s the persistent one.


Related benchmarks
------------------

benchmarks/charm++/cuda/gpudirect holds the latency, bandwidth and
latency-persistent microbenchmarks for the same transfer path.
