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

To exercise all three transfer modes, run across at least two hosts. Which
shapes are available depends on the example: verify, persistent and the three
benchmarks require exactly 2 PEs, while sdag, the jacobi2d pair and jacobi3d
take any count.

    srun -N2 -n2 ./verify              # 2 PEs, one per host: RDMA
    srun -N2 -n2 ./sdag +pe 4          # 4 PEs over 2 hosts: memcpy and RDMA
    srun -N1 -n2 ./sdag +pe 4          # one host, two processes: memcpy and IPC


Known limitations, as measured on Frontier (2026-09-01)
-------------------------------------------------------

These are properties of the runtime, not of the examples, but they decide how
the examples can be run, so they are recorded here.

1. CkDevicePersistent does not work between hosts. The runtime rejects it
   outright -- "Persistent GPU messaging is currently not supported for
   inter-node messages", from the two guards in ck-core/ckrdmadevice.C -- so
   the persistent example and the latency-persistent benchmark abort
   immediately when run across more than one host. Both work within a host, on
   the device-memcpy and IPC paths. This is an unimplemented feature rather
   than a defect; nothing to work around, just run them on one host.

2. Load balancing and GPU-direct transfers cannot be combined across processes.
   The sender resolves the destination PE before the transfer; if the
   destination chare has since migrated to another process, the receiver
   rejects the buffer and aborts with "Destination process does not match the
   one the sender determined". Inside a single process, migration and
   GPU-direct transfers work together, and jacobi2d warns at startup when the
   combination cannot work.

   This one is not fixable at the receiver: when the sender chose the
   device-memcpy mode it never staged the data anywhere the new owner's process
   can reach, so by the time the mismatch is visible the bytes are simply not
   available. It needs the sender to learn about the migration first -- the
   runtime's message names CMK_GLOBAL_LOCATION_UPDATE, which is referenced in
   ck-core but not defined by any build -- or a re-send handshake.

3. The bandwidth benchmark's zerocopy phase exhausts LCI's memory registrations
   at the default 4 MB maximum size ("register_memory_impl ... No space left on
   device"). It reallocates its device buffers per message size and the
   registrations are not released. -x 65536 stays under the limit.

4. +gpucommbuffer, +gpulbbuffer and +gpuipceventpool configure the
   shared-memory inter-process path and only take effect alongside +gpushm.
   Passing one without it is now reported as such at startup.

Fixed since these examples were first run, and required by them:

  - Startup used to deadlock whenever physical node 0 held exactly one PE --
    the ordinary one-PE-per-host multi-node launch. PE 0 waited in
    CmiCheckAffinity for affinity messages that no PE would ever send. That
    made every inter-node run here hang before reaching main, and it was not
    GPU-specific: ckhello hung identically. Fixed in reconverse 2c50813e7
    (charmplusplus/reconverse#212); this tree pins it.

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
