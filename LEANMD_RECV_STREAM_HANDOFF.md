# LeanMD isolated receive streams

## Change

LeanMD now opts into `CkDeviceBufferPost::isolated` for position inputs, force
contributions, and particle exchanges. The runtime gives each nonempty receive
operation an exclusive nonblocking CUDA stream. Sender-event waits and copies
execute there, leaving the application's shared compute streams free to run
ready work. Other active receives cannot share the leased stream either.

Both network and CUDA completion handlers return the lease before delivering
the completed receive entry. LeanMD already launches pair computation only
after its input entries arrive, and accumulates force contributions after their
receive entries arrive. No extra GPU wait is inserted into the compute stream.
No application SDAG changes or cross-process clock assumptions are needed.

The new posting flag defaults to false. Other applications retain their current
stream-ordering semantics. An application may opt in only if the destination
buffer is already safe to overwrite: earlier work on its posted application
stream no longer orders the copy. LeanMD's position buffers are reusable after
the preceding force computation completes; force and particle inputs use
separate per-neighbor landing slots. Cell step callbacks order prior uses before
the next exchange. Existing migration/receive admission and completion protocols
remain responsible for keeping in-flight buffers alive.

For a matched comparison with the old behavior, set
`LEANMD_SHARED_RECV_STREAM=1`; unset it for isolation. This switch changes only
LeanMD's receive posting, not placement, compute stream assignment, or the
communication accounting fix.

## Resource management

The pool is per worker thread and per CUDA device. It caches up to 64 completed
streams per device; additional completed streams are destroyed. Active streams
are never reused, and the number of active leases follows the number of
outstanding nonempty posted receives. Empty messages do not acquire a stream.
Cached CUDA resources are reclaimed with the process context at exit.

This removes the shared-stream dependency, but stream creation/destruction and
larger concurrent receive counts can add overhead. A performance win or reduced
variance has not been established by the host tests. If creation churn is high,
measure the active high-water mark before tuning cache retention; do not bound
active leases by sharing a stream with an unfinished receive.

## Files and tests

- `src/ck-core/ckrdmadevice.h`: opt-in receive posting flag.
- `src/ck-core/ckrdmadevice.C`: stream leasing and completion release.
- `src/ck-core/ckdevicerecvpool.h`: exclusive stream ownership and device cache.
- LeanMD `Cell.h`, `Cell.cc`, `Compute.cc`: enable isolation and comparison switch.
- `tests/charm++/load_balancing/lbdriver/recv_stream_test.C`: host-only pool test.

`make -C tests/charm++/load_balancing/lbdriver check-recv-stream` passes. It checks
that an unfinished receive does not block allocation/reuse for another receive,
device separation, out-of-order completion, normal unleased completion, and
bounded idle caching. It does not execute CUDA or validate the transport.

`cmake --build multicore-linux-x86_64-cuda --target ck -j4` and the LeanMD build
pass. A two-node A40 correctness run, job **22158228**, was submitted with a saved
binary and isolated logs under `leanmd/recv-stream-test/`. It runs a small
8x4x4 grid for 30 steps, period 10, noLB/sync/async (lag 4), and enables pointer
validation. This is a correctness smoke test, not the 16x8x8 performance case.

The posting struct changed size: rebuild callers against the new header before
linking against the new runtime. In particular, saved old object files that
construct arrays of `CkDeviceBufferPost` are not ABI-compatible.
