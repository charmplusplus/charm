# commbench

Point-to-point latency of every Charm++ entity and send method, plus the
Converse-level sends, in one program with one argument set and one output
format (JSON lines). This is milestone M1: host memory, pattern `lat`
(ping-pong, one-way = round trip / 2).

## Build and run

```
make                      # uses ../../../bin/charmc; make CHARMDIR=... for another tree
./commbench +pe 2 --out run.jsonl                  # P0: one process, 2 PEs
lcrun -n 2 ./commbench +pe 2 --out run.jsonl       # P2 on a workstation
./commbench +pe 2 --list                           # print the cases and exit
run/commbench-run.sh SITE PLACEMENT [ARGS...]      # the recorded way (below)
```

Options (Charm++ runtime options start with `+`, commbench's with `--`):

| option | default | meaning |
|---|---|---|
| `--entities LIST` | all | `handler chare group nodegroup array` |
| `--methods LIST` | all | method names; a prefix ending at a dot selects all below it (`nocopy`, `get`, `get.reg`, `persistent`) |
| `--sizes LIST` or `MIN:MAX:FACTOR` | 8 B to 16 MB in powers of 2, plus 8096 and 8097 | payload bytes; `k` and `m` suffixes are powers of 1024 |
| `--extra-sizes LIST` | | sizes added to the list |
| `--reps N` | 5 | repeats; every repeat is a sample record |
| `--sample-time SEC` | 0.1 | target duration of one sample, used by calibration |
| `--pairs auto` or `A:B,C:D` | auto | PE pairs; auto takes PE 0 and the first PE of each locality class present |
| `--seed N` | time-based | seed of the per-repeat shuffle (recorded) |
| `--out FILE` | stdout | output; stdout also carries the runtime's banner lines |
| `--deliver message\|user` | message | `user`: received data must end in the receiver's own buffer (see "User buffers") |
| `--source message\|user` | message | `user`: sent data starts in the sender's own buffer |
| `--no-validate` | | skip the validation round (smoke runs only) |
| `--list` | | print the case list and exit |
| `--progress` | | print each step on stderr, to locate a hang |
| `--run-id ID`, `--machine NAME` | host and time | fields of the run record |

## Cases

A case is (entity, method, memory=host, pattern=lat, pair, size).

Charm++ entities (`chare`, `group`, `nodegroup`, `array`) run these methods:

| method | what one half round trip is |
|---|---|
| `marshal` | `entry recvMarshal(int n, char d[n])` |
| `msg` | a varsize `CommMsg`; the receiver sends the message it got back, so the loop allocates nothing |
| `nocopy.{reg,unreg,prereg}` | `entry recvNocopy(int n, nocopy char d[n])`, `CkSendBuffer(buf, mode)` on every send |
| `post.{reg,unreg,prereg}` | `nocopypost`; the post entry sets `regMode`, calls `CkMatchBuffer` and `CkPostBuffer` |
| `get.{reg,unreg,prereg}.{nodereg,dereg}` | a 16 B notify, then the receiver's `CkNcpyBuffer::get`; ends when the get completes (`notify+get`) |
| `put.{reg,unreg,prereg}.{nodereg,dereg}` | `CkNcpyBuffer::put`; the half ends when the runtime's destination acknowledgement runs the receiver's handle callback (`put+destcb`) |

For `get` and `put`, `nodereg` handles are created and exchanged once at
setup. A `dereg` handle is good for one transfer (the runtime deregisters it
when the transfer completes). For `get` each notify carries a new handle
(`notify(handle)+get`). For `put` the receiver sends the sender a new receive
handle in a separate message once the data has landed, and the half ends when
both the data and the peer's handle are in (`put+destcb+handle`); the put's
data path cannot carry the handle. `prereg` buffers come
from `CkRdmaAlloc` on both sides.

The `handler` entity runs Converse-level methods; setup arrives through a
group, the timed traffic uses Converse handlers only:

| method | |
|---|---|
| `sync` | `CmiSyncSendAndFree`, the message sent back |
| `node` | `CmiSyncNodeSendAndFree` to the peer's node; `aux.recv_pes_*` lists the PEs that ran the handler |
| `rget`, `rput` | `CmiNcpyBuffer` `rdmaGet` / `rdmaPut` between buffers registered at setup: `notify+rget`, and `rput+notify`, the notify sent from the initiator's ack handler, which the runtime runs off the `+ipc` pool |
| `persistent.send` | `CmiCreatePersistent` at the sender, sends inside `CmiUsePersistentHandle` |
| `persistent.recv` | `CmiCreateReceiverPersistent` at the receiver; the `PersistentReq` travels to the sender, which calls `CmiRegisterReceivePersistent` |

Converse messages carry a 16 B commbench header after the Converse header;
`size` counts only the payload. `aux.persistent_path` is `local_forward`
(same process: the channel forwards to the plain send), `rma_put`, or
`inline_message` (no RMA, or a peer reached by the `+ipc` pool).
`aux.rdma_path` of the zerocopy methods is `local`, `rma` or `message`
(copy-based fallback).

Locality of a pair: `same_process`, `same_host`, `cross_host`, from
`CmiNodeOf` and `CmiPhysicalNodeID`. Inside one process a nodegroup pair and a
`node` send can only reach their own process, so they are written as skip
records, not timed. `same_pe` pairs are not supported.

## User buffers and copies

On reconverse with LCI, neither a large plain message nor a `nocopy` receive
copies the payload: both land by RDMA in a buffer the runtime allocates, which
becomes the message. Only `post.*`, `get.*`, `put.*`, `rget` and `rput` land in
a buffer the application chose. An application that needs the data in its own
buffer therefore copies out of a plain or nocopy message, and one whose data
starts in its own buffer copies into a message (marshal does that copy itself,
as the pack). `--deliver` and `--source` make commbench do that work, so the
methods can be compared on what such an application must do.

Every PE has a send and a receive buffer of the largest size of the run,
allocated and with every page written at the first case that uses it (the
`prereg` methods have their own pair, from `CkRdmaAlloc`). These are the user
buffers.

- `--deliver user`: when a half round trip arrives somewhere other than the
  receive buffer, the receiver memcpys the payload into it inside the timed
  loop, before it replies or counts the arrival (`Side::halfArrived`).
  Validation then checks the receive buffer, the final destination.
- `--source user`: the methods that send a message the program builds (`msg`,
  `sync`, `node`, `persistent.*`) memcpy the send buffer into it before each
  send (`Side::fillOutgoing`). The other methods already send from the send
  buffer: marshal packs from it, nocopy and post pass it to `CkSendBuffer`,
  get/put handles and rget/rput handles point at it.

Per one-way message, `aux.copies_sender` and `aux.copies_receiver` count this
program's payload memcpys (checked against the copies actually done in each
run; a mismatch fails validation):

| method | `--source user` sender | `--deliver user` receiver | lands in |
|---|---|---|---|
| `marshal` | 0 (the runtime packs from the send buffer) | 1 | the message |
| `msg`, `sync`, `node`, `persistent.*` | 1 | 1 | the message |
| `nocopy.*` | 0 | 1 | a fresh message (`ckrdma.C:823`) |
| `post.*` | 0 | 0 | the posted receive buffer |
| `get.*`, `put.*` (`nodereg` and `dereg`) | 0 | 0 | the receive buffer; a `dereg` handle is re-created on the same buffer |
| `rget`, `rput` | 0 | 0 | the receive buffer |

With `message` for either flag the corresponding column is 0. `flags.deliver`
and `flags.source` in the run record hold the settings.

### Runtime copies (estimate)

`aux.copies_runtime_est` is the number of payload copies the runtime itself
makes per one-way message, sender and receiver together, and
`aux.copies_runtime_path` names the path it assumes; `aux.msg_bytes` is the
size of the message that carries the payload, headers included. **This is an
estimate from reading the code, not a measurement.** It counts copies in
Charm++, reconverse and LCI; copies inside the network provider (libfabric
tcp on a Mac, the NIC's inline send of very small messages) are not counted.
`null` means the path is not covered.

| path | copies | reasoning |
|---|---|---|
| `local_pointer` | 0 | a message to a PE of the same process is handed over by pointer (`contrib/reconverse/src/convcore.cpp:1098-1099`) |
| `ipc_pool` | 1 | `+ipc`, a peer process on the host, message within the cutoff: the sender copies the message into a pool block (`cmishmem.cpp:190`), which the receiver takes as the message (`cmishmem.cpp:206-221`). If the message already sits in a pool block for that peer, no copy is made |
| `lci_eager` | 1 | message of at most packet size - 76 bytes (8116 B for the default 8192 B packet, `CMakeCache` `LCI_PACKET_SIZE_DEFAULT`, env `LCI_ATTR_PACKET_SIZE`; `lci-src/src/core/communicate.cpp:841`, `packet_pool/packet_pool.hpp:36`). The receiver copies the packet into a `CmiAlloc` buffer (`lci-src/src/core/progress.cpp:50-52`; 0 with `LCI_ATTR_ZERO_COPY_AM`). The sender makes no copy: a `CmiAlloc` message carries its pool block's registration (`comm_backend_lci.cpp:154`), so LCI sends it `eager_zcopy` (`communicate.cpp:219`); `eager_bcopy`, the copy into a packet (`communicate.cpp:303`), is taken only for an unregistered buffer (`communicate.cpp:211-214`). The memcpy at `rendezvous.hpp:33` belongs to LCI's matched send/receive, which reconverse does not use (it sends active messages) |
| `lci_rendezvous` | 0 | larger messages: the receiver allocates through reconverse's allocator (`rendezvous.hpp:108`, `comm_backend_lci.h:16`), the data lands there by RDMA, and `CommRemoteHandler` hands that buffer to the scheduler (`convcore.cpp:224-231`) |
| `pack+...` | 1 + the message path | marshal: the generated sender copies the array into the message (`commbench.def.h`, `memcpy(impl_buf+impl_off_d,d,impl_cnt_d)`); the receiver gets a pointer into the message |
| `memcpy` | 1 | zerocopy methods within one process: `CkNcpyBuffer::get`/`put` memcpy (`ckrdma.C:53-54`, `133`), the nocopy and post receive (`ckrdma.C:511`, `633`), and Converse `rdmaGet`/`rdmaPut` (`conv-rdma.cpp:631`, `659`) |
| `rdma` | 0 | zerocopy methods between processes (`ckrdma.C:639`, `conv-rdma.cpp:637-643`). CMA is not used on reconverse: `useCMAForZC` is never set (`conv-rdma.cpp:21`) |
| `rma_put` | 0 | persistent: a put into the receiver's registered buffer (`persist-comm.cpp:362`), which the receiver hands over as the message (`persist-comm.cpp:462`) |
| `inline_message+...` | 2 + the notification's path | persistent to a peer reached by `+ipc` or without RMA: the payload is copied into the notification (`persist-comm.cpp:327`) and out of it into the receive buffer (`persist-comm.cpp:451`) |
| `copy_based_rdma`, `lci_shm`, `unknown_backend` | null | not estimated: `CmiUseCopyBasedRDMA`, a build with `LCI_WITH_SHM=ON` between processes on one host, a build without an LCI packet size |

On this Mac (`LCI_PACKET_SIZE_DEFAULT` 8192), the `msg` message of an 8097 B
payload is 8192 B and the marshal one 8208 B, both above the eager limit; an
8 B `sync` message is 48 B.

## Measurement

For each (case, size), before any timing: one validation round, in which both
sides fill their send buffers with a pattern that depends on offset, role and
size, prefill their receive buffers with its complement, and check every byte
and the size of what arrives. A failure is a skip record and the case is not
timed. Then one calibration run of 10 iterations sets the count so that a
sample takes about `--sample-time`, clamped to [10, 100000]; the count is the
same for all repeats. Each sample runs max(2, iters/10) untimed warmup
iterations, then `iters` timed round trips on side A with `CmiWallTimer`.
The repeat loop is outermost and shuffles (case, size) each time.

## Output

JSON lines: one `run` record, then `sample` and `skip` records. The schema is
section 3 of the commbench design note (to be published with the results
repository as `schema/`). `run` holds the commits embedded at build time
(`charm_commit`, `charm_dirty`, `reconverse_commit`, `lci_commit`), the
build target and `CMK_*` flags, topology, `argv` and `argv_runtime`, the IPC
settings in `flags` (`ipc`, `ipcmode`, `ipccutoff`, `ipcpoolsize`,
`ipc_active` from `CmiIpcEnabled()`), `deliver`, `source` and
`lci_packet_size` in `flags`, the options and the seed. A sample carries
`value` (`lat_oneway_us`), `aux.lat_roundtrip_us`, the copy counts
`aux.copies_sender`, `aux.copies_receiver`, `aux.copies_runtime_est` and
`aux.copies_runtime_path`, `iters`, `warmup`, `protocol` and `validated`.

## Placements and commbench-run.sh

`run/commbench-run.sh SITE PLACEMENT [ARGS...]` builds the launch line, runs
it with `+showcpuaffinity`, and writes `<run-id>.jsonl` plus
`<run-id>.jsonl.sidecar.json` (launch line, `FI_*`, `LCI_*`, `LCT_*`,
`CmiIpc*` and Slurm environment, modules, job id, host names, and the
CMakeCache values `LCI_USE_REG_CACHE`, `LCI_NETWORK_BACKENDS`,
`LCI_PACKET_SIZE_DEFAULT`, `RECONVERSE_ENABLE_CPU_AFFINITY`, `CMK_USE_SHMEM`,
`CMAKE_BUILD_TYPE`). SITE is `mac`, `anvil`, `delta`, `frontier` or
`generic`; extra ARGS (runtime options such as `++ipccutoff 65536` and
commbench options) are passed through. Environment: `PEMAP` replaces the
`+pemap` list, `CPU_BIND` the srun `--cpu-bind` value, `P0_CPUS` the `-c` of
the P0 step (srun `-c` implies `--exact`, so a `+pemap` outside the step's
CPUs aborts with `CmiSetCPUAffinity failed`), `LAUNCHER` the launcher, and
`TIMEOUT` bounds the run with GNU `timeout` (the sidecar is still written,
with `exit_status` 124). Pinned to the cores next to the NIC on Anvil
(mlx5_0 is on NUMA node 2):

```
PEMAP=32 CPU_BIND=map_cpu:32 run/commbench-run.sh anvil P1 --reps 5
PEMAP=32,33 CPU_BIND=map_cpu:32,33 run/commbench-run.sh anvil P2 --reps 5
PEMAP=32,33 CPU_BIND=none P0_CPUS=128 run/commbench-run.sh anvil P0 --reps 5
```

| placement | meaning |
|---|---|
| P0 | 1 process, 2 PEs |
| P1 | 2 hosts x 1 process |
| P2 | 1 host x 2 processes |
| P2i | P2 with `+ipc` |
| P3 | 2 hosts x 2 processes; `--pairs auto` then covers every locality class |

On a Mac only P0 and P2 exist, and cross-process traffic is TCP loopback:
use them to develop, not to measure.
