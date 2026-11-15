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
| `put.{reg,unreg,prereg}.{nodereg,dereg}` | `CkNcpyBuffer::put`, then a 16 B notify from its local completion (`put+notify`) |

For `get` and `put`, `nodereg` handles are created and exchanged once at
setup. A `dereg` handle is good for one transfer (the runtime deregisters it
when the transfer completes), so each notify carries a new handle for the next
transfer (`notify(handle)+get`, `put+notify(handle)`). `prereg` buffers come
from `CkRdmaAlloc` on both sides.

The `handler` entity runs Converse-level methods; setup arrives through a
group, the timed traffic uses Converse handlers only:

| method | |
|---|---|
| `sync` | `CmiSyncSendAndFree`, the message sent back |
| `node` | `CmiSyncNodeSendAndFree` to the peer's node; `aux.recv_pes_*` lists the PEs that ran the handler |
| `rget`, `rput` | `CmiNcpyBuffer` `rdmaGet` / `rdmaPut` between buffers registered at setup, with the get/put protocol above |
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
`ipc_active` from `CmiIpcEnabled()`), the options and the seed. A sample
carries `value` (`lat_oneway_us`), `aux.lat_roundtrip_us`, `iters`, `warmup`,
`protocol` and `validated`.

## Placements and commbench-run.sh

`run/commbench-run.sh SITE PLACEMENT [ARGS...]` builds the launch line, runs
it with `+showcpuaffinity`, and writes `<run-id>.jsonl` plus
`<run-id>.jsonl.sidecar.json` (launch line, `FI_*`, `LCI_*`, `LCT_*`,
`CmiIpc*` and Slurm environment, modules, job id, host names, and the
CMakeCache values `LCI_USE_REG_CACHE`, `LCI_NETWORK_BACKENDS`,
`LCI_PACKET_SIZE_DEFAULT`, `RECONVERSE_ENABLE_CPU_AFFINITY`, `CMK_USE_SHMEM`,
`CMAKE_BUILD_TYPE`). SITE is `mac`, `anvil`, `delta`, `frontier` or
`generic`; extra ARGS (runtime options such as `++ipccutoff 65536` and
commbench options) are passed through.

| placement | meaning |
|---|---|
| P0 | 1 process, 2 PEs |
| P1 | 2 hosts x 1 process |
| P2 | 1 host x 2 processes |
| P2i | P2 with `+ipc` |
| P3 | 2 hosts x 2 processes; `--pairs auto` then covers every locality class |

On a Mac only P0 and P2 exist, and cross-process traffic is TCP loopback:
use them to develop, not to measure.
