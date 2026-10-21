# GPU-initiated communication for migratable device actors: research directions

2026-10-20. Brainstorm after the device-channel work (CkDeviceChannel: persistent, one-sided,
credit flow control, epochs/resolutions for migration, write with the note as immediate data,
shared-memory notes on the same host).

## Where the field is (October 2026)

Three systems cover device-initiated *transport*; all assume static placement, one rank per GPU,
preallocated buffers and no flow control.

- **NVSHMEM**: device-side puts into a symmetric heap. On Slingshot the NIC work is proxied through
  a host thread; only InfiniBand (IBGDA) is truly device-initiated. "Demystifying NVSHMEM"
  (arXiv 2606.05951, 2026) analyses the device-initiated path with DeepEP (MoE dispatch) as the case.
- **GICC** (arXiv 2604.22126, April 2026): kernels trigger NIC operations on Slingshot and
  InfiniBand. On OFI/CXI the host pre-posts deferred work (fi_control FI_QUEUE_WORK) with a
  threshold on a GPU-visible counter; a kernel rings the counter and the NIC fires the put; a host
  thread recycles NIC resources off the fast path. Application: stencil halos sent as boundary
  regions finish. Claims: up to 229x lower per-coordination latency on Slingshot, 1.95x lower put
  latency than NVSHMEM on InfiniBand, 42% vs 35.4% parallel efficiency at 64 MI250X GCDs.
- **CharminG** (Choi, Richards, Kale, HPDC 2021): the Charm++ scheduler and chares on the GPU,
  NVSHMEM transport, Jacobi 2D on Lassen. No migration, no load balancing, a separate runtime.
- Also: MPI partitioned communication with GPU-side pready; HPE's stream-triggered MPI on
  Slingshot (same CXI triggered operations); libfabric fi_trigger / deferred work queue.

The claim that distinguishes this branch: *they move the communication onto the GPU; we move the
placement decisions with it.* Every idea below attacks one of the static-placement assumptions.

## Ideas (roughly most to least novel)

### 1. Location-independent device-initiated puts
A kernel issues a put against a device-side endpoint descriptor carrying an epoch. After the
target migrates, a stale put is caught and redirected without a host round trip: the receiver's
epoch check lands it in a bounce region and a pre-armed triggered forward sends it on, or the old
location's descriptor is updated by a device-visible resolution. This is the array location
manager moved onto the device.
- New vs the field: none of NVSHMEM / GICC / CharminG has a moving target.
- Builds on: channel epochs, the resolution protocol, the older-resolution guard (2026-10-19).
- Evaluate: leanmd migration storms (750 Compute moves per LB event) with no kernel stopped.
- Risk: the hardest item; device-side forwarding needs bounce memory and ordering proofs.

### 2. Live migration of device actors
Today an LB event drains streams and closes channels with round trips (Closing -> ack -> pack ->
transfer -> unpack -> reopen -> resolve). With counters the two-phase close becomes arithmetic:
the pack starts when the departing actor's issued-put count equals the completed-put count, and
the arrival re-arms its triggered operations. An LB event costs microseconds of stream time, not
milliseconds of host time; asynchronous LB becomes asynchronous at the device level.
- Much of it is a protocol change worth doing for async LB regardless of GPU-initiated transport
  (2026-10-20, built: PERSISTENT_COMM_PLAN.md "the move window"): senders told the new location when
  the element leaves (at emigrate, NOT at the decision: between the decision and the pack the element
  keeps stepping at the old home and needs its messages there), invariant state shipped at the decision
  (PUP::DeviceLife::Stable, hashed at both ends), rebuildable state left out of the pup and priced by
  setObjRebuildCost, several migration streams so one slow element's pack does not hold the others.
  Withdrawn: pulling the state in place instead of the staging copy (removed 2026-08-30, commit
  1bbd70297: unbounded pinned memory, the memory contract is written on staging), and the
  redirect-before-arrival for channels (the new home's buffer is known only when the element unpacks).
  The channel's closing ack IS the switch marker; its two waits are measured by the tally first.
- Builds on: async LB, the memory contract / landing reserve, stream-ordered pup.
- Evaluate: per-migration window (decision -> first put at the new home) and partners' idle during
  LB events, sph3d (375 MB objects: transfer-bound) and leanmd (small objects: round-trip-bound).

### 3. Streaming partitions into persistent channels
Thread blocks publish the slab they finished (as MPI partitioned pready / GICC boundary-first)
into a migratable one-sided channel with per-partition flags; the consuming kernel starts on
partitions as they land. GICC does the sender half for one rank per GPU; the receiver half, the
flow control and 32 patches x 26 neighbours per GPU are new.
- Answers the question interior-first could not: does sph3d's round-boundary skew shrink when
  halos stream?
- Risk: partition flags in the one-buffer channel; receiver kernels that start before all data.

### 4. Trigger resources as a scheduled resource under over-decomposition
NIC counters and deferred-work slots are finite. 32 actors x 26 channels x 2 rounds per GPU exceeds
what a NIC arms at once. A channel scheduler that arms triggered operations in dependency order,
multiplexing thousands of channels over hundreds of slots, is a systems problem the
one-rank-per-GPU systems never meet. The actor model becomes the technical core here.

### 5. Device-side flow control tied to the memory contract
The symmetric heap pushes buffer management onto the user. Here receive buffers come from a pool
with a landing reserve the balancer sizes. Making the pool's admission a device-visible credit gives
GPU-to-GPU back-pressure without the host, and turns the 2026-10-09 landing/departure deadlock into
a counter invariant rather than a timeout.

### 6. Two-timescale balancing
Intra-node: an idle GPU steals ready work from a peer GPU's device-side queue through peer memory,
in microseconds, with queue wait time as the load signal (no CUPTI). Inter-node: the host balancer
migrates actors in seconds using the cost model. Nobody has combined dynamic placement with
GPU-initiated communication. Inputs: sph2d's moving dam break, moe's reshuffling hot set.

### 7. NIC-offloaded reductions with changing membership
Convergence checks and the global time step keep the host in the loop. A deferred-work chain on the
NIC can execute a reduction tree; re-forming the tree when members migrate is new, and it is what
makes a whole SPH step host-free rather than only its exchanges.

### 8. The SM cost of communication
Threads spinning on flags and packing into remote buffers take SM time from compute; under
over-decomposition that cost is real and unpriced. Measure it, add a device-initiated tier to the
balancer's cost table, and let the runtime choose per channel between host-driven and
device-initiated paths. A bounded study with a clean result.

### 9. Host and device actors in one program
CharminG is all device, Charm++ all host. Tree building and I/O stay host chares while patches are
device actors; SDAG `when` compiles to stream dependencies; messages cross between the two worlds.
A programming-model contribution the other systems cannot make.

### 10. MoE as the headline application
DeepEP dispatches to statically placed experts over NVSHMEM. Dynamic expert placement with migration
plus device-initiated dispatch sits in the hottest corner of the field; the moe mini-app with its
bit-exact PyTorch twin is already here.

## A thesis shape
Core claim: existing systems move communication onto the GPU; this work moves the placement
decisions with it.
- Chapter 1: location-independent channels (1) and live migration (2).
- Chapter 2: streaming partitions (3), flow control (5) and trigger-slot scheduling (4) under
  over-decomposition.
- Chapter 3: balancing on device-visible signals at two timescales (6), reductions (7), cost (8).
- Applications: jacobi3d (build it), sph3d (skew), leanmd (migration stress), moe (visibility).

## First step
A feasibility microbenchmark on Delta: arm one put through libfabric's deferred work queue with a
counter threshold and ring the counter from a kernel. GICC reports this works on Slingshot; whether
Delta's libfabric exposes GPU-writable counters decides whether the transport is NIC-triggered or
proxied. Every idea above works either way, with different constants. (Known: CXI 2.3.1 has no RMA
write with immediate; LCI emulates it with a zero-length message, 2026-10-19.)

## Sources
- GICC: https://arxiv.org/pdf/2604.22126
- Demystifying NVSHMEM: https://arxiv.org/pdf/2606.05951
- The Landscape of GPU-Centric Communication: https://arxiv.org/pdf/2409.09874
- CharminG: https://experts.illinois.edu/en/publications/charming-a-scalable-gpu-resident-runtime-system/
  slides: https://charm.cs.illinois.edu/charmWorkshop/slides/talk_choi.pdf
- fi_trigger(3): https://man.archlinux.org/man/fi_trigger.3.en
