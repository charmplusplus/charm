# Configurable process / PE / GPU split: implementation plan

Date: 2026-10-11. Branch: rate-aware-gpu-lb. Written from a read of the current tree
(three parallel code surveys: HAPI pool/IPC/messaging, the balancers, the apps and scripts).
Line numbers are pointers into today's files; they will drift.

## 1. The configuration space

A split is a triple (P, G, K):

| symbol | meaning | today |
|---|---|---|
| P | processes per node | 4 |
| G | GPUs per process | 1 |
| K | PEs per GPU | 4 (8, 12 tried) |

PEs per process (`+ppn`) = G x K. Constraint: P x G <= GPUs visible on the node (4 on gpuA40x4).
P x G = 4 uses the node. P x G > 4 means several processes share a GPU; HAPI's mapping allows it
(hapi_impl.cpp:2231-2234) and the plan keeps it legal, but it is not a target to optimise.

Examples that must all work from the same binary and the same scripts:

| split | +ppn | what it exercises |
|---|---|---|
| (4,1,4) | 4 | today's configuration, the control for everything |
| (4,1,8) | 8 | more PEs per GPU, no device change |
| (2,2,4) | 8 | peer copies in-process AND IPC across processes in one run |
| (1,4,4) | 16 | one process per node, no IPC on the node at all |
| (2,2,8) | 16 | the (2,2) split with the larger PE count |

Every runtime path must be written against (process, device), never against "process". The
split must appear nowhere in the code as a constant or as a formula that is only right at
G = 1 or P = 1. Section 3 lists the places where that is not true today.

### How the runtime learns the split

- G is derived today: `device_count = visible / processes_per_host` (hapi_impl.cpp:2217, the line
  with the "?????" comment). Keep the derivation, add `+gpudevs G` to override it (fewer GPUs per
  process than the node offers), print the resulting (P, G, K) at startup, and refuse a `+ppn`
  that G does not divide. No environment variables.
- Two independent assignments live in `hapiMapping` and must be separated:
  1. process -> its set of devices. Must be contiguous: process R gets `R*G + i`, for any P and G.
     Today only RoundRobin gets this right (`(G*R + i) % visible`, hapi_impl.cpp:2249); Block's
     formula `(R*visible + i) / P` (2254) collapses to one device for P = 2, G = 2 (both i give
     device 0 for R = 0, device 2 for R = 1). It is right only at P = 1 or G = 1.
  2. PE -> device within the process. Must be contiguous blocks of K: PE r -> device r / K
     (Block's PE rule, 2280). RoundRobin's `r % G` (2287) interleaves devices across adjacent PEs,
     against the array map and DiffusionLB's `node*nodeSize` addressing.
  Fix: one formula for (1), always; `+gpumap` selects only (2), and the runtime defaults to block
  whenever G > 1.
- Device representatives (the PE that creates the arena, the IPC handle, the comm buffer and
  peer access for its device): first PE of each device, `r % K == 0`. Today's `r < G`
  (hapi_impl.cpp:2282, 2289) puts every rep on device 0 under the block PE rule.

### Launch and pinning

run3d.sh grows three knobs: `PPN` (P, default 4), `GPP` (G, default 4/PPN), `PES` (G x K, default
4 x GPP). It derives `NP = NODES x PPN`, `--ntasks-per-node=PPN`, `--cpus-per-task=64/PPN`, keeps
`--gpus-per-node=4`, and passes `+ppn $PES +gpumap block`. `+gpupoolsize` is already per device
(hapi_impl.cpp:2473) and stays at its value.

The wrapper becomes a function of (R, G, K): process R owns devices R*G .. R*G+G-1; A40 d sits on
NUMA domain 3-d (reversed order, leanmd/run_numa.sh); the K PEs of device d go on cores
(3-d)*16 .. (3-d)*16+K-1, in PE-rank order:

```
# numa_wrap_sph.sh, generalised
R=${SLURM_LOCALID:-0}; G=${GPP:-1}; K=$((PES / G)); MAP=""
for i in $(seq 0 $((G-1))); do d=$((R*G + i)); n=$((3 - d)); f=$((n*16)); MAP="$MAP${MAP:+,}$f-$((f+K-1))"; done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec "$@" +pemap $MAP
```

(2,2,4) gives process 0 `48-51,32-35` and process 1 `16-19,0-3`; (4,1,4) gives today's map.
Never narrow CUDA_VISIBLE_DEVICES (numa_wrap_sph.sh:3-7: the IPC path needs every peer visible).

Why do this (what a result has to show): with G > 1 a chare moving between two GPUs of one process
is a peer copy with no shm handshake, no IPC import and no landing buffer; the balancer sees several
devices inside one process; host-side state is shared. With the split configurable, the trade
between fewer network endpoints (small P) and in-process peer moves (large G) is measurable
instead of assumed. The risks that could cancel it are in section 5.

## 2. What is already per device (build on it, do not rebuild)

- PE-to-device mapping: `GPUManager::device_managers`, `pes_per_device`, per-PE
  `my_device` / `my_device_id` / `device_rep`, `device_map` (hapi_impl.cpp:2169-2300).
- Device pool: arenas keyed by `my_device_id`, one vmm heap per (process, device) with the CUDA
  ordinal taken from `device_managers[dev].global_index` (hapi_impl.cpp:4238-4268, 4540-4584);
  one `hapi_pool_shm_entry` per (process, device) (3622-3646).
- IPC event pool indexed `device_count*NodeRankLocal + my_device_id` (3570-3669, ckrdmadevice.C:2333 ff).
- gpushm layout: one file per host, one chunk per (process, device) (3473-3567). Every process on
  the host must compute the same G (2217); the `+gpudevs` check enforces it.
- Stream pool keyed by device, streams returned to their owner (5883-6061).
- Per-PE event pools split by device; `recordEvent` switches to the stream's device
  (commit 6467f9191; hapi_impl.cpp:3028-3149).
- Flag-poll ring per PE, used only when the stream's device is the PE's (2354-2359, 3041, 6330).
- CUPTI attribution by the record's `deviceId`, normalised against that device's SM count
  (hapi_impl.cpp:1401-1407, 1699-1742).
- MetisLB, GreedyRefineCentralGPULB, the central criticality and MetaBalancer group by
  `gpu_device_id` (MetisLB.C:820-834, 946-980; GreedyRefineCentralGPULB.C:460-477; LBLoadDim.h:281-306).
- Migration: packing on the source DeviceManager's `migration_stream`, landing allocated on the
  destination PE's own device, admission check per device (cklocation.C:4931-4969, 5199-5225;
  hapi_impl.cpp:4801-4872); the memory contract is per device (LBMemoryContract.h:134-196).
- Peer access: each device rep enables access to every other visible device, on by default,
  `+gpunopeer` turns it off (hapi_impl.cpp:2660-2699).

## 3. What breaks, by layer

### 3a. HAPI runtime

| # | Site | Problem | Fix |
|---|------|---------|-----|
| R1 | hapi_impl.cpp:4479-4486 (`mapChunks`), 3910-3919 (`hapiVmmImportLocked`) | vmm chunks and imported mirrors are made accessible to ONE device (`cuMemSetAccess` with a single descriptor). Peer access does not cover vmm memory. Every same-process cross-device read of pool memory faults. | One access descriptor per local device in both places. |
| R2 | hapi_impl.cpp:2247-2256 | Block's process-to-device formula is wrong for 1 < P and 1 < G (section 1). | One contiguous formula `R*G + i` for both map types. |
| R3 | hapi_impl.cpp:2280-2290 | Reps are `r < G`, all on device 0 under the block PE rule. Devices 1.. get no comm/LB buffer, no IPC handle, no peer access; device 0 gets several reps (double create, 3451-3456). | Rep = `r % K == 0`. |
| R4 | hapi_impl.cpp:2217, 2240 | G derived with integer division and never checked; `pes_per_device = ppn/G` silently rounds down, and a PE's event-pool slice can run off the end. | `+gpudevs`, print (P, G, K), refuse `ppn % G != 0`. |
| R5 | hapi_impl.cpp:3649, 3659 | `ipc_export_cache` seeded by each rep without `ipc_cache_lock`; with G > 1 several reps per process run concurrently. | Take the lock. |
| R6 | hapi_impl.cpp:4367-4377, buddy_allocator.cpp:103 | Buddy arena `cudaMalloc` runs on the current device, not the `dev` key. | `cudaSetDevice(global_index)` around the create. Low priority (vmm is the backend in use). |
| R7 | hapi_impl.cpp:4334-4351 | Fabric registration cap is one 4096 MB total per process across all devices' arenas; G times as binding. | Decide per device or per process after Phase 2's registration test. |
| R8 | hapi_impl.cpp:2425-2428, 2768-2842 | The submitter is one per process on one device and refuses `device_count != 1`. | Leave as is. sph3d runs with `+gpusubmit` off (it loses there). Per-device submitter only if an arm ever wants it. |
| R9 | gpumanager.h:632-722 | Legacy `GPUManager::streams_` is one process-wide array created on the caller's device. | Check for callers (`hapiGetStream`); the apps use the pool. Remove or make per device. |
| R10 | hapi_impl.cpp:814, 893 | CUPTI attach/finalize `cudaDeviceSynchronize` syncs the current device only. | Loop over local devices. |
| R11 | hapi_impl.cpp:1657-1682 | Driver-busy union assumes one CUDA context per process (feeds the launch dimension). | Per-context union, or document that the launch term is per process. |

### 3b. Device messaging (ckrdmadevice.C, cklocation.C)

| # | Site | Problem | Fix |
|---|------|---------|-----|
| M1 | ckrdmadevice.C:65 `findTransferModeDevice` | Returns MEMCPY for any two PEs of the same process, whatever their devices. | Split: MEMCPY (same device) vs PEER (same process, other device). PEER = `cudaMemcpyPeerAsync` or Default with peer access, on a stream of the destination device, completion event recorded on that device (recordEvent already switches). The IPC modes stay for other processes. |
| M2 | ckrdmadevice.C:904, 937, 2427-2428, 4424-4425 | Four explicit `DeviceToDevice` copies; the branch's own comment at 1597-1601 says that kind is rejected cross-device. | Use Default / PeerAsync; make the stream match the device. |
| M3 | ckrdmadevice.C:1202-1240, 1576, 1584-1602, 3352-3355, 1788 | `cudaMemcpyDefault` with a raw pointer: fine for cudaMalloc memory under peer access, faults on vmm blocks until R1 lands. | Covered by R1; verify each site after. |
| M4 | cklocation.C:360-371 | Residency table comment and logic: "same process is same GPU". Handoff already requires `hapiDeviceForPe` to match (157-158, 4846-4855), so a cross-device in-process migration must take the staged `transport()` path (LBMemoryContract.h:248-258). | Confirm the staged path is taken and that the stage copy is a peer copy. |
| M5 | hapi_impl.cpp:6450-6452 `hapiDeviceForPe` | Valid only for PEs in the calling process. | Fine for the runtime; the balancer needs a job-wide map (L3). |
| M6 | ckrdmadevice.C:488-517, batch copies | Unchecked: the arena registration cache is process-wide keyed by address; `cudaMemcpyBatchAsync` with peer pointers. | Test in Phases 1 and 2. |

### 3c. Load balancing

| # | Site | Problem | Fix |
|---|------|---------|-----|
| L1 | DiffusionLB.C:159-183, 414-430, 460, 669-688; DiffusionHelper.C:126-143, 290-365; DiffusionCore.C:117-153; DiffusionPseudo.C:127-141; DiffusionLoad.h:26-29, 56-63 | The diffusion "node" is a process (`nodeSize = CkNodeSize(0)`); `loadDimReport` reports `gpus=1, procs=1` per node; `nodeDevSum`, `capG`, `diffusionNodeDeviceBound`, `computeNodeMemory` are per process; the within-node phase assumes all PEs of a process share one device and has no cost model or memory check. Stats carry no `gpu_device_id`. | Make the diffusion node a DEVICE: nodeSize = K, node = (process, device), addressed by its first PE (the block PE rule makes that `node*nodeSize`). Across-node machinery (ring/MST/comm neighbours, capG, memory) then works unchanged per device for every split; the within-node phase is correct again. Carry `gpu_device_id` in the stats. Derive K from HAPI's mapping, not from `CHARM_DIFFUSION_NODE_SIZE`. |
| L2 | DiffusionCostModel.h:43-50, .C:31-48, 138-151; MetisLB.C:1010-1014, 1158-1172, 1513; lbcalib.C:201; sph2d/lbcost.delta-a40.conf | `tierBetween` returns INTRA_PROCESS on `CmiNodeOf` equality before any device check; `diffDeviceOfPe` guesses devices from the config's `devices_per_host` instead of HAPI's map; `migrateCost` ignores the tier and within-node moves are never priced; MetisLB prices same-process cross-device edges as IPC_CROSS_GPU. The conf's `intra_process` tier was measured as same-GPU. | Tiers become: SAME_DEVICE, INTRA_PROCESS_PEER (same process, other device), IPC_SAME_GPU, IPC_CROSS_GPU, INTER_NODE. `tierBetween` consults the real device map. Tier-aware `migrateCost` (handoff vs peer copy vs staged IPC). Loader defaults the new key so existing confs load. Measure the peer tier with lbcalib at (2,2,K). |
| L3 | DiffusionMetric.C:22-40; DiffusionHelper.C:35-58, 381-399 | The balancer has no job-wide PE-to-device map; a neighbour's tier is computed from the two rank-0 PEs. | Every stats message carries (process, `gpu_device_id`) per PE (central already does); build the map from it. |
| L4 | LBManager.C:1348-1423 `ProcessorGPUSpeed` | Function-static cache shared by all PE threads, filled by whichever PE calls first, unsynchronised. Consumer: rateAware at TreeLevel.h:1364. | Cache per device. |
| L5 | LBLoadDim.h:291, 295 | The launch term counts processes via `CmiNodeOf`. | Decide: per process (one driver) or per context. Tied to R11. |
| L6 | hapi_impl.cpp:604-607 | `descriptor.instanceId` from `CmiPhysicalNodeID(CmiMyPe())` on the helper thread, where `CmiMyPe()` is 0 (reconverse thread_local state). Pre-existing; affects gpuScaling only. | Pass the host id explicitly. |

### 3d. Applications and scripts

| # | Site | Problem | Fix |
|---|------|---------|-----|
| A1 | sph3d.C:203-230, sph2d.C:105-131, leanmd Compute.cc:200-228 | Process-wide `cudaEvent` pools; events created on the first PE's device and recycled to PEs on other devices. `cudaEventRecord` across devices fails with InvalidResourceHandle, surfacing as a sticky error in an unrelated chare (the 2026-09-08 bug class). | Pool per device, keyed by `hapiGetDeviceNum()`; create with the PE's device current. |
| A2 | sph3d.C:1092-1101, sph3d.cu:157-168 | `__constant__ c_interior_lo` set once per process behind a racy function-static. Other devices keep the default (harmless at SPH_INTERIOR_LAYERS=2, wrong at 1). | Set once per device; drop the static. |
| A3 | sph3d.C:371-375, 1661-1673, 146, 765 | `peTier` and SPH_PRIO_BORDER use "same process" as "same GPU". | Use the partner PE's device (`hapiDeviceForPe` in-process). |
| A4 | sph3d.C:155-201, 174-180; leanmd Compute.cc:155-194 | Shared pinned slabs land on whichever NUMA domain first touches them. With G > 1 a process spans G domains. | Allocate per PE on its own domain (after Phase 1 shows a host-side regression, not before). |
| A5 | run3d.sh:34-43, numa_wrap_sph.sh | Launch shape and pinning are hard-wired to (4,1,K). | The `PPN` / `GPP` / `PES` knobs and the generalised wrapper of section 1. |

## 4. Phases

Each phase is one allocation at most, one variable, an in-session control, a named kill criterion.
A40 partition only. One runner per allocation. Nothing is submitted without the user asking.
K = 4 is held fixed through Phase 5 so the per-device numbers stay comparable with the existing
pe4 results; a K sweep is Phase 6.

| Phase | Work | Validation arms (one node unless stated) | Control | Kill criterion |
|-------|------|------------------------------------------|---------|----------------|
| 1. Split plumbing, no LB | R1-R5, M1-M2, A1-A2, A5. Build. | sph3d 128 patches (`4 4 8`), noLB, 500 steps, at (2,2,4) then (1,4,4). The startup line must print the split. | (4,1,4) noLB in the same allocation. | Any fault, any checks mismatch, or step time > 1.10 x control at either split. A 10% loss means the peer path is wrong, not the idea. |
| 2. Two nodes | R7 decision, M6 (registration of G heaps per process, batch copies with peer pointers). | 2 nodes, 256 patches, noLB, 2000 steps, (2,2,4) and (1,4,4). | (4,1,4). | Registration abort, or > 1.10 x control. This arm also measures the endpoint-count risk (section 5). |
| 3. Migration across devices | M4, L2 (tiers + pricing), L3, lbcalib measurement of the peer tier. | 2 nodes, 256 patches, MetisLB at the first event only, then no balancer, 2000 steps, sort 100, at (2,2,4): the one split where same-device, peer and IPC moves all occur in one run. | Same flags at (4,1,4). | Fault, mismatch, or a cross-device in-process move that takes the IPC path (check the transfer-mode counters). |
| 4. DiffusionLB per device | L1 (node = device), L4. | 2 nodes, 256 patches, sync, Metis once then Diffusion, period 25 lag 10, sort 100, at (2,2,4) and (1,4,4). | Sync at (4,1,4), identical flags. | Diffusion makes no within-process cross-device moves when the devices are imbalanced; or step time > control + draw band (2.6 ms). |
| 5. Async | Nothing new in theory; the flush lock now blocks G x K PEs per flush and the CUPTI buffers carry G devices' records. R10, R11/L5. | Same as 4 with `+LBAsync`, equal period and lag, plus the kernels-only trace (`n2_gputrace.sh`, 200 steps). | Async at (4,1,4). | Async slower than sync at equal period, or a flush stall visible in the trace. |
| 6. K sweep and cleanups | K = 8 at the best split; R6, R8 if a submitter arm is wanted, R9, A3, A4, L6. | (P,G,8) vs (4,1,8). | | |

Phase 1 is where almost all the risk is. If both splits pass clean at the control's step time,
phases 2 and 3 are plumbing; phase 4 is the real LB work and the reason to do any of this.

## 5. Risks and open questions

- **Expected gain is structural, not a measured number.** Today's cross-process path is already
  a peer copy through an IPC-opened pointer, so raw copy bandwidth does not change. What changes
  is the protocol (no shm handshake, no import, no landing buffer for in-process moves) and what
  the balancer can see. Phase 1 must at least tie the control at every split.
- **Fewer processes per node = fewer network endpoints.** reconverse has no comm thread; PE threads
  drive progress. (1,4,K) has one endpoint per node where (4,1,K) has four. Off-node traffic may
  serialise at small P. Phase 2 measures this; it is the reason the split must stay configurable
  rather than fixed at P = 1.
- **CUPTI volume and the flush lock scale with G.** One subscriber per process carries G devices'
  kernel records, and the exclusive flush at the window close stops all G x K PEs. Phase 5.
- **vmm plus peer access.** `cuMemSetAccess` with several descriptors on one VA range is the
  documented way; cross-device reads of vmm memory then need P2P between the A40s, which the
  cross-process IPC path already relies on. Confirm with tests/util/vmmprobe before Phase 1.
- **The 4 GB per-process registration cap** was measured on the buddy backend; vmm registers per
  chunk. Unknown whether G heaps of 32 GB in one process register at two nodes (R7).
- **NUMA.** A process with G > 1 spans G domains. The pemap handles the PEs; host allocations do
  not follow (A4). A host-side regression in Phase 1 points here.
- **GPU sharing (P x G > 4)** stays legal through the mapping's modulo but needs `+gpupoolsize`
  halved per process and a pemap of 8 cores; not measured in this plan.
- **`+gpusubmit` stays off** for sph3d (it loses there anyway). Not a blocker.

## 6. Out of scope

- A PE driving more than one device.
- NVLink topology awareness in the balancer (A40 nodes on Delta are PCIe peers).
- Changing the default: (4,1,4) stays the default and the control.

## Status

- 2026-10-11 22:30: Phase 1 code done and built (R1-R5, M2, A1-A2, A5; M1 turned out unnecessary: the same-process receive path already copies with cudaMemcpyDefault, which handles the cross-device case once R1 grants access). Chain script: sph3d/n1_split1.sh <jobid>. Validation run pending an allocation.
- 2026-10-11 22:50, job 22818398 (gpub043): **Phase 1 PASSED.** One node, 64 patches at half the z extent, noLB, 500 steps, all three splits rc=0 with 20 checks and no faults, no access fallback printed. Whole-run ms/step: (4,1,4) control 62.28, (2,2,4) 61.19, (1,4,4) 61.44. Chain: sph3d/n1_split1.sh, summary in sph3d/logs/split1_chain.out.
- 2026-10-11 22:50, same node, sync LB (Metis once then Diffusion, period 25 lag 10) with the Phase 1 runtime but the OLD balancer: all clean, 54 Metis moves at every split, but (1,4,4) 51.20 vs control 45.20 and (2,2,4) 46.92. The one-process log shows DiffusionLB saw "1 nodes": the process-as-node premise (L1). Phase 3/4 code (peer tier, device-id pricing, node = device, device-id in the stats, per-PE GPU speed) built at 22:56; rerun pending in sph3d/logs/split2b_chain.out.
- 2026-10-11 23:05, same node, Phase 3/4 build (peer tier, device-id pricing, DiffusionLB node = device, device id in the Diffusion stats, per-PE GPU speed). First attempt segfaulted at every split, control included: the tier enum grew but MetisLB inside moduleEveryLB/moduleMetisLB kept the old DiffusionCostConfig layout (header skew; rebuild the LB modules after any ck-ldb header change). With the modules rebuilt, all clean, DiffusionLB reports "over 4 nodes" at every split:

  | ms/step, 64 patches, 500 steps | (4,1,4) | (2,2,4) | (1,4,4) |
  |---|---|---|---|
  | sync, Metis once then Diffusion 25/10 | 45.55 | 45.73 | 47.15 |
  | async, same period and lag | 45.19 | 45.50 | 47.10 |
  | sync at K = 8: (4,1,8) / (2,2,8) / (1,4,8) | 45.97 | 46.48 | 47.48 |

  The (1,4,K) residual is +3.5% (was +13% with the old balancer); LB-step costs are within 0.2 s of the control over the run, and the second-half step time carries the difference (43.07 vs 44.88 ms sync), so it is steady-state, post-migration. peTier (A3) was only diagnostics, now device-aware. Host-profile diagnostic chain: sph3d/n1_split5.sh.
- 2026-10-11 23:12, kernels-only traces on the same node (gputrace_merge.py, trace_compare.py and gap_report.py now group by (rank, device); the merge writes the device into every kernel record). Sync LB, last 1 s: control busy 89.3% per GPU; (1,4,4) 85.5% (87.9% with the LB barrier excluded). The (1,4,4) residual splits in two: the sync LB step inside the window is 24-29 ms against 10-12 ms in the control (the 54 in-process cross-device migrations and the barrier in one process), and the halo-round waits (gaps ended by cellCount) are 4.9% against 3.6%. Without LB and with CUPTI active through +gputrace alone, the two tie (62.89 vs 62.47 ms/step, busy 62.6 vs 63.7%), so the tracing subscriber is not the cost. Open: why async at (1,4,4) carries the same 4% when its barrier is gone; a 2-node trace in Phase 2/5 is the next look.
- 2026-10-11 23:25, the (1,4,4) residual located. Full trace (copies + message records): with the LB barrier excluded the two layouts tie (86.8% vs 87.3% busy); the whole difference is the LB step, 68-74 ms per event inside one process against 19-28 ms in the control, every PE idle in it. Per-event CUPTI work measured with CHARM_LB_CUPTI_TIME=1: control flush 1 ms + process 4-5 ms + normalize 1 ms = 6-7 ms per process; (1,4,4) flush 4 ms + process 8-13 ms + normalize 3 ms = 15-20 ms, i.e. one subscriber doing four devices' records with every PE of the process stopped for the flush. The remaining ~30 ms of the step is the in-process event's stats, strategy and staged cross-device migrations. Twenty events of ~44 ms over 500 steps is the 1.8 ms/step residual. The in-process receive path itself hides no synchronization: sends are issued from completion callbacks, so both paths are host-gated alike, and the copy is the same peer copy; the IPC handshake it skips is ~0.1 ms per PE per step (sendHalo 0.119 -> 0.062 s per PE over the run), off the critical path. The balancer's "launch 10%" figure for one process is the per-process driver union reported by every device-node (R11/L5), an artifact.
- Next work items from this: (1) per-context CUPTI flush and parse so a device's window closes without stopping the process's other devices (cuptiActivityFlush per context under a per-device lock instead of the process-wide exclusive flush); (2) a peer handoff for cross-device migration inside a process (pack straight into the destination's landing buffer with a peer copy, no staged path); (3) R11/L5. Phase 2 (two nodes) stays next for the endpoint question.
- 2026-10-11 23:48, items from the root cause, built (validation pending, job sph3d_split6):
  1. CUPTI drain thread (hapi_impl.cpp hapiCuptiStartDrainThread): buffers are delivered every 10 ms (cuptiActivityFlushPeriod at attach, default now) and parsed as they arrive under cupti_prepare_lock_, so the window close parses only the last period. The runtime-API attribution accumulates across the drains of a window (*_acc_ fields) and is published at the close, so a PE never reads a partial round. Activity buffers are 1 MB, quarantined for 0.5 s of wall time instead of two drains, and reused through a free list.
  3. Per-device driver share: each PE thread's OS id is recorded with its device at mapping time; API intervals carry the device; hapiNormalizeCuptiLoads computes the busy union per device and hapiCuptiDriverScale() gives each PE its own device's union/total, used by DistBaseLB, CentralLB and MetaBalancer in place of the process-wide ratio. Removes the "launch 10%" artifact of one process reporting its whole driver time at every device-node.
  2. In-place unpack for a same-process cross-device migration: NOT done. Measured bound: the Metis event with 54 moves took 551 ms in one process against 530 ms in the control, so the staged copy is not in the residual; skipping it would save one peer copy of the packed bytes per move (tens of ms per event at this size). Design, if wanted: CkDeviceBufferPost gets an in-place flag; immigrateGPU sets it for a same-process source (the packed pool block is readable from every local device since R1); the Memcpy receive kind then delivers the source pointer without a landing copy; immigrate skips the landing free and releases any pre-granted landing set; the source's block is freed by the existing same-process ack (finishGPUSend). Touches ckrdmadevice.C receive, cklocation.C immigrate and the admission accounting.
- 2026-10-12 01:04, job 22821909 (gpub072), the drain-thread + per-device-share build, 64 patches, 500 steps:

  | ms/step | (4,1,4) control | (2,2,4) | (1,4,4) |
  |---|---|---|---|
  | sync, Metis once then Diffusion 25/10 | 45.22 | 45.88 | 47.69 |
  | async, same period and lag | 45.02 | 45.17 | 45.72 |

  Async (1,4,4) went from +4% to +1.5%; sync did not move (+5.5%). The close still costs 14.9 ms in one process against 4.0 in the control (flush 4.1, process 7.5, normalize 3.2), and the launch share still prints 9% against 3%: either the periodic delivery is not reaching the drain thread (only full buffers are delivered while a kernel is running) or the API records are not being tied to their device (CUPTI thread id type). Diagnostics added: cuptiSetThreadIdType(SYSTEM), buffers parsed at the close vs by the drain, per-device api total and busy on the timing line. Under sync the second-half step is 45.47 vs 42.90 ms with the LB step itself accounting for only 0.56 ms/step of that, so the rest is steady-state between events, i.e. while CUPTI is attached: an A/B with CUPTI never attached (CHARM_LB_NO_CUPTI=1) is running to settle whether CUPTI's per-call intercept in one 16-thread process is the cost.
- 2026-10-12 01:07, A/B with CUPTI never attached (CHARM_LB_NO_CUPTI=1, sync LB on host loads only, 500 steps): control 52.33, (1,4,4) 54.41 ms/step (second half 50.29 vs 52.85). The one-process steady-state cost under load balancing is NOT CUPTI's per-call intercept. It appears only once the balancer has moved patches (noLB ties), so the suspect is the per-message cost of the in-process cross-device exchange path after Metis's cut replaces the block layout's: the trace showed the halo-round waits at 4.9% vs 3.6% and an envelope-latency tail 5x longer. Measuring it directly with the device-message latency benchmark: an in-process cross-device pair (1 process, 2 GPUs) against a cross-process IPC pair and an in-process same-device pair.
- 2026-10-12 01:13, device-message latency benchmark (zerocopy, 200 iterations), gpub072:

  | bytes | in-process, two GPUs | two processes, IPC | in-process, one GPU |
  |---|---|---|---|
  | 64 KB | 33.8 us | 56.1 us | 30.0 us |
  | 1 MB | 81.8 us | 80.4 us | 81.2 us |
  | 4 MB | 210.7 us | 211.5 us | 206.3 us |

  The in-process cross-device path is not slower per message; at small sizes it is 20 us faster (the handshake it skips). So the one-process steady-state cost under load balancing is neither CUPTI nor the exchange path. Drain thread diagnostics: CUPTI delivered no buffer between closes even with cuptiActivityFlushPeriod set (16 buffers arrived at each close), so the drain thread now issues a non-forced flush every ~20 ms under the flush lock, timed; three alternating sync pairs are running to get the noise band of the remaining difference.
- 2026-10-12 01:24, drain thread REVERTED (see memory cupti-drain-thread-dead-end: CUPTI delivers full buffers only at a flush; a non-forced flush from the drain thread cost 11-17 ms each under the flush lock and corrupted the heap through returned incomplete buffers). Kept: per-device driver share, thread id type, accumulate-and-publish of the API attribution. Re-validation on gpub072, 500 steps:

  | ms/step | (4,1,4) control | (1,4,4) |
  |---|---|---|
  | sync, two alternating pairs | 44.94 / 45.30 | 47.65 / 47.18 |
  | async | 45.18 | 46.32 |

  CUPTI close 16.5 vs 4.3 ms per event (the parse of four devices' records, 0.5 ms/step over the run). The "launch 12%" was my own loadDimReport change (procs = CkNumNodes, one process): the driver term is per CUDA context, one per device-node, so each report counts one again. The remaining sync residual (~2 ms/step steady state between events, async ~1 ms) has now been cleared of CUPTI (A/B without it: same gap), the exchange path (latency benchmark: in-process is faster at small sizes, equal above 1 MB) and the migration path (Metis event 447 vs 480 ms). Running sync at period 100 against 25 to separate per-event from per-step cost.
- 2026-10-12 01:28, the sync residual LOCATED (period 100 vs 25, 500 steps, gpub072): between events the one-process layout runs at 42.8-43.1 ms/step against 41.7-42.9 (+0.7 ms, 1.6%); the 25-step block holding an LB event runs at 45.7-47.4 against 42.9-44.2, and the AtSync-to-resume time per event is 65-97 ms in one process against 22-46 (period 100; at period 25: 20-28 vs 7-16). The per-event cost scales with the window length, i.e. with the records the close parses: one thread parses four devices' records (16.5 ms per close at period 25 against 4.3; roughly four times that at period 100). So: whole-run +2.4% at period 100 and +5 to 8% at period 25 is almost entirely the serial CUPTI parse at the close, times the number of events. The remaining 0.7 ms/step steady-state is the next candidate for A4 (host allocations across four NUMA domains). Launch share now 3% at every split (per-context driver count restored).
- NEXT (item 1 done right): parse the close's buffers in parallel, one thread per device: cuptiBufferCompleted records the buffer's context, the close groups the queue by context, each group is parsed into a per-thread accumulator (correlation maps, per-object kernel records, API raw/intervals, counters) and merged under cupti_prepare_lock_; the forced flush stays one per process. Expected: the close back to the control's 4 ms, the (1,4,4) sync gap to ~1.5%. Then item 2 (in-place unpack) and A4.
- 2026-10-12 09:40-09:47, job 22846356 (gpub098), in order:
  - Parallel CUPTI parse, first by context: CUPTI's completion callback hands every buffer a null context, so one shard. Reworked as round-robin shards over the buffers (one per device), pass 1 on persistent worker threads, pass 2 serial with cross-shard correlation lookup: one-process close 16 -> 11.7 ms (flush 4.3, parse 4.3, normalize 3.1); control 4.2. Sync (1,4,4) 46.91/46.96 vs control 45.24/45.83.
  - CUPTI buffer reuse after the two-drain quarantine (no free/mmap churn): no change in the steady state.
  - Per-PE pinned host pools with owner-routed frees (A4): sync 47.30/47.30 vs 45.33/45.39, async 46.27 vs 45.78: at most 0.4 ms, within noise. Kept (correct by construction).
  - Batched landing copies: zero fallbacks in both layouts; per-message runtime host cost equal (recv-issue 1.3 us both; send 0.9 us -> 0.0 in-process). Kernel time per device per step equal (51.4-51.8 vs 51.1-51.9 ms). The one-process gap is GPU idle: halo-round waits, with the in-process envelope latency tail 5x longer (p99 2.0 vs 0.36 ms). Thread count is not the variable: K=2 (8 threads) shows the same +2.3%.
  - Traced per PE: the entries that land device messages take 3x longer per step in one process (receiveHalo 1.2-1.6 ms vs 0.5; receiveParticles 0.9-1.1 vs 0.2) while the runtime's own receive-issue timer is equal, so the time is in message pickup/dispatch, not in the issue. The reconverse per-PE queue is lock-free (moodycamel, MPSC) whose try_dequeue rotates over per-producer sub-queues one element per call: 15 producers per PE in one process against 3. Testing a bulk-staged dequeue (try_dequeue_bulk into a consumer-owned buffer).
- 2026-10-12 09:55, reconverse: every PE drives LCI progress each scheduler iteration (and unconditionally when idle); `+backend_poll_thread 4` and `+backend_poll_freq 8` on (1,4,4) changed nothing (47.00 / 46.85 vs 47.01, control 45.77), so progress-lock contention is not the one-process cost either (the idle path now honours the knob). The bulk-staged PE queue dequeue changed nothing (to be reverted). LCI devices: `+lci_ndevices N` (reconverse comm_backend_lci2.cpp:183), default ONE per process, PE threads assigned to devices in contiguous blocks of ceil(PEs/N): the control has four LCI devices per node, a one-process layout one. Irrelevant to (1,4,K) on one node (no message leaves the process) but it is the endpoint-count risk of Phase 2 and it touches (2,2,K) through shared memory: run3d.sh will pass `+lci_ndevices $GPP` so every split keeps one LCI device per GPU group. Testing (2,2,4) with and without it on the last node minutes.
- 2026-10-12 10:01, end of the one-node cycle (gpub098, every change in place: sharded parse, buffer reuse, per-PE pinned pools, per-device driver share, idle-path poll knob; the bulk queue dequeue reverted, no effect):

  | ms/step, 64 patches, 500 steps | (4,1,4) control | (2,2,4) | (1,4,4) |
  |---|---|---|---|
  | async, alternating x2 | 45.33 / 46.97 | 45.25 / 45.79 | 46.29 / 46.89 |
  | sync, alternating x2 | 45.24 / 45.83 | 46.27 / 46.29 | 46.91 / 46.96 |

  Async: (2,2,4) is within noise of the control and beat it in both pairs; (1,4,4) is within 1%. Sync: (2,2,4) +1.5%, (1,4,4) +2.5 to 3.7%, of which the LB step itself (one-process CUPTI close 11.7 vs 4.2 ms, AtSync-to-resume 0.91 vs 0.64-0.76 s over the run) is about 0.5 ms/step; the rest is GPU idle between events that no instrumented runtime cost explains (per-op receive, send, batch, queue, progress, kernel time all equal or better in-process). `+lci_ndevices 2` on (2,2,4) changed nothing at one node (46.27 vs 46.29 sync), as expected with almost no cross-process traffic. Node released by expiry.
- NEXT: Phase 2 at two nodes (256 patches, the real configuration), the same three splits, noLB / sync / async, with one LCI device per GPU group. That is where a process count below four per node changes the network side, and where the in-process layout either wins or loses for a structural reason.
- 2026-10-12 10:28, PHASE 2, two nodes (gpub[034,090], job 22847020), 256 patches, 98M particles, 1000 steps, one LCI device per process (the default):

  | ms/step | (4,1,4) control | (2,2,4) | (1,4,4) |
  |---|---|---|---|
  | noLB | 56.50 | 55.46 | 56.43 |
  | sync, Metis once then Diffusion 25/10 | 41.47 | 43.68 | 51.77 |
  | async, same period and lag | 39.13 / 38.50 | 41.38 | 45.89 |

  Without load balancing the splits tie (the block map's cross-node cut is one face). With it, Metis's cut puts 230-240 patches' traffic across the fabric and the one-process layout loses 18-25%: all of a node's off-node device traffic goes through ONE LCI device (one endpoint, one completion queue, 16 PEs) against the control's four per node. That is the endpoint risk measured. The first attempt with `+lci_ndevices 2/4` aborted at startup: the device-message descriptor carries one LCI rmr per device and CMK_NOCOPY_DIRECT_BYTES was 32 (one rmr). Raised to 128 in reconverse/include/charm-config.h and conv-rdma.h (copied into the build's include/), full rebuild, rerun with LCIDEV=1 (run3d.sh adds `+lci_ndevices $GPP`).
- 2026-10-12 10:39, PHASE 2 with one LCI device per GPU group (`+lci_ndevices $GPP`, CMK_NOCOPY_DIRECT_BYTES 128, full rebuild; the first rebuild missed libckmain/libtrace-converse/libckqt and the app objects, which segfaulted every arm until the full build and a clean app rebuild). Two nodes, 256 patches, 1000 steps, same nodes as the control arms above:

  | ms/step (second half) | (4,1,4) control | (2,2,4) | (1,4,4) |
  |---|---|---|---|
  | async | 40.01 (39.02) | 39.07 (37.73) | 46.23 (44.11) |
  | sync | 42.05 (41.43) | 41.30 (40.30) | 41.22 (39.42) |

  (2,2,4) beats the control in both modes (-2.3% async, -1.8% sync; -3.3% / -2.7% on the second half); (1,4,4) beats it in sync (-2.0%, -4.9% second half) and loses in async (+15%). The single LCI device per process was the whole two-node loss for (2,2,4) and the sync (1,4,4). The async (1,4,4) residual is the next target: the async LB work of four device-nodes in one process (the exclusive CUPTI flush stalling 16 PEs mid-step, four strategies and their migrations through one helper thread and one pool) overlaps the steps instead of the barrier hiding it. Second round running with per-event timing.
- 2026-10-12 10:52, third two-node round (alternating, no extra instrumentation): async control 37.96 / 39.75, (2,2,4) 42.43 / 41.61; sync control 39.33, (1,4,4) 43.51, (2,2,4) 43.15; async (1,4,4) 45.28. Against round one (control 40.01 / 42.05, (2,2,4) 39.07 / 41.30, (1,4,4) 46.23 / 41.22) the same arms moved by 3 ms between rounds on the same node pair, so at two nodes the splits are within a +/-5% drift band of the control except async (1,4,4), which is 15% slower every time. No "faster" claim survives the drift yet; a cycle of alternating repeats on a fresh pair is needed once the per-device window close is in.
- IN PROGRESS: the async (1,4,4) fix, the balancer's work partitioned per device inside a process: (1) one CUPTI flush gate per device and a per-context flush (cuptiActivityFlush) at each device group's window close, so the other devices' PEs keep launching; (2) the window close per device group in DistBaseLB and CentralLB (arrival count = PEs per device, loads built, normalised and cleared per device); (3) one LB helper thread per device (lanes). Still to do: per-device migration admission and pool locking.
- 2026-10-12 11:50, while job 22849462 (two nodes, the per-device window close) waits in the queue: the two-node penalty of (1,4,4) decomposed from the round-three logs (async, 1000 steps, 44.8 vs 39.4 s): the CUPTI attach costs 0.2 s once (wall at step 25: 1.6 vs 1.4 s; 1.5 s for (2,2,4); equal to the noLB arms' 1.4 s without it), the Metis event at step 25 costs 0.8 s more (wall 25->50: 2.1 vs 1.3 s; sync 1.8 vs 1.4), and the rest -- 4.4 s -- is a steady 43.3 vs 38.6 ms/step from step 50 to the end (sync: +3.3 ms/step) through 38 Diffusion events whose max ratio is 0.017 in both layouts (same balance). Without LB the three layouts tie at every point of the run (55-58 ms/step), so the steady penalty belongs to the balancer's presence: the window close (22-27 vs 3-8 ms per event, about 1 ms/step), and 3-4 ms/step unexplained -- the traced steps (10 of every 25) or the in-process landing path (one-node traces: entries landing device messages 3x longer per step in one process). The runtime API totals per device at the close are alike in the two layouts, so the driver is not slower per call. CUPTI 13.2's activity buffers are already per thread (CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER defaults to 1): not a lever.
- Instrumentation added (CHARM_LB_CUPTI_TIME only, no behaviour change; libhybridapi + sph3d relinked 11:50, before the job starts): the attach time, and per device at each close the launch-path tallies -- driver calls through the gate, how many waited for a flush and how long, correlation pushes and their cost -- plus the drain's record counts. Follow-up chain n2_devclose2.sh (waits for devclose_chain.out's done marker, same nodes, same binary): LB without CUPTI (CHARM_LB_NO_CUPTI) s14/ctl; +LBDebug 2 per-call API breakdown s14/ctl; +gputrace last second s14/ctl (busy %, gaps, message latency by tier); alternating plain async.
- 2026-10-12 14:51-15:00, job 22849462 (gpub[055,090]) granted 14:50 after 3.5 h in the queue; the per-device window close validated, all arms rc=0, checks clean:

  | ms/step (second half) | (4,1,4) control | (2,2,4) | (1,4,4) |
  |---|---|---|---|
  | async, pair 1 | 40.15 (38.98) | 38.66 (37.26) | 45.47 (43.36) |
  | async, pair 2 | 40.28 (39.25) | 38.86 (38.18) | 45.99 (43.99) |
  | sync | 42.11 (41.19) | 40.73 (39.50) | 47.38 (45.54) |

  (2,2,4) beats the control in every pair, async and sync (-3.5% / -3.3%), and async beats sync in every layout. (1,4,4) still loses 13% in both modes; the per-device close did not move it (close 10.1 vs 8.0 ms per event). Alternating plain async afterwards (devclose2_chain.out): control 38.35 / 39.54 / 41.12, (1,4,4) 45.38 / 47.62.
- 2026-10-12 15:00, the (1,4,4) penalty FOUND with the launch-path tallies (CHARM_LB_CUPTI_TIME): tracing is attached from the first LB step to the end of the run, and every entry method pays a CUPTI external-correlation push and pop (~600 per PE per step). Per device per LB period the pushes cost 0.45-1.0 s in the 16-PE process against 0.04-0.32 s in a 4-PE process (same counts, 60-80k): 2.5-15 us per call against 0.5-4 us, i.e. 4.5-10 ms of every step per PE against 0.4-3 ms. The gate waits (0-3 per period), the device pool (lock waits 1 ms per period, no event fallbacks, no parked frees) and the close itself (10 vs 8 ms per event) are not it. The no-CUPTI arms (CHARM_LB_NO_CUPTI=1) are not a usable A/B: with host-only loads Metis moves nothing and Diffusion keeps migrating by host time, 88-91 ms/step in both layouts.
- 2026-10-12 15:13, launchcost in the split's own shapes on gpub055 (devclose3_chain.out), us per driver call (launch), CUPTI off / records only / records + push-pop:

  | shape | off | records | records + push/pop |
  |---|---|---|---|
  | one process, 4 threads x 1 device | 2.62 | 4.20 | 4.95 |
  | one process, 16 threads x 1 device | 13.66 | 16.44 | 30.83 |
  | one process, 16 threads x 4 devices (the (1,4,4) shape) | 4.65 | 7.13 | 32.75 |
  | four processes x 4 threads x 1 device each, concurrent (the control's shape) | 2.6 | - | 5.0 |

  The push/pop is the contention: 25 of the 33 us in the 16-thread process, process-wide, independent of the device count (CUPTI keeps the per-thread correlation stacks behind one lock). Without it the one-process shape pays 7.1 vs 5.0 us per call (2 us x ~340 gated calls per PE per step = ~0.7 ms/step), and 4.65 vs 2.6 with CUPTI off (hidden under the GPU in the noLB tie).
- FIX BUILT 15:14 (hapi_impl.cpp, CuptiEntryLog): no CUPTI call per entry. The PE appends a span {start, end, token} to its own log (two cuptiGetTimestamp calls, a vector push under its own uncontended mutex; nested entries keep a parent link), the window close copies each thread's log once and joins every runtime API record to the span of its thread containing the record's start, and the API record's correlation id names the kernel exactly as the EXTERNAL_CORRELATION record did (same tables, deferral and one-drain carry-over). The EXTERNAL_CORRELATION kind is no longer enabled; the work-tag lane (CUSTOM0) and the submitter use the same log. Logs are pruned at each PE's window-close arrival to the spans closed after the arrival two windows back. Validation running on the same allocation (n2_spanlog.sh -> logs/spanlog_chain.out): attribution check first (device share of the measured load, Metis moves), then alternating async control / (1,4,4) / (2,2,4), then sync.
- OPEN (seen in the Dbg arms): under the per-device close the records land unevenly across the device closes in the one-process layout (PE 0's device: api=57541 kernels=15616 at the first close, then 1762/604, 525/83, while a 4-PE process sees 12-28k/3.7-6.2k every close): the per-context flush does not confine the delivery to the closing device. The balance quality came out equal (Diffusion max ratio 0.017 both layouts), so the per-device build tolerates it, but the routing should be checked before the per-device close is trusted further.
- 2026-10-12 15:15-15:40, the span log validated on the same pair (spanlog_chain.out, spanlog2_chain.out; every arm rc=0, checks clean, attribution intact: device share of the measured load 54-83%, Metis moves 220-246 as before). Async ms/step (second half), arms in time order:

  | | (4,1,4) control | (1,4,4) | (2,2,4) |
  |---|---|---|---|
  | before (14:51-15:12) | 40.15 / 40.28 / 38.35 / 39.54 / 41.12 | 45.47 / 45.99 / 45.38 / 47.62 | 38.66 / 38.86 |
  | span log, CUPTI clock (15:14) | 38.74 / 39.18 | 43.99 (timed) / 40.98 / 42.25 | 41.48 |
  | span log, calibrated clock (15:19) | 38.88 / 39.11 / 40.61 / 38.44 / 38.22 / 38.82 / 40.26 / 39.00 | 41.73 / 41.51 / 44.40 / 42.88 (timed) / 42.70 / 42.04 / 42.62 / 42.70 / 42.46 | 40.46 / 39.82 / 40.11 / 39.61 |
  | sync, calibrated clock | 40.70 | 41.46 | - |

  (1,4,4) gained 4 ms/step (46.1 -> 42.3 mean) and trails the control by 3.5 ms async (9%) and 0.8 ms sync (2%); the control gained 1 ms (it paid the push/pop too); (2,2,4) sits 1 ms above the control now where it was 1.5 below (five arms, 39.6-41.5). The span clock: CLOCK_REALTIME, offset 0 / -244 ns (CUPTI's timestamps are CLOCK_REALTIME here).
- 2026-10-12 15:38, NODE FAULT FOUND: gpub090 runs on the HPET clocksource (tsc not even listed as available; gpub055: tsc). A clock_gettime there costs 1.7 us against 0.19 (python loop, both cores) and HPET reads serialise node-wide. Every timestamp pays it: CUPTI's two per API record, CmiWallTimer per entry in the LB instrumentation, the span log's. It is why every per-entry cost printed bimodally by node (CUPTI push 0.5 vs 4 us in the control, span open+close 0.2 vs 6.3 us), and it inflates both layouts alike (16 PEs per node either way), so it does not explain the remaining (1,4,4) gap, but it makes the pair slow and noisy. The chain now prints each node's clocksource at the start (n2_spanlog3.sh).
- 15:40, built (libhybridapi + sph3d, sph2d, leanmd relinked 15:41): one clock read per entry (an outermost span runs until the next span starts on its thread; only nested spans are stamped at close; prune keeps the span straddling the cutoff), and the close's span lookups run-cached per thread (records of a thread arrive in time order) with the snapshot copies shared by the shards -- the close had grown from 10 to 27 ms per event (flush 7-11 + process 7-19 ms), 1.1 ms/step at period 25. Validation: a 300-step timed (1,4,4) arm on the last minutes of job 22849462 (spanlog_quick.out), then a fresh pair (n2_spanlog3.sh -> spanlog3_chain.out, queued 15:42).
- REMAINING (1,4,4) candidates after the span log, ~3.5 ms/step async: the close (27 -> ? ms per event, on the critical path for the device group that closes last), the CUDA runtime's own 16-thread contention (7.1 vs 5.0 us per call, ~340 gated calls per PE per step = 0.7 ms), the records landing unevenly across the per-device closes (see OPEN above), and whatever the +gputrace timeline shows once the first two are out.
- 2026-10-12 15:50, HYPOTHESIS for the remaining async gap (3.5 ms async, 0.8 ms sync): the forced CUPTI flush at each device's window close stalls every CUDA call of the PROCESS for its duration (CUPTI blocks its launch callbacks while a flush is in progress; the flush waits for the device's queued kernels). One process per node closes four devices per event, 7-11 ms each (flush= in the timed closes, 0-8 ms in a 4-PE process): ~40 ms of process-wide launch stall per event = 1.6 ms/step, against 0.2 in the control, and sync hides it at the barrier. Test on the pending pair (job 22853463, n2_spanlog3b.sh -> spanlog3_chain.out): CHARM_LB_CUPTI_PERIODIC=10 (existing switch: CUPTI delivers its buffers on its own thread every 10 ms, the close forces nothing, loads lag <= 10 ms) on (1,4,4) and the control, timed, then alternating ctl / s14 / s14-periodic / s22-periodic, sync, and alternating. If it holds, periodic delivery becomes the default (no new switch).
- 2026-10-12 18:07, job 22853463 granted (gpub[056,072], BOTH on the tsc clocksource). One clock read per entry costs 0.15 us (pushes 68667 in 0.0101 s); the (1,4,4) attribution check: device share 91%, Metis 227 moves, 41.25 ms/step (second half 39.82) against the control's 38.58 (37.61) right after; a plain (1,4,4) arm 41.76 (40.49). The close: 29.6 ms mean over 320 (flush 0-23 ms + process 1-17 ms; it varies by which device's buffers the forced flush has to wait for).
- CHARM_LB_CUPTI_PERIODIC=10 is DEAD: CUPTI's worker thread delivered no buffers at all in 1000 steps (process=0.000s at every close, device share 0%, Metis 0 moves, 67-78 ms/step in both layouts as with no device loads). The hypothesis (the forced flush's process-wide stall, four per event in one process) stays untested by that route; the soft flush was ruled out earlier (blinds at 256 KB, stalls at 16 KB). Next evidence: the +gputrace timelines with the span-log binary (n2_spanlog5.sh): four simultaneous all-GPU gaps per event on the one-process node against one per process would show it; otherwise the remaining gap is elsewhere (in-process message pickup, the CUDA runtime's 16-thread contention).
- 2026-10-12 18:17-18:40, job 22853463 (tsc pair), eight alternating async rounds + sync, ms/step whole run (second half):

  | | (4,1,4) control | (1,4,4) | (2,2,4) |
  |---|---|---|---|
  | async, 8 rounds, mean | 38.82 (37.8) | 42.56 (41.2) | 39.32 (38.3) |
  | sync | 40.62 (39.95) | 41.25 (39.72) | 41.29 (40.16) |

  (2,2,4) ties the control in async (+0.5, within the round-to-round spread) and sync (+0.7). (1,4,4) ties in sync (second half 39.72 vs 39.95) and loses 3.7 ms in async -- and is the only layout whose async is slower than its sync (42.6 vs 41.3). The trace (last second of a 300-step run) put the one-process node's GPUs at 20.7% busy against 57.0%, with all PEs idle through a 261-370 ms gap per rank at the LB event against 121-157 in the control.
- THE REMAINING PENALTY IS THE LB EVENT, not the steps (from the +LBDebug lines, 38 events each): the lag window, AtSyncStart to resume, is 0.441 s on (1,4,4) against 0.373 on the control and 0.389 on (2,2,4) -- 68 ms more per event, which is the whole 75 ms-per-period gap; the balancer's own measured interval is SHORTER on (1,4,4) (0.84-0.89 s per 25 steps against 0.89-0.94: between events the one-process layout steps faster). The objects never wait at the barrier (AtSyncWait entered = resumed in both layouts); the steps inside the lag window run slower. Sync: resume 0.088 vs 0.046-0.058 s, +30 ms per event = the +0.6 ms/step. Per-device closes (timed, async): flush 13-43 ms each (the forced flush waits for that device's queued kernels) + parse 2-18 ms, and the four closes of a process were SERIALISED by cupti_prepare_lock_ (taken before the flush) although each runs on its own helper lane: 4 x (flush + parse) ~ 95 ms per event against ~20 in a one-device process. Also seen: the launch dimension is blind on (1,4,4) (launch 0-1% of the interval, drv=0 per object, against 6-8%): the per-object API time is lost when the records land in another device's drain -- open, not performance-relevant (the step dimension is used).
- FIX BUILT 18:52 (hapi_impl.cpp hapiPrepareCuptiLoadsImpl): the forced flush runs outside cupti_prepare_lock_, so the device groups' flushes wait concurrently and only the parses serialise (the first takes the bulk). Validation on the last minutes of job 22853463 (n2_spanlog7.sh -> spanlog7_chain.out): s14, ctl, s14, s22 async, timed.
- A tooling note: the Monitor tool's tail -f pipelines ending in `cut` delivered no events (cut block-buffers); end such pipelines with a line-buffered stage (stdbuf -oL cut, or grep --line-buffered alone).
- 2026-10-12 18:52-18:56, the concurrent flush (binary 18:52) on the last minutes of job 22853463, async, timed arms, ms/step whole run (second half), lag window AtSyncStart-to-resume:
  - (1,4,4): 39.75 (37.69), resume 0.394 s -- beats the control's second half; then 43.05 (41.22), resume 0.428 s.
  - control: 39.14 (38.43), resume 0.379 s. (2,2,4): 39.29 (38.39), resume 0.385 s.
  The per-device closes now overlap their flushes (25 ms each, concurrent), and the serialisation moved to the parse under cupti_prepare_lock_: the later closers' "process" time is 17-32 ms (waiting for the first closer's parse of the bulk, then their own). The (1,4,4) event cost is now sometimes equal to the control's (0.394 vs 0.379) and sometimes +50 ms; the two arms disagree by 3.3 ms/step, so more samples are needed (three more arms queued on the final minutes: n2_spanlog8.sh). Next lever if the variance holds: shrink the parse (snapshot only the spans since the previous close; per-device parse in parallel rather than one closer taking the bulk).
- OPEN, diagnosed 19:00: the launch dimension is blind on multi-device processes because hapiNormalizeCuptiLoads(onlyDevice) publishes and resets the PROCESS-wide API accumulators (cupti_obj_api_raw_acc_, cupti_api_total_acc_, cupti_api_total_dev_acc_, cupti_api_intervals_acc_) at every device's close: the first closer of an event publishes everything, the next closers publish a near-empty map, and the objects of the later devices (and of the first, once the map is replaced) read drv=0. Fix (not yet built, kept out of the concurrent-flush validation): publish per device after the sweep -- move the acc entries of the objects g_cupti_obj_device places on onlyDevice, the device's own intervals (union -> cupti_driver_busy_dev_[d]) and its api_total_dev; leave the other devices' entries accumulating; drop entries of objects with no device after 2 x device_count closes. Low performance stake (the step dimension is used; drv is 6-8% of the interval, lost uniformly), a correctness defect of the per-device close.
- 18:56-18:59, three more arms (same binary): (1,4,4) 40.73 (38.04) resume 0.408 s, control 38.11 (37.54) resume 0.370, (1,4,4) 42.20 (40.12) resume 0.422. With the concurrent flush, over four (1,4,4) arms: 41.4 whole run (39.3 second half) against the control's 38.6 (38.0); lag window 0.413 s against 0.375 (+38 ms per event, was +68). Two of the four arms are within 0.6-1.3 ms of the control on the second half, two are +2.7-3.7. What remains per event is the parse chain under cupti_prepare_lock_: the closes of one event read process= 17, 28, 30, 32 ms (each later closer waits for the earlier parses, then parses its own device's buffers, delivered by its own flush after the first closer took the queue), and the parse itself grew with the span log's snapshots (every drain copies every thread's log, 16 x ~1.5 MB, four drains per event).
- NEXT (building now): incremental span snapshots -- the drain keeps each thread's copy between drains and appends only the spans logged since (absolute span indices survive the owner's pruning; the spans still open at the previous copy are re-copied), so the second to fourth closers of an event copy kilobytes, not 24 MB. Then validation on the queued pair (n2_spanlog9.sh). The launch-dimension publish-per-device fix stays out of that binary until the concurrent flush + snapshots are measured.
- 19:02, BUILT: incremental span snapshots (hapi_impl.cpp CuptiLaneCopy / hapiCuptiLaneRefresh; CuptiEntryLane.pruned carries the absolute index); libhybridapi + sph3d, sph2d, leanmd relinked 19:02. Job 22853463 ended 19:02; n2_spanlog9.sh submits the next two-node request and runs: clocksource, 8 alternating async rounds ctl / s14 / s22, sync x3, soft-flush check s14 / ctl, alternating ctl / s14 -- all with +LBDebug 1 and the lag-window (resume) mean in each summary line.

## 2026-10-13 04:20 — job 22856533 (gpub[002,091]; 091 on HPET) read, two more root causes, fixes built, job 22866653 queued

**Results (sph3d 256 patches, 1000 steps, Metis once then Diffusion 25/10).** Sixteen alternating async rounds:
control 39.72 ms/step whole run / 38.97 second half / lag window 0.387 s; (1,4,4) 41.39 / 39.57 / 0.408; (2,2,4) (eight
rounds) 40.11 / 38.98 / 0.394. Sync: control 40.5, (1,4,4) 45.8 then 43.1, (2,2,4) 42.2. No balancer: (1,4,4) 56.5 vs
control 56.8 (tie; the whole gap is LB-event cost; a +0.2 s startup difference at step 25 exists without LB too).
Soft flush at the close: (1,4,4) 45.0, dead end confirmed.

**Decomposition of the (1,4,4) async gap (+1.67 ms/step = 1.67 s):** the Metis window (event 1, "LB at step 25:
resumed") 1.42 / 1.12 / 0.96 s against the control's 0.59 (+0.4-0.8 s); events 2-39 +21 ms each (+0.8 s); between
events the layout steps 0.4 ms FASTER than the control.

**ROOT CAUSE 3 — the Metis window is pool chunk registration.** Every 256 MB vmm chunk run registers with the fabric at
its first off-node transfer (ckrdmadevice.C acquireArenaRegistration): 449 registrations per run in BOTH layouts, 10 ms
median, 268 ms max, 8.2 s summed (timed build). The control registers 274 of them during steps 1-25 (its same-node halo
exchange crosses processes and takes the fabric path) and 147 in the Metis window; the one-process layout, whose
same-node exchange is in-process, registers 98 before step 25 and 301 inside the Metis window -- under one process-wide
mutex held across the fabric call. Fixes built: (a) the fabric call runs with the lock released (busy set + condvar;
single sample: window 0.957 s); (b) registration AT THE MAPPING: `hapiDevPoolSegmentHook` (hapi.h) is set by ck-core to
`ckRegisterPoolSegments`; the heap queues each mapped chunk run (and each buddy arena) and the pool lock's destructor
hands them over after unlocking; one summary line per heap ("registered for RDMA at the mapping"); nothing registers on
one node (CmiNumPhysicalNodes() <= 1). Expected: startup +1.3 s per heap (128 x 10 ms), no registration inside any step.

**ROOT CAUSE 4 — the drain grows over the run.** With CHARM_LB_CUPTI_TIME the close line now carries wait / retire /
pass1 / pass2 / merge: the first parse pass of the same ~70-90k API records grows 9 -> 33-38 ms from event 1 to 40 in
the (1,4,4) layout (6 -> 17 in the control, four times fewer threads); closers 2-4 wait on cupti_prepare_lock_ behind it
(24 ms mean, 85 max); empty closes cost nothing. The prune (hapiCuptiLanePrune) could not pass an open span and the
refresh re-copied from the earliest open one, so one span left open on a thread pins its log and every drain copies all
of it. Fix built: the prune compacts past open spans (kept at the front, parents remapped, a generation bump makes the
copies rebuild once), keeps the closed ancestor chain of the boundary span, and the refresh patches only the end stamps
of spans that were open at the previous copy. Host test (scratchpad spantest, the functions extracted from
hapi_impl.cpp): 619k lookups identical to the never-pruned log with and without a permanently open span; the lane stays
at two windows. Diagnostic lines (CHARM_LB_CUPTI_TIME): per-arrival "log: spans= open= bottom_open= pruned= gen=",
close line "snap=<s>/<KB>".

**Queued:** job 22866653 (sph3d_spanlog10, estimate 05:51), chain `n2_spanlog10.sh` -> `logs/spanlog10_chain.out`
(waits for `logs/spanlog10.ready`): timed (1,4,4) / control / (2,2,4), eight async rounds, sync twice, alternating.
Judge: `startreg` lines present and `lazyreg window25-50` ~0; `metis-window` near the control's 0.6 s; pass1 flat across
events; log sizes bounded; (1,4,4) async against the control and against its own sync.

**Still open:** the launch dimension's per-device publish (outlined 2026-10-12 18:56); the (1,4,4) sync spread
(45.8 / 43.1 here, 41.25 on the tsc pair).

**04:21 addendum — the launch dimension's per-device publish is IN this build too** (hapiNormalizeCuptiLoads, the
`onlyDevice >= 0` branch, which the control also takes with its one device): a device's close publishes its own API
intervals (union -> cupti_driver_busy_dev_[d]), its total, and the API time of the objects the last sweep placed on it
(plus any object no device has claimed), leaving the other devices' in the accumulators; the process-wide figures are
the devices' sums. Judge in spanlog10: the `[MetisLB] measured loads explain ... launch N%` line of the (1,4,4) arms
(was 0% against the control's 4%). The prune fix also keeps the closed ancestor chain of the boundary span (the host
test caught the gap-after-a-nested-span case at the cutoff). Binary 04:21:28; sph2d and leanmd relinked.

## 2026-10-13 07:45 — job 22866653 read (gpub[089,091]; 091 on HPET): the 04:21 fixes hold, a fifth root cause, job 22868252 running

**Results (sections 2 and 4, async, whole-run ms/step / second half / lag window):** control n=24
39.58 / 38.72 / 0.386 s; (1,4,4) n=23 40.67 / 39.62 /
0.405 s; (2,2,4) n=8 39.90 / 39.07 / 0.390 s. The (1,4,4) gap is
+1.08 ms/step (was +1.67 in job 22856533). Sync: control 40.98, (1,4,4) 41.38, (2,2,4)
41.54 (the (1,4,4) sync arms were 43.1-45.8 before): async beats sync in every layout. Both 04:21 fixes
did what they were built for: the Metis window of the (1,4,4) arms is 0.58-0.75 s against the control's 0.54-0.77
(was 1.1-1.4 vs 0.6), `lazyreg` is 0 everywhere (the chunk runs register at the mapping: 128 x 256 MB per device,
0.20 s per device in the control, 0.97-1.85 s per device in the one-process layout -- see below), the drain's first
pass is flat over the run (pass1 max per event (1,4,4): 8 10 12 6 ... 6 6 7 11 ms; control 8 6 7 4 ... 4 4 5 5) and
the span logs are bounded (pe0 36-43k spans with its one open span at the top, pe5 32k spans compacting at every
arrival, gen 7/17/27/37). The launch dimension now reads for the (1,4,4) arms as well.

**What remains is the close (timed arms):** (1,4,4) 27.9 ms per close x 320 against the control's 9.3 ms; per close
flush=12-17 ms (control 1-6), and the four closers of an event serialise behind the first one's parse (wait=11-22 ms,
their own pass1=0: the first closer parsed everything). The lag window (resume) is +20 ms per event
(0.405 vs 0.386), which at 40 events is 0.8 of the remaining 1.1 ms/step.

**ROOT CAUSE 5 — there is no per-context CUPTI flush.** cupti_activity.h (CUDA 13.2): `cuptiActivityFlush(context,
streamId, flag)` is "DEPRECATED ... CONTEXT and STREAMID will be ignored. Use cuptiActivityFlushAll". So the per-device
close built on 2026-10-12 did four process-wide forced flushes per LB event in a four-device process, each 12-17 ms,
under a gate that only stopped that device's PEs (the other twelve kept launching into the flush -- the hazard the
gate exists for; no fault seen). Why a process-wide forced flush costs 12-17 ms in a 16-PE process and 1-6 ms in a
4-PE one: CUPTI keeps one activity buffer per thread (CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER defaults to 1),
the forced flush hands back every thread's buffer full or not, and each replacement is zeroed by CUPTI
(CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER default 0) -- at the 5 MB buffer size that is 80 MB of buffers per
flush against 20; and the kernel timestamps sit in CUPTI's device-side buffers (device memory since CUDA 12.0u1,
3.2 MB per buffer per context) that the flush reads back, one context after another. The two are hypotheses with a
measurement each in the running chain.

**Built 07:39 (uncommitted), hapi_impl.cpp / hapi.h / DistBaseLB.C / CentralLB.C:** (1) one window close per process
per LB event again: `hapiDeviceGroupSize()` is the process's PE count, `hapiDeviceGroupFirstPe()` its first PE, one
arrival counter, `hapiPrepareCuptiLoadsDevice` builds the process (`dev = -1`), `hapiClearCuptiDataDevice` clears the
process; `hapiCuptiFlushDeviceGated` and the per-device context record are gone (the per-device parse/normalise path
of `hapiPrepareCuptiLoadsImpl`/`hapiNormalizeCuptiLoads` stays, unused). (2) The host activity buffer default is 1 MB
(was 5; `CHARM_LB_CUPTI_BUFKB` still overrides): a PE writes ~0.6 MB per 25-step window, a full buffer is delivered
early and parsed at the next close. (3) CUPTI's device-side profiling buffers in pinned host memory
(`CUPTI_ACTIVITY_ATTR_MEM_ALLOCATION_TYPE_HOST_PINNED = 1`, set at the attach before the kinds are enabled;
`HAPI_CUPTI_HOST_PINNED_BUFFERS` is a build-time macro, default 1). (4) The close's timing line adds
`bufs=<n>/<KB>` (buffers taken from the queue and their record bytes). Binaries: `sph3d` = `sph3d.pin` (default),
`sph3d.nopin` (macro 0) for the A/B.

**Also found (open): LCI registers every MR on every LCI device of the process** (reconverse
comm_backend_lci2.cpp `registerMemory`: one `register_memory_x` per device, remote keys for all of them). With
`+lci_ndevices 4` the one-process layout registers each 256 MB chunk run four times, so its startup registration
is 0.97-1.85 s per device (max 584 ms for one chunk: the four lanes contend) against 0.20 in the control, and its
first Diffusion step starts ~1 s later (7.34 vs 6.32 s). Not in ms/step; in time to solution. Registering a device
buffer only on its own group's LCI device needs the protocol to pick the LCI device by the TARGET buffer's device.

**Job 22868252 (gpub[085,089], started 07:30, the chain joined at 07:40), `n2_spanlog11.sh` ->
`logs/spanlog11_chain.out`:** timed async (1,4,4) pinned, control, (1,4,4) nopin, (1,4,4) pinned at BUFKB=5120,
(2,2,4); the rounds use nopin only if its (1,4,4) flush mean is 20% below pinned's; then 8 rounds async
ctl/s14/s22, sync x2, alternating ctl/s14. Judge: one close line per process per event (80 for (1,4,4), 320 for the
control); (1,4,4) flush and total close near the control's; `bufs=` ~16-20 per close holding a few MB; lag window
0.385 like the control; (1,4,4) whole-run at or below the control; the (1,4,4) BUFKB=5120 arm tells the buffer
size's share of the flush, the nopin arm the pinned buffers' share.

## 2026-10-13 07:55 — job 22868252 timed arms (gpub[085,089], both tsc): the flush is the GATE, not CUPTI's work

**Timed async arms, 07:39 binary (one close per process, 1 MB buffers, pinned device buffers):** (1,4,4) close 27.6 ms
x 80 (one per process per event now): flush=14.5 (max 31) process=8.0 (pass1 6.3, snap 1.8 / 15 MB) normalize=5.1;
control 9.9 x 320: flush=5.0 (max 20) process=3.7 normalize=1.4; (2,2,4) 15.5 x 160: flush=7.0. Neither hypothesis of
07:45 moved the flush: (1,4,4) nopin 13.5 ms, (1,4,4) at CHARM_LB_CUPTI_BUFKB=5120 14.9 ms. The flush scales with the
number of GATES the flush takes (1 / 2 / 4 devices: 5 / 7 / 14.5 ms), not with buffer size or buffer location, and
the PE-side tallies show `waited=0` at every close in every layout: no PE ever waited for the gate. glibc's default
rwlock PREFERS READERS -- new readers keep acquiring while a writer waits -- so the flush's write lock waited for a
moment with no launch in flight on that device, ~3.5 ms per gate with four PEs launching every ~60 us, and the
four-device process takes four gates in turn. (The 2026-10-12 "the forced flush waits for the device's queued
kernels" reading was this wait.) Whole-run: the (1,4,4) timed arms scatter 37.1-43.7 ms/step (same binary, the nopin
arm the fastest at 37.1) against the control's 37.5: single arms do not resolve 1 ms on this pair; the rounds decide.
Rounds so far (3/3/2): control 40.49, (1,4,4) 41.09, (2,2,4) 39.11; lag window 0.394 / 0.409 / 0.380.

**Built (lib 08:00, relink after the chain; uncommitted):** the gates are `PTHREAD_RWLOCK_WRITER_NONRECURSIVE_INITIALIZER_NP`
(a waiting flush stops new launches at once and gets the gate when the in-flight ones finish; our readers never
recurse at the lock level, one read lock per thread). The close line splits the flush into `gate=` (the wait for the
gates), `call=` (cuptiActivityFlushAll) and `cbs=<n>/<KB>` (buffers it delivered). Expected: gate ~0, and the PEs'
`waited` count rises to one wait per PE per flush of the call's length. Chain `n2_spanlog12.sh` (job sph3d_spanlog12,
submitted by `scratchpad/submit12.sh` when 22868252 leaves the queue -- the interactive QOS refuses a second
submitted job): timed (1,4,4) pinned / control / (1,4,4) nopin / (2,2,4), the rounds with nopin only if its flush CALL
is 20% below pinned's, then 8 rounds, sync x2, alternating. Judge: (1,4,4) flush ~ control's; lag window equal;
(1,4,4) whole-run at or below the control over the rounds; sync (1,4,4) vs control (the close is on sync's critical
path: process 8 + normalize 5 ms per event against 3.7 + 1.4 -- the per-device sweep of hapiNormalizeCuptiLoads runs
serially over the devices, parallelisable on the parse workers if the sync gap remains).

**Also open:** the (1,4,4) startup registration (+1 s, [LCI registers on all devices]); the pinned-buffer attribute
stays only if `call=` is lower with it.

## 2026-10-13 08:30 — job 22868252 rounds read (gpub[085,089], tsc pair), job 22868681 queued with the writer gates

**Rounds (07:39 binary: one close per process, 1 MB buffers, pinned device buffers; all 53 arms clean):** async
control n=17 39.63 ms/step (sd 1.26) / (1,4,4) n=17 39.95 (sd 0.90) / (2,2,4) n=8 38.83; second halves 38.85 / 39.14 /
37.67; lag windows 0.385 / 0.397 / 0.378 s. Paired (1,4,4) minus the preceding control: +0.32 ms/step (sd 1.56, n=17:
within noise); (2,2,4) minus control -0.98. Sync x2: control 40.53, (1,4,4) 41.10, (2,2,4) 39.83. Async beats sync in
every layout. So after three fixes (registration at the mapping, bounded span logs, one close per process) the
(1,4,4) layout ties the control async and trails 0.6 sync, (2,2,4) beats it by 1.0 / 0.7, and the measured residual is
the (1,4,4) close (27.6 vs 9.9 ms per event: flush 14.5 = the reader-preferring gate, see 07:55).

**Validation of the writer-preferring gates:** job 22868681 (sph3d_spanlog12, queued 08:26 by the auto-submitter,
Priority), binaries relinked 08:24 (sph3d = sph3d.pin, sph3d.nopin; the `gate=`/`call=`/`cbs=` close line). Chain
n2_spanlog12.sh: timed (1,4,4) / control / (1,4,4) nopin / (2,2,4); LB-period-50 arms (ctl / s14 / s22) whose windows
without an event (ending at steps = 0 mod 50) measure each layout's pure step cost and whose event windows (= 25 mod
50) the event's; then 8 rounds, sync x2, alternating. Judge: `gate=` ~0 and `call=` a few ms in every layout, the
(1,4,4) close within ~5 ms of the control's; the (1,4,4) lag window = the control's; the period-50 clean windows
tell whether the one-process layout steps faster or slower than the control between events (the launch-path
tallies show 2x the gated driver calls per PE in the multi-GPU processes: 5042 vs 2424 per 25 steps, also in
(2,2,4) which is nevertheless the fastest layout).

## 2026-10-17 10:30 — job 22868681 read (chain 12, gpub[001,002] tsc, 08:24 binary = writer-preferring gates): the gate was not it

**Timed closes (CHARM_LB_CUPTI_TIME=1):** (1,4,4) total 28.0 ms per close = flush 14.4 (gate 0.0, call 14.4, max 25) +
process 8.1 (pass1 6.5) + normalize 5.5 + snap 1.7 (14.7 MB); 16 buffers / 7.5 MB delivered per close. Control 10.3
(call 5.2, process 3.7, normalize 1.2; 4 buffers / 1.9 MB). (2,2,4) 20.1 (call 11.5; 8 buffers). (1,4,4) with the
device buffers in device memory (nopin): call 14.5 -- the pinned attribute changed nothing and is removed. PEs
`waited=0` everywhere. So the cuptiActivityFlushAll CALL itself is the cost and it scales with the contexts (buffers)
of the process: 5.2 / 11.5 / 14.4 ms for 1 / 2 / 4 devices. The reader-preferring-gate explanation (08:05) is
falsified. Gated driver calls per close: (1,4,4) 50-81k per 16 PEs, control 9.7k per 4 PEs (2x per PE, as before).

**Rounds (22 ctl / 22 s14 / 8 s22 async, 61 arms clean):** control 38.67 ms/step (sd 0.80), (1,4,4) 39.67 (sd 1.03),
(2,2,4) 39.12 (sd 1.05); paired against the preceding control: (1,4,4) +1.00 (sd 1.54, sem 0.33 -- three sigma),
(2,2,4) +0.68 (sem 0.37; chain 11 had it at -0.98: node-pair scatter). Second halves 37.70 / 38.71 / 38.30; lag windows
0.376 / 0.393 / 0.383 s. Sync x2: 39.61 / 40.69 / 39.50. Async beats sync in every layout. Accounting: the close is on
the step's critical path (the closer's chares stall their neighbours), so (28.0 - 10.3) ms x 40 events / 1000 steps
= 0.71 ms/step -- the (1,4,4) gap within its error.

**LB-period-50 single arms:** control clean windows 38.94 / event windows 38.82 ms/step; (1,4,4) 35.69 / 36.34;
(2,2,4) 36.46 / 36.44. The (1,4,4) event costs 0.65 ms/step x 50 = 32 ms (its close), the control's ~0. But the
clean-window ORDER ((1,4,4) 3 ms/step faster between events) is the reverse of the rounds at n=1 (whole-run 37.6 vs
40.4 in those two arms against 39.7 vs 38.7 over the rounds): chain 13 runs three rounds of them.

**Hypothesis 7 (built 10:26, lib + sph3d = sph3d.v13):** the forced flush's call time is the device-side profiling
buffer readback and record completion that CUPTI's worker thread had not done yet -- cupti_activity.h: the worker
"offloads ... synchronization of profiling resources between host and device, delivery of the activity buffers",
woken "based on certain heuristics", and `cuptiActivityFlushPeriod(ms)` overrides them; "it's allowed to use
cuptiActivityFlushAll to flush the data on-demand, even when client sets the periodic flush". So: a fixed worker
period of 10 ms by default (CHARM_LB_CUPTI_PERIODIC=<ms>; 0 restores the heuristics), the gated FORCED flush at the
close unchanged. Removed: CHARM_LB_CUPTI_SOFT_FLUSH (dead end, documented), HAPI_CUPTI_HOST_PINNED_BUFFERS (no effect),
the "periodic = no forced flush" mode (blind by a buffer per thread). New instrumentation (CHARM_LB_CUPTI_TIME): the
arrival time of every buffer within the flush call, its context and KB, printed for the first three closes per process
-- the call's structure (per context? per buffer? all at the end?). Also first linked here: the parallel lane snapshot,
the parallel per-device sweep in hapiNormalizeCuptiLoads, the per-kind gated-call tallies (closure / batch / event /
memcpy). Chain `n2_spanlog13.sh` (job 22953167, sph3d_spanlog13, queued 10:25): timed (1,4,4) default / control /
(1,4,4) PERIODIC=0 / control PERIODIC=0 / (2,2,4); the rounds use the default only if the (1,4,4) call mean with the
period is 20% below the one without; LB-period-50 x3 rounds; 8 async rounds; sync x2; alternating. Judge: (1,4,4)
`call=` a few ms with the period and ~14 without (else hypothesis 7 is dead and the arrivals line says where the time
goes); process + normalize + snap down from 15.3 toward the control's 5; (1,4,4) whole-run <= control over the rounds.

## 2026-10-17 11:20 — job 22953167 (gpub[060,083] tsc): hypothesis 7 dead; the flush call dissected; zeroing fix built

**Chain 13 (sph3d.v13, periodic worker flush 10 ms vs CUPTI heuristics):** (1,4,4) flush call 15.9 ms with the period,
14.4 without; control 6.3 / --. Hypothesis 7 falsified; the default goes back to CUPTI's heuristics (keep
CHARM_LB_CUPTI_PERIODIC as the one knob, 0 = heuristics). The arrivals line gave the call's structure: NOTHING is
delivered for 2-42 ms, then the 16-17 buffers arrive one every ~0.3 ms (one every 0.04 ms at 64 KB). The parallel lane
snapshot and per-device sweep changed little (process 8.6, normalize 3.8, snap 3.2 vs 8.1 / 5.5 / 1.7). Gated-call
tallies per PE per 25 steps: (1,4,4) 4628 closures + 387 events, control 4655 + 380 -- EQUAL per PE; the "2x driver
calls per PE" of 08:30 compared closes of unequal windows and is closed. Period-50 arms, 3 rounds: control clean
36.48 / 36.54 / 35.71, event 36.61 / 36.44 / 35.57; (1,4,4) clean 35.99 / 36.75 / 36.71, event 36.76 / 37.23 / 36.86;
(2,2,4) clean 35.35 / 36.08?, event 35.23 / 36.08?. The (1,4,4) event costs +0.15..+0.77 ms/step over its 50-step
window = 8-38 ms per event (mean ~23); the control's ~0; the clean windows do not separate the layouts (+-1 ms/step
scatter at n=3). Chain 13 was stopped past the period-50 arms (the rounds would have re-measured a known +1.0).

**Chain 14 (diagnostic binaries, same job):** sph3d.sync14 synchronizes every local device inside the gate before the
forced flush: (1,4,4) sync=9.6 ms (max 20) + call=6.4 (max 27) = the same 16; control sync=5.3 (max 18) + call=1.1
(max 2). So the flush call is (a) the GPU QUEUE DRAIN -- CUPTI's readback of the device-side records is stream-ordered
behind the queued kernels, the max over the process's four GPUs in (1,4,4) -- and (b) a handover of ~0.3 ms per 1 MB
buffer, which the arrivals show evenly spaced and the CUPTI header explains: CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER
defaults to 0 and "CUPTI will zero out the buffer in the main thread ... a few stalls may appear in critical path".
sph3d.soft14 (the non-forced flush, 64 KB buffers): call 13.7 ms with the same pre-delivery wait (the drain is not a
forced-flush artefact) and 101 buffers delivered per close; the non-forced flush buys nothing. The close already runs on
the LB helper thread (CHARM_LB_HELPER=near is in every arm): what the PEs feel is the GATE held for the whole call
(drain + zeroing), during which no PE of the process launches, so the four GPUs run dry and restart together. Slurm's
controller timed out for ~4 minutes mid-chain: left_s came back empty and four arms were skipped (ctl soft, s22 sync,
one period-50 round); chain 15 computes the time left from the job's end time read once.

**Built 11:10 (lib v15, sph3d.v15 linked 11:15):** CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER = 1 at the attach;
fresh buffers were memset already, recycled ones are zeroed when they leave quarantine (hapiCuptiFreeRetired, on the
helper, outside g_cupti_free_lock). Expected: (1,4,4) call ~10 (the drain alone), control ~5.5. The drain is the part
no flush avoids: it is the queue depth at the close (5-10 ms, max 18-20), and under the gate it is also the GPUs'
idle time until the launches resume. Chain 15 (job 22953167 from 11:15): kernel-only +gputrace timelines of the
control and (1,4,4) (gap_report.py: kernel gaps > 3 ms per rank/GPU around the LB events, and what the PEs did in the
largest), then timed (1,4,4) / control with v15, then alternating v15 arms.

**Open after this:** if the drain is what remains, the options are (i) flush without the gate on the helper so launches
continue while CUPTI drains -- needs the forced-flush race with the launch intercept settled another way (the quarantine
alone was not enough, job 22809309), or (ii) close the window at a shallow-queue moment, or (iii) accept it: ~5 ms
(control) vs ~10 (max of four queues) per event, 0.2 ms/step.

## 2026-10-17 11:40 — chain 15 (job 22953167): the timelines, and v15 does not shorten the call

**Kernel-only GPU timelines (sph3d.v14, last 2.5 s of a 1000-step async run = 2-3 LB events; gap_report.py, gaps
> 3 ms):** control: the largest gap per rank is 4.5-18.8 ms (median 11.4); the biggest (18.8 ms, rank 7) is a
machine-wide pause (every PE of every rank idle 17-18 ms), the other ranks' large gaps are NOT simultaneous (rank 5's
GPU has eleven gaps > 3 ms in 2.5 s, rank 7's dozens). (1,4,4): the largest gap per rank is 14.6-19.5 ms (median
16.1), and the gaps at 40936-40940 (8.5-14.7 ms) and 41887-41898 (15.4-17.7 ms) are on ALL EIGHT GPUs of both nodes at
once, 953 ms apart = one LB period: each LB event is a ~15 ms all-GPU stop. Every PE is idle through it (16-19 ms,
entries receiveHalo/leaversPacked/haloPacked at fractions of a ms): nobody is computing, everybody waits. That is the
gated flush: no launch in the 16-PE process for the call's 16 ms, the four GPUs run dry together, the other node's
process starves on the halos. x40 events = 0.6 ms/step, the gap. The control's per-process flush (5 ms, one GPU, the
eight processes at slightly different moments) is absorbed.

**v15 (CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER = 1):** (1,4,4) call 16.5 ms (max 34), control 4.7: no change.
The per-buffer spacing stays ~0.3 ms at 1 MB (0.1-0.15 in the control's process): it is CUPTI completing/copying the
records, proportional to the records, not the zeroing. The attribute stays (one memset fewer per buffer, off the
critical path) but is not a cure. The arrivals also show the drain is SERIAL per context: in one (1,4,4) close eight
buffers arrived at 21.1-22.6 ms, then nothing until 33.2, then the other eight -- CUPTI drains the contexts one after
another, so the one-process call is nearer the SUM of four queue drains than their max.

**Settled mechanism:** flush call = sum over the process's contexts of (queue drain, stream-ordered device-record
readback, 2-20 ms each) + record completion (~0.3 ms per MB of records, 4x the control's volume), all under the gate.
Nothing tried shrinks it: writer gates, pinned device buffers, buffer size, worker flush period, zeroed buffers (five
hypotheses, all falsified by measurement). Per PE the driver-call counts are equal across layouts; the close already
runs on the helper thread; the parse (12 ms) is off the critical path. The ONLY lever left is the gate itself.

**Design E2 (not built): a close that needs no gate.** The gate exists because the FORCED flush races with CUPTI's
launch intercept (job 22809309). The NON-forced flush is what CUPTI's own worker thread does concurrently with launches
by design, so it needs no gate; soft14 measured it at 13.7 ms per call delivering 81% of the window's bytes with 64 KB
buffers (the open buffer per thread is withheld) and the SAME kernel counts as the forced arms within noise. On the
helper, without the gate, its 14 ms stall nothing. What it costs: (a) each thread's window ends at its last delivered
buffer, so a thread's objects are under-counted by the withheld fraction (0-100% of a 64 KB buffer = 0-3.4 steps of a
10-step window) -- correct by scaling each thread's objects' device time by (wall window) / (that thread's covered
span: first..last record of its records in the drain); the withheld records arrive at the next drain and belong to
the previous window: the parse already carries unresolved kernels one drain, the sweep must clip or drop records that
end before this window's start (tracing is never switched off, the window is "since the last drain", so a per-drain
timestamp must be recorded). (b) 16-64 KB buffers: more callbacks, same record volume (the handover cost is per
record, not per buffer, once zeroing is off). (c) the forced flush stays for hapiTraceStop / the end of the run. Which
buffer holds the kernel records (the launching thread's, or a context's) decides whether the coverage is per thread
or per device: print per-thread first/last record times in the soft arm to see. Expected: the (1,4,4) LB event
stops no GPU; the layout ties the control (the period-50 clean windows show no between-event advantage on sph3d:
3 rounds, +-1 ms/step). Cost: ~half a day plus one 2-node validation. Parity is the ceiling on sph3d; a win needs a
workload with many small same-node messages or many same-node migrations (leanmd).

**Chain 15 alternating v15 arms (n=10 control / 9 (1,4,4), all 23 arms of the chain clean):** control 38.92 (sd 0.64),
(1,4,4) 39.90 (sd 0.57); paired +0.90 (sd 0.99, sem 0.33). Same gap as chains 12-13: the zeroed-buffer build
changes nothing, as its close line said. Job 22953167 ran out at 11:38 (node kept to its end, never cancelled).

**State of the question (2026-10-17 11:45):** the (1,4,4) async layout trails the control by 0.9-1.0 ms/step (2.5%)
and sync by ~1; (2,2,4) by 0.7 async on this pair (-1.0 on the chain-11 pair). The whole residual is one mechanism,
the gated CUPTI flush at each LB window close, understood to its parts and confirmed by the all-GPU timeline. The one
lever left is E2 above. Between events the layouts step alike on sph3d (period-50 clean windows, 3 rounds), so E2
buys parity, not a win. A win needs an application whose same-node traffic or migration volume the one-process layout
actually saves: leanmd (tens of thousands of small same-node Cell/Compute device messages per step through the IPC
slot protocol, ~750 Compute migrations per event). Next: a leanmd runner with the split knobs (leanmd/run_lmd.sh,
numa_wrap_lmd.sh: 8 PEs per GPU on the first 8 cores of the GPU's NUMA domain, GPU order reversed as in sph3d's
wrapper), then no-LB (4,1,4)/(1,4,4)/(2,2,4) first -- a per-step advantage shows without any balancer -- then sync
and async at 16x8x8, 100 steps, period 20, lag 16 (scale.sh's strong configuration, 128 cells/GPU on two nodes).

## 2026-10-17 12:40 — leanmd under the split (job 22953717, gpub[006,013] tsc)

**Runner:** leanmd/run_lmd.sh + numa_wrap_lmd.sh (PPN/GPP/PES knobs as run3d.sh; 8 PEs per GPU on the GPU's own NUMA
domain). The first chain's arms all died at startup for want of PMI_MAX_KVS_ENTRIES=8192, FI_MR_CACHE_MONITOR=disabled
and the LCI_LIB preload (the stock LCI pushes a NULL owning context for vmm buffers: cuCtxPushCurrent fails in
accelerator_cuda.cpp:154 in every multi-GPU process). Fixed; chain n2_lmd2.sh.

**No LB, 16x8x8 (128 cells/GPU), 100 steps, mean of steps 21-100:** control (4,1,8) 1496.2 ms, (1,4,8) 1491.5, (2,2,8)
1490.1; wall 144.8 / 144.6 / 144.6 s. A TIE (0.3-0.4%): at 1.5 s per step the Cell/Compute message protocol is nothing
against the compute, same-node or not. Sync and async Metis+Diffusion arms follow (the one-process LB window close on
leanmd has far more kernel records per process than sph3d's, so the gated flush should be worse there, not better).
Chain n2_lmd3.sh is written for the job's last quarter hour: 8x4x4 (16 cells/GPU, tiny kernels, the step IS the
device-message traffic) no-LB x3 rounds then async -- the one regime left where in-process delivery could show.

**leanmd 16x8x8 with LB (one arm each; leanmd's LB run-to-run spread is 5-10 s of a ~115 s run, so n=1 decides
nothing):** mean of steps 21-100 / 42-100 (the second excludes the LB steps):
sync   control 1117.7 / 994.7   (1,4,8) 1135.6 / 990.2   (2,2,8) 1123.1 / 989.7   wall 113.9 / 115.7 / 114.4 s
async  control 1124.4 / 998.6   (1,4,8) 1110.0 / 974.3   (2,2,8) 1093.7 / 962.3   wall 114.5 / 113.7 / 112.2 s
No-LB round 2: 1496.0 / 1487.0 / 1489.6 (tie again). LB wins 33% on every layout. The async post-LB window hints at
the multi-GPU layouts ahead (2.4% and 3.6%) with the control's sync and async equal; the Metis step moved 8.6-9.0k
objects in every layout. Needs n >= 6 pairs on one allocation before it means anything. The (1,4,8) "launch 18%" is
the known process-wide launch-dimension artefact. Chain 2 stopped after two more async pairs (section 4); chain 3
(8x4x4, 16 cells/GPU) takes the job's last ~15 minutes.

**leanmd 8x4x4 (16 cells/GPU, the communication-bound regime), no LB, three rounds, steps 21-100:** control 202.0 /
202.2 / 202.3 ms, (1,4,8) 201.1 / 201.1 / 200.9, (2,2,8) 201.4 / 201.4 / 201.6. The one-process layout IS faster when the
step is message traffic -- by 0.5%, consistently (the scatter is 0.1). That is the size of the same-node protocol saving.
Second async pair at 16x8x8: control 998.6 / (1,4,8) 1014.3 (post-LB window): the first pair's 2.4% reversed; n=2 is a tie.

**Found and fixed (LBLoadDim.h, libck 13:05, unvalidated):** the central criticality form (MetisLB) spread the driver
term over PROCESSES (CmiNodeOf), so a process driving four devices got a quarter of the contexts and a 4x launch share
("launch 18%" vs 5% leanmd, 15% vs 4% sph3d). Now one context per (process, device) pair, as DiffusionLB's distributed
form already counted. No decision flipped in these runs (device 51-65% stayed the larger share), but a launch-heavy
application would have balanced the wrong dimension in a multi-GPU process. Goes into the next leanmd/sph3d links.

**leanmd 8x4x4 async Metis+Diffusion (lag 16), 18 pairs alternating control / (1,4,8), post-LB steps 42-100:** control
114.92 ms/step (sd 1.25), (1,4,8) 114.08 (sd 1.74); paired -0.83 (sd 2.32, sem 0.55) = -0.7%, 1.5 sigma; wall equal
(14.18 s both). LB wins 43% on both layouts (202 -> 115). The same ~0.5% lean as without LB: the one-process layout's
same-node saving is real and small. Job 22953717 ran out at 13:19 (node kept).

**Next (chain n2_lmd4.sh, queued 13:20):** 16x8x8 async, seven rounds of control / (1,4,8) / (2,2,8) with the
relinked binaries (libck criticality-contexts fix + libhybridapi v16) -- the statistics the n=2 pairs lacked.

## 2026-10-17 14:05 — leanmd chain 4 (job 22955173, gpub[006,013]): seven async rounds, the lean collapses to a tie

16x8x8, 100 steps, async Metis+Diffusion lag 16, binaries with the criticality-contexts fix (libck 13:05) and
libhybridapi v16; 21 arms clean. Post-LB steps 42-100: control 1009.7 ms/step (sd 27.1), (1,4,8) 999.9 (sd 13.3),
(2,2,8) 1006.6 (sd 13.1); paired (1,4,8) - control -9.7 (sem 10.8, -1.0%), (2,2,8) - control -3.1 (sem 12.8); whole run
(21-100) -4.5 / -1.7; wall -0.2 s of 115 for both. After four rounds both layouts had read -2.2% at 1.7 sigma; rounds
five to seven took it back. What stays: the multi-GPU layouts run STEADIER (sd 13 against the control's 27: the
control's slow arms coincide with its Metis steps sending 29-115 objects across nodes). Verdict for leanmd: the
one-process layout ties the control under LB, sync and async, and is 0.5-0.7% faster only where the step is message
traffic (8x4x4). Sync pairs run in the job's last ten minutes.

**Chain 4 sync pairs (16x8x8, post-LB 42-100 / whole 21-100 / wall):** control 1015.7 / 1133.4 / 115.3 s, (1,4,8) 1000.4 /
1143.4 / 116.3; control 995.4 / 1116.7 / 113.9, (1,4,8) 992.7 / 1137.9 / 115.9; control 998.2 / 1120.2 / 114.1. Paired
post-LB -9 +-6 (n=2) but whole-run +16 and wall +1.5 s: the one-process layout's first twenty steps are slower (first
step 1420-1582 ms against 1257-1305: the startup registration of every buffer on every LCI device, [LCI registers on
all devices], and a bigger first step) and the sync run never recovers that second. Job 22955173 ran out at 14:14,
26 arms clean; node kept to its end. leanmd verdict: TIE under LB, sync and async; steadier under async; 0.5-0.7%
faster only in the message-bound regime; ~1 s slower to start.

## 2026-10-17 15:05 — both fixes in flight: the startup registration probe and the gate-free close (E2), one arm each

**User direction:** do both, the registration first; and from now on ONE arm per configuration ("don't run the same
thing a hundred times. one is fine").

**Startup registration.** The cost is inherent to one libfabric endpoint per LCI device: the backend's mr_mode carries
FI_MR_ENDPOINT (backend_ofi.cpp), so a memory region is bound to one endpoint and a process with four LCI devices must
register every buffer four times; one LCI device per process was measured at -18..25% under LB at two nodes (10-12),
so fewer devices is not the answer. What remains is fewer registrations: the vmm pool registers one region per
cuMemCreate chunk (+gpupoolchunk, 256 MB default: 128 regions per 32 GB device, x4 devices = 512 per process-device).
If the cost is per region rather than per byte, 1 GB / 2 GB chunks cut it 4-8x. Probe: (1,4,4) arms with
+gpupoolchunk 1024 and 2048 against the 256 MB arms of the same chain ('startreg' sum/max per device, DiffLB1 time).

**E2 built (lib e2, sph3d.e2 / leanmd.e2 linked 15:04; the forced reference sph3d.v17 / leanmd.v17 from the same
source with HAPI_CUPTI_FLUSH_FORCED 1):** the non-forced flush takes no gate (hapiCuptiFlushGated gates only a FORCED
flush); the LB round stamps its window in the CUPTI clock before the flush (g_cupti_window_prev/close_ns); each thread's
coverage = its last runtime-API record in the drain (per shard cover_end, merged); the sweep drops records that ended
before the window and clips those straddling its start; each object's device load and API time are multiplied by
window / its thread's covered span (the thread is known from the API record that shares the kernel's correlation id:
thread_corr -> obj_thread), capped at 2x below half coverage (capped=); a device's driver-busy union by its threads'
mean scale; 64 KB activity buffers in the non-forced build. The timed close line prints cover= (worst thread's covered
fraction) scale= threads= capped=. Chain n2_spanlog16.sh (job 22956399, 25 min): timed e2 (1,4,4) / control; e2 async
(1,4,4) / control / (2,2,4); e2 sync (1,4,4) / control; v17 async (1,4,4) / control; the two chunk-size arms; leanmd e2
async (1,4,8) / control. Judge: e2 call= and total on the helper are irrelevant to the step (no gate: gate waits 0 and
the (1,4,4) async whole-run within the control's); cover= near 0.9 and capped= 0; device-share and Metis moves equal to
v17's (the balancer sees the same loads); checks all ok.

**15:49 — job 22956399 lost:** granted while the session was down; the chain runner and its salloc had died with the
session, so the allocation ended after 10 s with nothing run. Relaunched 17:02 as job 22958581 (setsid -f, own session).

## 2026-10-17 18:45 — chain 16 (job 22958581, gpub[041,097] tsc): gate-free close validated as a mechanism, made the default; registration cost is per byte x LCI devices

One arm per configuration, 12 arms + one traced arm, every arm clean except the 2 GB-chunk probe (below). Binaries:
sph3d.e2 / leanmd.e2 (non-forced flush, no gate, 64 KB buffers, coverage scaling) against sph3d.v17 / leanmd.v17
(forced, gated) from the same source; CUPTI buffers are never free()d in either (hapiCuptiFreeRetired recycles them).

**Timed e2 closes.** (1,4,4): 80 closes, total 33.9 ms on the helper = flush 20.8 (gate 0.0, sync 0.0, call 20.8, max 53)
+ process 10.3 (pass1 6.4) + normalize 2.7 + snap 3.2 (15 MB/close); 94.8 buffers / 5.6 MB delivered per close = 0.27
ms/MB, the record completion, now entirely on the helper. Launch path: gated=80906 waited=0 (0.0000 s) -- the PEs never
stop for the close. Coverage: min-cover mean 0.884 (worst 0.50), max-scale mean 1.144 (worst 2.00), 15.6 threads per
close, 2 of 80 closes capped. Control: close 11.5 (call 5.3), cover 0.917, 8 of 320 capped. Device share 62-70% and
Metis 226-236 moves in every arm of the chain: the balancer sees the same loads under e2 as under v17.

**Step times (ms/step, whole run / second half).** e2 async: (1,4,4) 39.198 / 37.98, control 37.920 / 37.02, (2,2,4)
39.174 / 37.78. e2 sync: (1,4,4) 41.756 / 40.64, control 42.623 / 42.02. v17 async: (1,4,4) 39.567 / 38.78, control
39.355 / 38.68. (1,4,4) async with +gpupoolchunk 1024: 39.350 / 38.44. Reading at n=1: the Cf series has sd ~0.6 per arm
(control 38.92 n=10, (1,4,4) 39.90 n=9 in chain 15), so one pair resolves +-0.85. The e2 (1,4,4) arm equals the best
(1,4,4) async ever recorded (39.209) and the e2 control sits at the control's low end; their gap (1.28 / 0.96) is
indistinguishable from the forced build's +0.98 (sem 0.33), and the same chain's v17 pair read +0.21. The sync pair
leans the other way (-0.87). No step-time verdict from one pair; the mechanism verdict below stands on the trace.

**Trace (e2 (1,4,4), +gputracenocopy, last 2.5 s, gap_report.py):** largest gap per rank min 8.8 / median 10.8 / max
14.3 ms, once per LB period (60343 and 61315 ms: 972 ms apart = 25 steps), every PE of both ranks idle through it. The
gated (1,4,4) trace of chain 15 read 14.6 / 16.1 / 19.5 and the gated control 4.5 / 11.4 / 18.8: the all-GPU stop at
the event shrank by ~5 ms, to the control's own size. 5 ms x 40 events / 1000 steps = 0.2 ms/step is all the gate was
worth per step; the ~11 ms stop that remains is in the control too (an LB-event cost of both layouts, next target).
The per-step launch cost is not the residual either: in the Diffusion rounds the measured launch share is equal (6-8%
(1,4,4) vs 7-10% control; the chain's "launch 16% vs 4%" is the Metis window's figure), and no-LB runs tie (56.5 vs
56.8). If a (1,4,4) residual of ~0.8 ms/step is real, its mechanism is not identified; the one per-event difference
left is the helper's 34 vs 11.5 ms of CPU, now off the PEs' path on the domain's idle cores 8-15.

**Decision 18:41:** HAPI_CUPTI_FLUSH_FORCED 0 is the default (hapi_impl.cpp:524); libhybridapi rebuilt, sph3d, sph3d.e2,
leanmd, leanmd.e2 relinked; sph3d.v17 / leanmd.v17 and libhybridapi.v17.a stay as the forced reference. Reasons: no
launch ever waits on a close; the LB-event stop shrank to the control's; the balancer's inputs are unchanged; 8 arms
and the traced arm clean (sph3d async/sync, leanmd async); no incomplete records are ever returned, so the quarantine
and the gate are no longer load-bearing (both still compiled, [async-segfault-open]).

**Registration probe (per device, 32 GB mapped at the start, 'registered for RDMA at the mapping'):** control (1 LCI
device) 0.19-0.21 s; (2,2,4) (2 devices) 0.44-0.46; (1,4,4) (4 devices) 0.93-1.90 with single chunks of 160-813 ms when
the four lanes contend. 1024 MB chunks: 0.89-1.82 s -- unchanged, so the cost is per byte x LCI devices, not per
region. 2048 MB chunks: 0.33-0.67 s, but the run died in LCI poll_comp (ADDR_OUT_OF_RANGE, prov_errno 52, on a remote
READ): 2 GB regions are beyond what the provider's MR addresses -- not usable. First Diffusion step: (1,4,4) 7.2-7.5 s,
control 6.4, (2,2,4) 6.0. Verdict: +1 s of startup stays for (1,4,4) until LCI registers a buffer only on the device
that serves it (or registers chunks lazily); (2,2,4) starts earliest.

**leanmd e2 async 16x8x8, one pair (lag 16, 100 steps):** (1,4,8) 1107.7 / 964.8 ms (steps 21-100 / 42-100), wall
113.4 s; control 1121.6 / 999.7, wall 114.3 s: -1.2% / -3.5%, same direction as the seven-round -1.0% +-1.1%. (1,4,8)
closes 410.7 ms (call 73.7) x 10 on the helper, control 98.7 (7.4) x 40; cover 0.934 / 0.970; both clean.

**Next:** one longer pair (4000 steps) per layout is the only way to resolve a 1 ms/step question at one arm per
configuration; then the ~11 ms LB-event stop common to both layouts (what every PE waits for at the close).

## 2026-10-17 19:25 — steady state only (user: "I don't care about startup costs. I only care about steady state")

**Per-window reading of the chain 16 pairs (steps >= 525, 20 windows of 25 steps, same step indices paired):** e2
(1,4,4) 37.98 vs e2 control 37.02 ms/step, paired +0.97 (sd 0.60), (1,4,4) slower in 18 of 20 windows -- a uniform
per-step cost, not a few bad windows; v17 pair +0.10 (9 of 20; that control arm was slow, 38.68); e2 (2,2,4) +0.76 (15
of 20). The control's own run-to-run spread (37.02 vs 38.68 second half, five minutes apart) exceeds the layout effect,
so pairs at n=1 cannot settle it; chain 17 (one 4000-step arm per layout, job 22961360) is queued for that.

**Trace steady-state picture (last 2.5 s; gated control, gated (1,4,4), e2 (1,4,4)):** GPU busy 80.9 / 82.0 / 82.5%;
idle gap time per second of trace 171 / 160 / 156 ms, of which gaps > 8 ms (the LB-event stop and the sort step) 14.8 /
25.4 / 12.2 ms. The e2 (1,4,4) GPUs are as busy as the control's; nothing in the GPU timeline is slower per step.

**Where the per-step cost could be: the cut.** The app's message-tier histograms (whole run, halo messages) read, for
the chain 16 arms: control same-GPU 1.14M / same-host 1.11M / other-host 1.07M; (1,4,4) 1.04M / 0.30M / 1.98M; (2,2,4)
1.15M / 0.38M / 1.78M; v17 pair 1.15/0.72/1.44M vs 1.15/0.29/1.87M; sync 0.75/0.96/1.61M vs 0.80/0.07/2.45M. Over all
logs: control async 1.11/0.86/1.34M (n=149), (1,4,4) 1.10/0.34/1.88M (n=157), (2,2,4) 1.15/0.57/1.60M (n=67). Same-GPU
traffic is equal (the within-group placement is equal); the split of the cross-GPU remainder is 61% cross-host in the
control and 85-97% in the one-process layout. Two readings: (a) real -- the balancer places the one-process layout's
regions so that almost every cross-GPU neighbour is on the other host (more network traffic per step = the +1 ms/step);
(b) an accounting artifact -- the app's peTier() went through CmiNodeOf/CmiPeOnSamePhysicalNode, and the 2026-10-12/13
no-LB (1,4,4) logs, whose block map is identical to the control's, read same-host 0 / other-host 2.97M against the
control's 1.83M / 1.14M: the classification was wrong for the one-process layout then, and the trace tool's tiers come
from the same function. MetisLB itself is host-blind: level one prices every cross-group edge at one tier (ipc_cross_gpu
= inter_node in lbcost.delta-a40.conf), so which cut edges cross the host boundary is left to the partition and the
relabel. For the control, 61% cross-host of the cross-GPU traffic is what a host-blind 8-way assignment gives (4 of 7
other groups are on the other host = 57%); a host-aware level (objects -> hosts -> GPUs -> PEs) would cut the control's
cross-host traffic too. That is a steady-state lever for both layouts, pending the measurement.

**Instrumentation built 19:20 (sph3d.diag = e2 + prints; libmoduleEveryLB/MetisLB rebuilt; sph3d.e2 and chain 17
untouched):** MetisLB prints its group table (PEs, process, physical node, device id) and, in the priced block, the
cross-group messages by host under the old and new mapping ("cross-group messages by host: same-host A -> B, cross-host
C -> D (share)"), exact from the balancer's own mapping; sph3d's peTier() now classifies by CmiPhysicalNodeID and the
host-local device index (hapiDeviceIndexOfPe vs hapiMyDeviceIndex). Chain 18 (n2_spanlog18.sh, 13 min, written, to
launch after chain 17): one instrumented async arm per layout plus one no-LB (1,4,4) arm, whose fixed block map must
reproduce the control's no-LB tiers (0.35M / 1.83M / 1.14M) if the accounting is now right. Judge: the Metis host-split
line decides (a) vs (b); if (a), the fix is a host level in MetisLB and host-aware pricing in DiffusionLB; if (b), the
one-process residual is still unexplained and the 4000-step pair is the measurement.

## 2026-10-17 19:45 — chains 17 and 18b (job 22961360, gpub[002,097] tsc): steady-state parity; the cross-host signal was an accounting artifact

**Chain 17, one 4000-step gate-free async arm per layout.** Whole run (1,4,4) 41.59 / control 41.62 / (2,2,4) 42.46
ms/step. Steady state, steps 1025-4000 paired by window (120 windows): (1,4,4) 42.02 vs control 42.27, paired -0.25
(sd 1.81), the (1,4,4) arm slower in 61 of 120; second half (steps 2025-4000) -0.80, slower in 28 of 80. (2,2,4) +1.22
(54 of 120), all of it in the last quarter. The dam break's late phase drifts hard (mean ms/step by 1000-step quarter:
(1,4,4) 39.9 / 40.7 / 41.7 / 43.7; control 39.2 / 39.9 / 41.2 / 45.8; (2,2,4) 38.9 / 39.3 / 40.7 / 50.5) and the
late-phase spread (per-window sd 3-6 ms) is a per-run LB draw, not a layout property. Verdict: the one-process layout
is at PARITY with the control at steady state; the +0.97 of the 1000-step pair earlier today was run-to-run spread
(the control's own second halves ranged 37.0-38.7 within an hour). No (1,4,4) per-step mechanism is left to find.

**Chain 18b, instrumented arms in the allocation's last six minutes.** MetisLB's group table is right in every layout
(groups 0-3 physical node 0, 4-7 node 1; processes 0/0-3 / 0-7; device ids unique). Its exact cross-group split by
host under the block map and under its own partition: same-host 111200 -> 39750 / cross-host 21000 -> 52600 messages
per interval in (1,4,4), 111200 -> 39050 / 21000 -> 47650 in the control, 111200 -> 38450 / 21000 -> 49850 in (2,2,4):
cross-host share 16% -> 57 / 55 / 56%. The balancer places the cut IDENTICALLY across layouts. The app's per-tier
histograms disagreed because they tier a message by ckLocMgr()->whichPe(partner), which returns -1 for an element
not in the PE's location cache, and peTier(-1) is "other node"; in a multi-device process the in-process partners on
other devices are never cached (their messages take the in-process path), so they were counted as cross-host -- the
no-LB (1,4,4) arm still read same-host 0 / other-host 2.97M with the physical-node form of peTier, which settles it as
a cache miss, not a predicate. The 2026-10-12..17 "other node" counts for (1,4,4) and (2,2,4), and the gputrace tiers
(same function), are therefore not cross-host traffic. Fix built 19:44 (sph3d, sph3d.e2, sph3d.diag; leanmd relinked
19:43 against the rebuilt EveryLB): every exchange entry carries the sender's PE (int src after ts in the .ci) and the
receiver tiers by it; peTier is by CmiPhysicalNodeID and the host-local device index. Validation: the next no-LB (1,4,4)
arm must read same GPU 347440 / same host ~1.83M / other host ~1.14M like the control's. Not yet run.

**What Metis does to the cut, in every layout:** from 16% cross-host under the block map to ~56% after its partition
(a host-blind 8-way assignment; 4 of the 7 other groups are on the other host). A host level would hold it near the
block map's 16%. Whether that is worth a step: in the control's trace the cross-host halo p50 was 1.84 ms against 2.81
for same-host IPC, so on this fabric the network tier is not the slow one; not pursued. The gate-free close stays the
default (0.2 ms/step per LB event in a multi-GPU process, and no launch ever waits).

**State of the binaries:** sph3d = sph3d.e2 = sph3d.diag (19:44: non-forced close, Metis host-split and group prints
under +LBDebug 1, sender-PE tiers); leanmd = leanmd.e2 (19:43). sph3d.v17 / leanmd.v17 remain the forced reference.

## 2026-10-17 19:55 — committed

No allocation left (squeue empty). Diagnostics removed before the commit (hapi_impl.cpp): the HAPI_CUPTI_FLUSH_SYNC
block, the per-buffer flush-arrival recording and its "closes 1-3/20-22" print, the launch-path tallies (gate waits,
push counts, call kinds), the device-pool traffic counters and print, the gate-wait timing in the launch path. Kept
under CHARM_LB_CUPTI_TIME: the close's phase line (now gate=/call=/cbs=/cover=/scale=), the record counts, the
per-device API totals, the log-arrival line, the attach time. Rebuilt and relinked clean (hybridapi, sph3d, leanmd).
Commits on rate-aware-gpu-lb: 9d315ab61 (runtime: the split, the entry-span log, the sharded parse, the per-process
and gate-free close, pool registration at the mapping, the peer tier, the contexts fix, the Metis prints) and
5d574c8b3 (sph3d / sph2d / leanmd: per-PE pinned pools, per-device event pools, sender-PE tiers, the split runners
and wrappers, the per-device trace tools, window_compare.py, trace_steady.py, chains 16-18 and n2_lmd4.sh, lbcalib
peer tier, launchcost multi-device). The older experiment chains (n1_*, n2_spanlog1-15, n2_lmd1-3, gdb wrappers) and
the planning notes stay untracked.
