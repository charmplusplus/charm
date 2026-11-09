# Starting on Frontier: the branch, the AMD build, and the first jobs

Written 2026-11-02, when development of `rate-aware-gpu-lb` moved from Delta's A40 nodes to Frontier's
MI250X. `HIP_PORT.md` is the port itself and the Frontier work of 2026-10-22 (its last section: the build
corrections, the driver findings, the profiler patch, the channel-transport fixes, the balancer figures).
This file is what has been added since, what it needs on Frontier, and what to run first. It replaces the
handover of 2026-10-21.

## What the branch holds now

Commits of 2026-11-02 on `rate-aware-gpu-lb`, made in the Delta clone `/u/bhosale/charm-reconverse`
(pushing them needs a credential from the user; none is stored anywhere):

| Commit | What |
|---|---|
| bfcdc743e | replication as a balancing decision: `setLBReplica`, the clone and retire decision kinds, `ckClone`/`ckRetire` and the notices to the other copies, ShedLB's group pass with the critical-path rule and the declared exchange cost, the memory contract's clone accounting, `LBManager::LastRoundDecisions`; `tests/charm++/load_balancing/replica_test`; `doc/internals/lb-multiplicity.md`; the manual |
| 5488fde06 | hapi: a vmm chunk of exactly 2048 MB is refused off-node (the registration ceiling is 2^31 bytes) |
| 07abb5815 | moe: experts as balancer-made copies (up to four), bf16, serving, real-routing replay, the transformer block, checksums every n steps, ShedLB linked by the Makefile; the FastMoE and Megatron-Core harnesses; the kernel bound found on 2026-11-02 and its fix |
| a4ab7a902 | this file |
| 778631db3, d7a4ad9d4 | the routing traces as a GitHub release; the FastMoE patch and build script |
| 7f3a67e46 | moe/ref: NCCL on Slingshot through the aws-ofi-nccl plugin, and the transport logged (correction 1 below) |
| a2c2a1b23 | moe: serving ran two of four partitions (correction 2); the matrix re-measured; `-DMOE_K_PART` |
| 3bfecb852 | moe/ref: Transformer Engine grouped experts, FasterMoE serving without the per-step collection, fp32 masters (corrections 3 to 5); not pushed at the time of writing |
| e6f9b8aa9 | moe: cuDNN fused attention for the block variant, with a startup self-check (item 6); the eight-GPU matrix with the baselines as their authors run them |
| e26ee53bf | moe: Frontier (MI250X, ROCm 7.14): the fused attention on aotriton, the runtime fixes, the one- and two-node matrices with the baselines |
| 2026-11-08 (with reconverse's of the same day) | the runtime clone (`ckCloneTo` through the element's pup); moves that ship their Stable device buffers first and switch at their landing; copies made on their source's PE and switched by the runtime at their landing (not yet run anywhere); the bulk-read queue in reconverse's LCI backend; the section "Since 2026-11-07" below |
| 2026-11-07 | the balancer chooses the copy count and prices it from the fan calibration (`-R` gone); the cross-host channel put announced and read by the owner, with the micro-benchmark that decided it (`benchmarks/charm++/cuda/gpudirect/rma_bw`); the built-in cost tables of both Delta machines re-measured by channels; the A100 rows; the section below |

Everything in them was measured on A40s only, one and two nodes (the tables in
`examples/charm++/cuda/gpudirect/moe/README.md`). On AMD nothing has run since 2026-10-22, when leanmd,
jacobi3d, chantest and sph3d ran on Frontier and moe only built. The replication code tests `CMK_GPU` and
never `CMK_CUDA`, the MoE example calls only `hapi*` names, its bf16 values are raw 16-bit words with
hand-written conversions, and `hapi_blas.h` has both vendors' GEMMs, so the port is expected to be a build,
not a rewrite.

## Corrections since 2026-11-02, and the mistakes behind them (written 2026-11-03)

Read this before trusting any number in the README or in memory. Between 2026-11-02 and 11-03 the user
asked what a fair comparison must match, and the audit that followed found that several published
numbers were artefacts of how I had run the baselines or the example, every one of them in the example's
favour. The pattern was the same each time: a baseline came out slow, and I wrote the number down without
checking that the baseline had been run the way its authors run it. The README's sections "Measured
2026-11-02" and "Measured 2026-11-03" hold the corrected tables; the earlier sections carry "superseded"
notes. Dated, with the cause and what now prevents it:

1. **The NCCL baselines ran without the Slingshot plugin** (every two-node baseline before 2026-11-02).
   `NCCL_DEBUG=INFO` showed `NET/IB : No device found`: socket transport across the nodes. The baselines
   gained 11 to 48% on the plugin and every published ratio shrank. `ref/refwrap.sh` now sets what
   `module load aws-ofi-nccl` sets and logs NCCL's INIT and NET lines, so a rank log without `Using
   network Libfabric` is known to have run without it. On Frontier the equivalent is RCCL with the
   `aws-ofi-rccl` plugin: make the harness log the transport there too, and read the line before the
   number.
2. **The example served two of its four partitions** (from the four-partition change of 2026-11-01
   17:20 until a2c2a1b23): the serving branch still looped over partitions 0 and 1, half the
   dispatchers got unwritten outputs, the serving step looked 30% slower and the serving checksums of the
   "Copies up to four" paragraph were wrong. Training was right. Now every partition is served and every
   plain-against-copy serving checksum pair agrees. `-DMOE_K_PART=2` builds the two-partition variant.
3. **The Megatron-Core arm never ran grouped GEMM.** Megatron-Core 0.20's local layer spec builds
   `SequentialMLP` whatever `moe_grouped_gemm` says (there is no `GroupedMLP` in 0.20); the grouped path
   is the Transformer Engine spec, Megatron-LM's default. The harness printed "grouped GEMM" from its flag.
   `ref/megatron_moe.py --te` runs TE's grouped experts (7 to 10% faster on SmartMoE and Mixtral, nothing
   on OLMoE) and the header line prints the expert module that ran. On ROCm, Transformer Engine is a
   separate fork; without it the Megatron arm is the sequential one, and the row must say so.
4. **FasterMoE's serving step paid a full garbage collection every step** (every FasterMoE serving row
   before 2026-11-03): my harness collected the autograd graphs FastMoE's fused forward records, and the
   collection walked every long-lived object, about 55 ms a step. `gc.freeze()` before the loop. With
   it FasterMoE serves SmartMoE's routing on four GPUs in 27.0 ms/step, equal to the example with ShedLB
   and ahead of the plain example: its shadow of the hottest expert does in serving what ShedLB's copy
   does. The published "3.6x over FasterMoE in serving" is void; job G (below) re-measures eight GPUs.
5. **The baseline harnesses updated bf16 parameters in place**, less work than Megatron-LM's bf16
   optimizer, which keeps fp32 master weights as the example does. `--master` on both harnesses; job G
   runs every training arm with it and one arm without, for its cost.
6. **The example's attention was unfused** (score GEMM, softmax kernel, PV GEMM, and their backward)
   while the references' attention is PyTorch's fused flash kernel. On SmartMoE's routing the example's
   attention cost twice the references' (21.6 against 10.4 ms a step in training on eight GPUs). The
   fused path is cuDNN's attention graphs (`MOE_CUDNN_ATTN`, the Makefile; bf16 only, the fp32 path and
   the HIP build stay unfused), with a startup self-check that runs one training step on both paths and
   aborts past a relative L2 error of 5e-2. On Frontier there is no cuDNN: the block rows there carry the
   mismatch until a fused HIP attention exists (composable kernels' FMHA is the candidate), and the
   "Block:" header line names the kernels so a row cannot hide it.
7. **Process mistakes**, each now a rule: I submitted one job to two partitions to race the queue (never;
   one job in one partition, and the partition the user names is the standing choice: on Delta that is
   now `gpuA40x4-interactive` by sbatch, after a batch job sat 13 hours); I rebuilt the `moe` binary while
   a job was running it (an arm died with rc=127; build variants under another name or after the job);
   and I passed an unparsed flag to a script and lost a job's first 20 seconds (dry-run the arms' argument
   lines against the scripts' parsers before submitting).

What I check before a baseline number enters a table, from now on: the log names the module and the
transport that ran, never the flag; a serving step is not slower than the same system's training
forward; a system's enhanced variant does not lose to its plain one without a written explanation. The
harness options that make the baselines what their authors run: Megatron `--te --grouped --master`,
FastMoE training `--master`, FasterMoE serving with the freeze (automatic).

**Pending at the time of writing.** Job G (23497275, two A40 nodes): the Megatron arms with TE experts and
masters, FastMoE and FasterMoE training with masters, FasterMoE serving fixed, on the three routings and
the two block variants. Job I (to follow, two nodes): the example's block rows with the fused attention
against the unfused build, plain and ShedLB, training and serving. Their tables go into a "Measured
2026-11-04" section of the README; the headline ratios change again, and the Frontier ladder (below)
should take its A40 reference rows from that section, not from 2026-11-01.

## Since 2026-11-04: the copy count, the A100 nodes, and the read path across hosts (written 2026-11-07)

All of it is in the commit of 2026-11-07 on `rate-aware-gpu-lb` (made in the Delta clone, not pushed) and
measured on Delta's A40 and A100 nodes; nothing has run on Frontier since 2026-11-04. In order:

1. **The balancer chooses the copy count** (2026-11-05). `-R` is gone. An expert declares that it can be
   copied, into how many parts, and what a copy's per-step exchange costs in bytes and messages
   (`setLBReplica` with `parts`, `exchangeBytes`, `exchangeMsgs`, `extraPerPeer`); ShedLB plans every copy of
   a round at once, up to four, and prices each count from the declared bytes against the calibration's fan
   entries (a two-, three- or four-way exchange on one host and across hosts, measured by `lbcalib -F 4`
   through device channels). `doc/internals/lb-multiplicity.md` and the manual carry the protocol. There is
   no feedback loop from measured steps: the user's rule is to fix the model or the cost, never to retry.
   GPU timelines of the example against FasterMoE on eight A40s (README "Measured 2026-11-05") showed
   FasterMoE's step to be NCCL-bound, its shadow of the hottest expert about 15 ms a step.
2. **Two A100 nodes** (2026-11-06, `gpuA100x4-interactive`). NVLink inside the node: a copy's exchange costs a
   peer 93 GB/s on an A100 host at any copy count, where the A40 host's PCIe gives 24, 16 and 14 at two,
   three and four copies (the fan-in collapse seen on 2026-11-05 is the PCIe fabric's). The example's
   outputs go back to the dispatchers through channels too. The serving step across the two A100 hosts came
   out *worse* than on the A40s (plain 24.8 ms against the A40's 29.4 was expected to halve), and that was
   root-caused the next morning.
3. **The read path across hosts** (2026-11-07). The cross-host channel put was a one-sided write with its
   note as immediate data (commit 88d4f724d, 2026-10-19). It was not the signal, not chunking (16 writes
   gave 6.3 GB/s), and not LCI's `FI_RMA_EVENT` registration flag (the NIC's restricted-packet counter
   proved the packet format changed while the rate did not). A library-level micro-benchmark, now
   `benchmarks/charm++/cuda/gpudirect/rma_bw` (LCI over libfabric, no Charm++; `make BUILD_HIP=1` here),
   measured the fabric on two A100 nodes at 128 MB:

   | GB/s                                        | one direction | both hosts sending at once |
   |---------------------------------------------|--------------:|---------------------------:|
   | host to host, put or get                    | 24.0          |                            |
   | device source into host memory, put or get  | 16.1          |                            |
   | host source into device memory, put         | 12.3          |                            |
   | host source into device memory, get         | 24.3          |                            |
   | device to device, put                       | 10.9          | 3.7                        |
   | device to device, get                       | 16.1          | 12.5                       |

   A write landing in device memory is the slow leg on Slingshot; the same bytes fetched by a read land at
   the link's rate. The verb matters only at the destination side, where the two verbs do the same PCIe work
   and differ in the NIC's role: on a write the target must accept, translate and acknowledge every packet
   and the sender's window waits on it; on a read the requester writes the responses locally and nobody
   waits. On InfiniBand the usual finding is the reverse (writes at least as fast as reads; the slow leg
   there is a NIC *reading* GPU memory), so the verb across hosts is a machine property, not a protocol
   constant. The runtime now announces every cross-host put and the owner reads it (`pull_mode` across hosts
   in `CkDeviceChannel::onResolved`; the path the channels already used for a put made during a move), the
   write stays behind `CHARM_CHAN_WRITE=1` for measurement, and two defects of the dormant read path were
   root-caused from the channel's event trail and fixed (the closing ack reported announced puts as written;
   a zero-byte put's note overtook the owner's queue; a release that finds no put now dumps the trail).
   Exchange across hosts, per peer: A40 4.9 to 14.2 GB/s, A100 3.0 to 12.1. The built-in cost tables of both
   Delta machines were re-measured by channels on this path, and ShedLB prints each copy's placement
   outcome and the host-time refusals without `+LBDebug`. The rows, one binary, one job each, steps 30-41,
   every ShedLB checksum equal to plain's:

   | two A100 nodes, ms/step (job 23638138; the first run on another pair in brackets) | serve | train |
   |---|---:|---:|
   | example plain | 14.5 (14.7) | 37.9 (37.3) |
   | example ShedLB, four copies placed with their inputs | 6.2 (6.3) | 20.3 (22.6) |
   | FasterMoE, eight ranks, its A100 parameters, `NODES=2` | 19.8 (15.1) | 32.2 (32.1) |

   | eight A40 GPUs, ms/step (job 23637512) | plain serve | ShedLB serve | ShedLB train |
   |---|---:|---:|---:|
   | SmartMoE (plain train 81.2) | 29.4 | 11.1 | 45.5 |
   | Mixtral (no copy made) | 63.8 | 63.2 | 144.5 |
   | OLMoE (no copy made, host-bound) | 124.8 | 115.4 | 120.8 |

   Two process mistakes on the way, both now rules: a FasterMoE arm ran without `NODES=2` exported (its
   `run_fm.sh` defaults to one node; the two rows were four-GPU runs and were discarded; the job script now
   exports it and prints the rank count), and a job pointed at a binary in the session's `/tmp` scratchpad,
   which the compute nodes cannot see (15 s of failed launches; binaries for a job live under `/scratch`).

**What this means on Frontier.** Its NICs are Slingshot too (Cassini, four per node), so every two-node
Frontier row of 2026-11-02 and 11-04 ran its cross-host exchange on the write path. The first job after the
pull, before any row: `rma_bw` on two nodes, one rank per node, `dd` and `hh`, put and get, alone and `both`,
128 MB. It says whether the MI250X's write-into-device leg has the same asymmetry. If the write wins there,
`CHARM_CHAN_WRITE=1` in the runner restores the 2026-10-19 path (the proper form, a startup probe or a machine
profile instead of an environment variable, is open item 10). Then the two-node rows again on the read path;
open item (3) of the two-node notes, OLMoE's exchange crossing at one NIC's bandwidth, has a second candidate
cause now. The transfer-cost table must be re-measured: the built-ins are Delta's and MI250X gets the A40 one
without `+LBCostConfig`. Read ShedLB's placement lines in the rank log: a copy placed away from its inputs makes
the cross-host exchange the step. The Makefiles' `-DHAPI_CUDA_CALLBACK` was dead text (the flag is the
library's) and is gone; nothing to do here.

## Since 2026-11-07: the runtime clone, and copies switched at their landing (written 2026-11-08)

One charm commit and one reconverse commit of 2026-11-08 on `rate-aware-gpu-lb`, pushed together. Pull both:
charm's `ckrdmadevice.C` calls reconverse's new `CmiRdmaSetBulk`. `pup.h` (a new virtual in `PUP::er`),
`cklocation.h`, `cklocrec.h` and `ckmigratable.h` changed, so the tree needs a full rebuild. A single-target
`make` gave a binary that died at start-up on Delta.

**Status, plainly.** Items 1 and 2 ran on Delta: A40 and A100, two nodes each. Item 3 is the A100 evidence
for what was still wrong. Item 4 is the fix for that; it is built but has not run on any machine yet. Its
first runs are queued on Delta's A40s (job 23674111, tasks `10_rs_serve` to `50_rs_tests` in
`/scratch/mzu/bhosale/moe_runs/a40ab/queue`). Frontier is free to run it first.

1. **The runtime clone.** `CkMigratable::ckCloneTo(toPe, newIdx)` makes a copy of an element at another
   index of the same array (the array needs `CkArrayOptions::setBounds`). The element's own pup runs with
   `PUP::er::IS_CLONING` on both sides, so the pup decides what a copy inherits (`p.isCloning()`). The
   state ships as a migration's does while the source keeps running. The copy gets `ckJustCloned` once
   its state is in, and the source gets `ckCloneSent` once the pack has finished reading its buffers. A copy
   starts with no SDAG state (`charm++.h` skips `_sdag_pup` when cloning): the source's open `when`s and
   their buffered messages stay with the source. The MoE example now makes every copy this way; its own
   `takeState`/`takeMoments` protocol is gone. A copy is told its first step by `startCopy`.
2. **Moves that ship first and switch at their landing.** A device buffer pupped as
   `p.pup_device(ptr, n, PUP::DeviceLife::Stable)` promises that nothing writes it. Under `+LBAsync` the
   balancer's decision ships a snapshot of the Stable buffers at once (`stableSnapshot`), and the element
   keeps stepping. The move is held until the snapshot has landed: `finishStableSend` kicks it. The pack
   then carries only the rest. `CHARM_DEBUG_MIGRATE=1` hashes each Stable buffer at both points and aborts on
   a broken promise. The example declares its weights Stable in serving and Mutable in training. During a
   round, an expert parks only once the round is over (`LB_PARK_MAX` bounds the wait), and a dispatcher
   sends no tokens ahead until two steps after `roundResumed`. Four bugs were found and fixed on the way:
   - **ABI skew.** The new `PUP::er` virtual needs the full rebuild above.
   - **A move whose device buffers were all Stable.** It unpacked from a null packed region; the manifest is
     now passed whenever either region came (`immigrate`).
   - **Copies inherited the source's SDAG buffers.** Fixed by the `charm++.h` change in item 1.
   - **ShedLB's host cap refused copies.** It refused `replica_test`'s copy. A copy now passes the cap when
     the destination's host time stays below the source's.

   Validation on Delta (A40):

   | Check | Result |
   |---|---|
   | chantest | 6/6 arms pass |
   | Stable arms | 8/8 pass |
   | `replica_test` | PASS |
   | serving, training and Mixtral checksums | equal to plain's, on A40 and on A100 |

   Two problems are still open:
   - a token-conservation abort in 1 of 21 runs;
   - a segfault at exit (rc 139) in some serving runs of the 2026-11-07 binary.
3. **What the A100 rounds showed** (2026-11-08, two nodes). Steady state was good. The balancing rounds
   were not, and every earlier comparison with FasterMoE had left them out, which flattered ShedLB.
   Balancing runs every 20 steps; a period is one round plus the steps after it. ms/step:

   | | After the rounds | Period with round 2 (40-59) | Period with round 1 (20-39) | FasterMoE |
   |---|---:|---:|---:|---|
   | SmartMoE serving, ShedLB | 3.7 | 12.4 | 19.8 | 19.8 (steps 30-41), 21.2 (5-43) |
   | SmartMoE serving, plain | 14.2 | 14.2 | 14.7 | |
   | Mixtral serving, ShedLB | 21.1 | 35.9 | 59.4 | 51.3 (2026-11-07) |
   | Mixtral serving, plain | 52.0 | 52.2 | 53.0 | |
   | SmartMoE training, ShedLB | 23.6 | (40-step run) | 26.6 | 32.2 (30-41), 36.5 (5-43) |
   | SmartMoE training, plain | 37.9 | | 38.7 | |

   The FasterMoE rows come from other A100 nodes on 2026-11-07. Every comparison with FasterMoE from now on
   reports the full-period means beside the after-round window, because FasterMoE's figures include its own
   per-step rebalancing.

   Inside the rounds the GPUs were 11-16% busy, against 84-86% in steady state. The cause: every dispatcher
   routes to a new copy from a fixed step, whether or not the copy's state has landed. Copies land one after
   another at link speed, and the step is all-to-all, so the whole step waits for the last copy. Round 2 of
   one run:

   | | Time from step 46's start |
   |---|---:|
   | 18 copies decided onto host 1; their state (134 MB each) starts shipping | -14 ms |
   | first copy lands | +26 ms |
   | 5th / 10th / 15th copy lands | +46 / +79 / +111 ms |
   | last copy lands | +146 ms |
   | step 46 ends on every dispatcher | +148 to +152 ms |

   Most of those copies were locality copies of nearly idle experts: 26 of the 34 in that run. Pricing a
   copy against what it saves over one period is an open item; the balancer has no such price yet.

   A first diagnosis blamed contention on the link. The bulk-read queue built on it changed nothing:

   | | Payloads whole | Bulk queue on |
   |---|---:|---:|
   | steady state, 48-61 | 3.7 ms/step | 3.8 ms/step |
   | GPU busy, round 1 (19-27) | 11% | 12% |
   | GPU busy, round 2 (40-48) | 16% | 16% |

   The queue is in this commit anyway. Reads of migration state (`CkIsStatePayloadEntry`) go to the LCI
   backend's bulk queue: 2 MB pieces, at most 8 MB in flight, 2 MB while other gets are pending. It is on
   by default; `CHARM_RDMA_BULK=0` turns it off, and `CHARM_RDMA_BULK_{CHUNK,CAP,BUSY}_KB` set the sizes.
   chantest has an arm for it (`rdma_lb_big`). It is a candidate for removal.
4. **Copies switched by the runtime at their landing (new, not yet run).** No step should wait for a copy,
   no copy for another copy, and the application should not handle any of it. `CkLocMgr::cloneElement`
   now unpacks the copy on the source's own PE whenever its state has Stable device buffers (and the
   device pool and arena mode are on). It does so through a local `immigrate` message, after a host wait
   on the migration stream for the device-to-device pack. Messages can reach the copy at once.
   `immigrate` then calls `CkLocMgr::moveCopy`, which treats the copy as a balancer's deferred move:
   - the snapshot of its Stable buffers ships to the balancer's PE while the copy runs;
   - `finishStableSend` kicks the move when they have landed;
   - the pack carries the rest, and the copy switches.

   Each copy switches on its own landing. Until then it serves from its source's GPU, which carries the
   load it carried before the round.

   The copy's arrival is not counted by the balancer: `CentralLB::Migrated` would count it toward
   `migrates_expected`, so `copyInTransit` on the record makes it `lbUntracked` on the message. A
   balancer's own later move of the copy clears the flag and is counted.

   - **Comparing.** `CHARM_CLONE_DIRECT=1` restores the old path (the copy unpacked at its destination,
     messages waiting for it), so one binary gives the A/B.
   - **Training.** Its weights are Mutable, so training takes the old path, unchanged.
   - **The example.** The application-level switch is removed. A copy starts at the round's release step,
     as before, wherever the runtime has it at the time. In serving, a copy's pack skips
     `pup_device_order`, since nothing writes the weights.
   - **Memory on the source's GPU, per copy.** The pack's payload block until the unpack, plus the copy's
     buffers, plus the snapshot block until it lands. On Delta that is about 0.5 GB per SmartMoE copy and
     1.6 GB per Mixtral copy.
   - **Not built.** Replica routing (senders address the object and the runtime picks a replica). Here each
     copy stays an element with its own index. Also not built: a test that reaches `moveCopy` (only the
     MoE runs reach it), the `doc/internals` chapter, and the price for locality copies above.

**The experiment.** Run this on Frontier, or on Delta if it gets there first. Use one binary, two nodes
(16 GCDs), the buddy pool flags of the Run section, and `+LBCostConfig` with a Frontier table. Serving,
60 steps, rounds at 20 and 40, two runs of each arm for the band:

    SM60="-e 32 -m 2048 -h 8192 -k 1 -t 8192 -i 60 -u 3 -B -E traces/smartmoe_L02.trace"
    MX60="-e 8 -m 4096 -h 14336 -k 2 -t 8192 -i 60 -u 3 -B -E traces/mixtral_L12.trace"
    run_app rs_plain   $SM60 -G -f 99999
    run_app rs_new     $SM60 -G -f 20 -b 20 -a $COST                          # copies switched at landing
    CHARM_CLONE_DIRECT=1 run_app rs_direct $SM60 -G -f 20 -b 20 -a $COST     # the old copy
    # the same three with $MX60; then once each with +gputrace <dir> +gputracenocopy for the GPU busy time,
    # and once rs_new under CHARM_DEBUG_MIGRATE=1 for the mechanism

Read the results this way:
- **The table.** Mean ms/step over 20-39 and 40-59 (the full periods), 48-61 (after the rounds) and 5-62,
  one row per run.
- **Correctness.** Checksums (`out2`, `w2` on the `Step n:` lines) equal to plain's. Kill the experiment
  on a checksum that differs, an abort, or a hang.
- **GPU busy time.** `python3 ranklogs/gpubusy.py <trace dir>` gives each GPU's busy fraction in steps
  48-61, 19-27 (round 1), 40-48 (round 2) and 5-62. `python3 ranklogs/gaps.py <trace dir> --rounds 23,44`
  gives the long steps after the rounds. Both read the runtime's own `+gputrace` files, so they should work
  unchanged with the rocprofiler backend; that is not yet checked.
- **The mechanism**, under `CHARM_DEBUG_MIGRATE=1`. `[CLONE n] ... here, then to pe m` is a copy made on
  its source's PE. `[CLONE n] ...: made here, moves to pe m when its Stable buffers have landed` is its move
  held. `[STABLE n] id=... landed at pe m after x ms; move kicked` is the switch. `[CLONE n] ... on pe m`
  is a copy that took the old path.
- **Success** means rounds whose GPU busy time is near steady state's, and full-period means near the
  after-round window. Compare FasterMoE on those full-period means.

## What Frontier changes for the experiments

| | Delta, gpuA40x4 | Frontier |
|---|---|---|
| GPUs per node | 4 A40 | 4 MI250X = 8 GCDs of 64 GB; one GCD per process (`+gpudevs 1`) |
| the eight-GPU matrix | 2 nodes | 1 node; 2 nodes = 16 GCDs (two nodes stay the cap) |
| pool across nodes | vmm, `+gpupoolsize 16384 +gpupoolchunk 2046` | buddy only (ROCr cannot dma-buf-export `hipMemCreate` memory): `+gpupoolalloc buddy +gpupoolsize 2048 +gpupoolarenas 4`, `CK_GPU_ARENA_TOTAL_MAX_MB=16384` |
| pool on one node | vmm | vmm works (the IPC import keeps the fd open until the grant) |
| a one-node `srun` step | plain | `--network=single_node_vni`, else cxi finds no VNI and LCI aborts |
| transfer-cost table | built in (the A40 and the A100 tables, by channels on the read path, 2026-11-07; any other device name gets the A40 table, MI250X included) | `+LBCostConfig lbcost.frontier.chan.conf`, measured 2026-11-07 by `lbcalib -F 4` through channels in the example's layout (`apps/lbcalib`, launched by `irun.sh`; `moe_run.sh` passes it, `LBCOST=` overrides); the 2026-10-22 `lbcost.frontier.conf` predates the fan entries |
| per-rank wrapper | `ranklogs/moewrap.sh` (NUMA node 3 - local rank, 16 cores each) | `frontier_wrap.sh` (a GCD to its L3 region, the region's first core left to the OS); adapt `moewrap.sh` to that map |
| LCI devices | one per process | one per PE when a process has several (`+lci_ndevices 7` for leanmd); moe runs one PE per process |
| completion flags | memset writer | kernel writer (the HIP default; `+gpuflagmemset` restores the event-and-memset one) |
| profiler | CUPTI | rocprofiler-sdk 1.3.2 with the pool-release patch (`ext/rocprofiler_patched`, loaded ahead of ROCm's by `irun.sh`); the legacy intercept queue is preset by the runtime |
| MR cache monitor | `FI_MR_CACHE_MONITOR=uffd` | `kdreg2` or `uffd` (within noise of each other) |
| application streams | one per chare | `+gpustreams 32` (the HIP default) |
| allocation | `sbatch -A mzu-delta-gpu -p gpuA40x4` | `sbatch -A csc710 -p batch` |

The ledger question in the README ("no bin can hold one" for a third copy of a default-shape expert on a
16 GB pool) meets 64 GB GCDs here. The ledger prints what each node advertises at the first balancing
step; a buddy pool advertises arena by arena (one 2048 MB arena advertised 0 MB on 2026-10-22, below the
reserve; two advertised about 1.4 GB).

## Build

Paths on Frontier as of 2026-10-22: `/lustre/orion/csc710/scratch/adityapb/hip` with `env.sh` (the
modules; do not load `craype-accel-amd-gfx90a`), the HIP LCI at `lci_install_hip`, the charm clone,
`irun.sh` and `wrun.sh` (a run, and a run with a stall catcher that dumps every process's stacks),
`frontier_wrap.sh`, `ext/` (the patched profiler, the crash catcher `segvbt.so`), `vmmtest/` (the driver
repros).

1. **Sources.** Charm: fetch `rate-aware-gpu-lb` and fast-forward to the commits above. Reconverse: the
   branch was rebased on 2026-10-31 (`038893f`, forced over the `76fe574` Frontier has), so in the
   reconverse clone `git fetch && git reset --hard origin/rate-aware-gpu-lb`. The recharm merge of
   2026-10-31 (`9327afa2b`) replaced the build system: reconverse is the only network,
   `cmake/fetch_reconverse` is gone, `--with-reconverse-dir` names the clone, and the HIP option is spelled
   `amd` (or `hip`) with a `-amd` tree suffix. The 2026-10-22 tree, `multicore-linux-x86_64-hip`, predates
   all of that: build a new one.
2. **Environment.** `source env.sh`; `export CC=cc CXX=CC FC=ftn` (else the system GCC 7.5 builds it);
   `$ROCM_PATH/lib/llvm/bin` on `PATH` (charmc calls bare `clang`).
3. **LCI.** The HIP install of 2026-10-22 (UIUC-PPL/lci master 662b4736, ofi, no CUDA) still serves;
   `ldd $LCI_HIP/lib64/liblci.so` must show no CUDA library. The fork's next commit, b7c289ae of 2026-11-07
   (`/u/bhosale/lci`, master; not pushed), requests `FI_RMA_EVENT` only under `LCI_OFI_RMA_EVENT=1`: the
   2026-10-19 request put every remote write into cxi's ordered packet format and bought nothing (3.0 against
   3.1 GB/s on Delta's A100 nodes), so the old install is not wrong, only heavier. Rebuilt from b7c289ae on
   2026-11-07 (`cmake --build lci/build-hip -j16 && cmake --install lci/build-hip`; no CUDA library in `ldd`).
4. **Runtime.** The Delta configuration with the AMD option and the HIP LCI:

       CC=cc CXX=CC ./buildcmake charm++ reconverse-linux-x86_64 amd -j16 \
         --with-reconverse-dir=$PWD/reconverse \
         --with-cmake-args="-DLCI_ROOT=$LCI_HIP -DLCI_DIR=$LCI_HIP/lib64/cmake \
           -DRECONVERSE_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
           -DCMK_USE_SHMEM=ON -DLBUSERDATA=1 -DCMAKE_BUILD_TYPE=Release"

   Expect the tree `reconverse-linux-x86_64-amd`. Run it once: `buildcmake` removes everything under an
   existing tree (`rm -rf "$builddir"/*`), `apps/` with the traces and binaries included. After a pull rebuild
   in place with `make -j16` inside the tree (cmake tracks the headers; three minutes on 2026-11-07), then the
   apps from clean objects. This line was last run on 2026-11-02. The
   2026-10-22 corrections to watch for are in `HIP_PORT.md`: without `-DRECONVERSE_ENABLE_HIP=ON`
   reconverse builds with `CMK_HIP 0`; autofetched non-HIP `liblci*.so` in the tree's `lib/` shadow the HIP
   LCI at run time (move them aside); clang rejected `void*` arithmetic in `buddy_allocator.cpp` (fixed
   then). After any `./buildcmake` the root symlinks `bin include lib` point at the last tree built. The LB
   modules come with the tree (`EveryLB` holds ShedLB; `DiffusionLB` is its own module). Headers of the LB
   database changed in bfcdc743e (`lbdb.h`, `LBManager.h`, `BaseLB.h`, `LBMemoryContract.h`,
   `ckmigratable.h`, `cklocrec.h`); after any later header edit rebuild the tree, a single-target `make`
   leaves ABI skew.
5. **Applications.** As on 2026-10-22, as copies in the tree: rsync `examples/charm++/cuda/gpudirect/moe/`
   to `<tree>/apps/moe/` with `hapi_blas.h` and `hapi_kokkos.h` beside it, remove the A40 binaries (`moe`,
   `moe.*`) and objects, `make CHARM_DIR=<tree> moe`. The Makefile reads `BUILD_HIP` and `HIP_ARCHS` from
   the tree's `conv-mach-opt.mak`, compiles `moe.cu` with `hipcc -x hip` and links `-lhipblas`. New since
   the last AMD build, and checked only by compiling: `hipblasGemmEx` and `hipblasGemmStridedBatchedEx`
   with `HIP_R_16BF` operands and `HIPBLAS_COMPUTE_32F`. Also `tests/charm++/load_balancing/replica_test`
   (`make CHARM_DIR=<tree>`): host-only, but run inside an allocation like everything else (`+p4`).
6. **Traces.** The three routings are 3.3 GB of `.trace` files, ignored by git, in Delta's
   `/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/moe/traces/` (72 layer files and
   `mixtral_tokens.npy`), and published, zstd-packed to 473 MB, as release `traces-2026-11-01` of
   <https://github.com/adityapb/charm-moe-traces> (private; a token with repo scope downloads it). On
   Frontier: `GH_TOKEN=<token> ./fetch_traces.sh apps/moe/traces` with the script from that repository
   (it checks the sums), then `zstd -dc <set>_traces.tar.zst | tar -xf - -C apps/moe/traces` for each of
   the three sets; its README has the format and the generating commands. The README's rows use
   `smartmoe_L02`, `mixtral_L12` and `olmoe_L12`. Regenerating them instead needs the checkpoints (Mixtral-8x7B 87 GB and OLMoE-1B-7B
   13 GB, in Delta's `/scratch/mzu/bhosale/hf`) and SmartMoE-AE's `moe_trace` tables (`smartmoe_src` in
   the rescued scratchpads below); `traces/*_stats.txt` are the per-layer skews that chose the layers.
7. **Baselines**, for the matrix only. `ref/moe_ref.py` (the PyTorch reference, the example's twin) runs
   on ROCm PyTorch with RCCL; `ref/megatron_moe.py` needs Megatron-Core (its ROCm fork); FastMoE's CUDA
   extension has no ROCm build, so `ref/fastmoe_moe.py` and FasterMoE are a port (hipify) or stay A40
   figures. `ref/moe_ref.py --lb sync --place smartmoe -f 10 -b 10` is SmartMoE's online placement
   (ATC'23) ported from its artifact, added 2026-11-05 with one-node A40 rows in README "Measured
   2026-11-05"; it runs wherever the reference runs, and `ref/smartmoe_check.py` needs the artifact's
   `mapping.py` only to re-check the port. FlexMoE (SIGMOD'23) has no code; the README cites its paper.

## Run

A job is one `sbatch` script that runs every arm and ends; nothing is queued without work attached; one
runner per allocation; `squeue -u $USER` before anything else; `ulimit -c 0`; at most one hour and two nodes.

    #!/bin/bash
    #SBATCH -A csc710 -p batch -N 1 -t 01:00:00 -J moe-ladder
    #SBATCH --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=8
    #SBATCH -o /lustre/orion/csc710/scratch/adityapb/hip/runs/%x-%j.out
    source /lustre/orion/csc710/scratch/adityapb/hip/env.sh; ulimit -c 0
    export LD_LIBRARY_PATH=$TREE/lib:$LCI_HIP/lib64:$ROCM_PATH/lib:$LD_LIBRARY_PATH
    export LD_PRELOAD=$LCI_HIP/lib64/liblci.so    # irun.sh adds the patched profiler and the crash catcher
    export FI_MR_CACHE_MONITOR=kdreg2 CHARM_LB_HELPER=near PMI_MAX_KVS_ENTRIES=8192
    run_app() { tag=$1; shift; RL=$RUNS/ranklogs/$tag; rm -rf $RL; mkdir -p $RL
      RANKLOG_DIR=$RL timeout 300 srun -N 1 -n 8 --ntasks-per-node=8 --gpus-per-node=8 \
        --cpu-bind=none --exact --kill-on-bad-exit=1 --unbuffered --network=single_node_vni \
        $APP/ranklogs/moewrap.sh stdbuf -oL $APP/moe "$@" +gpudevs 1 \
        +gpushm +gpupool +gpupoolsize 16384 +gpuipceventpool 256 +gpupoolalloc vmm +gpupoolchunk 2046 \
        > $RL/srun.log 2>&1; }

Two nodes take `-N 2 -n 16` without `--network=single_node_vni`, and the buddy pool (`+gpupoolalloc buddy
+gpupoolsize 2048 +gpupoolarenas 4`, `CK_GPU_ARENA_TOTAL_MAX_MB=16384`) in place of the vmm flags. After
any kill remove `/dev/shm/charm-chan-*` and `/dev/shm/charm-hapi-host*` on the node. The map
`frontier_wrap.sh` pins by, from the 2026-10-21 handover: GCD 0 cores 48-55, 1: 56-63, 2: 16-23, 3: 24-31,
4: 0-7, 5: 8-15, 6: 32-39, 7: 40-47, the first core of each region to the OS in low-noise mode; confirm it
with `rocm-smi --showtopo` before trusting it (an unpinned run was 18-60x slower on Delta).

The three routings and their shapes (`$SM`, `$MIX`, `$OL` below):

| Routing | Model arguments | Trace |
|---|---|---|
| SmartMoE's training routing, layer 2 (hottest expert 48% of the tokens) | `-e 32 -m 2048 -h 8192 -k 1 -t 8192` | `traces/smartmoe_L02.trace` |
| Mixtral-8x7B on WikiText-103, layer 12 (15.9%) | `-e 8 -m 4096 -h 14336 -k 2 -t 8192` | `traces/mixtral_L12.trace` |
| OLMoE-1B-7B on WikiText-103, layer 12 (5.7%) | `-e 64 -m 2048 -h 1024 -k 8 -t 8192` | `traces/olmoe_L12.trace` |

Common: `-i 40 -u 3 -B`, steady state steps 23-43; plain `-f 99999`; ShedLB `-f 20 -b 20 -a` (since
2026-11-05 the balancer chooses the copy count, up to four; `-R` is gone, so the `-R n` arms below are the
ladder as it was run on 2026-11-02, and `-R 4 -f 10 -b 10 -a` reads `-f 10 -b 10 -a` now);
serving `-G`; the transformer block `-w`; checksums `-C 4`. The per-rank log `rank_0.log` carries the
`Step n:` lines with `out2` and `w2`; `ShedLB: replicate` and `ShedLB: retire` lines name the decisions.

## The first job: the validation ladder

*Done 2026-11-02 (job 5833510, interactive): Parts 0-2 pass after three fixes (the runtime's
imported-pointer copies through the batch API on HIP, a new copy's landing buffers made before its state,
the serving branch over all four partitions); the rows are in the README's "Measured 2026-11-02:
Frontier" section. Open: one of seven `-R 4` runs of the fixed app still aborted at a new copy's first step
("a block sum from copy 1 for step 24 but this copy has 1 copies", logs/moe/sm_R4 of job 5833510); the six others,
five of them traced (`CHARM_MOE_TRACE=1`), passed bit-identical to plain. Part 3 waits on the baselines. The scripts: `moe_run.sh`, `moe_diag.sh`,
`ladder_int.sh <1|2>` under the scratch root, `JID=` the interactive allocation.*

*Done 2026-11-02, later (job 5837505, two nodes, frontier02930+03188, one hour): the three-way comparison
on 16 GCDs, every shape, training and serving, plus the `-w` rows: the README's "Measured 2026-11-02, later:
Frontier, two nodes" section. Scripts: `two_node2.sh` (the chain that ran: arms, checksum pairs, `scancel` at
the end; `two_node.sh` is its first draft), `n2_table.sh` (the README rows from `logs/moe/two_node.txt` and
`logs/ref/results_2n.txt`), `node_clean.sh` (leftover ranks and `/dev/shm` segments after a kill),
`moe_run.sh`/`ref_run.sh` take `NODES=`. The baselines' RCCL across nodes needs OLCF's aws-ofi-nccl plugin
(`/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib`, the `rccl-net-plugin` module's variables), set by
`ref_wrap.sh` when the step spans nodes; the test is `SLURM_STEP_NUM_NODES`, because `SLURM_JOB_NUM_NODES` is
unset in a step of a `--no-shell` allocation (the first three baseline runs went over TCP sockets for that and are
kept under `_socket` tags). A two-hour two-node debug request had a next-morning estimate; cut to one hour it
started in 35 minutes. Open from the run: (1) the reference's EPLB arm hangs in RCCL's channel setup before its
fourth step on 16 ranks (twice, both transports; `logs/ref/n2_sm_ref_eplb`); (2) three of four Mixtral ShedLB arms
segfault on PE 14 at exit, a device-channel receive completing inside the LCI backend's exit drain whose handler
sends (`logs/moe/n2_mix_train_R2/rank_14.log`; steps and checksums complete); (3) OLMoE's exchange crosses the
link at one NIC's bandwidth where RCCL's plugin uses four: whether LCI's endpoints spread over a node's NICs;
(4) ShedLB's one-copy-per-round rule cannot use an empty node when eight bins are loaded alike (Mixtral, 8
experts on 16 GCDs).*

*Pulled 2026-11-04 (five commits, d7a4ad9d4..e6f9b8aa9, the corrections section above; the nested reconverse is
unchanged at 038893f; the local changes were stashed, fast-forwarded and reapplied, two conflicts resolved by hand:
the serving loop of `moe.C` is upstream's, the README's Frontier sections sit before the Delta "Measured
2026-11-02" section). The merged tree's HIP `moe` builds (16 s; the attention unfused, which the "Block:" line now
names). What the corrections mean for the Frontier rows: (1) the two-node baselines already ran on the RCCL plugin
and the one-node ones cross no link; `ref_wrap.sh` now logs the transport by default (`NCCL_DEBUG=INFO`,
`INIT,NET`); (2) every Frontier serving row came from the fixed binary (built 14:54 on 11-02, "fixed app" in
`logs/moe/ladder_serve.txt`; the two-node job ran it), so none is invalid; (3) the Megatron rows say "sequential
experts" in the log header and in the tables, and Transformer Engine does not reach MI250X: AMD's prebuilt
`transformer_engine_rocm` 2.1.0 (repo.radeon.com, rocm-rel-7.0) carries gfx942 and gfx950 code only, a source
build of ROCm/TransformerEngine would need `NVTE_ROCM_ARCH=gfx90a` against the venv's torch, and its
single-launch grouped-GEMM backends are gfx942/gfx950 only, so TE's grouped experts on MI250X would be one
hipBLASLt GEMM per expert in any case; (4) no FasterMoE here; (5) the seven Megatron training rows (one node:
SmartMoE, SmartMoE cap 1.25, Mixtral, OLMoE; two nodes: SmartMoE, SmartMoE cap 1.25, OLMoE) ran without fp32
masters and were re-measured with `--master` on job 5847934 (`mg_master.sh`, tags `mgm_*`, `n2_mgm_*`; masters
cost 0 to 5 ms a step: SmartMoE 76.9 against 73.8 on one node, 142.0 against 137.4 on two); the serving rows do
not update weights and stand; (6) the HIP
build's attention is unfused and no baseline block row exists on Frontier, so nothing measured is affected, but a
block comparison here waits for a fused HIP attention (open item 9). The README's one-node section now compares
against the corrected A40 rows, and the two-node section's OLMoE paragraph no longer calls the Delta order
"the opposite": with the plugin the baselines are ahead on OLMoE there too.*

*2026-11-04, the fused attention on Frontier (correction 6 above, open item 9). The HIP build has it now:
aotriton, the library PyTorch's `scaled_dot_product_attention` dispatches to on ROCm, ships inside the torch
2.13+rocm7.1 wheel (`torch/lib/libaotriton_v2.so.0.12.0`, the kernel images under
`torch/lib/aotriton.images/amd-gfx90a`, the headers under `torch/include/aotriton`; the library exports the v3
API only, the params structs). `$W/aotriton/{lib,include}` holds copies, because the library's `NEEDED
libamdhip64.so` with `RPATH $ORIGIN` would, used in place, load the wheel's HIP runtime beside the runtime's
/opt/rocm-7.14.1 one; staged without it, the name resolves through the library path to
`/opt/rocm-7.14.1/lib/libamdhip64.so`, the same file as the already loaded `libamdhip64.so.7` (ldd shows one).
`moe.cu`: `MOE_AOTRITON_ATTN`, the same strided views of the packed qkv and dqkv arrays as the cuDNN path, the
causal forward with its persistent atomic counter zeroed per call, the backward with eager delta rows and an
fp32 dQ accumulator (zeroed per call), dropout off, as PyTorch 2.13's `mha_all_aot.hip` calls it. The Makefile's
HIP branch sets it when `$(AOTRITON_DIR)/include/aotriton/flash.h` exists (`AOTRITON_DIR`, default
`$W/aotriton`; `MOE_AOTRITON_ATTN=0` builds the unfused kernels). The startup self-check (one training step,
fused against unfused, abort past a relative L2 error of 5e-2) runs here as on CUDA, and the "Block:" line
names the kernels. Binaries: `apps/moe/moe` (fused), `apps/moe/moe.attn_unfused` (the merged tree's unfused
build, kept under its own name); `moe_run.sh` takes `BIN=`. The chain `fused_attn.sh` (the block rows plain
and ShedLB, training and serving, one node and two, the unfused build's plain arms on the same nodes, then
`mg_master.sh` with its `scancel`): job 5847934 (frontier04265 and 04266, 01:03 to 01:29), results
`logs/moe/fused_attn.txt`, the README section "Measured 2026-11-04: Frontier, the block rows with the fused
attention". Outcome: aotriton 0.12 (built for ROCm 7.1) runs under the 7.14.1 runtime; the self-check error is
the A40 value (3.6e-3 on the output, 1.1e-3 to 2.1e-3 on the weight gradients, both head sizes); the
attention's cost in training halves on one node (SmartMoE 13.3 to 6.4 ms, Mixtral 40.1 to 29.5), the serving
step on two nodes is exchange-bound either way; the Mixtral ShedLB two-node arms still take the exit segfault
(open item 2, PE 10 this time). The baselines' block rows then ran on the same allocation (`block_ref.sh`,
`--attn`, tags `smb_ref_*`, `mixb_ref_*`, `mgm_smb_*`, `mgm_mixb_*`, `n2_*`; every two-node log says "Using
network Libfabric", every Megatron header "SequentialMLP (local spec)" with the masters line), so the block
comparison exists on Frontier now, the same fixture on both sides: the README section has the tables.
Transformer Engine does not reach MI250X (the pull note above), so the Megatron rows stay the sequential ones.*

*Pulled 2026-11-07 (charm 4537c12c9, LCI b7c289ae; reconverse unchanged) and the two-node comparison re-run on the
read path: job 5877110 (frontier09162 and 09164, 11:42 to 12:19, one hour of debug QOS), the README section "Measured
2026-11-07: Frontier, two nodes on the read path". Builds: LCI reinstalled, the tree rebuilt in place (item 4 above),
`moe` and `moe.attn_unfused` from clean objects, `apps/rma_bw` (`make BUILD_HIP=1 CXX=hipcc`: `CXX ?=` loses to make's
default) and `apps/lbcalib` (its device report was guarded by `CMK_CUDA`, now `CMK_CUDA || CMK_HIP`; the HIP tree
defines both). What the job found, in the order the section above asked: (1) `rma_bw` on two Frontier nodes: put and
get both at 24 GB/s into device memory, alone and with both hosts sending (23.3 each way), every memory class alike;
no write penalty on MI250X and Cassini, so the verb is a wash here and the read path (the default) ran the rows;
registrations of 4 and 8 GB still fail on cxi without `FI_RMA_EVENT`, 2 GB runs at 24 GB/s (`rma_bw_sweep.sh`,
`logs/rma_bw.txt`). (2) `lbcost.frontier.chan.conf` by `lbcalib -F 4` in the example's layout: 23.4 GB/s per peer
across hosts, 145 / 34.8 / 50.3 on one host at fan 2 / 3 / 4 (the two dies of one MI250X, then another package);
`moe_run.sh` passes it. (3) The rows: SmartMoE ShedLB 67.8 training and 20.3 serving with one round (three copies at
once, two on the other host; 90.8 and 32.7 before), 51.7 and 17.8 with rounds every 10 steps; 2.6x over Megatron-Core
and 2.0x over the reference's best arm in training, 3.0x and 3.3x in serving. Mixtral serving plain 112.4 (164.6)
with the outputs through the channels, level with the reference; its ShedLB now makes one copy and gains nothing.
OLMoE could not run as pulled: with the outputs through channels the dispatcher's receive slab is 64 fixed segments,
3 GB a parity, over the 2048 MB arena and the 16384 MB per-process registration caps; `moe.C` now allocates it in
chunks of whole segments of at most 1024 MB (`recvRowPtr`, the combine kernel maps a row to its chunk; one chunk
wherever it fits, bit-identical there) and the rows ran with `-r 7.5`, the trace's own maximum (7680 rows): 124.4
and 131.6 (159.2 and 163.1 before), the entry-method output path 129.0 and 132.3 on the same nodes, ShedLB placing
nothing (host-bound). Open: the Mixtral ShedLB exit segfault persists (the PE holding the copy); the reference's EPLB
arm hung a third time; OLMoE still trails the baselines 1.7x to 2.6x. Uncommitted: `moe.C`, `moe.cu`, `moe.h`,
`moe_kokkos.cpp` (the chunked slab), `lbcalib.C` (the guard), this file and the README. Scripts under the scratch
root: `read_path.sh` (the chain, `SKIP_RB=1` skips its `rma_bw` section), `rma_bw_sweep.sh`, `ol_rows.sh`,
`ol_rows2.sh`, `rp_table.sh`, `swap_tail.sh` (stops a chain between arms). Process notes: the first `rma_bw` runs
failed on `taskset` because a one-task step with `-c 8` has no cores 49-55 (a full-node step, `-c 56`, has them);
pausing a chain with SIGSTOP ends it (the parent shell reads the stop as an exit), so a chain is replaced by killing
it between arms and starting the next script on the same allocation.*

*Pulled again 2026-11-07 ~16:25 (charm 1f0db0b17: ShedLB copies for locality, the declared load pooled per device,
LBClearLoads at the timed steps; LCI b7c289ae and reconverse 038893f unchanged) over the local changes (stash, ff-pull,
pop: no conflicts); the runtime rebuilt in place (`make -j16`, rebuild_1107b.log), apps/moe rebuilt from clean objects
(build_moe_1107b.sh: the chunked slab and ExpertMap included, so the OLMoE rows run from apps/moe at -r 7.5). The two-node
comparison re-run at one PE per GCD on job 5879259 (frontier05256-05257, 16:33-17:0x): read_path2.sh (tags rp2_*,
logs/moe/read_path2.txt, logs/ref/results_rp2.txt), rp2_table.sh (the tables beside the morning's), ss.sh (a mean over
any step range: the protocol's 23-43 window now holds the landing step of eight-copy rounds). Verdict in the README
("Measured 2026-11-07, evening"): Mixtral serving 111.8 -> 57.4 after the round (one copy of every expert on the other
host, 2.1x the reference), SmartMoE serving's 31 locality copies buy nothing on top of the hot expert's three, training
unchanged, OLMoE serving -20%. Open: the exit segfault on a copy's PE in 4 of 12 ShedLB rows (upstream's known item),
the 5.7 s landing of the eight copies in the MoE-only Mixtral serving row (216 ms in the block row), the reference EPLB arm hung a fourth time (rc=124). Job released 17:00.*

*Four PEs per GCD, 2026-11-07 15:57-16:20 (job 5878615), then dropped.* `moe_run.sh PPN=4` (`+ppn 4`, the wrapper pins
four cores of the GCD's region), `lbcost.frontier.chan.ppn4.conf` (lbcalib `+ppn 4`), `apps/moe3` (the example with
`ExpertMap`: experts blocked over the GPUs as at one PE, since the default map put SmartMoE on one host and Mixtral four
to a GCD past the 16384 MB ceiling), scripts `ppn4*.sh`, `p4_table.sh`, rows in `logs/moe/ppn4.txt`. Verdict in the
README ("Measured 2026-11-07, later"): a wash on this machine, the balancer's host-time guard flips at four PEs (the
expert's entry-method wall time fills its own PE's interval; a group is abandoned when two copies fail the cap although
three or four would pass), and `+lci_ndevices 4` takes OLMoE from 124 to 96 ms a step (four PEs on one LCI device: 150 M
send-lock retries). The user dropped the plan at 16:25; the queued follow-up job (5879192, `ppn4e.sh`) was cancelled
unrun. Still uncommitted: `moe.C`/`moe.ci` (`ExpertMap`; harmless at one PE, the controls reproduce the morning's rows).

This is the job cancelled unrun on Delta (`/scratch/mzu/bhosale/moe_runs/v4b_fix.sbatch`), the one the
kernel-bound fix of 2026-11-02 still owes (the README's "A kernel bound, found and removed").

- **Part 0.** `replica_test +p4 +balancer ShedLB +LBDebug 1` (the flags its Makefile runs it with; without the
  balancer it aborts "expected the balancer to replicate worker 0"). It prints no PASS line: it passes when it
  exits 0 after `worker 0 cloned at step 10` and `retire decision at step 40`, and aborts otherwise.
- **Part 1, the kill criterion** (the last step's `out2 w2` checksums must be one set per shape):

  | Tag | Arguments |
  |---|---|
  | sm_plain | `$SM -i 40 -u 3 -B -C 4 -E $TS -R 1 -f 99999` (the reference) |
  | sm_X | `... -R 2 -f 99999 -X` |
  | sm_R4 | `... -R 4 -f 10 -b 10 -a` |
  | sm_R3 | `... -R 3 -f 10 -b 10 -a` |
  | mix_plain | `$MIX -i 40 -u 3 -B -C 4 -E $TM -R 1 -f 99999` (the reference) |
  | mix_X | `... -R 2 -f 99999 -X` |

  A difference stops the job: on the A40s the pre-fix binary differed from step 24, the first step with
  three copies, in copy 0 alone.
- **Part 2, the rows measured before the fix.** For SmartMoE and Mixtral: `-w` in training and serving,
  plain and ShedLB `-R 2`; for all three routings: the MoE layer alone in training and serving, plain and
  ShedLB `-R 2`; for SmartMoE also ShedLB `-R 4`. About 27 arms, 15 minutes on the A40s. These become the
  first rows of a "Measured on Frontier" section of the README, beside the A40 rows they re-measure.
- **Part 3.** The matrix (`mat8.sh` in the rescued scratchpads: example plain and ShedLB, the reference
  static, EPLB-style packing and one copy, Megatron-Core with and without a capacity factor, FastMoE and
  FasterMoE, training and serving, bands of three) once the baselines run on ROCm.

## Open questions, in order

1. The ladder above: checksums at three and four copies on eight GCDs, and the `-w` rows.
2. The memory ledger refusing the third copy of a default-shape expert on a 16 GB pool: what it advertises
   per node and why, now against 64 GB GCDs.
3. The cost model has no per-message latency: on eight A40s SmartMoE with `-R 4` ran slower than with
   `-R 2` (the copies' exchanges dominate at 8 MB weights), and the start-up probe sends one 128 MB partial;
   a small-message probe and a latency term beside the bytes term in the declared cost.
4. The idle-round path (`LastRoundDecisions == 0`, the dispatchers keep the lookahead through the round)
   has never been exercised: every OLMoE round made two or three decisions. `+LBShedEps 1` on OLMoE makes
   the threshold twice the mean, so no bin sheds and the round decides nothing; the step time should then
   equal the plain arm's.
5. The Kokkos variant against the four-partition tree: `moe.kokkos` has no add kernel and may still be the
   two-partition scheme; its `-X` and `-R 3` checksums against `moe`'s say so.
6. Copies in serving: the declared exchange cost is zero when serving, so ShedLB should copy the Mixtral
   expert in serving (a 10% gain on the A40s before the critical-path rule); verify the copy is made.
7. Platform items from `HIP_PORT.md`: the vmm pool cannot be NIC-registered on ROCr (dma-buf export),
   `hapi_memory_daemon` is CUDA-only, `tests/util/streamorder` and `benchmarks/charm++/cuda/gpudirect/*`
   spell CUDA names, the rocprofiler pool-release patch belongs upstream, the channel transport's remaining
   hold under the balancer.
8. FasterMoE's bf16 numerics (NaN from step 3 to 8; its harness, documented in `ref/README.md`), A40-only
   until FastMoE builds on ROCm.
9. Done 2026-11-04: the fused attention for the HIP build is aotriton (the paragraph under the ladder), and
   `ref_wrap.sh` logs the RCCL transport by default (correction 1). Left: the fp32 path is unfused on both
   vendors by design.
10. The verb across hosts as a machine property: on Slingshot the owner reads, on InfiniBand the sender
    would write; today that is `CHARM_CHAN_WRITE=1` or not. A startup probe (one put and one get of a few MB
    between the first two hosts) or a machine profile should choose it. Measured on two Frontier nodes
    2026-11-07: a wash (24 GB/s either verb into device memory, 23.3 each way with both hosts sending), so
    the probe would choose nothing there; the A100 asymmetry is the NIC's, not Slingshot's as such.
11. ShedLB's host cap (`max(T, hostMean * 1.1)`) refuses placements relative to the hosts' mean host time; on
    the A40s OLMoE is host-bound (the hosts' mean 2.4x the device target) and most placements are refused
    there. The refusal totals and the closest refused PE are printed now; the rule is to be revisited with
    those numbers.
12. One serving run on Delta (gpua011 and gpua040, 2026-11-06) placed all three copies on one host, 22.7 ms
    against 6.2; never reproduced once the placement diagnostics existed. If a Frontier ShedLB row is far off
    its band, read the placement lines before anything else.
13. The receive slab's fixed per-expert segments against the registration caps (2048 MB an arena, 16384 MB a
    process on cxi, both re-confirmed 2026-11-07): OLMoE on 16 GCDs needs the chunked slab and the trace's own
    capacity (`-r 7.5`). Sizing each expert's segment from the trace (or the hot set) instead of one `cap_src`
    for all would put the auto capacity back; the per-process cap itself is unexplained (ADDR_OUT_OF_RANGE past
    16 GB registered).
14. OLMoE on the read path: 124 to 132 ms a step against the 159 to 163 of the write path, the same on either
    output path, so the gain is the read protocol's or LCI's packet format's (b7c289ae); `CHARM_CHAN_WRITE=1`
    on the same nodes separates them. The baselines still lead 1.7x to 2.6x there (item 3 of the two-node notes).

## Delta artefacts and where they are

- `/scratch/mzu/bhosale/moe_runs/`: the run scripts and console logs of 2026-11-01 and 11-02
  (`v4b_fix.sbatch`, `v2a.sh`, `cmp_dump.py`, `mat8_table.py`), and `scratchpad_dt-login01/` and
  `scratchpad_dt-login04/` (166 MB each, copied on 2026-11-02 from the node-local session scratchpads:
  `mat8.sh`, `bands_table.py`, `fm8*.sh`, `alloc_*.sh`, `perf*.sh`, `trace1.sh`, `build_fastmoe.sh`,
  `fastmoe_src/`, `smartmoe_src/`). Delta's scratch is purged; copy what is needed to Frontier.
- `examples/charm++/cuda/gpudirect/moe/ranklogs/` in the Delta clone: the per-rank logs of every arm in
  the README; `traces/` beside it (above, and on GitHub as a release).
- The checkpoint cache `/scratch/mzu/bhosale/hf` (100 GB; ungated downloads, so re-downloadable).
- The agent's memory, `/u/bhosale/.claude/projects/-u-bhosale-charm-reconverse/memory/` (facts, rules,
  the measured history). Claude Code keys it by the clone's path, so on Frontier copy it to
  `~/.claude/projects/<clone path with / written as ->/memory/`, and link the practitioner notes as the
  skill (`AGENTS.md`).

## Working rules that carry over

- Batch only, at most one hour and two nodes; the sbatch script runs the work and the job ends with it;
  nothing pending at the end of a turn; `squeue -u $USER` first; one runner per allocation; never release
  an allocation with open work.
- Read the run's own log in the call that launches it; steady state only (steps 23-43); results in
  tables, arm by metric, in the README and in chat; a repeat needs a stated purpose (three for a band).
- A hang is a bug to root-cause (stacks; `wrun.sh`'s catcher), never run past; a mitigation is not a fix.
- Every run: `CHARM_LB_HELPER=near`, `+gpushm +gpupool`, `ulimit -c 0`; no new environment variables for
  one-off experiments; no balancer flags (the runtime self-configures), `+LBShedEps` only for item 4.
- One job in one partition, the partition the user named; never rebuild a binary a running job uses; a
  baseline row is written only after its log names what ran (module, transport, attention kernels) and
  its serving step is no slower than its training forward (the section above).
