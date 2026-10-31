d2dtest -- acceptance test for GPU-direct device-to-device messaging
===================================================================
GPU migration plan stage 9.2.

One self-checking program. Every check is a CkEnforce, so a failure aborts with
a file and line; success prints exactly one line:

    d2dtest PASS: <PEs> PEs, <procs> processes, <hosts> physical nodes, <n> doubles verified, <t> s

It is meant to be the thing a reviewer runs on their own GPUs. Per the runtime
validation protocol (#3950) CI for this series is build-only; this test is what
makes "reviewers run on GPU types of their choice until satisfied" concrete,
and it is deliberately backend-neutral so an NVIDIA run and an AMD run are
comparable verdicts on the same source.

What it checks
--------------
  1. All three transfer modes, and that the ones the job layout makes reachable
     were actually taken. findTransferModeDevice() picks MEMCPY (same process),
     IPC (same physical node, different process) or RDMA (different physical
     nodes), and each is a separate path through CkRdmaDeviceIssueRgets and
     CkRdmaDeviceOnSender. Senders tally the mode they used; Main checks the
     tally against CmiNumNodes() and CmiNumPhysicalNodes(), so a run that
     silently collapsed onto one path fails rather than passing.

  2. Payload correctness against the *sender's* pattern. Every buffer is filled
     from the sender's index, the iteration number and which buffer it is, and
     refilled every iteration. A receiver that gets a stale buffer, its own
     buffer, or its neighbour's other buffer sees a mismatch. The values are
     small integers scaled by powers of ten, so the comparison is exact.

  3. Two device buffers in one entry method, so the numops > 1 bookkeeping in
     DeviceRdmaInfo -- the n_ops/counter pair that decides when the real entry
     method finally runs -- is driven rather than the degenerate single-buffer
     case.

  4. Source callbacks: one per CkDeviceBuffer sent, counted and checked exactly.
     This is what tells a sender its buffer is reusable, and it is delivered
     from a different place in each of the three modes.

  5. Quiescence bracketing. Iterations are separated by CkWaitQD(), so the
     QdCreate/QdProcess pairs the D2D path adds -- including the ones around
     loopback_bridge, which bounces an inter-node completion to the destination
     PE -- must balance. Over-count and QD never fires (hang); under-count and
     QD fires before the data lands, which check 2 then catches.

  6. Chare arrays, groups and nodegroups, which reach the device path through
     different proxy machinery. The nodegroup case in particular only says
     anything on a run with more than one PE per process: with one PE per
     process, node and PE numbering coincide and a whole class of confusion
     between the two is invisible.

Building
--------
  make GPU=hip  CHARM_DIR=<build>          # AMD/ROCm; HIP_ARCH defaults to gfx90a
  make GPU=cuda CHARM_DIR=<build>          # NVIDIA;   CUDA_ARCH defaults to sm_80

CHARM_DIR defaults to ../../../.. , which is right when the directory is
reached through the build tree.

Running
-------
Needs at least 2 PEs. What gets covered depends on the shape of the job:

  1 process                     MEMCPY only
  >1 process, 1 physical node   MEMCPY + IPC
  >1 physical node              MEMCPY + IPC + RDMA

Use more than one PE per process (see check 6). On Frontier, two nodes:

  srun -N2 -n4 -c28 --gpus-per-node=8 ./d2dtest +pe 8

and on a single node add --network=single_node_vni. Flags: -s doubles per
buffer, -i iterations, -e ring elements, -v verbose.

Known coverage gaps
-------------------
  - Migration. CkRdmaDeviceIssueRgets aborts if a device buffer arrives at a
    process other than the one the sender addressed, which is what happens when
    an object migrates without CMK_GLOBAL_LOCATION_UPDATE. Nothing here
    migrates, so that path is not exercised.
  - CkDevicePersistent (the Direct/persistent device API) is not covered here;
    it has no inter-node implementation yet (both get() and put() abort for
    CkNcpyModeDevice::RDMA).
