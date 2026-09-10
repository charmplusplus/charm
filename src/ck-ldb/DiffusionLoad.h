#ifndef _DIFFUSION_LOAD_H
#define _DIFFUSION_LOAD_H

// The two load accessors DiffusionLB decides on. In their own header, with no
// dependency on the balancer's chare, so that the object-selection metric
// (DiffusionMetric.h), the flow arithmetic (DiffusionFlow.h) and the offline
// simulator (tests/charm++/load_balancing/lbdriver/lbsim.C) can all use them
// without pulling in DiffusionLB.h -- which defines a global and includes a
// heap implementation, and so cannot be included from a second translation
// unit that also links the module.

#include "LBManager.h"
#include "lbdb.h"

// DiffusionLB balances two different resources at its two levels, and they are not
// interchangeable:
//
//   Across nodes. Under one process per device a node IS a GPU, so the scarce
//   resource is device occupancy and the quantity to equalise is the sum of GPU
//   time over the node's objects. Selected with +LBDiffusionGpuDim.
//
//   Within a node. The PEs of a process SHARE that device, so moving a chare from
//   one PE to another does not relieve the GPU by a microsecond -- the kernel still
//   runs on the same card. Only host-side work relocates. The intra-node heap must
//   therefore balance CPU time alone; charging it GPU time would have it believe it
//   is rebalancing something it structurally cannot.
//
// Hence two accessors. diffusionObjLoad() is the diffused dimension (what crosses
// node boundaries); diffusionObjCpuLoad() is always host time (what moves between
// PEs inside a node).
//
// Note deliberately NOT max(cpu, gpu): summing per-object maxima over-counts every
// object whose two timelines overlap. A node's step time is
// max(sum of gpuTime, max over PEs of sum of wallTime) -- aggregate first, then take
// the max, never the other way round.

// The dimension diffused across nodes. Defaults to host time so that CPU-only
// workloads keep working; +LBDiffusionGpuDim switches it to device occupancy for
// GPU-bound runs. An automatic choice would have to be identical on every node --
// nodes disagreeing about which resource they are equalising would diffuse
// incoherently -- so it is an explicit flag rather than a local heuristic.
static inline double diffusionObjLoad(const LDObjData& o)
{
#if CMK_CUDA
  if (_lb_args.diffusionGpuDim()) return o.gpuTime;
#endif
  return o.wallTime;
}

// Host time, always. Used for per-PE totals and the within-node heap, which can only
// ever move host work between PEs that share a device.
static inline double diffusionObjCpuLoad(const LDObjData& o) { return o.wallTime; }

#endif
