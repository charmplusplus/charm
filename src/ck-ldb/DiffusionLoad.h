#ifndef _DIFFUSION_LOAD_H
#define _DIFFUSION_LOAD_H

// The load accessors DiffusionLB decides on. In their own header, with no
// dependency on the balancer's chare, so that the object-selection metric
// (DiffusionMetric.h), the flow arithmetic (DiffusionFlow.h) and the offline
// simulator (tests/charm++/load_balancing/lbdriver/lbsim.C) can all use them
// without pulling in DiffusionLB.h -- which defines a global and includes a
// heap implementation, and so cannot be included from a second translation
// unit that also links the module.

#include "LBLoadDim.h"
#include "LBManager.h"
#include "lbdb.h"

// DiffusionLB balances two different resources at its two levels, and they are not
// interchangeable:
//
//   Across nodes. Under one process per device a node IS a GPU, so the scarce
//   resource is device occupancy and the quantity to equalise is the sum of GPU
//   time over the node's objects -- when the device is what binds. When the host
//   binds, it is the node's host time. When neither has much slack, it is the
//   node's step time itself: the larger of its host time per PE and its device
//   time (LB_MODE_STEP).
//
//   Within a node. The PEs of a process SHARE that device, so moving a chare from
//   one PE to another does not relieve the GPU by a microsecond -- the kernel still
//   runs on the same card. Only host-side work relocates. The intra-node heap must
//   therefore balance CPU time alone; charging it GPU time would have it believe it
//   is rebalancing something it structurally cannot.
//
// Hence the accessors. diffusionObjLoad() is the object's share of the diffused
// quantity (what crosses node boundaries); diffusionObjCpuLoad() is always host
// time (what moves between PEs inside a node).
//
// Note deliberately NOT max(cpu, gpu): summing per-object maxima over-counts every
// object whose two timelines overlap. A node's step time is
// max(sum of gpuTime, max over PEs of sum of wallTime) -- aggregate first, then take
// the max, never the other way round.

// What the rounds diffuse this step, chosen once per step by which dimension
// binds job-wide (LBLoadDim.h): every node reports its totals to PE 0, PE 0
// resolves, and the verdict is broadcast before the rounds start, so every
// node diffuses the same thing -- nodes disagreeing about which resource they
// are equalising would diffuse incoherently. +LBDiffusionGpuDim and
// +LBDiffusionHostDim override the measurement. Until a verdict has arrived
// (the first step's stats assembly, or a build without the balancer) an
// override applies if given and the host dimension otherwise, which is what
// this balancer diffused before the choice was measured.
//
// Defined in DiffusionLB.C, so the offline simulator gets it by linking the
// module. -1 = no verdict yet, else an LBLoadMode.
extern int diffusionLoadDimDevice;

// In LB_MODE_STEP the diffused quantity is the node's step time, and an
// object's share of it depends on which of the node's two terms is the
// larger: g(o) on a device-bound node, h(o)/ppn on a host-bound one. These
// are the node's side and its PE count, set by the node's rank-0 PE with the
// verdict. Process-wide, which is exact for one node per process and only
// approximate under the CHARM_DIFFUSION_NODE_SIZE test hook, where several
// logical nodes share a process; the simulator sets them per virtual node
// before each node's decisions.
extern int diffusionNodeDeviceBound;
extern int diffusionPpn;

static inline int diffusionLoadMode()
{
  switch (lbLoadDimOverride())
  {
    case LB_DIM_DEVICE: return LB_MODE_DEVICE;
    case LB_DIM_HOST: return LB_MODE_HOST;
    default: return diffusionLoadDimDevice < 0 ? LB_MODE_HOST : diffusionLoadDimDevice;
  }
}
static inline bool diffusionDeviceDim() { return diffusionLoadMode() == LB_MODE_DEVICE; }
static inline bool diffusionStepMode() { return diffusionLoadMode() == LB_MODE_STEP; }

// The object's load in the group dimension -- device time, or driver time
// when the launch term is the larger bound (LBLoadDim.h). Named for what it
// was before the launch term existed; every consumer of the node's "device"
// total means this.
static inline double diffusionObjGpuLoad(const LDObjData& o) { return lbObjGroupLoad(o); }

// The object's share of the diffused quantity, in the units the rounds
// planned in. Under one dimension that is the object's load in it; under
// LB_MODE_STEP it is what the object contributes to THIS node's binding term.
static inline double diffusionObjLoad(const LDObjData& o)
{
  switch (diffusionLoadMode())
  {
    case LB_MODE_DEVICE: return diffusionObjGpuLoad(o);
    case LB_MODE_STEP:
      return diffusionNodeDeviceBound ? diffusionObjGpuLoad(o)
                                      : o.wallTime / (diffusionPpn > 0 ? diffusionPpn : 1);
    default: return o.wallTime;
  }
}

// Host time, always. Used for per-PE totals and the within-node heap, which can only
// ever move host work between PEs that share a device.
static inline double diffusionObjCpuLoad(const LDObjData& o) { return o.wallTime; }

#endif
