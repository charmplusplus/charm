#ifndef LB_LOAD_DIM_H
#define LB_LOAD_DIM_H

// Which load dimension binds: host time or device occupancy -- or both.
//
// With one GPU per process and ppn PEs driving it, an object's two loads live
// at different places in the machine. Host time is a per-PE resource and a
// group of PEs sharing a device has ppn of it; device occupancy is one
// resource per group. A group's step is over when both are done:
//
//   T_q = max( max_{p in q} sum_p h , sum_q g )
//
// so the job's step cannot be shorter than either dimension's total spread
// evenly over the units it aggregates at -- or than its largest single
// object, which cannot be split:
//
//   T_h = max( sum_h / P , h_max )      T_g = max( sum_g / G , g_max )
//
// The dimension with the larger bound binds. The other one has slack in the
// ratio alpha_k = T_k / max(T_h, T_g): at alpha 0.2 it can be five times out
// of balance before it decides the step, and a balancer that spends moves
// equalising it is optimising the wrong thing. Measured 11 Sep 2026: sph2d at
// 12.5M particles is device-bound with alpha_h = 0.21, pic2d at 16 objects per
// node is host-bound with alpha_g = 0.24; the flag-selected dimension was
// right for one and wrong for the other.
//
// When neither dimension has much slack -- both alphas at or above
// +LBLoadVectorAbove -- balancing one of them leaves the step set by the
// other wherever the other's load is concentrated, and the balancers switch
// to their two-dimensional forms: DiffusionLB diffuses each node's step time
// T_q itself and selects objects by what they cost the receiver in BOTH
// dimensions (LB_MODE_STEP), MetisLB carries the second dimension as an extra
// partition constraint at the tolerance its slack allows. Measured on the
// lbsim cross case with one-dimensional balancing: the step stops improving
// as soon as the other dimension's hot region holds the maximum.
//
// Every strategy chooses here, from the same arithmetic, so a central
// partition and a distributed diffusion agree on what they are equalising.
// The flags remain as overrides: +LBDiffusionGpuDim forces device,
// +LBDiffusionHostDim forces host, neither lets the measurement decide.
// Header-only, CUDA-independent: without CMK_CUDA there is no device
// dimension and host always binds.

#include <algorithm>
#include <cstdint>
#include <unordered_set>

#include "BaseLB.h"
#include "LBManager.h"
#include "lbdb.h"

struct LBCriticality
{
  double sumHost = 0.0, maxHost = 0.0;
  double sumDev = 0.0, maxDev = 0.0;
  int pes = 0;   // P: PEs the host dimension is spread over
  int gpus = 0;  // G: devices the device dimension is spread over
  // Wall time of the balancing interval, the longest any PE saw. What the
  // two measured loads are held against: alpha says which of them is the
  // larger relative to its resources, and nothing about whether either is
  // what holds the step open. A dimension's share of the interval says
  // that. When both shares are small the step is spent elsewhere --
  // communication, waiting, launch latency -- and no choice of dimension
  // can move it much. Zero when unknown (the offline simulator).
  double period = 0.0;

  void addObject(double h, double g)
  {
    sumHost += h;
    if (h > maxHost) maxHost = h;
    sumDev += g;
    if (g > maxDev) maxDev = g;
  }
  // Fold another node's (or PE's) totals in, for a distributed reduction.
  void merge(const LBCriticality& o)
  {
    sumHost += o.sumHost;
    maxHost = std::max(maxHost, o.maxHost);
    sumDev += o.sumDev;
    maxDev = std::max(maxDev, o.maxDev);
    pes += o.pes;
    gpus += o.gpus;
    period = std::max(period, o.period);
  }
  // Each dimension's even-spread bound as a share of the interval.
  double hostShare() const { return period > 0.0 ? boundHost() / period : 0.0; }
  double devShare() const { return period > 0.0 ? boundDev() / period : 0.0; }
  // Neither dimension accounts for as much as `below` of the interval.
  bool unexplained(double below) const
  {
    return period > 0.0 && hostShare() < below && devShare() < below;
  }

  double boundHost() const { return pes > 0 ? std::max(sumHost / pes, maxHost) : 0.0; }
  double boundDev() const { return gpus > 0 ? std::max(sumDev / gpus, maxDev) : 0.0; }
  double lowerBound() const { return std::max(boundHost(), boundDev()); }
  double alphaHost() const
  {
    const double lb = lowerBound();
    return lb > 0.0 ? boundHost() / lb : 0.0;
  }
  double alphaDev() const
  {
    const double lb = lowerBound();
    return lb > 0.0 ? boundDev() / lb : 0.0;
  }
  // Ties go to the host: a job with no device load at all has both bounds at
  // zero and must keep balancing host time, as it always did.
  bool deviceBinds() const { return boundDev() > boundHost(); }
  // Neither dimension has enough slack to be ignored.
  bool comparable(double above) const
  {
    return boundHost() > 0.0 && boundDev() > 0.0 &&
           std::min(alphaHost(), alphaDev()) >= above;
  }
};

enum LBLoadDimOverride
{
  LB_DIM_AUTO = 0,
  LB_DIM_HOST = 1,
  LB_DIM_DEVICE = 2
};

// What a balancer balances this step. The first two are one dimension; the
// third is the step time itself, both dimensions at once.
enum LBLoadMode
{
  LB_MODE_HOST = 0,
  LB_MODE_DEVICE = 1,
  LB_MODE_STEP = 2
};

static inline LBLoadDimOverride lbLoadDimOverride()
{
#if CMK_CUDA
  if (_lb_args.diffusionGpuDim()) return LB_DIM_DEVICE;
  if (_lb_args.diffusionHostDim()) return LB_DIM_HOST;
  return LB_DIM_AUTO;
#else
  return LB_DIM_HOST;
#endif
}

// The alpha at or above which both dimensions count (+LBLoadVectorAbove).
static inline double lbLoadVectorAbove() { return _lb_args.loadVectorAbove(); }

// The one-dimensional decision: an override wins, otherwise whichever
// dimension binds.
static inline bool lbResolveDeviceDim(const LBCriticality& c)
{
  switch (lbLoadDimOverride())
  {
    case LB_DIM_DEVICE: return true;
    case LB_DIM_HOST: return false;
    default: return c.deviceBinds();
  }
}

// The full decision: an override forces one dimension; otherwise both when
// they are comparable, else the one that binds.
static inline LBLoadMode lbResolveLoadMode(const LBCriticality& c)
{
  switch (lbLoadDimOverride())
  {
    case LB_DIM_DEVICE: return LB_MODE_DEVICE;
    case LB_DIM_HOST: return LB_MODE_HOST;
    default:
      if (c.comparable(lbLoadVectorAbove())) return LB_MODE_STEP;
      return c.deviceBinds() ? LB_MODE_DEVICE : LB_MODE_HOST;
  }
}

static inline const char* lbLoadDimName(bool device) { return device ? "device" : "host"; }
static inline const char* lbLoadModeName(int mode)
{
  return mode == LB_MODE_STEP ? "step (both)" : mode == LB_MODE_DEVICE ? "device" : "host";
}

// Below this share of the interval a dimension is not what the step waits
// on. Half: at less than half, even a perfect balance of that dimension
// leaves the majority of the step to whatever else the time went to.
static const double kLBUnexplainedBelow = 0.5;

// One line of diagnosis for a strategy's debug output: the shares, and a
// warning when neither dimension explains the interval.
static inline void lbPrintExplained(const char* who, const LBCriticality& c)
{
  if (c.period <= 0.0) return;
  // A load larger than the interval it is held against means the two do not
  // cover the same span: an application declaring a per-step load against a
  // multi-step interval, or a stats window that started after the interval
  // did (a central balancer's first step, before any ClearLoads). The shares
  // then say nothing, and the verdict is withheld.
  const bool spans = c.hostShare() > 1.5 || c.devShare() > 1.5;
  CkPrintf("%s measured loads explain host %.0f%%, device %.0f%% of the %.3f s interval%s\n",
           who, 100.0 * c.hostShare(), 100.0 * c.devShare(), c.period,
           spans ? " -- loads exceed the interval: they do not cover the same span (declared "
                   "per-step loads, or a stats window shorter than the interval); shares "
                   "not meaningful"
           : c.unexplained(kLBUnexplainedBelow)
               ? " -- NEITHER dimension holds the step open; balancing either cannot move it much"
               : "");
}

// The central form: everything a central strategy needs is in the stats it
// was handed. P counts the available PEs, G the distinct devices among them.
static inline LBCriticality lbCriticalityOf(const BaseLB::LDStats* stats)
{
  LBCriticality c;
  std::unordered_set<uint64_t> devices;
  for (int pe = 0; pe < stats->nprocs(); pe++)
  {
    if (!stats->procs[pe].available) continue;
    c.pes++;
    devices.insert(stats->procs[pe].gpu_device_id);
    c.period = std::max(c.period, (double)stats->procs[pe].total_walltime);
  }
  c.gpus = (int)devices.size();
  for (size_t i = 0; i < stats->objData.size(); i++)
  {
    const LDObjData& o = stats->objData[i];
#if CMK_CUDA
    c.addObject(o.wallTime, o.gpuTime);
#else
    c.addObject(o.wallTime, 0.0);
#endif
  }
  return c;
}

#endif
