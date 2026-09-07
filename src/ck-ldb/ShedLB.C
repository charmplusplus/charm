/**
 * \addtogroup CkLdb
*/
/*@{*/

/**
 * ShedLB: threshold refinement minimising migrated bytes. See ShedLB.h for
 * what it guarantees and why it is not another repacking strategy.
*/

#include "charm++.h"
#include "ShedLB.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

extern int quietModeRequested;

CreateLBFunc_Def(ShedLB, "Threshold refinement that minimises migrated bytes")

ShedLB::ShedLB(const CkLBOptions& opt) : CBase_ShedLB(opt)
{
  lbname = "ShedLB";
  if ((CkMyPe() == 0) && !quietModeRequested)
    CkPrintf("CharmLB> ShedLB created.\n");
}

ShedLB::ShedLB(CkMigrateMessage* m) : CBase_ShedLB(m) { lbname = "ShedLB"; }

namespace
{
// What one migration of this object stages. Unknown sizes fall back to 1, so
// the objective degenerates to minimising the NUMBER of moves rather than
// silently treating every object as free.
inline double objBytes(const LDObjData& od)
{
#if CMK_CUDA
  const double b = (double)od.gpuPupSize;
  return b > 0.0 ? b : 1.0;
#else
  (void)od;
  return 1.0;
#endif
}
}  // namespace

void ShedLB::work(LDStats* stats)
{
  const int n_pes = stats->nprocs();
  const int n_objs = (int)stats->objData.size();
  if (n_pes <= 0 || n_objs <= 0) return;

  // ---- which resource is being balanced --------------------------------
  // Device time when there is any, because a migratable object's cost is
  // then its kernels and the PEs sharing a GPU are one bin. Otherwise host
  // time, one bin per PE.
  bool useGpu = false;
#if CMK_CUDA
  {
    double gpuTotal = 0.0;
    for (int i = 0; i < n_objs; i++)
      if (stats->objData[i].migratable) gpuTotal += stats->objData[i].gpuTime;
    useGpu = gpuTotal > 0.0;
  }
#endif

  // ---- bins ------------------------------------------------------------
  std::vector<Bin> bins;
  std::vector<int> peToBin(n_pes, -1);
  {
    std::unordered_map<uint64_t, int> devToBin;
    for (int pe = 0; pe < n_pes; pe++)
    {
      if (!stats->procs[pe].available) continue;
      int b;
      if (useGpu)
      {
#if CMK_CUDA
        const uint64_t dev = stats->procs[pe].gpu_device_id;
        auto it = devToBin.find(dev);
        if (it == devToBin.end()) { b = (int)bins.size(); devToBin[dev] = b; bins.push_back(Bin()); }
        else b = it->second;
#else
        b = (int)bins.size(); bins.push_back(Bin());
#endif
      }
      else
      {
        b = (int)bins.size();
        bins.push_back(Bin());
      }
      peToBin[pe] = b;
      bins[b].pes.push_back(pe);
    }
  }
  const int nbins = (int)bins.size();
  if (nbins <= 1) return;   // nothing to balance across

  // ---- loads, and the candidates each bin could shed -------------------
  // Non-migratable objects are background: they count toward a bin's load and
  // can never be shed, which is what keeps the threshold honest.
  std::vector<double> peCpu(n_pes, 0.0);
  std::vector<Cand> cands;
  cands.reserve(n_objs);
  double total = 0.0;
  for (int i = 0; i < n_objs; i++)
  {
    const LDObjData& od = stats->objData[i];
    const int pe = stats->from_proc[i];
    if (pe < 0 || pe >= n_pes || peToBin[pe] < 0) continue;
    const int b = peToBin[pe];
#if CMK_CUDA
    const double load = useGpu ? od.gpuTime : od.wallTime;
#else
    const double load = od.wallTime;
#endif
    bins[b].load += load;
    total += load;
    peCpu[pe] += od.wallTime;
    if (od.migratable) cands.push_back(Cand{i, load, objBytes(od), b});
  }
  if (total <= 0.0) return;

  const double avg = total / nbins;
  double eps = _lb_args.shedEps();
  if (eps < 0.0) eps = 0.0;
  const double T = avg * (1.0 + eps);

  double maxLoad = 0.0;
  for (int b = 0; b < nbins; b++) maxLoad = std::max(maxLoad, bins[b].load);
  if (maxLoad <= T)
  {
    if (_lb_args.debug() > 0 && CkMyPe() == cur_ld_balancer)
      CkPrintf("[ShedLB] max/avg %.3f already within 1+eps (%.3f): no moves\n",
               maxLoad / avg, 1.0 + eps);
    return;
  }

  // ---- shed and place -------------------------------------------------
  // One transfer at a time, always off the fullest bin. A candidate is only
  // worth moving if some other bin can take it and stay under T: an object
  // larger than a bin's whole share cannot be placed anywhere, so moving it
  // buys nothing and costs its bytes. Deciding shed and placement together is
  // what keeps that case from silently consuming the bin's excess -- an
  // earlier version debited the excess for such an object, put it straight
  // back where it came from, and left the bin overloaded having moved nothing.
  //
  // Among the objects that can be placed, prefer the ones that fit inside the
  // excess (so the bin is not sent below its share for nothing), and among
  // those the highest load per byte: the most excess cleared per byte staged.
  // If none fits inside the excess, take the smallest load, which overshoots
  // least. That last case is the granularity term in the bound.
  std::vector<char> moved(cands.size(), 0);
  int moves = 0;
  double shedBytes = 0.0;

  for (;;)
  {
    int h = 0;
    for (int b = 1; b < nbins; b++)
      if (bins[b].load > bins[h].load) h = b;
    if (bins[h].load <= T) break;
    const double excess = bins[h].load - T;

    int pick = -1, pickDst = -1;
    double pickScore = -1.0, pickSmallest = std::numeric_limits<double>::max();
    for (size_t k = 0; k < cands.size(); k++)
    {
      if (moved[k]) continue;
      const Cand& c = cands[k];
      if (c.bin != h) continue;

      // Best fit among the other bins: fullest that still clears T.
      int dst = -1;
      double bestAfter = -1.0;
      for (int b = 0; b < nbins; b++)
      {
        if (b == h) continue;
        const double after = bins[b].load + c.load;
        if (after <= T && after > bestAfter) { bestAfter = after; dst = b; }
      }
      if (dst < 0) continue;   // nowhere to put it: not a candidate at all

      if (c.load <= excess)
      {
        const double score = c.load / c.bytes;
        if (pickScore < 0.0 || score > pickScore) { pickScore = score; pick = (int)k; pickDst = dst; }
      }
      else if (pickScore < 0.0 && c.load < pickSmallest)
      {
        pickSmallest = c.load;   // only if nothing fits inside the excess
        pick = (int)k;
        pickDst = dst;
      }
    }
    if (pick < 0) break;   // this bin cannot be improved; neither can any fuller one

    Cand& c = cands[pick];
    moved[pick] = 1;
    bins[h].load -= c.load;
    bins[pickDst].load += c.load;
    shedBytes += c.bytes;

    // Within the destination bin, the least host-loaded PE. Host time decides
    // only this, never which bin the object lands on.
    const int oldPe = stats->from_proc[c.idx];
    int dstPe = bins[pickDst].pes[0];
    for (int pe : bins[pickDst].pes)
      if (peCpu[pe] < peCpu[dstPe]) dstPe = pe;
    if (dstPe != oldPe)
    {
      stats->to_proc[c.idx] = dstPe;
      peCpu[dstPe] += stats->objData[c.idx].wallTime;
      peCpu[oldPe] -= stats->objData[c.idx].wallTime;
      moves++;
    }
  }

  if (_lb_args.debug() > 0 && CkMyPe() == cur_ld_balancer)
  {
    double after = 0.0;
    for (int b = 0; b < nbins; b++) after = std::max(after, bins[b].load);
    CkPrintf("[ShedLB] %d bins on %s time: max/avg %.3f -> %.3f (target %.3f), "
             "%d of %d objects moved, %.1f MB staged\n",
             nbins, useGpu ? "device" : "host", maxLoad / avg, after / avg,
             1.0 + eps, moves, (int)cands.size(), shedBytes / 1e6);
  }
}

#include "ShedLB.def.h"

/*@}*/
