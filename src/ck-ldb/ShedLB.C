#include <set>
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
#include "DiffusionCostModel.h"
#if CMK_GPU
#include "LBMemoryContract.h"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
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
#if CMK_GPU
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
#if CMK_GPU
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
#if CMK_GPU
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
  int nHosts = 0;
  {
    std::set<int> hostIds;
    for (int pe = 0; pe < n_pes; pe++)
      if (stats->procs[pe].available && peToBin[pe] >= 0) hostIds.insert(CmiPhysicalNodeID(pe));
    nHosts = (int)hostIds.size();
  }

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
#if CMK_GPU
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
  // The host is a second resource. An object's entry-method time runs on its
  // PE's host whatever its device load is (an expert's chunk launches and
  // output puts: 0.5 to 2.4 ms a step on the A100 nodes), and a PE's step is
  // bound by the larger of its device load and its host time. Packing ten
  // experts on one PE by device load alone gave it 10 ms of host time a
  // step and the serving step (job 23612682). So no move or copy may raise
  // a host above the device target, or above the hosts' own mean with the
  // same slack where that is higher.
  double hostMean = 0.0;
  {
    int np = 0;
    for (int pe = 0; pe < n_pes; pe++)
      if (stats->procs[pe].available && peToBin[pe] >= 0) { hostMean += peCpu[pe]; np++; }
    if (np > 0) hostMean /= np;
  }
  const double hostCap = std::max(T, hostMean * (1.0 + eps));
  int hostRefusals = 0;
  if (_lb_args.debug() > 0 && CkMyPe() == cur_ld_balancer)
  {
    // The two resources per bin as the strategy sees them: the bin's device
    // load and each of its PEs' host time (the sum of its objects' wall time
    // after the driver correction), for reading a layout of several PEs per
    // device against one.
    for (int b = 0; b < nbins; b++)
    {
      std::string hosts;
      for (int pe : bins[b].pes) { char buf[48]; snprintf(buf, sizeof buf, " pe%d %.3f", pe, peCpu[pe]); hosts += buf; }
      CkPrintf("[ShedLB] bin %d: device %.3f (target %.3f); host%s (cap %.3f)\n", b, bins[b].load, T, hosts.c_str(), hostCap);
    }
  }

  double maxLoad = 0.0;
  for (int b = 0; b < nbins; b++) maxLoad = std::max(maxLoad, bins[b].load);

  // Memory is a constraint here, not the objective: the bytes a move stages
  // are still what shedding minimises, and a destination is only a candidate
  // if it can hold what the object holds (the memory contract's I-final). The
  // two differ under the device pool -- a Compute stages its live prefix but
  // holds its capacity. Staging and IPC slots are the batch planner's.
#if CMK_GPU
  LBMemoryModel memModel;
  memModel.build(stats);
  MemoryLedger ledger;
  ledger.init(&memModel, 0.95, /*waveStaging=*/false);
  int memRefusals = 0;
#endif
  // Where an object placed in each bin would go: its least host-loaded PE.
  auto landingPe = [&](int b) {
    int dstPe = bins[b].pes[0];
    for (int pe : bins[b].pes)
      if (peCpu[pe] < peCpu[dstPe]) dstPe = pe;
    return dstPe;
  };
  std::vector<int> binPe(nbins);
  for (int b = 0; b < nbins; b++) binPe[b] = landingPe(b);

  // ---- multiplicity: replicable groups -------------------------------
  // An object above T cannot be placed anywhere, so for objects the
  // application declared replicable (LDObjData::replicaGroup) the balancer
  // changes how many copies exist before it places anything. A group's load
  // L_g is the sum over its copies; it needs k = ceil(L_g / T) copies, each
  // carrying L_g / k, up to the application's maximum. Copies are added one
  // at a time until the group has what it needs or a copy would not lower
  // the critical path, and one is taken back per step, the latter only when the remaining
  // copies would sit below the mean (T (1 - eps) ~ avg), so the two decisions
  // do not alternate on a load that hovers around T. A copy costs the
  // object's bytes at its destination like a move; nothing is staged for a
  // retirement and the bytes it frees are not spent in the same step.
  // Members of a group never share a bin; the source of a copy and a retiring
  // copy are left out of the shedding below.
  std::vector<char> pinned(cands.size(), 0);
  std::unordered_map<int, std::vector<int>> groupMembers;   // group -> cand indices
  for (size_t k = 0; k < cands.size(); k++)
  {
    const LDObjData& od = stats->objData[cands[k].idx];
    if (od.replicable()) groupMembers[od.replicaGroup].push_back((int)k);
  }
  auto groupHoldsBin = [&](int g, int b) {
    auto it = groupMembers.find(g);
    if (it == groupMembers.end()) return false;
    for (int k : it->second)
      if (cands[k].bin == b) return true;
    return false;
  };
  int clonesPlanned = 0, retiresPlanned = 0;
  static DiffusionCostConfig costCfg;
  static bool costCfgTried = false;
  if (!costCfgTried && !groupMembers.empty())
  {
    costCfgTried = true;
    costCfg.loadDefault(_lb_args.costConfig());
  }
  // The groups in descending load: the heaviest copies are placed first and
  // the lighter ones fill in behind them (longest processing time first),
  // which keeps two heavy copies off one bin. Planned by group id, Mixtral's
  // two heaviest experts shared a GPU at 0.98 against 0.50 on another
  // (two A100 nodes, job 23646171).
  std::vector<std::pair<double, int>> groupOrder;
  for (auto& gm : groupMembers)
  {
    double L = 0.0;
    for (int k : gm.second) L += cands[k].load;
    groupOrder.push_back(std::make_pair(-L, gm.first));
  }
  std::sort(groupOrder.begin(), groupOrder.end());
  for (const auto& go : groupOrder)
  {
    const int g = go.second;
    std::vector<int>& members = groupMembers[g];
    double L = 0.0;
    int maxCount = 1;
    for (int k : members)
    {
      L += cands[k].load;
      maxCount = std::max(maxCount, stats->objData[cands[k].idx].replicaMax);
    }
    const int count = (int)members.size();
    if (_lb_args.debug() > 0 && CkMyPe() == cur_ld_balancer)
      for (int k : members)
        CkPrintf("[ShedLB] group %d: copy %d of %d on PE %d, load %.4f (max %d)\n", g,
                 stats->objData[cands[k].idx].replicaRank, count, stats->from_proc[cands[k].idx],
                 cands[k].load, maxCount);
    // An idle split group still takes its copies: the split needs them before its demand arrives, not a round after.
    if (L <= 0.0 && stats->objData[cands[members[0]].idx].replicaParts >= 0) continue;
    int want = (int)std::ceil(L / T);
    if (want < 1) want = 1;
    // A copy's share is a block of the declared partitions, not 1/k: with P partitions and k copies the largest is
    // ceil(P/k)/P. Counts are tried up to the first whose largest share fits the target (2026-11-08, SmartMoE on
    // 16 dispatchers: ceil(L/T) = 7 copies leave blocks of 3, which do not lower the fullest bin, so the planner
    // stopped at 6 and the fullest GPU carried 1.5x the mean; 8 copies give every copy 2).
    {
      const LDObjData& odp = stats->objData[cands[members[0]].idx];
      if (odp.replicaParts > 0)
        for (int k = want; k <= maxCount; k++)
        {
          double big = 0.0;
          for (int r = 0; r < k; r++) big = std::max(big, odp.replicaShare(r, k));
          want = k;
          if (L * big <= T) break;
        }
    }
    if (want > maxCount) want = maxCount;
    // A group whose copies split every source's work between them each step (replicaParts < 0: the runtime's
    // CkReplicaSplit, from the step's demand and where the copies are) is not bound to the count that fits: the
    // split moves the group's work between the GPUs its copies are on, so each copy past the count that fits is
    // room for the step's demand to shift into. Two of them: replaying the SmartMoE trace on 16 GPUs
    // (2026-11-09, scratchpad sim3.py), ceil(L/T) + 2 copies placed at step 5 kept every step through 61 at 91% of
    // the mean or better (mean 93%, and 98% with rounds at 20 and 40), where ceil(L/T) + 1 fell to 46% and fixed
    // blocks of sources reached 87% at best even re-placed every step.
    const bool splitGroup = stats->objData[cands[members[0]].idx].replicaParts < 0;
    static const int splitExtra = getenv("CHARM_SHED_SPLIT_EXTRA") ? atoi(getenv("CHARM_SHED_SPLIT_EXTRA")) : 2;
    // CHARM_SHED_SPLIT_FLOOR=1: a split group's count is taken for at least the mean group's load, so a group idle
    // at this round has room for its share to come back before the next (2026-11-09, SmartMoE: experts idle at
    // step 5 carried 10-20% of the tokens by step 20 on two or three copies, steps 11-27 at 67-78% busy).
    static const bool splitFloor = getenv("CHARM_SHED_SPLIT_FLOOR") != NULL;
    const double Lfloor = (splitFloor && !groupMembers.empty()) ? std::max(L, total / (double)groupMembers.size()) : L;
    if (splitGroup) want = std::max(1, std::min(std::min(maxCount, nbins), (int)std::ceil(Lfloor / T - 1e-9) + splitExtra));
    // The count at which every copy of the group fits the target (above).
    const int fitK = want;
    // Copies for locality. An object that declared partitions and no exchange
    // (an expert in serving) is fed by every host, and a copy placed with its
    // block of partitions serves that host's tokens without the link; the
    // copy costs memory and nothing per step. One copy per host, then, when
    // the object can have that many, accepted below if it leaves the fullest
    // bin no worse than it is: two A100 nodes, 2026-11-07, a serving step of
    // 6.4 ms carried about 35 MB each way per host through the one NIC for
    // the tokens of the experts that had no copy, against 2.4 to 3.3 ms of
    // compute on the fullest GPU.
    const LDObjData& odg0 = stats->objData[cands[members[0]].idx];
    // The host that feeds the first block of partitions (rank 0's, under one
    // copy per host). An original elsewhere is moved there first (the pass
    // before the shed loop) and copied the round after: a copy beside it
    // would serve its own host and leave the original's block on the link.
    auto blockHost = [&](const LDObjData& od, int rank, int k) {
      const int P = od.replicaParts, lo = (P * rank) / k, hi = (P * (rank + 1)) / k;
      std::map<int, int> feeders;
      for (int pe = 0; pe < stats->nprocs(); pe++)
      {
        const int part = stats->procs[pe].partition;
        if (part >= lo && part < hi) feeders[CmiPhysicalNodeID(pe)]++;
      }
      int most = 0, host = -1;
      for (const auto& kv : feeders) if (kv.second > most) { most = kv.second; host = kv.first; }
      return host;
    };
    const bool exchangeFree = odg0.replicaParts > 0 && odg0.replicaExchangeBytes == 0 && nHosts > 1;
    const bool onBlock0Host = exchangeFree &&
        CmiPhysicalNodeID(stats->from_proc[cands[members[0]].idx]) == blockHost(odg0, 0, nHosts);
    // Not for a group that takes next to nothing: a copy of an idle expert keeps no tokens off the link and costs
    // a clone and its memory (2026-11-09, SmartMoE on 16 GCDs: the round at step 5 copied eight experts of load
    // 0.000-0.012 for locality).
    const bool localRule = exchangeFree && onBlock0Host && count < nHosts && maxCount >= nHosts && L > eps * avg;
    if (localRule && want < nHosts) want = nHosts;

    // The recurring cost of a copy, as the application declared it: what
    // one exchange with one peer adds to a copy per balancing interval, on
    // the same host or across hosts (the physical node: a Charm++ node is a
    // process). The measured loads leave it out, so each copy's exchanges
    // with the others are counted into its bin here.
    const LDObjData& odg = stats->objData[cands[members[0]].idx];
    auto memberPe = [&](int k) { return binPe[cands[k].bin]; };   // the bin's PE: right for a copy planned this round too
    // What one peer costs a copy in a group of `fan`: what the copy measured,
    // when it has exchanged (replicaCostNode / replicaCostRemote), else the
    // calibrated transfer-cost table's figure for the declared bytes and
    // messages at that fan (lbcalib's fan sweep; the tier alone when the
    // table has no fan entry).
    // A measurement was taken at the group's present size, `count`; at another
    // fan it is scaled by the table's ratio between the two (the measurement
    // sets the level, the table the shape: two copies that measured 7 ms a
    // peer say nothing about four, which the table puts at twenty times that).
    auto peerCost = [&](int peA, int peB, int fan) {
      const bool same = CmiPhysicalNodeID(peA) == CmiPhysicalNodeID(peB);
      const double table = costCfg.exchangePerPeer(same, fan, odg.replicaExchangeMsgs, (double)odg.replicaExchangeBytes);
      const double measured = same ? odg.replicaCostNode : odg.replicaCostRemote;
      if (measured <= 0.0 || count < 2) return table;
      if (fan == count) return measured;
      const double tableNow = costCfg.exchangePerPeer(same, count, odg.replicaExchangeMsgs, (double)odg.replicaExchangeBytes);
      return tableNow > 0.0 ? measured * table / tableNow : measured;
    };
    std::vector<double> xcost(members.size(), 0.0);
    if (count > 1)
      for (size_t a = 0; a < members.size(); a++)
      {
        for (size_t b = 0; b < members.size(); b++)
          if (a != b) xcost[a] += peerCost(memberPe(members[a]), memberPe(members[b]), count);
        bins[cands[members[a]].bin].load += xcost[a];
      }
    double before = 0.0;
    for (int b = 0; b < nbins; b++) before = std::max(before, bins[b].load);

    if (want > count)
    {
      // More copies. Every count k from count + 1 to the need is planned in
      // turn, each plan extending the one before by one copy, priced with the
      // copies' real shares (replicaShare: the declared partition blocks, or
      // 1/k) and their exchanges; the count that leaves the step's critical
      // path, the fullest bin anywhere, lowest is taken if it lowers it by
      // more than the tolerance that defines T. A count that gains nothing on
      // the way is passed over: three copies of a four-partition object leave
      // one copy with half of it. The copies that exist now are the only
      // sources a clone can take its state from and the only ones a notice
      // can reach. A copy is paid for every step, so relieving a bin that was
      // not the fullest, or leaving another bin as full as the one relieved,
      // buys nothing: eight experts on eight GPUs, each 0.7-1.4x the mean,
      // got two copies that lowered nothing (job 23388275: +29% in bf16
      // training, +35% in bf16 serving); on Mixtral's real routing (hottest
      // 1.27x, the next 1.18x) a copy that lowered its own bin but not the
      // fullest cost +15% in training and gained 10% in serving, where the
      // exchange is free (job 23411075): the declared cost is what tells the
      // two apart. Three copies of the SmartMoE routing's hot expert doubled
      // the training step (job 23581818): the exchange grew with the copies
      // while the fullest copy kept half the tokens, which the block shares
      // and the measured exchange cost now put into the plan.
      const std::vector<int> existing = members;
      auto homePe = [&](int k) { return stats->from_proc[cands[k].idx]; };
      int nextRank = 0;
      for (bool used = true; used; nextRank++)
      {
        used = false;
        for (int k : existing)
          if (stats->objData[cands[k].idx].replicaRank == nextRank) used = true;
      }
      nextRank--;
      // The tentative state the plans are priced on: the bins' loads and, per
      // member (existing, then planned), its rank, bin and load. Rebuilt for
      // every count tried, and the count's copies placed from scratch under
      // it: a copy's block of partitions, so its share and the host that
      // feeds it, depends on the count (rank 1 of 2 serves half the
      // partitions, rank 1 of 4 a quarter), and a copy placed for one count
      // sat on the wrong host under the next (job 23627965).
      // prefHost: the host of the copy's inputs (-1: none declared); withInputs: it
      // got a bin there; otherwise hostRefused/memRefused say why not, and
      // hostMostLoaded/hostMostCpu the host-time refusal that was closest.
      struct PlannedCopy { int src; int dst; int rank; bool srcAlone; double after;
                           int prefHost; bool withInputs; int hostRefused; int memRefused; int hostMostLoaded; double hostMostCpu; };
      std::vector<double> tbins(nbins);
      std::vector<int> trank, tbin;
      std::vector<double> tload, txc;   // txc: each member's exchanges in the tentative state
      std::vector<int> binObjs(nbins, 0);
      auto resetTentative = [&]() {
        std::fill(binObjs.begin(), binObjs.end(), 0);
        for (const Cand& cd : cands) if (cd.bin >= 0 && cd.bin < nbins) binObjs[cd.bin]++;
        for (int b = 0; b < nbins; b++) tbins[b] = bins[b].load;
        trank.clear(); tbin.clear(); tload.clear(); txc = xcost;
        for (size_t a = 0; a < members.size(); a++)
        {
          trank.push_back(stats->objData[cands[members[a]].idx].replicaRank);
          tbin.push_back(cands[members[a]].bin);
          tload.push_back(cands[members[a]].load);
        }
      };
      std::vector<PlannedCopy> plan, bestPlan, localPlan, fitPlan;
      double fitAfter = 0.0;
      std::vector<double> afterAt;   // per count tried, the fullest bin it leaves
      auto holdsBin = [&](int b) { for (int x : tbin) if (x == b) return true; return false; };
      double bestAfter = before - eps * avg;   // what a plan must beat
      int bestK = count;
      bool unplaceable = false;
      for (int k = count + 1; k <= want && !unplaceable; k++)
      {
        resetTentative();
        plan.clear();
        for (int c = count; c < k; c++)
        {
          const int now = (int)trank.size();
          size_t srcA = 0;
          for (size_t a = 0; a < existing.size(); a++)
            if (tbins[tbin[a]] > tbins[tbin[srcA]]) srcA = a;
          const int src = existing[srcA];
          const LDObjData& sod = stats->objData[cands[src].idx];
          const int srcPe = homePe(src);
          const int rank = nextRank + (int)plan.size();
          const double share = L * sod.replicaShare(rank, k);
          // The destination: the fullest bin that still clears T with the copy's
          // share, else the emptiest; never a bin the group is already on, and
          // only where the memory ledger says the copy fits. When the copies
          // have no exchange to pay for and the object declared partitions, the
          // bins of the host that originates most of this copy's block of them
          // (LBManager::setPartition on the feeding PEs) are tried first, and
          // the others only when none of those can take it: the copy's inputs
          // then stay on their host (MoE serving, two A100 nodes: every copy of
          // the hot expert on one host put 100 MB a step each way over the link).
          int prefHost = -1;
          if (sod.replicaParts > 0 && sod.replicaExchangeBytes == 0)
          {
            const int P = sod.replicaParts, lo = (P * rank) / k, hi = (P * (rank + 1)) / k;
            std::map<int, int> feeders;
            for (int pe = 0; pe < stats->nprocs(); pe++)
            {
              const int part = stats->procs[pe].partition;
              if (part >= lo && part < hi) feeders[CmiPhysicalNodeID(pe)]++;
            }
            int most = 0;
            for (const auto& kv : feeders) if (kv.second > most) { most = kv.second; prefHost = kv.first; }
          }
          int dst = -1;
          bool withInputs = false;
          int hostRefused0 = 0, memRefused0 = 0, hostMostLoaded0 = -1;
          double hostMostCpu0 = -1.0;
          for (int pass = (prefHost >= 0 ? 0 : 1); pass < 2 && dst < 0; pass++)
          {
          double bestFit = -1.0, emptiest = std::numeric_limits<double>::max();
          int emptiestBin = -1;
          int hostRefusedHere = 0, memRefusedHere = 0, hostMostLoaded = -1;
          double hostMostCpu = -1.0;
          for (int b = 0; b < nbins; b++)
          {
            if (holdsBin(b)) continue;
            if (pass == 0 && CmiPhysicalNodeID(binPe[b]) != prefHost) continue;
            // Past the cap only where the copy would leave this PE's host time
            // at or above the source's: below it, the busiest host still drops
            // (a host-only run, where host time is the load, needs exactly that).
            if (peCpu[binPe[b]] + sod.wallTime / k > hostCap &&
                peCpu[binPe[b]] + sod.wallTime / k >= peCpu[srcPe])
            {
              hostRefusals++; hostRefusedHere++;
              if (peCpu[binPe[b]] > hostMostCpu) { hostMostCpu = peCpu[binPe[b]]; hostMostLoaded = binPe[b]; }
              continue;
            }
  #if CMK_GPU
            if (!ledger.feasibleArrival(memModel.footprint(cands[src].idx) + sod.replicaExtraFor(k), binPe[b]))
            {
              memRefusals++; memRefusedHere++;
              continue;
            }
  #endif
            const double after = tbins[b] + share;
            if (splitGroup)
            {
              // A split group's copy: by load, the fewest objects breaking ties; a copy that carries next to nothing
              // now by the fewest objects alone, since by load every such copy lands in the one emptiest bin, which
              // it leaves the emptiest (2026-11-09, SmartMoE on 16 GPUs: 24 copies of idle experts on one GPU,
              // which then ran out of device memory at step 21).
              const bool idleCopy = share <= eps * avg;
              const double kb = idleCopy ? (double)binObjs[b] : tbins[b];
              const double kd = dst < 0 ? 0.0 : (idleCopy ? (double)binObjs[dst] : tbins[dst]);
              const double tb = idleCopy ? tbins[b] : (double)binObjs[b];
              const double td = dst < 0 ? 0.0 : (idleCopy ? tbins[dst] : (double)binObjs[dst]);
              if (dst < 0 || kb < kd || (kb == kd && tb < td)) { bestFit = tbins[b]; dst = b; }
            }
            else if (sod.replicaExchangeBytes == 0)
            {
              // Nothing is saved by packing a copy that pays no exchange: the
              // emptiest bin that still clears T (two A100 nodes, Mixtral: the
              // fullest-that-fits rule stacked two copies on three GPUs and left
              // the fourth empty, 30.5 against 19.6 ms a step, job 23646171).
              if (after <= T && (dst < 0 || tbins[b] < bestFit)) { bestFit = tbins[b]; dst = b; }
            }
            else if (after <= T && after > bestFit) { bestFit = after; dst = b; }
            if (tbins[b] < emptiest) { emptiest = tbins[b]; emptiestBin = b; }
          }
          if (dst < 0) dst = emptiestBin;
          if (pass == 0)
          {
            withInputs = dst >= 0;
            hostRefused0 = hostRefusedHere; memRefused0 = memRefusedHere;
            hostMostLoaded0 = hostMostLoaded; hostMostCpu0 = hostMostCpu;
          }
          }
  #if CMK_GPU
          // The source grows by the exchange state too when it was alone.
          if (dst >= 0 && now == 1 && !ledger.feasibleArrival(sod.replicaExtraFor(2), srcPe))
          {
            memRefusals++;
            dst = -1;
          }
  #endif
          if (dst < 0)
          {
            if (CkMyPe() == cur_ld_balancer)
              CkPrintf("CharmLB> ShedLB: group %d carries %.3f of a mean %.3f and would need %d copies, "
                       "but no bin can hold copy %d\n", g, L, avg, want, now + 1);
            unplaceable = true;
            break;
          }
          const int dstPe = binPe[dst];
          // Every member's share under k copies and its exchanges at fan k (the
          // fan prices every peer anew), on its bin; the new copy's share and all
          // its exchanges on the destination.
          double dstx = 0.0;
          for (size_t a = 0; a < trank.size(); a++)
          {
            const double nl = L * sod.replicaShare(trank[a], k);
            double nx = peerCost(binPe[tbin[a]], dstPe, k);
            for (size_t b = 0; b < trank.size(); b++)
              if (b != a) nx += peerCost(binPe[tbin[a]], binPe[tbin[b]], k);
            tbins[tbin[a]] += (nl - tload[a]) + (nx - txc[a]);
            tload[a] = nl;
            txc[a] = nx;
            dstx += peerCost(dstPe, binPe[tbin[a]], k);
          }
          tbins[dst] += share + dstx;
          binObjs[dst]++;
          txc.push_back(dstx);
          trank.push_back(rank); tbin.push_back(dst); tload.push_back(share);
          double after = 0.0;
          for (int b = 0; b < nbins; b++) after = std::max(after, tbins[b]);
          plan.push_back(PlannedCopy{src, dst, rank, now == 1, after, prefHost, withInputs,
                                     hostRefused0, memRefused0, hostMostLoaded0, hostMostCpu0});
          if (_lb_args.debug() > 0 && CkMyPe() == cur_ld_balancer)
            CkPrintf("[ShedLB] group %d: %d copies (copy %d on PE %d, share %.3f, exchange %.3f) would leave the "
                     "fullest bin at %.3f; now %.3f, to beat %.3f\n", g, k, rank, dstPe, share, dstx, after, before,
                     before - eps * avg);
        }
        if (unplaceable || (int)plan.size() != k - count) break;
        const double after = plan.back().after;
        afterAt.push_back(after);
        if (after < bestAfter) { bestAfter = after; bestK = k; bestPlan = plan; }
        if (k == fitK) { fitAfter = after; fitPlan = plan; }
        if (localRule && k == nHosts && after <= std::max(before, T)) localPlan = plan;
      }
      plan = bestPlan;
      // Prefer the count at which every copy fits the target when it leaves the fullest bin no worse than the best
      // count within the tolerance: past the best, the fullest bin is another object's (the shed loop's to move),
      // and fewer copies would leave each one of this group over the target for the moves to work around
      // (2026-11-08, SmartMoE on 16 GCDs: 3 copies where 8 fit, the hot GPU at 1.5x the mean after the round). No
      // worse is enough: while another group holds the fullest bin, every group's fitting count leaves it as it is,
      // and a strict improvement would give the next-heaviest groups no copies until a later round.
      if (!fitPlan.empty() && fitK > bestK && fitAfter <= before && fitAfter <= bestAfter + eps * avg)
      {
        bestK = fitK;
        bestAfter = fitAfter;
        plan = fitPlan;
      }
      // A split group takes its count whatever the fullest bin does now: the per-step split, not this model of
      // equal shares, decides the bins' loads.
      if (splitGroup && !fitPlan.empty() && fitK > count)
      {
        bestK = fitK;
        bestAfter = fitAfter;
        plan = fitPlan;
      }
      bool localChosen = false;
      if (bestK == count && localRule && !localPlan.empty())
      {
        bestK = nHosts; plan = localPlan; localChosen = true;
      }
      if (bestK == count)
      {
        if (CkMyPe() == cur_ld_balancer && !afterAt.empty())
        {
          double lowest = afterAt[0];
          int atK = count + 1;
          for (size_t i = 1; i < afterAt.size(); i++)
            if (afterAt[i] < lowest) { lowest = afterAt[i]; atK = count + 1 + (int)i; }
          CkPrintf("CharmLB> ShedLB: group %d: load %.3f exceeds the per-bin target %.3f, but no count up to %d "
                   "lowers the fullest bin enough (%d copies: %.3f of %.3f): no copy\n",
                   g, L, T, count + (int)afterAt.size(), atK, lowest, before);
        }
        continue;
      }
      // A split group's new copies are numbered nearest first: the copies serve in rank order (each count is a
      // prefix of the ranks), and a copy on its source's host lands first (2026-11-09, SmartMoE on 16 GPUs: the hot
      // expert's copies within its node landed 90-110 ms after the round, those across 220-240 ms, and ranks 1-4
      // were the far ones, so the near three served only with them).
      if (splitGroup && bestK - count > 1)
      {
        std::vector<PlannedCopy> taken(plan.begin(), plan.begin() + (bestK - count));
        int lowRank = taken[0].rank;
        for (const PlannedCopy& pc : taken) lowRank = std::min(lowRank, pc.rank);
        std::stable_sort(taken.begin(), taken.end(), [&](const PlannedCopy& x, const PlannedCopy& y) {
          const bool xn = CmiPhysicalNodeID(binPe[x.dst]) == CmiPhysicalNodeID(homePe(x.src));
          const bool yn = CmiPhysicalNodeID(binPe[y.dst]) == CmiPhysicalNodeID(homePe(y.src));
          return xn && !yn;
        });
        for (size_t i = 0; i < taken.size(); i++) { taken[i].rank = lowRank + (int)i; plan[i] = taken[i]; }
      }
      // The plan up to bestK is taken: the clones and their notices, the
      // ledger, the members and their loads.
      for (int i = 0; i < bestK - count; i++)
      {
        const PlannedCopy& pc = plan[i];
        const LDObjData& sod = stats->objData[cands[pc.src].idx];
        const int srcPe = homePe(pc.src), dstPe = binPe[pc.dst];
#if CMK_GPU
        ledger.commitArrival(memModel.footprint(cands[pc.src].idx) + sod.replicaExtraFor(bestK), dstPe);
        if (pc.srcAlone) ledger.commitArrival(sod.replicaExtraFor(2), srcPe);
#endif
        peCpu[dstPe] += sod.wallTime / bestK;
        pinned[pc.src] = 1;
        stats->clones.push_back(LBAction{cands[pc.src].idx, srcPe, dstPe, pc.rank});
        for (int k : existing)
          if (k != pc.src) stats->cloneNotices.push_back(LBAction{cands[k].idx, homePe(k), dstPe, pc.rank});
        clonesPlanned++;
        if (CkMyPe() == cur_ld_balancer && localChosen)
          CkPrintf("CharmLB> ShedLB: replicate group %d for locality: load %.3f, no exchange to pay, one copy per "
                   "host; copy %d from PE %d to PE %d, %d copies in all, this one %.3f; the fullest bin %.3f -> %.3f\n",
                   g, L, pc.rank, srcPe, dstPe, bestK, L * sod.replicaShare(pc.rank, bestK), before, pc.after);
        else if (CkMyPe() == cur_ld_balancer)
          CkPrintf("CharmLB> ShedLB: replicate group %d: load %.3f exceeds the per-bin target %.3f "
                   "(mean %.3f); copy %d from PE %d to PE %d, %d copies in all, this one %.3f; "
                   "the fullest bin %.3f -> %.3f\n",
                   g, L, T, avg, pc.rank, srcPe, dstPe, bestK, L * sod.replicaShare(pc.rank, bestK), before, pc.after);
        // Where the copy's inputs are, or why not: the placement is a decision
        // the run should show, as the copy is.
        if (CkMyPe() == cur_ld_balancer && pc.prefHost >= 0)
        {
          if (pc.withInputs)
            CkPrintf("[ShedLB] group %d copy %d: placed with its inputs, on host %d\n", g, pc.rank, pc.prefHost);
          else
            CkPrintf("[ShedLB] group %d copy %d: no bin on host %d, where its inputs are, could take it: "
                     "%d refused for host time (the closest, PE %d, at %.3f against the cap %.3f), %d for memory\n",
                     g, pc.rank, pc.prefHost, pc.hostRefused, pc.hostMostLoaded, pc.hostMostCpu, hostCap, pc.memRefused);
        }
        // The new copy is a member for the same-bin rule and the loads for the rest of the step.
        cands.push_back(Cand{cands[pc.src].idx, 0.0, cands[pc.src].bytes, pc.dst});
        pinned.push_back(1);
        members.push_back((int)cands.size() - 1);
        xcost.push_back(0.0);
      }
      for (size_t a = 0; a < members.size(); a++)
      {
        const int k = members[a];
        const int r = a < existing.size() ? stats->objData[cands[k].idx].replicaRank : plan[a - existing.size()].rank;
        const double nl = L * odg.replicaShare(r, bestK);
        double nx = 0.0;
        for (size_t c = 0; c < members.size(); c++)
          if (c != a) nx += peerCost(binPe[cands[k].bin], binPe[cands[members[c]].bin], bestK);
        bins[cands[k].bin].load += (nl + nx) - (cands[k].load + xcost[a]);
        cands[k].load = nl;
        xcost[a] = nx;
      }
    }
    else if (count > 1)
    {
      // One copy fewer: the highest-ranked copy, so the application's
      // partition of the work over ranks 0..k-1 stays contiguous. Taken back
      // when the remaining copies would sit below the mean (T (1 - eps) ~
      // avg), or when the exchanges cost the critical path more than the
      // sharing saves it.
      // Each remaining copy's share under count - 1 copies (its partition block).
      auto shareOf = [&](int k) { return L * odg.replicaShare(stats->objData[cands[k].idx].replicaRank, count - 1); };
      double share = 0.0;   // the largest of them: what must fit below the target
      for (int k : members)
        if (stats->objData[cands[k].idx].replicaRank < count - 1) share = std::max(share, shareOf(k));
      int victim = -1;
      for (int k : members)
      {
        if (stats->objData[cands[k].idx].replicaRank == 0) continue;
        if (victim < 0 || stats->objData[cands[k].idx].replicaRank >
                              stats->objData[cands[victim].idx].replicaRank)
          victim = k;
      }
      if (victim < 0) continue;
      const int vPe = stats->from_proc[cands[victim].idx];
      size_t va = 0;
      for (size_t a = 0; a < members.size(); a++) if (members[a] == victim) va = a;
      // The remaining copies' exchanges at the smaller fan.
      std::vector<double> nxc(members.size(), 0.0);
      for (size_t a = 0; a < members.size(); a++)
        for (size_t c = 0; c < members.size(); c++)
          if (c != a && members[c] != victim && members[a] != victim)
            nxc[a] += peerCost(memberPe(members[a]), memberPe(members[c]), count - 1);
      double after = 0.0;
      for (int b = 0; b < nbins; b++)
      {
        double l = bins[b].load;
        for (size_t a = 0; a < members.size(); a++)
        {
          const int k = members[a];
          if (cands[k].bin != b) continue;
          if (k == victim) l -= cands[k].load + xcost[a];
          else l += (shareOf(k) + nxc[a]) - (cands[k].load + xcost[a]);
        }
        after = std::max(after, l);
      }
      // Copies made for locality (an exchange-free partitioned group with one
      // copy per host) are kept: their share is small by design, and retiring
      // one puts its host's tokens back on the link.
      if (odg.replicaParts > 0 && odg.replicaExchangeBytes == 0 && nHosts > 1 && count <= nHosts) continue;
      // A split group keeps its copies up to one past its count (the count is ceil(L/T) + 2, above), so a load
      // that hovers at a boundary does not clone and retire on alternate rounds.
      if (splitGroup && count <= want + 1) continue;
      const bool fits = splitGroup || share <= T * (1.0 - eps);
      const bool cheaper = after <= before - eps * avg;
      if (!fits && !cheaper) continue;
      bins[cands[victim].bin].load -= cands[victim].load + xcost[va];
      peCpu[vPe] -= stats->objData[cands[victim].idx].wallTime;
      for (size_t a = 0; a < members.size(); a++)
      {
        const int k = members[a];
        if (k == victim) continue;
        bins[cands[k].bin].load += (shareOf(k) + nxc[a]) - (cands[k].load + xcost[a]);
        cands[k].load = shareOf(k);
        xcost[a] = nxc[a];
      }
      cands[victim].load = 0.0;
      pinned[victim] = 1;
      const int vRank = stats->objData[cands[victim].idx].replicaRank;
      stats->retires.push_back(LBAction{cands[victim].idx, vPe, vPe, vRank});
      for (int k : members)
        if (k != victim) stats->retireNotices.push_back(LBAction{cands[k].idx, stats->from_proc[cands[k].idx], vPe, vRank});
      retiresPlanned++;
      if (CkMyPe() == cur_ld_balancer)
        CkPrintf("CharmLB> ShedLB: retire replica %d of group %d on PE %d: %s (load %.3f, %d copies, "
                 "fullest bin %.3f -> %.3f, target %.3f, mean %.3f)\n",
                 stats->objData[cands[victim].idx].replicaRank, g, vPe,
                 fits ? "the group fits fewer copies" : "the exchange costs more than the sharing saves",
                 L, count - 1, before, after, T * (1.0 - eps), avg);
      xcost.erase(xcost.begin() + va);
      members.erase(std::find(members.begin(), members.end(), victim));
    }
  }
  if (clonesPlanned || retiresPlanned)
  {
    maxLoad = 0.0;
    for (int b = 0; b < nbins; b++) maxLoad = std::max(maxLoad, bins[b].load);
  }

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

  // Moves for locality: a lone object with partitions and no exchange (a
  // serving expert without a copy) whose host is not the one feeding its
  // first block of partitions goes to the emptiest bin of that host that can
  // take it; the round after, the copy rule above gives it a copy per host.
  for (size_t k = 0; k < cands.size(); k++)
  {
    if (moved[k] || pinned[k]) continue;
    Cand& c = cands[k];
    const LDObjData& cod = stats->objData[c.idx];
    if (!(cod.replicaParts > 0 && cod.replicaExchangeBytes == 0 && nHosts > 1 && cod.replicaCount == 1 &&
          cod.migratable)) continue;
    int block0Host = -1;
    {
      const int P = cod.replicaParts, hi = P / nHosts;
      std::map<int, int> feeders;
      for (int pe = 0; pe < stats->nprocs(); pe++)
      {
        const int part = stats->procs[pe].partition;
        if (part >= 0 && part < hi) feeders[CmiPhysicalNodeID(pe)]++;
      }
      int most = 0;
      for (const auto& kv : feeders) if (kv.second > most) { most = kv.second; block0Host = kv.first; }
    }
    const int oldPe = stats->from_proc[c.idx];
    if (block0Host < 0 || CmiPhysicalNodeID(oldPe) == block0Host) continue;
    // An object that takes next to nothing has no traffic to keep local, and moving it costs a migration; and since
    // it adds nothing to a bin, the emptiest bin stays the emptiest and would take every such object (2026-11-09,
    // SmartMoE on 16 GCDs: ten idle experts moved onto PE 5 at step 5, three of them among the hottest by step 20).
    if (c.load <= eps * avg) continue;
    for (int b = 0; b < nbins; b++) binPe[b] = landingPe(b);
    int dst = -1;
    double emptiest = std::numeric_limits<double>::max();
    for (int b = 0; b < nbins; b++)
    {
      if (b == c.bin || CmiPhysicalNodeID(binPe[b]) != block0Host) continue;
      // Not onto a bin it would push past the target: locality is worth a link crossing, not the fullest GPU
      // (2026-11-08, SmartMoE on 16 GCDs: the second-hottest expert, 1.8x the target alone, went onto a GPU
      // that held a copy of the hottest and stayed the job's fullest after both rounds). It stays, and the
      // replication pass splits it if it is over the target.
      if (bins[b].load + c.load > T) continue;
      if (peCpu[binPe[b]] + cod.wallTime > hostCap) { hostRefusals++; continue; }
#if CMK_GPU
      if (!ledger.feasible(c.idx, oldPe, binPe[b])) { memRefusals++; continue; }
#endif
      if (bins[b].load < emptiest) { emptiest = bins[b].load; dst = b; }
    }
    if (dst < 0) continue;
    const int dstPe = binPe[dst];
    bins[c.bin].load -= c.load;
    bins[dst].load += c.load;
    c.bin = dst;
    shedBytes += c.bytes;
#if CMK_GPU
    ledger.commit(c.idx, oldPe, dstPe);
#endif
    stats->to_proc[c.idx] = dstPe;
    peCpu[dstPe] += cod.wallTime;
    peCpu[oldPe] -= cod.wallTime;
    moves++;
    moved[k] = 1;
    if (CkMyPe() == cur_ld_balancer)
      CkPrintf("CharmLB> ShedLB: move group %d for locality: PE %d -> PE %d, the host that feeds its first block "
               "of partitions (load %.3f)\n", cod.replicaGroup, oldPe, dstPe, c.load);
  }

  for (;;)
  {
    int h = 0;
    for (int b = 1; b < nbins; b++)
      if (bins[b].load > bins[h].load) h = b;
    if (bins[h].load <= T) break;
    const double excess = bins[h].load - T;
    for (int b = 0; b < nbins; b++) binPe[b] = landingPe(b);

    int pick = -1, pickDst = -1;
    double pickScore = -1.0, pickSmallest = std::numeric_limits<double>::max();
    for (size_t k = 0; k < cands.size(); k++)
    {
      if (moved[k] || pinned[k]) continue;
      const Cand& c = cands[k];
      if (c.bin != h) continue;
      const LDObjData& cod = stats->objData[c.idx];

      // Best fit among the other bins: fullest that still clears T.
      int dst = -1;
      double bestAfter = -1.0;
      for (int b = 0; b < nbins; b++)
      {
        if (b == h) continue;
        // Copies of one group never share a bin.
        if (cod.replicable() && groupHoldsBin(cod.replicaGroup, b)) continue;
        // An object with partitions and no exchange stays on the host that
        // feeds its block of them: a move across hosts would put its tokens
        // back on the link (the locality rules above).
        if (cod.replicaParts > 0 && cod.replicaExchangeBytes == 0 && nHosts > 1)
        {
          const int k = std::max(1, cod.replicaCount), P = cod.replicaParts;
          const int lo = (P * cod.replicaRank) / k, hi = (P * (cod.replicaRank + 1)) / k;
          std::map<int, int> feeders;
          for (int pe = 0; pe < stats->nprocs(); pe++)
          {
            const int part = stats->procs[pe].partition;
            if (part >= lo && part < hi) feeders[CmiPhysicalNodeID(pe)]++;
          }
          int most = 0, host = -1;
          for (const auto& kv : feeders) if (kv.second > most) { most = kv.second; host = kv.first; }
          if (host >= 0 && CmiPhysicalNodeID(binPe[b]) != host) continue;
        }
        const double after = bins[b].load + c.load;
        if (!(after <= T && after > bestAfter)) continue;
        if (peCpu[binPe[b]] + cod.wallTime > hostCap) { hostRefusals++; continue; }
#if CMK_GPU
        if (!ledger.feasible(c.idx, stats->from_proc[c.idx], binPe[b]))
        {
          memRefusals++;
          continue;
        }
#endif
        bestAfter = after;
        dst = b;
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
    c.bin = pickDst;
    shedBytes += c.bytes;

    // Within the destination bin, the least host-loaded PE. Host time decides
    // only this, never which bin the object lands on.
    const int oldPe = stats->from_proc[c.idx];
    const int dstPe = binPe[pickDst];
    if (dstPe != oldPe)
    {
#if CMK_GPU
      ledger.commit(c.idx, oldPe, dstPe);
#endif
      stats->to_proc[c.idx] = dstPe;
      peCpu[dstPe] += stats->objData[c.idx].wallTime;
      peCpu[oldPe] -= stats->objData[c.idx].wallTime;
      moves++;
    }
  }

  if (CkMyPe() == cur_ld_balancer)
  {
    // The refusal counts are decisions the run should show; the bins' summary
    // is the debug view.
#if CMK_GPU
    if (memRefusals > 0)
      CkPrintf("[ShedLB] %d destination check(s) refused because the bin could not "
               "hold the object\n", memRefusals);
#endif
    if (hostRefusals > 0)
      CkPrintf("[ShedLB] %d destination check(s) refused because the PE's host time would pass %.3f "
               "(the device target %.3f, the hosts' mean %.3f)\n", hostRefusals, hostCap, T, hostMean);
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
