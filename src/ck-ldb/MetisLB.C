/** \file MetisLB.C
 *
 *  Updated by Abhinav Bhatele, 2010-11-26 to use ckgraph
 */

/**
 * \addtogroup CkLdb
 */

/*@{*/

#include "MetisLB.h"
#include "ckgraph.h"
#include "LBLoadDim.h"
#include "LBMemoryContract.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <unordered_map>
#include <metis.h>

extern int quietModeRequested;

static void lbinit()
{
  LBRegisterBalancer<MetisLB>("MetisLB", "Use Metis(tm) to partition object graph");
  LBTurnCommOn();
}

MetisLB::MetisLB(const CkLBOptions& opt) : CBase_MetisLB(opt)
{
  lbname = "MetisLB";
  if (CkMyPe() == 0 && !quietModeRequested)
    CkPrintf("CharmLB> MetisLB created.\n");
}

// ---- scratch-remap: relabel the parts for overlap --------------------------------
//
// METIS numbers its parts arbitrarily, so a partition computed from scratch
// sends most objects somewhere new even when it is nearly the partition they
// are already in. Relabelling the parts to the current owners they overlap
// most -- heaviest overlap first, greedily -- recovers what can be recovered
// without touching the partition itself; only parts with equal target shares
// may exchange labels, so the balance is exactly what METIS produced. This is
// the remap half of Schloegel, Karypis and Kumar's scratch-remap, and it is
// what makes a from-scratch partition a repartitioner for a large
// perturbation. Measured on a 128x128 stencil over 256 nodes with a 12x hot
// disc: one step from 4.49x to 1.12x with 67% of the objects moving once,
// where diffusion took eight steps and five times the migrations to 1.9x.
// Off with +LBMetisNoRemap. Read once per process, since Converse removes an
// option from argv when it is read and every PE of a process shares one argv.
static bool metisRemapOn()
{
  static const bool on = []() {
    return !CmiGetArgFlagDesc(CkGetArgv(), "+LBMetisNoRemap",
                              "MetisLB: do not relabel the new parts to the owners they "
                              "overlap most (same partition, more migrations)");
  }();
  return on;
}

// sigma[part]: the label that part takes. current[k] is vertex k's label now,
// or -1 when it has none among these; target[p] a part's share, and labels
// are exchanged only between parts of near-equal share.
//
// Near-equal, not equal: a PE's share carries its measured background load
// (fixedCpu), so on a uniform machine the shares differ by fractions of a
// percent and no two are ever exactly equal -- with an exact test the
// relabel never fired, and a block map went to a uniform partition with every
// object moving. Two parts whose shares are within this fraction of each
// other may exchange labels; the balance METIS produced is then off by at
// most that fraction of a share, inside the 10% it was allowed anyway.
static const double kRelabelShareTolerance = 0.05;

static std::vector<int> relabelForOverlap(const std::vector<idx_t>& partOf, int nparts,
                                          const std::vector<int>& current,
                                          const std::vector<std::vector<double>>& target)
{
  std::vector<std::vector<int>> overlap(nparts, std::vector<int>(nparts, 0));
  for (size_t k = 0; k < partOf.size(); k++)
    if (current[k] >= 0 && current[k] < nparts) overlap[partOf[k]][current[k]]++;
  auto sameShare = [&](int a, int b) {
    for (size_t c = 0; c < target[a].size() && c < target[b].size(); c++)
    {
      const double scale =
          std::max(1e-12, std::max(std::fabs(target[a][c]), std::fabs(target[b][c])));
      if (std::fabs(target[a][c] - target[b][c]) > kRelabelShareTolerance * scale)
        return false;
    }
    return true;
  };
  struct Pair { int n, part, label; };
  std::vector<Pair> pairs;
  for (int p = 0; p < nparts; p++)
    for (int l = 0; l < nparts; l++)
      if (overlap[p][l] > 0 && sameShare(p, l)) pairs.push_back(Pair{overlap[p][l], p, l});
  std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
    if (a.n != b.n) return a.n > b.n;
    if (a.part != b.part) return a.part < b.part;
    return a.label < b.label;
  });
  std::vector<int> sigma(nparts, -1);
  std::vector<char> used(nparts, 0);
  for (const Pair& pr : pairs)
    if (sigma[pr.part] < 0 && !used[pr.label])
    {
      sigma[pr.part] = pr.label;
      used[pr.label] = 1;
    }
  // What is left keeps its own number where it can, else takes any free label
  // of equal share; one always exists, since parts and labels are one set.
  for (int p = 0; p < nparts; p++)
    if (sigma[p] < 0 && !used[p])
    {
      sigma[p] = p;
      used[p] = 1;
    }
  for (int p = 0; p < nparts; p++)
    if (sigma[p] < 0)
      for (int l = 0; l < nparts; l++)
        if (!used[l] && sameShare(p, l))
        {
          sigma[p] = l;
          used[l] = 1;
          break;
        }
  for (int p = 0; p < nparts; p++)
    if (sigma[p] < 0)
      for (int l = 0; l < nparts; l++)
        if (!used[l])
        {
          sigma[p] = l;
          used[l] = 1;
          break;
        }
  return sigma;
}

void MetisLB::work(LDStats* stats)
{
  /** ========================== INITIALIZATION ============================= */
  ProcArray* parr = new ProcArray(stats);
  ObjGraph* ogr = new ObjGraph(stats);

  /** ============================= STRATEGY ================================ */
  if (_lb_args.debug() >= 2)
  {
    CkPrintf("[%d] In MetisLB Strategy...\n", CkMyPe());
  }

  // Which dimension the cross level partitions on, and how far the other one
  // is from binding. ObjGraph decided it (LBLoadDim.h); this is the record.
  if (_lb_args.debug() > 0 && CkMyPe() == cur_ld_balancer)
  {
    const LBCriticality c = lbCriticalityOf(stats);
    CkPrintf("[%d] MetisLB load dimension: %s by %s (T_h %.6f over %d PEs, T_g %.6f over "
             "%d GPUs; alpha_h %.2f, alpha_g %.2f)\n",
             CkMyPe(), lbLoadDimName(ogr->deviceDim),
             lbLoadDimOverride() == LB_DIM_AUTO ? "criticality" : "flag", c.boundHost(),
             c.pes, c.boundDev(), c.gpus, c.alphaHost(), c.alphaDev());
  }

  const idx_t numVertices = ogr->vertices.size();
  if (numVertices == 0 || parr->availProcSize < 1)
  {
    ogr->convertDecisions(stats);
    delete parr;
    delete ogr;
    return;
  }

  // Memory contract: when per-object device footprints exist, memory becomes
  // a second METIS balancing constraint, so the partitioner spreads resident
  // device bytes as well as load. The verifier in CentralLB::Strategy then
  // hardens METIS's soft tolerance into hard capacity compliance.
  int nConstraints = 1;
  bool memAware = false;
#if CMK_CUDA
  LBMemoryModel memModel;
  memModel.build(stats);
  if (memModel.numDevices() > 0)
    for (int i = 0; i < numVertices && !memAware; i++)
      if (memModel.footprint(ogr->vertices[i].getVertexId()) > 0) memAware = true;
  if (memAware) nConstraints = 2;
#endif

  // The second load dimension as a partition constraint, when neither
  // dimension has the slack to be ignored (LBLoadDim.h). METIS balances every
  // constraint to its own tolerance, so the slack dimension is given the
  // ratio it can be out of balance before it binds: a group's load in it may
  // reach the step's lower bound T_lb, and its mean across groups is
  // alpha * T_lb, hence (1 + eps) / alpha. At alpha 0.2 that is 5.5 and the
  // constraint is as good as absent, which is why it is not added there at
  // all; at alpha 0.8 it is 1.4, and METIS will not let one group take the
  // whole of the other dimension's hot region while it balances this one.
  // The verify-and-repair pass after level 2 still runs behind it.
  int slackIdx = -1;
  double slackUbvec = 1.1;
  double slackAlpha = 0.0;
#if CMK_CUDA
  {
    const LBCriticality crit = lbCriticalityOf(stats);
    if (lbLoadDimOverride() == LB_DIM_AUTO && crit.comparable(lbLoadVectorAbove()))
    {
      slackIdx = nConstraints++;
      slackAlpha = ogr->deviceDim ? crit.alphaHost() : crit.alphaDev();
      slackUbvec = std::max(1.1, 1.1 / std::max(slackAlpha, 1e-6));
      if (_lb_args.debug() > 0 && CkMyPe() == cur_ld_balancer)
        CkPrintf("[%d] MetisLB: %s dimension carried as constraint %d, tolerance %.2f "
                 "(alpha %.2f)\n",
                 CkMyPe(), lbLoadDimName(!ogr->deviceDim), slackIdx, slackUbvec, slackAlpha);
    }
  }
  auto slackOf = [&](int i) {
    return ogr->deviceDim ? (double)stats->objData[i].wallTime
                          : (double)stats->objData[i].gpuTime;
  };
#endif

  // The object graph as one merged adjacency per vertex. METIS requires a
  // clean undirected structure -- no self-loop, no neighbour repeated within a
  // vertex's list, positive weights -- and none of that is guaranteed: the
  // central statistics are the per-PE commData lists concatenated
  // (CentralLB::depositData) with no merge by (sender, receiver), so one object
  // pair contributes one edge per PE that recorded it. METIS does not reject
  // such input, it walks off its arrays. Merging here also lets any subset of
  // the graph be re-partitioned without rebuilding it.
  int selfLoops = 0, outOfRange = 0, duplicates = 0;
  size_t rawEdges = 0;
  std::vector<std::map<int, long long>> adj(numVertices);
  for (int i = 0; i < numVertices; i++)
  {
    auto addEdge = [&](int nbr, int bytes) {
      rawEdges++;
      if (nbr == i) { selfLoops++; return; }
      if (nbr < 0 || nbr >= numVertices) { outOfRange++; return; }
      const auto res = adj[i].emplace(nbr, 0LL);
      if (!res.second) duplicates++;
      if (bytes > 0) res.first->second += bytes;
    };
    for (const auto& outEdge : ogr->vertices[i].sendToList)
      addEdge(outEdge.getNeighborId(), outEdge.getNumBytes());
    for (const auto& inEdge : ogr->vertices[i].recvFromList)
      addEdge(inEdge.getNeighborId(), inEdge.getNumBytes());
  }
  if (selfLoops || outOfRange || duplicates)
    CkPrintf("CharmLB> MetisLB: dropped %d self-loop(s) and %d out-of-range "
             "edge(s), merged %d duplicate edge(s) of %zu\n",
             selfLoops, outOfRange, duplicates, rawEdges);

  // The two levels balance different resources, so they need different weights.
  // Across GPU groups what counts is device work -- and device memory, when
  // footprints exist, since that is a per-GPU capacity. Within a group every PE
  // drives the same GPU, so spreading device time among them changes nothing
  // and a memory constraint means nothing; what differs there is the host-side
  // work each object costs the PE that runs it. Same split
  // GreedyRefineCentralGPULB makes: cross-group on gpuTime, within-group on
  // wallTime. Vertex index is the objData index (ObjGraph numbers them that
  // way), so both are read off the same object.
  double maxCross = 0.0, maxIntra = 0.0;
  for (int i = 0; i < numVertices; i++)
  {
    maxCross = std::max(maxCross, ogr->vertices[i].getVertexLoad());
    maxIntra = std::max(maxIntra, (double)stats->objData[i].wallTime);
  }
  /** each object load is normalized to an integer between 1 and 256 */
  const double crossRatio = (maxCross == 0) ? 0 : 256.0 / maxCross;
  const double intraRatio = (maxIntra == 0) ? 0 : 256.0 / maxIntra;
  double slackRatio = 0.0;
#if CMK_CUDA
  if (slackIdx >= 0)
  {
    double maxSlack = 0.0;
    for (int i = 0; i < numVertices; i++) maxSlack = std::max(maxSlack, slackOf(i));
    slackRatio = (maxSlack == 0) ? 0 : 256.0 / maxSlack;
  }
#endif

  std::vector<idx_t> vwgtCross((size_t)numVertices * nConstraints);
  std::vector<idx_t> vwgtIntra(numVertices);
  for (int i = 0; i < numVertices; i++)
  {
    idx_t* w = &vwgtCross[(size_t)i * nConstraints];
    // Floored at 1 throughout: METIS needs a positive weight to balance on,
    // and ceil() of a zero load is zero.
    w[0] = std::max((idx_t)1, (idx_t)ceil(ogr->vertices[i].getVertexLoad() * crossRatio));
#if CMK_CUDA
    if (memAware) {
      // Footprint in MB, floored at 1 so every object has nonzero weight in
      // the memory dimension.
      size_t fp = memModel.footprint(ogr->vertices[i].getVertexId());
      w[1] = (idx_t)(fp >> 20) + 1;
    }
    if (slackIdx >= 0)
      w[slackIdx] = std::max((idx_t)1, (idx_t)ceil(slackOf(i) * slackRatio));
#endif
    vwgtIntra[i] =
        std::max((idx_t)1, (idx_t)ceil(stats->objData[i].wallTime * intraRatio));
  }

  std::array<idx_t, METIS_NOPTIONS> options;
  METIS_SetDefaultOptions(options.data());
  options[METIS_OPTION_NUMBERING] = 0;   // C style numbering

  // Partition the subgraph induced by `verts` into `nparts`, returning a part
  // number per entry of `verts`. Edges leaving the subset are dropped: at the
  // second level they are the traffic the first level already decided to pay.
  // `ubvecIn` gives each constraint its own tolerance; null means METIS's
  // usual 1.1 for every one.
  auto partitionSubset = [&](const std::vector<int>& verts, idx_t nparts,
                             const std::vector<real_t>* tpwgts,
                             const std::vector<idx_t>& weights,
                             int ncon_in,
                             const std::vector<real_t>* ubvecIn) -> std::vector<idx_t>
  {
    const idx_t nv = (idx_t)verts.size();
    std::vector<idx_t> parts(verts.size(), 0);
    if (nv == 0 || nparts <= 1) return parts;

    std::unordered_map<int, idx_t> local;
    local.reserve(verts.size() * 2);
    for (idx_t k = 0; k < nv; k++) local[verts[k]] = k;

    std::vector<idx_t> xadj(nv + 1), adjncy, adjwgt;
    std::vector<idx_t> lvwgt((size_t)nv * ncon_in);
    idx_t e = 0;
    for (idx_t k = 0; k < nv; k++)
    {
      xadj[k] = e;
      for (int c = 0; c < ncon_in; c++)
        lvwgt[(size_t)k * ncon_in + c] = weights[(size_t)verts[k] * ncon_in + c];
      for (const auto& nb : adj[verts[k]])
      {
        const auto it = local.find(nb.first);
        if (it == local.end()) continue;
        adjncy.push_back(it->second);
        // Positive and in range: idx_t is 32 bits in a stock METIS build and
        // these are byte counts summed over the whole instrumented window.
        // Both endpoints sum the same records, so the capped weight stays
        // symmetric, which METIS also requires.
        adjwgt.push_back((idx_t)std::min<long long>(
            std::max<long long>(nb.second, 1),
            (long long)std::numeric_limits<idx_t>::max()));
        e++;
      }
    }
    xadj[nv] = e;
    // METIS reads the zeroth element even when there are no edges.
    if (adjncy.empty()) { adjncy.push_back(0); adjwgt.push_back(1); }

    idx_t ncon = ncon_in, nv_arg = nv, np = nparts, edgecut = 0;
    std::vector<real_t> ubvec(ncon_in, (real_t)1.1);
    if (ubvecIn != nullptr)
      for (int c = 0; c < ncon_in && c < (int)ubvecIn->size(); c++) ubvec[c] = (*ubvecIn)[c];
    METIS_PartGraphRecursive(&nv_arg, &ncon, xadj.data(), adjncy.data(), lvwgt.data(),
                             nullptr, adjwgt.data(), &np,
                             tpwgts ? const_cast<real_t*>(tpwgts->data()) : nullptr,
                             ubvec.data(), options.data(), &edgecut, parts.data());
    return parts;
  };

  // Two-level, the way GreedyRefineCentralGPULB and DiffusionLB's GPU
  // dimension are: the resource an object contends for is the GPU, and on this
  // machine several PEs share one. Partitioning straight into availProcSize
  // parts balances per-PE load and minimises PE-to-PE edge cut, when what
  // decides the step time is the aggregate load per GPU, and traffic between
  // PEs sharing a GPU costs almost nothing. So cut across GPU groups first,
  // then split each group's objects over the PEs that drive that GPU. With one
  // group (a CPU-only run, where gpu_device_id is unset and identical) this
  // reduces to exactly the flat partition it replaces.
  std::vector<std::vector<int>> grpPes;
  std::unordered_map<uint64_t, int> gpuIdToIdx;
  for (int pe = 0; pe < (int)parr->procs.size(); pe++)
  {
    if (!stats->procs[pe].available) continue;
    const uint64_t devId = stats->procs[pe].gpu_device_id;
    const auto it = gpuIdToIdx.find(devId);
    if (it == gpuIdToIdx.end())
    {
      gpuIdToIdx[devId] = grpPes.size();
      grpPes.push_back(std::vector<int>(1, pe));
    }
    else
      grpPes[it->second].push_back(pe);
  }
  const idx_t nGroups = (idx_t)grpPes.size();
  if (nGroups < 1)
  {
    ogr->convertDecisions(stats);
    delete parr;
    delete ogr;
    return;
  }

  // Where each PE sits, and what load is already nailed down there. A
  // non-migratable object is not partitioned -- it is fixed load on the PE it
  // is already on, GPU-side for its group and CPU-side for its PE -- and the
  // PE's measured background walltime is fixed CPU load too. METIS has no
  // notion of a part that starts out partly full, so this is expressed by
  // shrinking that part's target share by what it already carries. Partitioning
  // pinned objects along with the rest and then declining to move them, which
  // is the obvious shortcut, leaves their weight counted against whichever part
  // METIS chose rather than the one they are actually on.
  std::vector<int> groupOfPe(parr->procs.size(), -1);
  for (idx_t g = 0; g < nGroups; g++)
    for (int pe : grpPes[g]) groupOfPe[pe] = (int)g;

  // Relative PE speed. normalize_speed() has already scaled these against the
  // fastest PE but has not touched any object's measured load, so the work
  // stays raw and the capacity is what varies: a PE of speed s should be
  // targeted with s times the share of an identical fast one. Expressed
  // through the target weights rather than the object weights, since it is a
  // property of the processor, not of the object.
  std::vector<double> peSpeed(parr->procs.size(), 1.0);
  for (int pe = 0; pe < (int)parr->procs.size(); pe++)
  {
    const double sp = stats->procs[pe].pe_speed;
    peSpeed[pe] = (sp > 0.0) ? sp : 1.0;
  }

  std::vector<double> fixedCross(nGroups, 0.0), fixedMem(nGroups, 0.0), fixedSlack(nGroups, 0.0);
  std::vector<double> fixedCpu(parr->procs.size(), 0.0);
  for (int pe = 0; pe < (int)parr->procs.size(); pe++)
    if (groupOfPe[pe] >= 0)
    {
      fixedCpu[pe] = stats->procs[pe].bg_walltime * intraRatio;
      // When host time is the slack constraint, the PEs' background time is
      // fixed host load on their group.
      if (slackIdx >= 0 && ogr->deviceDim)
        fixedSlack[groupOfPe[pe]] += stats->procs[pe].bg_walltime * slackRatio;
    }

  std::vector<int> migVerts;
  migVerts.reserve(numVertices);
  for (int i = 0; i < numVertices; i++)
  {
    if (ogr->vertices[i].isMigratable()) { migVerts.push_back(i); continue; }
    const int pe = ogr->vertices[i].getCurrentPe();
    if (pe < 0 || pe >= (int)groupOfPe.size() || groupOfPe[pe] < 0) continue;
    fixedCross[groupOfPe[pe]] += ogr->vertices[i].getVertexLoad() * crossRatio;
    fixedCpu[pe] += stats->objData[i].wallTime * intraRatio;
#if CMK_CUDA
    if (memAware)
      fixedMem[groupOfPe[pe]] +=
          (double)(memModel.footprint(ogr->vertices[i].getVertexId()) >> 20) + 1.0;
    if (slackIdx >= 0) fixedSlack[groupOfPe[pe]] += slackOf(i) * slackRatio;
#endif
  }

  // Level one: migratable objects to GPU groups, on device load (and device
  // memory), each group's share proportional to how many PEs feed it less what
  // it already carries.
  double movCross = 0.0, movMem = 0.0, movSlack = 0.0;
  for (int v : migVerts)
  {
    movCross += (double)vwgtCross[(size_t)v * nConstraints];
    if (memAware) movMem += (double)vwgtCross[(size_t)v * nConstraints + 1];
    if (slackIdx >= 0) movSlack += (double)vwgtCross[(size_t)v * nConstraints + slackIdx];
  }
  const double allCross =
      movCross + std::accumulate(fixedCross.begin(), fixedCross.end(), 0.0);
  const double allMem =
      movMem + std::accumulate(fixedMem.begin(), fixedMem.end(), 0.0);
  const double allSlack =
      movSlack + std::accumulate(fixedSlack.begin(), fixedSlack.end(), 0.0);

  std::vector<real_t> tpwgts((size_t)nGroups * nConstraints);
  {
    std::vector<double> want((size_t)nGroups * nConstraints, 0.0);
    std::vector<double> sum(nConstraints, 0.0);
    // What fraction of the whole each group should end up with. When the
    // balanced resource is the device, that is how many GPUs the group has --
    // one -- so groups are equal up to how many PEs feed them. When it is the
    // host (no GPU dimension), capacity is the summed speed of its PEs.
    double speedAll = 0.0;
    for (idx_t g2 = 0; g2 < nGroups; g2++)
      for (int pe : grpPes[g2]) speedAll += peSpeed[pe];
    const bool deviceIsResource = ogr->deviceDim;
    for (idx_t g = 0; g < nGroups; g++)
    {
      double grpSpeed = 0.0;
      for (int pe : grpPes[g]) grpSpeed += peSpeed[pe];
      const double share = deviceIsResource
          ? (double)grpPes[g].size() / (double)parr->availProcSize
          : (speedAll > 0.0 ? grpSpeed / speedAll
                            : (double)grpPes[g].size() / (double)parr->availProcSize);
      double* wg = &want[(size_t)g * nConstraints];
      wg[0] = std::max(1e-6, allCross * share - fixedCross[g]);
      sum[0] += wg[0];
      if (memAware)
      {
        wg[1] = std::max(1e-6, allMem * share - fixedMem[g]);
        sum[1] += wg[1];
      }
      if (slackIdx >= 0)
      {
        // Same share as the binding dimension: a group of ppn PEs takes ppn
        // PEs' worth of host time, one GPU's worth of device time.
        wg[slackIdx] = std::max(1e-6, allSlack * share - fixedSlack[g]);
        sum[slackIdx] += wg[slackIdx];
      }
    }
    for (idx_t g = 0; g < nGroups; g++)
      for (int c = 0; c < nConstraints; c++)
        tpwgts[(size_t)g * nConstraints + c] =
            (real_t)(want[(size_t)g * nConstraints + c] / sum[c]);
  }

  CkPrintf("Metis partitioning %d migratable of %d objects over %d GPU group(s) "
           "of %d PEs\n", (int)migVerts.size(), (int)numVertices, (int)nGroups,
           parr->availProcSize);

  std::vector<real_t> ubvecCross(nConstraints, (real_t)1.1);
  if (slackIdx >= 0) ubvecCross[slackIdx] = (real_t)slackUbvec;
  const std::vector<idx_t> groupOf = partitionSubset(
      migVerts, nGroups, nGroups > 1 ? &tpwgts : nullptr, vwgtCross, nConstraints, &ubvecCross);

  // Scratch-remap, level one: the group each part becomes is the group its
  // objects mostly come from, where the shares allow.
  std::vector<idx_t> groupLabel = groupOf;
  int relabelledGroups = 0, relabelledPes = 0;
  if (metisRemapOn() && nGroups > 1)
  {
    std::vector<int> cur(migVerts.size(), -1);
    for (size_t k = 0; k < migVerts.size(); k++)
    {
      const int pe = ogr->vertices[migVerts[k]].getCurrentPe();
      if (pe >= 0 && pe < (int)groupOfPe.size()) cur[k] = groupOfPe[pe];
    }
    std::vector<std::vector<double>> shares(nGroups);
    for (idx_t g = 0; g < nGroups; g++)
      for (int c = 0; c < nConstraints; c++)
        shares[g].push_back(tpwgts[(size_t)g * nConstraints + c]);
    const std::vector<int> sigma = relabelForOverlap(groupOf, (int)nGroups, cur, shares);
    for (idx_t g = 0; g < nGroups; g++)
      if (sigma[g] != (int)g) relabelledGroups++;
    for (size_t k = 0; k < migVerts.size(); k++) groupLabel[k] = sigma[groupOf[k]];
  }

  std::vector<std::vector<int>> grpVerts(nGroups);
  for (size_t k = 0; k < migVerts.size(); k++)
    grpVerts[groupLabel[k]].push_back(migVerts[k]);

  // Level two: within a group, over the PEs sharing that GPU, on host load --
  // every PE here drives the same device, so device time cannot separate them.
  // These are real PE ids already, so no remapping through availPeMap.
  std::vector<int> newPe(numVertices, -1);
  for (idx_t g = 0; g < nGroups; g++)
  {
    const std::vector<int>& pes = grpPes[g];
    if (grpVerts[g].empty()) continue;

    double movCpu = 0.0, fixSum = 0.0;
    for (int v : grpVerts[g]) movCpu += (double)vwgtIntra[v];
    for (int pe : pes) fixSum += fixedCpu[pe];
    const double allCpu = movCpu + fixSum;

    double grpSpeed = 0.0;
    for (int pe : pes) grpSpeed += peSpeed[pe];

    std::vector<real_t> tpwgts2(pes.size());
    double sum2 = 0.0;
    for (size_t k = 0; k < pes.size(); k++)
    {
      const double share = (grpSpeed > 0.0) ? peSpeed[pes[k]] / grpSpeed
                                            : 1.0 / (double)pes.size();
      const double w = std::max(1e-6, allCpu * share - fixedCpu[pes[k]]);
      tpwgts2[k] = (real_t)w;
      sum2 += w;
    }
    for (auto& t : tpwgts2) t = (real_t)(t / sum2);

    const std::vector<idx_t> peOf = partitionSubset(
        grpVerts[g], (idx_t)pes.size(), pes.size() > 1 ? &tpwgts2 : nullptr,
        vwgtIntra, 1, nullptr);

    // Scratch-remap, level two: the PE each part becomes is the PE its
    // objects mostly come from, where the shares allow.
    std::vector<idx_t> peLabel = peOf;
    if (metisRemapOn() && pes.size() > 1)
    {
      std::vector<int> cur(grpVerts[g].size(), -1);
      for (size_t k = 0; k < grpVerts[g].size(); k++)
      {
        const int pe = ogr->vertices[grpVerts[g][k]].getCurrentPe();
        for (size_t j = 0; j < pes.size(); j++)
          if (pes[j] == pe)
          {
            cur[k] = (int)j;
            break;
          }
      }
      std::vector<std::vector<double>> shares(pes.size());
      for (size_t j = 0; j < pes.size(); j++) shares[j].push_back(tpwgts2[j]);
      const std::vector<int> sigma = relabelForOverlap(peOf, (int)pes.size(), cur, shares);
      for (size_t j = 0; j < pes.size(); j++)
        if (sigma[j] != (int)j) relabelledPes++;
      for (size_t k = 0; k < grpVerts[g].size(); k++) peLabel[k] = sigma[peOf[k]];
    }
    for (size_t k = 0; k < grpVerts[g].size(); k++)
      newPe[grpVerts[g][k]] = pes[peLabel[k]];
  }

  // ---- the dimension that was not partitioned ----------------------------
  //
  // Level one balanced the binding dimension across groups and level two
  // spread host time within each group. Nothing above looked at the other
  // dimension. With the device binding, a group can leave here
  // device-balanced and still hold more host work than its PEs can finish in
  // the step; with the host binding, a group can be handed more device time
  // than its one GPU can run in it. Check every group against the step the
  // partition achieved in the binding dimension, and repair the groups the
  // slack dimension would hold open -- the verify-and-repair the memory
  // contract applies after the partition, in the other dimension.
  //
  // Repair is greedy and bounded: from the group whose slack term is highest
  // over the target, move the object carrying the most of that dimension to
  // the group that ends up lowest in it, provided the binding dimension there
  // stays under target; stop when no such move exists. Unexercised on a real
  // workload as of 12 Sep 2026: both measured applications sit at alpha ~ 0.2
  // on their slack dimension, where no group ever trips the check.
#if CMK_CUDA
  {
    const bool devBinds = ogr->deviceDim;
    const int nPe = (int)parr->procs.size();
    auto hostOf = [&](int i) { return (double)stats->objData[i].wallTime; };
    auto devOf = [&](int i) { return (double)stats->objData[i].gpuTime; };
    auto peOfObj = [&](int i) {
      return newPe[i] >= 0 ? newPe[i] : ogr->vertices[i].getCurrentPe();
    };

    std::vector<double> grpDev(nGroups, 0.0), peHost(nPe, 0.0);
    for (int pe = 0; pe < nPe; pe++)
      if (groupOfPe[pe] >= 0) peHost[pe] = stats->procs[pe].bg_walltime;
    for (int i = 0; i < numVertices; i++)
    {
      const int pe = peOfObj(i);
      if (pe < 0 || pe >= nPe || groupOfPe[pe] < 0) continue;
      grpDev[groupOfPe[pe]] += devOf(i);
      peHost[pe] += hostOf(i) / peSpeed[pe];
    }
    auto grpHostMax = [&](int g) {
      double m = 0.0;
      for (int pe : grpPes[g]) m = std::max(m, peHost[pe]);
      return m;
    };
    auto bindTerm = [&](int g) { return devBinds ? grpDev[g] : grpHostMax(g); };
    auto slackTerm = [&](int g) { return devBinds ? grpHostMax(g) : grpDev[g]; };

    double target = 0.0, slackBefore = 0.0;
    for (idx_t g = 0; g < nGroups; g++)
    {
      target = std::max(target, bindTerm((int)g));
      slackBefore = std::max(slackBefore, slackTerm((int)g));
    }
    // The slack dimension cannot be brought under what its granularity
    // allows, so the limit is the larger of that and the binding dimension's
    // achieved max -- otherwise, when a flag forces the partition onto the
    // dimension that does not bind, every group is over the limit and the
    // repair churns to no end. For host time the reachable per-PE maximum is
    // the list-scheduling bound, the even spread plus the largest object's
    // share of a PE's surplus, since objects are indivisible; for device time
    // it is the even spread or the largest object, whichever is more. METIS's
    // own tolerance (ubvec 1.1) applies to both.
    double slackEven = 0.0;
    if (devBinds)
    {
      int nAvail = 0, ppnMax = 1;
      double hMax = 0.0;
      for (int pe = 0; pe < nPe; pe++)
        if (groupOfPe[pe] >= 0) { slackEven += peHost[pe]; nAvail++; }
      slackEven = nAvail > 0 ? slackEven / nAvail : 0.0;
      for (idx_t g = 0; g < nGroups; g++) ppnMax = std::max(ppnMax, (int)grpPes[g].size());
      for (int i : migVerts)
      {
        const int pe = peOfObj(i);
        if (pe >= 0 && pe < nPe) hMax = std::max(hMax, hostOf(i) / peSpeed[pe]);
      }
      slackEven += hMax * (1.0 - 1.0 / ppnMax);
    }
    else
    {
      double gMax = 0.0;
      for (idx_t g = 0; g < nGroups; g++) slackEven += grpDev[g];
      slackEven /= (double)nGroups;
      for (int i : migVerts) gMax = std::max(gMax, devOf(i));
      slackEven = std::max(slackEven, gMax);
    }
    const double eps = 0.1;
    const double limit = std::max(target, slackEven) * (1.0 + eps);

    int repaired = 0;
    bool stuck = false;
    if (nGroups > 1 && target > 0.0)
    {
      for (size_t iter = 0; iter < migVerts.size(); iter++)
      {
        int worst = -1;
        double worstSlack = limit;
        for (idx_t g = 0; g < nGroups; g++)
          if (slackTerm((int)g) > worstSlack) { worstSlack = slackTerm((int)g); worst = (int)g; }
        if (worst < 0) break;

        // The object in `worst` carrying the most of the slack dimension. A
        // host slack term is the group's busiest PE, so only that PE's
        // objects can lower it.
        int busiestPe = -1;
        if (devBinds)
          for (int pe : grpPes[worst])
            if (busiestPe < 0 || peHost[pe] > peHost[busiestPe]) busiestPe = pe;
        int obj = -1;
        double objSlack = 0.0;
        for (int i : migVerts)
        {
          const int pe = peOfObj(i);
          if (pe < 0 || pe >= nPe || groupOfPe[pe] != worst) continue;
          if (devBinds && pe != busiestPe) continue;
          const double s = devBinds ? hostOf(i) / peSpeed[pe] : devOf(i);
          if (s > objSlack) { objSlack = s; obj = i; }
        }
        if (obj < 0) { stuck = true; break; }

        // Destination: the group, and its least-loaded PE, that ends lowest
        // in the slack dimension with the binding dimension still under the
        // limit -- and strictly better than where the object is now.
        int bestG = -1, bestPe = -1;
        double bestAfter = worstSlack;
        for (idx_t g = 0; g < nGroups; g++)
        {
          if ((int)g == worst) continue;
          int pe = -1;
          for (int p : grpPes[g])
            if (pe < 0 || peHost[p] < peHost[pe]) pe = p;
          if (pe < 0) continue;
          const double hostAfter =
              std::max(grpHostMax((int)g), peHost[pe] + hostOf(obj) / peSpeed[pe]);
          const double devAfter = grpDev[g] + devOf(obj);
          const double bindAfter = devBinds ? devAfter : hostAfter;
          const double slackAfter = devBinds ? hostAfter : devAfter;
          if (bindAfter > limit) continue;
          if (slackAfter < bestAfter) { bestAfter = slackAfter; bestG = (int)g; bestPe = pe; }
        }
        if (bestG < 0) { stuck = true; break; }

        const int from = peOfObj(obj);
        peHost[from] -= hostOf(obj) / peSpeed[from];
        grpDev[worst] -= devOf(obj);
        peHost[bestPe] += hostOf(obj) / peSpeed[bestPe];
        grpDev[bestG] += devOf(obj);
        newPe[obj] = bestPe;
        repaired++;
      }
    }
    double slackAfter = 0.0;
    for (idx_t g = 0; g < nGroups; g++) slackAfter = std::max(slackAfter, slackTerm((int)g));
    if ((_lb_args.debug() > 0 || repaired > 0) && CkMyPe() == cur_ld_balancer)
      CkPrintf("[%d] MetisLB %s dimension partitioned to %.6f; %s dimension max %.6f -> %.6f "
               "against limit %.6f; %d repair move(s)%s\n",
               CkMyPe(), lbLoadDimName(devBinds), target, lbLoadDimName(!devBinds),
               slackBefore, slackAfter, limit, repaired,
               stuck ? " (stopped with a group still over the limit)" : "");
  }
#endif

  if (metisRemapOn() && _lb_args.debug() > 0 && CkMyPe() == cur_ld_balancer)
  {
    int moved = 0;
    for (int v : migVerts)
      if (newPe[v] >= 0 && newPe[v] != ogr->vertices[v].getCurrentPe()) moved++;
    CkPrintf("[%d] MetisLB remap: relabelled %d group(s) and %d PE part(s); %d of %d "
             "migratable objects move\n",
             CkMyPe(), relabelledGroups, relabelledPes, moved, (int)migVerts.size());
  }

  if (_lb_args.debug() > 1 && CkMyPe() == cur_ld_balancer)
  {
    // Does the partition actually keep talkers together? Compare the edge
    // weight that crosses a group boundary under the old placement with the
    // same under the new one. If the graph has no edges at all then METIS has
    // no locality information whatsoever and its partition is an arbitrary
    // permutation -- which on a cyclic initial map means throwing spatial
    // locality away rather than improving it.
    long long edges = 0, wTotal = 0, cutOld = 0, cutNew = 0;
    for (int i = 0; i < numVertices; i++)
    {
      const int oldPeI = ogr->vertices[i].getCurrentPe();
      const int newPeI = (newPe[i] >= 0) ? newPe[i] : oldPeI;
      for (const auto& nb : adj[i])
      {
        const int j = nb.first;
        if (j <= i) continue;                       // count each edge once
        const int oldPeJ = ogr->vertices[j].getCurrentPe();
        const int newPeJ = (newPe[j] >= 0) ? newPe[j] : oldPeJ;
        edges++;
        wTotal += nb.second;
        const int gOldI = (oldPeI >= 0 && oldPeI < (int)groupOfPe.size()) ? groupOfPe[oldPeI] : -1;
        const int gOldJ = (oldPeJ >= 0 && oldPeJ < (int)groupOfPe.size()) ? groupOfPe[oldPeJ] : -1;
        const int gNewI = (newPeI >= 0 && newPeI < (int)groupOfPe.size()) ? groupOfPe[newPeI] : -1;
        const int gNewJ = (newPeJ >= 0 && newPeJ < (int)groupOfPe.size()) ? groupOfPe[newPeJ] : -1;
        if (gOldI != gOldJ) cutOld += nb.second;
        if (gNewI != gNewJ) cutNew += nb.second;
      }
    }
    int moved = 0;
    for (int i = 0; i < numVertices; i++)
      if (newPe[i] >= 0 && newPe[i] != ogr->vertices[i].getCurrentPe()) moved++;
    CkPrintf("[%d] MetisLB locality: %lld edge(s) over %d objects, total weight "
             "%lld; cross-group weight before=%lld after=%lld; %d object(s) moved\n",
             CkMyPe(), edges, (int)numVertices, wTotal, cutOld, cutNew, moved);
  }

  if ((_lb_args.debug() > 1) && (CkMyPe() == cur_ld_balancer))
  {
    // The quantities the two levels actually target, under the new assignment.
    // A step is not over until both resources are done with it, so a group's
    // step time is set by whichever is busier: its GPU, or its busiest PE.
    // Reporting only the device term hides every intra-group difference.
    std::vector<double> grpDev(nGroups, 0.0);
    std::vector<double> peHost(parr->procs.size(), 0.0);
    for (int pe = 0; pe < (int)parr->procs.size(); pe++)
      if (groupOfPe[pe] >= 0) peHost[pe] = stats->procs[pe].bg_walltime;
    for (int i = 0; i < numVertices; i++)
    {
      const int pe = (newPe[i] >= 0) ? newPe[i] : ogr->vertices[i].getCurrentPe();
      if (pe < 0 || pe >= (int)groupOfPe.size() || groupOfPe[pe] < 0) continue;
      grpDev[groupOfPe[pe]] += ogr->vertices[i].getVertexLoad();
      peHost[pe] += stats->objData[i].wallTime / peSpeed[pe];
    }
    double devMax = 0.0, devSum = 0.0;
    for (idx_t g = 0; g < nGroups; g++)
    {
      devMax = std::max(devMax, grpDev[g]);
      devSum += grpDev[g];
      double grpHostMax = 0.0;
      for (int pe : grpPes[g]) grpHostMax = std::max(grpHostMax, peHost[pe]);
      CkPrintf("[%d]   group %d (%d PEs): deviceLoad=%.6f busiestPeHost=%.6f\n",
               CkMyPe(), (int)g, (int)grpPes[g].size(), grpDev[g], grpHostMax);
    }
    double hostMax = 0.0, hostSum = 0.0;
    int nAvail = 0;
    for (int pe = 0; pe < (int)parr->procs.size(); pe++)
      if (groupOfPe[pe] >= 0)
      { hostMax = std::max(hostMax, peHost[pe]); hostSum += peHost[pe]; nAvail++; }
    CkPrintf("[%d] MetisLB after LB: device max/avg=%.3f over %d group(s), "
             "host max/avg=%.3f over %d PEs\n", CkMyPe(),
             devSum > 0 ? devMax * nGroups / devSum : 1.0, (int)nGroups,
             hostSum > 0 ? hostMax * nAvail / hostSum : 1.0, nAvail);
  }

  // Objects that declared themselves non-migratable (setMigratable(false)) were
  // never in the partitioned set, so newPe stays -1 for them and they keep the
  // PE they are on -- their load was accounted there instead.
  for (int i = 0; i < numVertices; i++)
    if (newPe[i] >= 0 && newPe[i] != ogr->vertices[i].getCurrentPe())
      ogr->vertices[i].setNewPe(newPe[i]);

  if (_lb_args.debug() >= 1)
  {
    CkPrintf("[%d] MetisLB done! \n", CkMyPe());
  }

  /** ============================== CLEANUP ================================ */
  ogr->convertDecisions(stats);
  delete parr;
  delete ogr;
}

#include "MetisLB.def.h"

/*@}*/
