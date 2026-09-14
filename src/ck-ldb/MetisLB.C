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
#include "DiffusionCostModel.h"
#include "LBLoadDim.h"
#include "LBMemoryContract.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <string>
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

static bool metisRemapOn();
static double metisMinGain();
static double metisStickiness();

MetisLB::MetisLB(const CkLBOptions& opt) : CBase_MetisLB(opt)
{
  lbname = "MetisLB";
  // Read the options now, at startup. Converse removes an option from argv
  // when it is read, and reading +LBMetisStickiness lazily at the first
  // balance left its value in argv for the application to parse: leanmd took
  // the 0.5 as one of its positional arguments, zeroed its balancing period,
  // and died of a divide by zero in its step test (SIGFPE in Cell::_if_7).
  metisRemapOn();
  metisMinGain();
  metisStickiness();
  // A partitioner is only as good as its edges. Communication instrumentation
  // is off by default, and MetisLB run on its own had been partitioning a
  // graph with no edges at all (every cut reported 0.0): its output was an
  // arbitrary balanced permutation. DiffusionLB switches this on from its
  // own constructor; so does every other comm-aware balancer here.
  LBTurnCommOn();
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

// The least the step bound must fall, as a fraction of what it is now, for a
// new mapping to be applied at all. METIS never sees the current placement:
// it partitions from scratch and the relabel only renames parts, so against a
// balanced state it returns a different tiling of the same balance and moves
// nearly everything for nothing. Measured 12 Sep 2026 on sph2d once the dam
// break had spread (busiest PE 1.05x the mean): 105 of 128 patches moved for
// a 1.6% fall in the bound, and the run then paid ~20% for the stale mapping
// until the end. A fall below the floor is inside the balance METIS is
// allowed to miss by and inside round-to-round noise in the loads; it is
// DiffusionLB's +LBDiffusionMinImbalance applied to the whole partition, and
// defaults to the same value. +LBMetisMinGain overrides it.
static double metisMinGain()
{
  static const double frac = []() {
    double f = _lb_args.diffusionMinImbalance();
    CmiGetArgDoubleDesc(CkGetArgv(), "+LBMetisMinGain", &f,
                        "MetisLB: least fall in the step bound, as a fraction of the current "
                        "bound, for a new mapping to be applied (default: +LBDiffusionMinImbalance)");
    return f;
  }();
  return frac;
}

// What staying put is worth to an object, as a fraction of the traffic it
// has with its neighbours: an edge of that weight to the anchor of the part
// it is in now (see partitionSubset). METIS partitions from scratch, and a
// vertex whose neighbours are split between two parts -- a leanmd compute,
// with one cell on each side -- pays the same cut in either, so from scratch
// half of them change sides for nothing: 5845 of 7168 moved at a step that
// balanced device load 0.93 -> 0.64, where DiffusionLB reached the same
// balance moving ~1200. With the fraction set, a move has to save at least
// that much traffic to be made. 0 (the default) keeps the from-scratch
// partition; +LBMetisStickiness overrides.
static double metisStickiness()
{
  static const double frac = []() {
    double f = 0.0;
    CmiGetArgDoubleDesc(CkGetArgv(), "+LBMetisStickiness", &f,
                        "MetisLB: weight of an object's current placement, as a fraction of "
                        "its communication (default 0: partition from scratch)");
    return f;
  }();
  return frac;
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

// The greedy assignment both relabels share: overlap[part][label] is what
// giving `part` the name `label` is worth -- how many of its vertices carry
// that label now (relabelForOverlap), or how much traffic its vertices have
// with the fixed objects under that label (the anchored partition in work()).
static std::vector<int> relabelByWeight(const std::vector<std::vector<double>>& overlap,
                                        int nparts,
                                        const std::vector<std::vector<double>>& target)
{
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
  struct Pair { double n; int part, label; };
  std::vector<Pair> pairs;
  for (int p = 0; p < nparts; p++)
    for (int l = 0; l < nparts; l++)
      if (overlap[p][l] > 0.0 && sameShare(p, l)) pairs.push_back(Pair{overlap[p][l], p, l});
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

static std::vector<int> relabelForOverlap(const std::vector<idx_t>& partOf, int nparts,
                                          const std::vector<int>& current,
                                          const std::vector<std::vector<double>>& target)
{
  std::vector<std::vector<double>> overlap(nparts, std::vector<double>(nparts, 0.0));
  for (size_t k = 0; k < partOf.size(); k++)
    if (current[k] >= 0 && current[k] < nparts) overlap[partOf[k]][current[k]] += 1.0;
  return relabelByWeight(overlap, nparts, target);
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
             "%d GPUs, T_l %.6f over %d processes; alpha_h %.2f, alpha_g %.2f, alpha_l %.2f)\n",
             CkMyPe(), lbLoadDimName(ogr->deviceDim),
             lbLoadDimOverride() == LB_DIM_AUTO ? "criticality" : "flag", c.boundHost(),
             c.pes, c.boundDev(), c.gpus, c.boundDrv(), c.procs, c.alphaHost(), c.alphaGpu(),
             c.alphaDrv());
    lbPrintExplained("[MetisLB]", c);
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
                          : lbObjGroupLoad(stats->objData[i]);
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
  struct EdgeSum { long long bytes = 0, msgs = 0; };
  std::vector<std::map<int, EdgeSum>> adj(numVertices);
  for (int i = 0; i < numVertices; i++)
  {
    auto addEdge = [&](int nbr, int bytes, int msgs) {
      rawEdges++;
      if (nbr == i) { selfLoops++; return; }
      if (nbr < 0 || nbr >= numVertices) { outOfRange++; return; }
      const auto res = adj[i].emplace(nbr, EdgeSum());
      if (!res.second) duplicates++;
      if (bytes > 0) res.first->second.bytes += bytes;
      if (msgs > 0) res.first->second.msgs += msgs;
    };
    for (const auto& outEdge : ogr->vertices[i].sendToList)
      addEdge(outEdge.getNeighborId(), outEdge.getNumBytes(), outEdge.getNumMsgs());
    for (const auto& inEdge : ogr->vertices[i].recvFromList)
      addEdge(inEdge.getNeighborId(), inEdge.getNumBytes(), inEdge.getNumMsgs());
  }
  if (selfLoops || outOfRange || duplicates)
    CkPrintf("CharmLB> MetisLB: dropped %d self-loop(s) and %d out-of-range "
             "edge(s), merged %d duplicate edge(s) of %zu\n",
             selfLoops, outOfRange, duplicates, rawEdges);

  // What an edge costs if the partition cuts it. With the table DiffusionLB
  // prices moves from (+LBCostConfig, written by lbcalib), that is the
  // transfer cost per interval at the tier the cut lands on: alpha seconds
  // per message plus beta per byte. Without it, the byte count, as before.
  //
  // Bytes alone say a boundary through objects that exchange nothing is
  // free, and it is not: every edge cut costs a message a step whatever it
  // carries. On sph2d at 60k particles per patch, where three quarters of
  // the patches are empty and exchange zero-length halos, the byte-weighted
  // cut ran through the empty region -- a third of the bytes, on paper --
  // and the host work per step rose 40-55% from the messages that created,
  // taking the step from 4.4 to 6.9 ms while the balance it was asked for
  // improved. The per-message term is what makes that cut expensive.
  static DiffusionCostConfig costCfg;
  static bool costCfgTried = false;
  if (!costCfgTried)
  {
    costCfgTried = true;
    if (_lb_args.costConfig() != NULL) costCfg.load(_lb_args.costConfig());
  }
  auto edgeCost = [&](const EdgeSum& e, DiffusionTier t) -> double {
    if (!costCfg.calibrated) return (double)e.bytes;
    return costCfg.tier[t].alpha * (double)e.msgs + costCfg.tier[t].beta * (double)e.bytes;
  };

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
    // The measured load, not getVertexLoad(): that one floors every object at
    // 0.1 s, and a patch's device or launch time per interval is a few
    // milliseconds to 0.09 s, so the floor made every vertex the same weight
    // and level one balanced object counts, not the dimension it was asked to.
    // Measured 12 Sep 2026 on sph2d: every group-level partition came out 10
    // to 44% over the mean against a 10% tolerance, at both sizes.
    maxCross = std::max(maxCross, ogr->vertices[i].getCompLoad());
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
  // Footprints on the same 1..256 scale as the loads. Whole megabytes, which
  // this used to be, weigh every object under 1 MB the same -- a LeanMD
  // Compute holds 147 KB -- and the memory constraint was an object count.
  double memRatio = 0.0;
  if (memAware)
  {
    size_t maxFp = 0;
    for (int i = 0; i < numVertices; i++)
      maxFp = std::max(maxFp, memModel.footprint(ogr->vertices[i].getVertexId()));
    memRatio = (maxFp == 0) ? 0 : 256.0 / (double)maxFp;
  }
  auto memWeight = [&](int i) {
    return std::max((idx_t)1,
                    (idx_t)ceil(memModel.footprint(ogr->vertices[i].getVertexId()) * memRatio));
  };
#endif

  std::vector<idx_t> vwgtCross((size_t)numVertices * nConstraints);
  std::vector<idx_t> vwgtIntra(numVertices);
  for (int i = 0; i < numVertices; i++)
  {
    idx_t* w = &vwgtCross[(size_t)i * nConstraints];
    // Floored at 1 throughout: METIS needs a positive weight to balance on,
    // and ceil() of a zero load is zero.
    w[0] = std::max((idx_t)1, (idx_t)ceil(ogr->vertices[i].getCompLoad() * crossRatio));
#if CMK_CUDA
    if (memAware) w[1] = memWeight(i);
    if (slackIdx >= 0)
      w[slackIdx] = std::max((idx_t)1, (idx_t)ceil(slackOf(i) * slackRatio));
#endif
    vwgtIntra[i] =
        std::max((idx_t)1, (idx_t)ceil(stats->objData[i].wallTime * intraRatio));
  }

  std::array<idx_t, METIS_NOPTIONS> options;
  METIS_SetDefaultOptions(options.data());
  options[METIS_OPTION_NUMBERING] = 0;   // C style numbering
  // Experiment knobs: how many partitions METIS computes before keeping the
  // best cut, and how many refinement passes each gets (METIS defaults 1 and
  // 10). Environment, not argv: read at the first balance, an argv option
  // would still be there for the application to parse (see the constructor).
  // Measured on leanmd 8x8x8 with the anchors pinned and stickiness 0.5:
  // 1 / 4 / 8 partitions left 485 / 492 / 460 MB of cross-GPU traffic per
  // window against the block map's 243 (this file's own accounting, below),
  // and the steps after the balance at 229 / 235 / 242 ms -- no gain. Do not
  // compare those figures with DiffusionLB's per-neighbour bytes: those are
  // sender-side records keyed by the receiver's last-known PE, stale for an
  // object that just moved, and undercount the traffic to migrated objects.
  {
    static const int ncuts = getenv("CHARM_METIS_NCUTS") ? atoi(getenv("CHARM_METIS_NCUTS")) : 0;
    static const int niter = getenv("CHARM_METIS_NITER") ? atoi(getenv("CHARM_METIS_NITER")) : 0;
    if (ncuts > 0) options[METIS_OPTION_NCUTS] = ncuts;
    if (niter > 0) options[METIS_OPTION_NITER] = niter;
  }

  // Edges to fixed objects the last partitionSubset call kept (see below).
  long long lastFixedEdges = 0;

  // Partition the subgraph induced by `verts` into `nparts`, returning a part
  // number per entry of `verts`. Edges leaving the subset are dropped: at the
  // second level they are the traffic the first level already decided to pay.
  // `ubvecIn` gives each constraint its own tolerance; null means METIS's
  // usual 1.1 for every one. `tier` is what a cut edge is paid at: across
  // groups the transport between two GPUs, within a group the one between
  // two PEs of a process.
  //
  // Edges to FIXED objects are kept. A non-migratable object is not a vertex
  // of the subgraph, but it sits in one of the parts being cut and the
  // traffic between it and a migratable object is paid exactly as any other
  // cut edge is, unless the object lands in that part. Dropping those edges
  // -- what this did -- left leanmd's computes, which talk only to their
  // (non-migratable) cells, as a graph with no edges at all: METIS balanced
  // their device load and scattered them, every position and force message
  // crossed a process, and the step after the balance ran slower than with
  // no balancing (276 vs 250 ms, host and launch terms up by half).
  //
  // METIS has no fixed vertices, so the traffic is carried by one anchor
  // vertex per part: an edge from a migratable vertex to the anchor of part p
  // weighs what that vertex exchanges with the fixed objects of p. Anchors
  // carry the least weight METIS allows in the load constraints, so the
  // balance is unchanged, and METIS keeps each anchor with the vertices that
  // talk to it.
  //
  // An anchor must also be ALONE in its part, or the encoding says nothing:
  // METIS balances load, not cell membership, and on leanmd's gradient the
  // two light GPUs' computes together fit one part's share, so the cheapest
  // cut merged both anchors into one part and split the heavy GPUs' computes
  // over the other three -- the same device balance DiffusionLB reaches, but
  // with three to five times the moves and a slower step from the computes
  // that landed away from their cells. So the anchors get a balance
  // constraint of their own (see below) that allows exactly one per part, and
  // the anchor's part is then its label. Should METIS still fail to keep them
  // apart, the caller relabels the parts to the fixed objects their vertices
  // talk to most (fixedOverlap[part][label], the summed edge cost, filled in
  // only then). `fixedPartOf` says which part a vertex outside the subset is
  // fixed in, or -1 for one that is not fixed here (a migratable vertex of
  // another group at the second level, whose edge stays dropped as before).
  // `anchorsPinned` reports which of the two happened.
  //
  // The same anchors carry stickiness (metisStickiness): a vertex's edge to
  // the anchor of the part it is in now, `currentPartOf`, worth that fraction
  // of its traffic, so that staying is preferred over a move that saves
  // nothing. That term counts toward the relabel too, since it says where
  // the vertices came from.
  auto partitionSubset = [&](const std::vector<int>& verts, idx_t nparts,
                             const std::vector<real_t>* tpwgts,
                             const std::vector<idx_t>& weights,
                             int ncon_in,
                             const std::vector<real_t>* ubvecIn,
                             DiffusionTier tier,
                             const std::function<int(int)>& fixedPartOf,
                             const std::function<int(int)>& currentPartOf,
                             std::vector<std::vector<double>>* fixedOverlap,
                             bool* anchorsPinned) -> std::vector<idx_t>
  {
    const idx_t nv = (idx_t)verts.size();
    std::vector<idx_t> parts(verts.size(), 0);
    if (fixedOverlap) fixedOverlap->clear();
    lastFixedEdges = 0;
    if (nv == 0 || nparts <= 1) return parts;

    std::unordered_map<int, idx_t> local;
    local.reserve(verts.size() * 2);
    for (idx_t k = 0; k < nv; k++) local[verts[k]] = k;

    // Per vertex, the cost of its traffic with the fixed objects of each
    // part, then its stickiness to the part it is in now.
    std::vector<std::map<int, double>> fixedAff(nv);
    long long fixedEdges = 0;
    const double stick = metisStickiness();
    for (idx_t k = 0; k < nv; k++)
    {
      double total = 0.0;
      for (const auto& nb : adj[verts[k]])
      {
        const double c = edgeCost(nb.second, tier);
        if (local.find(nb.first) != local.end()) { total += c; continue; }
        const int p = fixedPartOf(nb.first);
        if (p < 0 || p >= (int)nparts) continue;
        fixedAff[k][p] += c;
        total += c;
        fixedEdges++;
      }
      if (stick > 0.0 && total > 0.0)
      {
        const int cur = currentPartOf(verts[k]);
        if (cur >= 0 && cur < (int)nparts) fixedAff[k][cur] += stick * total;
      }
    }
    lastFixedEdges = fixedEdges;
    bool anchored = false;
    for (idx_t k = 0; k < nv && !anchored; k++) anchored = !fixedAff[k].empty();
    const idx_t nAll = nv + (anchored ? nparts : 0);
    // With anchors, one more balance constraint: each anchor weighs 1 in it
    // and every real vertex 0, every part is owed 1/nparts of it, and the
    // tolerance 1 + 1/nparts is below what a second anchor in any part -- or
    // in either half of any bisection on the way there -- would cost. So
    // every part holds exactly one anchor.
    const int nconAll = anchored ? ncon_in + 1 : ncon_in;

    std::vector<std::vector<std::pair<idx_t, double>>> anchorEdges(anchored ? nparts : 0);
    std::vector<idx_t> xadj(nAll + 1), adjncy, adjwgt;
    std::vector<double> cost;
    std::vector<idx_t> lvwgt((size_t)nAll * nconAll, 0);
    idx_t e = 0;
    for (idx_t k = 0; k < nv; k++)
    {
      xadj[k] = e;
      for (int c = 0; c < ncon_in; c++)
        lvwgt[(size_t)k * nconAll + c] = weights[(size_t)verts[k] * ncon_in + c];
      for (const auto& nb : adj[verts[k]])
      {
        const auto it = local.find(nb.first);
        if (it == local.end()) continue;
        adjncy.push_back(it->second);
        cost.push_back(edgeCost(nb.second, tier));
        e++;
      }
      for (const auto& fa : fixedAff[k])
      {
        const double c = fa.second;
        adjncy.push_back(nv + (idx_t)fa.first);
        cost.push_back(c);
        e++;
        anchorEdges[fa.first].push_back(std::make_pair(k, c));
      }
    }
    // The anchors: the least weight METIS allows in the load constraints, 1
    // in their own, and their own edge lists, since METIS wants every edge
    // from both ends.
    for (idx_t p = 0; p < (anchored ? nparts : 0); p++)
    {
      for (int c = 0; c < nconAll; c++) lvwgt[(size_t)(nv + p) * nconAll + c] = 1;
      xadj[nv + p] = e;
      for (const auto& ae : anchorEdges[p])
      {
        adjncy.push_back(ae.first);
        cost.push_back(ae.second);
        e++;
      }
    }
    xadj[nAll] = e;
    // Positive and in range: idx_t is 32 bits in a stock METIS build. Without
    // a table the weight is the byte count summed over the instrumented
    // window, capped; with one it is seconds per interval, scaled so the
    // heaviest edge of this subset is 2^20. Both endpoints sum the same
    // records, so the weight stays symmetric either way, which METIS
    // requires.
    double maxCost = 0.0;
    for (double c : cost) maxCost = std::max(maxCost, c);
    const double scale =
        (costCfg.calibrated && maxCost > 0.0) ? (double)(1 << 20) / maxCost : 1.0;
    adjwgt.reserve(cost.size());
    for (double c : cost)
      adjwgt.push_back((idx_t)std::min<long long>(
          std::max<long long>((long long)llround(c * scale), 1),
          (long long)std::numeric_limits<idx_t>::max()));
    // METIS reads the zeroth element even when there are no edges.
    if (adjncy.empty()) { adjncy.push_back(0); adjwgt.push_back(1); }

    idx_t ncon = nconAll, nv_arg = nAll, np = nparts, edgecut = 0;
    std::vector<real_t> ubvec(nconAll, (real_t)1.1);
    if (ubvecIn != nullptr)
      for (int c = 0; c < ncon_in && c < (int)ubvecIn->size(); c++) ubvec[c] = (*ubvecIn)[c];
    std::vector<real_t> tpwgtsAll;
    if (anchored)
    {
      ubvec[ncon_in] = (real_t)(1.0 + 1.0 / (double)nparts);
      tpwgtsAll.assign((size_t)nparts * nconAll, (real_t)(1.0 / (double)nparts));
      if (tpwgts != nullptr)
        for (idx_t p = 0; p < nparts; p++)
          for (int c = 0; c < ncon_in; c++)
            tpwgtsAll[(size_t)p * nconAll + c] = (*tpwgts)[(size_t)p * ncon_in + c];
    }
    std::vector<idx_t> partsAll(nAll, 0);
    METIS_PartGraphRecursive(&nv_arg, &ncon, xadj.data(), adjncy.data(), lvwgt.data(),
                             nullptr, adjwgt.data(), &np,
                             anchored ? tpwgtsAll.data()
                                      : (tpwgts ? const_cast<real_t*>(tpwgts->data()) : nullptr),
                             ubvec.data(), options.data(), &edgecut, partsAll.data());
    std::copy(partsAll.begin(), partsAll.begin() + nv, parts.begin());
    if (anchorsPinned) *anchorsPinned = false;
    if (anchored)
    {
      // Did METIS keep the anchors apart? Then part q is named after the
      // anchor it holds. Otherwise the caller names the parts after the fixed
      // objects their vertices talk to most, the best that is left.
      std::vector<int> label(nparts, -1);
      bool distinct = true;
      for (idx_t p = 0; p < nparts && distinct; p++)
      {
        const idx_t q = partsAll[nv + p];
        if (q < 0 || q >= nparts || label[q] >= 0) distinct = false;
        else label[q] = (int)p;
      }
      if (distinct)
      {
        for (idx_t k = 0; k < nv; k++) parts[k] = (idx_t)label[partsAll[k]];
        if (anchorsPinned) *anchorsPinned = true;
      }
      else if (fixedOverlap)
      {
        fixedOverlap->assign(nparts, std::vector<double>(nparts, 0.0));
        for (idx_t k = 0; k < nv; k++)
          for (const auto& fa : fixedAff[k])
            (*fixedOverlap)[partsAll[k]][fa.first] += fa.second;
      }
    }
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
    fixedCross[groupOfPe[pe]] += ogr->vertices[i].getCompLoad() * crossRatio;
    fixedCpu[pe] += stats->objData[i].wallTime * intraRatio;
#if CMK_CUDA
    if (memAware) fixedMem[groupOfPe[pe]] += (double)memWeight(i);
    if (slackIdx >= 0) fixedSlack[groupOfPe[pe]] += slackOf(i) * slackRatio;
#endif
  }

  // What each group can hold in the memory dimension, in its weight units: the
  // footprints already on it plus the room the contract plans against, H_g.
  // Its share of the whole is its share of the target, so a device with less
  // room is asked to hold fewer bytes. The tolerance is where a group would
  // actually overflow: with room to spare the constraint gets out of the load
  // dimension's way, and it tightens toward METIS's usual 1.1 as memory fills.
  // The verifier still repairs whatever residue the tolerance leaves.
  std::vector<double> memCap(nGroups, 0.0);
  double memUbvec = 1.1;
#if CMK_CUDA
  if (memAware)
  {
    MemoryLedger memLedger;
    memLedger.init(&memModel, 0.95, /*waveStaging=*/false);
    for (int i = 0; i < numVertices; i++)
    {
      const int pe = ogr->vertices[i].getCurrentPe();
      if (pe >= 0 && pe < (int)groupOfPe.size() && groupOfPe[pe] >= 0)
        memCap[groupOfPe[pe]] += (double)memWeight(i);
    }
    for (idx_t g = 0; g < nGroups; g++)
    {
      const int d = memModel.deviceIndexOf(stats->procs[grpPes[g][0]].gpu_device_id);
      if (d >= 0) memCap[g] += (double)memLedger.memAvailOn(d) * memRatio;
    }
  }
#endif

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
    const double memCapAll = std::accumulate(memCap.begin(), memCap.end(), 0.0);
    double memHeadroom = std::numeric_limits<double>::max();
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
        const double memShare = memCapAll > 0.0 ? memCap[g] / memCapAll : share;
        wg[1] = std::max(1e-6, allMem * memShare - fixedMem[g]);
        sum[1] += wg[1];
        // How far over its target group g's migratable bytes may run before
        // the group overflows.
        if (memCapAll > 0.0)
          memHeadroom = std::min(memHeadroom, (memCap[g] - fixedMem[g]) / wg[1]);
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
    // tpwgts are renormalised over the migratable total, which the want[]
    // above already sums to unless a group's share was clamped; the margin
    // absorbs that and METIS's own slop.
    if (memAware && memHeadroom < std::numeric_limits<double>::max())
      memUbvec = std::max(1.1, 0.9 * memHeadroom);
    if (memAware && _lb_args.debug() > 0 && CkMyPe() == cur_ld_balancer)
      CkPrintf("[%d] MetisLB: memory constraint tolerance %.2f (fill %.3f of what the "
               "groups can hold)\n",
               CkMyPe(), memUbvec, memCapAll > 0.0 ? allMem / memCapAll : 0.0);
  }

  CkPrintf("Metis partitioning %d migratable of %d objects over %d GPU group(s) "
           "of %d PEs\n", (int)migVerts.size(), (int)numVertices, (int)nGroups,
           parr->availProcSize);

  // The tier a cross-group edge is paid at. Groups are GPUs: two on one host
  // talk over CUDA IPC, two on different hosts over the network. METIS takes
  // one weight per edge, not one per pair of parts, so if any two groups sit
  // on different hosts every cross-group edge is priced at the network tier
  // -- the over-priced direction, which keeps objects where they are rather
  // than moving them onto a boundary the price understated.
  DiffusionTier crossTier = DIFF_TIER_IPC_CROSS_GPU;
  for (idx_t g = 1; g < nGroups; g++)
    if (!grpPes[g].empty() && !grpPes[0].empty() &&
        !CmiPeOnSamePhysicalNode(grpPes[0][0], grpPes[g][0]))
      crossTier = DIFF_TIER_INTER_NODE;

  std::vector<real_t> ubvecCross(nConstraints, (real_t)1.1);
  if (memAware) ubvecCross[1] = (real_t)memUbvec;
  if (slackIdx >= 0) ubvecCross[slackIdx] = (real_t)slackUbvec;
  // A fixed object's group, for the edges to it.
  auto fixedGroupOf = [&](int v) -> int {
    if (ogr->vertices[v].isMigratable()) return -1;
    const int pe = ogr->vertices[v].getCurrentPe();
    return (pe >= 0 && pe < (int)groupOfPe.size()) ? groupOfPe[pe] : -1;
  };
  auto currentGroupOf = [&](int v) -> int {
    const int pe = ogr->vertices[v].getCurrentPe();
    return (pe >= 0 && pe < (int)groupOfPe.size()) ? groupOfPe[pe] : -1;
  };
  std::vector<std::vector<double>> fixedOverlap;
  bool anchorsPinned = false;
  const std::vector<idx_t> groupOf = partitionSubset(
      migVerts, nGroups, nGroups > 1 ? &tpwgts : nullptr, vwgtCross, nConstraints, &ubvecCross,
      crossTier, fixedGroupOf, currentGroupOf, &fixedOverlap, &anchorsPinned);
  const long long fixedEdgesLevelOne = lastFixedEdges;

  // Level one's labels. With edges to fixed objects the part numbers are
  // relabelled to the groups whose fixed objects the parts talk to most:
  // that is what makes the anchors mean anything, so it is not optional.
  // Otherwise scratch-remap: the group each part becomes is the group its
  // objects mostly come from, where the shares allow.
  std::vector<idx_t> groupLabel = groupOf;
  int relabelledGroups = 0, relabelledPes = 0;
  if (nGroups > 1 && !anchorsPinned && (metisRemapOn() || !fixedOverlap.empty()))
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
    const std::vector<int> sigma =
        fixedOverlap.empty() ? relabelForOverlap(groupOf, (int)nGroups, cur, shares)
                             : relabelByWeight(fixedOverlap, (int)nGroups, shares);
    for (idx_t g = 0; g < nGroups; g++)
      if (sigma[g] != (int)g) relabelledGroups++;
    for (size_t k = 0; k < migVerts.size(); k++) groupLabel[k] = sigma[groupOf[k]];
  }
  if (_lb_args.debug() > 0 && CkMyPe() == cur_ld_balancer && fixedEdgesLevelOne > 0)
    CkPrintf("[%d] MetisLB level one: %lld edge(s) to fixed objects kept through %d anchor(s), "
             "%s\n",
             CkMyPe(), fixedEdgesLevelOne, (int)nGroups,
             anchorsPinned ? "each pinned to its own part"
                           : "NOT kept apart by METIS; parts relabelled to the fixed objects "
                             "they talk to");

  // What level one actually achieved in the dimension it balanced, group by
  // group, against where the objects came from. METIS's tolerance is a
  // request; with several constraints and heavy edges it is not always met,
  // and a partition that leaves one group far over the mean is worth seeing
  // before the levels below build on it.
  if (_lb_args.debug() > 1 && CkMyPe() == cur_ld_balancer && nGroups > 1)
  {
    std::vector<double> was(nGroups, 0.0), now(nGroups, 0.0);
    std::vector<double> loads;
    int zeros = 0;
    for (size_t k = 0; k < migVerts.size(); k++)
    {
      const double l = ogr->vertices[migVerts[k]].getCompLoad();
      const int pe = ogr->vertices[migVerts[k]].getCurrentPe();
      if (pe >= 0 && pe < (int)groupOfPe.size() && groupOfPe[pe] >= 0) was[groupOfPe[pe]] += l;
      now[groupLabel[k]] += l;
      loads.push_back(l);
      if (l <= 0.0) zeros++;
    }
    std::sort(loads.begin(), loads.end(), std::greater<double>());
    std::string line;
    char buf[96];
    for (idx_t g = 0; g < nGroups; g++)
    {
      snprintf(buf, sizeof buf, " g%d %.4f->%.4f", (int)g, was[g], now[g]);
      line += buf;
    }
    CkPrintf("[%d] MetisLB level one (%s):%s; %d of %zu objects carry none; largest %.4f %.4f %.4f %.4f %.4f\n",
             CkMyPe(), lbLoadDimName(ogr->deviceDim), line.c_str(), zeros, loads.size(),
             loads.size() > 0 ? loads[0] : 0.0, loads.size() > 1 ? loads[1] : 0.0,
             loads.size() > 2 ? loads[2] : 0.0, loads.size() > 3 ? loads[3] : 0.0,
             loads.size() > 4 ? loads[4] : 0.0);
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

    // A fixed object's PE within this group, for the edges to it; one in
    // another group is not this level's to place around.
    auto fixedPeIdxOf = [&](int v) -> int {
      if (ogr->vertices[v].isMigratable()) return -1;
      const int pe = ogr->vertices[v].getCurrentPe();
      for (size_t j = 0; j < pes.size(); j++)
        if (pes[j] == pe) return (int)j;
      return -1;
    };
    auto currentPeIdxOf = [&](int v) -> int {
      const int pe = ogr->vertices[v].getCurrentPe();
      for (size_t j = 0; j < pes.size(); j++)
        if (pes[j] == pe) return (int)j;
      return -1;  // arriving from another group: no place here to keep
    };
    std::vector<std::vector<double>> fixedOverlap2;
    bool anchorsPinned2 = false;
    const std::vector<idx_t> peOf = partitionSubset(
        grpVerts[g], (idx_t)pes.size(), pes.size() > 1 ? &tpwgts2 : nullptr,
        vwgtIntra, 1, nullptr, DIFF_TIER_INTRA_PROCESS, fixedPeIdxOf, currentPeIdxOf,
        &fixedOverlap2, &anchorsPinned2);
    if (_lb_args.debug() > 1 && CkMyPe() == cur_ld_balancer && lastFixedEdges > 0)
      CkPrintf("[%d] MetisLB level two, group %d: %lld edge(s) to fixed objects, anchors %s\n",
               CkMyPe(), (int)g, lastFixedEdges, anchorsPinned2 ? "pinned" : "NOT kept apart");

    // Level two's labels: to the PEs whose fixed objects the parts talk to
    // most when there are such edges, else scratch-remap to the PE each
    // part's objects mostly come from, where the shares allow.
    std::vector<idx_t> peLabel = peOf;
    if (pes.size() > 1 && !anchorsPinned2 && (metisRemapOn() || !fixedOverlap2.empty()))
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
      const std::vector<int> sigma =
          fixedOverlap2.empty() ? relabelForOverlap(peOf, (int)pes.size(), cur, shares)
                                : relabelByWeight(fixedOverlap2, (int)pes.size(), shares);
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
    auto devOf = [&](int i) { return lbObjGroupLoad(stats->objData[i]); };
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

  // ---- is the new mapping worth what it costs? ----------------------------
  //
  // METIS returns the least cut it can find subject to balance, and until now
  // whatever it returned was applied. That is the right rule only when moving
  // is free. With a table it is priced: a mapping is applied when the step
  // time it saves per interval exceeds what it costs per interval -- the
  // communication its cut adds over the current mapping's, plus the
  // migrations, amortised over the intervals the placement is expected to
  // last. It is the trade DiffusionLB's metric makes per move, made once here
  // for the whole partition. The saving is the fall in the larger of the two
  // bounds, the busiest GPU's device time and the busiest PE's host time,
  // which is the most the step can shorten by; the loads say nothing about
  // the rest of it. Without a table nothing is priced, only the floor below
  // applies. The memory contract is no reason to skip this: the
  // current mapping is where the objects already reside, so keeping it
  // cannot exceed a capacity they already occupy.
  //
  // sph2d at 60k particles per patch is the case this is for: a 1.55 host
  // imbalance, a partition that cuts the priced graph no better than the
  // block map it replaces (1.84 -> 1.88 s per interval) and moves 82 of 128
  // patches to bring the busiest PE down 13%, and a step that came out no
  // faster for it.
  //
  // Before any of that, and with or without a table: the bound must fall by
  // at least metisMinGain() of itself, or the current mapping is kept.
  {
    const int nPe = (int)parr->procs.size();
    auto peUnder = [&](int i, bool fresh) {
      const int cur = ogr->vertices[i].getCurrentPe();
      return (fresh && newPe[i] >= 0) ? newPe[i] : cur;
    };
    const DiffusionCostModel model(costCfg, DIFF_TIER_INTRA_PROCESS);
    // The step bound under a mapping: busiest GPU or busiest PE, host time
    // with the PE's background and its speed, as the levels above count them,
    // and the communication and migration the mapping puts on that same PE.
    //
    // The last part is the whole point. A bound is a MAXIMUM over PEs; a cut
    // is a SUM over the job. Priced against each other directly -- which is
    // what this did -- the cut arrives inflated by something close to the
    // number of PEs, and a partition that took the busiest PE down by a fifth
    // was refused for a communication cost that no single PE ever pays. On
    // sph2d with a load in motion that was every decision, twenty out of
    // twenty, each one leaving a measured 16-33% on the table. So the cut is
    // charged where it lands instead: a cut edge costs both of its endpoints,
    // each paying for its half of the exchange, and the comparison is then
    // one bound against another with nothing left outside it.
    auto bound = [&](bool fresh) {
      std::vector<double> grpDev(nGroups, 0.0), peHost(nPe, 0.0);
      for (int pe = 0; pe < nPe; pe++)
        if (groupOfPe[pe] >= 0) peHost[pe] = stats->procs[pe].bg_walltime;
      for (int i = 0; i < numVertices; i++)
      {
        const int pe = peUnder(i, fresh);
        if (pe < 0 || pe >= nPe || groupOfPe[pe] < 0) continue;
        grpDev[groupOfPe[pe]] += lbObjGroupLoad(stats->objData[i]);
        peHost[pe] += (double)stats->objData[i].wallTime / peSpeed[pe];
      }
      if (costCfg.calibrated)
      {
        for (int i = 0; i < numVertices; i++)
        {
          const int p = peUnder(i, fresh);
          if (p < 0 || p >= nPe || groupOfPe[p] < 0) continue;
          for (const auto& nb : adj[i])
          {
            if (nb.first <= i) continue;
            const int q = peUnder(nb.first, fresh);
            if (q < 0 || q >= nPe || groupOfPe[q] < 0 || p == q) continue;
            const double e = edgeCost(nb.second, DiffusionCostConfig::tierBetween(p, q));
            peHost[p] += e / peSpeed[p];
            peHost[q] += e / peSpeed[q];
          }
        }
        // What the move itself costs the PE that packs it, spread over the
        // intervals the placement is expected to last. Only the new mapping
        // pays it: staying put moves nothing.
        if (fresh && costCfg.placementLifetimeIntervals > 0.0)
          for (int i = 0; i < numVertices; i++)
          {
            const int cur = ogr->vertices[i].getCurrentPe();
            if (newPe[i] < 0 || newPe[i] == cur) continue;
            if (cur < 0 || cur >= nPe || groupOfPe[cur] < 0) continue;
            peHost[cur] += model.migrateCost(stats->objData[i]) /
                           (costCfg.placementLifetimeIntervals * peSpeed[cur]);
          }
      }
      double b = 0.0;
      for (idx_t g = 0; g < nGroups; g++) b = std::max(b, grpDev[g]);
      for (int pe = 0; pe < nPe; pe++) b = std::max(b, peHost[pe]);
      return b;
    };
    // What a mapping's cut costs per interval: each cut edge at the tier its
    // endpoints' PEs make -- nothing on one PE, the in-process tier across
    // the PEs of one process, IPC or the network beyond.
    auto cutCost = [&](bool fresh) {
      double c = 0.0;
      for (int i = 0; i < numVertices; i++)
      {
        const int p = peUnder(i, fresh);
        for (const auto& nb : adj[i])
        {
          if (nb.first <= i) continue;
          const int q = peUnder(nb.first, fresh);
          if (p < 0 || q < 0 || p == q) continue;
          c += edgeCost(nb.second, DiffusionCostConfig::tierBetween(p, q));
        }
      }
      return c;
    };
    double migration = 0.0;
    int moving = 0;
    for (int i = 0; i < numVertices; i++)
      if (newPe[i] >= 0 && newPe[i] != ogr->vertices[i].getCurrentPe())
      {
        if (costCfg.calibrated) migration += model.migrateCost(stats->objData[i]);
        moving++;
      }
    // Both bounds already carry the communication and the migration each PE
    // bears, so this is the whole trade: the busiest resource under the new
    // mapping against the busiest under the current one. What remains is the
    // floor, which keeps the current mapping when the difference is too small
    // to be worth disturbing.
    const double boundOld = bound(false), boundNew = bound(true);
    const double gain = boundOld - boundNew;
    const double minGain = metisMinGain() * boundOld;
    const bool worth = moving > 0 && gain >= minGain;
    const char* verdict = moving == 0 ? "nothing moves"
                          : !worth    ? "below the floor, current mapping kept"
                                      : "applied";
    // Reported, not priced: the job-wide cut says how much traffic the
    // partition makes, which is worth seeing, but it is a sum and the bounds
    // above are maxima, so it is not a quantity either of them can be
    // compared against.
    double cutOld = 0.0, cutNew = 0.0;
    if (costCfg.calibrated)
    {
      cutOld = cutCost(false);
      cutNew = cutCost(true);
    }
    if ((_lb_args.debug() > 0 || !worth) && CkMyPe() == cur_ld_balancer)
    {
      if (costCfg.calibrated)
        CkPrintf("[%d] MetisLB priced: step bound %.6f -> %.6f, comm and migration included "
                 "(saves %.6f s/interval, %.1f%% against a floor of %.1f%%); job cut %.6f -> %.6f; "
                 "%d migration(s) %.6f s/interval over %.0f interval(s); %s\n",
                 CkMyPe(), boundOld, boundNew, gain,
                 boundOld > 0.0 ? 100.0 * gain / boundOld : 0.0, 100.0 * metisMinGain(),
                 cutOld, cutNew, moving, migration, costCfg.placementLifetimeIntervals, verdict);
      else
        CkPrintf("[%d] MetisLB gate: step bound %.6f -> %.6f (saves %.6f s/interval, %.1f%% "
                 "against a floor of %.1f%%); %d migration(s); %s\n",
                 CkMyPe(), boundOld, boundNew, gain,
                 boundOld > 0.0 ? 100.0 * gain / boundOld : 0.0, 100.0 * metisMinGain(),
                 moving, verdict);
    }
    if (!worth) std::fill(newPe.begin(), newPe.end(), -1);
  }

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
    long long edges = 0;
    double wTotal = 0.0, cutOld = 0.0, cutNew = 0.0;
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
        const double w = edgeCost(nb.second, crossTier);
        wTotal += w;
        const int gOldI = (oldPeI >= 0 && oldPeI < (int)groupOfPe.size()) ? groupOfPe[oldPeI] : -1;
        const int gOldJ = (oldPeJ >= 0 && oldPeJ < (int)groupOfPe.size()) ? groupOfPe[oldPeJ] : -1;
        const int gNewI = (newPeI >= 0 && newPeI < (int)groupOfPe.size()) ? groupOfPe[newPeI] : -1;
        const int gNewJ = (newPeJ >= 0 && newPeJ < (int)groupOfPe.size()) ? groupOfPe[newPeJ] : -1;
        if (gOldI != gOldJ) cutOld += w;
        if (gNewI != gNewJ) cutNew += w;
      }
    }
    int moved = 0;
    for (int i = 0; i < numVertices; i++)
      if (newPe[i] >= 0 && newPe[i] != ogr->vertices[i].getCurrentPe()) moved++;
    CkPrintf("[%d] MetisLB locality: %lld edge(s) over %d objects, total weight "
             "%.6g %s; cross-group weight before=%.6g after=%.6g; %d object(s) moved\n",
             CkMyPe(), edges, (int)numVertices, wTotal,
             costCfg.calibrated ? "s/interval at the cross-group tier" : "bytes", cutOld,
             cutNew, moved);
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
      grpDev[groupOfPe[pe]] += ogr->vertices[i].getCompLoad();
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
