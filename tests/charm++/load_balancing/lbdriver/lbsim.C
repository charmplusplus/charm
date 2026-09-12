/**
 * lbsim -- DiffusionLB on N virtual nodes inside one process.
 *
 * The balancer's nodes are real PEs: with +p4 it has at most four of them, and
 * there is no offline replay for a distributed balancer the way +LBSim replays
 * a central one. This program gets around that by running the balancer's own
 * decision code on N nodes that exist only in memory. It links the DiffusionLB
 * module and calls the same functions the chare calls:
 *
 *   diffusionRoundFlows   (DiffusionFlow.h)    the pseudo-LB round
 *   MetricComm            (DiffusionMetric.h)  object selection and re-scoring
 *   diffusionSelectMoves  (DiffusionSelect.h)  the across-node selection loop
 *   DiffusionCostConfig   (DiffusionCostModel.h) the +LBCostConfig table
 *
 * so what it reports is what the balancer decides, not a re-implementation of
 * it. What IS simulated here, because in the balancer it is a message protocol
 * and not a function:
 *
 *   - stats assembly: each virtual node gets an LDStats holding exactly the
 *     records the runtime would have given it (its objects, and every send
 *     from one of them, to local and remote partners alike);
 *   - the neighbour graph: the ring backbone plus the ask/okay/ack handshake
 *     of DiffusionNeighbors.C, with the same rules and round count, processed
 *     one ask at a time in node order. The real handshake interleaves asks
 *     from every node, so it can settle on a different graph; this is one
 *     admissible serialisation of it, and a deterministic one;
 *   - the pseudo rounds' message exchange, run in lockstep, which is what the
 *     SDAG loop enforces anyway;
 *   - the receiver side of a move: with one PE per node there is no within-
 *     node phase, so an object lands on the node it was sent to.
 *
 * The input is the lbdriver stencil: an nx x ny grid of chares, four-point
 * ghost exchange of ghostBytes per iteration, weights flat for the initial
 * partition and then a 12x hot disc off-centre. The initial partition is METIS
 * on the same graph and weights MetisLB would hand it (minus the background
 * load, which does not exist here). Then `steps` DiffusionLB steps run, each on
 * the mapping the previous one produced.
 *
 *   lbsim nx ny nodes steps [ghostBytes] [iters] [initial] [hot]
 *
 *   initial   metis (default) or block
 *   hot       the disc's weight relative to the rest (default 12)
 *
 * Two builds: `make lbsim` is the Charm++ program (needs a compute node on
 * this machine, since the runtime's startup wants a GPU and the NIC);
 * `make lbsim-standalone` is the same simulation as a plain program that
 * runs on a login node, with the runtime stubbed by lbsim_stub.C. Same
 * arguments, same +LB flags, same decisions -- verified move for move.
 *
 * DiffusionLB's own flags apply, since it reads them from the same place:
 * +LBDiffusionNumNbors, +LBDiffusionMaxMoveFrac, +LBDiffusionMinImbalance,
 * +LBDiffusionBeta, +LBCostConfig, +LBnoMST, +LBDebug, and the environment
 * knobs CHARM_LB_DIFFUSION_ITERS, CHARM_LB_DIFFUSION_CONVERGE,
 * CHARM_DIFFUSION_GRAPH_REBUILD. +LBDiffusionCommOn is assumed: there are no
 * object positions here, so only the communication metric applies.
 *
 * Writes lbsim.json in lbdriver.json's format (plot_map.py, analyze.py read
 * both) and prints a one-line summary per step. Runs on one PE.
 *
 * Experiment knobs, all environment variables, none of which the chare has.
 * They exist to measure a design before deciding whether the chare gets it;
 * the measurements that decided each are in the comments where it is used.
 *
 *   LBSIM_MODE=route   plan with relaying, carry tokens with the rounds, heal
 *                      and smooth, migrate once (see the route section)
 *   LBSIM_MODE=remap   scratch-remap: METIS on the current weights, relabel
 *                      parts to the owners they overlap most, migrate once
 *   LBSIM_SHED=plan    across-node phase sheds the plan's outflow, not the
 *                      node's own excess
 *   LBSIM_PEEL=layers  chunks grow by layers from the destination boundary
 *   LBSIM_REFINE=1     heal detached pieces and smooth boundaries after the
 *                      balancer's own moves
 *   LBSIM_LOCAL_FLOOR  per-edge flow floor for route mode and plan shedding
 *                      (default 0.01; the chare uses +LBDiffusionMinImbalance)
 *   LBSIM_REMAP_ABOVE  the chare's +LBDiffusionRemapAbove: after the rounds,
 *                      hand the step to scratch-remap when the one-hop plan
 *                      leaves max/avg above this; the same decision function
 *   LBSIM_REFINE_PASSES, LBSIM_PATIENCE, LBSIM_TRACE   tuning and tracing
 *   LBSIM_VECTOR=host|device|cross
 *                      a second load dimension per object (device time beside
 *                      host time; see Grid::weightsVector). The step then
 *                      decides which dimension binds, diffuses that one, and
 *                      bounds what a receiver takes in the other -- the same
 *                      code the chare runs (LBLoadDim.h, DiffusionMetric.h).
 *                      LBSIM_HOT2 sets the device disc's hot factor in cross
 *                      mode (default: the host disc's).
 */

// Two builds of this file. The Charm++ one (default) is a chare program run
// with charmrun/srun and needs the runtime it links -- which on this
// machine calls cuInit at startup and so needs a compute node. The
// standalone one (-DLBSIM_STANDALONE, `make lbsim-standalone`) is a plain
// program: it compiles against the same Charm headers, links METIS and the
// decision code directly, and takes the handful of runtime symbols the
// decision code touches from lbsim_stub.C. It runs anywhere, the login node
// included. Same decisions either way; only the entry point and the +LB
// flag parsing differ.
#ifndef LBSIM_STANDALONE
#include "lbsim.decl.h"
#endif

#include "DiffusionCostModel.h"
#include "DiffusionFlow.h"
#include "DiffusionMetric.h"
#include "DiffusionSelect.h"

#include <metis.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <string>
#include <vector>

// Same weights as lbdriver, so the two are comparable. The hot factor is the
// eighth argument (default 12), so the size of the perturbation can be swept.
static const double kBaseWork = 1.0;
static double kHotWork = 12.0;
// Neighbour-finding rounds: ROUNDS in DiffusionNeighbors.C.
static const int kRounds = 20;

struct Grid
{
  int nx, ny;
  int iters;
  int ghostBytes;
  std::vector<double> w;  // per object, index i * ny + j

  int n() const { return nx * ny; }
  int at(int i, int j) const { return i * ny + j; }

  // The four stencil partners of a cell, in the order lbdriver sends to them.
  void partners(int k, std::vector<int>& out) const
  {
    out.clear();
    const int i = k / ny, j = k % ny;
    if (i > 0) out.push_back(at(i - 1, j));
    if (i < nx - 1) out.push_back(at(i + 1, j));
    if (j > 0) out.push_back(at(i, j - 1));
    if (j < ny - 1) out.push_back(at(i, j + 1));
  }

  void weightsUniform()
  {
    w.assign(n(), kBaseWork);
  }

  void weightsHotSpot()
  {
    w.resize(n());
    const double cx = nx * 0.30, cy = ny * 0.30;
    const double radius = 0.22 * (nx < ny ? nx : ny);
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++)
      {
        const double r = std::sqrt((i - cx) * (i - cx) + (j - cy) * (j - cy));
        w[at(i, j)] = (r <= radius) ? kHotWork : kBaseWork;
      }
  }

  // A second load dimension (LBSIM_VECTOR): device time per object in w2,
  // beside the host time in w. The balancer then decides which of the two
  // binds (LBLoadDim.h) and diffuses that one, and the receiver check bounds
  // what a neighbour takes in the other.
  //   host    the hot disc is host time, device time is flat
  //   device  the hot disc is device time, host time is flat
  //   cross   two discs: host at (0.30, 0.30), device at (0.70, 0.70) --
  //           both dimensions loaded, neither a copy of the other, which is
  //           the comparable-alpha regime the receiver check exists for.
  // A second hot factor for the device disc comes from LBSIM_HOT2 (default:
  // the same as the host disc's), so which dimension binds can be swept.
  std::vector<double> w2;
  void weightsVector(const std::string& mode)
  {
    weightsHotSpot();
    w2.assign(n(), kBaseWork);
    if (mode == "device")
    {
      w2 = w;
      w.assign(n(), kBaseWork);
    }
    else if (mode == "cross")
    {
      const char* hot2Env = getenv("LBSIM_HOT2");
      const double hot2 = hot2Env ? atof(hot2Env) : kHotWork;
      const double cx = nx * 0.70, cy = ny * 0.70;
      const double radius = 0.22 * (nx < ny ? nx : ny);
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
        {
          const double r = std::sqrt((i - cx) * (i - cx) + (j - cy) * (j - cy));
          if (r <= radius) w2[at(i, j)] = hot2;
        }
    }
    else if (mode != "host")
      CkAbort("lbsim: LBSIM_VECTOR must be host, device or cross\n");
  }
};

// Whether a second dimension is in play this run (LBSIM_VECTOR given).
static bool gVectorMode = false;

// One DiffusionLB node, as far as the decision code can tell. The fields mirror
// the chare's, named the same where they are the same thing.
struct VNode
{
  int id = 0;
  std::vector<int> objs;  // global object ids, in objData order
  BaseLB::LDStats* st = NULL;
  double my_load = 0.0;
  // Both dimensions' totals (nodeHostSum, nodeDevSum in the chare). One PE
  // per virtual node, so host per PE is the host total.
  double hostSum = 0.0, hostMax = 0.0, devSum = 0.0, devMax = 0.0;

  // The neighbour graph. Survives steps, like hs_graphCached.
  std::vector<int> sendToNeighbors;
  std::vector<long> bytesToNode;  // cost_for_neighbor, indexed by node
  std::vector<int> node_idx;      // preference order for the handshake
  int pick = 0;
  std::vector<int> holds;

  // The pseudo rounds.
  std::vector<double> loadNeighbors, toSendLoad, prevRoundToSend;
  double my_pseudo_load = 0.0, prev_pseudo_load = 0.0;

  bool hasNbor(int node) const
  {
    return std::find(sendToNeighbors.begin(), sendToNeighbors.end(), node) !=
           sendToNeighbors.end();
  }
  int nborIdx(int node) const
  {
    for (size_t i = 0; i < sendToNeighbors.size(); i++)
      if (sendToNeighbors[i] == node) return (int)i;
    return -1;
  }
};

static LDOMid gOmId;

static LDObjKey keyOf(int k)
{
  LDObjKey key;
  key.omId = gOmId;
  key.objId = (CmiUInt8)k;
  return key;
}

// What the runtime would have assembled on this node's rank-0 PE: its objects
// with their declared loads, and one comm record per (sender, receiver) pair
// among the sends its objects made, to local and remote partners alike. The
// hash the metric resolves senders and receivers through is built the same way
// BuildStats builds it.
static void buildNodeStats(VNode& n, const Grid& g, const std::vector<int>& map)
{
  delete n.st;
  n.st = new BaseLB::LDStats(1);
  n.st->procs[0].pe = n.id;
  n.st->procs[0].available = true;
  n.st->procs[0].n_objs = (int)n.objs.size();
  n.st->procs[0].pe_speed = 1;

  const int nobj = (int)n.objs.size();
  n.st->objData.resize(nobj);
  n.st->from_proc.resize(nobj);
  n.st->to_proc.resize(nobj);
  n.my_load = 0.0;
  n.hostSum = n.hostMax = n.devSum = n.devMax = 0.0;
  for (int a = 0; a < nobj; a++)
  {
    const int k = n.objs[a];
    LDObjData& od = n.st->objData[a];
    od = LDObjData();
    od.handle.omhandle.id = gOmId;
    od.handle.omhandle.handle = 0;
    od.handle.id = (CmiUInt8)k;
    od.handle.handle = k;
    od.wallTime = g.w[k];
#if CMK_LB_CPUTIMER
    od.cpuTime = g.w[k];
#endif
#if CMK_CUDA
    od.gpuTime = g.w2.empty() ? 0.0 : g.w2[k];
    od.gpuPupSize = 0;
#endif
    n.hostSum += od.wallTime;
    n.hostMax = std::max(n.hostMax, (double)od.wallTime);
    n.devSum += diffusionObjGpuLoad(od);
    n.devMax = std::max(n.devMax, diffusionObjGpuLoad(od));
    od.migratable = true;
    od.asyncArrival = false;
    od.prevPe = -1;
    od.prevStep = -1;
    // lbdriver's Cell pups its weight and its ghost buffer.
    od.pupSize = pup_encodeSize((size_t)g.ghostBytes + 16);
    n.st->from_proc[a] = n.id;
    n.st->to_proc[a] = -1;
    n.my_load += diffusionObjLoad(od);
  }
  n.st->n_migrateobjs = nobj;

  n.st->commData.clear();
  std::vector<int> p;
  for (int a = 0; a < nobj; a++)
  {
    const int k = n.objs[a];
    g.partners(k, p);
    for (int m : p)
    {
      LDCommData c;
      c.src_proc = -1;  // from an object, not a PE
      c.sender = keyOf(k);
      LDOMid om = gOmId;
      CmiUInt8 mid = (CmiUInt8)m;
      c.receiver.init_objmsg(om, mid, map[m]);
      c.sendHash = c.recvHash = -1;
      c.messages = g.iters;
      c.bytes = g.iters * g.ghostBytes;
      n.st->commData.push_back(c);
    }
  }
  n.st->deleteCommHash();
  n.st->makeCommHash();
}

// DiffusionLB::countCommToNodes: bytes this node's objects sent to each other
// node. Send side only, like the runtime's records.
static void countCommToNodes(VNode& n, const Grid& g, const std::vector<int>& map, int nnodes)
{
  n.bytesToNode.assign(nnodes, 0);
  std::vector<int> p;
  for (int k : n.objs)
  {
    g.partners(k, p);
    for (int m : p)
      if (map[m] != n.id) n.bytesToNode[map[m]] += (long)g.iters * g.ghostBytes;
  }
}

// DiffusionLB::sortArr: every other node, heaviest traffic first, ties broken
// toward the higher node id (the pair sort is ascending and then reversed).
static void preferenceOrder(VNode& n, int nnodes)
{
  std::vector<std::pair<long, int>> vp;
  for (int i = 0; i < nnodes; i++) vp.push_back(std::make_pair(n.bytesToNode[i], i));
  std::sort(vp.begin(), vp.end());
  std::reverse(vp.begin(), vp.end());
  n.node_idx.assign(nnodes, -1);
  int found = 0;
  for (int i = 0; i < nnodes; i++)
    if (vp[i].second != n.id) n.node_idx[found++] = vp[i].second;
}

// The ask/okay/ack handshake between two nodes, processed to completion. Each
// handler applies the test the chare's handler applies (DiffusionNeighbors.C:
// askNbor, okayNbor, ackNbor), against the state as it stands when it runs.
static void ask(VNode& a, VNode& t, int rnd, int K)
{
  // askNbor on the target: hold a spot this round if it still needs one.
  const int neededT = K - (int)t.sendToNeighbors.size() - t.holds[rnd];
  int agree = 0;
  if (neededT > 0 && !t.hasNbor(a.id))
  {
    agree = 1;
    t.holds[rnd]++;
  }
  // okayNbor on the asker: take the edge if it still needs one.
  const int neededA = K - (int)a.sendToNeighbors.size() - a.holds[rnd];
  if (neededA > 0 && agree && !a.hasNbor(t.id))
  {
    a.sendToNeighbors.push_back(t.id);
    // ackNbor on the target: make the edge symmetric.
    if (!t.hasNbor(a.id)) t.sendToNeighbors.push_back(a.id);
  }
}

// DiffusionLB::findNBors for every node: the ring backbone, then the rounds of
// findNBorsRound until no node needs a neighbour or the rounds run out.
static void buildGraph(std::vector<VNode>& nodes)
{
  const int N = (int)nodes.size();
  const int K = _lb_args.diffusionNumNbors();
  for (VNode& n : nodes)
  {
    n.sendToNeighbors.clear();
    n.holds.assign(kRounds + 1, 0);
    n.pick = 0;
    preferenceOrder(n, N);
    // buildRingBackbone
    if (!_lb_args.noMST() && N >= 2)
    {
      n.sendToNeighbors.push_back((n.id + 1) % N);
      if (N > 2) n.sendToNeighbors.push_back((n.id - 1 + N) % N);
    }
  }

  for (int round = 1; round < kRounds; round++)
  {
    int maxNeeded = 0;  // next_phase collects max(0, nborsNeeded) over nodes
    for (VNode& n : nodes)
    {
      const int nborsNeeded = K - (int)n.sendToNeighbors.size() - n.holds[round];
      if (nborsNeeded > maxNeeded) maxNeeded = nborsNeeded;
      if (nborsNeeded <= 0) continue;
      const int max_neighbors = N < K ? N : K;
      int local_tries = 0;
      while (local_tries < nborsNeeded / 2)
      {
        n.pick = (n.pick + 1) % max_neighbors;
        const int potentialNbor = n.node_idx[n.pick];
        if (potentialNbor == -1)
        {
          local_tries++;
          continue;
        }
        if (n.id != potentialNbor && !n.hasNbor(potentialNbor) && potentialNbor < N &&
            potentialNbor >= 0)
        {
          n.node_idx[n.pick] = -1;
          ask(n, nodes[potentialNbor], round, K);
        }
        local_tries++;
      }
    }
    if (maxNeeded == 0) break;
  }
}

// DiffusionLB::nborFlowAdjacent, comm rule only (there are no 1-D keys here),
// plus PseudoLoadBalancing's rule that a neighbour with no room in the
// dimension not being diffused takes no flow.
static void flowAdjacency(const VNode& n, const std::vector<VNode>& nodes,
                          std::vector<char>& adj)
{
  const int nc = (int)n.sendToNeighbors.size();
  adj.assign(nc, 0);
  bool any = false;
  for (int i = 0; i < nc; i++)
  {
    adj[i] = n.bytesToNode[n.sendToNeighbors[i]] > 0 ? 1 : 0;
    if (adj[i]) any = true;
  }
  if (!any) adj.assign(nc, 1);

  if (gVectorMode && nc > 0 && !diffusionStepMode())
  {
    const bool devDim = diffusionDeviceDim();
    const double eps = _lb_args.diffusionMinImbalance();
    double fairD = 0.0, fairOther = 0.0;
    for (int i = 0; i < nc; i++)
    {
      const VNode& nb = nodes[n.sendToNeighbors[i]];
      fairD += n.loadNeighbors[i];
      fairOther += devDim ? nb.hostSum : nb.devSum;
    }
    fairD /= nc;
    fairOther /= nc;
    const double limit = std::max(fairD, fairOther) * (1.0 + eps);
    for (int i = 0; i < nc; i++)
    {
      const VNode& nb = nodes[n.sendToNeighbors[i]];
      if ((devDim ? nb.hostSum : nb.devSum) >= limit) adj[i] = 0;
    }
  }
}

// Route mode carries tokens along with the rounds; defined with the route code.
struct TokenRouter;
static void routerSyncLoads(TokenRouter* router, std::vector<VNode>& nodes);
static bool routerMoveRound(TokenRouter* router, std::vector<VNode>& nodes,
                            const std::vector<std::vector<double>>& flows);
static bool routerStalled(TokenRouter* router);

// The pseudolb_rounds loop of DiffusionLB.ci, every node in lockstep. Returns
// the rounds run. `effMinImbalance` is the per-neighbourhood floor the round
// arithmetic applies (DiffusionLB passes +LBDiffusionMinImbalance). With a
// router, every round's flows are also carried out on tokens.
static int pseudoRounds(std::vector<VNode>& nodes, double effMinImbalance, bool relayHoldings,
                        TokenRouter* router = NULL)
{
  const double beta = _lb_args.diffusionBeta();
  for (VNode& n : nodes)
  {
    const int nc = (int)n.sendToNeighbors.size();
    n.loadNeighbors.assign(nc, 0.0);
    n.toSendLoad.assign(nc, 0.0);
    n.prevRoundToSend.assign(nc, 0.0);
    n.my_pseudo_load = n.my_load;
    n.prev_pseudo_load = n.my_load;
  }

  int itr = 0;
  bool converged = false;
  std::vector<std::vector<double>> flows(nodes.size());
  std::vector<char> adj;
  while (itr < diffusionIterations() && !converged)
  {
    // With tokens, the state each round plans from IS the token state: what
    // the last round could not carry is simply still there to be planned for.
    if (router != NULL) routerSyncLoads(router, nodes);

    // ReceiveLoadInfo: every node learns its neighbours' loads for the round.
    for (VNode& n : nodes)
      for (size_t i = 0; i < n.sendToNeighbors.size(); i++)
        n.loadNeighbors[i] = nodes[n.sendToNeighbors[i]].my_pseudo_load;

    // PseudoLoadBalancing on every node, from the same snapshot.
    for (VNode& n : nodes)
    {
      flowAdjacency(n, nodes, adj);
      diffusionRoundFlows(n.my_load, n.my_pseudo_load, effMinImbalance, beta, n.loadNeighbors,
                          adj, n.toSendLoad, n.prevRoundToSend, flows[n.id], relayHoldings);
    }

    if (router == NULL)
    {
      // Commit and deliver: the sender's bookkeeping, then PseudoLoad at the receiver.
      for (VNode& n : nodes)
      {
        for (size_t i = 0; i < n.sendToNeighbors.size(); i++)
        {
          const double f = flows[n.id][i];
          n.toSendLoad[i] += f;
          n.prevRoundToSend[i] = f;
          n.my_pseudo_load -= f;
          VNode& t = nodes[n.sendToNeighbors[i]];
          t.my_pseudo_load += f;
          const int pos = t.nborIdx(n.id);
          if (pos >= 0) t.toSendLoad[pos] -= f;
        }
      }

      // Convergence: the largest share of its own load any node still shifted.
      double maxRatio = 0.0;
      for (VNode& n : nodes)
      {
        const double denom = (n.my_load > 1e-12) ? n.my_load : 1e-12;
        const double m = std::fabs(n.my_pseudo_load - n.prev_pseudo_load) / denom;
        n.prev_pseudo_load = n.my_pseudo_load;
        if (m > maxRatio) maxRatio = m;
      }
      converged = (maxRatio <= diffusionPseudoConvergeRatio());
    }
    else
    {
      // Tokens carry this round's flows; the next round plans from where they
      // land. Done when nothing moved and no flow is left that a token could
      // ever carry.
      const bool moved = routerMoveRound(router, nodes, flows);
      double maxFlow = 0.0;
      for (const VNode& n : nodes)
        for (double f : flows[n.id])
          if (f > maxFlow) maxFlow = f;
      converged = (!moved && maxFlow < 1.0) || routerStalled(router);
    }
    itr++;
  }
  return itr;
}

// Experiment, simulator only (LBSIM_PEEL=layers): peel a chunk by LAYERS from
// the destination's boundary. Every object gets its graph distance, through
// this node's own comm edges, from the objects that talk to the destination;
// the pick is the nearest layer first, heaviest within a layer. MetricComm's
// growth (seed on the boundary, then best edge cut) was built for small
// chunks and can tunnel when asked for most of a node, leaving the remainder
// in pieces; this is the rule to compare it against. The chare does not have
// it.
class MetricLayers : public DiffusionMetric
{
  BaseLB::LDStats* st;
  int n;
  std::vector<int> nbors;
  std::vector<double> quota;
  std::vector<std::vector<int>> adj;   // internal comm adjacency, local indices
  std::vector<std::vector<int>> dist;  // [nbor][obj]: layers from that boundary, -1 if none
  std::vector<bool> avail;
  int accepted = 0;

public:
  MetricLayers(BaseLB::LDStats* ns, int nodeId, int nodeSize, const std::vector<double>& tSL,
               const std::vector<int>& sendToNbrs)
      : st(ns), n((int)ns->objData.size()), nbors(sendToNbrs), quota(tSL)
  {
    avail.assign(n, true);
    adj.resize(n);
    std::vector<std::vector<char>> border(nbors.size(), std::vector<char>(n, 0));
    for (LDCommData& c : ns->commData)
    {
      if (c.from_proc() || c.recv_type() != LD_OBJ_MSG) continue;
      const int from = ns->getHash(c.sender);
      if (from < 0 || from >= n) continue;
      const int toNode = c.receiver.lastKnown() / nodeSize;
      if (toNode == nodeId)
      {
        const int to = ns->getHash(c.receiver.get_destObj());
        if (to >= 0 && to < n)
        {
          adj[from].push_back(to);
          adj[to].push_back(from);
        }
      }
      else
      {
        for (size_t i = 0; i < nbors.size(); i++)
          if (nbors[i] == toNode) border[i][from] = 1;
      }
    }
    dist.assign(nbors.size(), std::vector<int>(n, -1));
    for (size_t i = 0; i < nbors.size(); i++)
    {
      std::deque<int> q;
      for (int o = 0; o < n; o++)
        if (border[i][o])
        {
          dist[i][o] = 0;
          q.push_back(o);
        }
      while (!q.empty())
      {
        const int o = q.front();
        q.pop_front();
        for (int p : adj[o])
          if (dist[i][p] < 0)
          {
            dist[i][p] = dist[i][o] + 1;
            q.push_back(p);
          }
      }
    }
  }

  int popBestObject(int nbor) override
  {
    int best = -1, bestD = INT_MAX;
    double bestLoad = -1.0;
    for (int o = 0; o < n; o++)
    {
      if (!avail[o] || !st->objData[o].migratable || !isAllowed(o)) continue;
      const double l = diffusionObjLoad(st->objData[o]);
      if (l <= 0.0 || l > quota[nbor]) continue;
      const int d = dist[nbor][o];
      if (d < 0) continue;  // no path to that boundary: never a candidate
      if (d < bestD || (d == bestD && l > bestLoad))
      {
        best = o;
        bestD = d;
        bestLoad = l;
      }
    }
    if (best >= 0) accepted++;
    return best;
  }
  int getBestNeighbor() override
  {
    for (size_t i = 0; i < quota.size(); i++)
      if (quota[i] > 0) return (int)i;
    return -1;
  }
  void updateState(int o, int nb) override
  {
    quota[nb] -= diffusionObjLoad(st->objData[o]);
    avail[o] = false;
  }
  int acceptedCount() const override { return accepted; }
};

struct StepStats
{
  int rounds = 0;
  int moves = 0;
  int accepted = 0, rejected = 0;
  int slackRefused = 0;  // candidates the receiver check turned away
  double unshed = 0.0;
};

// loadDimReport / loadDimVerdict: which dimension this step diffuses, from
// every node's totals, the way PE 0 decides it for the chare. Each virtual
// node is one PE and one device. Then my_load is priced in that dimension --
// buildNodeStats summed it before the verdict, as BuildStats does.
static void resolveLoadDim(std::vector<VNode>& nodes, int step)
{
  LBCriticality c;
  for (const VNode& n : nodes)
  {
    LBCriticality one;
    one.sumHost = n.hostSum;
    one.maxHost = n.hostMax;
    one.sumDev = n.devSum;
    one.maxDev = n.devMax;
    one.pes = 1;
    one.gpus = 1;
    c.merge(one);
  }
  const int mode = lbResolveLoadMode(c);
  diffusionLoadDimDevice = mode;
  for (VNode& n : nodes)
  {
    // The node's own side, for the step-time mode (DiffusionLoad.h).
    diffusionNodeDeviceBound = (n.devSum >= n.hostSum) ? 1 : 0;
    n.my_load = 0.0;
    for (const LDObjData& od : n.st->objData) n.my_load += diffusionObjLoad(od);
  }
  if (gVectorMode || _lb_args.debug() > 0)
    CkPrintf("lbsim> step %d load dimension: %s by %s (T_h %.1f, T_g %.1f; alpha_h %.2f,"
             " alpha_g %.2f)\n",
             step, lbLoadModeName(mode),
             lbLoadDimOverride() == LB_DIM_AUTO ? "criticality" : "flag", c.boundHost(),
             c.boundDev(), c.alphaHost(), c.alphaDev());
}

// AcrossNodeLB on every node, against the pre-step mapping, as the real nodes
// do concurrently; the moves are applied afterwards.
static void acrossNode(std::vector<VNode>& nodes, const DiffusionCostConfig& costCfg,
                       std::vector<int>& map, StepStats& ss)
{
  const double effMinImbalance = _lb_args.diffusionMinImbalance();
  std::vector<int> newMap = map;
  for (VNode& n : nodes)
  {
    const int nc = (int)n.sendToNeighbors.size();
    if (nc == 0) continue;
    // This node's side, for what each of its objects is worth (DiffusionLoad.h).
    diffusionNodeDeviceBound = (n.devSum >= n.hostSum) ? 1 : 0;

    // Shed the EXCESS over the neighbourhood mean, and nothing under the floor.
    double fair = 0.0;
    for (double l : n.loadNeighbors) fair += l;
    fair /= nc;
    const double excess = n.my_load - fair;
    double remaining = (excess > 0.0) ? excess : 0.0;
    if (n.my_load <= fair * (1.0 + effMinImbalance)) remaining = 0.0;

    // Experiment knob, simulator only. LBSIM_SHED=plan sheds the pseudo plan's
    // gross outflow on this node's positive-quota edges instead of the node's
    // own excess: a node the plan uses as a relay then forwards what the plan
    // routes through it (from what it holds now), rather than only its own
    // surplus. The chare does not have this; it is here to measure the
    // difference before deciding whether it should.
    double planOut = 0.0, planIn = 0.0;
    for (double q : n.toSendLoad)
      if (q > 0) planOut += q; else planIn -= q;
    const char* shedMode = getenv("LBSIM_SHED");
    if (shedMode != NULL && strcmp(shedMode, "plan") == 0)
      remaining = (planOut < n.my_load) ? planOut : n.my_load;
    if (_lb_args.debug() > 1)
      CkPrintf("[node %d] load %.0f nbr-mean %.0f excess %.0f | plan out %.0f in %.0f"
               " (net %.0f) | shedding %.0f\n",
               n.id, n.my_load, fair, excess, planOut, planIn, planOut - planIn, remaining);

    // The per-step cap, in the diffused dimension.
    double maxShed = std::numeric_limits<double>::max();
    const double frac = _lb_args.diffusionMaxMoveFrac();
    if (frac < 1.0)
    {
      double migLoad = 0.0;
      for (const LDObjData& od : n.st->objData)
        if (od.migratable) migLoad += diffusionObjLoad(od);
      maxShed = frac * migLoad;
    }

    double internalBytes = 0.0, externalBytes = 0.0;
    const char* peel = getenv("LBSIM_PEEL");
    DiffusionMetric* metric;
    if (peel != NULL && strcmp(peel, "layers") == 0)
      metric = new MetricLayers(n.st, n.id, 1, n.toSendLoad, n.sendToNeighbors);
    else
      metric = new MetricComm(n.st, n.id, 1, nc, n.toSendLoad, n.sendToNeighbors, internalBytes,
                              externalBytes, &costCfg);

    // The receiver check in the dimension not being diffused, as AcrossNodeLB
    // sets it: a neighbour may take that dimension up to the step-time level
    // the plan brings everyone to, and no further. One PE per node, so both
    // terms are already per node and no unit conversion is needed.
    if (gVectorMode)
    {
      const double inf = std::numeric_limits<double>::max();
      std::vector<double> capH(nc, inf), capG(nc, inf);
      if (diffusionStepMode())
      {
        const double limit = fair * (1.0 + effMinImbalance);
        for (int i = 0; i < nc; i++)
        {
          const VNode& nb = nodes[n.sendToNeighbors[i]];
          capH[i] = std::max(0.0, limit - nb.hostSum);
          capG[i] = std::max(0.0, limit - nb.devSum);
        }
        metric->setRiseKnown(true);
      }
      else
      {
        const bool devDim = diffusionDeviceDim();
        double fairOther = 0.0;
        for (int i = 0; i < nc; i++)
        {
          const VNode& nb = nodes[n.sendToNeighbors[i]];
          fairOther += devDim ? nb.hostSum : nb.devSum;
        }
        fairOther /= nc;
        const double limit = std::max(fair, fairOther) * (1.0 + effMinImbalance);
        for (int i = 0; i < nc; i++)
        {
          const VNode& nb = nodes[n.sendToNeighbors[i]];
          if (devDim) capH[i] = std::max(0.0, limit - nb.hostSum);
          else capG[i] = std::max(0.0, limit - nb.devSum);
        }
      }
      metric->setReceiverCapacity(capH, capG);
    }

    std::vector<DiffusionMove> moves;
    double shedThisStep = 0.0;
    int movesThisStep = 0;
    diffusionSelectMoves(*metric, *n.st, n.toSendLoad, remaining, maxShed,
                         std::function<void(int, std::vector<char>&)>(), n.id, moves,
                         shedThisStep, movesThisStep);

    for (const DiffusionMove& mv : moves)
      newMap[n.objs[mv.obj]] = n.sendToNeighbors[mv.nbor];
    ss.moves += (int)moves.size();
    ss.accepted += metric->acceptedCount();
    ss.rejected += metric->rejectedCount();
    ss.slackRefused += metric->slackRefusals;
    delete metric;
    if (remaining > 0.0) ss.unshed += remaining;
  }
  map = newMap;
}

// ---- route mode: plan, route tokens, refine, migrate once ----------------------
//
// LBSIM_MODE=route. The design that handles a deep imbalance with locality,
// prototyped here before the chare gets a protocol for it. The pseudo rounds
// plan the whole deformation (a 1% per-edge floor; whether to balance at all is
// a global decision on max/avg against +LBDiffusionMinImbalance). Then:
//
//   route    Tokens follow the rounds. Each round's flows are consistent by
//            construction -- every node computed them from the same state,
//            receivers included -- so each round every node peels, from its
//            boundary with each neighbour, as much as that round's flow to it
//            says, in layers from the boundary over what it holds right then.
//            A flow too small for the next token is carried forward until it
//            is not. Rings shift a little every round, adjacency is always the
//            current one, and relaying needs no ordering: a node that received
//            in round r has that load in its holdings for round r+1. Tokens
//            move; objects do not.
//
//            Two other executions were tried first and are worth not retrying.
//            Executing the SUMMED net edge flows in one pass fails because the
//            sum over 200 rounds has cycles on most nodes, so no upstream-first
//            order exists, and a relay that has handed its whole region on
//            leaves its upstream neighbour no boundary to route through (one
//            node took 1477 and forwarded 8). Executing the TARGETS -- every
//            node above its planned load peels to neighbours below theirs --
//            stalls after two rounds: the ring around a hot region fills to
//            target and then neither sheds nor accepts, and the interior is
//            boxed in. Targets cannot drive relaying; flows can.
//   refine   A detached piece goes to the neighbour it touches most. Then a few
//            rounds of boundary smoothing: an object with more edges into a
//            neighbour than into its own node moves there if both loads stay
//            within tolerance of their planned targets. One direction per
//            round, so two nodes never swap the same boundary at once.
//   migrate  Every object whose owner changed moves once, straight there.

struct RouteStats
{
  int migrated = 0;   // objects whose owner changed
  int forwarded = 0;  // tokens relayed beyond the node that originally held them
  int healed = 0;     // objects moved to reattach detached pieces
  int smoothed = 0;   // objects moved by boundary smoothing
  int rounds = 0;     // token rounds until nothing moved
  int tokenMoves = 0; // token hops in total (an object can hop more than once)
  int maxHops = 0;    // farthest an object migrated, in node-graph hops
  int multiHop = 0;   // objects that migrated more than one hop
  double residual = 0.0;  // load still above target + tol when routing stopped
};

// Layers of node v's holdings from its boundary with w: distance 0 is a token
// of v adjacent to a token of w, then outward through v's own tokens.
static void layersFrom(const Grid& g, const std::vector<int>& owner, int v, int w,
                       std::vector<int>& dist)
{
  dist.assign(g.n(), -1);
  std::deque<int> q;
  std::vector<int> p;
  for (int k = 0; k < g.n(); k++)
  {
    if (owner[k] != v) continue;
    g.partners(k, p);
    for (int m : p)
      if (owner[m] == w)
      {
        dist[k] = 0;
        q.push_back(k);
        break;
      }
  }
  while (!q.empty())
  {
    const int k = q.front();
    q.pop_front();
    g.partners(k, p);
    for (int m : p)
      if (owner[m] == v && dist[m] < 0)
      {
        dist[m] = dist[k] + 1;
        q.push_back(m);
      }
  }
}

struct TokenRouter
{
  const Grid& g;
  std::vector<int> orig;   // who held each object at the start of the step
  std::vector<int> owner;  // who holds its token now
  // Per node, per neighbour: flow the rounds have asked for that no token has
  // carried yet. A token moves once this covers its weight.
  std::vector<std::vector<double>> owed;
  RouteStats& rs;
  std::vector<int> dist, cand;

  // What is owed on an edge is capped at two of the heaviest tokens: the flow
  // is re-planned from the token state every round, so a shortfall is
  // re-requested rather than banked, and a bank would only burst later.
  double cap;
  bool movedThisRound = false;

  TokenRouter(const Grid& g_, const std::vector<int>& map, RouteStats& rs_)
      : g(g_), orig(map), owner(map), rs(rs_)
  {
    double maxW = 0.0;
    for (double x : g.w) maxW = std::max(maxW, x);
    cap = 2.0 * maxW;
  }

  // Stall detection on the token loads: the best max/avg seen and how many
  // rounds ago. Tokens are coarser than the flow floor, so the rounds do not
  // settle on their own; they chase, and the best state they pass through is
  // the one to keep.
  double bestImb = 1e300;
  int sinceBest = 0;
  std::vector<int> bestOwner;

  // The round plans from the token loads: what every node holds right now.
  void syncLoads(std::vector<VNode>& nodes)
  {
    std::vector<double> load(nodes.size(), 0.0);
    for (int k = 0; k < g.n(); k++) load[owner[k]] += g.w[k];
    double sum = 0.0, mx = 0.0;
    for (VNode& n : nodes)
    {
      n.my_pseudo_load = load[n.id];
      n.toSendLoad.assign(n.sendToNeighbors.size(), 0.0);
      sum += load[n.id];
      mx = std::max(mx, load[n.id]);
    }
    const double imb = sum > 0 ? mx / (sum / nodes.size()) : 0.0;
    if (imb < bestImb - 1e-9)
    {
      bestImb = imb;
      sinceBest = 0;
      bestOwner = owner;
    }
    else
      sinceBest++;
  }

  // Carry the flow owed on one edge with tokens, nearest the boundary first
  // and lightest within a layer, stopping at the first token what is owed
  // does not cover `minFrac` of -- no tunnelling past it for a lighter one.
  void drain(std::vector<VNode>& nodes, int v, int i, double minFrac)
  {
    const int w = nodes[v].sendToNeighbors[i];
    double& due = owed[v][i];
    if (due < minFrac * 1.0) return;
    layersFrom(g, owner, v, w, dist);
    cand.clear();
    for (int k = 0; k < g.n(); k++)
      if (owner[k] == v && dist[k] >= 0) cand.push_back(k);
    if (cand.empty()) return;
    std::sort(cand.begin(), cand.end(), [&](int a, int b) {
      if (dist[a] != dist[b]) return dist[a] < dist[b];
      if (g.w[a] != g.w[b]) return g.w[a] < g.w[b];
      return a < b;
    });
    for (int k : cand)
    {
      if (due <= 0.0 || due < minFrac * g.w[k]) break;
      owner[k] = w;
      due -= g.w[k];
      rs.tokenMoves++;
      movedThisRound = true;
      if (orig[k] != v) rs.forwarded++;
    }
  }

  bool moveRound(std::vector<VNode>& nodes, const std::vector<std::vector<double>>& flows)
  {
    if (owed.empty()) owed.resize(nodes.size());
    movedThisRound = false;
    for (VNode& n : nodes)
    {
      if (owed[n.id].size() != n.sendToNeighbors.size())
        owed[n.id].assign(n.sendToNeighbors.size(), 0.0);
      for (size_t i = 0; i < n.sendToNeighbors.size(); i++)
        owed[n.id][i] = std::min(owed[n.id][i] + flows[n.id][i], cap);
    }
    for (VNode& n : nodes)
      for (size_t i = 0; i < n.sendToNeighbors.size(); i++)
        if (owed[n.id][i] >= 1.0) drain(nodes, n.id, (int)i, 1.0);
    rs.rounds++;
    return movedThisRound;
  }

  // After the last round: a token whose weight is at least half covered goes.
  void finish(std::vector<VNode>& nodes)
  {
    for (VNode& n : nodes)
      for (size_t i = 0; i < n.sendToNeighbors.size() && i < owed[n.id].size(); i++)
        if (owed[n.id][i] > 0.0) drain(nodes, n.id, (int)i, 0.5);
    for (VNode& n : nodes)
      for (size_t i = 0; i < owed[n.id].size(); i++)
        if (owed[n.id][i] > 0.0) rs.residual += owed[n.id][i];
  }
};

static void routerSyncLoads(TokenRouter* router, std::vector<VNode>& nodes)
{
  router->syncLoads(nodes);
}

static bool routerMoveRound(TokenRouter* router, std::vector<VNode>& nodes,
                            const std::vector<std::vector<double>>& flows)
{
  return router->moveRound(nodes, flows);
}

// LBSIM_PATIENCE rounds without a better token state ends the rounds; the
// best state seen is what gets refined and migrated.
static bool routerStalled(TokenRouter* router)
{
  static const int patience = getenv("LBSIM_PATIENCE") ? atoi(getenv("LBSIM_PATIENCE")) : 40;
  if (router->sinceBest < patience) return false;
  router->owner = router->bestOwner;
  return true;
}

// Reattach every detached piece to the neighbour it shares the most edges with.
static int healPieces(const Grid& g, std::vector<int>& owner, int nnodes)
{
  const int n = g.n();
  std::vector<int> comp(n, -1);
  std::vector<double> compLoad;
  std::vector<int> compOwner;
  std::vector<std::vector<int>> compCells;
  std::vector<int> p;
  for (int s = 0; s < n; s++)
  {
    if (comp[s] >= 0) continue;
    const int c = (int)compLoad.size();
    compLoad.push_back(0.0);
    compOwner.push_back(owner[s]);
    compCells.push_back(std::vector<int>());
    std::deque<int> q;
    q.push_back(s);
    comp[s] = c;
    while (!q.empty())
    {
      const int k = q.front();
      q.pop_front();
      compLoad[c] += g.w[k];
      compCells[c].push_back(k);
      g.partners(k, p);
      for (int m : p)
        if (comp[m] < 0 && owner[m] == owner[k])
        {
          comp[m] = c;
          q.push_back(m);
        }
    }
  }
  // The main piece of each node is its heaviest.
  std::vector<int> mainComp(nnodes, -1);
  for (size_t c = 0; c < compLoad.size(); c++)
  {
    const int o = compOwner[c];
    if (mainComp[o] < 0 || compLoad[c] > compLoad[mainComp[o]]) mainComp[o] = (int)c;
  }
  int moved = 0;
  for (size_t c = 0; c < compLoad.size(); c++)
  {
    if ((int)c == mainComp[compOwner[c]]) continue;
    std::vector<int> touch(nnodes, 0);
    for (int k : compCells[c])
    {
      g.partners(k, p);
      for (int m : p)
        if (owner[m] != compOwner[c]) touch[owner[m]]++;
    }
    int best = -1;
    for (int t = 0; t < nnodes; t++)
      if (touch[t] > 0 && (best < 0 || touch[t] > touch[best])) best = t;
    if (best < 0) continue;
    for (int k : compCells[c]) owner[k] = best;
    moved += (int)compCells[c].size();
  }
  return moved;
}

// One round of boundary smoothing. `parity` fixes which side of a pair may act.
static int smoothBoundaries(const Grid& g, std::vector<int>& owner, int nnodes,
                            const std::vector<double>& target, double tol, int parity)
{
  std::vector<double> load(nnodes, 0.0);
  for (int k = 0; k < g.n(); k++) load[owner[k]] += g.w[k];
  std::vector<int> p;
  int moved = 0;
  for (int k = 0; k < g.n(); k++)
  {
    const int v = owner[k];
    g.partners(k, p);
    int own = 0, bestCnt = 0, best = -1;
    // Count edges per neighbouring owner; the partner list is at most four long.
    for (int m : p)
      if (owner[m] == v) own++;
    for (int m : p)
    {
      const int t = owner[m];
      if (t == v) continue;
      int cnt = 0;
      for (int m2 : p)
        if (owner[m2] == t) cnt++;
      if (cnt > bestCnt) { bestCnt = cnt; best = t; }
    }
    if (best < 0 || bestCnt <= own) continue;
    if ((parity == 0) != (v < best)) continue;
    if (load[v] - g.w[k] < target[v] - tol) continue;
    if (load[best] + g.w[k] > target[best] + tol) continue;
    owner[k] = best;
    load[v] -= g.w[k];
    load[best] += g.w[k];
    moved++;
  }
  return moved;
}

// Refinement and the single migration, on the map the tokens settled on.
static void routeStep(std::vector<VNode>& nodes, const Grid& g, std::vector<int>& map,
                      const std::vector<int>& routed, double tol, int passes, RouteStats& rs)
{
  const int N = (int)nodes.size();
  std::vector<int> owner = routed;
  std::vector<double> target(N), load(N, 0.0);
  for (const VNode& n : nodes) target[n.id] = n.my_pseudo_load;
  for (int k = 0; k < g.n(); k++) load[owner[k]] += g.w[k];

  // LBSIM_TRACE: the nodes that ended farthest above their planned load.
  if (getenv("LBSIM_TRACE") != NULL)
  {
    std::vector<int> idx(N);
    for (int v = 0; v < N; v++) idx[v] = v;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
      return load[a] - target[a] > load[b] - target[b];
    });
    for (int i = 0; i < 4 && i < N; i++)
    {
      const int v = idx[i];
      std::string nb;
      for (int w : nodes[v].sendToNeighbors)
      {
        char b[48];
        snprintf(b, sizeof(b), " %d(%.0f/%.0f)", w, load[w], target[w]);
        nb += b;
      }
      CkPrintf("lbsim>   node %3d: start %.0f, target %.0f, after routing %.0f; neighbours"
               " load/target:%s\n",
               v, nodes[v].my_load, target[v], load[v], nb.c_str());
    }
  }

  // Refine.
  for (int pass = 0; pass < passes; pass++)
  {
    rs.healed += healPieces(g, owner, N);
    rs.smoothed += smoothBoundaries(g, owner, N, target, tol, pass % 2);
  }
  rs.healed += healPieces(g, owner, N);

  // Migrate once. Hops are measured on the neighbour graph.
  std::vector<std::vector<int>> hop(N);
  for (int k = 0; k < g.n(); k++)
  {
    if (owner[k] == map[k]) continue;
    rs.migrated++;
    const int a = map[k], b = owner[k];
    if (hop[a].empty())
    {
      hop[a].assign(N, -1);
      std::deque<int> q;
      q.push_back(a);
      hop[a][a] = 0;
      while (!q.empty())
      {
        const int x = q.front();
        q.pop_front();
        for (int y : nodes[x].sendToNeighbors)
          if (hop[a][y] < 0)
          {
            hop[a][y] = hop[a][x] + 1;
            q.push_back(y);
          }
      }
    }
    const int h = hop[a][b] < 0 ? 99 : hop[a][b];
    if (h > rs.maxHops) rs.maxHops = h;
    if (h > 1) rs.multiHop++;
  }
  map = owner;
}

// ---- the initial partition ---------------------------------------------------

// METIS on the same graph MetisLB would build: vertex weights normalised to
// 1..256, edge weights the recorded bytes, recursive bisection with a 1.1
// balance tolerance.
static std::vector<int> metisPartition(const Grid& g, int nparts)
{
  const int nv = g.n();
  std::vector<int> part(nv, 0);
  if (nparts <= 1) return part;

  double maxw = 0.0;
  for (double x : g.w) maxw = std::max(maxw, x);
  const double ratio = (maxw == 0) ? 0 : 256.0 / maxw;

  std::vector<idx_t> xadj(nv + 1), adjncy, adjwgt, vwgt(nv), parts(nv, 0);
  std::vector<int> p;
  idx_t e = 0;
  for (int k = 0; k < nv; k++)
  {
    xadj[k] = e;
    vwgt[k] = std::max((idx_t)1, (idx_t)std::ceil(g.w[k] * ratio));
    g.partners(k, p);
    for (int m : p)
    {
      adjncy.push_back(m);
      adjwgt.push_back((idx_t)std::min<long long>(
          std::max<long long>((long long)g.iters * g.ghostBytes, 1),
          (long long)std::numeric_limits<idx_t>::max()));
      e++;
    }
  }
  xadj[nv] = e;

  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_NUMBERING] = 0;
  idx_t ncon = 1, nvt = nv, np = nparts, edgecut = 0;
  real_t ubvec = (real_t)1.1;
  METIS_PartGraphRecursive(&nvt, &ncon, xadj.data(), adjncy.data(), vwgt.data(), NULL,
                           adjwgt.data(), &np, NULL, &ubvec, options, &edgecut, parts.data());
  for (int k = 0; k < nv; k++) part[k] = (int)parts[k];
  return part;
}

// A px x py block map, px the largest divisor of nnodes not above its root.
static std::vector<int> blockPartition(const Grid& g, int nnodes)
{
  int px = (int)std::sqrt((double)nnodes);
  while (px > 1 && nnodes % px != 0) px--;
  const int py = nnodes / px;
  std::vector<int> part(g.n());
  for (int i = 0; i < g.nx; i++)
    for (int j = 0; j < g.ny; j++)
      part[g.at(i, j)] = (i * px / g.nx) * py + (j * py / g.ny);
  return part;
}

// ---- remap mode: partition from scratch, relabel for overlap, migrate once ----
//
// LBSIM_MODE=remap. Schloegel, Karypis and Kumar's scratch-remap: METIS on the
// current weights gives a balanced, contiguous partition that knows nothing of
// the current one; relabelling its parts to the current owners they overlap
// most (greedy, heaviest overlap first) recovers what can be recovered of the
// current placement, and every object then migrates once, straight to its
// final owner. This is the literature's answer for a large perturbation, and
// the measurement here says why: for a 12x disc the balanced partition puts
// most of the nodes inside the disc, so most regions relocate wholesale, and
// no hop-by-hop transport reaches that cheaply.
static void remapStep(const Grid& g, std::vector<int>& map, int nnodes, int& migrated)
{
  const std::vector<int> fresh = metisPartition(g, nnodes);
  std::vector<std::vector<double>> overlap(nnodes, std::vector<double>(nnodes, 0.0));
  for (int k = 0; k < g.n(); k++) overlap[map[k]][fresh[k]] += g.w[k];

  struct Pair { double load; int oldPart, newPart; };
  std::vector<Pair> pairs;
  for (int o = 0; o < nnodes; o++)
    for (int n = 0; n < nnodes; n++)
      if (overlap[o][n] > 0.0) pairs.push_back(Pair{overlap[o][n], o, n});
  std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
    if (a.load != b.load) return a.load > b.load;
    if (a.oldPart != b.oldPart) return a.oldPart < b.oldPart;
    return a.newPart < b.newPart;
  });
  std::vector<int> labelOf(nnodes, -1);
  std::vector<char> taken(nnodes, 0);
  for (const Pair& p : pairs)
    if (labelOf[p.newPart] < 0 && !taken[p.oldPart])
    {
      labelOf[p.newPart] = p.oldPart;
      taken[p.oldPart] = 1;
    }
  int spare = 0;
  for (int n = 0; n < nnodes; n++)
    if (labelOf[n] < 0)
    {
      while (taken[spare]) spare++;
      labelOf[n] = spare;
      taken[spare] = 1;
    }

  migrated = 0;
  for (int k = 0; k < g.n(); k++)
  {
    const int to = labelOf[fresh[k]];
    if (to != map[k]) migrated++;
    map[k] = to;
  }
}

// ---- measures ----------------------------------------------------------------

static int edgeCut(const Grid& g, const std::vector<int>& map)
{
  int c = 0;
  for (int i = 0; i < g.nx; i++)
    for (int j = 0; j < g.ny; j++)
    {
      const int k = g.at(i, j);
      if (i + 1 < g.nx && map[k] != map[g.at(i + 1, j)]) c++;
      if (j + 1 < g.ny && map[k] != map[g.at(i, j + 1)]) c++;
    }
  return c;
}

// Connected pieces (4-connectivity) beyond one per node that holds anything.
static int detachedPieces(const Grid& g, const std::vector<int>& map, int nnodes)
{
  std::vector<char> seen(g.n(), 0);
  std::vector<int> pieces(nnodes, 0);
  std::vector<int> p;
  for (int s = 0; s < g.n(); s++)
  {
    if (seen[s]) continue;
    pieces[map[s]]++;
    std::deque<int> q;
    q.push_back(s);
    seen[s] = 1;
    while (!q.empty())
    {
      const int k = q.front();
      q.pop_front();
      g.partners(k, p);
      for (int m : p)
        if (!seen[m] && map[m] == map[k])
        {
          seen[m] = 1;
          q.push_back(m);
        }
    }
  }
  int extra = 0;
  for (int n = 0; n < nnodes; n++)
    if (pieces[n] > 1) extra += pieces[n] - 1;
  return extra;
}

static double imbalanceOf(const std::vector<double>& w, const std::vector<int>& map, int nnodes)
{
  std::vector<double> load(nnodes, 0.0);
  for (size_t k = 0; k < w.size(); k++) load[map[k]] += w[k];
  double sum = 0.0, mx = 0.0;
  for (double l : load)
  {
    sum += l;
    mx = std::max(mx, l);
  }
  const double avg = sum / nnodes;
  return avg > 0 ? mx / avg : 0.0;
}

// max/avg in the host dimension -- the one the scalar experiments report.
static double imbalance(const Grid& g, const std::vector<int>& map, int nnodes)
{
  return imbalanceOf(g.w, map, nnodes);
}

// The step a node's two terms would set, max/avg over nodes: what the
// balancer is actually minimising once both dimensions are loaded.
static double stepImbalance(const Grid& g, const std::vector<int>& map, int nnodes)
{
  if (g.w2.empty()) return imbalance(g, map, nnodes);
  std::vector<double> h(nnodes, 0.0), d(nnodes, 0.0);
  for (int k = 0; k < g.n(); k++)
  {
    h[map[k]] += g.w[k];
    d[map[k]] += g.w2[k];
  }
  double sum = 0.0, mx = 0.0;
  for (int q = 0; q < nnodes; q++)
  {
    const double t = std::max(h[q], d[q]);
    sum += t;
    mx = std::max(mx, t);
  }
  const double avg = sum / nnodes;
  return avg > 0 ? mx / avg : 0.0;
}

struct Phase
{
  std::string name;
  std::vector<int> map;
  std::vector<double> w;
};

static void writeJson(const char* path, const Grid& g, int nnodes, const std::vector<Phase>& ph)
{
  FILE* f = fopen(path, "w");
  if (f == NULL)
  {
    CkPrintf("lbsim> could not open %s for writing\n", path);
    return;
  }
  fprintf(f, "{\n");
  fprintf(f, "  \"tool\": \"lbsim\",\n");
  fprintf(f, "  \"nx\": %d,\n  \"ny\": %d,\n  \"npes\": %d,\n", g.nx, g.ny, nnodes);
  fprintf(f, "  \"phases\": [\n");
  for (size_t p = 0; p < ph.size(); p++)
  {
    fprintf(f, "    {\n      \"name\": \"%s\",\n", ph[p].name.c_str());
    fprintf(f, "      \"map\": [");
    for (size_t k = 0; k < ph[p].map.size(); k++) fprintf(f, "%s%d", k ? "," : "", ph[p].map[k]);
    fprintf(f, "],\n");
    fprintf(f, "      \"weights\": [");
    for (size_t k = 0; k < ph[p].w.size(); k++) fprintf(f, "%s%g", k ? "," : "", ph[p].w[k]);
    fprintf(f, "]\n");
    fprintf(f, "    }%s\n", p + 1 < ph.size() ? "," : "");
  }
  fprintf(f, "  ]\n}\n");
  fclose(f);
  CkPrintf("lbsim> wrote %s (%zu phases)\n", path, ph.size());
}

// The simulation, from the positional arguments (the +LB flags have already
// been taken out of argv, by the runtime or by the standalone parser).
static void lbsimRun(int argc, char** argv)
{
  {
    Grid g;
    g.nx = 128;
    g.ny = 128;
    int nnodes = 256;
    int steps = 1;
    g.ghostBytes = 4096;
    g.iters = 6;
    std::string initial = "metis";
    if (argc > 1) g.nx = atoi(argv[1]);
    if (argc > 2) g.ny = atoi(argv[2]);
    if (argc > 3) nnodes = atoi(argv[3]);
    if (argc > 4) steps = atoi(argv[4]);
    if (argc > 5) g.ghostBytes = atoi(argv[5]);
    if (argc > 6) g.iters = atoi(argv[6]);
    if (argc > 7) initial = argv[7];
    if (argc > 8) kHotWork = atof(argv[8]);

    if (nnodes < 2) CkAbort("lbsim: need at least 2 nodes\n");
    if (steps < 1) CkAbort("lbsim: need at least 1 step\n");

    // The one metric that needs no positions. The chare would take
    // MetricCentroid without this flag and abort for want of positions.
    if (!_lb_args.diffusionCommOn())
    {
      _lb_args.diffusionCommOn() = true;
      CkPrintf("lbsim> +LBDiffusionCommOn assumed (no object positions here)\n");
    }
    // Tiers: the metric prices every neighbour as a separate node, as it does
    // under the logical-node hook, rather than calling every pair intra-process.
    setenv("CHARM_DIFFUSION_NODE_SIZE", "1", 1);
    // One PE per virtual node: host time per PE is the node's host total.
    diffusionPpn = 1;

    DiffusionCostConfig costCfg;
    if (_lb_args.costConfig() != NULL) costCfg.load(_lb_args.costConfig());

    gOmId.id.idx = 1;

    CkPrintf("lbsim> %d x %d = %d objects on %d virtual nodes, %d step(s), %d-byte ghosts x %d,"
             " initial %s, %d neighbour(s) per node\n",
             g.nx, g.ny, g.n(), nnodes, steps, g.ghostBytes, g.iters, initial.c_str(),
             _lb_args.diffusionNumNbors());

    std::vector<Phase> phases;

    // Phase 0: the initial partition, on flat weights.
    g.weightsUniform();
    std::vector<int> map = (initial == "block") ? blockPartition(g, nnodes) : metisPartition(g, nnodes);
    {
      Phase p;
      p.name = (initial == "block") ? "Block map (uniform weights)" : "METIS (uniform weights)";
      p.map = map;
      p.w = g.w;
      phases.push_back(p);
    }
    CkPrintf("lbsim> %-34s max/avg %.3f, edge cut %d, detached pieces %d\n",
             phases.back().name.c_str(), imbalance(g, map, nnodes), edgeCut(g, map),
             detachedPieces(g, map, nnodes));

    // Then the hot region appears and DiffusionLB has to repair it. With
    // LBSIM_VECTOR the region is in one of two load dimensions, or in both.
    const char* vectorEnv = getenv("LBSIM_VECTOR");
    if (vectorEnv != NULL)
    {
      gVectorMode = true;
      g.weightsVector(vectorEnv);
      CkPrintf("lbsim> vector loads: %s (host disc x%.0f%s)\n", vectorEnv, kHotWork,
               strcmp(vectorEnv, "cross") == 0 ? ", device disc from LBSIM_HOT2" : "");
    }
    else
      g.weightsHotSpot();

    std::vector<VNode> nodes(nnodes);
    for (int n = 0; n < nnodes; n++) nodes[n].id = n;
    bool graphBuilt = false;

    const char* modeEnv = getenv("LBSIM_MODE");
    const bool route = (modeEnv != NULL && strcmp(modeEnv, "route") == 0);
    const bool remap = (modeEnv != NULL && strcmp(modeEnv, "remap") == 0);
    const double effMinImbalance = _lb_args.diffusionMinImbalance();
    const char* localFloorEnv = getenv("LBSIM_LOCAL_FLOOR");
    const double localFloor = localFloorEnv ? atof(localFloorEnv) : 0.01;
    const char* passesEnv = getenv("LBSIM_REFINE_PASSES");
    const int refinePasses = passesEnv ? atoi(passesEnv) : 4;
    if (route)
      CkPrintf("lbsim> route mode: per-edge floor %.3f, global floor %.3f, %d refinement"
               " passes\n",
               localFloor, effMinImbalance, refinePasses);

    for (int step = 1; step <= steps; step++)
    {
      // Stats assembly.
      for (VNode& n : nodes) n.objs.clear();
      for (int k = 0; k < g.n(); k++) nodes[map[k]].objs.push_back(k);
      for (VNode& n : nodes)
      {
        buildNodeStats(n, g, map);
        countCommToNodes(n, g, map, nnodes);
      }
      resolveLoadDim(nodes, step);

      // Neighbour graph: built once, kept across steps; the adjacency the flow
      // rule reads is recounted every step.
      if (!graphBuilt || getenv("CHARM_DIFFUSION_GRAPH_REBUILD") != NULL)
      {
        buildGraph(nodes);
        graphBuilt = true;
        int edges = 0, bordering = 0, waived = 0;
        for (const VNode& n : nodes)
        {
          bool any = false;
          for (int nb : n.sendToNeighbors)
          {
            edges++;
            if (n.bytesToNode[nb] > 0) { bordering++; any = true; }
          }
          if (!any && !n.sendToNeighbors.empty()) waived++;
        }
        CkPrintf("lbsim> neighbour graph: %.2f neighbours per node, %d of %d directed edges"
                 " carry traffic, %d node(s) border none of theirs\n",
                 (double)edges / nnodes, bordering, edges, waived);
      }

      const double imbBefore = imbalance(g, map, nnodes);
      const double devBefore = g.w2.empty() ? 0.0 : imbalanceOf(g.w2, map, nnodes);
      const double stepBefore = stepImbalance(g, map, nnodes);
      const int cutBefore = edgeCut(g, map);

      Phase p;
      char name[96];
      snprintf(name, sizeof(name), "DiffusionLB step %d%s", step,
               step == 1 ? " (hot region added)" : "");
      p.name = name;

      if (remap)
      {
        int migrated = 0;
        if (imbBefore <= 1.0 + effMinImbalance)
          CkPrintf("lbsim> %-34s max/avg %.3f is within the global floor; nothing to do\n",
                   p.name.c_str(), imbBefore);
        else
          remapStep(g, map, nnodes, migrated);
        CkPrintf("lbsim> %-34s scratch-remap: %d migrated (%.0f%% of objects),"
                 " max/avg %.3f -> %.3f, edge cut %d -> %d (%+.1f%%), detached pieces %d\n",
                 p.name.c_str(), migrated, 100.0 * migrated / g.n(), imbBefore,
                 imbalance(g, map, nnodes), cutBefore, edgeCut(g, map),
                 cutBefore ? 100.0 * (edgeCut(g, map) - cutBefore) / cutBefore : 0.0,
                 detachedPieces(g, map, nnodes));
      }
      else if (route)
      {
        if (imbBefore <= 1.0 + effMinImbalance)
        {
          CkPrintf("lbsim> %-34s max/avg %.3f is within the global floor; nothing to do\n",
                   p.name.c_str(), imbBefore);
        }
        else
        {
          RouteStats rs;
          TokenRouter router(g, map, rs);
          const int rounds = pseudoRounds(nodes, localFloor, true, &router);
          router.finish(nodes);
          // What the plan itself promises: the pseudo loads the rounds settled
          // on. The tokens follow them up to granularity.
          double planMax = 0.0, planSum = 0.0;
          for (const VNode& n : nodes)
          {
            planSum += n.my_pseudo_load;
            if (n.my_pseudo_load > planMax) planMax = n.my_pseudo_load;
          }
          CkPrintf("lbsim>   plan after %d rounds: max/avg %.3f\n", rounds,
                   planSum > 0 ? planMax / (planSum / nnodes) : 0.0);
          // Refinement may move a node this far from its planned target load.
          double total = 0.0;
          for (double x : g.w) total += x;
          const double tol = effMinImbalance * total / nnodes;
          routeStep(nodes, g, map, router.owner, tol, refinePasses, rs);
          CkPrintf("lbsim> %-34s %d rounds, %d token hops (%d relayed),"
                   " %d migrated (%d beyond one hop, max %d), %d healed, %d smoothed,"
                   " flow uncarried %.0f, max/avg %.3f -> %.3f, edge cut %d -> %d (%+.1f%%),"
                   " detached pieces %d\n",
                   p.name.c_str(), rounds, rs.tokenMoves, rs.forwarded, rs.migrated, rs.multiHop,
                   rs.maxHops, rs.healed, rs.smoothed, rs.residual, imbBefore,
                   imbalance(g, map, nnodes),
                   cutBefore, edgeCut(g, map),
                   cutBefore ? 100.0 * (edgeCut(g, map) - cutBefore) / cutBefore : 0.0,
                   detachedPieces(g, map, nnodes));
        }
      }
      else
      {
        StepStats ss;
        // LBSIM_SHED=plan uses the fine per-edge floor; the plan must NOT
        // relay, since a node executing it physically can only hand over what
        // it holds (a relaying plan executed this way empties nodes into their
        // neighbours: measured 8.6x). The balancer's own execution keeps its floor.
        const char* shedEnv = getenv("LBSIM_SHED");
        const bool planShed = (shedEnv != NULL && strcmp(shedEnv, "plan") == 0);
        ss.rounds = pseudoRounds(nodes, planShed ? localFloor : effMinImbalance, false);

        // What the plan predicts, summed the way DiffusionLB sums it on PE 0
        // (DiffusionPlanSummary), and the same verdict the chare would reach
        // with +LBDiffusionRemapAbove set to LBSIM_REMAP_ABOVE: execute the
        // plan, or hand the step to the scratch-remap.
        DiffusionPlanSummary plan;
        for (const VNode& n : nodes)
        {
          double in = 0.0, out = 0.0;
          for (double q : n.toSendLoad)
            if (q > 0) out += q; else in -= q;
          plan.add(n.my_load, n.my_pseudo_load, in, out);
        }
        const char* remapAboveEnv = getenv("LBSIM_REMAP_ABOVE");
        const double remapAbove = remapAboveEnv ? atof(remapAboveEnv) : 0.0;
        const bool handOff = diffusionShouldRemap(plan, remapAbove);
        CkPrintf("lbsim>   plan: max/avg %.3f now, %.3f after one hop; %.0f%% of the load"
                 " would move, relay share %.0f%%%s\n",
                 plan.currentImbalance(), plan.predictedImbalance(),
                 100.0 * plan.movedShare(), 100.0 * plan.relayShare(),
                 remapAbove > 0.0 ? (handOff ? "; handing off to scratch-remap"
                                             : "; executing the plan")
                                  : "");
        if (handOff)
        {
          int migrated = 0;
          remapStep(g, map, nnodes, migrated);
          CkPrintf("lbsim> %-34s scratch-remap: %d migrated (%.0f%% of objects),"
                   " max/avg %.3f -> %.3f, edge cut %d -> %d (%+.1f%%), detached pieces %d\n",
                   p.name.c_str(), migrated, 100.0 * migrated / g.n(), imbBefore,
                   imbalance(g, map, nnodes), cutBefore, edgeCut(g, map),
                   cutBefore ? 100.0 * (edgeCut(g, map) - cutBefore) / cutBefore : 0.0,
                   detachedPieces(g, map, nnodes));
          p.map = map;
          p.w = g.w;
          phases.push_back(p);
          continue;
        }

        acrossNode(nodes, costCfg, map, ss);

        // LBSIM_REFINE=1: heal and smooth after the moves, as route mode does,
        // so the multi-step physical execution can be compared on equal terms.
        int healed = 0, smoothed = 0;
        if (getenv("LBSIM_REFINE") != NULL)
        {
          std::vector<double> target(nnodes);
          for (const VNode& n : nodes) target[n.id] = n.my_pseudo_load;
          double total = 0.0;
          for (double x : g.w) total += x;
          const double tol = effMinImbalance * total / nnodes;
          for (int pass = 0; pass < refinePasses; pass++)
          {
            healed += healPieces(g, map, nnodes);
            smoothed += smoothBoundaries(g, map, nnodes, target, tol, pass % 2);
          }
          healed += healPieces(g, map, nnodes);
        }

        CkPrintf("lbsim> %-34s %d pseudo rounds, %d moved (%d priced out), unshed %.2f,"
                 " %d healed, %d smoothed,"
                 " max/avg %.3f -> %.3f, edge cut %d -> %d (%+.1f%%), detached pieces %d\n",
                 p.name.c_str(), ss.rounds, ss.moves, ss.rejected, ss.unshed, healed, smoothed,
                 imbBefore, imbalance(g, map, nnodes), cutBefore, edgeCut(g, map),
                 cutBefore ? 100.0 * (edgeCut(g, map) - cutBefore) / cutBefore : 0.0,
                 detachedPieces(g, map, nnodes));
        // With two dimensions, the max/avg above is the host one; what the
        // balancer minimises is the step, max over nodes of the larger term.
        if (gVectorMode)
          CkPrintf("lbsim>   dimensions: host max/avg %.3f -> %.3f, device %.3f -> %.3f,"
                   " step %.3f -> %.3f; %d candidate(s) refused by the receiver check%s\n",
                   imbBefore, imbalance(g, map, nnodes), devBefore,
                   imbalanceOf(g.w2, map, nnodes), stepBefore, stepImbalance(g, map, nnodes),
                   ss.slackRefused,
                   diffusionStepMode() ? ""
                   : diffusionDeviceDim() ? " (host dimension)" : " (device dimension)");
      }

      p.map = map;
      p.w = g.w;
      phases.push_back(p);
    }

    writeJson("lbsim.json", g, nnodes, phases);
    for (VNode& n : nodes) delete n.st;
  }
}

#ifdef LBSIM_STANDALONE
// lbsim_stub.C: takes the +LB flags out of argv into _lb_args, the way the
// runtime's LBManager does in the Charm++ build.
void lbsimParseArgs(int& argc, char** argv);
int main(int argc, char** argv)
{
  lbsimParseArgs(argc, argv);
  lbsimRun(argc, argv);
  return 0;
}
#else
class Main : public CBase_Main
{
public:
  Main(CkArgMsg* m)
  {
    lbsimRun(m->argc, m->argv);
    delete m;
    CkExit();
  }
};

#include "lbsim.def.h"
#endif
