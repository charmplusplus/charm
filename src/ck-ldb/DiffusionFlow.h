#ifndef _DIFFUSION_FLOW_H
#define _DIFFUSION_FLOW_H

// The pseudo-LB round of DiffusionLB as a pure function: one node's flows for
// one round, computed from that node's view of its neighbourhood and nothing
// else. DiffusionLB::PseudoLoadBalancing gathers the inputs from the chare and
// commits the result; the offline simulator (tests/charm++/load_balancing/
// lbdriver/lbsim.C) runs N nodes through the same function in lockstep. There
// is one copy of the arithmetic, and the simulator is a simulation of it rather
// than a second implementation that drifts.

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <utility>
#include <vector>

// Rounds of the pseudo-load diffusion loop. Fixed, with no convergence check:
// every round costs two neighbour exchanges and the SDAG waits between them, so
// the strategy pays all 40 even when the load is already even. On a GPU-bound
// run where diffusion wants to move ~1% of the load, that is the single largest
// cost of load balancing -- larger than the migration it decides on.
// CHARM_LB_DIFFUSION_ITERS overrides it so the trade can be measured.
inline int diffusionIterations()
{
  static const int n = []() {
    const char* s = getenv("CHARM_LB_DIFFUSION_ITERS");
    const int v = s ? atoi(s) : 40;
    return v > 0 ? v : 40;
  }();
  return n;
}

// Diffusion rounds stop once no node wants to shift more than this fraction of
// its own load. Measured on a GPU-bound run, diffusion asks to move ~1.6% on the
// first round and converges immediately after, so the fixed 40 rounds were
// almost entirely wasted. The default is THRESHOLD (DiffusionLB.C) percent.
inline double diffusionPseudoConvergeRatio()
{
  static const double r = []() {
    const char* s = getenv("CHARM_LB_DIFFUSION_CONVERGE");
    const double v = s ? atof(s) : 0.02;
    return v > 0.0 ? v : 0.02;
  }();
  return r;
}

// What the pseudo rounds reveal about the step before any object moves, summed
// over the nodes on PE 0. Enough to tell a one-hop plan from a multi-hop one:
// a node with both planned inflow and outflow is a relay, and the balancer's
// one-hop execution cannot forward what a relay receives. Measured on a
// 128x128 stencil over 256 nodes: a hot disc of any weight, 1.5x to 12x, left
// the interior nodes with no gradient and the balancer never reduced the
// maximum; the relay share was 21% at 1.5x and 50-65% beyond, against 0% when
// the plan was one hop deep. So the decision is on what the plan predicts,
// not on how large the imbalance is.
struct DiffusionPlanSummary
{
  double maxCurrent = 0.0;    // largest node load now
  double maxPredicted = 0.0;  // largest node load if the one-hop plan is executed
  double sumLoad = 0.0;
  double planOut = 0.0;       // planned outflow, summed
  double planRelay = 0.0;     // of which leaves a node that also has inflow
  double planMoved = 0.0;     // load that leaves the node it is on
  int nodes = 0;

  void add(double load, double predicted, double planIn, double planOutNode)
  {
    if (load > maxCurrent) maxCurrent = load;
    if (predicted > maxPredicted) maxPredicted = predicted;
    sumLoad += load;
    planOut += planOutNode;
    planRelay += (planIn < planOutNode) ? planIn : planOutNode;
    if (load > predicted) planMoved += load - predicted;
    nodes++;
  }
  double avg() const { return nodes > 0 ? sumLoad / nodes : 0.0; }
  double currentImbalance() const { return avg() > 0 ? maxCurrent / avg() : 0.0; }
  double predictedImbalance() const { return avg() > 0 ? maxPredicted / avg() : 0.0; }
  double relayShare() const { return planOut > 0 ? planRelay / planOut : 0.0; }
  double movedShare() const { return sumLoad > 0 ? planMoved / sumLoad : 0.0; }
};

// Whether to hand the step to a scratch-remap instead of executing the plan:
// when the plan, executed one hop, would still leave max/avg above
// `remapAbove`. Zero or less never hands off.
inline bool diffusionShouldRemap(const DiffusionPlanSummary& s, double remapAbove)
{
  return remapAbove > 0.0 && s.nodes > 0 && s.predictedImbalance() > remapAbove;
}

// One node's flows for one round.
//
//   my_load          the node's load at the start of the step (never changes)
//   my_pseudo_load   its notional load now, after the rounds so far
//   effMinImbalance  the decision floor, as a fraction of the neighbourhood mean
//   beta             second-order momentum in [1, 2); 1.0 is first-order
//   loadNeighbors    each neighbour's notional load this round
//   flowAdjacent     per neighbour, whether load may flow to it
//                    (DiffusionLB::nborFlowAdjacent); empty means all may
//   toSendLoad       flow committed to each neighbour so far this step, signed
//   prevRoundToSend  last round's flow to each neighbour, for the momentum term
//   thisRoundToSend  OUT: this round's flow to each neighbour
//   relayHoldings    whether a node may plan to send what it has notionally
//                    RECEIVED, not only what it started the step with. False
//                    is the balancer's one-hop execution, where a node can only
//                    ever hand over objects it held at the start of the step;
//                    true is for a plan that will be routed through relays,
//                    without which pseudo load cannot pass through a node
//                    beyond that node's own load and the plan stalls short of
//                    balance on any graph deeper than one hop.
//
// The caller commits: toSendLoad += thisRoundToSend, prevRoundToSend =
// thisRoundToSend, my_pseudo_load -= sum, and tells each neighbour its share.
inline void diffusionRoundFlows(double my_load, double my_pseudo_load,
                                double effMinImbalance, double beta,
                                const std::vector<double>& loadNeighbors,
                                const std::vector<char>& flowAdjacent,
                                const std::vector<double>& toSendLoad,
                                const std::vector<double>& prevRoundToSend,
                                std::vector<double>& thisRoundToSend,
                                bool relayHoldings = false)
{
  const int neighborCount = (int)loadNeighbors.size();
  thisRoundToSend.assign(neighborCount, 0.0);
  if (neighborCount == 0) return;

  // The floor, as a fraction of the average neighbour load: differences under
  // it are noise and produce no flow. This used to be a fixed 1%, far under
  // the step-to-step noise of a balanced GPU run (~5-10% between nodes), so a
  // uniform workload diffused every step and never settled.
  double avgLoadNeighbor =
      std::accumulate(loadNeighbors.begin(), loadNeighbors.end(), 0.0) / neighborCount;
  double threshold = effMinImbalance * avgLoadNeighbor;

  // create pairs for sorting
  std::vector<std::pair<int, double>> nborPairs;
  for (int i = 0; i < neighborCount; i++)
  {
    nborPairs.push_back(std::make_pair(i, loadNeighbors[i]));
  }

  // sort by load
  std::sort(nborPairs.begin(), nborPairs.end(),
            [](const std::pair<int, double>& a, const std::pair<int, double>& b)
            { return a.second < b.second; });

  // find the neighbors that I should balance with (set such that I am the only one with
  // more load than set average)
  std::vector<std::pair<int, double>> nborsToBalance;

  double sumNeighborLoads = 0.0;
  double currAverage = my_pseudo_load;  // start with just me
  for (std::pair<int, double> p : nborPairs)
  {
    int id = p.first;
    double load = p.second;

    // Load flows only to neighbours this node borders (nborFlowAdjacent): the
    // next key interval when the objects carry 1-D keys, a node its objects
    // talk to when they carry comm. A neighbour on the far side of another
    // one still takes part in the threshold's neighbourhood mean -- it is
    // part of the neighbourhood -- but receives no flow from here; the flow
    // reaches it through the node in between, one step later, with every
    // chunk kept whole on the way.
    if (!flowAdjacent.empty() && id < (int)flowAdjacent.size() && !flowAdjacent[id])
      continue;

    // Calculate current average including me and all selected neighbors so far
    currAverage = (my_pseudo_load + sumNeighborLoads) / (nborsToBalance.size() + 1);

    // Only consider neighbors that are significantly underloaded (below threshold)
    if (load >= currAverage - threshold)
    {
      break;
    }

    nborsToBalance.push_back(p);
    sumNeighborLoads += load;
  }
  currAverage = (my_pseudo_load + sumNeighborLoads) / (nborsToBalance.size() + 1);

  // No early return when nborsToBalance is empty. Under second-order diffusion a
  // round with no first-order flow can still carry a decaying tail of the previous
  // round's flow, and that tail is exactly what accelerates convergence -- dropping
  // it would discard the momentum. The loops below are simply no-ops when the list
  // is empty, and the caller still emits one message per neighbour, which the SDAG
  // round requires.

  // balance with neighborstobalance
  double myOverload = my_pseudo_load - currAverage;

  // Don't bother balancing if my overload is insignificant.
  if (myOverload < threshold)
  {
    myOverload = 0;
  }

  // adjust my overload for what I've already sent out
  double alreadySent = std::accumulate(toSendLoad.begin(), toSendLoad.end(), 0.0,
                                       [](double sum, double value)
                                       { return value > 0 ? sum + value : sum; });

  double leftToSend = my_load - alreadySent;  // my_load is original load
  // A relay may plan to pass on what it holds now, receipts included.
  if (relayHoldings) leftToSend = (my_pseudo_load > 0.0) ? my_pseudo_load : 0.0;
  myOverload = std::min(myOverload, leftToSend);

  // First pass: calculate ideal send amounts (ignoring overload limits)
  // and handle negative edges
  double totalUnderLoad = 0.0;
  std::vector<double> idealSend(neighborCount, 0.0);

  for (std::pair<int, double> p : nborsToBalance)
  {
    int id = p.first;
    double load = p.second;

    double trySend = currAverage - load;

    // First, handle negative edges (past receives we need to offset)
    if (toSendLoad[id] < 0)
    {
      double offset = std::min(-toSendLoad[id], trySend);
      idealSend[id] += offset;
      trySend -= offset;
    }

    // Add remaining ideal send amount
    if (trySend > 0)
    {
      idealSend[id] += trySend;
      totalUnderLoad += trySend;
    }
  }

  // Second pass: scale down proportionally if we don't have enough overload
  // This ensures all neighbors get a fair share
  double scaleFactor = 1.0;
  if (totalUnderLoad > myOverload && totalUnderLoad > 0)
  {
    scaleFactor = myOverload / totalUnderLoad;
  }

  // First-order flows for this round: what plain diffusion would send.
  for (std::pair<int, double> p : nborsToBalance)
  {
    int id = p.first;

    double toSend = idealSend[id] * scaleFactor;

    // Only actually send if the amount is significant (exceeds threshold)
    // This prevents tiny transfers that have high overhead relative to benefit
    if (toSend < threshold)
    {
      toSend = 0;
    }

    thisRoundToSend[id] = toSend;
  }

  // ---- Second-order diffusion --------------------------------------------
  // First-order diffusion is Jacobi iteration on the load vector: each round moves
  // load proportional to the local gradient, so information crosses one edge per
  // round and the error decays by the graph's spectral gap. On path-like graphs that
  // gap scales as 1/D^2, so equilibration takes ~D^2 rounds -- far more than the
  // fixed ITERATIONS budget once the neighbour graph is any size.
  //
  // The second-order scheme (Diekmann, Frommer & Monien) adds momentum, standing in
  // the same relation to first-order diffusion as SOR does to Jacobi:
  //
  //     f_k = BETA * f_firstOrder + (BETA - 1) * f_{k-1}
  //
  // A node that sent load in one direction last round keeps pushing that way, so a
  // gradient no longer has to be rediscovered hop by hop. That improves the round
  // count from ~D^2 toward ~D, which is what makes a fixed round budget viable.
  //
  // BETA must lie in [1, 2): 1.0 disables momentum and reduces this exactly to
  // first-order diffusion; the optimum depends on the graph's second eigenvalue,
  // which is not known here.
  //
  // MEASURED, and the reason the default is 1.0 rather than the textbook 1.5:
  // momentum only pays when the round budget is the binding constraint. At 4 nodes
  // (diameter 2) first-order already converges well inside ITERATIONS, so momentum
  // has nothing to accelerate and only overshoots -- five-run mean final max/avg was
  // 1.115 at BETA=1.5 against 1.060 at BETA=1.0. The regime where it wins is a
  // slow-mixing graph (large diameter, D^2 rounds needed, budget exhausted), which
  // does not exist at this node count. Left as a runtime knob so it can be swept
  // where that regime does exist rather than guessed at here.
  const double BETA = beta;

  double totalSend = 0.0;
  for (int i = 0; i < neighborCount; i++)
  {
    const double prev = (i < (int)prevRoundToSend.size()) ? prevRoundToSend[i] : 0.0;
    double flow = BETA * thisRoundToSend[i] + (BETA - 1.0) * prev;

    // Momentum may sustain or accelerate a flow, never reverse it: a negative send
    // would mean pulling load back, which this protocol's accounting (alreadySent
    // sums only positive entries) does not model.
    if (flow < threshold)
    {
      flow = 0.0;
    }

    thisRoundToSend[i] = flow;
    totalSend += flow;
  }

  // Momentum can push the total past what this node actually still holds. The
  // first-order path was bounded by scaleFactor against myOverload; re-apply the
  // same bound to the boosted flows.
  if (totalSend > leftToSend && totalSend > 0.0)
  {
    const double rescale = (leftToSend > 0.0) ? (leftToSend / totalSend) : 0.0;
    for (int i = 0; i < neighborCount; i++)
      thisRoundToSend[i] *= rescale;
  }
}

#endif
