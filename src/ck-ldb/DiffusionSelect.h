#ifndef _DIFFUSION_SELECT_H
#define _DIFFUSION_SELECT_H

// The across-node selection loop of DiffusionLB as a pure function over a
// metric: which objects leave this node, for which neighbour, in what order.
// DiffusionLB::AcrossNodeLB calls it and then does the messaging for each move
// it returns; the offline simulator (tests/charm++/load_balancing/lbdriver/
// lbsim.C) calls it and applies the moves to its map. One loop, two callers.

#include <functional>
#include <vector>

#include "BaseLB.h"
#include "DiffusionLoad.h"
#include "DiffusionMetric.h"
#include "LBManager.h"

struct DiffusionMove
{
  int obj;          // index into nodeStats.objData
  int nbor;         // index into the node's neighbour list
  double shedLoad;  // the object's load in the diffused dimension
};

// Runs the selection loop until the node's obligation is met, the per-step cap
// is reached, or every neighbour has refused a candidate.
//
//   metric        the selection metric, already built for this node's stats
//   nodeStats     the node's objects (read for their loads)
//   toSendLoad    per-neighbour quota from the pseudo rounds; only its sign is
//                 read here (the metric holds its own copy and retires it)
//   remaining     IN/OUT: the load this node still has to shed
//   maxShed       the per-step cap in the diffused dimension
//   allowedEnds   for a 1-D keyed node, fills the per-object allow mask for a
//                 neighbour (DiffusionLB::allowedEndsFor); null otherwise
//   myNodeId      for diagnostics only
//   moves         OUT: accepted moves, in acceptance order
//   shedThisStep  IN/OUT: load accepted so far this step
//   movesThisStep IN/OUT: moves accepted so far this step
inline void diffusionSelectMoves(
    DiffusionMetric& metric, const BaseLB::LDStats& nodeStats,
    const std::vector<double>& toSendLoad, double& remaining, double maxShed,
    const std::function<void(int, std::vector<char>&)>& allowedEnds, int myNodeId,
    std::vector<DiffusionMove>& moves, double& shedThisStep, int& movesThisStep)
{
  const int neighborCount = (int)toSendLoad.size();
  int loadReceivers = 0;
  for (double q : toSendLoad)
    if (q > 0) loadReceivers++;
  if (loadReceivers <= 0 || neighborCount <= 0) return;

  std::vector<char> tries(neighborCount, 0);
  std::vector<char> allowed;

  // The quota the rounds actually planned per neighbour. A destination that
  // never appears here was refused by the FLOW gate; one that appears with a
  // quota but yields no candidate below was refused by the METRIC.
  if (_lb_args.debug() > 1)
    for (int i = 0; i < neighborCount; i++)
      CkPrintf("[QUOTA node %d] nbor idx %d: planned %.6f\n", myNodeId, i, toSendLoad[i]);

  // Stay with one neighbour until it can take nothing more, then move on --
  // rather than advancing to the next neighbour after every accepted move.
  //
  // The rotation this replaces defeated the metric's own locality mechanism.
  // MetricComm::updateState re-scores after each move so that the NEXT
  // candidate is the one adjacent to the object just sent: that is what makes
  // a shed peel a contiguous chunk off the boundary instead of taking objects
  // from all over. Rotating the destination immediately handed that carefully
  // chosen neighbour-of-the-last-move to a DIFFERENT PE, so the two mechanisms
  // worked against each other and the departing set came out interleaved.
  //
  // Measured on the lbdriver stencil (32x32 chares, 4 one-PE nodes, a 12x
  // heavy disc dropped on a MetisLB partition): with the rotation, all 64
  // moved objects alternated between two destinations along each row, the
  // edge cut went 72 -> 113 and one PE was left in three disconnected pieces.
  //
  // Each recipient is still capped by its own quota (toSendLoad), so draining
  // one neighbour cannot overload it; and the termination condition is
  // unchanged, since a neighbour that yields no candidate is marked in tries[]
  // exactly as before.
  int nid = 0;
  while (remaining > 0)
  {
    const int nborId = nid;

    if (shedThisStep >= maxShed)
    {
      if (_lb_args.debug() > 1)
        CkPrintf("[node %d] AcrossNodeLB: shed cap %.6f reached after %d move(s), "
                 "%.6f left unshed\n",
                 myNodeId, maxShed, movesThisStep, remaining);
      break;
    }

    // What this node still owes, so the metric can cap a candidate's benefit:
    // load shed beyond the fair share buys nothing and must not pay for a
    // move. Refreshed every iteration because each accepted move reduces it.
    metric.setRemainingShed(remaining);

    // With a 1-D ordering key only the interval's ends may go, and only
    // toward the side this neighbour is on. Both metrics honour the filter.
    if (allowedEnds)
    {
      allowedEnds(nborId, allowed);
      metric.setAllowed(&allowed);
    }

    int v_id = metric.popBestObject(nborId);
    // Both metrics refuse a zero-load candidate; should one ever not, it is
    // treated as no candidate here rather than retiring nothing forever.
    if (v_id != -1 && diffusionObjLoad(nodeStats.objData[v_id]) <= 0.0) v_id = -1;

    if (v_id == -1)
    {
      if (_lb_args.debug() > 1)
        CkPrintf("[NOCAND node %d] nbor idx %d: quota %.6f, metric supplied no object\n",
                 myNodeId, nborId, toSendLoad[nborId]);
      tries[nborId] = 1;
      bool not_done = false;
      for (int i = 0; i < neighborCount; i++)
        if (tries[i] == 0)
          not_done = true;
      if (!not_done)
        break;  // no more objects to send
      // This neighbour is done; advance to the next one still open. Skipping
      // the exhausted ones matters now that the cursor no longer moves on its
      // own -- otherwise the loop would sit on a neighbour that can never
      // supply a candidate again.
      do { nid = (nid + 1) % neighborCount; } while (tries[nid] != 0);
      continue;
    }

    // In the diffused dimension: it retires this node's obligation and the
    // per-neighbour quota, both of which the pseudo-LB rounds expressed in that
    // dimension. Using anything else would retire the budget in different units
    // from the ones it was computed in. Unfloored, like the budget it retires
    // (BuildStats): the metric never hands back a zero-load object, so this is
    // positive, and crediting a move with load the object does not carry is
    // exactly what let a node "retire" a quarter of its obligation by shedding
    // its empty objects.
    const double shedLoad = diffusionObjLoad(nodeStats.objData[v_id]);

    remaining -= shedLoad;
    metric.updateState(v_id, nborId);  // update state to keep track of migrations
    movesThisStep++;
    shedThisStep += shedLoad;
    moves.push_back(DiffusionMove{v_id, nborId, shedLoad});
  }

  // The mask is a local of this function; do not leave the metric pointing at it.
  metric.setAllowed(NULL);
}

#endif
