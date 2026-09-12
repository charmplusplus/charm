
// The class declarations are in DiffusionMetric.h so the offline simulator can
// use the metrics; the definitions stay here, in DiffusionLB's translation unit.
#include "DiffusionMetric.h"




MetricComm::MetricComm(BaseLB::LDStats* ns, int nodeId, int nodeSize_, int nCount,
                       std::vector<double> tSL, std::vector<int> sendToNbrs, 
                       double &internalbytes, double &externalbytes,
                       const DiffusionCostConfig* cfg)
    : nodeStats(ns),
      myNodeId(nodeId),
      nodeSize(nodeSize_),
      neighborCount(nCount),
      n_objs(ns->objData.size()),
      toSendLoad(tSL),
      sendToNeighbors(sendToNbrs),
      costCfg((cfg != NULL && cfg->calibrated) ? cfg : NULL),
      localTier(DIFF_TIER_INTRA_PROCESS),
      remainingShed(0.0)
{
  // Which transport each neighbour is reached over. A DiffusionLB node is a
  // Charm node, i.e. a process, so its rank0 PE stands for the whole node here.
  //
  // Except under CHARM_DIFFUSION_NODE_SIZE, where the "nodes" are PE groups
  // inside one process and tierBetween would call every pair intra-process.
  // localTier would then equal every destTier, commDelta would be identically
  // zero, and the cost model would price communication at nothing -- silently
  // reducing itself to migration cost alone, which is the one behaviour it
  // exists to avoid. The override says to treat these groups as separate nodes,
  // so their transfers are priced as separate nodes too.
  const bool logicalNodes = (getenv("CHARM_DIFFUSION_NODE_SIZE") != NULL);
  nborTier.resize(neighborCount, DIFF_TIER_INTER_NODE);
  for (int i = 0; i < neighborCount && i < (int)sendToNeighbors.size(); i++)
    nborTier[i] = logicalNodes
                      ? DIFF_TIER_INTER_NODE
                      : DiffusionCostConfig::tierBetween(myNodeId * nodeSize,
                                                   sendToNeighbors[i] * nodeSize);

  internalComm.resize(n_objs, 0);
  internalMsgs.resize(n_objs, 0);
  for (int nbor = 0; nbor < neighborCount; nbor++)
  {
    std::vector<double> nborComm;
    nborComm.resize(n_objs, 0);
    externalComm.push_back(nborComm);
    externalMsgs.push_back(nborComm);
  }

  objAvailable.resize(n_objs, true);
  movesTo.assign(neighborCount, 0);
  objCommEdges.resize(n_objs);
  for (int edge = 0; edge < nodeStats->commData.size(); edge++)
  {
    LDCommData& commData = nodeStats->commData[edge];

    if ((!commData.from_proc()) && (commData.recv_type() == LD_OBJ_MSG))
    {
      LDObjKey from = commData.sender;
      LDObjKey to = commData.receiver.get_destObj();

      int fromNode = myNodeId;
      int toPE = commData.receiver.lastKnown();
      int toNode = toPE / nodeSize;

      if (fromNode == toNode)
      {
        // internal communication
        int fromObj = nodeStats->getHash(from);
        int toObj = nodeStats->getHash(to);
        // LBDatabase::Send only ever records a locally-running sender, so this
        // should always resolve; guard anyway, because the miss would index
        // internalComm at -1 rather than announce itself.
        if (fromObj == -1 || fromObj >= n_objs)
          continue;
        internalComm[fromObj] += commData.bytes;
        internalMsgs[fromObj] += commData.messages;

        if (toObj != -1 && toObj < n_objs)
        {
          internalComm[toObj] += commData.bytes;
          internalMsgs[toObj] += commData.messages;
          objCommEdges[toObj].push_back(
              CommEdge{fromObj, (double)commData.bytes, (double)commData.messages});
          objCommEdges[fromObj].push_back(
              CommEdge{toObj, (double)commData.bytes, (double)commData.messages});
        }

        internalbytes += commData.bytes;
      }
      else
      {
        int nborId = getNborId(toNode);
        externalbytes += commData.bytes;


        if (nborId == -1)  // could comm with node that is not a "neighbor".. ignore
          continue;

        int fromObj = nodeStats->getHash(from);
        if (fromObj != -1 && fromObj < n_objs)
        {
          // Incident, not outbound. LBDatabase::Send records a message on the
          // sender's node only, so this node sees i->nbor but never nbor->i:
          // that half is in the peer's stats and never reaches us. Internal
          // traffic has no such gap -- both endpoints are local, so both
          // directions are recorded here and internalComm is already incident.
          //
          // Comparing the two as they stood put a bidirectional internal figure
          // against a one-directional external one, understating every external
          // edge by 2x and so making every candidate look more expensive to move
          // than it is. Complete the external side by assuming the reciprocal
          // edge carries what the observed one carries, which holds for the
          // neighbour exchanges this balancer is aimed at. The residual error is
          // a per-object scale factor on one term, which the alpha/beta
          // calibration absorbs; the previous mismatch was a systematic bias
          // between two terms that no calibration could.
          externalComm[nborId][fromObj] += kIncidentFactor * commData.bytes;
          externalMsgs[nborId][fromObj] += kIncidentFactor * commData.messages;
        }
      }
    }
  }
};

// Load band for the uncalibrated pickers below. An object is a candidate only
// if it carries at least this fraction of the heaviest load that fits the
// neighbour's quota; the metric's own criterion (edge cut, distance) ranks
// within the band. 0.5 keeps every object of a uniform application in the
// band -- those rank exactly as before -- while excluding the light objects
// of a skewed one until the heavy ones no longer fit.
static const double kLoadBand = 0.5;

int MetricComm::popBestObject(int nbor)
{
  // Pick the object whose move costs this node the least edge cut: the bytes it
  // already sends across to `nbor` (which the move turns into local traffic)
  // minus the bytes that are local today (which the move turns into cut).
  //
  // Ranking on externalComm alone is only a tie-break away from arbitrary. On a
  // well-placed stencil almost all traffic is internal, so externalComm[nbor][i]
  // is 0 for nearly every object; with a -1 seed the first candidate that fits
  // capacity wins and no later 0 can beat it, so the choice collapses to the
  // lowest object index that fits -- always the same corner of the index space,
  // regardless of who the object talks to. That is what shreds the locality the
  // BLOCK map starts with: measured on pic2d, cross-node traffic went 8.9 -> 48
  // MB across two LB rounds and the iteration time went 40 -> 98 ms.
  //
  // Subtracting internalComm restores a real signal in exactly that case: with
  // no external traffic to discriminate on, the least-wired-in object wins
  // instead of the lowest-numbered one. Doubles throughout -- externalComm and
  // internalComm are byte counts and were being truncated to int.
  // With a calibrated cost table the same loop answers both questions at once:
  // which object is best for this neighbour, and whether moving any of them is
  // worth doing. Splitting those apart would let the cheapest candidate be
  // chosen by one criterion and then vetoed by another, which is how a
  // quota-driven balancer ends up making the least-bad move rather than no move.
  //
  // Score is the marginal effect on this node's per-interval time:
  //   benefit  load this move sheds, capped at what is still owed
  //   cost     the migration itself, amortised, plus the change in steady-state
  //            communication the new placement leaves behind
  // Both sides are seconds per balancer interval, which is the unit the load
  // figures already come in (see the horizon note in DiffusionCostModel.h).
  //
  // Uncalibrated, this degrades to the edge-cut ranking with no veto -- the
  // behaviour before the cost model existed. Refusing moves on guessed
  // constants would be worse than not pricing them at all.
  double bestScore = -std::numeric_limits<double>::max();
  int bestObject = -1;

  double nborCapacity = toSendLoad[nbor];
  // Static, not a temporary: DiffusionCostModel keeps a reference, and binding
  // one to a temporary in a constructor does not extend its lifetime. The
  // uncalibrated path never reads it, but a dangling reference that happens to
  // go untouched is a trap for the next person to add a branch here.
  static const DiffusionCostConfig kNoCostConfig;
  const DiffusionCostModel model(costCfg ? *costCfg : kNoCostConfig, localTier);
  const DiffusionTier destTier =
      (nbor >= 0 && nbor < (int)nborTier.size()) ? nborTier[nbor]
                                                 : DIFF_TIER_INTER_NODE;

  // Load first, edge cut second. The quota this phase retires is load, so the
  // heaviest object that fits is the natural candidate. Ranking on edge cut
  // alone -- what this did before -- picks the LEAST wired object, which on a
  // stencil is the one with nothing to exchange and therefore nothing to shed.
  // Measured on sph2d with 24 fluid patches on one GPU and 13x the load of the
  // other three: the first step moved that node's eight empty patches, its
  // GPU share went UP in the next window, and after three steps it still held
  // 67% of the job's GPU time with 19 of 128 objects.
  //
  // Edge cut keeps its real job inside a band of near-equal loads: of several
  // equally heavy candidates, take the one whose departure cuts the least, so
  // the retained set stays contiguous and updateState pulls the next pick
  // next to it. The calibrated path below already weighs load against
  // communication in one unit and is left as it is.
  // The seed must border the destination.
  //
  // A chunk grows outward from its seed and stays contiguous with itself, but
  // that is not enough: it has to arrive ATTACHED to the receiver. When the
  // seed is chosen on load alone it lands wherever this node's heaviest object
  // happens to be, and if that is not on the shared boundary the whole chunk
  // arrives as an island -- the receiver goes from one region to two, and the
  // partition fragments even though nothing about the move looked wrong
  // locally.
  //
  // Measured on the lbdriver stencil, over six runs: every chunk with no cell
  // bordering its receiver fragmented that receiver, and every chunk with at
  // least one did not. The correlation was exact, and it explained variance I
  // had wrongly attributed to the size of the per-step cap.
  //
  // "Borders the destination" is just externalComm[nbor][i] > 0 -- the object
  // already exchanges messages with something on that node. If no such object
  // exists (this node may not border that neighbour at all, or the comm graph
  // may be empty) the restriction lifts rather than refusing to shed, since an
  // unattached chunk still beats an unmet obligation.
  //
  // Applied on BOTH scoring paths. It was first fenced to the uncalibrated one
  // on the assumption that the calibrated score's communication term would do
  // the same job on its own. Measured, it does not: on the lbdriver stencil a
  // fully interior cell's comm penalty is ~1.6 units against a 12-unit load
  // benefit, so the calibrated path seeded inside the region and fragmented
  // the receiver in 1 of 5 trials. Where a chunk starts is geometry; the cost
  // table prices a move; the first does not depend on the second.
  //
  // Applied to the seed only. Extending it to every pick -- gating the
  // candidate SET on the frontier rather than letting the ranking find it --
  // was tried and is a regression on both paths: the gate has to lift or refuse
  // when no frontier object also clears the band and the capacity filter, and
  // refusing ends the neighbour early, leaving a half-peeled layer. Measured
  // over five trials, the uncalibrated path went +1% edge cut and 66 moves to
  // +14% and 40, with cut worsening as moves fell (34 moves -> +25%, 53 -> +1%).
  // A partial peel has more perimeter than either a whole one or none.
  const bool seeding =
      (nbor >= 0 && nbor < (int)movesTo.size()) ? (movesTo[nbor] == 0) : true;
  bool seedOnBoundary = false;
  if (seeding)
  {
    for (int i = 0; i < n_objs; i++)
    {
      if (!objAvailable[i] || !nodeStats->objData[i].migratable || !isAllowed(i)) continue;
      if (diffusionObjLoad(nodeStats->objData[i]) > nborCapacity) continue;
      if (!slackFits(nodeStats->objData[i], nbor)) continue;
      if (externalComm[nbor][i] > 0.0) { seedOnBoundary = true; break; }
    }
  }

  // What the band ranks on. One dimension: the object's load in it. Both
  // (LB_MODE_STEP): the load it takes off this node's binding term per unit
  // of step time it adds to the receiver -- an object that lands on the
  // receiver's slack term costs it nothing and ranks highest, one that lands
  // on its binding term ranks as its load would. That is the dot-product
  // rule of vector bin packing in local form, and the only place the second
  // dimension enters the ranking; the edge-cut ranking below it is unchanged.
  const bool stepMode = diffusionStepMode();
  auto effOf = [&](int i) {
    const double l = diffusionObjLoad(nodeStats->objData[i]);
    if (!stepMode) return l;
    const double rise = receiverRise(nodeStats->objData[i], nbor, nodeSize);
    return l / (rise + 0.01 * l + 1e-12);
  };

  double heaviest = 0.0;
  if (costCfg == NULL)
  {
    int refusedHere = 0;
    for (int i = 0; i < n_objs; i++)
    {
      if (!objAvailable[i] || !nodeStats->objData[i].migratable || !isAllowed(i)) continue;
      if (seedOnBoundary && externalComm[nbor][i] <= 0.0) continue;
      const double objLoad = diffusionObjLoad(nodeStats->objData[i]);
      if (objLoad > nborCapacity) continue;
      if (!slackFits(nodeStats->objData[i], nbor)) { refusedHere++; continue; }
      const double e = effOf(i);
      if (e > heaviest) heaviest = e;
    }
    // Nothing with measurable load fits. Moving a zero-load object retires
    // nothing and still costs a migration, so there is no candidate -- not
    // the lightest one, which is what a plain minimum would hand back. When
    // the slack check is what emptied the field, say so in the count.
    if (heaviest <= 0.0)
    {
      slackRefusals += refusedHere;
      return -1;
    }
  }

  for (int i = 0; i < n_objs; i++)
  {
    if (!objAvailable[i]) continue;
    if (!nodeStats->objData[i].migratable) continue;
    if (!isAllowed(i)) continue;

    double objLoad = diffusionObjLoad(nodeStats->objData[i]);
    if (objLoad > nborCapacity) continue;
    if (!slackFits(nodeStats->objData[i], nbor)) { slackRefusals++; continue; }
    if (seedOnBoundary && externalComm[nbor][i] <= 0.0) continue;

    double score;
    if (costCfg == NULL)
    {
      // The band picks the SEED of a chunk, not every object in it.
      //
      // Its job is to stop a node shedding its empty objects while its heavy
      // ones stay put (sph2d: eight empty fluid patches moved, the node's GPU
      // share went up). That risk is real only for the first pick, which has no
      // context to go on but load. Once a move to this neighbour has been
      // accepted, updateState has re-scored the departing object's partners so
      // the edge-cut term names the objects ADJACENT to it -- and applying the
      // band again there throws that away, because it re-ranks by load and
      // jumps to whatever heavy object sits elsewhere on the node.
      //
      // That is what carved the hot region on the lbdriver stencil: every one of
      // the 64 moved objects was a heavy cell, drawn from across the disc rather
      // than peeled off its boundary, so the partition came apart. Growing the
      // chunk instead lets cheap boundary objects join the move, which is how a
      // partition boundary shifts without fragmenting.
      //
      // Zero-load objects stay excluded even while growing: they retire no
      // budget, so a chunk made of them would never end.
      if (seeding && effOf(i) < kLoadBand * heaviest) continue;
      if (objLoad <= 0.0) continue;
      score = externalComm[nbor][i] - internalComm[i];
    }
    else
    {
      const double benefit = DiffusionCostModel::loadBenefit(objLoad, remainingShed);
      const double cost =
          model.migrateCost(nodeStats->objData[i]) +
          model.commDelta(internalComm[i], internalMsgs[i], externalComm[nbor][i],
                          externalMsgs[nbor][i], destTier);
      score = benefit - cost;
    }

    if (score > bestScore)
    {
      bestScore = score;
      bestObject = i;
    }
  }

  // A move that does not pay for itself is not made, even with quota left. An
  // unmet quota is the correct answer when meeting it costs more than the
  // imbalance it removes; the caller already treats -1 as "nothing for this
  // neighbour" and stops once every neighbour says so.
  if (costCfg != NULL && bestObject != -1 && bestScore <= 0.0)
  {
    rejectedMoves++;
    bestObject = -1;
  }
  else if (bestObject != -1)
  {
    acceptedMoves++;
  }

  // if (bestObject != -1)
  // {
  //   assert(objAvailable[bestObject]);
  //   objAvailable[bestObject] = false;
  // }
  // else
  // {
  //   CkPrintf("No object found for neighbor %d, with capacity %f\n", nbor,
  //   nborCapacity);
  // }
  return bestObject;
};

int MetricComm::getBestNeighbor()
{
  int bestNeighbor = -1;
  for (int i = 0; i < neighborCount; i++)
  {
    if (toSendLoad[i] > 0)
    {
      bestNeighbor = i;
      break;
    }
  }
  return bestNeighbor;
}

void MetricComm::updateState(int objId, int destNbor)
{
  // Must match what popBestObject weighed against the quota, or the quota is
  // retired in different units from the ones it was tested in.
  double objLoad = diffusionObjLoad(nodeStats->objData[objId]);
  if(_lb_args.debug() > 2)
    CkPrintf("Node %d: migrating obj %d (load %.6f) to neighbor %d (tosend before: %.6f, after: %.6f)\n", 
            myNodeId, objId, objLoad, sendToNeighbors[destNbor], 
            toSendLoad[destNbor], toSendLoad[destNbor] - objLoad);
  toSendLoad[destNbor] -= objLoad;
  slackTake(nodeStats->objData[objId], destNbor);
  if (destNbor >= 0 && destNbor < (int)movesTo.size()) movesTo[destNbor]++;
  if(objId<0 || objId>=n_objs)
    return;
  // Every local edge incident to the departing object stops being local for the
  // partner that stays: it now runs between that partner and destNbor. Both
  // sides of the ledger move together, and both dimensions move with them, so
  // the next candidate is weighed against the partition this move just created.
  for (const CommEdge& edge : objCommEdges[objId])
  {
    int toObj = edge.obj;
    if(toObj<0 || toObj>=n_objs) CkAbort("Error: invalid toObj %d in MetricComm::updateState\n", toObj);
    if (objAvailable[toObj])
    {
      externalComm[destNbor][toObj] += edge.bytes;
      internalComm[toObj] -= edge.bytes;
      externalMsgs[destNbor][toObj] += edge.msgs;
      internalMsgs[toObj] -= edge.msgs;
    }
  }
  objAvailable[objId] = false;
}

MetricCentroid::MetricCentroid(std::vector<std::vector<double>> nborCentroids,
                               std::vector<double> nborDistances,
                               std::vector<LBRealType> myCentroid, BaseLB::LDStats* ns,
                               int nodeId, std::vector<double> tSL,
                               std::vector<int> sendToNbrs, std::vector<int> nborObjCount)
    : nodeStats(ns),
      myNodeId(nodeId),
      nborCentroids(nborCentroids),
      myCentroid(myCentroid),
      nborDistances(nborDistances),
      toSendLoad(tSL),
      sendToNeighbors(sendToNbrs),
      nborObjCount(nborObjCount)
{
  position_dim = myCentroid.size();
  neighborCount = nborCentroids.size();
  n_objs = ns->objData.size();

  if (sendToNeighbors.size() != neighborCount)
  {
    CkAbort("Error: on node %d, sendToNeighbors size %d does not match neighborCount %d\n",
            myNodeId, sendToNeighbors.size(), neighborCount);
  }

  objAvailable.resize(n_objs, true);
  objPosition.resize(n_objs);
  objNborDistances.resize(n_objs);

  for (int i = 0; i < n_objs; i++)
  {
    if (ns->objData[i].position.size() == 0)
    {
      continue;  // Skip objects without position data, but process the rest
    }
    else if (ns->objData[i].position.size() != position_dim)
    {
      CkAbort("Error: object %d has position with %d dimensions, expected %d\n", i,
              ns->objData[i].position.size(), position_dim);
    }
    objPosition[i].resize(position_dim);
    for (int j = 0; j < position_dim; j++)
    {
      objPosition[i][j] = ns->objData[i].position[j];
    }

    objNborDistances[i].resize(neighborCount);
    for (int j = 0; j < neighborCount; j++)
    {
      objNborDistances[i][j] = computeDistance(objPosition[i], nborCentroids[j]);
    }
  }
  // print sendtoneighbors
  // std::string neighbors = "Node %d has neighbors: ";
  // for (int i = 0; i < neighborCount; i++)
  // {
  //   neighbors += std::to_string(sendToNeighbors[i]) + " (" +
  //                std::to_string(nborDistances[i]) + ") ";
  // }
  // neighbors += "\n";
  // CkPrintf(neighbors.c_str(), myNodeId);

  // print object neighbor distances
  // for (int i = 0; i < n_objs; i++)
  // {
  //   std::string distances = "Node %d: Object %d (load %f) has distances: ";
  //   for (int j = 0; j < neighborCount; j++)
  //   {
  //     distances += std::to_string(sendToNeighbors[j]) + " (" +
  //                  std::to_string(objNborDistances[i][j]) + ") ";
  //   }
  //   distances += "\n";
  //   CkPrintf(distances.c_str(), myNodeId, i, ns->objData[i].wallTime);
  // }
}

int MetricCentroid::popBestObject(int nbor)
{
  // find index of object with min distance to neighbor centroid
  double minDistance = std::numeric_limits<double>::max();
  int bestObject = -1;
  
  // Validate neighbor index
  if (nbor < 0 || nbor >= neighborCount)
  {
    CkAbort("Error: on node %d, invalid neighbor index %d (must be 0 to %d) in MetricCentroid::popBestObject\n", 
            myNodeId, nbor, neighborCount - 1);
  }
  
  // Snapshot the current capacity to avoid stale reads
  double nborCapacity = toSendLoad[nbor];

  if(_lb_args.debug() == 3)
    CkPrintf("Node %d: popBestObject for neighbor %d with capacity %.6f\n", myNodeId, nbor, nborCapacity);

  // With a one-dimensional position the application is handing us an ORDERING
  // key, not a point in space -- an SFC index, say -- and asking that this
  // node keep one contiguous interval of it. Only the two ends of the interval
  // may leave.
  //
  // The distance rule alone is a preference, and it is not enough: the cost
  // this is really minimising is each PE's bounding box, which is set by
  // extremes, so a few objects left far from the centre inflate it as much as
  // a fully scattered set would. Shaving only the ends keeps the remaining set
  // an interval by construction, so its extent shrinks monotonically instead
  // of merely tending to.
  int loEnd = -1, hiEnd = -1;
  if (position_dim == 1)
  {
    double loKey = 0, hiKey = 0;
    for (int i = 0; i < n_objs; i++)
    {
      if (!objAvailable[i] || !nodeStats->objData[i].migratable) continue;
      if (nodeStats->objData[i].position.size() != 1) continue;
      const double k = nodeStats->objData[i].position[0];
      if (loEnd == -1 || k < loKey) { loEnd = i; loKey = k; }
      if (hiEnd == -1 || k > hiKey) { hiEnd = i; hiKey = k; }
    }
  }

  // Load first, distance second, for the reason given at
  // MetricComm::popBestObject: the quota is load, and a picker that ignores it
  // hands over the objects with nothing to shed. Distance ranks inside the
  // band of near-equal loads.
  double heaviest = 0.0;
  for (int i = 0; i < n_objs; i++)
  {
    if (position_dim == 1 && i != loEnd && i != hiEnd) continue;
    if (!isAllowed(i) || !objAvailable[i] || !nodeStats->objData[i].migratable) continue;
    if (objNborDistances[i].size() <= nbor) continue;
    const double objLoad = diffusionObjLoad(nodeStats->objData[i]);
    if (objLoad > nborCapacity) continue;
    if (!slackFits(nodeStats->objData[i], nbor)) continue;
    if (objLoad > heaviest) heaviest = objLoad;
  }
  if (heaviest <= 0.0) return -1;  // nothing with measurable load fits

  for (int i = 0; i < n_objs; i++)
  {
    double objLoad = diffusionObjLoad(nodeStats->objData[i]);

    if (position_dim == 1 && i != loEnd && i != hiEnd) continue;
    if (!isAllowed(i)) continue;

    if (objNborDistances[i].size() <= nbor)
    {
      if (nodeStats->objData[i].position.size() != 0){
        CkAbort("Error: on node %d invalid neighbor %d for object %d in MetricCentroid::popBestObject\n", myNodeId, nbor, i);
      }
      continue;
    }
    double testDistance = objNborDistances[i][nbor];
    bool migratable = nodeStats->objData[i].migratable;
    bool available = objAvailable[i];

    if (testDistance < minDistance && available && migratable &&
        (objLoad <= nborCapacity) && objLoad >= kLoadBand * heaviest &&
        slackFits(nodeStats->objData[i], nbor))
    {
      minDistance = testDistance;
      bestObject = i;
    } else if (_lb_args.debug() == 3) {
      if (myNodeId == 5) {
        CkPrintf("Node %d: Object %d rejected - ", myNodeId, i);
        if (testDistance >= minDistance)
          CkPrintf("distance %.6f >= current min %.6f ", testDistance, minDistance);
        if (!available)
          CkPrintf("not available ");
        if (!migratable)
          CkPrintf("not migratable ");
        if (objLoad > nborCapacity)
          CkPrintf("load %.6f > capacity %.6f", objLoad, nborCapacity);
        CkPrintf("\n");
      }
    }
  }

  return bestObject;
}

int MetricCentroid::getBestNeighbor()
{
  int bestNeighbor = -1;

  for (int i = 0; i < neighborCount; i++)
  {
    if (toSendLoad[i] > 0)
    {
      bestNeighbor = i;
      break;
    }
  }
  return bestNeighbor;
}

void MetricCentroid::updateState(int objId, int destNbor)
{
  if(objId<0 || objId>=n_objs)
    CkAbort("Error: invalid objId %d in MetricCentroid::updateState\n", objId);
  objAvailable[objId] = false;
  toSendLoad[destNbor] -= diffusionObjLoad(nodeStats->objData[objId]);
  slackTake(nodeStats->objData[objId], destNbor);
  if(nborCentroids.size()<=destNbor) return;

  // TODO: update my centroid (not used anywhere rn)
  // update neighbor centroid
  std::vector<LBRealType> old = nborCentroids[destNbor];
  int count = nborObjCount[destNbor];
  for (int i = 0; i < position_dim; i++)
  {
    nborCentroids[destNbor][i] = (old[i] * count + objPosition[objId][i]) / (count + 1);
  }

  // update objNborDistances
  for (int i = 0; i < n_objs; i++)
  {
    if (objNborDistances[i].size() <= destNbor)
    {
      continue;
    }
    objNborDistances[i][destNbor] =
        computeDistance(objPosition[i], nborCentroids[destNbor]);
  }
}
