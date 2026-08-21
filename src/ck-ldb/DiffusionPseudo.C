void DiffusionLB::startStrategyBarrier()
{
  if (++rank0_barrier_counter < numNodes)
    return;

    rank0_barrier_counter = 0;

  startStrategy();
}

void DiffusionLB::startStrategy(){
  // End neighbor selection timing
  endNeighborTiming();

  if (CkMyPe() == 0 && numNodes == 1) {
    CkCallback cb(CkIndex_DiffusionLB::WithinNodeLB(), thisProxy);
    CkStartQD(cb);
  }
  else if (CkMyPe() == 0)
  {
    CkCallback cb(CkIndex_DiffusionLB::AcrossNodeLB(), thisProxy);
    CkStartQD(cb);
  }

  if (_lb_args.debug() > 1) CkPrintf("--------NEIGHBOR SELECTION COMPLETE (Using Comm? %s)--------\n",
           _lb_args.diffusionCommOn() ? "true" : "false");
  fflush(stdout);
  
  // Start pseudo LB timing
  startPseudoLBTiming();
  
  if (numNodes > 1)
  for (int i = 0; i < numNodes; i++) thisProxy[i * nodeSize].pseudolb_rounds();
}


// pseudolb_barrier removed. It was the global convergence check for the pseudo-LB
// rounds: every node reported "nothing left to send" to PE 0, which ANDed the votes
// and broadcast the verdict so all nodes could break out of the round loop early.
//
// Two reasons it went. It was the only global synchronisation in an otherwise
// neighbour-local phase, costing an O(N) fan-in plus a broadcast every round -- a
// central coordinator inside a balancer whose whole premise is not having one. And
// its benefit shrinks as the machine grows: the exit condition is unanimous, so at
// large node counts some node almost always still has a slightly underloaded
// neighbour and the check never fires while still being paid for every round.
//
// The round loop is bounded by ITERATIONS regardless, so removing the check cannot
// run forever; it only means always running the full count. Second-order diffusion
// in PseudoLoadBalancing is what makes that fixed count sufficient.


/* In combination with the pseudolb_rounds SDAG code, this builds the toReceiveLoad and
 * toSendLoad vectors for each node. It is onlyl called on rank0PEs*/
void DiffusionLB::PseudoLoadBalancing()
{
  std::vector<double> thisRoundToSend(sendToNeighbors.size(), 0.0);

  // Define threshold as a percentage of average neighbor load
  // Prevents micro-migrations when loads are very similar
  const double THRESHOLD_PERCENT = 1.0;  // 1% threshold
  double avgLoadNeighbor = std::accumulate(loadNeighbors.begin(), loadNeighbors.begin() + neighborCount, 0.0) / neighborCount;
  double threshold = THRESHOLD_PERCENT * avgLoadNeighbor / 100.0;

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
  // is empty, and the unified send at the end still emits one message per neighbour,
  // which the SDAG round requires.

  // balance with neighborstobalance
  double myOverload = my_pseudo_load - currAverage;

  // Don't bother balancing if my overload is insignificant
  if (myOverload < threshold)
  {
    myOverload = 0;
  }

  // adjust my overload for what I've already sent out
  double alreadySent = std::accumulate(toSendLoad.begin(), toSendLoad.end(), 0.0,
                                       [](double sum, double value)
                                       { return value > 0 ? sum + value : sum; });

  double leftToSend = my_load - alreadySent;  // my_load is original load
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
  const double BETA = _lb_args.diffusionBeta();

  double totalSend = 0.0;
  for (int i = 0; i < neighborCount; i++)
  {
    double flow = BETA * thisRoundToSend[i] + (BETA - 1.0) * prevRoundToSend[i];

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

  // Commit: record the flow for next round's momentum, charge it against this
  // node's notional load, and tell each neighbour what it is receiving. Exactly one
  // message per neighbour per round -- the SDAG round waits for that many.
  for (int i = 0; i < neighborCount; i++)
  {
    int nbor_node = sendToNeighbors[i];

    toSendLoad[i] += thisRoundToSend[i];
    prevRoundToSend[i] = thisRoundToSend[i];
    my_pseudo_load -= thisRoundToSend[i];

    thisProxy[nbor_node * nodeSize].PseudoLoad(pseudo_itr, thisRoundToSend[i], myNodeId);
  }

  // double threshold = THRESHOLD * avgLoadNeighbor / 100.0;

  // avgLoadNeighbor = (avgLoadNeighbor + my_pseudo_load) / 2;
  // double totalOverload = my_pseudo_load - avgLoadNeighbor;
  // double totalUnderLoad = 0.0;
  // double thisIterToSend[neighborCount];
  // for (int i = 0; i < neighborCount; i++) thisIterToSend[i] = 0.0;
  // if (totalOverload > 0)
  //   for (int i = 0; i < neighborCount; i++)
  //   {
  //     if (loadNeighbors[i] < (avgLoadNeighbor - threshold))
  //     {
  //       thisIterToSend[i] = avgLoadNeighbor - loadNeighbors[i];
  //       totalUnderLoad += avgLoadNeighbor - loadNeighbors[i];
  //       //        DEBUGL2(("[PE-%d] iteration %d thisIterToSend %f avgLoadNeighbor %f
  //       //        loadNeighbors[%d] %f to node %d\n",
  //       //                thisIndex, itr, thisIterToSend[i], avgLoadNeighbor, i,
  //       //                loadNeighbors[i], sendToNeighbors[i]));
  //     }
  //   }
  // if (totalUnderLoad > 0 && totalOverload > 0 && totalUnderLoad > totalOverload)
  //   totalOverload += threshold;
  // else
  //   totalOverload = totalUnderLoad;

  // for (int i = 0; i < neighborCount; i++)
  // {
  //   if (totalOverload > 0 && totalUnderLoad > 0 && thisIterToSend[i] > 0)
  //   {
  //     //      DEBUGL2(("[%d] GRD: Pseudo Load Balancing Sending, iteration %d node
  //     //      %d(pe-%d) toSend %lf totalToSend %lf\n", CkMyPe(), itr,
  //     //      sendToNeighbors[i], CkNodeFirst(sendToNeighbors[i]), thisIterToSend[i],
  //     //      (thisIterToSend[i]*totalOverload)/totalUnderLoad));
  //     thisIterToSend[i] *= totalOverload / totalUnderLoad;
  //     toSendLoad[i] += thisIterToSend[i];
  //     if (my_pseudo_load - thisIterToSend[i] < 0)
  //       CkAbort("Error: my_pseudo_load (%f) - thisIterToSend[i] (%f) < 0\n",
  //               my_pseudo_load, thisIterToSend[i]);
  //     my_pseudo_load -= thisIterToSend[i];
  //   }
  //   if (thisIterToSend[i] < 0.0)
  //     thisIterToSend[i] = 0.0;
  //   int nbor_node = sendToNeighbors[i];
  //   thisProxy[nbor_node * nodeSize].PseudoLoad(pseudo_itr, thisIterToSend[i],
  //   myNodeId);
  // }
}
