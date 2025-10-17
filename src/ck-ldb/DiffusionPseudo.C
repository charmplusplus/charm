void DiffusionLB::startStrategy()
{
  if (++rank0_barrier_counter < numNodes)
    return;

  if (CkMyPe() == 0 && numNodes == 1) {
    CkCallback cb(CkIndex_DiffusionLB::WithinNodeLB(), thisProxy);
    CkStartQD(cb);
  }
  else if (CkMyPe() == 0)
  {
    CkCallback cb(CkIndex_DiffusionLB::AcrossNodeLB(), thisProxy);
    CkStartQD(cb);
  }

  rank0_barrier_counter = 0;
  if (_lb_args.debug()) CkPrintf("--------NEIGHBOR SELECTION COMPLETE (Using Comm? %s)--------\n",
           _lb_args.diffusionCommOn() ? "true" : "false");
  fflush(stdout);
  for (int i = 0; i < numNodes; i++) thisProxy[i * nodeSize].pseudolb_rounds();
}


void DiffusionLB::pseudolb_barrier(int allZero)
{
  if (!allZero)
  {
    pseudo_done = false;
  }

  if (++rank0_barrier_counter < numNodes)
    return;

  for (int node = 0; node < numNodes; node++)
  {
    thisProxy[node * nodeSize].pseudoDone(pseudo_done);
  }
  pseudo_done = true;  // set up for next round
  rank0_barrier_counter = 0;
}


/* In combination with the pseudolb_rounds SDAG code, this builds the toReceiveLoad and
 * toSendLoad vectors for each node. It is onlyl called on rank0PEs*/
void DiffusionLB::PseudoLoadBalancing()
{
  std::vector<double> thisRoundToSend(sendToNeighbors.size(), 0.0);

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

  double currAverage = my_pseudo_load;
  for (std::pair<int, double> p : nborPairs)
  {
    int id = p.first;
    double load = p.second;

    if (load >= currAverage)
    {
      break;
    }

    nborsToBalance.push_back(p);
    currAverage =
        (currAverage * nborsToBalance.size() + load) / (nborsToBalance.size() + 1);
  }

  // balance with neighborstobalance
  double myOverload = my_pseudo_load - currAverage;

  // adjust my overload for what I've already sent out
  double alreadySent = std::accumulate(toSendLoad.begin(), toSendLoad.end(), 0.0,
                                       [](double sum, double value)
                                       { return value > 0 ? sum + value : sum; });

  double leftToSend = my_load - alreadySent;  // my_load is original load
  myOverload = std::min(myOverload, leftToSend);

  for (std::pair<int, double> p : nborsToBalance)
  {
    int id = p.first;
    double load = p.second;

    double trySend = currAverage - load;
    double toSend = 0;

    // exhaust a negative edge first
    if (toSendLoad[id] < 0)
    {
      toSend += std::min(-toSendLoad[id], trySend);
      trySend -= toSend;
    }

    // either edge was enough (trySend == 0)
    // or we need to send more

    if (trySend > 0)
    {
      toSend += std::min(myOverload, trySend);
      trySend -= toSend;
      myOverload -= toSend;
    }

    toSendLoad[id] += toSend;
    thisRoundToSend[id] = toSend;
  }

  bool allZero = true;

  for (int i = 0; i < neighborCount; i++)
  {
    int nbor_node = sendToNeighbors[i];

    if (thisRoundToSend[i] > 0)
    {
      allZero = false;
    }

    my_pseudo_load -= thisRoundToSend[i];
    thisProxy[nbor_node * nodeSize].PseudoLoad(pseudo_itr, thisRoundToSend[i], myNodeId);
  }

  // contribute to reduction to check if round is over
  thisProxy[0].pseudolb_barrier(allZero);

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
