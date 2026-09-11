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
    // One node: no handshake happened and no rounds will run, so there is
    // nothing of the balancer's in flight to drain. Straight to within-node.
    thisProxy.WithinNodeLB();
  }

  if (_lb_args.debug() > 1) CkPrintf("--------NEIGHBOR SELECTION COMPLETE (Using Comm? %s)--------\n",
           _lb_args.diffusionCommOn() ? "true" : "false");
  fflush(stdout);

  // Start pseudo LB timing
  startPseudoLBTiming();

  if (numNodes > 1)
  {
    // Drain the neighbour handshake before starting the rounds.
    //
    // Neighbour selection ends on a counting barrier (startStrategyBarrier /
    // next_phase) that each node reports to as soon as it has *issued* its asks
    // for the round -- not once the ask/okay/ack exchange has finished. The
    // final ack is what makes an edge symmetric: okayNbor adds the peer and
    // sends ackNbor, and the peer adds this node only when that ack lands. An
    // ack still in flight when the barrier completes therefore leaves one node
    // holding an edge its peer does not.
    //
    // The rounds below wait for exactly sendToNeighbors.size() messages each,
    // so a one-sided edge deadlocks them: the node missing it never sends, and
    // its peer waits forever.
    //
    // This is latent rather than observed at small node counts. Whenever
    // numNodes <= NUM_NEIGHBORS + 1 every node ends up adjacent to every other,
    // so the graph is complete and symmetric no matter how the acks race --
    // measured at 4 nodes, zero asymmetric edges over ten runs with and without
    // this drain. It becomes reachable once the graph is a genuine subgraph.
    //
    // Quiescence used to close it, cheaply in sync mode -- but under +LBAsync
    // the application keeps iterating through the step, the network never
    // drains, and every phase of this balancer stalled behind global silence
    // that was never going to fall. The race is now closed at its source: a
    // node's barrier contribution is held until its own asks are answered and
    // its own edge-adds are confirmed processed (hsMaybeAdvance), so barrier
    // completion itself proves the graph final and symmetric. Nothing is left
    // to drain, and the rounds can start directly.
    beginPseudoRounds();
  }
}

// PE 0, at quiescence: the neighbour graph is now final and symmetric, so arm
// the post-rounds quiescence detector and kick the rounds off.
void DiffusionLB::beginPseudoRounds()
{
  // AcrossNodeLB now starts from the roundsDone counting barrier rather than
  // from quiescence; see the note in the pseudolb_rounds loop.

  // Build the section of diffusing PEs once (one per node) and delegate it to
  // a multicast manager, so the per-round convergence reduction runs over
  // exactly those members. The seeding multicast below is what gives each
  // member its section cookie; after that the rounds only reduce.
  if (!pseudoSectionBuilt)
  {
    pseudoMcastGid = CProxy_CkMulticastMgr::ckNew();
    std::vector<int> pelist(numNodes);
    for (int i = 0; i < numNodes; i++) pelist[i] = i * nodeSize;
    // Group sections are built by constructor, not ckNew.
    pseudoSection =
        CProxySection_DiffusionLB(thisgroup, pelist.data(), numNodes);
    CkMulticastMgr* mg = CProxy_CkMulticastMgr(pseudoMcastGid).ckLocalBranch();
    pseudoSection.ckSectionDelegate(mg);
    pseudoSectionBuilt = true;
  }
  PseudoRoundMsg* m = new PseudoRoundMsg;
  m->mcastGid = pseudoMcastGid;
  m->maxRatio = 0.0;
  pseudoSection.pseudoRoundStart(m);
}


// The global convergence check used to be pseudolb_barrier: every node reported
// "nothing left to send" to PE 0 by point-to-point message, PE 0 ANDed the votes
// and broadcast the verdict. It was dropped because it put an O(N) fan-in plus a
// broadcast on PE 0 every round -- a central coordinator inside a balancer whose
// premise is not having one -- and the round loop is bounded by ITERATIONS anyway.
//
// It is back, in the form above: a reduction over a CkMulticast section holding
// exactly the diffusing PEs (one per node), with the verdict multicast back over
// the same section. The fan-in is now a spanning tree over the section rather than
// N messages into PE 0, and PEs that do not diffuse are not dragged into the round
// lockstep at all. Paying for that buys back most of the round budget: the rounds
// converge in 1-3 at 4 nodes against a fixed count of 40.


/* In combination with the pseudolb_rounds SDAG code, this builds the toReceiveLoad and
 * toSendLoad vectors for each node. It is onlyl called on rank0PEs*/
void DiffusionLB::PseudoLoadBalancing()
{
  // The arithmetic is diffusionRoundFlows (DiffusionFlow.h), shared with the
  // offline simulator so the two cannot drift apart. This member gathers the
  // chare's inputs, calls it, and commits the result.
  std::vector<char> flowAdjacent(neighborCount, 1);
  for (int i = 0; i < neighborCount; i++) flowAdjacent[i] = nborFlowAdjacent(i) ? 1 : 0;

  // Which rule refuses a destination, and on what evidence. cost_for_neighbor
  // treats an ABSENT neighbour as bordering and only a present-and-zero one as
  // silent, so "refused" and "never counted" have to be told apart by eye.
  if (_lb_args.debug() > 1 && pseudo_itr == 0)
  {
    for (int i = 0; i < neighborCount; i++)
    {
      const int nborNode = sendToNeighbors[i];
      const auto it = cost_for_neighbor.find(nborNode);
      const char* comm = cost_for_neighbor.empty() ? "no-comm-data"
                         : (it == cost_for_neighbor.end()) ? "absent"
                         : (it->second > 0.0)              ? "bytes>0"
                                                           : "zero";
      CkPrintf("[FLOWGATE node %d] nbor %d: flow %d  keyAdj %d  commAdj %d (%s %.0f)  "
               "keyed1D %d  load %.6f (mine %.6f)\n",
               myNodeId, nborNode, (int)flowAdjacent[i], nborKeyAdjacent(i) ? 1 : 0,
               nborCommAdjacent(i) ? 1 : 0, comm,
               (it == cost_for_neighbor.end()) ? -1.0 : (double)it->second,
               keyed1D ? 1 : 0, loadNeighbors[i], my_pseudo_load);
    }
  }

  std::vector<double> thisRoundToSend;
  diffusionRoundFlows(my_load, my_pseudo_load, effMinImbalance, _lb_args.diffusionBeta(),
                      loadNeighbors, flowAdjacent, toSendLoad, prevRoundToSend,
                      thisRoundToSend);

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


}


// Section members land here first: take the cookie out of the multicast, note
// the multicast manager it belongs to, then run the round loop.
void DiffusionLB::pseudoRoundStart(PseudoRoundMsg* m)
{
  CkGetSectionInfo(pseudoCookie, m);
  pseudoMcastGid = m->mcastGid;
  delete m;
  thisProxy[CkMyPe()].pseudolb_rounds();
}

// The section reduction delivers its result here on PE 0, which hands the same
// verdict to every member so they all leave the loop on the same round.
//
// The verdict travels in a PseudoRoundMsg rather than as a marshalled double:
// CkMulticastMgr::sendToSection writes the section cookie and entry point over
// the head of whatever message it is handed, which for a marshalled send lands
// squarely on CkMarshallMsg::msgBuf and the payload behind it. The receiver then
// unpacks its argument through a corrupted pointer. It has to be a message that
// starts with CkMcastBaseMsg.
void DiffusionLB::pseudoVerdictRoot(double maxRatio)
{
  PseudoRoundMsg* m = new PseudoRoundMsg;
  m->mcastGid = pseudoMcastGid;
  m->maxRatio = maxRatio;
  pseudoSection.pseudoConvergeResult(m);
}

// Every section member, once per round: refresh the cookie from the multicast
// (the documented CkMulticast contract) and hand the verdict to the SDAG loop.
void DiffusionLB::pseudoConvergeResult(PseudoRoundMsg* m)
{
  CkGetSectionInfo(pseudoCookie, m);
  const double maxRatio = m->maxRatio;
  delete m;
  thisProxy[CkMyPe()].pseudoVerdict(maxRatio);
}

// PE 0, once per round: the max over the diffusing PEs' metrics is the
// verdict, handed straight back to each of them. A member contributes to
// round k+1 only after it has this verdict for round k, so the count can never
// mix rounds.
void DiffusionLB::pseudoMetricContribute(double metric)
{
  if (metric > pseudoMaxMetric) pseudoMaxMetric = metric;
  if (++pseudoContribCount < numNodes) return;
  const double verdict = pseudoMaxMetric;
  pseudoContribCount = 0;
  pseudoMaxMetric = 0.0;
  for (int i = 0; i < numNodes; i++)
    thisProxy[i * nodeSize].pseudoVerdict(verdict);
}

// PE 0: every section member has left its round loop.
void DiffusionLB::roundsDone()
{
  if (++roundsDoneCount < numNodes) return;
  roundsDoneCount = 0;
  thisProxy.AcrossNodeLB();
}
