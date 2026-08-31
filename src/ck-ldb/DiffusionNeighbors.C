#include <assert.h>

#define ROUNDS 20

/* Pick NUM_NEIGHBORS in random */
/*readonly*/ bool centroid;



int DiffusionLB::findNborIdx(int node)
{
  for (int i = 0; i < sendToNeighbors.size(); i++)
    if (sendToNeighbors[i] == node)
      return i;
  return -1;
}

/* Entry point for neighbor building. Only rank0PEs call findNBors*/
void DiffusionLB::findNBors(int do_again)
{
    if (thisIndex != rank0PE)
        return;

    startNeighborTiming();

    hs_asksOut = 0;
    hs_confirmOut = 0;
    hs_phaseOwed = false;
    hs_barrierOwed = false;


    // Start timing neighbor selection
   
    if (numNodes == 1)
    {
        if (_lb_args.debug() > 1)
        {
            CkPrintf("One node only - no neighbors\n");
        }
        thisProxy[0].startStrategy();
        return;
    }

    // Reuse the graph a previous step built. Everything from here to the
    // rounds exists to construct sendToNeighbors, and it is the most
    // expensive phase of a step; the graph it produces changes slowly, so a
    // step that already has one skips straight to the rounds. One node kicks
    // the strategy -- startStrategy must run exactly once.
    if (hs_graphCached && getenv("CHARM_DIFFUSION_GRAPH_REBUILD") == NULL)
    {
        if (thisIndex == 0) thisProxy[0].startStrategy();
        return;
    }
    sendToNeighbors.clear();

    // general setup
    holds = new int[ROUNDS + 1];
    for (int i = 0; i < ROUNDS + 1; i++)
        holds[i] = 0;

    cost_for_neighbor.clear(); // dictionary of nbor keys to cost
    sendToNeighbors.clear();

    pick = 0;

    // build graph for comm vs centroid
    if (_lb_args.diffusionCommOn())
    {
        createCommList();

        beginMST();
    }
    else
    {
        nborCentroids.clear();
        allNodeCentroids.clear();
        allNodeObjCount.clear();
        allNodeDistances.clear();
        nborDistances.clear();
        nborObjCount.clear();
        myCentroid.clear();

        // sdag calls beginMST();
        thisProxy[thisIndex].createCentroidList();
    }
}

// ******** FUNCTIONS FOR MST BUILDING ********

void DiffusionLB::startMSTBarrier() {
    buildMSTinRounds(best_weight, best_from, best_to);
}
// Build the connectivity backbone for the diffusion graph.
//
// Phase 2 (pseudo-LB) only converges if the neighbour graph is CONNECTED. A split
// graph converges to two different averages and stays permanently imbalanced, with
// no error reported anywhere. Neighbours derived from the communication graph do not
// guarantee connectivity on their own -- "top-k heaviest partners" can easily yield
// isolated clusters -- so something has to supply that guarantee.
//
// A ring supplies it in O(1) per node with no protocol at all: every node links to
// its successor and predecessor, so the graph is connected by construction. The MST
// this replaces needed O(numNodes) rounds, rebuilt from scratch on every LB step,
// and closed each round with a *group* reduction that only rank-0 PEs ever
// contributed to -- so it could never complete and simply hung.
//
// Ring edges may carry little traffic, which is fine: they exist so that load can
// FLOW anywhere, while popBestObject still routes individual objects along the
// high-affinity edges added by the neighbour rounds that follow this.
void DiffusionLB::buildRingBackbone()
{
    assert(thisIndex == rank0PE);
    if (numNodes < 2)
        return;

    addNeighbor((myNodeId + 1) % numNodes);

    // With exactly two nodes the successor and the predecessor are the same node;
    // adding it twice would double-count it in every subsequent load exchange.
    if (numNodes > 2)
        addNeighbor((myNodeId - 1 + numNodes) % numNodes);

    if (_lb_args.debug() > 1)
    {
        std::string s = "Ring backbone: Node " + std::to_string(myNodeId) + ": ";
        for (size_t i = 0; i < sendToNeighbors.size(); i++)
            s += std::to_string(sendToNeighbors[i]) + " ";
        CkPrintf("%s\n", s.c_str());
    }
}

void DiffusionLB::beginMST()
{
    assert(thisIndex == rank0PE);

    // The MST that used to run here is replaced by the ring backbone above: same
    // connectivity guarantee, no rounds, no reduction, nothing to hang. +LBnoMST
    // retains the historical behaviour of building no backbone at all, which leaves
    // connectivity to chance -- kept only for A/B comparison.
    if (!_lb_args.noMST())
        buildRingBackbone();

    thisProxy[0].startFirstRound();
    return;

    // ---- unreachable: former MST construction, retained for reference ----
    // Known broken: the contribute() below is a group reduction, but only rank-0
    // PEs reach this function (findNBors returns early otherwise), so the reduction
    // can never complete. Every other barrier in this file is hand-rolled with a
    // counter for exactly that reason.
    if (_lb_args.debug() > 1)
    {
        CkPrintf("Beginning MST building\n");
    }

    mstVisitedPes.clear();
    mstVisitedPes.push_back(0);

    round = 0;
    rank0_barrier_counter = 0;

    // initialize vars for mst
    resetVarsMST();

    visited = false;

    if (thisIndex == 0)
        visited = true;

    contribute(CkCallback(CkReductionTarget(DiffusionLB, startMSTBarrier), thisProxy));
   

    //  findRemainingNbors(0);
    // thisProxy[0].startFirstRound();
}



void DiffusionLB::buildMSTinRounds(double best_weight, int best_from, int best_to)
{
    // correctness checks for reduction input
    // note: if from = -1, this is fine because this is how we initialize the graph
    // TODO: optimization: remove the first round of this algo and just start with node 0 in
    // the graph

    // CkPrintf("Node-%d: best_to = %d, best_from = %d, best_weight = %f\n", thisIndex,
    //          best_to, best_from, best_weight);

    // Ensure that best_to is not already in mstVisitedPes

    int to = best_to;
    int from = best_from;

    if (_lb_args.debug() > 1) {
        CkPrintf("Node %d: building in rounds, from %d, to %d, weight %f\n", myNodeId, from, to, best_weight);
    }

    // current edge is valid
    if (to != -1)
    {
        assert(from != -1);
        // initiator is new node added to graph
        // assert that to is not already in graph
        if (myNodeId == to)
        {
            visited = true;
            assert(from < numNodes && from >= 0);
            addNeighbor(from);
        }

        if (myNodeId == from)
        {
            assert(visited == true);
            addNeighbor(to);
        }

        mstVisitedPes.push_back(to);
    }

    if (mstVisitedPes.size() == numNodes)
    {
        // all nodes have been visited, MST is complete
        if (!visited)
            CkAbort("Node %d: MST is complete, but I am not in it\n", myNodeId);
        assert(std::find(mstVisitedPes.begin(), mstVisitedPes.end(), myNodeId) !=
               mstVisitedPes.end());
        assert(sendToNeighbors.size() >= 1);

        if (_lb_args.debug() > 1) {
          std::string myNbors = "After MST: Node " + std::to_string(myNodeId) + ": Neighbors: ";
          for (int i = 0; i < sendToNeighbors.size(); i++)
          {
              myNbors += std::to_string(sendToNeighbors[i]) + " ";
          }
          CkPrintf("%s\n", myNbors.c_str());
        }
        thisProxy[0].startFirstRound();
        return;
    }

    // find best new edge to add, based on cost
    int newNbor = -1;
    int newParent = -1;
    double newweight = -1; // TODO: cost is a misnomer, we want to maximize the cost

    if (visited)
    {
        // node in visited set
        // pick best edge (it is best because nbors are sorted by preference)
        for (int id = 0; id < numNodes; id++)
        {
            int nbor = node_idx[id];

            if (std::find(mstVisitedPes.begin(), mstVisitedPes.end(), nbor) ==
                    mstVisitedPes.end() &&
                nbor != myNodeId && nbor < numNodes && nbor >= 0 &&
                sendToNeighbors.size() < NUM_NEIGHBORS // dont build too many nieghbors
            )
            {
                newNbor = nbor;
                newParent = myNodeId;
                newweight = cost_for_neighbor[newNbor];
                break;
            }
        }
    }

    thisProxy[0].next_MSTphase(newweight, newParent, newNbor);
}
void DiffusionLB::resetVarsMST()
{
    // reset vars for next round
    best_weight = -1;
    best_from = -1;
    best_to = -1;

    all_tos_negative = 1;
    acks = 0;
}
void DiffusionLB::next_MSTphase(double newweight, int newparent, int newto)
{
    acks++;

    

    if (newto >= 0)
        all_tos_negative = 0;

    if (newweight > best_weight)
    {
        assert(newto != -1);
        best_weight = newweight;
        best_to = newto;
        best_from = newparent;
    }

    if (acks == numNodes)
    {
        if (all_tos_negative)
        {
            CkPrintf("ERROR: MST can't add enough edges... Try adjusting NUM_NEIGHBORS\n");
            CkExit(1);
        }

        for (int i = 0; i < numNodes; i++)
            thisProxy[i * nodeSize].buildMSTinRounds(best_weight, best_from, best_to);

        resetVarsMST();
    }
}

// ******** FUNCTIONS FOR FINDING REMAINING NBORS ********
void DiffusionLB::startFirstRound()
{
    rank0_barrier_counter++;
    if (rank0_barrier_counter == numNodes)
    {
        rank0_barrier_counter = 0;
        if (_lb_args.debug() > 1)
        {
            CkPrintf("MST is built. Begin finding remaining neighbors.\n");
        }

        for (int i = 0; i < numNodes; i++)
            thisProxy[i * nodeSize].findNBorsRound();
    }
}
void DiffusionLB::findNBorsRound()
{
    if (thisIndex != rank0PE) return;

    round++;

    neighborCount = sendToNeighbors.size();
    if (round == ROUNDS)
    {
        

        if (_lb_args.debug() > 1)
        {
            std::string myNbors = "After Nbor Finding: Node " + std::to_string(myNodeId) + ": Neighbors: ";
            for (int i = 0; i < sendToNeighbors.size(); i++)
            {
                myNbors += std::to_string(sendToNeighbors[i]) + " ";
            }
            CkPrintf("%s\n", myNbors.c_str());
        }
        hs_barrierOwed = true;
        hsMaybeAdvance();
        return;
    }

    int nborsNeeded = NUM_NEIGHBORS - sendToNeighbors.size() - holds[round];
    int local_tries = 0;

    if (nborsNeeded > 0)
    {
        while (local_tries < nborsNeeded / 2)
        {
            int max_neighbors = numNodes < NUM_NEIGHBORS ? numNodes : NUM_NEIGHBORS;
            pick = (pick + 1) % max_neighbors;
            int potentialNbor = node_idx[pick]; // pick - better logic needed here

            if (potentialNbor == -1)
            {
                local_tries++;
                continue;
            }
            if (myNodeId != potentialNbor &&
                std::find(sendToNeighbors.begin(), sendToNeighbors.end(), potentialNbor) == sendToNeighbors.end() &&
                potentialNbor < numNodes &&
                potentialNbor >= 0)
            {
                node_idx[pick] = -1;
                hs_asksOut++;
                thisProxy[potentialNbor * nodeSize].askNbor(myNodeId, round);
            }
            local_tries++;
        }
    }


    // Held, not sent: the barrier used to count a node as done the moment its
    // asks were *issued*, and a quiescence detector then drained the in-flight
    // okay/ack tail before the rounds started. The hold replaces that detector:
    // the contribution goes out only once every ask this node issued has been
    // answered and every edge it added at a peer has been confirmed processed.
    hs_phaseOwed = true;
    hs_phaseVal = nborsNeeded;
    hsMaybeAdvance();
}

// The barrier contributions this node owes go out only when it has no
// handshake message in flight. An ask is in flight until its okayNbor lands;
// an ackNbor -- the message whose processing is what makes an edge symmetric
// at the peer -- is in flight until the peer confirms it with ackNborDone. A
// node that has neither owes nothing and contributes immediately. The barrier
// therefore completes only when the graph is final and symmetric, which is the
// property the quiescence detector used to buy.
void DiffusionLB::hsMaybeAdvance()
{
    if (hs_asksOut > 0 || hs_confirmOut > 0) return;
    if (hs_phaseOwed)
    {
        hs_phaseOwed = false;
        thisProxy[0].next_phase(hs_phaseVal);
    }
    if (hs_barrierOwed)
    {
        hs_barrierOwed = false;
        thisProxy[0].startStrategyBarrier();
    }
}
void DiffusionLB::next_phase(int val)
{
    acks++;
    if (val > max)
        max = val;
    if (acks == numNodes)
    {
        acks = 0;
        
        if (max > 0) {
            for (int i = 0; i < numNodes; i++)
                thisProxy[i * nodeSize].findNBorsRound();
        } else {
            thisProxy[0].startStrategy();
        }
        max = 0;
    }
}
void DiffusionLB::proposeNbor(int nborId)
{
    int agree = 0;
    if ((NUM_NEIGHBORS - sendToNeighbors.size()) - requests_sent > 0 &&
        sendToNeighbors.size() < NUM_NEIGHBORS &&
        std::find(sendToNeighbors.begin(), sendToNeighbors.end(), nborId) ==
            sendToNeighbors.end())
    {
        agree = 1;
        // sendToNeighbors.push_back(nborId);
        addNeighbor(nborId);
    }
    thisProxy[nborId * nodeSize].okayNbor(agree, myNodeId);
}
void DiffusionLB::askNbor(int nborId, int rnd)
{
    int agree = 0;
    int nborsNeeded = NUM_NEIGHBORS - sendToNeighbors.size() - holds[rnd];
    if (nborsNeeded > 0 &&
        std::find(sendToNeighbors.begin(), sendToNeighbors.end(), nborId) == sendToNeighbors.end())
    {
        // Hold a spot on this round
        agree = 1;
        holds[rnd]++;
    }
    else
    {
        if (_lb_args.debug() == 3)
        {
            CkPrintf("\nNode-%d, round =%d Rejecting %d ", thisIndex, round, nborId);
        }
    }
    if (_lb_args.debug() == 3)
    {
        CkPrintf("\n[PE-%d(node-%d)]Sending okay to nbor PE-%d(%d*%d)", thisIndex, myNodeId, nborId * nodeSize, nborId, nodeSize);
    }
    thisProxy[nborId * nodeSize].okayNbor(agree, myNodeId /*thisIndex*/);
}
void DiffusionLB::okayNbor(int agree, int nborId)
{
    hs_asksOut--;
    int nborsNeeded = NUM_NEIGHBORS - sendToNeighbors.size() - holds[round];
    if (nborsNeeded > 0 && agree && std::find(sendToNeighbors.begin(), sendToNeighbors.end(), nborId) == sendToNeighbors.end())
    {
        if (_lb_args.debug() == 3) CkPrintf("\n[Node-%d, round-%d] Rcvd ack, adding %d as nbor (neighbors:%d/%d, holds[%d]=%d)", thisIndex, round, nborId, sendToNeighbors.size(), NUM_NEIGHBORS, round, holds[round]);
        addNeighbor(nborId);
        hs_confirmOut++;
        thisProxy[nborId * nodeSize].ackNbor(myNodeId /*thisIndex*/);
    }
    else
    {
        if (_lb_args.debug() == 3) CkPrintf("\n[Node-%d] Decided not to pursue orig request to node %d", thisIndex, nborId);
    }
    hsMaybeAdvance();
}
void DiffusionLB::ackNbor(int nborId)
{
    if (std::find(sendToNeighbors.begin(), sendToNeighbors.end(), nborId) == sendToNeighbors.end())
    {
        if (_lb_args.debug() == 3) CkPrintf("\n[Node-%d] Adding neighbor [%d] through final ack (neighbors:%d/%d)", thisIndex, nborId, sendToNeighbors.size(), NUM_NEIGHBORS);
        addNeighbor(nborId);
    }
    // The edge is now in place on this side; releasing the asker is what lets
    // it report its round done.
    thisProxy[nborId * nodeSize].ackNborDone();
}

void DiffusionLB::ackNborDone()
{
    hs_confirmOut--;
    hsMaybeAdvance();
}
void DiffusionLB::sortArr(long arr[], int n, int *nbors)
{
    std::vector<std::pair<long, int>> vp;
    // Inserting element in pair vector
    // to keep track of previous indexes
    for (int i = 0; i < n; ++i)
    {
        vp.push_back(std::make_pair(arr[i], i));
    }
    // Sorting pair vector
    sort(vp.begin(), vp.end());
    reverse(vp.begin(), vp.end());
    int found = 0;
    for (int i = 0; i < numNodes; i++)
        if (myNodeId != vp[i].second)
        {
            assert(vp[i].second != myNodeId);
            // Ideally we shouldn't need to check this
            nbors[found++] = vp[i].second;
        }
    if (found == 0 && numNodes > 1)
        CkAbort("Error: No neighbors found on %d\n", CmiMyPe());
}
void DiffusionLB::addNeighbor(int nbor)
{
    if (!(_lb_args.diffusionCommOn()))
    {
        std::vector<LBRealType> centroid = allNodeCentroids[nbor];
        double distance = allNodeDistances[nbor];
        int nborCount = allNodeObjCount[nbor];

        nborDistances.push_back(distance);
        nborCentroids.push_back(centroid);
        nborObjCount.push_back(nborCount);
    }

    sendToNeighbors.push_back(nbor);
}

// ******** CENTROID METHOD FUNCTIONS ********
void DiffusionLB::initializeCentroid()
{
    node_idx = new int[numNodes];

    allNodeCentroids.resize(numNodes);
    allNodeObjCount.resize(numNodes);
    allNodeDistances.resize(numNodes);

    int position_dim = 0;
    if (nodeStats->objData.size() > 0)
    {
        position_dim = nodeStats->objData[0].position.size();
    }

    // initialize centroid structures
    myCentroid.resize(position_dim, 0);
    for (int nbor = 0; nbor < numNodes; nbor++)
    {
        node_idx[nbor] = nbor;
        allNodeCentroids[nbor].resize(position_dim, 0);
    }

    int totalObjCount = 0;
    // compute my own centroid
    for (int objIdx = 0; objIdx < nodeStats->objData.size(); objIdx++)
    {
        LDObjData &objData = nodeStats->objData[objIdx];
        std::vector<LBRealType> position = objData.position;

        if (objData.position.size() != position_dim)
        {
            if (_lb_args.debug() > 0) CkPrintf("Object %d has position of size %d, but expected %d\n", objIdx,
                     objData.position.size(), position_dim);
            continue;
        }

        totalObjCount++;
        for (int i = 0; i < position_dim; i++)
        {
            myCentroid[i] += position[i];
        }
        // store centroid
    }
    if (totalObjCount != 0)
    {
        for (int i = 0; i < position_dim; i++)
            myCentroid[i] /= totalObjCount;
    }

    for (int i = 0; i < numNodes; i++)
        thisProxy[i * nodeSize].receiveCentroid(myNodeId, myCentroid, totalObjCount);
}
void DiffusionLB::processReceiveCentroid(int node, std::vector<LBRealType> centroid, int objCount)
{
    position_dim = 3;
    // CkPrintf(
    //     "Node %d received centroid from %d with length %d, dest has size %d, "
    //     "mycentroid has size %d, allnode sitances has size %d\n",
    //     myNodeId, node, centroid.size(), allNodeCentroids[node].size(),
    //     myCentroid.size(), allNodeDistances.size());
    double dist = 0;
    if (centroid.size() != position_dim)
    {
        CkAbort("Node %d received centroid of size %d from node %d, expected size %d\n",
                myNodeId, centroid.size(), node, position_dim);
    }

     if (myCentroid.size() != position_dim)
    {
        CkAbort("Node %d has myCentroid of size %d from node %d, expected size %d\n",
                myNodeId, myCentroid.size(), node, position_dim);
    }

    for (int i = 0; i < position_dim; i++)
    {
        allNodeCentroids[node][i] = centroid[i];
    }

    for (int i = 0; i < position_dim; i++)
    {
        dist += (myCentroid[i] - centroid[i]) * (myCentroid[i] - centroid[i]);
    }
    dist = sqrt(dist);

    if (node != myNodeId)
    {
        allNodeDistances[node] = dist;
    }
    else
    {
        allNodeDistances[node] = 10000;
    }

    allNodeObjCount[node] = objCount;
}
void DiffusionLB::finishCentroidList()
{
    assert(thisIndex == rank0PE);

    pairedSort(node_idx, allNodeDistances);
    beginMST();
}

// ******** COMMUNICATION METHOD FUNCTIONS ********
void DiffusionLB::createCommList()
{

    long ebytes[numNodes];
    std::fill_n(ebytes, numNodes, 0);

    node_idx = new int[numNodes];
    for (int i = 0; i < numNodes; i++)
        node_idx[i] = -1;

    for (int edge = 0; edge < nodeStats->commData.size(); edge++)
    {
        LDCommData &commData = nodeStats->commData[edge];
        if ((!commData.from_proc()) && (commData.recv_type() == LD_OBJ_MSG))
        {
            LDObjKey from = commData.sender;
            LDObjKey to = commData.receiver.get_destObj();

            int fromobj = nodeStats->getHash(from); // this replaces the simulator get_obj_idx
            int toobj = nodeStats->getHash(to);

            // if (fromobj == -1 || toobj == -1)
            //   continue;

            int fromNode = myNodeId;
            int toPE = commData.receiver.lastKnown();
            int toNode = toPE / nodeSize;
            if (myNodeId != toNode && toNode != -1 && toNode < numNodes)
                ebytes[toNode] += commData.bytes;
        }
    }

    // initialize cost per neighbor (cost is a misnomer: higher cost is better neighbor)
    // TODO: note that this cost can be zero... is this okay?
    for (int i = 0; i < numNodes; i++)
    {
        cost_for_neighbor[i] = ebytes[i];
        // CkPrintf("\n[PE-%d] ebytes[%d] = %d", thisIndex, i, ebytes[i]);
    }

    sortArr(ebytes, numNodes, node_idx);
}
