#include <assert.h>

#define DEBUGL(x) x
#define ROUNDS 20

/* Pick NUM_NEIGHBORS in random */
/*readonly*/ bool centroid;

/* Entry point for neighbor building. Only rank0PEs call findNBors*/
void DiffusionLB::findNBors(int do_again)
{
    if (thisIndex != rank0PE)
        return;

    if (numNodes == 1)
    {
        CkPrintf("One node only - no neighbors\n");
        thisProxy[0].startStrategy();
        return;
    }

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
        // sdag calls beginMST();
        thisProxy[thisIndex].createCentroidList();
    }
}

// ******** FUNCTIONS FOR MST BUILDING ********
void DiffusionLB::beginMST()
{
    assert(thisIndex == rank0PE);

    mstVisitedPes.clear();
    mstVisitedPes.push_back(0);

    round = 0;
    rank0_barrier_counter = 0;

    // initialize vars for mst
    resetVarsMST();

    visited = false;

    if (thisIndex == 0)
        visited = true;

    buildMSTinRounds(best_weight, best_from, best_to);

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

        std::string myNbors = "After MST: Node " + std::to_string(myNodeId) + ": Neighbors: ";
        for (int i = 0; i < sendToNeighbors.size(); i++)
        {
            myNbors += std::to_string(sendToNeighbors[i]) + " ";
        }
        CkPrintf("%s\n", myNbors.c_str());
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
            CkPrintf("ERROR: MST can't add any more edges... Try adjusting NUM_NEIGHBORS\n");
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
        DEBUGL("MST is built. Begin finding remaining neighbors.\n");

        for (int i = 0; i < numNodes; i++)
            thisProxy[i * nodeSize].findNBorsRound();
    }
}
void DiffusionLB::findNBorsRound()
{
    if (thisIndex != rank0PE) return;

    round++;
    DEBUGL(("\nPE-%d, with round = %d", thisIndex, round));
    if (round < ROUNDS && thisIndex == 0)
    {
        CkCallback cb(CkIndex_DiffusionLB::findNBorsRound(), thisProxy);
        CkStartQD(cb);
    }
    if (round == ROUNDS)
    {
        neighborCount = sendToNeighbors.size();

        std::string myNbors = "After Nbor Finding: Node " + std::to_string(myNodeId) + ": Neighbors: ";
        for (int i = 0; i < sendToNeighbors.size(); i++)
        {
            myNbors += std::to_string(sendToNeighbors[i]) + " ";
        }
        CkPrintf("%s\n", myNbors.c_str());
        thisProxy[0].startStrategy();
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
                DEBUGL(("Node-%d sending request round =%d, potentialNbor = Node-%d\n", myNodeId, round, potentialNbor));
                thisProxy[potentialNbor * nodeSize].askNbor(myNodeId, round);
            }
            local_tries++;
        }
    }
    /*
      else
      {
        int do_again = 0;
        thisProxy[0].next_phase(do_again);
      }
    */
}
void DiffusionLB::next_phase(int val)
{
    acks++;
    if (val > max)
        max = val;
    if (acks == numNodes)
    {
        acks = 0;
        for (int i = 0; i < numNodes; i++)
            thisProxy[i * nodeSize].findNBorsRound();
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

        //    sendToNeighbors.push_back(nborId);
        DEBUGL(("\nNode-%d (holds[%d]=%d), (%d- %d- %d> 0?) round =%d Agreeing to hold for %d ", thisIndex, rnd, holds[rnd], NUM_NEIGHBORS, sendToNeighbors.size(), holds[rnd] - 1,
                round, nborId));
    }
    else
    {
        DEBUGL(("\nNode-%d, round =%d Rejecting %d ", thisIndex, round, nborId));
    }
    DEBUGL(("\n[PE-%d(node-%d)]Sending okay to nbor PE-%d(%d*%d)", thisIndex, myNodeId, nborId * nodeSize, nborId, nodeSize));
    thisProxy[nborId * nodeSize].okayNbor(agree, myNodeId /*thisIndex*/);
}
void DiffusionLB::okayNbor(int agree, int nborId)
{
    int nborsNeeded = NUM_NEIGHBORS - sendToNeighbors.size() - holds[round];
    if (nborsNeeded > 0 && agree && std::find(sendToNeighbors.begin(), sendToNeighbors.end(), nborId) == sendToNeighbors.end())
    {
        DEBUGL(("\n[Node-%d, round-%d] Rcvd ack, adding %d as nbor (neighbors:%d/%d, holds[%d]=%d)", thisIndex, round, nborId, sendToNeighbors.size(), NUM_NEIGHBORS, round, holds[round]));
        sendToNeighbors.push_back(nborId);
        thisProxy[nborId * nodeSize].ackNbor(myNodeId /*thisIndex*/);
    }
    else
    {
        DEBUGL(("\n[Node-%d] Decided not to pursue orig request to node %d", thisIndex, nborId));
    }
}
void DiffusionLB::ackNbor(int nborId)
{
    if (std::find(sendToNeighbors.begin(), sendToNeighbors.end(), nborId) == sendToNeighbors.end())
    {
        DEBUGL(("\n[Node-%d] Adding neighbor [%d] through final ack (neighbors:%d/%d)", thisIndex, nborId, sendToNeighbors.size(), NUM_NEIGHBORS));
        sendToNeighbors.push_back(nborId);
    }
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
            CkPrintf("Object %d has position of size %d, but expected %d\n", objIdx,
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
