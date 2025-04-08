/* Pick NUM_NEIGHBORS in random */
/*readonly*/ bool centroid;

#define SELF_IDX NUM_NEIGHBORS
#define EXT_IDX NUM_NEIGHBORS + 1
#define NUM_NEIGHBORS 4

#define COMM

#include <assert.h>

#define ROUNDS 50

/* Entry point for neighbor building. Only rank0PEs call findNBors*/
void DiffusionLB::findNBors(int do_again)
{
  if(thisIndex==0) {
    CkCallback cb(CkIndex_DiffusionLB::begin(), thisProxy);
    CkStartQD(cb);
  }
  if (thisIndex != rank0PE)
  {
    return;
  }

  if (numNodes == 1)
  {
    CkPrintf("One node only - no neighbors\n");
    thisProxy[0].startStrategy();
    return;
  }

  // DEBUGL(("\nNode-%d, round =%d, sendToNeighbors.size() = %d", thisIndex, round,
  //         sendToNeighbors.size()));
  if (round == 0)
  {
    cost_for_neighbor = {};  // dictionary of nbor keys to cost

#ifdef COMM
    createCommList();
#else
    thisProxy[thisIndex]
        .createCentroidList();  // this is SDAG!! only works here because the result isn't
// used until LB across nodes
#endif
  }

}

void DiffusionLB::begin() {
  if (CkMyPe() != rank0PE) return;
  mstVisitedPes.clear();
  round = 0;
  rank0_barrier_counter = 0;

  // initialize vars for mst
  best_weight = 0;
  best_from = -1;
  best_to = 0;

  all_tos_negative = 1;

  visited = false;

#ifdef COMM
  buildMSTinRounds(best_weight, best_from, best_to);
#endif
  //  findRemainingNbors(0);
//  thisProxy[0].startFirstRound();
}

void DiffusionLB::buildMSTinRounds(double best_weight, int best_from, int best_to)
{
  // correctness checks for reduction input
  // note: if from = -1, this is fine because this is how we initialize the graph
  // TODO: optimization: remove the first round of this algo and just start with node 0 in
  // the graph

  // CkPrintf("Node-%d: best_to = %d, best_from = %d, best_weight = %f\n", thisIndex,
  //          best_to, best_from, best_weight);

  int to = best_to;
  int from = best_from;

  CkAssert(thisIndex == rank0PE);

  CkAssert(to != from && to != -1);


  // initiator is new node added to graph
  // assert that to is not already in graph
  if (myNodeId == to)
  {
    visited = true;
    if (from != -1)
    {
      // this check ensures that during the first round (when to = 0, from = -1), we don't
      // add -1 to the neighbors

      CkAssert(from < numNodes);
      // sendToNeighbors.push_back(from);

      addNeighbor(from);
    }
  }

  if (myNodeId == from)
  {
    visited = true;
    addNeighbor(to);
  }

  mstVisitedPes.push_back(to);

  if (mstVisitedPes.size() == numNodes)
  {
    // all nodes have been visited, MST is complete
    int do_again = 1;
    CkAssert(visited);
    CkAssert(std::find(mstVisitedPes.begin(), mstVisitedPes.end(), myNodeId) !=
           mstVisitedPes.end());

    CkAssert(sendToNeighbors.size() >= 1);

    mstVisitedPes.clear();

    thisProxy[0].startFirstRound();
    return;
  }

  // if (mstVisitedPes.size() == numNodes)
  // {
  //   // all nodes have been visited, MST is complete
  //   int do_again = 1;
  //   CkPrintf(
  //       "MST IS BUILT ON %d with numneighbors = %d, numnodes = %d, mstVisitedPes.size =
  //       "
  //       "%d\n",
  //       myNodeId, sendToNeighbors.size(), numNodes, mstVisitedPes.size());
  //   CkExit();
  //   // thisProxy[0].startFirstRound();
  // }

  // find best new edge to add, based on cost
  double newNbor = -1;
  double newParent = -1;
  double newweight = 0;  // TODO: cost is a misnomer, we want to maximize the cost

  // check if thisIndex is in mstVisitedPes
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
          sendToNeighbors.size() < NUM_NEIGHBORS  // dont build too many nieghbors
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

void DiffusionLB::next_MSTphase(double newweight, int newparent, int newto)
{
  acks++;

  if (newto >= 0)
    all_tos_negative = 0;

  // if (newto == -1 && newparent == -1)
  //   // this edge is invalid, no contribution

  if (newweight >= best_weight &&
      (newto != -1))  // TODO: this shouldn't really have to be >=... whats wrong?
  {
    best_weight = newweight;
    best_to = newto;
    best_from = newparent;
  }

  if (acks == numNodes)
  {
    assert(!all_tos_negative);  // all inputs should never have invalid edges

    if (best_to == best_from)
    {
      CkAbort("ERROR: MST can't add any more edges... Try adjusting NUM_NEIGHBORS\n");
    }

    CkPrintf("Adding MST edge from %d to %d\n", best_from, best_to);

    acks = 0;
    all_tos_negative = 1;

    for (int i = 0; i < numNodes; i++)
      thisProxy[i * nodeSize].buildMSTinRounds(best_weight, best_from, best_to);

    best_weight = 0;
    best_from = -1;
    best_to = -1;
  }
}

/* Custom reduction: wait until all rank0PEs have completed comm list building, then start
 * finding nbors.*/
void DiffusionLB::startFirstRound()
{
  rank0_barrier_counter++;
  if (rank0_barrier_counter == numNodes)
  {
    rank0_barrier_counter = 0;
    CkPrintf("MST is built. Begin finding remaining neighbors.\n");
    for (int i = 0; i < numNodes; i++) thisProxy[i * nodeSize].findNBorsRound();
  }
}

void DiffusionLB::findNBorsRound()
{
  if(CkMyPe()%nodeSize!=0) return;
//  assert(thisIndex % nodeSize == 0);  // only node managers should call this
  round++;
  DEBUGL(("\nPE-%d, with round = %d", CkMyPe(), round));
  if(round < ROUNDS && thisIndex==0) {

    CkCallback cb(CkIndex_DiffusionLB::findNBorsRound(), thisProxy);
    CkStartQD(cb);
  }
  if (round == ROUNDS)
  { 
    neighborCount = sendToNeighbors.size();
   /* 
    loadNeighbors = new double[neighborCount];
    toSendLoad = new double[neighborCount];
    toReceiveLoad = new double[neighborCount];
  */
//    if(thisIndex==0) {
//      CkCallback cb(CkIndex_DiffusionLB::startStrategy()/*startDiffusion()*/, thisProxy);
//      CkStartQD(cb);//contribute(cb);
//    }
    thisProxy[0].startStrategy();
    return;
  }

  int nborsNeeded = NUM_NEIGHBORS - sendToNeighbors.size() - holds[round];
  int local_tries = 0;

  if (nborsNeeded > 0)
  {
    while(local_tries < nborsNeeded/2)
    {
      int max_neighbors = numNodes<NUM_NEIGHBORS?numNodes:NUM_NEIGHBORS;
      pick = (pick + 1)%max_neighbors;
      int potentialNbor = node_idx[pick]; //pick - better logic needed here

      if(potentialNbor == -1) {
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
        thisProxy[potentialNbor*nodeSize].askNbor(myNodeId, round);
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

void DiffusionLB::createCommList()
{
  holds = new int[ROUNDS+1];
    for(int i=0;i<ROUNDS+1;i++)
      holds[i] = 0;
  pick = 0;

  long ebytes[numNodes];
  std::fill_n(ebytes, numNodes, 0);

  node_idx = new int[numNodes];
  for (int i = 0; i < numNodes; i++) node_idx[i] = -1;

  sendToNeighbors.clear();
  for (int edge = 0; edge < nodeStats->commData.size(); edge++)
  {
    LDCommData& commData = nodeStats->commData[edge];
    if ((!commData.from_proc()) && (commData.recv_type() == LD_OBJ_MSG))
    {
      LDObjKey from = commData.sender;
      LDObjKey to = commData.receiver.get_destObj();

      int fromobj = nodeStats->getHash(from);  // this replaces the simulator get_obj_idx
      int toobj = nodeStats->getHash(to);

      //if (fromobj == -1 || toobj == -1)
      //  continue;

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
    //CkPrintf("\n[PE-%d] ebytes[%d] = %d", CkMyPe(), i, ebytes[i]);
  }

  sortArr(ebytes, numNodes, node_idx);
}

void DiffusionLB::next_phase(int val)
{
  acks++;
  if (val > max)
    max = val;
  if (acks == numNodes)
  {
    acks = 0;
    for (int i = 0; i < numNodes; i++) thisProxy[i * nodeSize].findNBorsRound();
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

#if 0
void DiffusionLB::okayNbor(int agree, int nborId)
{
  if (sendToNeighbors.size() < NUM_NEIGHBORS && agree &&
      std::find(sendToNeighbors.begin(), sendToNeighbors.end(), nborId) ==
          sendToNeighbors.end())
  {
    // sendToNeighbors.push_back(nborId);
    addNeighbor(nborId);
  }

  requests_sent--;
  if (requests_sent > 0)
    return;

  int do_again = 0;
  if (sendToNeighbors.size() < NUM_NEIGHBORS)
    do_again = 1;
  round++;

  thisProxy[0].next_phase(do_again);
}
#endif
void DiffusionLB::askNbor(int nborId, int rnd)
{
  int agree = 0;
  int nborsNeeded = NUM_NEIGHBORS - sendToNeighbors.size() - holds[rnd];
  if (nborsNeeded>0 &&
      std::find(sendToNeighbors.begin(), sendToNeighbors.end(), nborId) == sendToNeighbors.end())
  {
    //HOLD A SPOT THOUGH on THIS ROUND!!
    agree = 1;
    holds[rnd]++;

//    sendToNeighbors.push_back(nborId);
    DEBUGL(("\nNode-%d (holds[%d]=%d), (%d- %d- %d> 0?) round =%d Agreeing to hold for %d ", thisIndex, rnd, holds[rnd], NUM_NEIGHBORS, sendToNeighbors.size(), holds[rnd]-1,
    round, nborId));
  }
  else
  {
    DEBUGL(("\nNode-%d, round =%d Rejecting %d ", thisIndex, round, nborId));
  }
  DEBUGL(("\n[PE-%d(node-%d)]Sending okay to nbor PE-%d(%d*%d)", CkMyPe(), myNodeId, nborId*nodeSize, nborId, nodeSize));
  thisProxy[nborId*nodeSize].okayNbor(agree, myNodeId/*thisIndex*/);
}

void DiffusionLB::okayNbor(int agree, int nborId)
{
  int nborsNeeded = NUM_NEIGHBORS - sendToNeighbors.size() - holds[round];
  if (nborsNeeded > 0 && agree && std::find(sendToNeighbors.begin(), sendToNeighbors.end(), nborId) == sendToNeighbors.end())
  {
    DEBUGL(("\n[Node-%d, round-%d] Rcvd ack, adding %d as nbor (neighbors:%d/%d, holds[%d]=%d)", thisIndex, round, nborId,sendToNeighbors.size(), NUM_NEIGHBORS, round, holds[round]));
    sendToNeighbors.push_back(nborId);
    thisProxy[nborId*nodeSize].ackNbor(myNodeId/*thisIndex*/);
  } else {
    DEBUGL(("\n[Node-%d] Decided not to pursue orig request to node %d", thisIndex, nborId));
  }
}

void DiffusionLB::ackNbor(int nborId) {
  if(std::find(sendToNeighbors.begin(), sendToNeighbors.end(), nborId) == sendToNeighbors.end()) {
    CkPrintf("\n[Node-%d] Adding neighbor [%d] through final ack (neighbors:%d/%d)", thisIndex, nborId, sendToNeighbors.size(), NUM_NEIGHBORS);
    sendToNeighbors.push_back(nborId);
  }
}
void DiffusionLB::sortArr(long arr[], int n, int* nbors)
{
  std::vector<std::pair<long, int> > vp;
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

// helper function to add neighbors to the list
void DiffusionLB::addNeighbor(int nbor)
{
#ifndef COMM
  std::vector<LBRealType> centroid = allNodeCentroids[nbor];
  double distance = allNodeDistances[nbor];
  int nborCount = allNodeObjCount[nbor];

  nborDistances.push_back(distance);
  nborCentroids.push_back(centroid);
  nborObjCount.push_back(nborCount);
#endif

  sendToNeighbors.push_back(nbor);
}
