/* Pick NUM_NEIGHBORS in random */
/*readonly*/ bool centroid;

void DiffusionLB::createCommList()
{
  pick = 0;

  long ebytes[numNodes];
  std::fill_n(ebytes, numNodes, 0);

  nbors = new int[NUM_NEIGHBORS + numNodes];
  for (int i = 0; i < numNodes; i++)
    nbors[i] = -1;

  neighborCount = 0;
  sendToNeighbors.clear();
  for(int edge = 0; edge < nodeStats->commData.size(); edge++) {
    LDCommData &commData = nodeStats->commData[edge];
    if ((!commData.from_proc()) && (commData.recv_type() == LD_OBJ_MSG))
    {
      LDObjKey from = commData.sender;
      LDObjKey to = commData.receiver.get_destObj();
      
      int fromobj = nodeStats->getHash(from);
      int toobj = nodeStats->getHash(to);

      if (fromobj == -1 || toobj == -1)
        continue;

      int fromNode = myNodeId;
      int toPE = commData.receiver.lastKnown();
      int toNode = toPE/nodeSize;

      if (myNodeId != toNode && toNode != -1 && toNode<numNodes)
        ebytes[toNode] += commData.bytes;
    }
  }
  sortArr(ebytes, numNodes, nbors);
//  for(int i=0;i<numNodes-1;i++)
//  CkPrintf("\n[Node-%d]Potential Neighbors List = Node-%d", myNodeId, nbors[i]);
}
void DiffusionLB::findNBors(int do_again) {
  if(round==0) {
    round++;
    createCommList();
  }
  requests_sent = 0;
  if(!do_again || round == 100) {
    neighborCount = sendToNeighbors.size();
    std::string nbor_nodes;
    for(int i = 0; i < neighborCount; i++) {
      nbor_nodes += "node-"+ std::to_string(sendToNeighbors[i])+", ";
    }
    CkPrintf("\nnode-%d with nbors %s\n", myNodeId, nbor_nodes.c_str());
    fflush(stdout);

    loadNeighbors.resize(neighborCount);
    toSendLoad.resize(neighborCount);
    toReceiveLoad.resize(neighborCount);

    thisProxy[0].startStrategy();
//    CkCallback cb(CkReductionTarget(DiffusionLB, startStrategy/*startDiffusion*/), thisProxy);
//    contribute(cb);

    return;
  }
  int potentialNb = 0;
  int nborsNeeded = (NUM_NEIGHBORS - sendToNeighbors.size())/2;
  if(nborsNeeded > 0) {
    while(potentialNb < nborsNeeded) {
      int potentialNbor = nbors[pick++];//rand() % numNodes;
      if(potentialNbor == -1) {
        thisProxy[0].next_phase(0);
        return;
      }
      if(myNodeId != potentialNbor &&
          std::find(sendToNeighbors.begin(), sendToNeighbors.end(), potentialNbor) == sendToNeighbors.end()) {
        requests_sent++;
        CkPrintf("\n[Node-%d] sending a request to node-%d", myNodeId, potentialNbor);
        thisProxy[potentialNbor*nodeSize].proposeNbor(myNodeId);
        potentialNb++;
      }
    }
  } else {
    int do_again = 0;
    thisProxy[0].next_phase(do_again);
/*
    CkCallback cb(CkReductionTarget(DiffusionLB, findNBors), thisProxy);
    contribute(sizeof(int), &do_again, CkReduction::max_int, cb);
*/
  }
}

void DiffusionLB::next_phase(int val) {
  acks++;
  if(val > max) max = val;
  if(acks == CkNumNodes()) {
    acks = 0;
    for(int i=0;i<CkNumNodes();i++)
      thisProxy[i*nodeSize].findNBors(max);
    max = 0;
  }
}

void DiffusionLB::proposeNbor(int nborId) {
  if(round==0) {
    round++;
    createCommList();
  }
  int agree = 0;
  if((NUM_NEIGHBORS-sendToNeighbors.size())-requests_sent > 0 && sendToNeighbors.size() < NUM_NEIGHBORS &&
      std::find(sendToNeighbors.begin(), sendToNeighbors.end(), nborId) == sendToNeighbors.end()) {
    agree = 1;
    sendToNeighbors.push_back(nborId);
    DEBUGL(("\nNode-%d, round =%d Agreeing and adding %d ", myNodeId, round, nborId));
  } else {
    DEBUGL(("\nNode-%d, round =%d Rejecting %d ", myNodeId, round, nborId));
  }
  thisProxy[nborId*nodeSize].okayNbor(agree, myNodeId);
}

void DiffusionLB::okayNbor(int agree, int nborId) {
  if(sendToNeighbors.size() < NUM_NEIGHBORS && agree && std::find(sendToNeighbors.begin(), sendToNeighbors.end(), nborId) == sendToNeighbors.end()) {
    DEBUGL(("\n[Node-%d, round-%d] Rcvd ack, adding %d as nbor", myNodeId, round, nborId));
    sendToNeighbors.push_back(nborId);
  }

  requests_sent--;
  if(requests_sent > 0) return;

  int do_again = 0;
  if(sendToNeighbors.size()<NUM_NEIGHBORS)
    do_again = 1;
  round++;
  thisProxy[0].next_phase(do_again);
/*
  CkCallback cb(CkReductionTarget(DiffusionLB, findNBors), thisProxy);
  contribute(sizeof(int), &do_again, CkReduction::max_int, cb);
*/
}


void DiffusionLB::sortArr(long arr[], int n, int *nbors)
{
  std::vector<std::pair<long, int> > vp;
  // Inserting element in pair vector
  // to keep track of previous indexes
  for (int i = 0; i < n; ++i) {
      vp.push_back(std::make_pair(arr[i], i));
  }
  // Sorting pair vector
  sort(vp.begin(), vp.end());
  reverse(vp.begin(), vp.end());
  int found = 0;
  for(int i=0;i<numNodes;i++)
    if(myNodeId!=vp[i].second) //Ideally we shouldn't need to check this
      nbors[found++] = vp[i].second;
  if(found == 0)
    DEBUGL(("\nPE-%d Error!!!!!", CkMyPe()));
}
