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

  neighborCount = sendToNeighbors.size(); // neighborCount = NUM_NEIGHBORS/2;
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

      int fromNode = thisNode;
      int toPE = commData.receiver.lastKnown();
      int toNode = toPE/nodeSize;

      if (thisIndex != toNode && toNode != -1 && toNode<numNodes)
        ebytes[toNode] += commData.bytes;
    }
  }
  sortArr(ebytes, numNodes, nbors);
}

void DiffusionLB::findNeighbors(int do_again)
{

  DEBUGL(("\nNode-%d, round =%d, sendToNeighbors.size() = %d", thisNode, round, sendToNeighbors.size()));
  fflush(stdout);
  if (round == 0)
  {
/*
    if (centroid)
    {
      pick = 0;
      createDistNList();
    }
    else
*/
    {
      createCommList();
    }
  }

  requests_sent = 0;
  if (!do_again || round == 100)
  {
    neighborCount = sendToNeighbors.size();

    loadNeighbors.resize(neighborCount);
    toSendLoad.resize(neighborCount);
    toReceiveLoad.resize(neighborCount);

    thisProxy[0].startStrategy();
    return;
  }
  int potentialNb = 0;
  int nborsNeeded = (NUM_NEIGHBORS - sendToNeighbors.size()) / 2;

  if (nborsNeeded > 0 && pick < NUM_NEIGHBORS + numNodes)
  {
    // CkPrintf("neighbors still needed\n");
    while (potentialNb < nborsNeeded && pick < NUM_NEIGHBORS + numNodes-1)
    {
      int potentialNbor = nbors[pick++]; // rand() % numNodes;
      CkPrintf("\n[Node-%d] potentialNbor node=%d(pe=%d)", thisNode, potentialNbor, potentialNbor*nodeSize); fflush(stdout);
      if (thisNode != potentialNbor &&
          std::find(sendToNeighbors.begin(), sendToNeighbors.end(), potentialNbor) == sendToNeighbors.end())
      {
        requests_sent++;
        thisProxy[potentialNbor*nodeSize].proposeNbor(thisNode);
        potentialNb++;
      }
    }
  }
  else
  {
    int do_again = 0;
    findNeighbors(do_again);
/*
    CkCallback cb(CkReductionTarget(DiffusionLB, findNeighbors), thisProxy);
    contribute(sizeof(int), &do_again, CkReduction::max_int, cb);
*/
  }

}

#if 0

/* This function creates a list of neighbors, stored in nbors, and sorted by "position" distance from the current node */
void Diffusion::createDistNList()
{
  // initialization
  long distance[numNodes];
  nbors = new int[numNodes];

  // compute distance from local aggregate centroid to all other aggregate centroids
  if (getCentroid(thisIndex).size() == 0)
  {
    CkPrintf("Error: map_pe_centroid is empty\n");
    CkExit();
  }
  std::vector<LBRealType> myCentroid = getCentroid(thisIndex);

  for (int n = 0; n < numNodes; n++)
  {
    nbors[n] = n;
    distance[n] = 0;
    if (n == thisIndex)
    {
      continue;
    }

    std::vector<LBRealType> oppCentroid = getCentroid(n);
    for (int i = 0; i < myCentroid.size(); i++)
    {
      distance[n] += (myCentroid[i] - oppCentroid[i]) * (myCentroid[i] - oppCentroid[i]);
    }
  }

  // sort neighbors based on centroid distance
  pairedSort(nbors, distance, numNodes);
}
#endif

void DiffusionLB::proposeNbor(int nborId)
{
  int agree = 0;
  if ((NUM_NEIGHBORS - sendToNeighbors.size()) - requests_sent > 0 && sendToNeighbors.size() < NUM_NEIGHBORS &&
      std::find(sendToNeighbors.begin(), sendToNeighbors.end(), nborId) == sendToNeighbors.end())
  {
    agree = 1;
    sendToNeighbors.push_back(nborId);
    DEBUGL(("\nNode-%d, round =%d Agreeing and adding %d ", thisNode, round, nborId));
  }
  else
  {
    DEBUGL(("\nNode-%d, round =%d Rejecting %d ", thisNode, round, nborId));
  }
  thisProxy[nborId*nodeSize].okayNbor(agree, thisNode);
}

void DiffusionLB::okayNbor(int agree, int nborId)
{
  if (sendToNeighbors.size() < NUM_NEIGHBORS && agree && std::find(sendToNeighbors.begin(), sendToNeighbors.end(), nborId) == sendToNeighbors.end())
  {
    // CkPrintf("\n[Node-%d, round-%d] Rcvd ack, adding %d as nbor", thisIndex, round, nborId);
    sendToNeighbors.push_back(nborId);
  }

  requests_sent--;
  if (requests_sent > 0)
    return;

  int do_again = 0;
  if (sendToNeighbors.size() < NUM_NEIGHBORS || round < 100)
    do_again = 1;
  round++;
  findNeighbors(do_again);
/*
  CkCallback cb(CkReductionTarget(DiffusionLB, findNeighbors), thisProxy);
  contribute(sizeof(int), &do_again, CkReduction::max_int, cb);
*/
}

#if 0
/* 3D and 2D neighbors for each cell in 3D/2D grid */

void Diffusion::pick3DNbors()
{
#if NBORS_3D
  int x = getX(thisIndex);
  int y = getY(thisIndex);
  int z = getZ(thisIndex);

  // 6 neighbors along face of cell
  sendToNeighbors.push_back(getNodeId(x - 1, y, z));
  sendToNeighbors.push_back(getNodeId(x + 1, y, z));
  sendToNeighbors.push_back(getNodeId(x, y - 1, z));
  sendToNeighbors.push_back(getNodeId(x, y + 1, z));
  sendToNeighbors.push_back(getNodeId(x, y, z - 1));
  sendToNeighbors.push_back(getNodeId(x, y, z + 1));

  // 12 neighbors along edges
  sendToNeighbors.push_back(getNodeId(x - 1, y - 1, z));
  sendToNeighbors.push_back(getNodeId(x - 1, y + 1, z));
  sendToNeighbors.push_back(getNodeId(x + 1, y - 1, z));
  sendToNeighbors.push_back(getNodeId(x + 1, y + 1, z));

  sendToNeighbors.push_back(getNodeId(x - 1, y, z - 1));
  sendToNeighbors.push_back(getNodeId(x - 1, y, z + 1));
  sendToNeighbors.push_back(getNodeId(x + 1, y, z - 1));
  sendToNeighbors.push_back(getNodeId(x + 1, y, z + 1));

  sendToNeighbors.push_back(getNodeId(x, y - 1, z - 1));
  sendToNeighbors.push_back(getNodeId(x, y - 1, z + 1));
  sendToNeighbors.push_back(getNodeId(x, y + 1, z - 1));
  sendToNeighbors.push_back(getNodeId(x, y + 1, z + 1));
#if 0
  //neighbors at vertices
  sendToNeighbors.push_back(getNodeId(x-1,y-1,z-1));
  sendToNeighbors.push_back(getNodeId(x-1,y-1,z+1));
  sendToNeighbors.push_back(getNodeId(x-1,y+1,z-1));
  sendToNeighbors.push_back(getNodeId(x-1,y+1,z+1));

  sendToNeighbors.push_back(getNodeId(x+1,y-1,z-1));
  sendToNeighbors.push_back(getNodeId(x+1,y-1,z+1));
  sendToNeighbors.push_back(getNodeId(x+1,y+1,z-1));
  sendToNeighbors.push_back(getNodeId(x+1,y+1,z+1));
#endif

  // Create 2d neighbors
#if 0
  if(thisIndex.x > 0) sendToNeighbors.push_back(getNodeId(thisIndex.x-1, thisIndex.y));
  if(thisIndex.x < N-1) sendToNeighbors.push_back(getNodeId(thisIndex.x+1, thisIndex.y));
  if(thisIndex.y > 0) sendToNeighbors.push_back(getNodeId(thisIndex.x, thisIndex.y-1));
  if(thisIndex.y < N-1) sendToNeighbors.push_back(getNodeId(thisIndex.x, thisIndex.y+1));
#endif

  int size = sendToNeighbors.size();
  int count = 0;

  for (int i = 0; i < size - count; i++)
  {
    if (sendToNeighbors[i] < 0)
    {
      sendToNeighbors[i] = sendToNeighbors[size - 1 - count];
      sendToNeighbors[size - 1 - count] = -1;
      i -= 1;
      count++;
    }
  }
  sendToNeighbors.resize(size - count);

  findNeighbors(0);
#endif
}

void Diffusion::pairedSort(int *A, long *B, int n)
{
  // sort array A based on corresponding values in B (both of size n)
  std::vector<std::pair<long, int>> vp;
  for (int i = 0; i < n; ++i)
  {
    vp.push_back(std::make_pair(B[i], A[i]));
  }

  sort(vp.begin(), vp.end());

  // convert A back to array
  for (int i = 0; i < n; ++i)
  {
    A[i] = vp[i].second;
  }
}

void Diffusion::sortArr(long arr[], int n, int *nbors)
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
    if (thisIndex != vp[i].second) // Ideally we shouldn't need to check this
      nbors[found++] = vp[i].second;
  if (found == 0)
    DEBUGL(("\nPE-%d Error!!!!!", CkMyPe()));
}
#endif
