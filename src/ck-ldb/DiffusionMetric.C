
#include <limits>
#include <vector>

class DiffusionMetric
{
public:
  // Pure virtual function providing interface framework.
  virtual int popBestObject(int nbor) = 0;
  virtual int getBestNeighbor() = 0;
  virtual void updateState(int objId, int destNbor) = 0;
};

class MetricComm : public DiffusionMetric
{
private:
  // All four of these are INCIDENT traffic -- both directions of every edge
  // touching the object -- so that internal and external are on the same basis
  // and their difference is meaningful. See the constructor for how the
  // inbound half of an external edge is recovered.
  std::vector<double> internalComm;               // internal comm for each obj
  std::vector<std::vector<double>> externalComm;  // external comm for each obj for each nbor
  std::vector<double> internalMsgs;               // message counts, same basis
  std::vector<std::vector<double>> externalMsgs;

  std::vector<double> toSendLoad;  // comm outward to each neighbor
  BaseLB::LDStats* nodeStats;

  std::vector<int> sendToNeighbors;
  std::vector<bool> objAvailable;

  // One entry per recorded local edge incident to the object, carrying both
  // dimensions. A symmetric exchange between a and b yields two records (a->b
  // and b->a), so a's list holds b twice and the sum over the list is the
  // incident traffic between them -- which is what updateState moves.
  struct CommEdge { int obj; double bytes; double msgs; };
  std::vector<std::vector<CommEdge>>
      objCommEdges;  // for each object, list of internal comm edges

  int n_objs;
  int neighborCount;
  int myNodeId;

  // The inbound half of an external edge is recorded on the peer node, so a
  // locally-observed external byte stands for this many incident bytes. Exactly
  // 2 under the symmetric-exchange assumption documented in the constructor;
  // named so the assumption is greppable rather than a bare literal.
  static constexpr double kIncidentFactor = 2.0;

  int getNborId(int nbor)
  {
    for (int i = 0; i < sendToNeighbors.size(); i++)
      if (sendToNeighbors[i] == nbor)
        return i;
    return -1;
  }

public:
  MetricComm(BaseLB::LDStats* ns, int nodeId, int nodeSize, int nCount,
             std::vector<double> tSL, std::vector<int> sendToNbrs, double &internal, double &external);
  int popBestObject(int nbor) override;
  int getBestNeighbor() override;
  void updateState(int objId, int destNbor) override;
};

class MetricCentroid : public DiffusionMetric
{
private:
  std::vector<std::vector<LBRealType>> nborCentroids;
  std::vector<std::vector<LBRealType>> objPosition;

  std::vector<double> nborDistances;
  std::vector<int> nborObjCount;
  std::vector<LBRealType> myCentroid;
  int position_dim;

  std::vector<double> toSendLoad;  // comm outward to each neighbor
  BaseLB::LDStats* nodeStats;

  std::vector<int> sendToNeighbors;
  std::vector<bool> objAvailable;

  std::vector<std::vector<double>> objNborDistances;

  int n_objs;
  int neighborCount;
  int myNodeId;

  int computeDistance(std::vector<LBRealType> objPos,
                      std::vector<LBRealType> nborCentroid)
  {
    double distance = 0;
    for (int i = 0; i < position_dim; i++)
    {
      distance += (objPos[i] - nborCentroid[i]) * (objPos[i] - nborCentroid[i]);
    }
    return distance;
  }

public:
  MetricCentroid(std::vector<std::vector<double>> nborCentroids,
                 std::vector<double> nborDistances, std::vector<LBRealType> myCentroid,
                 BaseLB::LDStats* ns, int nodeId, std::vector<double> tSL,
                 std::vector<int> sendToNbrs, std::vector<int> nborObjCount);
  int popBestObject(int nbor) override;
  int getBestNeighbor() override;
  void updateState(int objId, int destNbor) override;
};




MetricComm::MetricComm(BaseLB::LDStats* ns, int nodeId, int nodeSize, int nCount,
                       std::vector<double> tSL, std::vector<int> sendToNbrs, 
                       double &internalbytes, double &externalbytes)
    : nodeStats(ns),
      myNodeId(nodeId),
      n_objs(ns->objData.size()),
      neighborCount(nCount),
      toSendLoad(tSL),
      sendToNeighbors(sendToNbrs)
{
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
  double bestGain = -std::numeric_limits<double>::max();
  int bestObject = -1;

  double nborCapacity = toSendLoad[nbor];

  for (int i = 0; i < n_objs; i++)
  {
    if (!objAvailable[i]) continue;
    if (!nodeStats->objData[i].migratable) continue;

    double objLoad = diffusionObjLoad(nodeStats->objData[i]);
    if (objLoad > nborCapacity) continue;

    const double gain = externalComm[nbor][i] - internalComm[i];

    if (gain > bestGain)
    {
      bestGain = gain;
      bestObject = i;
    }
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

  for (int i = 0; i < n_objs; i++)
  {
    double objLoad = diffusionObjLoad(nodeStats->objData[i]);

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
        (objLoad <= nborCapacity))
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
