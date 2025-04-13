
#include <vector>

class DiffusionMetric
{
public:
  // Pure virtual function providing interface framework.
  virtual int popBestObject(int nbor) = 0;
  virtual int getBestNeighbor() = 0;
  virtual void updateState(int objId, int destNbor) = 0;
};

class MetricCommEI : public DiffusionMetric
{
private:
  std::vector<int> internalComm;    // internal comm for each obj
  std::vector<int> externalComm;    // external comm for each obj
  std::vector<int> commDifference;  // external - internal comm for each obj

  std::vector<double> toSendLoad;  // comm outward to each neighbor
  BaseLB::LDStats* nodeStats;

  std::vector<bool> objAvailable;

  int n_objs;
  int neighborCount;
  int myNodeId;

public:
  MetricCommEI(BaseLB::LDStats* ns, int nodeId, int nodeSize, int nCount,
               std::vector<double> tSL);

  int popBestObject(int nbor) override;
  int getBestNeighbor() override;
  void updateState(int objId, int destNbor) override;
};

class MetricComm : public DiffusionMetric
{
private:
  std::vector<int> internalComm;               // internal comm for each obj
  std::vector<std::vector<int>> externalComm;  // external comm for each obj for each nbor

  std::vector<double> toSendLoad;  // comm outward to each neighbor
  BaseLB::LDStats* nodeStats;

  std::vector<int> sendToNeighbors;
  std::vector<bool> objAvailable;

  std::vector<std::vector<std::pair<int, int>>>
      objCommEdges;  // for each object, list of internal comm edges

  int n_objs;
  int neighborCount;
  int myNodeId;

  int getNborId(int nbor)
  {
    for (int i = 0; i < sendToNeighbors.size(); i++)
      if (sendToNeighbors[i] == nbor)
        return i;
    return -1;
  }

public:
  MetricComm(BaseLB::LDStats* ns, int nodeId, int nodeSize, int nCount,
             std::vector<double> tSL, std::vector<int> sendToNbrs);
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

MetricCommEI::MetricCommEI(BaseLB::LDStats* ns, int nodeId, int nodeSize, int nCount,
                           std::vector<double> tSL)
{
  nodeStats = ns;
  myNodeId = nodeId;
  n_objs = nodeStats->objData.size();
  neighborCount = nCount;
  toSendLoad = tSL;

  internalComm.resize(n_objs, 0);
  externalComm.resize(n_objs, 0);
  commDifference.resize(n_objs, 0);
  objAvailable.resize(n_objs, true);

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
        internalComm[fromObj] += commData.bytes;

        if (toObj != -1 && toObj < n_objs)
          internalComm[toObj] += commData.bytes;
      }
      else
      {
        // External communication
        int fromObj = nodeStats->getHash(from);
        if (fromObj != -1 && fromObj < n_objs)
          externalComm[fromObj] += commData.bytes;

        // CkPrintf("Updated external comm at %d to %d\n", fromObj,
        // externalComm[fromObj]);
      }
    }
  }

  for (int i = 0; i < n_objs; i++)
  {
    commDifference[i] = externalComm[i] - internalComm[i];
  }
}

int MetricCommEI::popBestObject(int nbor)
{
  // find index of object with max internal comm
  int maxInternalComm = std::numeric_limits<int>::min();
  int bestObject = -1;
  for (int i = 0; i < n_objs; i++)
  {
    int testComm = commDifference[i];
    if (testComm > maxInternalComm && objAvailable[i] && nodeStats->objData[i].migratable)
    {
      maxInternalComm = commDifference[i];
      bestObject = i;
    }
  }

  return bestObject;
}

int MetricCommEI::getBestNeighbor()
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

void MetricCommEI::updateState(int objId, int destNbor)
{
  objAvailable[objId] = false;
  toSendLoad[destNbor] -= nodeStats->objData[objId].wallTime;
};

MetricComm::MetricComm(BaseLB::LDStats* ns, int nodeId, int nodeSize, int nCount,
                       std::vector<double> tSL, std::vector<int> sendToNbrs)
    : nodeStats(ns),
      myNodeId(nodeId),
      n_objs(ns->objData.size()),
      neighborCount(nCount),
      toSendLoad(tSL),
      sendToNeighbors(sendToNbrs)
{
  internalComm.resize(n_objs, 0);
  for (int nbor = 0; nbor < neighborCount; nbor++)
  {
    std::vector<int> nborComm;
    nborComm.resize(n_objs, 0);
    externalComm.push_back(nborComm);
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
        internalComm[fromObj] += commData.bytes;

        if (toObj != -1 && toObj < n_objs)
        {
          internalComm[toObj] += commData.bytes;
          objCommEdges[toObj].push_back(std::make_pair(toObj, commData.bytes));
          objCommEdges[fromObj].push_back(std::make_pair(toObj, commData.bytes));
        }
      }
      else
      {
        int nborId = getNborId(toNode);

        if (nborId == -1)  // could comm with node that is not a "neighbor".. ignore
          continue;

        int fromObj = nodeStats->getHash(from);
        if (fromObj != -1 && fromObj < n_objs)
          externalComm[nborId][fromObj] += commData.bytes;
      }
    }
  }
};

int MetricComm::popBestObject(int nbor)
{
  // find index of object with max internal comm
  int maxExternalComm = -1;
  int bestObject = -1;

  double nborCapacity = toSendLoad[nbor];

  for (int i = 0; i < n_objs; i++)
  {
    if(!objAvailable[i]) continue;
    double objLoad = nodeStats->objData[i].wallTime;

    int testComm = externalComm[nbor][i];

    if ((testComm > maxExternalComm) && objAvailable[i] &&
        (nodeStats->objData[i].migratable) && (objLoad <= nborCapacity*1.1))
    {
      maxExternalComm = testComm;
      bestObject = i;
    }
  }

  if (bestObject != -1)
  {
    assert(objAvailable[bestObject]);
    objAvailable[bestObject] = false;
  }
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
  toSendLoad[destNbor] -= nodeStats->objData[objId].wallTime;
  if(objId<0 || objId>=n_objs)
    return;
  for (std::pair<int, int> edge : objCommEdges[objId])
  {
    int toObj = edge.first;
    int comm = edge.second;
//    CkPrintf("\n[%d][%d]", toObj, comm);
    if(toObj<0 || toObj>=n_objs) continue; //replace with assert
    if (objAvailable[toObj])
    {
      externalComm[destNbor][toObj] += comm;
      internalComm[toObj] -= comm;
    }
  }

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

  objAvailable.resize(n_objs, true);
  objPosition.resize(n_objs);
  objNborDistances.resize(n_objs);

  for (int i = 0; i < n_objs; i++)
  {
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
  // find index of object with max internal comm
  double minDistance = std::numeric_limits<double>::max();
  int bestObject = -1;

  for (int i = 0; i < n_objs; i++)
  {
    double objLoad = nodeStats->objData[i].wallTime;
    int testDistance = objNborDistances[i][nbor];
    bool migratable = nodeStats->objData[i].migratable;
    bool available = objAvailable[i];

    if (testDistance < minDistance && available && migratable &&
        (objLoad <= toSendLoad[nbor]))
    {
      minDistance = testDistance;
      bestObject = i;
    }
  }

  if (bestObject != -1)
  {
    objAvailable[bestObject] = false;
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
    return;
  toSendLoad[destNbor] -= nodeStats->objData[objId].wallTime;
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
    objNborDistances[i][destNbor] =
        computeDistance(objPosition[i], nborCentroids[destNbor]);
  }
}
