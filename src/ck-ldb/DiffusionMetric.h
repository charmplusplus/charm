#ifndef DIFFUSIONMETRIC_H
#define DIFFUSIONMETRIC_H

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

    CkPrintf("Metric has %d nonzero external comm values\n",
             std::count_if(commDifference.begin(), commDifference.end(),
                           [](int i) { return i > 0; }));
  }

  ~MetricCommEI() {}

  int popBestObject(int nbor) override
  {
    // find index of object with max internal comm
    int maxInternalComm = std::numeric_limits<int>::min();
    int bestObject = -1;
    for (int i = 0; i < n_objs; i++)
    {
      int testComm = commDifference[i];
      if (testComm > maxInternalComm && objAvailable[i] &&
          nodeStats->objData[i].migratable)
      {
        maxInternalComm = commDifference[i];
        bestObject = i;
      }
    }

    CkPrintf("Best object for neighbor %d is %d with comm diff %d\n", nbor, bestObject,
             maxInternalComm);
    return bestObject;
  }

  int getBestNeighbor() override
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

  void updateState(int objId, int destNbor)
  {
    objAvailable[objId] = false;
    toSendLoad[destNbor] -= nodeStats->objData[objId].wallTime;
  }
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

    CkPrintf("Metric traversing %d edges, building %d entries \n",
             nodeStats->commData.size(), n_objs);
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
          int nborId = getNborId(toNode);

          if (nborId == -1)  // could comm with node that is not a "neighbor".. ignore
            continue;

          int fromObj = nodeStats->getHash(from);
          if (fromObj != -1 && fromObj < n_objs)
            externalComm[nborId][fromObj] += commData.bytes;
        }
      }
    }
  }

  ~MetricComm() {}

  int popBestObject(int nbor) override
  {
    // find index of object with max internal comm
    int maxExternalComm = -1;
    int bestObject = -1;
    for (int i = 0; i < n_objs; i++)
    {
      int testComm = externalComm[nbor][i];
      assert(testComm >= 0);
      if (testComm > maxExternalComm && objAvailable[i] &&
          (nodeStats->objData[i].migratable == true))
      {
        maxExternalComm = testComm;
        bestObject = i;
      }
    }

    return bestObject;
  }

  int getBestNeighbor() override
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

  void updateState(int objId, int destNbor)
  {
    objAvailable[objId] = false;
    toSendLoad[destNbor] -= nodeStats->objData[objId].wallTime;
  }
};
#endif  // DIFFUSIONMETRIC_H