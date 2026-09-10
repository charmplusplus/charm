#ifndef _DIFFUSION_METRIC_H
#define _DIFFUSION_METRIC_H

// The object-selection metrics of DiffusionLB's across-node phase. Declared
// here, apart from the balancer's chare, so that the offline simulator
// (tests/charm++/load_balancing/lbdriver/lbsim.C) can run the very same
// selection code on N virtual nodes inside one process. The definitions stay in
// DiffusionMetric.C, which DiffusionLB.C includes into its single translation
// unit, so linking -module DiffusionLB provides them.

#include <limits>
#include <vector>

#include "BaseLB.h"
#include "DiffusionCostModel.h"
#include "DiffusionLoad.h"

class DiffusionMetric
{
public:
  // Pure virtual function providing interface framework.
  virtual int popBestObject(int nbor) = 0;
  virtual int getBestNeighbor() = 0;
  virtual void updateState(int objId, int destNbor) = 0;
  virtual ~DiffusionMetric() {}
  // This node's remaining shed obligation, refreshed before each selection. A
  // metric that prices moves needs it to cap a move's benefit: shedding past
  // the fair share buys nothing, so an object bigger than what is left over is
  // only worth the excess it actually removes. Metrics that do not price moves
  // ignore it.
  virtual void setRemainingShed(double) {}
  // How the last decision loop split. Zero from a metric that does not price
  // moves, which cannot reject one.
  virtual int acceptedCount() const { return 0; }
  virtual int rejectedCount() const { return 0; }
  // Candidate filter the balancer sets before each selection: with a 1-D
  // ordering key only the ends of the node's interval may leave, and only
  // toward the neighbour on their side (DiffusionLB::allowedEndsFor). Null
  // means no restriction. Honoured by every metric so the interval property
  // does not depend on which metric was selected.
  void setAllowed(const std::vector<char>* a) { allowed_ = a; }
  bool isAllowed(int i) const
  {
    return allowed_ == NULL || i < 0 || i >= (int)allowed_->size() || (*allowed_)[i];
  }
protected:
  const std::vector<char>* allowed_ = NULL;
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
  int nodeSize;

  // Set when a calibrated cost table was loaded (+LBCostConfig). Null means no
  // model: selection falls back to edge-cut ranking and never refuses a move,
  // which is what this balancer did before the model existed.
  const DiffusionCostConfig* costCfg;
  // Transport tier reached by each neighbour, and the tier traffic that stays
  // here runs at. A DiffusionLB "node" is a process, so local means intra-process.
  std::vector<DiffusionTier> nborTier;
  DiffusionTier localTier;
  double remainingShed;

  // Objects already accepted for each neighbour this step. Non-zero means a
  // chunk is being grown toward that neighbour, which changes what the load
  // band is for (see popBestObject).
  std::vector<int> movesTo;

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
             std::vector<double> tSL, std::vector<int> sendToNbrs, double &internal, double &external,
             const DiffusionCostConfig* cfg);
  int popBestObject(int nbor) override;
  int getBestNeighbor() override;
  void updateState(int objId, int destNbor) override;
  void setRemainingShed(double r) override { remainingShed = r; }
  // Diagnostics: how the last decision loop split, so a step that moved nothing
  // can say whether it found nothing to move or priced everything out.
  int acceptedMoves = 0, rejectedMoves = 0;
  int acceptedCount() const override { return acceptedMoves; }
  int rejectedCount() const override { return rejectedMoves; }
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

#endif
